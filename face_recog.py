#!/usr/bin/env python3
"""
face_recog.py - Face recognition inference with InsightFace
(SCRFD face detector + ArcFace 512-d embedder, run through ONNX Runtime).

Install
-------
    pip install insightface onnxruntime opencv-python numpy
    # NVIDIA GPU: replace onnxruntime with onnxruntime-gpu

Model packs (auto-downloaded to ~/.insightface/models/ on first run)
-------------------------------------------------------------------
    buffalo_l   SCRFD-10GF + ResNet50@WebFace600K   default, best speed/accuracy balance
    antelopev2  SCRFD-10GF + ResNet100@Glint360K    strongest embedder, slower
    buffalo_s   SCRFD-500MF + MobileFaceNet         lightweight, CPU / edge

    Licensing: the InsightFace code is MIT, but these pretrained model packs are
    released for NON-COMMERCIAL RESEARCH use only. For a commercial product you
    need a commercial license from InsightFace or a model trained on data you
    are allowed to use.

CLI usage
---------
    # 1) Enroll known people. Layout:  gallery/<person_name>/*.jpg  (3-10 photos each is ideal)
    python face_recog.py enroll   --gallery gallery/ --db faces.npz

    # 2) Identify every face in an image, write an annotated copy
    python face_recog.py identify --db faces.npz --image group.jpg --out annotated.jpg

    # 3) 1:1 verification - are these the same person?
    python face_recog.py verify   --image a.jpg --image2 b.jpg

    # 4) Webcam (0) or a video file
    python face_recog.py video    --db faces.npz --source 0
    python face_recog.py video    --db faces.npz --source clip.mp4 --out clip_annotated.mp4 --no-show

    Common flags: --model buffalo_l|antelopev2|buffalo_s   --threshold 0.40   --det-size 640   --cpu

Python API
----------
    from face_recog import FaceRecognizer
    fr = FaceRecognizer(model="buffalo_l")
    fr.enroll_folder("gallery/"); fr.save_db("faces.npz")
    results = fr.identify(fr.read_image("group.jpg"))   # list of dicts: name, score, bbox, det_score
    score, same = fr.verify(fr.read_image("a.jpg"), fr.read_image("b.jpg"))
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Cosine-similarity threshold for "same person" with ArcFace embeddings.
# 0.40 is a sensible starting point; calibrate on your own data:
#   ~0.30 lenient (fewer misses, more false matches) ... ~0.50 strict.
DEFAULT_THRESHOLD = 0.40


class FaceRecognizer:
    def __init__(
        self,
        model: str = "buffalo_l",
        det_size: int = 640,
        det_thresh: float = 0.5,
        min_face: int = 40,
        threshold: float = DEFAULT_THRESHOLD,
        use_gpu: bool = True,
        model_root: str = "~/.insightface",
    ):
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        available = ort.get_available_providers()
        if use_gpu and "CUDAExecutionProvider" in available:
            providers, ctx_id = ["CUDAExecutionProvider", "CPUExecutionProvider"], 0
        else:
            providers, ctx_id = ["CPUExecutionProvider"], -1

        root = os.path.expanduser(model_root)
        _fix_nested_model_dir(root, model)
        try:
            self.app = FaceAnalysis(
                name=model,
                root=root,
                allowed_modules=["detection", "recognition"],  # skip age/gender/landmark heads -> faster
                providers=providers,
            )
        except AssertionError:
            # First download of some packs (e.g. antelopev2) unzips into a nested folder.
            _fix_nested_model_dir(root, model)
            self.app = FaceAnalysis(
                name=model, root=root,
                allowed_modules=["detection", "recognition"], providers=providers,
            )
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size), det_thresh=det_thresh)

        self.model = model
        self.device = "GPU" if ctx_id >= 0 else "CPU"
        self.min_face = min_face
        self.threshold = threshold
        self.names: list[str] = []
        self.templates = np.zeros((0, 512), dtype=np.float32)

    # ------------------------------------------------------------------ I/O
    @staticmethod
    def read_image(path: str | Path) -> np.ndarray:
        """Read an image as BGR. Works with non-ASCII paths (cv2.imread does not on Windows)."""
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        return img

    def save_db(self, path: str | Path) -> None:
        np.savez(path, names=np.array(self.names), templates=self.templates, model=self.model)

    def load_db(self, path: str | Path) -> None:
        d = np.load(path, allow_pickle=False)
        db_model = str(d["model"]) if "model" in d.files else None
        if db_model and db_model != self.model:
            raise ValueError(
                f"DB was built with model '{db_model}' but recognizer uses '{self.model}'. "
                "Embeddings from different models are not comparable - re-enroll or pass --model."
            )
        self.names = [str(n) for n in d["names"]]
        self.templates = d["templates"].astype(np.float32)

    # ------------------------------------------------------------ detection
    def detect(self, img_bgr: np.ndarray):
        """Return detected faces (largest first), filtered by minimum face size."""
        faces = self.app.get(img_bgr)
        faces = [f for f in faces if min(f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]) >= self.min_face]
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        return faces

    def embed_largest(self, img_bgr: np.ndarray) -> np.ndarray | None:
        faces = self.detect(img_bgr)
        return faces[0].normed_embedding.astype(np.float32) if faces else None

    # ----------------------------------------------------------- enrollment
    def enroll(self, name: str, images: list[np.ndarray]) -> int:
        """Build one template per person: mean of L2-normalised embeddings, re-normalised."""
        embs = []
        for img in images:
            e = self.embed_largest(img)
            if e is not None:
                embs.append(e)
        if not embs:
            return 0
        t = np.mean(embs, axis=0)
        t /= np.linalg.norm(t) + 1e-12
        if name in self.names:  # replace existing template
            self.templates[self.names.index(name)] = t
        else:
            self.names.append(name)
            self.templates = np.vstack([self.templates, t[None, :]])
        return len(embs)

    def enroll_folder(self, gallery_dir: str | Path, verbose: bool = True) -> None:
        gallery_dir = Path(gallery_dir)
        people = sorted(p for p in gallery_dir.iterdir() if p.is_dir())
        if not people:
            raise ValueError(f"No person sub-folders in {gallery_dir} (expected gallery/<name>/*.jpg)")
        for person in people:
            files = sorted(f for f in person.iterdir() if f.suffix.lower() in IMG_EXTS)
            imgs = []
            for f in files:
                try:
                    imgs.append(self.read_image(f))
                except ValueError as e:
                    print(f"  ! {e}", file=sys.stderr)
            n = self.enroll(person.name, imgs)
            if verbose:
                status = f"{n}/{len(files)} images used" if n else "NO FACE FOUND - skipped"
                print(f"  {person.name:<24} {status}")

    # ------------------------------------------------------- recognition
    def identify(self, img_bgr: np.ndarray, top_k: int = 1) -> list[dict]:
        """Detect all faces and match each against the enrolled gallery."""
        faces = self.detect(img_bgr)
        results = []
        for f in faces:
            r = {
                "bbox": [int(v) for v in f.bbox],
                "det_score": round(float(f.det_score), 3),
                "name": "unknown",
                "score": None,
            }
            if len(self.names):
                sims = self.templates @ f.normed_embedding.astype(np.float32)
                order = np.argsort(-sims)[:top_k]
                best = int(order[0])
                r["score"] = round(float(sims[best]), 3)
                if sims[best] >= self.threshold:
                    r["name"] = self.names[best]
                if top_k > 1:
                    r["candidates"] = [(self.names[i], round(float(sims[i]), 3)) for i in order]
            results.append(r)
        return results

    def verify(self, img1: np.ndarray, img2: np.ndarray) -> tuple[float, bool]:
        """1:1 comparison of the largest face in each image."""
        e1, e2 = self.embed_largest(img1), self.embed_largest(img2)
        if e1 is None or e2 is None:
            raise ValueError("No face found in " + ("first" if e1 is None else "second") + " image")
        score = float(np.dot(e1, e2))
        return score, score >= self.threshold

    # ------------------------------------------------------------- drawing
    @staticmethod
    def draw(img_bgr: np.ndarray, results: list[dict]) -> np.ndarray:
        out = img_bgr.copy()
        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            known = r["name"] != "unknown"
            color = (0, 200, 0) if known else (0, 0, 230)
            label = r["name"] if r["score"] is None else f"{r['name']} {r['score']:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ty = max(y1, th + 8)
            cv2.rectangle(out, (x1, ty - th - 8), (x1 + tw + 6, ty), color, -1)
            cv2.putText(out, label, (x1 + 3, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return out


def _fix_nested_model_dir(root: str, model: str) -> None:
    """Some packs unzip to models/<name>/<name>/*.onnx; FaceAnalysis expects models/<name>/*.onnx."""
    outer = Path(root) / "models" / model
    inner = outer / model
    if inner.is_dir() and not any(outer.glob("*.onnx")):
        for f in inner.glob("*.onnx"):
            f.rename(outer / f.name)


# ====================================================================== CLI
def _build_recognizer(a) -> FaceRecognizer:
    t0 = time.time()
    fr = FaceRecognizer(
        model=a.model, det_size=a.det_size, threshold=a.threshold,
        min_face=a.min_face, use_gpu=not a.cpu,
    )
    print(f"[model {a.model} on {fr.device}, loaded in {time.time() - t0:.1f}s]", file=sys.stderr)
    return fr


def cmd_enroll(a):
    fr = _build_recognizer(a)
    if a.append and Path(a.db).exists():
        fr.load_db(a.db)
    print(f"Enrolling from {a.gallery}")
    fr.enroll_folder(a.gallery)
    fr.save_db(a.db)
    print(f"Saved {len(fr.names)} identities -> {a.db}")


def cmd_identify(a):
    fr = _build_recognizer(a)
    fr.load_db(a.db)
    img = fr.read_image(a.image)
    t0 = time.time()
    results = fr.identify(img, top_k=a.top_k)
    dt = (time.time() - t0) * 1000
    print(json.dumps({"image": a.image, "faces": results, "ms": round(dt, 1)}, indent=2))
    if a.out:
        cv2.imwrite(a.out, fr.draw(img, results))
        print(f"Annotated image -> {a.out}", file=sys.stderr)


def cmd_verify(a):
    fr = _build_recognizer(a)
    score, same = fr.verify(fr.read_image(a.image), fr.read_image(a.image2))
    print(json.dumps({"similarity": round(score, 4), "threshold": a.threshold, "same_person": same}))


def cmd_video(a):
    fr = _build_recognizer(a)
    fr.load_db(a.db)
    src = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"Cannot open video source {a.source}")
    writer, last, i, fps_t = None, [], 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % a.every == 0:  # run the model every N frames, reuse boxes in between
            last = fr.identify(frame)
        i += 1
        vis = fr.draw(frame, last)
        fps = i / (time.time() - fps_t)
        cv2.putText(vis, f"{fps:.1f} FPS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if a.out:
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                         cap.get(cv2.CAP_PROP_FPS) or 25, (w, h))
            writer.write(vis)
        if not a.no_show:
            cv2.imshow("face_recog (q to quit)", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default="buffalo_l", help="buffalo_l | antelopev2 | buffalo_m | buffalo_s")
    common.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="cosine match threshold")
    common.add_argument("--det-size", type=int, default=640, help="detector input size (320 faster, 640 default)")
    common.add_argument("--min-face", type=int, default=40, help="ignore faces smaller than N px")
    common.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")

    p = argparse.ArgumentParser(description="Face recognition with InsightFace (SCRFD + ArcFace)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("enroll", parents=[common], help="build gallery DB from gallery/<name>/*.jpg")
    s.add_argument("--gallery", required=True)
    s.add_argument("--db", default="faces.npz")
    s.add_argument("--append", action="store_true", help="add to / update an existing DB")
    s.set_defaults(fn=cmd_enroll)

    s = sub.add_parser("identify", parents=[common], help="recognise all faces in one image")
    s.add_argument("--db", default="faces.npz")
    s.add_argument("--image", required=True)
    s.add_argument("--out", help="write annotated image here")
    s.add_argument("--top-k", type=int, default=1)
    s.set_defaults(fn=cmd_identify)

    s = sub.add_parser("verify", parents=[common], help="1:1 compare two images")
    s.add_argument("--image", required=True)
    s.add_argument("--image2", required=True)
    s.set_defaults(fn=cmd_verify)

    s = sub.add_parser("video", parents=[common], help="webcam or video file")
    s.add_argument("--db", default="faces.npz")
    s.add_argument("--source", default="0", help="camera index or video path")
    s.add_argument("--every", type=int, default=1, help="run recognition every N frames")
    s.add_argument("--out", help="write annotated video (.mp4)")
    s.add_argument("--no-show", action="store_true", help="don't open a window (headless)")
    s.set_defaults(fn=cmd_video)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
