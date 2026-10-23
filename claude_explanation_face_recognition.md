I'll check what's currently the strongest open face-recognition setup, then write and test a Python inference script for you.

Starting the install in the background while I write the script.

InsightFace installed as v2.1, newer than what I wrote against, so I'll check the API matches before testing.

Model downloads and detects 6 faces in the bundled sample. Now an end-to-end test of enroll → identify → verify.

Genuine matches score ~0.94–0.97, the nearest wrong person at most 0.23, so the 0.40 threshold has a wide margin. Quick look at the annotated output:

The model is **InsightFace `buffalo_l`**: an SCRFD face detector plus an ArcFace ResNet50 embedder, run on ONNX Runtime. It's still the standard open choice, because ArcFace is arguably the most widely used loss in production face recognition as of 2026, with InsightFace as the reference implementation. On benchmarks, buffalo_l scores 99.83 on LFW and 97.25 on IJB-C. For maximum accuracy, use `--model antelopev2`, which pairs SCRFD-10GF with a ResNet100 trained on Glint360K.

**Setup and usage** (`face_recog.py`):
```bash
pip install insightface onnxruntime opencv-python   # onnxruntime-gpu for NVIDIA
python face_recog.py enroll   --gallery gallery/ --db faces.npz      # gallery/<name>/*.jpg
python face_recog.py identify --db faces.npz --image group.jpg --out annotated.jpg
python face_recog.py verify   --image a.jpg --image2 b.jpg
python face_recog.py video    --db faces.npz --source 0               # webcam
```
The model weights download automatically on the first run. The script uses a CUDA GPU if one is available. It can also be imported as a `FaceRecognizer` class.

**Test results** (on InsightFace's sample images):
- Six of six faces were identified correctly. Correct matches scored 0.94–0.97 similarity, and the closest wrong person scored at most 0.23.
- Same-person verification scored 0.96, and a different-person pair scored −0.04.
- The default match threshold is 0.40. Calibrate it on your own photos: lower means more lenient, higher means stricter.

**Licence:** the library code is MIT-licensed, but all the pretrained models are for non-commercial research purposes only. If this is for a commercial product, you'll need a commercial licence or differently licensed weights.

If you need better accuracy on low-quality CCTV-style footage, AdaFace is the stronger option. I can also add FAISS search for very large galleries.

Sources:
- [InsightFace model zoo](https://github.com/deepinsight/insightface/blob/e68b8b076fb11e9f75c87244d234385f0f318181/model_zoo/README.md)
- [insightface on PyPI](https://pypi.org/project/insightface/1.0/)
- [Face recognition models overview, 2026](https://facecheck.id/Face-Search-face-recognition-api)