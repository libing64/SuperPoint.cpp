# SuperPoint + LightGlue inference (ONNX Runtime / TensorRT)

C++ inference for SuperPoint and LightGlue using the same **cvg/LightGlue** weights as the `deep_matching` PyTorch stack. SuperPoint ONNX only emits dense score logits and descriptors; NMS / top-k / bilinear sampling live in shared C++ so ORT and TensorRT stay aligned.

## Layout

```
include/splg/     public types and backends
src/common/       npy IO, SuperPoint postprocess, LightGlue filter
src/onnx/         ONNX Runtime SuperPoint + LightGlue
src/trt/          TensorRT engines (optional)
tools/            infer_onnx, infer_trt, bench_onnx, bench_trt
python/           dump / export / TRT build / compare / HPatches bench
```

## Environment

```bash
conda activate deep_matching
pip install 'numpy<2' onnx onnxruntime onnxscript   # export + optional Python smoke
```

TensorRT comes from the system install (`libnvinfer-dev`, `trtexec` on `PATH`). LightGlue ONNX export uses the torch dynamo exporter (`onnxscript`).

## 1. Dump PyTorch reference

```bash
conda activate deep_matching
python python/dump_pytorch_ref.py --out outputs/ref --device cpu
```

Uses HPatches `i_crownday` when present, otherwise a synthetic checkerboard. Resize short edge 480, pad to a multiple of 8. LightGlue runs with `flash=False`, `depth_confidence=-1`, `width_confidence=-1`.

## 2. Export ONNX

```bash
python python/export_onnx.py --out models
```

- `models/superpoint.onnx`: `image [1,1,H,W] -> score_logits [1,65,h,w], descriptors [1,256,h,w]`
- `models/superpoint_lightglue.onnx`: `kpts0/1, desc0/1, image_size0/1 -> matches0, mscores0`

## 3. Build C++

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

ONNX Runtime is picked up from `onnxruntime_ROOT`, COLMAP’s cached copy, or downloaded (v1.24.4).

## 4. Run ORT

```bash
./build/bin/infer_onnx \
  --sp models/superpoint.onnx \
  --lg models/superpoint_lightglue.onnx \
  --ref outputs/ref \
  --out outputs/onnx
```

## 5. TensorRT (optional)

```bash
python python/build_trt_engine.py --onnx-dir models --out models
# or: trtexec --onnx=... --saveEngine=... --noTF32 --skipInference --minShapes=... --optShapes=... --maxShapes=...
./build/bin/infer_trt \
  --sp models/superpoint.engine \
  --lg models/superpoint_lightglue.engine \
  --ref outputs/ref \
  --out outputs/trt
```

Optimization profiles: SuperPoint H/W 64–1024 (opt 480×640); LightGlue N 1–2048 (opt 512). Build with `--noTF32` for FP32-accurate comparison.

## 6. Compare

```bash
python python/compare_accuracy.py --ref outputs/ref --onnx outputs/onnx --trt outputs/trt
```

| Check | ORT vs PT | TRT vs PT |
|-------|-----------|-----------|
| SuperPoint dense score / desc | max abs < 1e-4, cosine > 0.999 | max abs < 1e-3, cosine > 0.999 |
| SuperPoint kpts | IoU@1px > 0.99 | IoU@1px > 0.99 |
| LightGlue on PyTorch kpts | match agree > 99% | match agree > 98% |
| End-to-end pair IoU | > 0.95 | > 0.95 |

`lg_ref_matches0.npy` is LightGlue run on the PyTorch keypoints so matcher error is not mixed with detector jitter.

## 7. HPatches timing + accuracy

Same preprocess as the dump (ITU gray, `resize_short=480`, pad to a multiple of 8). PyTorch runs on GPU when available; C++ ORT is CPU; C++ TensorRT is GPU. Models are loaded once per backend.

```bash
conda activate deep_matching
cmake --build build -j --target bench_onnx bench_trt
python python/bench_hpatches.py --out outputs/hpatches
# subset: python python/bench_hpatches.py --max-pairs 20 --out outputs/hpatches
```

Writes `outputs/hpatches/summary.json`, per-backend `results.jsonl`, and match visualizations under `outputs/hpatches/viz/` (`i_crownday`, `v_bark`, `i_books`, `v_bricks` by default). Reports MMA@1/3/5, homography AUC@3/5/10, and SuperPoint / LightGlue / end-to-end milliseconds.
