#!/usr/bin/env python3
"""Export SuperPoint (dense) and LightGlue to ONNX from cvg/LightGlue weights.

Run inside conda env deep_matching (needs onnx; onnxruntime optional for check):
  pip install 'numpy<2' onnx onnxruntime onnxscript
  python python/export_onnx.py --out models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from splg_common import LightGlueExport, SuperPointDense, load_models  # noqa: E402


def _export(model, args, path: Path, input_names, output_names, dynamic_axes, opset: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    last_err = None
    for dynamo in (True, False):
        try:
            try:
                torch.onnx.export(model, args, str(path), dynamo=dynamo, **kwargs)
            except TypeError:
                if dynamo:
                    raise
                torch.onnx.export(model, args, str(path), **kwargs)
            last_err = None
            print(f"exported {path} (dynamo={dynamo})")
            break
        except Exception as exc:
            last_err = exc
            print(f"export dynamo={dynamo} failed: {type(exc).__name__}: {exc}")
    if last_err is not None:
        raise last_err


def _check_onnx(path: Path) -> None:
    try:
        import onnx

        model = onnx.load(str(path))
        onnx.checker.check_model(model)
        print(f"onnx check ok: {path}")
    except ImportError:
        print("onnx not installed; skip checker (pip install onnx)")
    except Exception as exc:
        print(f"onnx check warning for {path}: {exc}")


def _ort_smoke_sp(path: Path, image: torch.Tensor) -> None:
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("onnxruntime not installed; skip ORT smoke")
        return
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"image": image.numpy()})
    print(f"ORT SuperPoint: {[np.asarray(x).shape for x in out]}")


def _ort_smoke_lg(path: Path, feed: dict) -> None:
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        return
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {k: v.numpy() for k, v in feed.items()})
    print(f"ORT LightGlue: {[np.asarray(x).shape for x in out]}")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "models")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num-keypoints", type=int, default=256)
    args = parser.parse_args()

    sp, lg = load_models(device=args.device, max_keypoints=2048)
    dense = SuperPointDense(sp).eval()
    lg_exp = LightGlueExport(lg).eval()

    image = torch.rand(1, 1, args.height, args.width, dtype=torch.float32)
    sp_path = args.out / "superpoint.onnx"
    _export(
        dense,
        (image,),
        sp_path,
        input_names=["image"],
        output_names=["score_logits", "descriptors"],
        dynamic_axes={
            "image": {2: "H", 3: "W"},
            "score_logits": {2: "h", 3: "w"},
            "descriptors": {2: "h", 3: "w"},
        },
        opset=args.opset,
    )
    _check_onnx(sp_path)
    _ort_smoke_sp(sp_path, image)

    n0 = args.num_keypoints
    n1 = args.num_keypoints
    kpts0 = torch.rand(1, n0, 2) * torch.tensor([args.width, args.height], dtype=torch.float32)
    kpts1 = torch.rand(1, n1, 2) * torch.tensor([args.width, args.height], dtype=torch.float32)
    desc0 = torch.nn.functional.normalize(torch.randn(1, n0, 256), dim=-1)
    desc1 = torch.nn.functional.normalize(torch.randn(1, n1, 256), dim=-1)
    size0 = torch.tensor([[args.width, args.height]], dtype=torch.float32)
    size1 = torch.tensor([[args.width, args.height]], dtype=torch.float32)

    lg_path = args.out / "superpoint_lightglue.onnx"
    _export(
        lg_exp,
        (kpts0, kpts1, desc0, desc1, size0, size1),
        lg_path,
        input_names=["kpts0", "kpts1", "desc0", "desc1", "image_size0", "image_size1"],
        output_names=["matches0", "mscores0"],
        dynamic_axes={
            "kpts0": {1: "N0"},
            "kpts1": {1: "N1"},
            "desc0": {1: "N0"},
            "desc1": {1: "N1"},
            "matches0": {1: "N0"},
            "mscores0": {1: "N0"},
        },
        opset=args.opset,
    )
    _check_onnx(lg_path)
    _ort_smoke_lg(
        lg_path,
        {
            "kpts0": kpts0,
            "kpts1": kpts1,
            "desc0": desc0,
            "desc1": desc1,
            "image_size0": size0,
            "image_size1": size1,
        },
    )


if __name__ == "__main__":
    main()
