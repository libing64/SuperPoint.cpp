#!/usr/bin/env python3
"""Build TensorRT engines from exported SuperPoint / LightGlue ONNX models.

  python python/build_trt_engine.py --onnx-dir models --out models
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_with_tensorrt(
    onnx_path: Path,
    engine_path: Path,
    profiles: list[tuple[str, tuple[int, ...], tuple[int, ...], tuple[int, ...]]],
    workspace_gb: int,
) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i), file=sys.stderr)
            raise RuntimeError(f"ONNX parse failed: {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    profile = builder.create_optimization_profile()
    for name, mn, opt, mx in profiles:
        profile.set_shape(name, mn, opt, mx)
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT build failed: {onnx_path}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    print(f"wrote {engine_path}")


def build_with_trtexec(
    onnx_path: Path,
    engine_path: Path,
    min_shapes: str,
    opt_shapes: str,
    max_shapes: str,
) -> None:
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--noTF32",
        f"--minShapes={min_shapes}",
        f"--optShapes={opt_shapes}",
        f"--maxShapes={max_shapes}",
    ]
    print(" ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--out", type=Path, default=ROOT / "models")
    parser.add_argument("--opt-h", type=int, default=480)
    parser.add_argument("--opt-w", type=int, default=640)
    parser.add_argument("--min-hw", type=int, default=64)
    parser.add_argument("--max-hw", type=int, default=1536)
    parser.add_argument("--opt-n", type=int, default=512)
    parser.add_argument("--max-n", type=int, default=2048)
    parser.add_argument("--workspace-gb", type=int, default=4)
    args = parser.parse_args()

    sp_onnx = args.onnx_dir / "superpoint.onnx"
    lg_onnx = args.onnx_dir / "superpoint_lightglue.onnx"
    if not sp_onnx.exists() or not lg_onnx.exists():
        raise SystemExit(f"missing ONNX under {args.onnx_dir}; run export_onnx.py first")

    h0, w0 = args.min_hw, args.min_hw
    h1, w1 = args.opt_h, args.opt_w
    h2, w2 = args.max_hw, args.max_hw

    try:
        import tensorrt as trt  # noqa: F401

        build_with_tensorrt(
            sp_onnx,
            args.out / "superpoint.engine",
            [("image", (1, 1, h0, w0), (1, 1, h1, w1), (1, 1, h2, w2))],
            args.workspace_gb,
        )
        n0, n1, n2 = 1, args.opt_n, args.max_n
        build_with_tensorrt(
            lg_onnx,
            args.out / "superpoint_lightglue.engine",
            [
                ("kpts0", (1, n0, 2), (1, n1, 2), (1, n2, 2)),
                ("kpts1", (1, n0, 2), (1, n1, 2), (1, n2, 2)),
                ("desc0", (1, n0, 256), (1, n1, 256), (1, n2, 256)),
                ("desc1", (1, n0, 256), (1, n1, 256), (1, n2, 256)),
                ("image_size0", (1, 2), (1, 2), (1, 2)),
                ("image_size1", (1, 2), (1, 2), (1, 2)),
            ],
            args.workspace_gb,
        )
        return
    except ImportError:
        print("python package tensorrt not found; trying trtexec")

    try:
        build_with_trtexec(
            sp_onnx,
            args.out / "superpoint.engine",
            f"image:1x1x{h0}x{w0}",
            f"image:1x1x{h1}x{w1}",
            f"image:1x1x{h2}x{w2}",
        )
        n0, n1, n2 = 1, args.opt_n, args.max_n
        build_with_trtexec(
            lg_onnx,
            args.out / "superpoint_lightglue.engine",
            f"kpts0:1x{n0}x2,kpts1:1x{n0}x2,desc0:1x{n0}x256,desc1:1x{n0}x256,image_size0:1x2,image_size1:1x2",
            f"kpts0:1x{n1}x2,kpts1:1x{n1}x2,desc0:1x{n1}x256,desc1:1x{n1}x256,image_size0:1x2,image_size1:1x2",
            f"kpts0:1x{n2}x2,kpts1:1x{n2}x2,desc0:1x{n2}x256,desc1:1x{n2}x256,image_size0:1x2,image_size1:1x2",
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            "Neither tensorrt Python package nor trtexec is available. "
            "Install TensorRT or skip this step; ONNX Runtime still works."
        ) from exc


if __name__ == "__main__":
    main()
