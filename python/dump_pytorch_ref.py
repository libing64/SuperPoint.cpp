#!/usr/bin/env python3
"""Dump SuperPoint + LightGlue PyTorch reference tensors for accuracy comparison.

Run inside conda env deep_matching:
  python python/dump_pytorch_ref.py --out outputs/ref
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from splg_common import (  # noqa: E402
    SuperPointDense,
    load_models,
    load_or_make_pair,
    save_npy_dir,
    to_nchw,
)


def default_hpatches() -> tuple[str | None, str | None]:
    base = Path("/home/libing/source/cv/deep_matching/data/hpatches-sequences-release")
    p0 = base / "i_crownday" / "1.ppm"
    p1 = base / "i_crownday" / "2.ppm"
    if p0.exists() and p1.exists():
        return str(p0), str(p1)
    return None, None


@torch.no_grad()
def main() -> None:
    hp0, hp1 = default_hpatches()
    parser = argparse.ArgumentParser()
    parser.add_argument("--image0", default=hp0)
    parser.add_argument("--image1", default=hp1)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "ref")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resize-short", type=int, default=480)
    parser.add_argument("--max-keypoints", type=int, default=2048)
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    gray0, gray1 = load_or_make_pair(args.image0, args.image1, args.resize_short)
    t0 = to_nchw(gray0).to(device)
    t1 = to_nchw(gray1).to(device)

    sp, lg = load_models(device=device, max_keypoints=args.max_keypoints)
    dense = SuperPointDense(sp).eval()

    logits0, desc_dense0 = dense(t0)
    logits1, desc_dense1 = dense(t1)
    encoder0 = dense.encode(t0)
    encoder1 = dense.encode(t1)

    feats0 = sp.forward({"image": t0})
    feats1 = sp.forward({"image": t1})

    # LightGlue expects image_size as (W, H)
    hw0 = torch.tensor([[gray0.shape[1], gray0.shape[0]]], dtype=torch.float32, device=device)
    hw1 = torch.tensor([[gray1.shape[1], gray1.shape[0]]], dtype=torch.float32, device=device)
    feats0["image_size"] = hw0
    feats1["image_size"] = hw1

    pred = lg({"image0": feats0, "image1": feats1})

    kpts0 = feats0["keypoints"][0].cpu().numpy().astype(np.float32)
    kpts1 = feats1["keypoints"][0].cpu().numpy().astype(np.float32)
    scores0 = feats0["keypoint_scores"][0].cpu().numpy().astype(np.float32)
    scores1 = feats1["keypoint_scores"][0].cpu().numpy().astype(np.float32)
    desc0 = feats0["descriptors"][0].cpu().numpy().astype(np.float32)
    desc1 = feats1["descriptors"][0].cpu().numpy().astype(np.float32)
    matches0 = pred["matches0"][0].cpu().numpy().astype(np.int64)
    mscores0 = pred["matching_scores0"][0].cpu().numpy().astype(np.float32)

    valid = matches0 > -1
    match_pairs = np.stack([np.where(valid)[0], matches0[valid]], axis=1).astype(np.int64)
    match_scores = mscores0[valid].astype(np.float32)

    arrays = {
        "image0": t0.cpu().numpy().astype(np.float32),
        "image1": t1.cpu().numpy().astype(np.float32),
        "encoder0": encoder0.cpu().numpy().astype(np.float32),
        "encoder1": encoder1.cpu().numpy().astype(np.float32),
        "score_logits0": logits0.cpu().numpy().astype(np.float32),
        "score_logits1": logits1.cpu().numpy().astype(np.float32),
        "desc_dense0": desc_dense0.cpu().numpy().astype(np.float32),
        "desc_dense1": desc_dense1.cpu().numpy().astype(np.float32),
        "kpts0": kpts0,
        "kpts1": kpts1,
        "scores0": scores0,
        "scores1": scores1,
        "desc0": desc0,
        "desc1": desc1,
        "image_size0": hw0.cpu().numpy().astype(np.float32),
        "image_size1": hw1.cpu().numpy().astype(np.float32),
        "matches0": matches0,
        "mscores0": mscores0,
        "match_pairs": match_pairs,
        "match_scores": match_scores,
    }
    save_npy_dir(args.out, arrays)

    meta = {
        "backend": "pytorch",
        "device": device,
        "h0": int(gray0.shape[0]),
        "w0": int(gray0.shape[1]),
        "h1": int(gray1.shape[0]),
        "w1": int(gray1.shape[1]),
        "n0": int(kpts0.shape[0]),
        "n1": int(kpts1.shape[0]),
        "n_matches": int(match_pairs.shape[0]),
        "max_keypoints": args.max_keypoints,
        "resize_short": args.resize_short,
        "image0": args.image0,
        "image1": args.image1,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
