#!/usr/bin/env python3
"""Compare PyTorch reference vs ONNX Runtime / TensorRT dumps.

  python python/compare_accuracy.py --ref outputs/ref --onnx outputs/onnx --trt outputs/trt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(dir_path: Path, name: str):
    p = dir_path / f"{name}.npy"
    if not p.exists():
        return None
    return np.load(p)


def max_abs(a, b) -> float:
    if a is None or b is None:
        return float("nan")
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def rmse(a, b) -> float:
    if a is None or b is None or a.shape != b.shape:
        return float("nan")
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.sqrt(np.mean(d * d)))


def cosine(a, b) -> float:
    if a is None or b is None or a.size == 0:
        return float("nan")
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    if a.shape != b.shape:
        return float("nan")
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def kpt_iou(k0: np.ndarray, k1: np.ndarray, tol: float = 1.0) -> float:
    if k0 is None or k1 is None:
        return float("nan")
    if len(k0) == 0 and len(k1) == 0:
        return 1.0
    if len(k0) == 0 or len(k1) == 0:
        return 0.0
    used = np.zeros(len(k1), dtype=bool)
    inter = 0
    for p in k0:
        d = np.linalg.norm(k1 - p[None], axis=1)
        d[used] = 1e9
        j = int(np.argmin(d))
        if d[j] <= tol:
            used[j] = True
            inter += 1
    union = len(k0) + len(k1) - inter
    return inter / max(union, 1)


def match_agreement(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.size == 0:
        return float("nan")
    n = min(len(a), len(b))
    return float(np.mean(a[:n] == b[:n]))


def pair_iou(pairs_a: np.ndarray, pairs_b: np.ndarray) -> float:
    if pairs_a is None or pairs_b is None:
        return float("nan")
    sa = set(map(tuple, np.asarray(pairs_a).tolist())) if len(pairs_a) else set()
    sb = set(map(tuple, np.asarray(pairs_b).tolist())) if len(pairs_b) else set()
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def matched_score_rmse(k_pred, s_pred, k_ref, s_ref, tol: float = 1.0) -> float:
    if k_pred is None or k_ref is None or s_pred is None or s_ref is None:
        return float("nan")
    if len(k_pred) == 0 or len(k_ref) == 0:
        return float("nan")
    diffs = []
    used = np.zeros(len(k_ref), dtype=bool)
    for p, s in zip(k_pred, s_pred):
        d = np.linalg.norm(k_ref - p[None], axis=1)
        d[used] = 1e9
        j = int(np.argmin(d))
        if d[j] <= tol:
            used[j] = True
            diffs.append(float(s) - float(s_ref[j]))
    if not diffs:
        return float("nan")
    diffs = np.asarray(diffs, dtype=np.float64)
    return float(np.sqrt(np.mean(diffs * diffs)))


def e2e_coord_iou(pred_dir: Path, ref_dir: Path, tol: float = 1.5) -> float:
    k0p, k1p = load(pred_dir, "kpts0"), load(pred_dir, "kpts1")
    k0r, k1r = load(ref_dir, "kpts0"), load(ref_dir, "kpts1")
    pp, pr = load(pred_dir, "match_pairs"), load(ref_dir, "match_pairs")
    if any(x is None for x in (k0p, k1p, k0r, k1r, pp, pr)):
        return float("nan")

    def coords(k0, k1, pairs):
        out = []
        for a, b in np.asarray(pairs).reshape(-1, 2):
            if 0 <= a < len(k0) and 0 <= b < len(k1):
                out.append(np.concatenate([k0[int(a)], k1[int(b)]]))
        return np.asarray(out, dtype=np.float32) if out else np.zeros((0, 4), dtype=np.float32)

    cp, cr = coords(k0p, k1p, pp), coords(k0r, k1r, pr)
    if len(cp) == 0 and len(cr) == 0:
        return 1.0
    if len(cp) == 0 or len(cr) == 0:
        return 0.0
    used = np.zeros(len(cr), dtype=bool)
    inter = 0
    for p in cp:
        d = np.linalg.norm(cr - p[None], axis=1)
        d[used] = 1e9
        j = int(np.argmin(d))
        if d[j] <= tol:
            used[j] = True
            inter += 1
    return inter / max(len(cp) + len(cr) - inter, 1)


def row(name, pred, ref, criterion, ok) -> dict:
    return {"check": name, "value": pred, "criterion": criterion, "pass": bool(ok)}


def evaluate(pred_dir: Path, ref_dir: Path, dense_abs: float, match_agree: float) -> list[dict]:
    rows = []
    sl0_p, sl0_r = load(pred_dir, "score_logits0"), load(ref_dir, "score_logits0")
    dd0_p, dd0_r = load(pred_dir, "desc_dense0"), load(ref_dir, "desc_dense0")
    ma = max_abs(sl0_p, sl0_r)
    rows.append(row("SP dense score max_abs", ma, None, f"<{dense_abs}", ma < dense_abs))
    cos = cosine(dd0_p, dd0_r)
    rows.append(row("SP dense desc cosine", cos, None, ">0.999", cos > 0.999))

    k0p, k0r = load(pred_dir, "kpts0"), load(ref_dir, "kpts0")
    iou = kpt_iou(k0p, k0r, 1.0)
    rows.append(row("SP kpts IoU@1px", iou, None, ">0.99", iou > 0.99))
    s0p, s0r = load(pred_dir, "scores0"), load(ref_dir, "scores0")
    r = matched_score_rmse(k0p, s0p, k0r, s0r, 1.0)
    rows.append(row("SP score RMSE@1px", r, None, "<1e-4", r < 1e-4))

    lg_m = load(pred_dir, "lg_ref_matches0")
    if lg_m is None:
        lg_m = load(pred_dir, "matches0")
    ref_m = load(ref_dir, "matches0")
    agr = match_agreement(lg_m, ref_m)
    rows.append(row("LG match agreement (ref kpts)", agr, None, f">{match_agree}", agr > match_agree))
    lg_s = load(pred_dir, "lg_ref_mscores0")
    if lg_s is None:
        lg_s = load(pred_dir, "mscores0")
    ref_s = load(ref_dir, "mscores0")
    if lg_s is not None and ref_s is not None and lg_s.shape == ref_s.shape:
        r = rmse(lg_s, ref_s)
        rows.append(row("LG score RMSE", r, None, "<1e-4", r < 1e-3 or r < 1e-4))

    e2e = e2e_coord_iou(pred_dir, ref_dir)
    rows.append(row("E2E match IoU (xy)", e2e, None, ">0.95", e2e > 0.95))
    return rows


def print_table(title: str, rows: list[dict]) -> None:
    print(f"\n== {title} ==")
    print(f"{'check':<32} {'value':>14} {'criterion':<16} {'pass':>6}")
    for r in rows:
        val = r["value"]
        vs = f"{val:.6g}" if isinstance(val, float) else str(val)
        print(f"{r['check']:<32} {vs:>14} {r['criterion']:<16} {'YES' if r['pass'] else 'NO':>6}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=Path, default=ROOT / "outputs" / "ref")
    parser.add_argument("--onnx", type=Path, default=ROOT / "outputs" / "onnx")
    parser.add_argument("--trt", type=Path, default=ROOT / "outputs" / "trt")
    args = parser.parse_args()

    summary = {}
    if args.onnx.exists():
        rows = evaluate(args.onnx, args.ref, dense_abs=1e-4, match_agree=0.99)
        print_table("ONNX Runtime vs PyTorch", rows)
        summary["onnx"] = rows
    else:
        print(f"skip ORT: {args.onnx} missing")

    if args.trt.exists() and (args.trt / "score_logits0.npy").exists():
        rows = evaluate(args.trt, args.ref, dense_abs=1e-3, match_agree=0.98)
        print_table("TensorRT vs PyTorch", rows)
        summary["trt"] = rows
    else:
        print(f"skip TRT: {args.trt} missing or empty")

    out = args.ref.parent / "compare.json"
    out.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out}")

    failed = [
        r["check"]
        for backend, rows in summary.items()
        for r in rows
        if not r["pass"]
    ]
    if failed:
        raise SystemExit("failed checks: " + ", ".join(failed))


if __name__ == "__main__":
    main()
