#!/usr/bin/env python3
"""HPatches timing + MMA/homography AUC for PyTorch, C++ ORT, and C++ TensorRT.

  conda activate deep_matching
  python python/bench_hpatches.py --out outputs/hpatches
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from splg_common import (  # noqa: E402
    kpts_to_original,
    load_models,
    prepare_pair,
    save_npy_dir,
    to_nchw,
)

DEFAULT_HPATCHES = Path(
    "/home/libing/source/cv/deep_matching/data/hpatches-sequences-release"
)
DEFAULT_VIZ = (
    "i_crownday/1-2",
    "i_crownday/1-6",
    "v_bark/1-2",
    "v_bark/1-6",
    "i_books/1-3",
    "v_bricks/1-4",
)
MMA_THRESHOLDS = (1, 3, 5)
HOMO_THRESHOLDS = (3, 5, 10)


def pair_dir_name(pair_id: str) -> str:
    return pair_id.replace("/", "_")


def default_hpatches_root() -> Path:
    try:
        from deep_matching.data.hpatches import default_hpatches_root as _root

        p = Path(_root())
        if p.exists():
            return p
    except Exception:
        pass
    return DEFAULT_HPATCHES


def iter_hpatches(root: Path, max_pairs: int | None = None) -> list[dict[str, Any]]:
    try:
        from deep_matching.data.hpatches import iter_hpatches as _iter

        pairs = []
        for p in _iter(root, max_pairs=max_pairs):
            pairs.append(
                {
                    "pair_id": p.pair_id,
                    "scene": p.scene,
                    "split": p.split,
                    "image0": Path(p.image0_path),
                    "image1": Path(p.image1_path),
                    "H_gt": np.asarray(p.H_gt, dtype=np.float64),
                }
            )
        return pairs
    except Exception:
        return _iter_hpatches_local(root, max_pairs)


def _iter_hpatches_local(root: Path, max_pairs: int | None) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    scenes = sorted(p for p in root.iterdir() if p.is_dir() and p.name[:2] in {"i_", "v_"})
    for scene_dir in scenes:
        split = "illumination" if scene_dir.name.startswith("i_") else "viewpoint"
        ref = _find_frame(scene_dir, 1)
        if ref is None:
            continue
        for idx in range(2, 7):
            tgt = _find_frame(scene_dir, idx)
            h_path = scene_dir / f"H_1_{idx}"
            if tgt is None or not h_path.exists():
                continue
            pairs.append(
                {
                    "pair_id": f"{scene_dir.name}/1-{idx}",
                    "scene": scene_dir.name,
                    "split": split,
                    "image0": ref,
                    "image1": tgt,
                    "H_gt": np.loadtxt(h_path).reshape(3, 3),
                }
            )
            if max_pairs is not None and len(pairs) >= max_pairs:
                return pairs
    return pairs


def _find_frame(scene_dir: Path, index: int) -> Path | None:
    for ext in (".ppm", ".png", ".jpg", ".jpeg"):
        path = scene_dir / f"{index}{ext}"
        if path.exists():
            return path
    return None


def warp_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return pts
    ones = np.ones((len(pts), 1), dtype=np.float64)
    proj = (H @ np.hstack([pts, ones]).T).T
    return proj[:, :2] / np.clip(proj[:, 2:3], 1e-12, None)


def match_errors(k0: np.ndarray, k1: np.ndarray, H_gt: np.ndarray) -> np.ndarray:
    if len(k0) == 0:
        return np.zeros((0,), dtype=np.float64)
    return np.linalg.norm(warp_points(H_gt, k0) - np.asarray(k1, dtype=np.float64), axis=1)


def mean_matching_accuracy(errors: np.ndarray, thresholds: Iterable[float]) -> dict[str, float]:
    if len(errors) == 0:
        return {f"mma@{int(t)}px": 0.0 for t in thresholds}
    return {f"mma@{int(t)}px": float(np.mean(errors < t)) for t in thresholds}


def estimate_homography(k0: np.ndarray, k1: np.ndarray, ransac_th: float = 2.0):
    if len(k0) < 4:
        return None, 0
    import cv2

    H, mask = cv2.findHomography(
        np.asarray(k0, dtype=np.float64),
        np.asarray(k1, dtype=np.float64),
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=ransac_th,
        confidence=0.999,
        maxIters=10000,
    )
    inliers = int(mask.sum()) if mask is not None else 0
    return H, inliers


def homography_corner_error(H_est: np.ndarray, H_gt: np.ndarray, width: int, height: int) -> float:
    corners = np.array(
        [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
        dtype=np.float64,
    )
    err = np.linalg.norm(warp_points(H_est, corners) - warp_points(H_gt, corners), axis=1)
    return float(err.mean())


def error_auc(errors: Iterable[float], thresholds: Iterable[float]) -> dict[str, float]:
    values = [0.0] + sorted(float(x) for x in errors)
    recall = list(np.linspace(0.0, 1.0, len(values)))
    aucs: dict[str, float] = {}
    trapz = getattr(np, "trapezoid", np.trapz)
    for thr in thresholds:
        last_index = int(np.searchsorted(values, thr))
        tag = str(int(thr)) if float(thr).is_integer() else str(thr)
        if last_index == 0:
            aucs[f"auc@{tag}"] = 0.0
            continue
        y = recall[:last_index] + [recall[last_index - 1]]
        x = values[:last_index] + [float(thr)]
        aucs[f"auc@{tag}"] = float(trapz(y, x) / float(thr))
    return aucs


def evaluate_pair(
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    matches0: np.ndarray,
    H_gt: np.ndarray,
    orig_hw0: tuple[int, int],
) -> dict[str, Any]:
    valid = np.asarray(matches0) >= 0
    idx0 = np.where(valid)[0]
    idx1 = np.asarray(matches0)[valid].astype(np.int64)
    mk0 = np.asarray(kpts0, dtype=np.float64)[idx0] if len(idx0) else np.zeros((0, 2))
    mk1 = np.asarray(kpts1, dtype=np.float64)[idx1] if len(idx1) else np.zeros((0, 2))
    errors = match_errors(mk0, mk1, H_gt)
    H_est, n_inliers = estimate_homography(mk0, mk1)
    height, width = orig_hw0
    corner = float("inf") if H_est is None else homography_corner_error(H_est, H_gt, width, height)
    return {
        "n_matches": int(len(idx0)),
        "n_inliers": int(n_inliers),
        "corner_error": corner,
        **mean_matching_accuracy(errors, MMA_THRESHOLDS),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"n_pairs": 0}

    def _mean(keys: list[str], rows: list[dict[str, Any]]) -> dict[str, float]:
        return {k: float(sum(r[k] for r in rows) / len(rows)) for k in keys}

    mma_keys = [f"mma@{t}px" for t in MMA_THRESHOLDS]
    time_keys = [k for k in ("ms_sp", "ms_lg", "ms_e2e") if k in records[0]]
    timed = [r for r in records if not r.get("warmup")]
    if not timed:
        timed = records

    summary: dict[str, Any] = {
        "n_pairs": len(records),
        "mean_matches": float(sum(r["n_matches"] for r in records) / len(records)),
        **_mean(mma_keys, records),
        **error_auc([r["corner_error"] for r in records], HOMO_THRESHOLDS),
    }
    if time_keys and timed:
        summary.update(_mean(time_keys, timed))
        summary["n_timed"] = len(timed)

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_split[row["split"]].append(row)
    summary["splits"] = {}
    for name, rows in by_split.items():
        trows = [r for r in rows if not r.get("warmup")] or rows
        split = {
            "n_pairs": len(rows),
            "mean_matches": float(sum(r["n_matches"] for r in rows) / len(rows)),
            **_mean(mma_keys, rows),
            **error_auc([r["corner_error"] for r in rows], HOMO_THRESHOLDS),
        }
        if time_keys:
            split.update(_mean(time_keys, trows))
        summary["splits"][name] = split
    return summary


def matched_xy(k0: np.ndarray, k1: np.ndarray, matches0: np.ndarray) -> np.ndarray:
    valid = np.asarray(matches0) >= 0
    i0 = np.where(valid)[0]
    if len(i0) == 0:
        return np.zeros((0, 4), dtype=np.float64)
    i1 = np.asarray(matches0)[valid].astype(np.int64)
    return np.concatenate([k0[i0], k1[i1]], axis=1)


def match_iou(a: np.ndarray, b: np.ndarray, thr: float = 1.0) -> float:
    if len(a) == 0 and len(b) == 0:
        return 1.0
    if len(a) == 0 or len(b) == 0:
        return 0.0
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    used = np.zeros(len(b), dtype=bool)
    hit = 0
    for i in np.argsort(d.min(axis=1)):
        j = int(np.argmin(d[i]))
        if d[i, j] <= thr and not used[j]:
            used[j] = True
            hit += 1
    return float(hit) / float(len(a) + len(b) - hit)


def sync_cuda(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def run_pytorch(
    pairs: list[dict[str, Any]],
    work: dict[str, dict[str, Any]],
    out_dir: Path,
    device: str,
    max_keypoints: int,
    warmup: int,
) -> list[dict[str, Any]]:
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    sp, lg = load_models(device=device, max_keypoints=max_keypoints)
    records = []
    for i, pair in enumerate(pairs):
        prep = work[pair["pair_id"]]
        t0 = to_nchw(prep["gray0"]).to(device)
        t1 = to_nchw(prep["gray1"]).to(device)
        sync_cuda(device)
        t_a = time.perf_counter()
        with torch.no_grad():
            f0 = sp.forward({"image": t0})
            f1 = sp.forward({"image": t1})
            sync_cuda(device)
            t_b = time.perf_counter()
            hw0 = torch.tensor(
                [[prep["gray0"].shape[1], prep["gray0"].shape[0]]],
                dtype=torch.float32,
                device=device,
            )
            hw1 = torch.tensor(
                [[prep["gray1"].shape[1], prep["gray1"].shape[0]]],
                dtype=torch.float32,
                device=device,
            )
            f0["image_size"] = hw0
            f1["image_size"] = hw1
            pred = lg({"image0": f0, "image1": f1})
            sync_cuda(device)
            t_c = time.perf_counter()

        k0 = f0["keypoints"][0].detach().cpu().numpy().astype(np.float32)
        k1 = f1["keypoints"][0].detach().cpu().numpy().astype(np.float32)
        m0 = pred["matches0"][0].detach().cpu().numpy().astype(np.int64)
        ms = pred["matching_scores0"][0].detach().cpu().numpy().astype(np.float32)
        rec = {
            "pair_id": pair["pair_id"],
            "scene": pair["scene"],
            "split": pair["split"],
            "backend": "pytorch",
            "device": device,
            "ms_sp": (t_b - t_a) * 1000.0,
            "ms_lg": (t_c - t_b) * 1000.0,
            "ms_e2e": (t_c - t_a) * 1000.0,
            "warmup": i < warmup,
            "n0": int(k0.shape[0]),
            "n1": int(k1.shape[0]),
            **evaluate_pair(
                kpts_to_original(k0, prep["scale0"]),
                kpts_to_original(k1, prep["scale1"]),
                m0,
                pair["H_gt"],
                prep["orig_hw0"],
            ),
        }
        records.append(rec)
        pdir = out_dir / pair_dir_name(pair["pair_id"])
        save_npy_dir(
            pdir,
            {"kpts0": k0, "kpts1": k1, "matches0": m0, "mscores0": ms},
        )
        print(
            f"[pytorch {i + 1}/{len(pairs)}] {pair['pair_id']}  "
            f"N0={rec['n0']} N1={rec['n1']} matches={rec['n_matches']}  "
            f"SP={rec['ms_sp']:.2f} ms LG={rec['ms_lg']:.2f} ms e2e={rec['ms_e2e']:.2f} ms"
            f"{' (warmup)' if rec['warmup'] else ''}"
        )
    (out_dir / "results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    return records


def write_work_and_list(
    pairs: list[dict[str, Any]],
    work: dict[str, dict[str, Any]],
    work_dir: Path,
) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    list_path = work_dir / "pairs.txt"
    lines = []
    for pair in pairs:
        pid = pair_dir_name(pair["pair_id"])
        pdir = work_dir / pid
        pdir.mkdir(parents=True, exist_ok=True)
        prep = work[pair["pair_id"]]
        np.save(pdir / "image0.npy", np.ascontiguousarray(prep["gray0"].astype(np.float32)))
        np.save(pdir / "image1.npy", np.ascontiguousarray(prep["gray1"].astype(np.float32)))
        meta = {
            "pair_id": pair["pair_id"],
            "scene": pair["scene"],
            "split": pair["split"],
            "orig_hw0": list(prep["orig_hw0"]),
            "orig_hw1": list(prep["orig_hw1"]),
            "scale0": prep["scale0"].tolist(),
            "scale1": prep["scale1"].tolist(),
            "work_hw0": [int(prep["gray0"].shape[0]), int(prep["gray0"].shape[1])],
            "work_hw1": [int(prep["gray1"].shape[0]), int(prep["gray1"].shape[1])],
        }
        (pdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        lines.append(f"{pid} {pdir / 'image0.npy'} {pdir / 'image1.npy'}\n")
    list_path.write_text("".join(lines), encoding="utf-8")
    return list_path


def run_cpp_backend(
    exe: Path,
    sp_path: Path,
    lg_path: Path,
    list_path: Path,
    out_dir: Path,
    max_keypoints: int,
    warmup: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(exe),
        "--sp",
        str(sp_path),
        "--lg",
        str(lg_path),
        "--list",
        str(list_path),
        "--out",
        str(out_dir),
        "--max-keypoints",
        str(max_keypoints),
        "--warmup",
        str(warmup),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def load_cpp_records(
    pairs: list[dict[str, Any]],
    work: dict[str, dict[str, Any]],
    out_dir: Path,
    backend: str,
) -> list[dict[str, Any]]:
    times = {}
    jsonl = out_dir / "times.jsonl"
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                times[rec["pair_id"]] = rec
    records = []
    for pair in pairs:
        pid = pair_dir_name(pair["pair_id"])
        pdir = out_dir / pid
        k0 = np.load(pdir / "kpts0.npy")
        k1 = np.load(pdir / "kpts1.npy")
        m0 = np.load(pdir / "matches0.npy")
        prep = work[pair["pair_id"]]
        t = times.get(pid, {})
        rec = {
            "pair_id": pair["pair_id"],
            "scene": pair["scene"],
            "split": pair["split"],
            "backend": backend,
            "n0": int(k0.shape[0]),
            "n1": int(k1.shape[0]),
            "ms_sp": float(t.get("ms_sp", 0.0)),
            "ms_lg": float(t.get("ms_lg", 0.0)),
            "ms_e2e": float(t.get("ms_e2e", 0.0)),
            "warmup": bool(t.get("warmup", 0)),
            **evaluate_pair(
                kpts_to_original(k0, prep["scale0"]),
                kpts_to_original(k1, prep["scale1"]),
                m0,
                pair["H_gt"],
                prep["orig_hw0"],
            ),
        }
        records.append(rec)
    (out_dir / "results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    return records


def draw_matches(
    rgb0: np.ndarray,
    rgb1: np.ndarray,
    k0: np.ndarray,
    k1: np.ndarray,
    matches0: np.ndarray,
    scores0: np.ndarray | None,
    title: str,
    max_draw: int = 200,
) -> np.ndarray:
    import cv2

    def _to_bgr(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        if img.dtype != np.uint8:
            img = np.clip(img * 255.0 if img.max() <= 1.5 else img, 0, 255).astype(np.uint8)
        if img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    a = _to_bgr(rgb0)
    b = _to_bgr(rgb1)
    h = max(a.shape[0], b.shape[0])
    canvas = np.zeros((h, a.shape[1] + b.shape[1], 3), dtype=np.uint8)
    canvas[: a.shape[0], : a.shape[1]] = a
    canvas[: b.shape[0], a.shape[1] : a.shape[1] + b.shape[1]] = b

    valid = np.asarray(matches0) >= 0
    idx0 = np.where(valid)[0]
    if len(idx0) == 0:
        cv2.putText(canvas, f"{title}  0 matches", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return canvas
    idx1 = np.asarray(matches0)[valid].astype(np.int64)
    sc = np.asarray(scores0)[valid] if scores0 is not None else np.ones(len(idx0), dtype=np.float32)
    order = np.argsort(-sc)
    if max_draw > 0:
        order = order[:max_draw]
    for j in order:
        p0 = k0[idx0[j]]
        p1 = k1[idx1[j]]
        x0, y0 = int(round(float(p0[0]))), int(round(float(p0[1])))
        x1, y1 = int(round(float(p1[0]))) + a.shape[1], int(round(float(p1[1])))
        t = float(sc[j])
        color = (
            int(255 * (1.0 - t)),
            int(80 + 175 * t),
            int(40 + 40 * (1.0 - t)),
        )
        cv2.line(canvas, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
        cv2.circle(canvas, (x0, y0), 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x1, y1), 2, color, -1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"{title}  {len(idx0)} matches (draw {len(order)})",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def save_viz(
    pair: dict[str, Any],
    prep: dict[str, Any],
    backend_dirs: dict[str, Path],
    viz_dir: Path,
) -> None:
    import cv2

    viz_dir.mkdir(parents=True, exist_ok=True)
    pid = pair_dir_name(pair["pair_id"])
    rows = []
    for name, bdir in backend_dirs.items():
        pdir = bdir / pid
        if not (pdir / "kpts0.npy").exists():
            continue
        k0 = np.load(pdir / "kpts0.npy")
        k1 = np.load(pdir / "kpts1.npy")
        m0 = np.load(pdir / "matches0.npy")
        ms = np.load(pdir / "mscores0.npy") if (pdir / "mscores0.npy").exists() else None
        img = draw_matches(prep["rgb0"], prep["rgb1"], k0, k1, m0, ms, f"{pair['pair_id']}  {name}")
        out = viz_dir / f"{pid}_{name}.png"
        cv2.imwrite(str(out), img)
        print(f"wrote {out}")
        rows.append(img)
    if len(rows) >= 2:
        target_w = max(im.shape[1] for im in rows)
        padded = []
        for im in rows:
            if im.shape[1] == target_w:
                padded.append(im)
            else:
                canvas = np.zeros((im.shape[0], target_w, 3), dtype=np.uint8)
                canvas[:, : im.shape[1]] = im
                padded.append(canvas)
        stacked = np.vstack(padded)
        out = viz_dir / f"{pid}_compare.png"
        cv2.imwrite(str(out), stacked)
        print(f"wrote {out}")


def fmt_table(summaries: dict[str, dict[str, Any]]) -> str:
    keys = [
        ("backend", None),
        ("n_pairs", "n_pairs"),
        ("matches", "mean_matches"),
        ("mma@1", "mma@1px"),
        ("mma@3", "mma@3px"),
        ("mma@5", "mma@5px"),
        ("auc@3", "auc@3"),
        ("auc@5", "auc@5"),
        ("auc@10", "auc@10"),
        ("SP ms", "ms_sp"),
        ("LG ms", "ms_lg"),
        ("e2e ms", "ms_e2e"),
        ("IoU vs PT", "match_iou_vs_pt"),
    ]
    rows = []
    header = [k[0] for k in keys]
    rows.append(header)
    for name, s in summaries.items():
        row = [name]
        for _, key in keys[1:]:
            v = s.get(key)
            if v is None:
                row.append("-")
            elif isinstance(v, float):
                row.append(
                    f"{v:.4f}"
                    if any(s in key for s in ("mma", "auc", "matches", "iou"))
                    else f"{v:.2f}"
                )
            else:
                row.append(str(v))
        rows.append(row)
    widths = [max(len(r[c]) for r in rows) for c in range(len(header))]
    lines = []
    for i, row in enumerate(rows):
        line = " | ".join(cell.ljust(widths[c]) for c, cell in enumerate(row))
        lines.append(line)
        if i == 0:
            lines.append("-+-".join("-" * w for w in widths))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpatches", type=Path, default=default_hpatches_root())
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "hpatches")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--bin-dir", type=Path, default=ROOT / "build" / "bin")
    parser.add_argument("--backends", default="pytorch,onnx,trt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resize-short", type=int, default=480)
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--viz-pairs", default=",".join(DEFAULT_VIZ))
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    pairs = iter_hpatches(args.hpatches, max_pairs=args.max_pairs)
    if not pairs:
        raise SystemExit(f"no HPatches pairs under {args.hpatches}")
    print(f"HPatches {args.hpatches}  n_pairs={len(pairs)}  backends={backends}  pt_device={device}")

    args.out.mkdir(parents=True, exist_ok=True)
    work_dir = args.out / "work"
    work: dict[str, dict[str, Any]] = {}
    for i, pair in enumerate(pairs):
        work[pair["pair_id"]] = prepare_pair(
            str(pair["image0"]), str(pair["image1"]), args.resize_short
        )
        if (i + 1) % 50 == 0 or i + 1 == len(pairs):
            print(f"preprocessed {i + 1}/{len(pairs)}")

    results: dict[str, list[dict[str, Any]]] = {}
    backend_dirs: dict[str, Path] = {}

    if "pytorch" in backends:
        backend_dirs["pytorch"] = args.out / "pytorch"
        results["pytorch"] = run_pytorch(
            pairs, work, backend_dirs["pytorch"], device, args.max_keypoints, args.warmup
        )

    need_cpp = [b for b in backends if b in {"onnx", "trt"}]
    list_path = None
    if need_cpp:
        list_path = write_work_and_list(pairs, work, work_dir)

    if "onnx" in backends:
        exe = args.bin_dir / "bench_onnx"
        if not exe.exists():
            raise SystemExit(f"missing {exe}; build with cmake --build build -j")
        backend_dirs["onnx"] = args.out / "onnx"
        run_cpp_backend(
            exe,
            args.models / "superpoint.onnx",
            args.models / "superpoint_lightglue.onnx",
            list_path,
            backend_dirs["onnx"],
            args.max_keypoints,
            args.warmup,
        )
        results["onnx"] = load_cpp_records(pairs, work, backend_dirs["onnx"], "onnx")

    if "trt" in backends:
        exe = args.bin_dir / "bench_trt"
        if not exe.exists():
            raise SystemExit(f"missing {exe}; build TensorRT target bench_trt")
        backend_dirs["trt"] = args.out / "trt"
        run_cpp_backend(
            exe,
            args.models / "superpoint.engine",
            args.models / "superpoint_lightglue.engine",
            list_path,
            backend_dirs["trt"],
            args.max_keypoints,
            args.warmup,
        )
        results["trt"] = load_cpp_records(pairs, work, backend_dirs["trt"], "trt")

    summaries = {name: summarize(recs) for name, recs in results.items()}
    if "pytorch" in results:
        pt_xy = {
            r["pair_id"]: matched_xy(
                np.load(backend_dirs["pytorch"] / pair_dir_name(r["pair_id"]) / "kpts0.npy"),
                np.load(backend_dirs["pytorch"] / pair_dir_name(r["pair_id"]) / "kpts1.npy"),
                np.load(backend_dirs["pytorch"] / pair_dir_name(r["pair_id"]) / "matches0.npy"),
            )
            for r in results["pytorch"]
        }
        for name in ("onnx", "trt"):
            if name not in results:
                continue
            ious = []
            for r in results[name]:
                pdir = backend_dirs[name] / pair_dir_name(r["pair_id"])
                xy = matched_xy(
                    np.load(pdir / "kpts0.npy"),
                    np.load(pdir / "kpts1.npy"),
                    np.load(pdir / "matches0.npy"),
                )
                ious.append(match_iou(pt_xy[r["pair_id"]], xy))
            summaries[name]["match_iou_vs_pt"] = float(np.mean(ious)) if ious else 0.0

    viz_ids = [s.strip() for s in args.viz_pairs.split(",") if s.strip()]
    pair_by_id = {p["pair_id"]: p for p in pairs}
    for vid in viz_ids:
        if vid not in pair_by_id:
            print(f"skip viz {vid}: not in this run")
            continue
        save_viz(pair_by_id[vid], work[vid], backend_dirs, args.out / "viz")

    report = {
        "hpatches": str(args.hpatches),
        "n_pairs": len(pairs),
        "resize_short": args.resize_short,
        "max_keypoints": args.max_keypoints,
        "pytorch_device": device,
        "onnx_device": "cpu",
        "trt_device": "cuda",
        "summaries": summaries,
        "pairs": {name: recs for name, recs in results.items()},
    }
    (args.out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    table = fmt_table(summaries)
    print("\nHPatches SuperPoint + LightGlue\n")
    print(table)
    for name, s in summaries.items():
        if "splits" not in s:
            continue
        print(f"\n{name} by split")
        print(fmt_table(s["splits"]))
    print(f"\nwrote {args.out / 'summary.json'}")
    print(f"viz {args.out / 'viz'}")

    if list_path is not None and not args.keep_work:
        for npy in work_dir.glob("*/*.npy"):
            npy.unlink()


if __name__ == "__main__":
    main()
