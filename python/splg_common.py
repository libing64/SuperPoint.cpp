"""Shared SuperPoint / LightGlue helpers for dump and ONNX export."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ITU_R = (0.299, 0.587, 0.114)


def pad_to_multiple(h: int, w: int, m: int = 8) -> tuple[int, int]:
    return ((h + m - 1) // m) * m, ((w + m - 1) // m) * m


def rgb_to_gray_itu(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image.astype(np.float32)
        if gray.max() > 1.5:
            gray = gray / 255.0
        return gray
    rgb = image.astype(np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    return ITU_R[0] * rgb[..., 0] + ITU_R[1] * rgb[..., 1] + ITU_R[2] * rgb[..., 2]


def resize_short_edge(image: np.ndarray, short: int) -> tuple[np.ndarray, np.ndarray]:
    """Resize so min(H, W) == short. Returns RGB/gray uint8 or float and [sx, sy]."""
    h, w = image.shape[:2]
    scale = float(short) / float(min(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    try:
        import cv2

        out = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    except Exception:
        try:
            from PIL import Image

            if image.ndim == 2:
                pil = Image.fromarray((image * 255).astype(np.uint8) if image.dtype != np.uint8 else image)
            else:
                pil = Image.fromarray(image)
            pil = pil.resize((new_w, new_h), Image.BILINEAR)
            out = np.asarray(pil)
        except Exception:
            ys = (np.linspace(0, image.shape[0] - 1, new_h)).astype(np.int64)
            xs = (np.linspace(0, image.shape[1] - 1, new_w)).astype(np.int64)
            out = image[ys][:, xs]
    sx = w / float(new_w)
    sy = h / float(new_h)
    return out, np.array([sx, sy], dtype=np.float32)


def pad_image(gray: np.ndarray, multiple: int = 8) -> np.ndarray:
    h, w = gray.shape[:2]
    ph, pw = pad_to_multiple(h, w, multiple)
    if (ph, pw) == (h, w):
        return gray
    out = np.zeros((ph, pw), dtype=gray.dtype)
    out[:h, :w] = gray
    return out


def load_or_make_pair(
    image0: Optional[str],
    image1: Optional[str],
    resize_short: Optional[int] = 480,
) -> tuple[np.ndarray, np.ndarray]:
    """Return two grayscale float32 images in [0, 1], H/W multiple of 8."""
    if image0 and image1 and Path(image0).exists() and Path(image1).exists():
        g0 = _read_any(image0)
        g1 = _read_any(image1)
    else:
        g0 = _checkerboard(480, 640, phase=0)
        g1 = _checkerboard(480, 640, phase=6)

    if resize_short:
        g0, _ = resize_short_edge(g0, resize_short)
        g1, _ = resize_short_edge(g1, resize_short)
        g0 = rgb_to_gray_itu(g0) if g0.ndim == 3 else rgb_to_gray_itu(g0)
        g1 = rgb_to_gray_itu(g1) if g1.ndim == 3 else rgb_to_gray_itu(g1)
    else:
        g0 = rgb_to_gray_itu(g0)
        g1 = rgb_to_gray_itu(g1)

    return pad_image(g0.astype(np.float32)), pad_image(g1.astype(np.float32))


def _read_any(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() in {".ppm", ".pgm", ".pbm"}:
        return _read_pnm(p)
    try:
        import cv2

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = img[:, :, :3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception:
        from PIL import Image

        return np.asarray(Image.open(path).convert("RGB"))


def _read_pnm(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        magic = f.readline().strip()
        def _tok():
            while True:
                line = f.readline()
                if not line:
                    raise ValueError(f"truncated PNM: {path}")
                if line.startswith(b"#"):
                    continue
                return line.split()

        if magic not in {b"P5", b"P6"}:
            raise ValueError(f"unsupported PNM {magic} in {path}")
        header = _tok()
        while len(header) < 3:
            header += _tok()
        w, h, maxval = int(header[0]), int(header[1]), int(header[2])
        data = f.read()
        if magic == b"P5":
            arr = np.frombuffer(data, dtype=np.uint8, count=w * h).reshape(h, w)
            return arr
        arr = np.frombuffer(data, dtype=np.uint8, count=w * h * 3).reshape(h, w, 3)
        return arr


def _checkerboard(h: int, w: int, cell: int = 32, phase: int = 0) -> np.ndarray:
    yy, xx = np.indices((h, w))
    board = ((yy // cell + xx // cell + phase) % 2).astype(np.float32)
    noise = (np.sin((yy + phase) * 0.07) * np.cos((xx - phase) * 0.05) + 1.0) * 0.15
    img = np.clip(board * 0.7 + 0.15 + noise, 0.0, 1.0)
    rgb = np.stack([img, img * 0.95, img * 0.9], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def to_nchw(gray: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(gray.astype(np.float32))[None, None]


def save_npy_dir(out_dir: Path, arrays: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, value in arrays.items():
        np.save(out_dir / f"{name}.npy", np.ascontiguousarray(value))


def load_models(
    device: str = "cpu",
    max_keypoints: int = 2048,
):
    from lightglue.lightglue import LightGlue
    from lightglue.superpoint import SuperPoint

    sp = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
    lg = LightGlue(
        features="superpoint",
        flash=False,
        depth_confidence=-1,
        width_confidence=-1,
        filter_threshold=0.1,
    ).eval().to(device)
    return sp, lg


class SuperPointDense(nn.Module):
    """CNN only: image -> score logits [1,65,h,w] and L2-normalized desc [1,256,h,w]."""

    def __init__(self, sp: nn.Module):
        super().__init__()
        self.relu = sp.relu
        self.pool = sp.pool
        self.conv1a = sp.conv1a
        self.conv1b = sp.conv1b
        self.conv2a = sp.conv2a
        self.conv2b = sp.conv2b
        self.conv3a = sp.conv3a
        self.conv3b = sp.conv3b
        self.conv4a = sp.conv4a
        self.conv4b = sp.conv4b
        self.convPa = sp.convPa
        self.convPb = sp.convPb
        self.convDa = sp.convDa
        self.convDb = sp.convDb

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1a(image))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        return x

    def forward(self, image: torch.Tensor):
        x = self.encode(image)
        score_logits = self.convPb(self.relu(self.convPa(x)))
        desc = F.normalize(self.convDb(self.relu(self.convDa(x))), p=2, dim=1)
        return score_logits, desc


def rotate_half_onnx(x: torch.Tensor) -> torch.Tensor:
    dim = x.shape[-1]
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2], dim // 2, 2)
    x1 = x[..., 0]
    x2 = x[..., 1]
    return torch.stack((-x2, x1), dim=-1).reshape(x.shape[0], x.shape[1], x.shape[2], dim)


def apply_rope(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return t * freqs[0] + rotate_half_onnx(t) * freqs[1]


def attn_matmul(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scale = q.shape[-1] ** -0.5
    sim = torch.matmul(q, k.permute(0, 1, 3, 2)) * scale
    attn = torch.softmax(sim, dim=3)
    return torch.matmul(attn, v)


class LightGlueExport(nn.Module):
    """ONNX / TRT friendly LightGlue: tensors in, matches0 + mscores0 out."""

    def __init__(self, lg: nn.Module):
        super().__init__()
        self.lg = lg
        self.filter_threshold = float(lg.conf.filter_threshold)
        self.n_layers = int(lg.conf.n_layers)
        self.num_heads = int(lg.conf.num_heads)
        self.descriptor_dim = int(lg.conf.descriptor_dim)
        self.head_dim = self.descriptor_dim // self.num_heads

    def _normalize_kpts(self, kpts: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
        shift = size / 2.0
        scale = torch.amax(size, dim=1, keepdim=True) / 2.0
        return (kpts - shift.unsqueeze(1)) / scale.unsqueeze(1)

    def _self_block(self, block, x: torch.Tensor, encoding: torch.Tensor) -> torch.Tensor:
        qkv = block.Wqkv(x)
        bsz, n_pts, _ = qkv.shape
        qkv = qkv.reshape(bsz, n_pts, self.num_heads, self.head_dim, 3)
        qkv = qkv.permute(0, 2, 1, 3, 4)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = apply_rope(encoding, q)
        k = apply_rope(encoding, k)
        context = attn_matmul(q, k, v)
        message = block.out_proj(
            context.permute(0, 2, 1, 3).reshape(bsz, n_pts, self.descriptor_dim)
        )
        return x + block.ffn(torch.cat([x, message], dim=-1))

    def _cross_block(self, block, x0: torch.Tensor, x1: torch.Tensor):
        qk0 = block.to_qk(x0)
        qk1 = block.to_qk(x1)
        v0 = block.to_v(x0)
        v1 = block.to_v(x1)
        bsz, n0, _ = qk0.shape
        n1 = qk1.shape[1]
        qk0 = qk0.reshape(bsz, n0, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        qk1 = qk1.reshape(bsz, n1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v0 = v0.reshape(bsz, n0, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v1 = v1.reshape(bsz, n1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        qk0 = qk0 * (block.scale**0.5)
        qk1 = qk1 * (block.scale**0.5)
        sim = torch.matmul(qk0, qk1.permute(0, 1, 3, 2))
        attn01 = torch.softmax(sim, dim=3)
        attn10 = torch.softmax(sim.permute(0, 1, 3, 2), dim=3)
        m0 = torch.matmul(attn01, v1)
        m1 = torch.matmul(attn10, v0)
        m0 = block.to_out(m0.permute(0, 2, 1, 3).reshape(bsz, n0, self.descriptor_dim))
        m1 = block.to_out(m1.permute(0, 2, 1, 3).reshape(bsz, n1, self.descriptor_dim))
        x0 = x0 + block.ffn(torch.cat([x0, m0], dim=-1))
        x1 = x1 + block.ffn(torch.cat([x1, m1], dim=-1))
        return x0, x1

    def _assignment_scores(self, desc0: torch.Tensor, desc1: torch.Tensor) -> torch.Tensor:
        assign = self.lg.log_assignment[-1]
        mdesc0 = assign.final_proj(desc0)
        mdesc1 = assign.final_proj(desc1)
        dim = mdesc0.shape[-1]
        mdesc0 = mdesc0 / (dim**0.25)
        mdesc1 = mdesc1 / (dim**0.25)
        sim = torch.matmul(mdesc0, mdesc1.permute(0, 2, 1))
        z0 = assign.matchability(desc0)
        z1 = assign.matchability(desc1)
        certainties = F.logsigmoid(z0) + F.logsigmoid(z1).permute(0, 2, 1)
        scores0 = F.log_softmax(sim, dim=2)
        scores1 = F.log_softmax(sim.permute(0, 2, 1), dim=2).permute(0, 2, 1)
        inner = scores0 + scores1 + certainties
        bin0 = F.logsigmoid(-z0.squeeze(-1)).unsqueeze(-1)
        bin1 = F.logsigmoid(-z1.squeeze(-1))
        row = torch.cat([inner, bin0], dim=-1)
        corner = inner.new_zeros(inner.shape[0], 1, 1)
        last = torch.cat([bin1.unsqueeze(1), corner], dim=-1)
        return torch.cat([row, last], dim=1)

    def _filter_matches(self, scores: torch.Tensor):
        inner = scores[:, :-1, :-1]
        max0 = inner.max(dim=2)
        max1 = inner.max(dim=1)
        m0 = max0.indices
        m1 = max1.indices
        n0 = m0.shape[1]
        indices0 = torch.arange(n0, dtype=torch.int64).unsqueeze(0).expand_as(m0)
        mutual0 = torch.eq(indices0, torch.gather(m1, 1, m0))
        mscores0 = torch.where(mutual0, max0.values.exp(), torch.zeros_like(max0.values))
        valid0 = mutual0 & (mscores0 > self.filter_threshold)
        m0 = torch.where(valid0, m0, torch.full_like(m0, -1))
        return m0.to(torch.int64), mscores0

    def forward(
        self,
        kpts0: torch.Tensor,
        kpts1: torch.Tensor,
        desc0: torch.Tensor,
        desc1: torch.Tensor,
        image_size0: torch.Tensor,
        image_size1: torch.Tensor,
    ):
        k0 = self._normalize_kpts(kpts0, image_size0)
        k1 = self._normalize_kpts(kpts1, image_size1)
        d0 = self.lg.input_proj(desc0)
        d1 = self.lg.input_proj(desc1)
        enc0 = self.lg.posenc(k0)
        enc1 = self.lg.posenc(k1)
        for i in range(self.n_layers):
            layer = self.lg.transformers[i]
            d0 = self._self_block(layer.self_attn, d0, enc0)
            d1 = self._self_block(layer.self_attn, d1, enc1)
            d0, d1 = self._cross_block(layer.cross_attn, d0, d1)
        scores = self._assignment_scores(d0, d1)
        matches0, mscores0 = self._filter_matches(scores)
        return matches0, mscores0
