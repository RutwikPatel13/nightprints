"""Seamless tiling for SDXL and a metric to check it worked.

Setting every Conv2d to circular padding makes the UNet and VAE treat the
image as wrapping around at its edges, so the left edge continues into the
right edge and the top into the bottom.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def set_tiling(pipe, enabled: bool) -> None:
    import torch

    mode = "circular" if enabled else "zeros"
    for module in (pipe.unet, pipe.vae):
        for layer in module.modules():
            if isinstance(layer, torch.nn.Conv2d):
                layer.padding_mode = mode


def _gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float32)


def seam_score(img: Image.Image) -> float:
    """How much rougher the wrap seam is than the inside of the image.

    For each axis, takes the average pixel jump across the wrap boundary
    (last column to first, last row to first) and divides it by the 90th
    percentile of the same jump measured between neighbouring columns or
    rows inside the image. Using a high percentile instead of the mean keeps
    busy patterns from being flagged just because they have sharp edges.

    Around 1 or below means the seam blends in. Much higher means a visible
    line when the tile repeats.
    """
    g = _gray(img)
    eps = 1e-6
    col_jumps = np.abs(np.diff(g, axis=1)).mean(axis=0)
    row_jumps = np.abs(np.diff(g, axis=0)).mean(axis=1)
    seam_x = np.abs(g[:, -1] - g[:, 0]).mean()
    seam_y = np.abs(g[-1, :] - g[0, :]).mean()
    sx = seam_x / (np.percentile(col_jumps, 90) + eps)
    sy = seam_y / (np.percentile(row_jumps, 90) + eps)
    return float(max(sx, sy))


def preview_repeat(img: Image.Image, n: int = 2) -> Image.Image:
    """n x n repeat, handy for eyeballing seams."""
    w, h = img.size
    out = Image.new(img.mode, (w * n, h * n))
    for i in range(n):
        for j in range(n):
            out.paste(img, (i * w, j * h))
    return out
