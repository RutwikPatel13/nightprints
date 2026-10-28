"""Turn a raw generation into a file a fabric printer can use.

All CPU. Nothing here needs a GPU.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

UPSCALER_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
)


def remove_background(img: Image.Image) -> Image.Image:
    from rembg import remove

    return remove(img.convert("RGB"))  # RGBA


def reduce_palette(img: Image.Image, colors: int) -> Image.Image:
    """Snap to a small fixed palette. Keeps alpha if present."""
    alpha = img.getchannel("A") if img.mode == "RGBA" else None
    q = img.convert("RGB").quantize(colors=colors, method=Image.Quantize.MEDIANCUT,
                                    dither=Image.Dither.NONE).convert("RGB")
    if alpha is not None:
        q.putalpha(alpha)
    return q


_upscaler = None


def _load_upscaler(weights_dir: Path):
    global _upscaler
    if _upscaler is None:
        import torch
        from spandrel import ModelLoader

        weights_dir.mkdir(parents=True, exist_ok=True)
        path = weights_dir / "RealESRGAN_x4plus.pth"
        if not path.exists():
            torch.hub.download_url_to_file(UPSCALER_URL, str(path))
        _upscaler = ModelLoader().load_from_file(str(path)).eval()
    return _upscaler


def _esrgan(img: Image.Image, weights_dir: Path) -> Image.Image:
    import torch

    model = _load_upscaler(weights_dir)
    x = torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0)
    x = x.permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        y = model(x).clamp(0, 1)
    arr = (y[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def upscale(img: Image.Image, factor: int, tileable: bool = False,
            use_esrgan: bool = True, weights_dir: Path = Path("weights")) -> Image.Image:
    """Upscale by factor. For tiles, wrap-pad first so the edges stay seamless.

    Falls back to Lanczos if Real-ESRGAN can't load (offline, no torch).
    """
    pad = 32 if tileable else 0
    src = wrap_pad(img, pad) if pad else img

    alpha = src.getchannel("A") if src.mode == "RGBA" else None
    try:
        if not use_esrgan:
            raise RuntimeError("esrgan disabled")
        out = _esrgan(src, weights_dir)
        if out.size != (src.width * 4, src.height * 4):
            raise RuntimeError("unexpected upscaler output")
        if factor != 4:
            out = out.resize((src.width * factor, src.height * factor), Image.Resampling.LANCZOS)
    except Exception:
        out = src.convert("RGB").resize((src.width * factor, src.height * factor),
                                        Image.Resampling.LANCZOS)
    if alpha is not None:
        out.putalpha(alpha.resize(out.size, Image.Resampling.LANCZOS))

    if pad:
        p = pad * factor
        out = out.crop((p, p, out.width - p, out.height - p))
    return out


def wrap_pad(img: Image.Image, pad: int) -> Image.Image:
    arr = np.asarray(img)
    widths = ((pad, pad), (pad, pad)) + (((0, 0),) if arr.ndim == 3 else ())
    return Image.fromarray(np.pad(arr, widths, mode="wrap"))


def save_print(img: Image.Image, path: Path, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, dpi=(dpi, dpi))
    return path


def finish(img: Image.Image, mode: str, post_cfg: dict, use_esrgan: bool = True) -> Image.Image:
    """The full post-processing chain for one image."""
    if mode == "placement":
        img = remove_background(img)
    img = reduce_palette(img, post_cfg["palette_colors"])
    img = upscale(img, post_cfg["upscale"], tileable=(mode == "pattern"), use_esrgan=use_esrgan)
    # Upscaling adds in-between colors back; snap again so the printer gets a clean palette.
    return reduce_palette(img, post_cfg["palette_colors"])
