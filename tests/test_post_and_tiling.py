import numpy as np
from PIL import Image

from nightprints.config import load_style
from nightprints.post import reduce_palette, upscale, wrap_pad
from nightprints.tiling import preview_repeat, seam_score


def tileable_image(size=128):
    """Sine pattern whose period divides the size, so it wraps perfectly."""
    x = np.linspace(0, 2 * np.pi, size, endpoint=False)
    xx, yy = np.meshgrid(x, x)
    v = (np.sin(3 * xx) * np.cos(2 * yy) + 1) * 127.5
    rgb = np.stack([v, v * 0.6, 255 - v], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb)


def broken_image(size=128):
    """Horizontal gradient: big jump from the right edge back to the left."""
    v = np.tile(np.linspace(0, 255, size), (size, 1))
    return Image.fromarray(np.stack([v] * 3, axis=-1).astype(np.uint8))


def test_seam_score_separates_good_and_bad():
    assert seam_score(tileable_image()) < 1.5
    assert seam_score(broken_image()) > 10


def test_reduce_palette_limits_colors():
    out = reduce_palette(tileable_image(), 8)
    assert len(out.getcolors(maxcolors=10_000)) <= 8


def test_reduce_palette_keeps_alpha():
    img = tileable_image().convert("RGBA")
    img.putalpha(Image.new("L", img.size, 77))
    out = reduce_palette(img, 6)
    assert out.mode == "RGBA" and out.getchannel("A").getextrema() == (77, 77)


def test_wrap_pad_shape():
    assert wrap_pad(tileable_image(64), 8).size == (80, 80)


def test_upscale_keeps_tile_seamless():
    img = tileable_image(96)
    out = upscale(img, 4, tileable=True, use_esrgan=False)
    assert out.size == (384, 384)
    assert seam_score(out) < 1.5


def test_preview_repeat_size():
    assert preview_repeat(tileable_image(50), 3).size == (150, 150)


def test_style_config_builds_prompts():
    s = load_style("configs/styles/dino.yaml")
    items = s.dataset_subjects()
    assert {m for m, _ in items} == {"placement", "pattern"}
    assert len(items) == 6 * 6 + 6 * 4
    assert s.build_prompt("pattern", "dinos").startswith(s.trigger)
