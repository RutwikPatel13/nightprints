import numpy as np
from PIL import Image

from nightprints.mockup import (BACKGROUND, CANVAS, garment_mask, light_color, mockup,
                                pattern_mockup, placement_mockup, tile_inches, NIGHTDRESS)


def stripe_tile(size=256):
    a = np.zeros((size, size, 3), np.uint8)
    a[:, : size // 2] = (200, 40, 60)
    a[:, size // 2:] = (250, 240, 220)
    return Image.fromarray(a)


def test_mask_covers_garment_not_corners():
    m = np.asarray(garment_mask(NIGHTDRESS))
    assert m[600, 500] == 255 and m[5, 5] == 0


def test_pattern_fills_garment_and_leaves_background():
    out = np.asarray(pattern_mockup(stripe_tile(), "nightdress", repeat_in=6))
    assert out.shape[:2] == (CANVAS[1], CANVAS[0])
    assert tuple(out[5, 5]) == BACKGROUND
    inside = out[500:700, 350:650].reshape(-1, 3)
    assert (inside[:, 0] > inside[:, 2] + 60).any()  # red stripes visible on the dress


def test_repeat_size_from_dpi():
    t = stripe_tile(600)
    t.info["dpi"] = (300, 300)
    assert tile_inches(t) == 2.0
    assert tile_inches(stripe_tile()) == 8.0


def test_placement_and_combined():
    art = Image.new("RGBA", (300, 300), (30, 120, 90, 255))
    assert placement_mockup(art, "pajamas").size == CANVAS
    assert mockup(art, "placement").width > CANVAS[0] * 2
    assert all(v > 180 for v in light_color(art))
