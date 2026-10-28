"""Product mockups: show a print on a kids nightdress or pajama set.

Everything is drawn with Pillow, no image assets or GPU needed. Garments are
flat-lay silhouettes sized for a roughly age 6 child, so the pattern repeat
shows at a realistic scale: a 13 inch tile on a 28 inch nightdress looks the
way it will look when printed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

SS = 2  # supersampling factor for smooth edges
CANVAS = (1000, 1200)
BACKGROUND = (244, 242, 238)
OUTLINE = (90, 88, 84)


@dataclass
class Garment:
    name: str
    height_in: float  # real-world height of the drawn garment, for repeat scale
    polygons: list[list[tuple[int, int]]]
    cutouts: list[tuple[int, int, int, int]]  # ellipses removed (necklines)
    chest: tuple[int, int]  # where a placement print is centered
    chest_width: int  # placement print width in canvas px


NIGHTDRESS = Garment(
    name="nightdress",
    height_in=28,
    polygons=[[
        (395, 150), (605, 150), (700, 185), (820, 330), (745, 385), (665, 330),
        (700, 620), (765, 960), (800, 1110), (650, 1128), (500, 1134), (350, 1128),
        (200, 1110), (235, 960), (300, 620), (335, 330), (255, 385), (180, 330), (300, 185),
    ]],
    cutouts=[(400, 75, 600, 250)],
    chest=(500, 420),
    chest_width=250,
)

PAJAMAS = Garment(
    name="pajamas",
    height_in=40,
    polygons=[
        # top: boxy shirt with long sleeves
        [(405, 110), (595, 110), (690, 140), (850, 470), (785, 505), (670, 300),
         (680, 590), (320, 590), (330, 300), (215, 505), (150, 470), (310, 140)],
        # pants: waistband to two legs
        [(335, 620), (665, 620), (690, 1150), (530, 1150), (500, 760), (470, 1150), (310, 1150)],
    ],
    cutouts=[(410, 50, 590, 200)],
    chest=(500, 330),
    chest_width=230,
)

GARMENTS = {"nightdress": NIGHTDRESS, "pajamas": PAJAMAS}


def _scaled(points, s=SS):
    return [(x * s, y * s) for x, y in points]


def garment_mask(g: Garment) -> Image.Image:
    """Antialiased L-mode mask of the garment at canvas size."""
    w, h = CANVAS
    m = Image.new("L", (w * SS, h * SS), 0)
    d = ImageDraw.Draw(m)
    for poly in g.polygons:
        d.polygon(_scaled(poly), fill=255)
    for x0, y0, x1, y1 in g.cutouts:
        d.ellipse((x0 * SS, y0 * SS, x1 * SS, y1 * SS), fill=0)
    return m.resize(CANVAS, Image.Resampling.LANCZOS)


def _shading(mask: Image.Image) -> Image.Image:
    """Soft fabric shading: darker toward the garment edges, like a flat-lay photo."""
    blurred = mask.filter(ImageFilter.GaussianBlur(28))
    a = np.asarray(blurred, dtype=np.float32) / 255.0
    shade = 0.80 + 0.20 * np.clip(a * 1.25, 0, 1)
    rgb = (np.stack([shade] * 3, axis=-1) * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def _outline(mask: Image.Image) -> Image.Image:
    edge = mask.point(lambda v: 255 if v > 127 else 0).filter(ImageFilter.FIND_EDGES)
    return edge.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(0.8))


def tile_fill(tile: Image.Image, tile_px: int) -> Image.Image:
    """Repeat a tile across the whole canvas at tile_px per repeat."""
    t = tile.convert("RGB").resize((tile_px, tile_px), Image.Resampling.LANCZOS)
    w, h = CANVAS
    out = Image.new("RGB", CANVAS)
    for y in range(0, h, tile_px):
        for x in range(0, w, tile_px):
            out.paste(t, (x, y))
    return out


def tile_inches(tile: Image.Image, default: float = 8.0) -> float:
    """Real-world repeat size from the file's DPI, if it has one."""
    dpi = tile.info.get("dpi")
    if dpi and dpi[0] and dpi[0] > 1:
        return tile.width / float(dpi[0])
    return default


def light_color(img: Image.Image) -> tuple[int, int, int]:
    """A pale fabric color that suits the print: the print's mean color, lifted toward white."""
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    if img.mode == "RGBA":
        a = np.asarray(img.getchannel("A")) > 127
        rgb = rgb[a] if a.any() else rgb.reshape(-1, 3)
    else:
        rgb = rgb.reshape(-1, 3)
    mean = rgb.mean(axis=0)
    return tuple(int(v) for v in (mean * 0.25 + 255 * 0.75))


def _compose(fabric: Image.Image, mask: Image.Image) -> Image.Image:
    shaded = ImageChops.multiply(fabric, _shading(mask))
    out = Image.new("RGB", CANVAS, BACKGROUND)
    out.paste(shaded, (0, 0), mask)
    out.paste(Image.new("RGB", CANVAS, OUTLINE), (0, 0), _outline(mask))
    return out


def pattern_mockup(tile: Image.Image, garment: str = "nightdress",
                   repeat_in: float | None = None) -> Image.Image:
    g = GARMENTS[garment]
    repeat_in = repeat_in or tile_inches(tile)
    ys = [p[1] for poly in g.polygons for p in poly]
    px_per_in = (max(ys) - min(ys)) / g.height_in
    tile_px = max(24, int(round(repeat_in * px_per_in)))
    return _compose(tile_fill(tile, tile_px), garment_mask(g))


def placement_mockup(print_img: Image.Image, garment: str = "nightdress",
                     fabric: tuple[int, int, int] | None = None) -> Image.Image:
    g = GARMENTS[garment]
    mask = garment_mask(g)
    base = Image.new("RGB", CANVAS, fabric or light_color(print_img))
    art = print_img.convert("RGBA")
    scale = g.chest_width / art.width
    art = art.resize((g.chest_width, int(art.height * scale)), Image.Resampling.LANCZOS)
    cx, cy = g.chest
    base.paste(art, (cx - art.width // 2, cy - art.height // 2), art)
    return _compose(base, mask)


def side_by_side(images: list[Image.Image], gap: int = 40) -> Image.Image:
    w = sum(i.width for i in images) + gap * (len(images) - 1)
    h = max(i.height for i in images)
    out = Image.new("RGB", (w, h), BACKGROUND)
    x = 0
    for i in images:
        out.paste(i, (x, 0))
        x += i.width + gap
    return out


def mockup(img: Image.Image, mode: str, garments=("nightdress", "pajamas")) -> Image.Image:
    """Both garments side by side, the way the app shows them."""
    fn = pattern_mockup if mode == "pattern" else placement_mockup
    return side_by_side([fn(img, g) for g in garments])
