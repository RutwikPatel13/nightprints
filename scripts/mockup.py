"""Preview a print on a kids nightdress and pajama set.

  python scripts/mockup.py outputs/prints/dino/<run>/0_print.png --mode pattern
  python scripts/mockup.py data/raw/dino/placement_000_0.png --mode placement
  python scripts/mockup.py tile.png --mode pattern --repeat-in 6   # force repeat size

Pattern repeat size comes from the file's DPI (print files are saved at 300 DPI),
or --repeat-in. Writes <input>_mockup.png next to the input unless --out is given.
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from PIL import Image
from nightprints.mockup import GARMENTS, pattern_mockup, placement_mockup, side_by_side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--mode", choices=["pattern", "placement"], required=True)
    ap.add_argument("--garment", choices=[*GARMENTS, "both"], default="both")
    ap.add_argument("--repeat-in", type=float, default=None, help="pattern repeat size in inches")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    img = Image.open(args.image)
    names = list(GARMENTS) if args.garment == "both" else [args.garment]
    if args.mode == "pattern":
        shots = [pattern_mockup(img, g, args.repeat_in) for g in names]
    else:
        shots = [placement_mockup(img, g) for g in names]
    out = Path(args.out) if args.out else Path(args.image).with_name(Path(args.image).stem + "_mockup.png")
    side_by_side(shots).save(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
