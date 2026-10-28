"""Pick the best LoRA checkpoint by looking at them side by side.

Training saves a checkpoint every train.checkpoint_every steps. Later is not
always better: past a point the LoRA starts copying training images instead of
applying the style to new prompts. This renders the same few eval prompts with
every checkpoint (same seeds) into one grid, one row per checkpoint.

  python scripts/compare_checkpoints.py --style configs/styles/dino.yaml

About 20 seconds per image on a T4, so 5 checkpoints x 4 prompts is ~7 minutes.
Copy the winner's pytorch_lora_weights.safetensors up into outputs/lora/<style>/
if it isn't the final one.
"""
import argparse
import re
from pathlib import Path

import _bootstrap  # noqa: F401
from PIL import Image, ImageDraw
from nightprints.config import load_style
from nightprints.models import generate, load_pipeline
from nightprints.tiling import set_tiling

WEIGHTS = "pytorch_lora_weights.safetensors"


def find_checkpoints(lora_dir: Path) -> list[tuple[str, Path]]:
    found = []
    for d in lora_dir.glob("checkpoint-*"):
        m = re.fullmatch(r"checkpoint-(\d+)", d.name)
        if m and (d / WEIGHTS).exists():
            found.append((int(m.group(1)), d))
    rows = [(f"step {n}", d) for n, d in sorted(found)]
    if (lora_dir / WEIGHTS).exists():
        rows.append(("final", lora_dir))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", required=True)
    ap.add_argument("--lora-dir", default=None, help="defaults to paths.lora in the style YAML")
    ap.add_argument("--per-mode", type=int, default=2, help="eval prompts per mode")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    style = load_style(args.style)
    lora_dir = Path(args.lora_dir) if args.lora_dir else style.p("lora")
    rows = find_checkpoints(lora_dir)
    if not rows:
        raise SystemExit(f"no checkpoints with {WEIGHTS} in {lora_dir}")

    prompts = [(m, p) for m in ("placement", "pattern") for p in style["eval_prompts"][m][: args.per_mode]]
    pipe = load_pipeline(style)
    size, label_w = 320, 110
    grid = Image.new("RGB", (label_w + size * len(prompts), size * len(rows)), "white")
    draw = ImageDraw.Draw(grid)

    for r, (label, path) in enumerate(rows):
        pipe.load_lora_weights(str(path), adapter_name="style")
        draw.text((10, r * size + size // 2), label, fill="black")
        for c, (mode, p) in enumerate(prompts):
            set_tiling(pipe, mode == "pattern")
            img = generate(pipe, style.build_prompt(mode, p), style.negative, seed=args.seed)
            grid.paste(img.resize((size, size)), (label_w + c * size, r * size))
        pipe.unload_lora_weights()
        print(f"rendered {label}", flush=True)

    out = style.p("eval") / "compare_checkpoints.png"
    grid.save(out)
    print(f"saved {out}  (rows: {', '.join(l for l, _ in rows)})")


if __name__ == "__main__":
    main()
