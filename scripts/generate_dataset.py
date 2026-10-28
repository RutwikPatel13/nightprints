"""Phase 1: generate candidate training images with SDXL base.

No LoRA yet, so the trigger word is left out of prompts here. Each image gets
a sidecar .json with its mode and subject, which caption.py uses later.
"""
import argparse
import json

import _bootstrap  # noqa: F401
from nightprints.config import load_style
from nightprints.models import generate, load_pipeline
from nightprints.tiling import seam_score, set_tiling


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", required=True)
    ap.add_argument("--per-prompt", type=int, default=2, help="seeds per prompt")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many prompts (0 = all)")
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()

    style = load_style(args.style)
    out = style.p("raw")
    pipe = load_pipeline(style)
    max_seam = style["post"]["max_seam_score"]

    items = style.dataset_subjects()
    if args.limit:
        items = items[: args.limit]

    made = skipped = 0
    for idx, (mode, subject) in enumerate(items):
        set_tiling(pipe, mode == "pattern")
        prompt = style.build_prompt(mode, subject, with_trigger=False)
        for k in range(args.per_prompt):
            seed = 1000 * idx + k
            name = f"{mode}_{idx:03d}_{k}"
            if (out / f"{name}.png").exists():
                continue
            img = generate(pipe, prompt, style.negative, seed=seed, steps=args.steps)
            meta = {"mode": mode, "subject": subject, "prompt": prompt, "seed": seed}
            if mode == "pattern":
                meta["seam_score"] = seam_score(img)
                if meta["seam_score"] > max_seam:
                    skipped += 1
                    continue
            img.save(out / f"{name}.png")
            (out / f"{name}.json").write_text(json.dumps(meta, indent=2))
            made += 1
        print(f"[{idx + 1}/{len(items)}] {mode}: {subject}  (kept {made}, seam-rejected {skipped})")

    print(f"done: {made} images in {out}")


if __name__ == "__main__":
    main()
