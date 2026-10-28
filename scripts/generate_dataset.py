"""Phase 1: generate candidate training images with SDXL base.

No LoRA yet, so the trigger word is left out of prompts here. Each image gets
a sidecar .json with its mode and subject, which caption.py uses later.

Multi-GPU: run one process per GPU with --shard i/n and CUDA_VISIBLE_DEVICES=i.
Each shard takes every n-th prompt, and file names and seeds come from the
prompt's position in the full list, so shards never collide and the result is
the same as a single-GPU run.
"""
import argparse
import json

import _bootstrap  # noqa: F401
from nightprints.config import load_style
from nightprints.models import generate, load_pipeline
from nightprints.tiling import seam_score, set_tiling


def parse_shard(s: str) -> tuple[int, int]:
    i, n = (int(x) for x in s.split("/"))
    if not 0 <= i < n:
        raise argparse.ArgumentTypeError("shard must look like 0/2 or 1/2")
    return i, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", required=True)
    ap.add_argument("--per-prompt", type=int, default=2, help="seeds per prompt")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many prompts (0 = all)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--shard", type=parse_shard, default=(0, 1), help="i/n, e.g. 0/2")
    args = ap.parse_args()

    style = load_style(args.style)
    out = style.p("raw")
    max_seam = style["post"]["max_seam_score"]
    shard_i, shard_n = args.shard

    items = style.dataset_subjects()
    if args.limit:
        items = items[: args.limit]
    mine = [(idx, m, s) for idx, (m, s) in enumerate(items) if idx % shard_n == shard_i]

    pipe = load_pipeline(style)
    tag = f"[shard {shard_i}/{shard_n}]" if shard_n > 1 else ""
    made = skipped = 0
    for done, (idx, mode, subject) in enumerate(mine, start=1):
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
        print(f"{tag}[{done}/{len(mine)}] {mode}: {subject}  (kept {made}, seam-rejected {skipped})", flush=True)

    print(f"{tag} done: {made} images in {out}")


if __name__ == "__main__":
    main()
