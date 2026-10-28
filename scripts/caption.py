"""Phase 2a: build the training folder.

Copies curated images into data/train/<style> and writes metadata.jsonl in
the format the diffusers training script reads (file_name, text).

Captions come from the prompt that made each image, so no captioning model
is needed. The trigger word goes first so the LoRA ties the style to it.
Images without a sidecar .json (ones you added yourself) get the style's
fallback_caption from its YAML.
"""
import argparse
import json
import shutil

import _bootstrap  # noqa: F401
from nightprints.config import load_style

DEFAULT_FALLBACK = "kids nightwear illustration"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", required=True)
    args = ap.parse_args()

    style = load_style(args.style)
    cur, train = style.p("curated"), style.p("train")
    images = sorted(p for p in cur.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    if not images:
        raise SystemExit(f"no images in {cur}")

    rows = []
    for p in images:
        side = p.with_suffix(".json")
        if side.exists():
            meta = json.loads(side.read_text())
            kind = "seamless pattern" if meta["mode"] == "pattern" else "character"
            text = f"{style.trigger}, {kind}, {meta['subject']}"
        else:
            text = f"{style.trigger}, {style.raw.get('fallback_caption', DEFAULT_FALLBACK)}"
        shutil.copy2(p, train / p.name)
        rows.append({"file_name": p.name, "text": text})

    with open(train / "metadata.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n_pat = sum("seamless pattern" in r["text"] for r in rows)
    print(f"{len(rows)} images ({len(rows) - n_pat} placement, {n_pat} pattern) -> {train}")
    if len(rows) < 20:
        print("warning: under 20 images, the LoRA will probably overfit")


if __name__ == "__main__":
    main()
