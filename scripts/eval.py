"""Phase 3: does the LoRA actually help?

Runs the eval prompts from the style YAML with the LoRA at scale 0 and 1,
same seeds, and records:
  clip_score         prompt match (want it about the same or higher)
  style_consistency  how alike outputs look across different prompts (want higher)
  seam_score         pattern mode only (want close to 1)
Writes report.md, results.json and side-by-side grids to outputs/eval/<style>.
"""
import argparse
import json
from statistics import mean

import _bootstrap  # noqa: F401
from PIL import Image, ImageDraw
from nightprints.config import load_style
from nightprints.metrics import clip_score, style_consistency
from nightprints.models import generate, load_pipeline, set_lora_scale
from nightprints.tiling import seam_score, set_tiling


def grid(pairs, label_a="base", label_b="lora", size=384):
    w = size * 2
    out = Image.new("RGB", (w, size * len(pairs) + 24), "white")
    d = ImageDraw.Draw(out)
    d.text((size // 2 - 15, 4), label_a, fill="black")
    d.text((size + size // 2 - 15, 4), label_b, fill="black")
    for r, (a, b) in enumerate(pairs):
        out.paste(a.resize((size, size)), (0, 24 + r * size))
        out.paste(b.resize((size, size)), (size, 24 + r * size))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()

    style = load_style(args.style)
    out = style.p("eval")
    pipe = load_pipeline(style, lora=args.lora)

    results = {}
    for mode, prompts in style["eval_prompts"].items():
        set_tiling(pipe, mode == "pattern")
        imgs = {0.0: [], 1.0: []}
        rows = {0.0: [], 1.0: []}
        pairs = []
        for p in prompts:
            for s in range(args.seeds):
                pair = []
                for scale in (0.0, 1.0):
                    set_lora_scale(pipe, scale)
                    # Base model doesn't know the trigger, so leave it out at scale 0.
                    prompt = style.build_prompt(mode, p, with_trigger=scale > 0)
                    img = generate(pipe, prompt, style.negative, seed=s)
                    imgs[scale].append(img)
                    row = {"prompt": p, "seed": s, "clip": clip_score(img, p)}
                    if mode == "pattern":
                        row["seam"] = seam_score(img)
                    rows[scale].append(row)
                    pair.append(img)
                if s == 0:
                    pairs.append(tuple(pair))
        grid(pairs).save(out / f"grid_{mode}.png")

        summary = {}
        for scale, name in ((0.0, "base"), (1.0, "lora")):
            summary[name] = {
                "clip_score": mean(r["clip"] for r in rows[scale]),
                "style_consistency": style_consistency(imgs[scale]),
            }
            if mode == "pattern":
                summary[name]["seam_score"] = mean(r["seam"] for r in rows[scale])
        results[mode] = {"summary": summary, "rows": {k: v for k, v in zip(("base", "lora"), rows.values())}}
        print(mode, json.dumps(summary, indent=2))

    (out / "results.json").write_text(json.dumps(results, indent=2))

    lines = ["# Eval", "", f"{args.seeds} seeds per prompt, same seeds for base and LoRA.", ""]
    for mode, r in results.items():
        lines += [f"## {mode}", "", "| metric | base | lora |", "|---|---|---|"]
        for k in r["summary"]["base"]:
            lines.append(f"| {k} | {r['summary']['base'][k]:.3f} | {r['summary']['lora'][k]:.3f} |")
        lines += ["", f"![{mode}](grid_{mode}.png)", ""]
    (out / "report.md").write_text("\n".join(lines))
    print(f"report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
