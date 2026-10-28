"""Phase 4: make print-ready files from a theme.

Example:
  python scripts/generate.py --style configs/styles/dino.yaml \
      --lora outputs/lora/dino --mode pattern \
      --theme "sleepy baby t-rex with moons and stars" --n 4
"""
import argparse
import json
import re
from datetime import datetime

import _bootstrap  # noqa: F401
from nightprints.config import load_style
from nightprints.models import generate, load_pipeline
from nightprints.post import finish, save_print
from nightprints.tiling import preview_repeat, seam_score, set_tiling


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40]


def make_prints(pipe, style, mode: str, theme: str, n: int, seed: int = 0,
                use_esrgan: bool = True, max_tries: int = 3):
    """Returns a list of (final_print, raw_image, info). Retries tiles with bad seams."""
    set_tiling(pipe, mode == "pattern")
    prompt = style.build_prompt(mode, theme)
    max_seam = style["post"]["max_seam_score"]
    results = []
    s = seed
    for _ in range(n):
        for attempt in range(max_tries):
            raw = generate(pipe, prompt, style.negative, seed=s)
            s += 1
            info = {"mode": mode, "theme": theme, "prompt": prompt, "seed": s - 1}
            if mode != "pattern":
                break
            info["seam_score"] = seam_score(raw)
            if info["seam_score"] <= max_seam or attempt == max_tries - 1:
                break
        final = finish(raw, mode, style["post"], use_esrgan=use_esrgan)
        results.append((final, raw, info))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--mode", choices=["pattern", "placement"], default="pattern")
    ap.add_argument("--theme", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-esrgan", action="store_true")
    args = ap.parse_args()

    style = load_style(args.style)
    pipe = load_pipeline(style, lora=args.lora)
    out = style.p("prints") / f"{datetime.now():%Y%m%d-%H%M}_{args.mode}_{slug(args.theme)}"
    out.mkdir(parents=True, exist_ok=True)

    for i, (final, raw, info) in enumerate(
        make_prints(pipe, style, args.mode, args.theme, args.n, args.seed, not args.no_esrgan)
    ):
        save_print(final, out / f"{i}_print.png", style["post"]["dpi"])
        raw.save(out / f"{i}_raw.png")
        if args.mode == "pattern":
            preview_repeat(raw, 3).resize((1536, 1536)).save(out / f"{i}_repeat_preview.png")
        (out / f"{i}.json").write_text(json.dumps(info, indent=2))
        print(f"saved {out / f'{i}_print.png'}  {info.get('seam_score', '')}")


if __name__ == "__main__":
    main()
