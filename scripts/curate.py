"""Phase 1b: pick keepers from data/raw with a small Gradio UI.

Shows one image at a time. Keep copies it (and its .json) to data/curated,
Skip moves on. Works from a phone through the share link on Kaggle.

What to keep: same drawing style as the others, cute and calm, no text,
no extra limbs, clean background. Aim for 30 to 40 total, about 2/3
placement and 1/3 pattern.
"""
import argparse
import shutil

import _bootstrap  # noqa: F401
import gradio as gr
from nightprints.config import load_style


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="configs/styles/dino.yaml")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    style = load_style(args.style)
    raw, cur = style.p("raw"), style.p("curated")
    files = sorted(raw.glob("*.png"))
    if not files:
        raise SystemExit(f"no images in {raw}, run generate_dataset.py first")

    def view(i):
        if i >= len(files):
            kept = len(list(cur.glob("*.png")))
            return None, f"Finished. {kept} images in {cur}", i
        kept = len(list(cur.glob("*.png")))
        return str(files[i]), f"{i + 1}/{len(files)}  ·  kept {kept}  ·  {files[i].name}", i

    def keep(i):
        if i < len(files):
            f = files[i]
            shutil.copy2(f, cur / f.name)
            side = f.with_suffix(".json")
            if side.exists():
                shutil.copy2(side, cur / side.name)
        return view(i + 1)

    def skip(i):
        return view(i + 1)

    with gr.Blocks(title="nightprints curate") as demo:
        idx = gr.State(0)
        img = gr.Image(type="filepath", height=600, label=None)
        info = gr.Markdown()
        with gr.Row():
            b_skip = gr.Button("Skip")
            b_keep = gr.Button("Keep", variant="primary")
        demo.load(view, idx, [img, info, idx])
        b_keep.click(keep, idx, [img, info, idx])
        b_skip.click(skip, idx, [img, info, idx])

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
