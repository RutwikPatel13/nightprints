"""Gradio app for the friend: type a theme, get print files.

  python app.py --style configs/styles/dino.yaml --lora outputs/lora/dino --share
"""
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

import gradio as gr  # noqa: E402
from generate import make_prints  # noqa: E402
from nightprints.config import load_style  # noqa: E402
from nightprints.models import load_pipeline  # noqa: E402
from nightprints.post import save_print  # noqa: E402
from nightprints.tiling import preview_repeat  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="configs/styles/dino.yaml")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    style = load_style(args.style)
    pipe = load_pipeline(style, lora=args.lora)
    dpi = style["post"]["dpi"]

    def run(theme, mode, n, seed):
        mode = "pattern" if mode.startswith("All-over") else "placement"
        tmp = Path(tempfile.mkdtemp())
        gallery, files = [], []
        for i, (final, raw, _) in enumerate(make_prints(pipe, style, mode, theme, int(n), int(seed))):
            path = save_print(final, tmp / f"{mode}_{i}.png", dpi)
            files.append(str(path))
            shown = preview_repeat(raw, 2) if mode == "pattern" else final
            gallery.append(shown)
        return gallery, files

    with gr.Blocks(title="nightprints") as demo:
        gr.Markdown("### Nightwear print generator")
        with gr.Row():
            theme = gr.Textbox(label="Theme", value="sleepy baby dinosaurs with moons and stars", scale=3)
            mode = gr.Radio(["All-over pattern", "Front print"], value="All-over pattern", label="Type")
        with gr.Row():
            n = gr.Slider(1, 4, value=2, step=1, label="How many")
            seed = gr.Number(value=0, label="Seed (change for new options)", precision=0)
            go = gr.Button("Generate", variant="primary")
        gallery = gr.Gallery(label="Preview (patterns shown repeated 2x2)", columns=2, height=600)
        files = gr.File(label="Print-ready files, 300 DPI", file_count="multiple")
        go.click(run, [theme, mode, n, seed], [gallery, files])

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
