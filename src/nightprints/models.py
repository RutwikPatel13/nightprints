from __future__ import annotations

from pathlib import Path

import torch

from .config import Style


def pick_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def load_pipeline(style: Style, lora: str | Path | None = None):
    from diffusers import AutoencoderKL, StableDiffusionXLPipeline

    device, dtype = pick_device()
    vae = AutoencoderKL.from_pretrained(style["model"]["vae"], torch_dtype=dtype)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        style["model"]["base"],
        vae=vae,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
    )
    if lora:
        pipe.load_lora_weights(str(lora), adapter_name="style")
    pipe.to(device)
    if device == "cuda":
        pipe.enable_vae_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def set_lora_scale(pipe, scale: float) -> None:
    """0 turns the LoRA off without reloading the pipeline."""
    if getattr(pipe, "get_active_adapters", None) and pipe.get_active_adapters():
        pipe.set_adapters(["style"], adapter_weights=[scale])


def generate(pipe, prompt: str, negative: str, seed: int, steps: int = 30,
             guidance: float = 6.5, size: int = 1024):
    device = pipe.device.type
    gen = torch.Generator(device="cpu" if device == "mps" else device).manual_seed(seed)
    return pipe(
        prompt=prompt,
        negative_prompt=negative,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=size,
        height=size,
        generator=gen,
    ).images[0]
