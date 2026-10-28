"""CLIP-based scores for eval.

clip_score: how well an image matches its prompt (higher is better).
style_consistency: mean pairwise similarity between images from different
prompts. If the LoRA learned a style, outputs should look more alike even
when the subject changes.
"""
from __future__ import annotations

import itertools

import torch
from PIL import Image

_model = None
_proc = None
CLIP_ID = "openai/clip-vit-large-patch14"


def _load():
    global _model, _proc
    if _model is None:
        from transformers import CLIPModel, CLIPProcessor

        _model = CLIPModel.from_pretrained(CLIP_ID).eval()
        _proc = CLIPProcessor.from_pretrained(CLIP_ID)
    return _model, _proc


@torch.no_grad()
def image_embeds(images: list[Image.Image]) -> torch.Tensor:
    model, proc = _load()
    inp = proc(images=[i.convert("RGB") for i in images], return_tensors="pt")
    e = model.get_image_features(**inp)
    return e / e.norm(dim=-1, keepdim=True)


@torch.no_grad()
def clip_score(image: Image.Image, prompt: str) -> float:
    model, proc = _load()
    inp = proc(text=[prompt], images=[image.convert("RGB")], return_tensors="pt",
               padding=True, truncation=True)
    img = model.get_image_features(pixel_values=inp["pixel_values"])
    txt = model.get_text_features(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
    img = img / img.norm(dim=-1, keepdim=True)
    txt = txt / txt.norm(dim=-1, keepdim=True)
    return float((img * txt).sum() * 100)


def style_consistency(images: list[Image.Image]) -> float:
    if len(images) < 2:
        return float("nan")
    e = image_embeds(images)
    sims = [float(e[i] @ e[j]) for i, j in itertools.combinations(range(len(images)), 2)]
    return sum(sims) / len(sims)
