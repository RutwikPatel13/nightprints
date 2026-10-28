from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Style:
    raw: dict
    path: Path

    def __getitem__(self, key):
        return self.raw[key]

    @property
    def trigger(self) -> str:
        return self.raw["trigger"]

    def p(self, key: str) -> Path:
        """Path from the paths: block, created if missing."""
        out = Path(self.raw["paths"][key])
        out.mkdir(parents=True, exist_ok=True)
        return out

    def build_prompt(self, mode: str, subject: str, with_trigger: bool = True) -> str:
        parts = []
        if with_trigger:
            parts.append(self.trigger)
        parts.append(subject)
        parts.append(self.raw["style"][mode])
        return ", ".join(parts)

    @property
    def negative(self) -> str:
        return self.raw["style"]["negative"]

    def dataset_subjects(self, seed: int = 0) -> list[tuple[str, str]]:
        """(mode, subject phrase) pairs for dataset generation.

        Placement: subject x action. Pattern: subject x background.
        Shuffled so a partial run still covers variety.
        """
        r = self.raw
        placement = [("placement", f"{s} {a}") for s, a in itertools.product(r["subjects"], r["actions"])]
        pattern = [("pattern", f"{s} motifs, {b}") for s, b in itertools.product(r["subjects"], r["backgrounds"])]
        items = placement + pattern
        random.Random(seed).shuffle(items)
        return items


def load_style(path: str | Path) -> Style:
    path = Path(path)
    with open(path) as f:
        return Style(yaml.safe_load(f), path)
