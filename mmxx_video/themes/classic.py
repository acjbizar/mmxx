from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Any
from ..scene import Scene
from ..anim.pulses import whiteness_at
from ..anim.color_math import mix_to_white, rgb_to_hex

@dataclass
class ClassicTheme:
    name: str = "classic"

    @classmethod
    def create(cls, *, scene: Scene, args: Any, rng: random.Random) -> "ClassicTheme":
        return cls()

    def apply_frame(self, scene: Scene, t: float) -> None:
        for idx, poly in enumerate(scene.polys):
            a = whiteness_at(t, scene.pulses_per_poly[idx])
            poly.set("fill", rgb_to_hex(mix_to_white(scene.base_rgbs[idx], a)))
