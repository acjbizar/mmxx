from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from ..scene import Scene
from ..anim.color_math import rgb_to_hex


@dataclass
class NoneTheme:
    """No animation: keep polygons at their base colors (honors --color override)."""

    name: str = "none"

    @classmethod
    def create(cls, *, scene: Scene, args: Any, rng: random.Random) -> "NoneTheme":
        return cls()

    def apply_frame(self, scene: Scene, t: float) -> None:
        for i, poly in enumerate(scene.polys):
            poly.set("fill", rgb_to_hex(scene.base_rgbs[i]))
