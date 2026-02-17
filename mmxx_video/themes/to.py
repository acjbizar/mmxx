from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Tuple

from ..scene import Scene
from ..svg.color import parse_css_color_to_rgb
from ..anim.color_math import mix_rgb, rgb_to_hex
from ..anim.pulses import whiteness_at

RGB = Tuple[int, int, int]


@dataclass
class ToTheme:
    """Pulse polygons toward a target color (defaults to white)."""

    target_rgb: RGB
    name: str = "to"

    @classmethod
    def create(cls, *, scene: Scene, args: Any, rng: random.Random) -> "ToTheme":
        raw = (getattr(args, "to", None) or "white").strip() or "white"
        target = parse_css_color_to_rgb(raw)
        return cls(target_rgb=target)

    def apply_frame(self, scene: Scene, t: float) -> None:
        for i, poly in enumerate(scene.polys):
            a = whiteness_at(t, scene.pulses_per_poly[i])
            rgb = mix_rgb(scene.base_rgbs[i], self.target_rgb, a)
            poly.set("fill", rgb_to_hex(rgb))
