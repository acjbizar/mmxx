from __future__ import annotations
import math, random
from dataclasses import dataclass
from typing import Any, List, Tuple
from ..scene import Scene
from ..anim.color_math import mix_rgb, rgb_to_hex
from ..anim.easing import cosine_ease
from ..svg.style import style_set

RGB = Tuple[int, int, int]

@dataclass
class DeideeTheme:
    name: str = "deidee"
    alpha: float = 0.5
    colors_per_poly: List[List[RGB]] = None  # type: ignore[assignment]
    seg_dur: List[float] = None  # type: ignore[assignment]
    phase: List[float] = None  # type: ignore[assignment]

    @classmethod
    def create(cls, *, scene: Scene, args: Any, rng: random.Random) -> "DeideeTheme":
        inst = cls()
        inst.colors_per_poly = []
        inst.seg_dur = []
        inst.phase = []
        for _ in scene.polys:
            k = rng.randint(4, 8)
            cols: List[RGB] = []
            for _j in range(k):
                r = int(round(rng.uniform(0.0, 0.5) * 255.0))
                g = int(round(rng.uniform(0.5, 1.0) * 255.0))
                b = int(round(rng.uniform(0.0, 0.75) * 255.0))
                cols.append((max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))))
            inst.colors_per_poly.append(cols)
            inst.seg_dur.append(rng.uniform(0.90, 2.60))
            inst.phase.append(rng.uniform(0.0, 10.0))

        for poly in scene.polys:
            poly.set("fill-opacity", f"{inst.alpha:.3f}")
            st = (poly.get("style") or "").strip()
            poly.set("style", style_set(st, "fill-opacity", f"{inst.alpha:.3f}"))
        return inst

    def apply_frame(self, scene: Scene, t: float) -> None:
        for idx, poly in enumerate(scene.polys):
            cols = self.colors_per_poly[idx]
            k = len(cols)
            seg = self.seg_dur[idx]
            ph = self.phase[idx]
            pos = (t + ph) / seg
            i0 = int(math.floor(pos)) % k
            i1 = (i0 + 1) % k
            f = pos - math.floor(pos)
            u = cosine_ease(f)
            rgb = mix_rgb(cols[i0], cols[i1], u)
            poly.set("fill", rgb_to_hex(rgb))
