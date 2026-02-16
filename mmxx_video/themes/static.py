from __future__ import annotations
import math, random
from dataclasses import dataclass
from typing import Any, List
from ..scene import Scene
from ..anim.easing import clamp01, cosine_ease
from ..anim.noise import hash01
from ..anim.color_math import hsv01_to_rgb255, rgb_to_hex

@dataclass
class StaticTheme:
    name: str = "static"
    seg: List[float] = None  # type: ignore[assignment]
    phase: List[float] = None  # type: ignore[assignment]
    color_prob: List[float] = None  # type: ignore[assignment]
    seedf: float = 0.0

    @classmethod
    def create(cls, *, scene: Scene, args: Any, rng: random.Random) -> "StaticTheme":
        inst = cls()
        inst.seedf = float(args.seed if args.seed is not None else rng.randint(0, 10_000_000))
        inst.seg = []
        inst.phase = []
        inst.color_prob = []
        for _ in scene.polys:
            inst.seg.append(rng.uniform(0.05, 0.13))
            inst.phase.append(rng.uniform(0.0, 10.0))
            inst.color_prob.append(rng.uniform(0.06, 0.14))
        return inst

    def apply_frame(self, scene: Scene, t: float) -> None:
        for idx, poly in enumerate(scene.polys):
            seg = self.seg[idx]
            ph = self.phase[idx]
            pos = (t + ph) / seg
            k0 = int(math.floor(pos))
            f = pos - math.floor(pos)
            u = cosine_ease(f)

            a0 = hash01(self.seedf * 0.001 + idx * 12.9898 + k0 * 78.233)
            a1 = hash01(self.seedf * 0.001 + idx * 12.9898 + (k0 + 1) * 78.233)
            a = (1.0 - u) * a0 + u * a1

            a = clamp01((a - 0.5) * 1.85 + 0.5)
            vv = 0.06 + 0.94 * a
            scan = 0.95 + 0.05 * math.sin(2.0 * math.pi * (scene.poly_ny[idx] * 90.0 + t * 1.25))
            vv = clamp01(vv * scan)

            csel0 = hash01(self.seedf * 0.002 + idx * 3.11 + k0 * 9.73)
            csel1 = hash01(self.seedf * 0.002 + idx * 3.11 + (k0 + 1) * 9.73)
            csel = (1.0 - u) * csel0 + u * csel1

            if csel < self.color_prob[idx]:
                h0 = hash01(self.seedf * 0.003 + idx * 0.77 + k0 * 2.17)
                h1 = hash01(self.seedf * 0.003 + idx * 0.77 + (k0 + 1) * 2.17)
                h = ((1.0 - u) * h0 + u * h1) % 1.0
                s = 0.55 + 0.45 * hash01(self.seedf * 0.004 + idx * 1.33 + k0 * 6.19)
                rgb = hsv01_to_rgb255(h, s, 0.25 + 0.75 * vv)
            else:
                g = int(round(vv * 255.0))
                rgb = (g, g, g)

            poly.set("fill", rgb_to_hex(rgb))
