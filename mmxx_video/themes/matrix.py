from __future__ import annotations
import math, random
from dataclasses import dataclass
from typing import Any, List
from ..scene import Scene
from ..anim.easing import clamp01, smoothstep
from ..anim.color_math import hsv01_to_rgb255, mix_rgb, rgb_to_hex

@dataclass(frozen=True)
class MatrixDrop:
    speed: float
    phase: float
    tail: float
    head: float
    strength: float
    flicker_freq: float
    flicker_phase: float

@dataclass
class MatrixTheme:
    name: str = "matrix"
    col_count: int = 0
    drops: List[MatrixDrop] = None  # type: ignore[assignment]

    @classmethod
    def create(cls, *, scene: Scene, args: Any, rng: random.Random) -> "MatrixTheme":
        inst = cls()
        inst.col_count = max(14, min(40, int(round(math.sqrt(len(scene.polys)) * 4))))
        inst.drops = []
        for _c in range(inst.col_count):
            inst.drops.append(MatrixDrop(
                speed=rng.uniform(0.10, 0.32),
                phase=rng.uniform(0.0, 1.0),
                tail=rng.uniform(0.18, 0.55),
                head=rng.uniform(0.02, 0.06),
                strength=rng.uniform(0.65, 1.00),
                flicker_freq=rng.uniform(3.0, 9.0),
                flicker_phase=rng.uniform(0.0, 1.0),
            ))
        return inst

    def apply_frame(self, scene: Scene, t: float) -> None:
        for idx, poly in enumerate(scene.polys):
            nx = scene.poly_nx[idx]
            ny = scene.poly_ny[idx]
            c = int(clamp01(nx) * (self.col_count - 1))
            d = self.drops[c]

            head = (d.phase + d.speed * t) % 1.0
            dist = (ny - head) % 1.0

            if dist <= d.head:
                inten = 1.0
            elif dist <= d.tail:
                z = 1.0 - (dist - d.head) / max(1e-6, (d.tail - d.head))
                inten = (z * z)
            else:
                inten = 0.0

            fl = 0.72 + 0.28 * math.sin(2.0 * math.pi * (d.flicker_freq * t + d.flicker_phase))
            inten = clamp01(inten * d.strength * fl)

            bg_v = 0.02 + 0.04 * (0.5 + 0.5 * math.sin(2.0 * math.pi * (0.18 * t + nx * 1.7 + ny * 0.9)))
            bg = hsv01_to_rgb255(120/360.0, 0.55, bg_v)

            if inten <= 0.0:
                rgb = bg
            else:
                headness = smoothstep(0.75, 1.0, inten)
                g_v = 0.10 + 0.90 * inten
                green = hsv01_to_rgb255(120/360.0, 1.0, g_v)
                rgb = mix_rgb(bg, green, 0.85)
                rgb = mix_rgb(rgb, (255, 255, 255), 0.40 * headness)

            poly.set("fill", rgb_to_hex(rgb))
