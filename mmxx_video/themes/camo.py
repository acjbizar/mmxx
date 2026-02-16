from __future__ import annotations
import math, random
from dataclasses import dataclass
from typing import Any, List, Tuple
from ..scene import Scene
from ..anim.easing import clamp01, smoothstep
from ..anim.noise import noise2, fbm2
from ..anim.color_math import mix_rgb, mix_to_white, rgb_to_hex
from ..svg.color import parse_css_color_to_rgb
from .configs import get_theme_config

RGB = Tuple[int, int, int]

@dataclass
class CamoTheme:
    name: str = "camo"
    camo_seed: float = 0.0
    offx: List[float] = None  # type: ignore[assignment]
    offy: List[float] = None  # type: ignore[assignment]
    phase: List[float] = None  # type: ignore[assignment]
    palette: List[RGB] = None  # type: ignore[assignment]

    @classmethod
    def create(cls, *, scene: Scene, args: Any, rng: random.Random) -> "CamoTheme":
        inst = cls()
        inst.camo_seed = float(args.seed if args.seed is not None else rng.randint(0, 10_000_000))
        inst.offx = []
        inst.offy = []
        inst.phase = []
        for _ in scene.polys:
            inst.offx.append(rng.uniform(0.0, 1000.0))
            inst.offy.append(rng.uniform(0.0, 1000.0))
            inst.phase.append(rng.uniform(0.0, 1.0))

        inst.palette = [
            parse_css_color_to_rgb("#1b2116"),
            parse_css_color_to_rgb("#2f3b2a"),
            parse_css_color_to_rgb("#4b5636"),
            parse_css_color_to_rgb("#6b6a45"),
            parse_css_color_to_rgb("#8a7751"),
            parse_css_color_to_rgb("#b2a378"),
            parse_css_color_to_rgb("#3c2f20"),
        ]
        return inst

    def apply_frame(self, scene: Scene, t: float) -> None:
        cfg = get_theme_config("camo")
        amb = cfg.amb_base + cfg.amb_amp * (0.5 + 0.5 * math.sin(2.0 * math.pi * (cfg.amb_freq * t)))

        th = [0.12, 0.24, 0.40, 0.56, 0.72, 0.86]
        bw = 0.045
        pal = self.palette

        def pick(nv: float) -> RGB:
            if nv <= th[0]:
                return pal[0]
            if nv >= th[-1]:
                return pal[-1]
            k = 0
            while k < len(th) and nv > th[k]:
                k += 1
            a = max(0, min(len(pal) - 2, k))
            b = a + 1
            edge0 = th[a] if a < len(th) else th[-1]
            edge1 = th[b] if b < len(th) else th[-1]
            t0 = smoothstep(edge0 - bw, edge0 + bw, nv)
            t1 = smoothstep(edge1 - bw, edge1 + bw, nv)
            tt = clamp01((t0 + t1) * 0.5)
            return mix_rgb(pal[a], pal[b], tt)

        for idx, poly in enumerate(scene.polys):
            nx = scene.poly_nx[idx]
            ny = scene.poly_ny[idx]

            drift_x = 0.020 * math.sin(2.0 * math.pi * (0.035 * t + self.phase[idx]))
            drift_y = 0.018 * math.cos(2.0 * math.pi * (0.030 * t + self.phase[idx] * 0.7))

            x = (nx * 3.2 + drift_x) + self.offx[idx] * 0.001
            y = (ny * 3.2 + drift_y) + self.offy[idx] * 0.001

            macro = fbm2(x * 0.85, y * 0.85, self.camo_seed + 1.7, octaves=4)
            mid   = fbm2(x * 2.10, y * 2.10, self.camo_seed + 7.9, octaves=3)
            micro = fbm2(x * 9.00, y * 9.00, self.camo_seed + 13.3, octaves=2)

            n = clamp01(0.62 * macro + 0.28 * mid + 0.10 * micro)

            shade = 0.90 + 0.10 * math.sin(2.0 * math.pi * (0.045 * t + nx * 2.1 + ny * 1.6))
            shade *= 0.94 + 0.06 * (0.5 + 0.5 * math.sin(2.0 * math.pi * (0.11 * t + self.phase[idx])))
            shade = clamp01(shade)

            rgb = pick(n)

            g = noise2(nx * 120.0 + t * 0.35, ny * 120.0 + t * 0.27, self.camo_seed + 99.1)
            grain = (g - 0.5) * 0.10
            rgb = (
                max(0, min(255, int(round(rgb[0] * (shade + grain))))),
                max(0, min(255, int(round(rgb[1] * (shade + grain))))),
                max(0, min(255, int(round(rgb[2] * (shade + grain))))),
            )

            rgb = mix_to_white(rgb, amb * 0.04)
            poly.set("fill", rgb_to_hex(rgb))
