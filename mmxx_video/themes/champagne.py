from __future__ import annotations
import math, random
from dataclasses import dataclass
from typing import Any, List
from ..scene import Scene
from ..anim.easing import clamp01, smoothstep
from ..anim.pulses import facet_shimmer
from ..anim.color_math import hsv01_to_rgb255, mix_rgb, mix_to_white, rgb_to_hex
from .configs import get_theme_config

@dataclass(frozen=True)
class Bubble:
    x: float
    y0: float
    r: float
    speed: float
    wob_amp: float
    wob_freq: float
    wob_phase: float
    strength: float

def bubble_influence(nx: float, ny: float, bx: float, by: float, r: float) -> float:
    dx = nx - bx
    dy = ny - by
    d2 = dx * dx + dy * dy
    rr = max(1e-6, r * r)
    return math.exp(-d2 / (2.2 * rr))

@dataclass
class ChampagneTheme:
    name: str = "champagne"
    bubbles: List[Bubble] = None  # type: ignore[assignment]
    ch_freq: List[float] = None  # type: ignore[assignment]
    ch_phase: List[float] = None  # type: ignore[assignment]

    @classmethod
    def create(cls, *, scene: Scene, args: Any, rng: random.Random) -> "ChampagneTheme":
        inst = cls()
        n_bub = min(90, max(24, int(round(math.sqrt(len(scene.polys)) * 10))))
        stream_x = [rng.uniform(0.12, 0.88) for _ in range(rng.randint(4, 7))]
        bubbles: List[Bubble] = []
        for _ in range(n_bub):
            sx = rng.choice(stream_x)
            x = clamp01(sx + rng.uniform(-0.06, 0.06))
            bubbles.append(Bubble(
                x=x,
                y0=rng.uniform(0.0, 1.0),
                r=rng.uniform(0.010, 0.040) * rng.uniform(0.8, 1.3),
                speed=rng.uniform(0.08, 0.26),
                wob_amp=rng.uniform(0.005, 0.020),
                wob_freq=rng.uniform(0.20, 0.80),
                wob_phase=rng.uniform(0.0, 1.0),
                strength=rng.uniform(0.35, 1.00),
            ))
        inst.bubbles = bubbles
        inst.ch_freq = []
        inst.ch_phase = []
        for _ in scene.polys:
            inst.ch_freq.append(rng.uniform(0.05, 0.22))
            inst.ch_phase.append(rng.uniform(0.0, 1.0))
        return inst

    def apply_frame(self, scene: Scene, t: float) -> None:
        cfg = get_theme_config("champagne")
        amb = cfg.amb_base + cfg.amb_amp * (0.5 + 0.5 * math.sin(2.0 * math.pi * (cfg.amb_freq * t)))

        for idx, poly in enumerate(scene.polys):
            nx = scene.poly_nx[idx]
            ny = scene.poly_ny[idx]

            h = (45/360.0) + 0.010 * math.sin(2.0 * math.pi * (0.04 * t + self.ch_phase[idx]))
            s = 0.18 + 0.12 * (0.5 + 0.5 * math.sin(2.0 * math.pi * (self.ch_freq[idx] * t + self.ch_phase[idx])))
            v = 0.30 + 0.55 * (1.0 - ny)
            v += 0.06 * (facet_shimmer(t, 0.12 + self.ch_freq[idx], self.ch_phase[idx]) - 0.5)
            v = clamp01(v)

            rgb = hsv01_to_rgb255(h, s, v)

            bub = 0.0
            bub_edge = 0.0
            for b in self.bubbles:
                by = (b.y0 - b.speed * t) % 1.0
                bx = clamp01(b.x + b.wob_amp * math.sin(2.0 * math.pi * (b.wob_freq * t + b.wob_phase)))
                infl = bubble_influence(nx, ny, bx, by, b.r)
                if infl > 1e-6:
                    bub = max(bub, infl * b.strength)
                    infl2 = bubble_influence(nx, ny, bx, by, b.r * 0.60)
                    bub_edge = max(bub_edge, infl2 * b.strength)

            bub = clamp01(bub)
            bub_edge = clamp01(bub_edge)

            if bub > 0.0:
                bubble_rgb = hsv01_to_rgb255(200/360.0, 0.08, 1.0)
                rgb = mix_rgb(rgb, bubble_rgb, 0.70 * bub)
                rgb = mix_rgb(rgb, (255, 255, 255), 0.55 * bub_edge)

                tint_gate = smoothstep(0.30, 0.90, bub_edge)
                if tint_gate > 0.0:
                    fh = (0.10 + 0.20 * math.sin(2.0 * math.pi * (0.06 * t + nx))) % 1.0
                    tint = hsv01_to_rgb255(fh, 0.55, 1.0)
                    rgb = mix_rgb(rgb, tint, 0.10 * tint_gate)

            rgb = mix_to_white(rgb, amb * 0.08)
            poly.set("fill", rgb_to_hex(rgb))
