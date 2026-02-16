from __future__ import annotations
import math, random
from dataclasses import dataclass
from typing import Any, List
from ..scene import Scene
from ..anim.easing import clamp01, smoothstep
from ..anim.pulses import whiteness_at, facet_shimmer
from ..anim.color_math import mix_rgb, hsv01_to_rgb255, mix_to_white, rgb_to_hex
from .configs import get_theme_config, ThemeConfig

@dataclass
class PolyState:
    v: float
    v_mul: float
    freq: float
    phase: float
    fire_enabled: bool
    fire_hue: float
    fire_sat_mul: float
    fire_white_mix: float

@dataclass
class DiamondTheme:
    cfg: ThemeConfig
    poly: List[PolyState]
    name: str = "diamond"

    @classmethod
    def create(cls, *, scene: Scene, args: Any, rng: random.Random) -> "DiamondTheme":
        cfg = get_theme_config("diamond")
        poly: List[PolyState] = []
        for idx in range(len(scene.polys)):
            nx = scene.poly_nx[idx]
            ny = scene.poly_ny[idx]
            freq = rng.uniform(0.18, 0.85)
            phase = rng.uniform(0.0, 1.0)

            u = rng.random()
            if u < 0.18:
                v = (u / 0.18) * 0.10
            elif u > 0.82:
                v = 0.90 + ((u - 0.82) / 0.18) * 0.10
            else:
                v = 0.10 + ((u - 0.18) / 0.64) * 0.80

            v *= 0.90 + 0.20 * (0.5 + 0.5 * math.sin(2 * math.pi * (nx * 1.3 + ny * 0.7)))
            v = clamp01(v)

            fire_enabled = (rng.random() < cfg.fire_prob)
            fh = rng.choice(cfg.fire_hues) if cfg.fire_hues else rng.random()
            fire_sat_mul = rng.uniform(cfg.fire_sat_mul_lo, cfg.fire_sat_mul_hi)
            fire_white_mix = rng.uniform(cfg.fire_white_mix_min, cfg.fire_white_mix_max)

            poly.append(PolyState(
                v=v,
                v_mul=rng.uniform(0.90, 1.10),
                freq=freq,
                phase=phase,
                fire_enabled=fire_enabled,
                fire_hue=(fh + rng.uniform(-cfg.fire_hue_jitter, cfg.fire_hue_jitter)) % 1.0,
                fire_sat_mul=fire_sat_mul,
                fire_white_mix=fire_white_mix,
            ))
        return cls(cfg=cfg, poly=poly)

    def apply_frame(self, scene: Scene, t: float) -> None:
        cfg = self.cfg
        amb = cfg.amb_base + cfg.amb_amp * (0.5 + 0.5 * math.sin(2.0 * math.pi * (cfg.amb_freq * t)))

        for idx, poly_el in enumerate(scene.polys):
            ps = self.poly[idx]
            pulse = whiteness_at(t, scene.pulses_per_poly[idx])
            shim = facet_shimmer(t, ps.freq, ps.phase)

            glint = max(cfg.gl_pulse_w * pulse, cfg.gl_shim_w * shim)
            spec = smoothstep(cfg.spec_edge0, 1.0, glint)
            spec = clamp01(spec * cfg.spec_scale)

            g0 = clamp01((ps.v * ps.v_mul))
            g = clamp01((g0 - 0.5) * 1.20 + 0.5)
            grey = int(round(g * 255.0))
            rgb = (grey, grey, grey)

            rgb = mix_rgb(rgb, (255, 255, 255), 0.18 + 0.72 * spec)

            if ps.fire_enabled:
                gate = smoothstep(cfg.fire_gate0, 1.0, glint)
                if gate > 0.001:
                    drift = cfg.fire_hue_drift_amp * math.sin(2.0 * math.pi * (cfg.fire_hue_drift_freq * t + ps.phase))
                    fh = (ps.fire_hue + drift) % 1.0
                    sat_base = cfg.fire_sat_base_min + (cfg.fire_sat_base_max - cfg.fire_sat_base_min) * (0.5 + 0.5 * math.sin(2*math.pi*(0.15*t + ps.phase)))
                    sat_peak = cfg.fire_sat_peak_min + (cfg.fire_sat_peak_max - cfg.fire_sat_peak_min) * (0.5 + 0.5 * math.sin(2*math.pi*(0.22*t + ps.freq)))
                    s = clamp01((sat_base + gate * sat_peak) * ps.fire_sat_mul)
                    v = clamp01(0.55 + 0.45 * gate)
                    fire_rgb = hsv01_to_rgb255(fh, s, v)
                    mix_amt = clamp01(gate * (0.20 + 0.55 * ps.fire_white_mix))
                    rgb = mix_rgb(rgb, fire_rgb, mix_amt)

            rgb = mix_to_white(rgb, amb * 0.20)
            poly_el.set("fill", rgb_to_hex(rgb))
