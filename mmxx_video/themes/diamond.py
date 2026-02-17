from __future__ import annotations

"""Diamond theme.

The original generator used a high-contrast greyscale body with a crisp specular
response driven by pulses and facet shimmer, plus occasional colorful "fire"
(glints/dispersion) in highlights. The refactor version drifted darker due to
aggressive ambient scaling.

This module mirrors the original look.
"""

import math
import random
from dataclasses import dataclass
from typing import List

from ..anim.color_math import hsv01_to_rgb255, mix_rgb, mix_to_white, rgb_to_hex
from ..anim.easing import clamp01, smoothstep
from ..anim.pulses import facet_shimmer, whiteness_at
from ..scene import Scene
from .configs import ThemeConfig, get_theme_config


GREY_DARK = (6, 6, 8)
GREY_LIGHT = (255, 255, 255)


@dataclass
class PolyDiamond:
    tone_base: float
    tone_phase: float
    shimmer_freq: float
    shimmer_phase: float

    fire_enabled: bool
    fire_hue: float
    fire_hue_phase: float
    fire_sat_base: float
    fire_sat_peak: float
    fire_sat_mul: float
    fire_white_mix: float


@dataclass
class DiamondTheme:
    cfg: ThemeConfig
    poly: List[PolyDiamond]
    name: str

    @classmethod
    def create(cls, scene: Scene, args, rng: random.Random) -> "DiamondTheme":
        cfg = get_theme_config(args.theme)
        assert cfg.kind == "diamond"

        poly: List[PolyDiamond] = []
        for _i in range(scene.n_polys):
            # legacy diamond tone: lots of near-black / near-white facets
            if rng.random() < 0.52:
                tone = (rng.random() ** 2.2) * 0.28
            else:
                tone = 1.0 - (rng.random() ** 2.2) * 0.28

            tone_phase = rng.uniform(0.0, 1.0)
            shimmer_freq = rng.uniform(0.05, 0.14)
            shimmer_phase = rng.uniform(0.0, 1.0)

            fire_enabled = rng.random() < cfg.fire_prob
            if cfg.fire_hues:
                fire_hue = rng.choice(cfg.fire_hues)
            else:
                fire_hue = rng.random()
            fire_hue = (fire_hue + rng.uniform(-cfg.fire_hue_jitter, cfg.fire_hue_jitter)) % 1.0
            fire_hue_phase = rng.uniform(0.0, 1.0)

            fire_sat_base = rng.uniform(cfg.fire_sat_base_min, cfg.fire_sat_base_max)
            fire_sat_peak = rng.uniform(cfg.fire_sat_peak_min, cfg.fire_sat_peak_max)
            fire_sat_mul = rng.uniform(cfg.fire_sat_mul_lo, cfg.fire_sat_mul_hi)

            fire_white_mix = rng.uniform(cfg.fire_white_mix_min, cfg.fire_white_mix_max)

            poly.append(
                PolyDiamond(
                    tone_base=tone,
                    tone_phase=tone_phase,
                    shimmer_freq=shimmer_freq,
                    shimmer_phase=shimmer_phase,
                    fire_enabled=fire_enabled,
                    fire_hue=fire_hue,
                    fire_hue_phase=fire_hue_phase,
                    fire_sat_base=fire_sat_base,
                    fire_sat_peak=fire_sat_peak,
                    fire_sat_mul=fire_sat_mul,
                    fire_white_mix=fire_white_mix,
                )
            )

        return cls(cfg=cfg, poly=poly, name=args.theme)

    def apply_frame(self, scene: Scene, t: float) -> None:
        cfg = self.cfg

        global_amb = cfg.amb_base + cfg.amb_amp * (0.5 + 0.5 * math.sin(2.0 * math.pi * (cfg.amb_freq * t)))

        for idx, poly_el in enumerate(scene.polys):
            pd = self.poly[idx]

            wobble = 0.08 * math.sin(2.0 * math.pi * (0.06 * t + pd.tone_phase))
            tone = clamp01(pd.tone_base + wobble)

            a = whiteness_at(t, scene.pulses_per_poly[idx])
            shim = facet_shimmer(t, pd.shimmer_freq, pd.shimmer_phase)
            gl = clamp01(cfg.gl_pulse_w * a + cfg.gl_shim_w * shim)

            spec = smoothstep(cfg.spec_edge0, 1.00, gl)
            spec_amt = clamp01(spec * cfg.spec_scale)

            body_grey = mix_rgb(GREY_DARK, GREY_LIGHT, tone)
            body = mix_to_white(body_grey, global_amb)

            if pd.fire_enabled:
                fire_gate = smoothstep(cfg.fire_gate0, 1.00, gl)
                hue = (
                    pd.fire_hue
                    + cfg.fire_hue_drift_amp * math.sin(2.0 * math.pi * (cfg.fire_hue_drift_freq * t + pd.fire_hue_phase))
                ) % 1.0

                sat = pd.fire_sat_base + (pd.fire_sat_peak - pd.fire_sat_base) * fire_gate
                sat = clamp01(sat * pd.fire_sat_mul)

                fire_rgb = hsv01_to_rgb255(hue, sat, 1.0)
                tw = mix_rgb(fire_rgb, (255, 255, 255), pd.fire_white_mix)
                rgb = mix_rgb(body, tw, spec_amt)
            else:
                rgb = mix_rgb(body, (255, 255, 255), spec_amt)

            poly_el.set("fill", rgb_to_hex(rgb))
