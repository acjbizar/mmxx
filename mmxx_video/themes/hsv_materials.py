from __future__ import annotations

"""HSV-based material themes.

These themes (silver/gold/bronze/ruby/jade/sapphire/emerald/rainbow/fire/ice)
used to share a richer material model in the original monolithic generator
script: tone-driven value, facet shimmer + pulses feeding specular, a proper
ambient lift (mix-to-white), and optional "fire" glints.

During the refactor, this material model was simplified and the ambient lift was
scaled down, which made several themes (notably `ruby`) darker and less vibrant.

This module intentionally mirrors the original look.
"""

import math
import random
from dataclasses import dataclass
from typing import List, Optional

from ..anim.color_math import (
    hsv01_to_rgb255,
    mix_rgb,
    mix_to_white,
    rgb255_to_hsv01,
    rgb_to_hex,
)
from ..anim.easing import clamp01, smoothstep
from ..anim.pulses import facet_shimmer, whiteness_at
from ..scene import Scene
from .configs import ThemeConfig, get_theme_config


@dataclass
class PolyHSV:
    # Tone & shimmer (drives value/spec)
    tone_base: float
    tone_phase: float
    shimmer_freq: float
    shimmer_phase: float

    # Body hue/sat
    body_hue: float
    body_sat: float

    # Optional fire/glint layer
    fire_enabled: bool
    fire_hue: float
    fire_hue_phase: float
    fire_sat_base: float
    fire_sat_peak: float
    fire_sat_mul: float
    fire_white_mix: float


@dataclass
class HSVTheme:
    cfg: ThemeConfig
    poly: List[PolyHSV]
    name: str

    @classmethod
    def create(cls, scene: Scene, args, rng: random.Random) -> "HSVTheme":
        cfg = get_theme_config(args.theme)
        assert cfg.kind == "hsv"

        override_hsv: Optional[tuple] = getattr(scene, "override_hsv", None)
        override_hsv_per_poly = getattr(scene, "override_hsv_per_poly", None)

        poly: List[PolyHSV] = []
        for _i in range(scene.n_polys):
            # ---- legacy tone distribution ----
            u = rng.random()
            if u < 0.86:
                tone = rng.uniform(0.38, 0.70)
            elif u < 0.97:
                tone = rng.uniform(0.30, 0.78)
            else:
                tone = rng.uniform(0.22, 0.86)

            tone_phase = rng.uniform(0.0, 1.0)
            shimmer_freq = rng.uniform(0.05, 0.14)
            shimmer_phase = rng.uniform(0.0, 1.0)

            # ---- body hue/sat ----
            # Per-poly override (from --colors) takes precedence over global --color
            ov = None
            if isinstance(override_hsv_per_poly, list) and _i < len(override_hsv_per_poly):
                ov = override_hsv_per_poly[_i]
            if ov is None:
                ov = override_hsv

            if ov is not None:
                h0, s0, _v0 = ov
                body_hue = float(h0)
                body_sat = max(
                    cfg.body_sat_min,
                    min(
                        cfg.body_sat_max,
                        max(float(s0), rng.uniform(cfg.body_sat_min, cfg.body_sat_max)) * 0.90,
                    ),
                )
            else:
                # Refactor configs don't carry the palette-bias fields from the legacy script.
                # When base_hue is None, pick any hue.
                if cfg.base_hue is None:
                    body_hue = rng.random()
                else:
                    body_hue = (cfg.base_hue + rng.uniform(-cfg.hue_jitter, cfg.hue_jitter)) % 1.0

                body_sat = rng.uniform(cfg.body_sat_min, cfg.body_sat_max)

            # ---- fire/glint settings ----
            fire_enabled = rng.random() < cfg.fire_prob

            if cfg.fire_hues is not None:
                fire_hue = rng.choice(cfg.fire_hues)
            else:
                fire_hue = body_hue
            fire_hue = (fire_hue + rng.uniform(-cfg.fire_hue_jitter, cfg.fire_hue_jitter)) % 1.0
            fire_hue_phase = rng.uniform(0.0, 1.0)

            fire_sat_base = rng.uniform(cfg.fire_sat_base_min, cfg.fire_sat_base_max)
            fire_sat_peak = rng.uniform(cfg.fire_sat_peak_min, cfg.fire_sat_peak_max)
            fire_sat_mul = rng.uniform(cfg.fire_sat_mul_lo, cfg.fire_sat_mul_hi)

            fire_white_mix = rng.uniform(cfg.fire_white_mix_min, cfg.fire_white_mix_max)

            poly.append(
                PolyHSV(
                    tone_base=tone,
                    tone_phase=tone_phase,
                    shimmer_freq=shimmer_freq,
                    shimmer_phase=shimmer_phase,
                    body_hue=body_hue,
                    body_sat=body_sat,
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

        # Legacy global ambient lift (do NOT downscale).
        global_amb = cfg.amb_base + cfg.amb_amp * (0.5 + 0.5 * math.sin(2.0 * math.pi * (cfg.amb_freq * t)))

        for idx, poly_el in enumerate(scene.polys):
            ph = self.poly[idx]

            # tone wobble (slow)
            wobble = 0.050 * math.sin(2.0 * math.pi * (0.06 * t + ph.tone_phase))
            tone = clamp01(ph.tone_base + wobble)

            pulse = whiteness_at(t, scene.pulses_per_poly[idx])
            shim = facet_shimmer(t, ph.shimmer_freq, ph.shimmer_phase)

            gl = clamp01(cfg.gl_pulse_w * pulse + cfg.gl_shim_w * shim)
            spec = smoothstep(cfg.spec_edge0, 1.00, gl)
            spec_amt = clamp01(spec * cfg.spec_scale)

            # body HSV driven by tone
            h = (ph.body_hue + cfg.hue_tone_amp * (tone - 0.5) + cfg.hue_shimmer_amp * (shim - 0.5)) % 1.0

            v = cfg.body_v_min + (cfg.body_v_max - cfg.body_v_min) * (tone ** cfg.body_v_gamma)
            if cfg.val_shimmer_amp:
                v = clamp01(v + cfg.val_shimmer_amp * (shim - 0.5))

            s = ph.body_sat * cfg.body_sat_mul
            s *= (1.0 + cfg.sat_dark_boost * (1.0 - tone))
            s = clamp01(s)

            body = hsv01_to_rgb255(h, s, v)
            body = mix_to_white(body, global_amb)

            if ph.fire_enabled:
                fire_gate = smoothstep(cfg.fire_gate0, 1.00, gl)

                hue = (
                    ph.fire_hue
                    + cfg.fire_hue_drift_amp * math.sin(2.0 * math.pi * (cfg.fire_hue_drift_freq * t + ph.fire_hue_phase))
                ) % 1.0

                sat = ph.fire_sat_base + (ph.fire_sat_peak - ph.fire_sat_base) * fire_gate
                sat = clamp01(sat * ph.fire_sat_mul)

                fire_rgb = hsv01_to_rgb255(hue, sat, 1.0)
                tw = mix_rgb(fire_rgb, (255, 255, 255), ph.fire_white_mix)

                rgb = mix_rgb(body, tw, spec_amt)
            else:
                # sheen highlight, mixed by spec_amt (legacy)
                hb, sb, _vb = rgb255_to_hsv01(body)
                hb2 = (hb + cfg.sheen_hue_shift + 0.010 * (shim - 0.5)) % 1.0
                sb2 = clamp01(sb * (1.0 + cfg.sheen_sat_boost))
                sheen_rgb = hsv01_to_rgb255(hb2, sb2, 1.0)
                sheen = mix_rgb(sheen_rgb, (255, 255, 255), cfg.sheen_mix)
                rgb = mix_rgb(body, sheen, spec_amt)

            poly_el.set("fill", rgb_to_hex(rgb))
