from __future__ import annotations
import math, random
from dataclasses import dataclass
from typing import Any, List
from ..scene import Scene
from ..anim.easing import clamp01, smoothstep
from ..anim.pulses import whiteness_at, facet_shimmer
from ..anim.color_math import mix_rgb, hsv01_to_rgb255, mix_to_white, rgb_to_hex
from .configs import ThemeConfig, get_theme_config

@dataclass
class PolyHSV:
    h: float
    s: float
    v: float
    sat_mul: float
    v_mul: float
    freq: float
    phase: float
    fire_enabled: bool
    fire_hue: float
    fire_sat_mul: float

@dataclass
class HSVTheme:
    cfg: ThemeConfig
    poly: List[PolyHSV]
    name: str

    @classmethod
    def create(cls, *, scene: Scene, args: Any, rng: random.Random) -> "HSVTheme":
        cfg = get_theme_config(args.theme)
        poly: List[PolyHSV] = []
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

            if cfg.base_hue is None:
                h = (nx + rng.uniform(-0.05, 0.05)) % 1.0
            else:
                h = (cfg.base_hue
                     + cfg.hue_jitter * rng.uniform(-1.0, 1.0)
                     + cfg.hue_tone_amp * (nx - 0.5)) % 1.0

            s = rng.uniform(cfg.body_sat_min, cfg.body_sat_max)
            sat_mul = rng.uniform(0.85, 1.25)

            ug = rng.random()
            vg = cfg.body_v_min + (cfg.body_v_max - cfg.body_v_min) * (ug ** cfg.body_v_gamma)
            vg = clamp01(vg + 0.10 * (0.5 - ny))
            v = vg
            v_mul = rng.uniform(0.92, 1.12)

            fire_enabled = (rng.random() < cfg.fire_prob)
            if cfg.fire_hues:
                fh = rng.choice(cfg.fire_hues)
            else:
                fh = (h + rng.uniform(-0.10, 0.10)) % 1.0
            fire_hue = (fh + rng.uniform(-cfg.fire_hue_jitter, cfg.fire_hue_jitter)) % 1.0
            fire_sat_mul = rng.uniform(cfg.fire_sat_mul_lo, cfg.fire_sat_mul_hi)

            poly.append(PolyHSV(
                h=h, s=s, v=v,
                sat_mul=sat_mul, v_mul=v_mul,
                freq=freq, phase=phase,
                fire_enabled=fire_enabled,
                fire_hue=fire_hue,
                fire_sat_mul=fire_sat_mul,
            ))
        return cls(cfg=cfg, poly=poly, name=args.theme)

    def apply_frame(self, scene: Scene, t: float) -> None:
        cfg = self.cfg
        amb = cfg.amb_base + cfg.amb_amp * (0.5 + 0.5 * math.sin(2.0 * math.pi * (cfg.amb_freq * t)))

        for idx, poly_el in enumerate(scene.polys):
            ph = self.poly[idx]
            pulse = whiteness_at(t, scene.pulses_per_poly[idx])
            shim = facet_shimmer(t, ph.freq, ph.phase)

            drift_h = cfg.hue_shimmer_amp * math.sin(2.0 * math.pi * (0.10 * t + ph.phase))
            h = (ph.h + drift_h) % 1.0

            v = clamp01((ph.v * ph.v_mul) + cfg.val_shimmer_amp * (shim - 0.5))
            v = clamp01(0.06 + 0.90 * v)

            s = ph.s * cfg.body_sat_mul * ph.sat_mul
            s = clamp01(s + cfg.sat_dark_boost * (1.0 - v))

            rgb = hsv01_to_rgb255(h, s, v)

            glint = max(cfg.gl_pulse_w * pulse, cfg.gl_shim_w * shim)
            spec = smoothstep(cfg.spec_edge0, 1.0, glint)
            spec = clamp01(spec * cfg.spec_scale)

            sh = (h + cfg.sheen_hue_shift) % 1.0
            ss = clamp01(s + cfg.sheen_sat_boost)
            sheen_rgb = hsv01_to_rgb255(sh, ss, 1.0)
            rgb = mix_rgb(rgb, sheen_rgb, cfg.sheen_mix * spec)

            rgb = mix_rgb(rgb, (255, 255, 255), 0.06 * spec)

            if ph.fire_enabled:
                gate = smoothstep(cfg.fire_gate0, 1.0, glint)
                if gate > 0.001:
                    drift = cfg.fire_hue_drift_amp * math.sin(2.0 * math.pi * (cfg.fire_hue_drift_freq * t + ph.phase))
                    fh = (ph.fire_hue + drift) % 1.0
                    sat_base = cfg.fire_sat_base_min + (cfg.fire_sat_base_max - cfg.fire_sat_base_min) * (0.5 + 0.5 * math.sin(2*math.pi*(0.15*t + ph.phase)))
                    sat_peak = cfg.fire_sat_peak_min + (cfg.fire_sat_peak_max - cfg.fire_sat_peak_min) * (0.5 + 0.5 * math.sin(2*math.pi*(0.22*t + ph.freq)))
                    s2 = clamp01((sat_base + gate * sat_peak) * ph.fire_sat_mul)
                    v2 = clamp01(0.55 + 0.45 * gate)
                    fire_rgb = hsv01_to_rgb255(fh, s2, v2)
                    rgb = mix_rgb(rgb, fire_rgb, clamp01(gate * 0.45))

            rgb = mix_to_white(rgb, amb * 0.10)
            poly_el.set("fill", rgb_to_hex(rgb))
