from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass(frozen=True)
class ThemeConfig:
    kind: str
    base_hue: Optional[float] = None
    hue_jitter: float = 0.0

    body_sat_min: float = 0.0
    body_sat_max: float = 0.0
    body_v_min: float = 0.0
    body_v_max: float = 1.0
    body_v_gamma: float = 1.0

    body_sat_mul: float = 1.0
    sat_dark_boost: float = 0.0

    hue_tone_amp: float = 0.0
    hue_shimmer_amp: float = 0.0
    val_shimmer_amp: float = 0.0

    amb_base: float = 0.05
    amb_amp: float = 0.03
    amb_freq: float = 0.025

    gl_pulse_w: float = 0.86
    gl_shim_w: float = 0.55

    spec_edge0: float = 0.48
    spec_scale: float = 0.98

    sheen_mix: float = 0.20
    sheen_sat_boost: float = 0.15
    sheen_hue_shift: float = 0.0

    fire_prob: float = 0.0
    fire_hues: Optional[List[float]] = None
    fire_hue_jitter: float = 0.0
    fire_hue_drift_amp: float = 0.015
    fire_hue_drift_freq: float = 0.06

    fire_gate0: float = 0.58
    fire_sat_base_min: float = 0.12
    fire_sat_base_max: float = 0.28
    fire_sat_peak_min: float = 0.40
    fire_sat_peak_max: float = 0.78

    fire_sat_mul_lo: float = 0.85
    fire_sat_mul_hi: float = 1.25

    fire_white_mix_min: float = 0.48
    fire_white_mix_max: float = 0.72

_FIRE_HUES_DEFAULT: List[float] = [
    200 / 360.0, 220 / 360.0, 255 / 360.0, 300 / 360.0,
    330 / 360.0, 45 / 360.0, 70 / 360.0, 150 / 360.0,
]

def _cfg_merge(common: Dict, specific: Dict) -> ThemeConfig:
    merged = dict(common)
    merged.update(specific)
    return ThemeConfig(**merged)

def get_theme_config(theme: str) -> ThemeConfig:
    if theme == "none":
        return ThemeConfig(kind="none")

    if theme == "clouds":
        return ThemeConfig(kind="clouds")

    if theme == "classic":
        return ThemeConfig(kind="classic")

    if theme == "diamond":
        return ThemeConfig(
            kind="diamond",
            amb_base=0.045, amb_amp=0.030, amb_freq=0.022,
            spec_edge0=0.48, spec_scale=0.98,
            fire_prob=0.38,
            fire_hues=_FIRE_HUES_DEFAULT,
            fire_hue_jitter=0.02,
            fire_sat_base_min=0.12, fire_sat_base_max=0.28,
            fire_sat_peak_min=0.45, fire_sat_peak_max=0.85,
            fire_white_mix_min=0.48, fire_white_mix_max=0.72,
        )

    if theme == "minecraft":
        return ThemeConfig(
            kind="minecraft",
            amb_base=0.02, amb_amp=0.03, amb_freq=0.030,
            gl_pulse_w=0.80, gl_shim_w=0.52,
            spec_edge0=0.36, spec_scale=0.72,
            sheen_mix=0.10, sheen_sat_boost=0.10,
            fire_prob=0.0,
        )

    if theme == "deidee":
        return ThemeConfig(kind="deidee")

    if theme == "heart":
        return ThemeConfig(kind="heart")

    if theme == "static":
        return ThemeConfig(kind="static")

    if theme == "champagne":
        return ThemeConfig(kind="champagne")

    if theme == "camo":
        # Matte, low-glint fabric-ish camo. These values match the last
        # known-good single-file generator (dark ambience, minimal spec).
        return ThemeConfig(
            kind="camo",
            amb_base=0.015, amb_amp=0.020, amb_freq=0.030,
            spec_edge0=0.72, spec_scale=0.20,
            sheen_mix=0.00,
            fire_prob=0.0,
        )

    if theme == "fireworks":
        # Dark base + bright additive bursts. Keep ambience low so the
        # firework arrows/bursts read as *light* on a night sky.
        return ThemeConfig(
            kind="fireworks",
            amb_base=0.010, amb_amp=0.012, amb_freq=0.040,
            spec_edge0=0.62, spec_scale=0.35,
            sheen_mix=0.00,
            fire_prob=0.0,
        )

    if theme == "nuke":
        return ThemeConfig(kind="nuke")

    common = dict(
        kind="hsv",
        amb_base=0.06,
        amb_amp=0.025,
        spec_edge0=0.54,
        spec_scale=0.86,
        sheen_mix=0.18,
        sheen_sat_boost=0.22,
        val_shimmer_amp=0.035,
        fire_white_mix_min=0.10,
        fire_white_mix_max=0.42,
    )

    if theme == "silver":
        return _cfg_merge(common, dict(
            base_hue=210/360.0, hue_jitter=0.012,
            body_sat_min=0.22, body_sat_max=0.55,
            body_v_min=0.22, body_v_max=0.92, body_v_gamma=1.0,
            body_sat_mul=1.06,
            sat_dark_boost=0.10,
            hue_tone_amp=0.012,
            hue_shimmer_amp=0.010,
            sheen_mix=0.24, sheen_sat_boost=0.18, sheen_hue_shift=0.006,
            fire_prob=0.26,
            fire_hues=[200/360.0, 210/360.0, 225/360.0, 240/360.0],
            fire_hue_jitter=0.04,
            fire_sat_base_min=0.18, fire_sat_base_max=0.40,
            fire_sat_peak_min=0.45, fire_sat_peak_max=0.85,
        ))

    if theme == "gold":
        return _cfg_merge(common, dict(
            base_hue=45/360.0, hue_jitter=0.024,
            body_sat_min=0.60, body_sat_max=1.00,
            body_v_min=0.20, body_v_max=0.92, body_v_gamma=1.0,
            body_sat_mul=1.08,
            sat_dark_boost=0.10,
            hue_tone_amp=0.018,
            hue_shimmer_amp=0.012,
            sheen_mix=0.22, sheen_sat_boost=0.22, sheen_hue_shift=0.010,
            fire_prob=0.38,
            fire_hues=[35/360.0, 45/360.0, 55/360.0, 30/360.0, 65/360.0],
            fire_hue_jitter=0.05,
            fire_sat_base_min=0.24, fire_sat_base_max=0.52,
            fire_sat_peak_min=0.70, fire_sat_peak_max=1.00,
        ))

    if theme == "bronze":
        return _cfg_merge(common, dict(
            base_hue=28/360.0, hue_jitter=0.030,
            body_sat_min=0.55, body_sat_max=1.00,
            body_v_min=0.18, body_v_max=0.90, body_v_gamma=1.05,
            body_sat_mul=1.08,
            sat_dark_boost=0.12,
            hue_tone_amp=0.020,
            hue_shimmer_amp=0.012,
            sheen_mix=0.22, sheen_sat_boost=0.22, sheen_hue_shift=0.010,
            fire_prob=0.34,
            fire_hues=[20/360.0, 28/360.0, 35/360.0, 12/360.0, 45/360.0],
            fire_hue_jitter=0.05,
            fire_sat_base_min=0.22, fire_sat_base_max=0.52,
            fire_sat_peak_min=0.65, fire_sat_peak_max=1.00,
        ))

    if theme == "matrix":
        return ThemeConfig(kind="matrix")

    if theme == "ruby":
        return _cfg_merge(common, dict(
            base_hue=350/360.0, hue_jitter=0.022,
            body_sat_min=0.82, body_sat_max=1.00,
            body_v_min=0.14, body_v_max=0.90, body_v_gamma=1.08,
            body_sat_mul=1.10,
            sat_dark_boost=0.30,
            hue_tone_amp=0.010,
            hue_shimmer_amp=0.016,
            sheen_mix=0.14, sheen_sat_boost=0.30,
            fire_prob=0.46,
            fire_hues=None,
            fire_hue_jitter=0.06,
            fire_sat_base_min=0.28, fire_sat_base_max=0.58,
            fire_sat_peak_min=0.80, fire_sat_peak_max=1.00,
        ))

    if theme == "jade":
        return _cfg_merge(common, dict(
            base_hue=145/360.0, hue_jitter=0.025,
            body_sat_min=0.70, body_sat_max=1.00,
            body_v_min=0.16, body_v_max=0.92, body_v_gamma=1.0,
            body_sat_mul=1.08,
            sat_dark_boost=0.25,
            hue_tone_amp=0.010,
            hue_shimmer_amp=0.016,
            sheen_mix=0.14, sheen_sat_boost=0.28,
            fire_prob=0.42,
            fire_hues=[130/360.0, 145/360.0, 160/360.0, 120/360.0, 175/360.0],
            fire_hue_jitter=0.06,
            fire_sat_base_min=0.26, fire_sat_base_max=0.54,
            fire_sat_peak_min=0.75, fire_sat_peak_max=1.00,
        ))

    if theme == "sapphire":
        return _cfg_merge(common, dict(
            base_hue=220/360.0, hue_jitter=0.025,
            body_sat_min=0.78, body_sat_max=1.00,
            body_v_min=0.14, body_v_max=0.90, body_v_gamma=1.08,
            body_sat_mul=1.10,
            sat_dark_boost=0.28,
            hue_tone_amp=0.010,
            hue_shimmer_amp=0.018,
            sheen_mix=0.14, sheen_sat_boost=0.30,
            fire_prob=0.44,
            fire_hues=[205/360.0, 220/360.0, 240/360.0, 255/360.0, 190/360.0],
            fire_hue_jitter=0.06,
            fire_sat_base_min=0.26, fire_sat_base_max=0.54,
            fire_sat_peak_min=0.80, fire_sat_peak_max=1.00,
        ))

    if theme == "emerald":
        return _cfg_merge(common, dict(
            base_hue=140/360.0, hue_jitter=0.025,
            body_sat_min=0.80, body_sat_max=1.00,
            body_v_min=0.14, body_v_max=0.90, body_v_gamma=1.08,
            body_sat_mul=1.10,
            sat_dark_boost=0.28,
            hue_tone_amp=0.010,
            hue_shimmer_amp=0.018,
            sheen_mix=0.14, sheen_sat_boost=0.30,
            fire_prob=0.44,
            fire_hues=[125/360.0, 140/360.0, 155/360.0, 165/360.0, 110/360.0],
            fire_hue_jitter=0.06,
            fire_sat_base_min=0.26, fire_sat_base_max=0.54,
            fire_sat_peak_min=0.80, fire_sat_peak_max=1.00,
        ))

    if theme == "rainbow":
        return _cfg_merge(common, dict(
            base_hue=None, hue_jitter=0.0,
            body_sat_min=0.82, body_sat_max=1.00,
            body_v_min=0.18, body_v_max=0.92, body_v_gamma=1.0,
            body_sat_mul=1.08,
            sat_dark_boost=0.18,
            hue_shimmer_amp=0.030,
            val_shimmer_amp=0.040,
            spec_edge0=0.50, spec_scale=0.90,
            sheen_mix=0.12, sheen_sat_boost=0.32,
            fire_prob=0.80,
            fire_hues=_FIRE_HUES_DEFAULT,
            fire_hue_jitter=0.10,
            fire_sat_base_min=0.34, fire_sat_base_max=0.70,
            fire_sat_peak_min=0.90, fire_sat_peak_max=1.00,
            fire_white_mix_min=0.06,
            fire_white_mix_max=0.30,
        ))

    if theme == "fire":
        return _cfg_merge(common, dict(
            base_hue=25/360.0, hue_jitter=0.070,
            body_sat_min=0.85, body_sat_max=1.00,
            body_v_min=0.12, body_v_max=0.90, body_v_gamma=1.08,
            body_sat_mul=1.10,
            sat_dark_boost=0.38,
            hue_tone_amp=0.012,
            hue_shimmer_amp=0.020,
            val_shimmer_amp=0.030,
            spec_edge0=0.46, spec_scale=0.92,
            sheen_mix=0.10, sheen_sat_boost=0.36, sheen_hue_shift=0.018,
            fire_prob=0.92,
            fire_hues=[0/360.0, 12/360.0, 25/360.0, 40/360.0, 55/360.0, 65/360.0, 330/360.0],
            fire_hue_jitter=0.11,
            fire_sat_base_min=0.40, fire_sat_base_max=0.80,
            fire_sat_peak_min=0.90, fire_sat_peak_max=1.00,
            fire_white_mix_min=0.05,
            fire_white_mix_max=0.22,
        ))

    if theme == "ice":
        return _cfg_merge(common, dict(
            base_hue=205/360.0, hue_jitter=0.055,
            body_sat_min=0.75, body_sat_max=1.00,
            body_v_min=0.18, body_v_max=0.96, body_v_gamma=0.95,
            body_sat_mul=1.08,
            sat_dark_boost=0.16,
            hue_tone_amp=0.010,
            hue_shimmer_amp=0.020,
            val_shimmer_amp=0.040,
            spec_edge0=0.52, spec_scale=0.92,
            sheen_mix=0.22, sheen_sat_boost=0.30, sheen_hue_shift=-0.010,
            fire_prob=0.62,
            fire_hues=[175/360.0, 195/360.0, 210/360.0, 225/360.0, 245/360.0, 275/360.0],
            fire_hue_jitter=0.10,
            fire_sat_base_min=0.25, fire_sat_base_max=0.55,
            fire_sat_peak_min=0.75, fire_sat_peak_max=1.00,
            fire_white_mix_min=0.12,
            fire_white_mix_max=0.35,
        ))

    if theme == "valentines":
        return _cfg_merge(common, dict(
            base_hue=335/360.0, hue_jitter=0.090,
            body_sat_min=0.78, body_sat_max=1.00,
            body_v_min=0.22, body_v_max=0.98, body_v_gamma=0.95,
            body_sat_mul=1.10,
            sat_dark_boost=0.18,
            hue_tone_amp=0.012,
            hue_shimmer_amp=0.028,
            val_shimmer_amp=0.045,
            spec_edge0=0.50, spec_scale=0.92,
            sheen_mix=0.16, sheen_sat_boost=0.34, sheen_hue_shift=0.010,
            fire_prob=0.78,
            fire_hues=[350/360.0, 0/360.0, 10/360.0, 330/360.0, 315/360.0, 45/360.0],
            fire_hue_jitter=0.10,
            fire_sat_base_min=0.28, fire_sat_base_max=0.62,
            fire_sat_peak_min=0.80, fire_sat_peak_max=1.00,
            fire_white_mix_min=0.10, fire_white_mix_max=0.38,
        ))

    if theme == "snow":
        return _cfg_merge(common, dict(
            base_hue=205/360.0, hue_jitter=0.05,
            body_sat_min=0.04, body_sat_max=0.30,
            body_v_min=0.10, body_v_max=1.00, body_v_gamma=0.90,
            body_sat_mul=1.00,
            sat_dark_boost=0.05,
            hue_tone_amp=0.004,
            hue_shimmer_amp=0.010,
            val_shimmer_amp=0.055,
            spec_edge0=0.52, spec_scale=0.96,
            sheen_mix=0.30, sheen_sat_boost=0.06, sheen_hue_shift=-0.004,
            fire_prob=0.55,
            fire_hues=[190/360.0, 205/360.0, 220/360.0, 235/360.0, 255/360.0],
            fire_hue_jitter=0.08,
            fire_sat_base_min=0.10, fire_sat_base_max=0.30,
            fire_sat_peak_min=0.25, fire_sat_peak_max=0.55,
            fire_white_mix_min=0.60, fire_white_mix_max=0.88,
        ))

    raise ValueError(f"Unknown theme: {theme!r}")
