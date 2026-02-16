from __future__ import annotations
import math, random
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from ..scene import Scene
from ..anim.easing import clamp01, smoothstep, cosine_ease
from ..anim.noise import fbm2
from ..anim.color_math import hsv01_to_rgb255, mix_rgb, add_rgb, rgb_to_hex

RGB = Tuple[int, int, int]

def _ellipse_mask(dx: float, dy: float, rx: float, ry: float, edge: float = 0.10) -> float:
    """Soft mask for an ellipse centered at (0,0). Returns 0..1."""
    rx = max(1e-6, rx)
    ry = max(1e-6, ry)
    u = math.sqrt((dx / rx) * (dx / rx) + (dy / ry) * (dy / ry))
    # u <= 1: inside. Fade out between 1..1+edge
    return clamp01(1.0 - smoothstep(1.0, 1.0 + max(1e-6, edge), u))

def _exp_ring(d: float, r: float, w: float) -> float:
    w = max(1e-6, w)
    z = (d - r) / w
    return math.exp(-0.5 * z * z)

@dataclass
class NukeTheme:
    """
    Animated mushroom cloud / nuclear blast inspired theme.

    It is a stylized effect built from:
      - initial flash
      - expanding fireball core
      - rising stem + cap (mushroom)
      - expanding shockwave ring close to the ground
      - smoke/soot takeover over time
    """
    name: str = "nuke"
    seed: float = 0.0
    offx: List[float] = None  # type: ignore[assignment]
    offy: List[float] = None  # type: ignore[assignment]
    freq: List[float] = None  # type: ignore[assignment]
    phase: List[float] = None  # type: ignore[assignment]
    tint_rgb: Optional[RGB] = None

    @classmethod
    def create(cls, *, scene: Scene, args: Any, rng: random.Random) -> "NukeTheme":
        inst = cls()
        inst.seed = float(args.seed if args.seed is not None else rng.randint(0, 10_000_000))
        n = len(scene.polys)
        inst.offx = [rng.uniform(0.0, 1000.0) for _ in range(n)]
        inst.offy = [rng.uniform(0.0, 1000.0) for _ in range(n)]
        inst.freq = [rng.uniform(0.20, 1.05) for _ in range(n)]
        inst.phase = [rng.uniform(0.0, 1.0) for _ in range(n)]

        # Optional artistic tint from --color override (kept subtle, still reads as a nuke).
        if scene.override_hsv is not None:
            h = float(scene.override_hsv[0])
            inst.tint_rgb = hsv01_to_rgb255(h, 0.85, 1.0)
        else:
            inst.tint_rgb = None
        return inst

    def apply_frame(self, scene: Scene, t: float) -> None:
        dur = max(1e-6, float(scene.duration))
        tn = clamp01(t / dur)

        # Ground-centered origin (a bit above the bottom so it reads well in a glyph canvas).
        ox = 0.50
        oy = 0.86

        # Phase controls
        flash = math.exp(-((tn - 0.035) / 0.035) ** 2.0)  # 0..1
        flash = clamp01(flash)

        # Rise + sizes (stylized)
        core_grow = smoothstep(0.02, 0.22, tn)
        core_r = 0.055 + 0.28 * core_grow
        core_y = oy - (0.03 + 0.33 * smoothstep(0.06, 0.55, tn))

        cap_rise = smoothstep(0.12, 0.80, tn)
        cap_y = oy - (0.14 + 0.58 * cap_rise)
        cap_rx = 0.10 + 0.50 * smoothstep(0.15, 0.78, tn)
        cap_ry = 0.07 + 0.22 * smoothstep(0.15, 0.70, tn)

        stem_build = smoothstep(0.10, 0.55, tn)
        stem_r = 0.030 + 0.080 * stem_build

        heat_decay = 1.0 - smoothstep(0.10, 0.82, tn)
        heat_decay = clamp01(heat_decay)

        smoke_build = smoothstep(0.10, 0.42, tn)
        smoke_fade = 1.0 - 0.55 * smoothstep(0.72, 1.00, tn)
        smoke_mul = clamp01(smoke_build * smoke_fade)

        # Shockwave ring
        wave_t = smoothstep(0.03, 0.30, tn) * (1.0 - smoothstep(0.30, 0.62, tn))
        wave_r = 0.02 + 1.10 * smoothstep(0.03, 0.42, tn)
        wave_w = 0.010 + 0.010 * (1.0 - smoothstep(0.03, 0.42, tn))

        # Palette
        hot_white: RGB = (255, 255, 255)
        hot_yellow: RGB = (255, 240, 190)
        hot_orange: RGB = (255, 155, 55)
        hot_red: RGB = (205, 65, 25)

        smoke_dark: RGB = (18, 17, 16)
        smoke_light: RGB = (96, 84, 70)
        dust_brown: RGB = (90, 60, 35)

        warm_glow: RGB = (255, 190, 95)

        for idx, poly in enumerate(scene.polys):
            nx = scene.poly_nx[idx]
            ny = scene.poly_ny[idx]

            # Base sky
            sky_v = 0.02 + 0.16 * ((1.0 - ny) ** 2.0)
            sky_v += 0.02 * math.sin(2.0 * math.pi * (0.03 * t + nx * 0.7 + ny * 0.23))
            sky = hsv01_to_rgb255(215 / 360.0, 0.70, clamp01(sky_v))

            # Distance to origin for falloffs
            dx0 = nx - ox
            dy0 = ny - oy
            d0 = math.sqrt(dx0 * dx0 + dy0 * dy0)

            # Moving noise field (adds turbulent edges / rolling smoke)
            drift_x = 0.06 * math.sin(2.0 * math.pi * (0.030 * t + self.phase[idx]))
            drift_y = -0.08 * (0.50 + 0.50 * math.sin(2.0 * math.pi * (0.022 * t + self.phase[idx] * 0.7)))
            n = fbm2(nx * 3.0 + self.offx[idx] * 0.001 + drift_x,
                     ny * 3.0 + self.offy[idx] * 0.001 + drift_y,
                     self.seed + 7.1,
                     octaves=4)
            # Make it a bit punchier
            n2 = clamp01((n - 0.5) * 1.35 + 0.5)

            # Fireball core (slightly squashed)
            core_mask = _ellipse_mask(nx - ox, ny - core_y,
                                      core_r * (1.05 + 0.22 * (n2 - 0.5)),
                                      core_r * (0.92 + 0.22 * (0.5 - (n2 - 0.5))),
                                      edge=0.12)

            # Cap (mushroom top)
            cap_mask = _ellipse_mask(nx - ox, ny - cap_y,
                                     cap_rx * (0.92 + 0.24 * n2),
                                     cap_ry * (0.92 + 0.28 * (1.0 - n2)),
                                     edge=0.14)

            # Stem (vertical column between core and cap)
            xnorm = abs(nx - ox) / max(1e-6, stem_r * (0.85 + 0.40 * n2))
            xmask = clamp01(1.0 - smoothstep(1.0, 1.22, xnorm))
            y_gate_top = smoothstep(cap_y - 0.06, cap_y + 0.02, ny)
            y_gate_bot = 1.0 - smoothstep(oy + 0.02, oy + 0.14, ny)
            stem_mask = clamp01(xmask * y_gate_top * y_gate_bot)

            plume = max(core_mask, cap_mask, stem_mask)

            # Heat (dominantly core early, then diminishes)
            flick = 0.85 + 0.15 * math.sin(2.0 * math.pi * (self.freq[idx] * t + self.phase[idx]))
            heat_local = (1.15 * core_mask + 0.85 * stem_mask + 0.75 * cap_mask)
            heat = clamp01(heat_local * heat_decay * flick)

            # Smoke takeover (stronger on cap + upper plume; noise makes rolling texture)
            smoke = smoke_mul * (1.10 * cap_mask + 0.75 * stem_mask + 0.25 * (plume - core_mask))
            smoke *= (0.55 + 0.95 * n2)
            smoke = clamp01(smoke)

            # Dust near the ground, especially after the flash
            dust = smoothstep(oy - 0.02, oy + 0.20, ny) * smoothstep(0.08, 0.30, tn)
            dust *= smoothstep(0.00, 0.45, abs(nx - ox))
            dust *= (0.55 + 0.75 * n2)
            dust = clamp01(dust)

            # Warm ambient glow around the origin (and the initial flash)
            glow_r = 0.18 + 0.36 * smoothstep(0.02, 0.25, tn)
            glow = math.exp(-(d0 * d0) / (2.0 * glow_r * glow_r))
            glow *= clamp01(0.25 + 0.90 * heat_decay + 0.85 * flash)
            sky = add_rgb(sky, warm_glow, 0.35 * glow)

            # Fire color progression
            fire = mix_rgb(hot_red, hot_orange, smoothstep(0.10, 0.35, heat))
            fire = mix_rgb(fire, hot_yellow, smoothstep(0.35, 0.70, heat))
            fire = mix_rgb(fire, hot_white, smoothstep(0.78, 1.00, heat))

            # Optional tint (subtle), still stays "nuke"
            if self.tint_rgb is not None and heat > 0.01:
                fire = mix_rgb(fire, self.tint_rgb, 0.22 * heat)

            # Smoke color (cooler/darker over time, slightly warmer near ground)
            smix = clamp01(0.35 + 0.45 * (1.0 - ny) + 0.25 * n2)
            smoke_rgb = mix_rgb(smoke_dark, smoke_light, smix)
            smoke_rgb = mix_rgb(smoke_rgb, dust_brown, 0.45 * dust)

            # Compose: sky + additive fire + smoky overlay
            rgb = sky
            rgb = add_rgb(rgb, fire, 0.95 * heat)

            if smoke > 0.0:
                rgb = mix_rgb(rgb, smoke_rgb, 0.90 * smoke)

            if dust > 0.0:
                rgb = mix_rgb(rgb, smoke_rgb, 0.55 * dust)

            # Flash bloom
            if flash > 1e-4:
                bloom = flash * math.exp(-(d0 * d0) / (2.0 * (0.20 * 0.20)))
                rgb = mix_rgb(rgb, hot_white, 0.65 * clamp01(bloom))

            # Shockwave ring near ground
            if wave_t > 1e-4:
                ring = _exp_ring(d0, wave_r, wave_w) * wave_t
                ring *= smoothstep(oy - 0.08, oy + 0.12, ny)  # keep it near the "ground"
                ring = clamp01(ring)
                if ring > 0.0:
                    rgb = add_rgb(rgb, hot_yellow, 0.35 * ring)
                    rgb = mix_rgb(rgb, hot_white, 0.55 * ring)

            poly.set("fill", rgb_to_hex(rgb))
