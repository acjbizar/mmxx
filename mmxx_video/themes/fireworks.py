from __future__ import annotations

"""Fireworks theme ("exploding fire arrows").

This file restores the older, better-looking fireworks style from the last
known-good single-file generator:

- Rocket phase: bright hot head + a glowing trail along a segment
- Burst phase: radial *arrow-like* streaks (body + hot head) with flicker
- Additive blending on top of a dark night-sky gradient

The refactor briefly replaced this with a simpler "ring/spokes" burst, which
is cheaper but visually much flatter.
"""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

from ..anim.color_math import hsv01_to_rgb255, mix_rgb, mix_to_white, rgb_to_hex
from ..anim.easing import clamp01, cosine_ease, smoothstep
from ..scene import Scene
from .configs import get_theme_config


@dataclass(frozen=True)
class FireRay:
    dx: float
    dy: float
    speed: float
    width: float
    hue_off: float
    phase: float


@dataclass(frozen=True)
class Firework:
    t0: float
    x0: float
    y0: float
    x1: float
    y1: float
    t_up: float
    t_burst: float
    hue: float
    rays: Tuple[FireRay, ...]
    sparkle_phase: float


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _dist2_point_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> Tuple[float, float]:
    """Return (squared distance, u) where u is closest-point parameter on AB in [0,1]."""
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    d = abx * abx + aby * aby
    if d <= 1e-12:
        return (apx * apx + apy * apy, 0.0)
    u = (apx * abx + apy * aby) / d
    u = clamp01(u)
    cx = ax + u * abx
    cy = ay + u * aby
    dx = px - cx
    dy = py - cy
    return (dx * dx + dy * dy, u)


def _firework_additive_rgb(nx: float, ny: float, t: float, fw: Firework) -> Tuple[float, float, float]:
    """Additive RGB (float) for a single firework at time t, at normalized position nx/ny."""
    dt0 = t - fw.t0
    r_acc = g_acc = b_acc = 0.0

    # ---------------- rocket phase ----------------
    if 0.0 <= dt0 <= fw.t_up:
        u = dt0 / max(1e-6, fw.t_up)
        u2 = 0.25 * u + 0.75 * cosine_ease(u)
        hx = _lerp(fw.x0, fw.x1, u2)
        hy = _lerp(fw.y0, fw.y1, u2)

        dx = fw.x1 - fw.x0
        dy = fw.y1 - fw.y0
        L = math.hypot(dx, dy)
        if L <= 1e-9:
            L = 1.0
        ux = dx / L
        uy = dy / L

        trail_len = 0.26
        tx = hx - ux * trail_len
        ty = hy - uy * trail_len

        d2, uu = _dist2_point_to_segment(nx, ny, tx, ty, hx, hy)
        sigma = 0.010
        trail = math.exp(-d2 / (2.0 * sigma * sigma)) * (0.15 + 0.85 * uu)

        hd2 = (nx - hx) ** 2 + (ny - hy) ** 2
        head = math.exp(-hd2 / (2.0 * (sigma * 1.75) ** 2))

        inten = clamp01(0.85 * trail + 1.30 * head)
        inten *= (0.65 + 0.35 * math.sin(2.0 * math.pi * (6.0 * dt0 + fw.sparkle_phase)))

        fire_rgb = hsv01_to_rgb255((fw.hue + 0.02) % 1.0, 0.95, 1.0)
        mixw = clamp01(0.55 * head + 0.25 * trail)
        hot = mix_rgb(fire_rgb, (255, 255, 255), mixw)

        r_acc += inten * hot[0]
        g_acc += inten * hot[1]
        b_acc += inten * hot[2]

    # ---------------- burst phase ----------------
    te = fw.t0 + fw.t_up
    dt = t - te
    if 0.0 <= dt <= fw.t_burst:
        fade = 1.0 - smoothstep(0.0, fw.t_burst, dt)
        cx, cy = fw.x1, fw.y1
        rx = nx - cx
        ry = ny - cy

        # soft colored core
        core_r = 0.065 + 0.070 * (dt / max(1e-6, fw.t_burst))
        glow = math.exp(-(rx * rx + ry * ry) / (2.0 * core_r * core_r))
        core = fade * 0.42 * glow
        core_rgb = hsv01_to_rgb255(fw.hue, 0.35, 1.0)
        r_acc += core * core_rgb[0]
        g_acc += core * core_rgb[1]
        b_acc += core * core_rgb[2]

        for ray in fw.rays:
            length = ray.speed * dt
            if length <= 1e-6:
                continue

            # projection along ray direction
            proj = rx * ray.dx + ry * ray.dy
            if proj <= 0.0:
                continue

            # perpendicular distance to ray axis
            px = rx - proj * ray.dx
            py = ry - proj * ray.dy
            perp2 = px * px + py * py

            sigma = ray.width
            if proj <= length:
                body = math.exp(-perp2 / (2.0 * sigma * sigma))
                taper = 0.25 + 0.75 * (1.0 - (proj / max(1e-6, length)))
                body *= taper
            else:
                body = 0.0

            d_tip = proj - length
            head = (
                math.exp(-perp2 / (2.0 * (sigma * 1.9) ** 2))
                * math.exp(-(d_tip * d_tip) / (2.0 * (sigma * 2.8) ** 2))
            )

            flick = 0.70 + 0.30 * math.sin(2.0 * math.pi * (7.0 * dt + ray.phase + 0.35 * proj))
            inten = fade * (0.70 * body + 1.35 * head) * flick
            if inten <= 1e-6:
                continue

            h = (fw.hue + ray.hue_off) % 1.0
            c = hsv01_to_rgb255(h, 0.95, 1.0)
            hot = mix_rgb(c, (255, 255, 255), clamp01(0.70 * head))

            r_acc += inten * hot[0]
            g_acc += inten * hot[1]
            b_acc += inten * hot[2]

    return (r_acc, g_acc, b_acc)


@dataclass
class FireworksTheme:
    name = "fireworks"
    fireworks: List[Firework]

    @classmethod
    def create(cls, *, scene: Scene, args, rng: random.Random):
        base_h = scene.override_hsv[0] if scene.override_hsv is not None else None
        dur = float(scene.duration)

        # Match the old single-file behaviour: a handful of staggered launches.
        n_fw = max(6, int(round(dur / 1.35)))
        fireworks: List[Firework] = []

        for _ in range(n_fw):
            t0 = rng.uniform(0.0, dur * 0.86)
            x0 = rng.uniform(0.10, 0.90)
            y0 = 1.08
            x1 = clamp01(x0 + rng.uniform(-0.22, 0.22))
            y1 = rng.uniform(0.18, 0.55)
            t_up = rng.uniform(0.85, 1.70)
            t_burst = rng.uniform(1.55, 2.85)

            if base_h is None:
                if rng.random() < 0.70:
                    hue = rng.choice([10 / 360.0, 20 / 360.0, 35 / 360.0, 45 / 360.0, 60 / 360.0, 330 / 360.0])
                else:
                    hue = rng.random()
            else:
                hue = (base_h + rng.uniform(-0.10, 0.10)) % 1.0

            rays: List[FireRay] = []
            n_rays = rng.randint(14, 24)
            for k in range(n_rays):
                ang = (2.0 * math.pi) * (k / float(n_rays)) + rng.uniform(-0.16, 0.16)
                dx = math.cos(ang)
                dy = math.sin(ang)
                rays.append(
                    FireRay(
                        dx=dx,
                        dy=dy,
                        speed=rng.uniform(0.33, 0.72),
                        width=rng.uniform(0.007, 0.014),
                        hue_off=rng.uniform(-0.09, 0.09),
                        phase=rng.uniform(0.0, 1.0),
                    )
                )

            fireworks.append(
                Firework(
                    t0=t0,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    t_up=t_up,
                    t_burst=t_burst,
                    hue=hue,
                    rays=tuple(rays),
                    sparkle_phase=rng.uniform(0.0, 1.0),
                )
            )

        return cls(fireworks=fireworks)

    def apply_frame(self, scene: Scene, t: float) -> None:
        cfg = get_theme_config("fireworks")
        amb = cfg.amb_base + cfg.amb_amp * (0.5 + 0.5 * math.sin(2.0 * math.pi * (cfg.amb_freq * t)))

        for idx, poly in enumerate(scene.polys):
            nx = scene.poly_nx[idx]
            ny = scene.poly_ny[idx]

            # Dark night-sky background with a subtle purple/blue drift
            bg_h = (250 / 360.0) + 0.015 * math.sin(2.0 * math.pi * (0.03 * t + nx * 1.3))
            bg_s = 0.50
            bg_v = 0.03 + 0.08 * (1.0 - ny) ** 1.35
            bg_v += 0.01 * math.sin(2.0 * math.pi * (0.06 * t + nx * 0.9 + ny * 0.7))
            bg_v = clamp01(bg_v)
            bg = hsv01_to_rgb255(bg_h % 1.0, bg_s, bg_v)

            r_acc = float(bg[0])
            g_acc = float(bg[1])
            b_acc = float(bg[2])

            for fw in self.fireworks:
                fr, fg, fb = _firework_additive_rgb(nx, ny, t, fw)
                r_acc += fr
                g_acc += fg
                b_acc += fb

            # If the additive sum gets very hot, bias towards white (spark burn)
            hot = max(r_acc, g_acc, b_acc) / 255.0
            hot_gate = smoothstep(0.85, 1.30, hot)
            if hot_gate > 0.0:
                r_acc = _lerp(r_acc, 255.0, 0.40 * hot_gate)
                g_acc = _lerp(g_acc, 255.0, 0.40 * hot_gate)
                b_acc = _lerp(b_acc, 255.0, 0.40 * hot_gate)

            rgb = (
                max(0, min(255, int(round(r_acc)))),
                max(0, min(255, int(round(g_acc)))),
                max(0, min(255, int(round(b_acc)))),
            )

            # Gentle ambient lift (keep it low)
            rgb = mix_to_white(rgb, amb * 0.04)
            # Scene polygons are lxml elements; set SVG attribute.
            poly.set("fill", rgb_to_hex(rgb))

