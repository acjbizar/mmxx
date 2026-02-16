from __future__ import annotations
import math, random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from ..scene import Scene
from ..anim.easing import clamp01, smoothstep, cosine_ease
from ..anim.pulses import facet_shimmer
from ..anim.color_math import hsv01_to_rgb255, mix_rgb, mix_to_white, rgb_to_hex

_HEART_BEAT_AMP = 0.11

def beat_wave_smooth(t: float) -> float:
    cycle = 2.25
    p = (t / cycle) % 1.0

    def bump(phase: float, center: float, width: float) -> float:
        d = abs((phase - center + 0.5) % 1.0 - 0.5)
        if d >= width:
            return 0.0
        x = d / width
        return 0.5 * (1.0 + math.cos(math.pi * x))

    b1 = bump(p, 0.16, 0.11)
    b2 = bump(p, 0.33, 0.15)
    tail = 0.20 * (1.0 - smoothstep(0.34, 0.98, p))

    raw = 0.04 + 0.50 * b1 + 1.00 * b2 + tail
    raw = clamp01(raw)

    eased = cosine_ease(raw)
    pulse = 0.55 * raw + 0.45 * eased
    return clamp01(pulse)

def heart_icon_val(x: float, y: float) -> float:
    x = abs(x)
    x *= 1.12
    y *= 1.02
    y += 0.10
    a = x * x + y * y - 1.0
    return (a * a * a) - (x * x) * (y * y * y)

def compute_heart_fit(margin: float = 0.004, headroom: float = (1.0 + _HEART_BEAT_AMP)) -> Dict[str, float]:
    scan_min = -1.85
    scan_max = 1.85
    steps = 460

    minx = 1e9
    maxx = -1e9
    miny = 1e9
    maxy = -1e9

    for j in range(steps):
        y = scan_min + (scan_max - scan_min) * (j / (steps - 1))
        for i in range(steps):
            x = scan_min + (scan_max - scan_min) * (i / (steps - 1))
            if heart_icon_val(x, y) <= 0.0:
                minx = min(minx, x)
                maxx = max(maxx, x)
                miny = min(miny, y)
                maxy = max(maxy, y)

    if not (minx < maxx and miny < maxy):
        return {"sx": 1.20, "sy": 1.20, "cx": 0.0, "cy": 0.0, "margin": margin, "headroom": headroom}

    cx = 0.5 * (minx + maxx)
    cy = 0.5 * (miny + maxy)
    w = (maxx - minx)
    h = (maxy - miny)

    target = 2.0 * (1.0 - margin) / max(1e-6, headroom)
    sx = target / max(1e-6, w)
    sy = target / max(1e-6, h)

    return {"sx": float(sx), "sy": float(sy), "cx": float(cx), "cy": float(cy), "margin": float(margin), "headroom": float(headroom)}

def heart_mask_icon_single(nx: float, ny: float, pulse: float, fit: Dict[str, float]) -> Tuple[float, float]:
    xs = nx * 2.0 - 1.0
    ys = (1.0 - ny) * 2.0 - 1.0

    beat_scale = 1.0 + _HEART_BEAT_AMP * pulse
    sx = fit["sx"] * beat_scale
    sy = fit["sy"] * beat_scale

    x = (xs / sx) + fit["cx"]
    y = (ys / sy) + fit["cy"]

    val = heart_icon_val(x, y)

    edge = 0.026
    mask = 1.0 - smoothstep(-edge, edge, val)
    glow = math.exp(-abs(val) * 9.5)
    return (clamp01(mask), clamp01(glow))

def heart_mask_icon_aa(nx: float, ny: float, pulse: float, fit: Dict[str, float]) -> Tuple[float, float]:
    eps = 0.0026
    samples = [
        (0.0, 0.0),
        (+eps, 0.0),
        (-eps, 0.0),
        (0.0, +eps),
        (0.0, -eps),
        (+eps * 0.7, +eps * 0.7),
        (-eps * 0.7, +eps * 0.7),
        (+eps * 0.7, -eps * 0.7),
        (-eps * 0.7, -eps * 0.7),
    ]

    m_sum = 0.0
    g_sum = 0.0
    for dx, dy in samples:
        m, g = heart_mask_icon_single(clamp01(nx + dx), clamp01(ny + dy), pulse, fit)
        m_sum += m
        g_sum += g

    inv = 1.0 / float(len(samples))
    return (clamp01(m_sum * inv), clamp01(g_sum * inv))

@dataclass
class HeartTheme:
    name: str = "heart"
    fit: Dict[str, float] = None  # type: ignore[assignment]
    hj: List[float] = None  # type: ignore[assignment]
    sj: List[float] = None  # type: ignore[assignment]
    tw_f: List[float] = None  # type: ignore[assignment]
    tw_p: List[float] = None  # type: ignore[assignment]
    fire_h: List[float] = None  # type: ignore[assignment]

    @classmethod
    def create(cls, *, scene: Scene, args: Any, rng: random.Random) -> "HeartTheme":
        inst = cls()
        inst.fit = compute_heart_fit(margin=0.004, headroom=(1.0 + _HEART_BEAT_AMP))
        fire_hues = [350/360.0, 0/360.0, 10/360.0, 330/360.0, 315/360.0, 45/360.0]
        inst.hj = []
        inst.sj = []
        inst.tw_f = []
        inst.tw_p = []
        inst.fire_h = []
        for _ in scene.polys:
            inst.hj.append(rng.uniform(-0.030, 0.030))
            inst.sj.append(rng.uniform(0.92, 1.22))
            inst.tw_f.append(rng.uniform(0.22, 0.95))
            inst.tw_p.append(rng.uniform(0.0, 1.0))
            inst.fire_h.append((rng.choice(fire_hues) + rng.uniform(-0.06, 0.06)) % 1.0)
        return inst

    def apply_frame(self, scene: Scene, t: float) -> None:
        pulse = beat_wave_smooth(t)
        base_h = (scene.override_hsv[0] if scene.override_hsv is not None else (335 / 360.0))
        amb = 0.06 + 0.025 * (0.5 + 0.5 * math.sin(2.0 * math.pi * (0.025 * t)))

        for idx, poly in enumerate(scene.polys):
            nx = scene.poly_nx[idx]
            ny = scene.poly_ny[idx]

            mask, glow = heart_mask_icon_aa(nx, ny, pulse, self.fit)

            bg_h = (300/360.0) + 0.010 * math.sin(2.0 * math.pi * (0.03 * t + nx * 0.7))
            bg_s = 0.35
            vign = math.sqrt((nx - 0.5) ** 2 + (ny - 0.5) ** 2)
            bg_v = 0.05 + 0.07 * (1.0 - smoothstep(0.15, 0.95, vign))
            bg_v += 0.02 * math.sin(2.0 * math.pi * (0.05 * t + nx * 1.1 + ny * 0.6))
            bg_v = clamp01(bg_v)
            bg_rgb = hsv01_to_rgb255(bg_h, bg_s, bg_v)

            h = (base_h
                 + 0.090 * (nx - 0.5)
                 + 0.040 * math.sin(2.0 * math.pi * (0.09 * t + ny * 0.9))
                 + self.hj[idx]) % 1.0

            sat_breathe = 0.85 + 0.30 * (0.5 + 0.5 * math.sin(2.0 * math.pi * (0.12 * t + self.tw_p[idx])))
            s = (0.72 + 0.30 * mask + 0.22 * glow) * self.sj[idx] * sat_breathe
            s = clamp01(s)

            v = 0.18 + 0.46 * mask
            v += 0.34 * mask * (0.18 + 0.82 * pulse)
            v += 0.22 * glow * (0.30 + 0.70 * pulse)
            v = clamp01(v)

            heart_rgb = hsv01_to_rgb255(h, s, v)
            rgb = mix_rgb(bg_rgb, heart_rgb, mask)

            edge = clamp01(glow * (0.22 + 0.40 * pulse))
            rgb = mix_rgb(rgb, (255, 255, 255), 0.40 * edge)

            fire_gate = smoothstep(0.35, 0.92, edge)
            if fire_gate > 0.0:
                fh = (self.fire_h[idx] + 0.02 * math.sin(2.0 * math.pi * (0.06 * t + nx))) % 1.0
                fire_rgb = hsv01_to_rgb255(fh, clamp01(0.65 + 0.35 * s), 1.0)
                rgb = mix_rgb(rgb, fire_rgb, 0.18 * fire_gate)

            tw = facet_shimmer(t, self.tw_f[idx], self.tw_p[idx])
            tw = (mask ** 1.55) * (tw ** 1.30)
            sparkle = clamp01(tw * (0.06 + 0.22 * pulse) + (mask ** 2.2) * (0.02 + 0.06 * pulse))
            if sparkle > 0.0:
                rgb = mix_rgb(rgb, (255, 255, 255), sparkle)
                fh2 = (self.fire_h[idx] + 0.015 * math.sin(2.0 * math.pi * (0.08 * t + ny))) % 1.0
                flare = hsv01_to_rgb255(fh2, 0.85, 1.0)
                rgb = mix_rgb(rgb, flare, 0.10 * sparkle)

            rgb = mix_to_white(rgb, amb * 0.10)
            poly.set("fill", rgb_to_hex(rgb))
