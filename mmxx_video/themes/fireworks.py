from __future__ import annotations
import math, random
from dataclasses import dataclass
from typing import Any, List
from ..scene import Scene
from ..anim.easing import clamp01, smoothstep, cosine_ease
from ..anim.color_math import hsv01_to_rgb255, mix_rgb, add_rgb, mix_to_white, rgb_to_hex
from .configs import get_theme_config

@dataclass(frozen=True)
class Firework:
    x: float
    yb: float
    t_launch: float
    t_burst: float
    vel: float
    ring_w: float
    decay: float
    hue: float

@dataclass
class FireworksTheme:
    name: str = "fireworks"
    fireworks: List[Firework] = None  # type: ignore[assignment]

    @classmethod
    def create(cls, *, scene: Scene, args: Any, rng: random.Random) -> "FireworksTheme":
        inst = cls()
        dur = float(args.duration)
        n_fw = max(6, min(14, int(round(dur * 1.0))))
        fws: List[Firework] = []
        for _ in range(n_fw):
            t_burst = rng.uniform(1.0, max(1.2, dur - 0.8))
            launch_lead = rng.uniform(0.55, 1.40)
            t_launch = max(0.0, t_burst - launch_lead)
            fws.append(Firework(
                x=rng.uniform(0.08, 0.92),
                yb=rng.uniform(0.18, 0.55),
                t_launch=t_launch,
                t_burst=t_burst,
                vel=rng.uniform(0.22, 0.55),
                ring_w=rng.uniform(0.018, 0.050),
                decay=rng.uniform(1.00, 2.20),
                hue=rng.random(),
            ))
        inst.fireworks = fws
        return inst

    def apply_frame(self, scene: Scene, t: float) -> None:
        cfg = get_theme_config("fireworks")
        amb = cfg.amb_base + cfg.amb_amp * (0.5 + 0.5 * math.sin(2.0 * math.pi * (cfg.amb_freq * t)))

        for idx, poly in enumerate(scene.polys):
            nx = scene.poly_nx[idx]
            ny = scene.poly_ny[idx]

            sky_rgb = hsv01_to_rgb255(215/360.0, 0.55, clamp01(0.04 + 0.10 * (1.0 - ny)))
            rgb = sky_rgb

            for fw in self.fireworks:
                if fw.t_launch <= t < fw.t_burst:
                    u = (t - fw.t_launch) / max(1e-6, (fw.t_burst - fw.t_launch))
                    u = clamp01(u)
                    uu = cosine_ease(u)
                    y = 1.05 + (fw.yb - 1.05) * uu
                    dx = abs(nx - fw.x)
                    dy = ny - y
                    if dy >= 0.0 and dy <= 0.30:
                        trail = math.exp(-(dx * dx) / (2.0 * 0.020 * 0.020)) * math.exp(-(dy * dy) / (2.0 * 0.18 * 0.18))
                        trail_rgb = hsv01_to_rgb255(35/360.0, 0.85, 0.80)
                        rgb = add_rgb(rgb, trail_rgb, 0.85 * clamp01(trail))

                if t >= fw.t_burst:
                    dt = t - fw.t_burst
                    if dt <= 4.0:
                        d = math.sqrt((nx - fw.x)**2 + (ny - fw.yb)**2)
                        r = fw.vel * dt
                        ring = math.exp(-((d - r) * (d - r)) / (2.0 * fw.ring_w * fw.ring_w)) * math.exp(-dt / max(1e-6, fw.decay))
                        inten = clamp01(ring)
                        if inten > 1e-5:
                            burst_rgb = hsv01_to_rgb255(fw.hue, 0.90, clamp01(0.30 + 0.70 * inten))
                            burst_rgb = mix_rgb(burst_rgb, (255, 255, 255), 0.20 * inten)
                            rgb = add_rgb(rgb, burst_rgb, 0.95 * inten)

            rgb = mix_to_white(rgb, amb * 0.05)
            poly.set("fill", rgb_to_hex(rgb))
