from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Tuple

from ..scene import Scene
from ..anim.noise import fbm2
from ..anim.easing import clamp01, smoothstep
from ..anim.color_math import mix_rgb, rgb_to_hex

RGB = Tuple[int, int, int]


@dataclass
class CloudsTheme:
    """Soft drifting clouds from left to right."""

    seed: int
    speed: float = 0.09   # drift left->right
    scale: float = 3.2    # cloud size (higher = smaller)

    sky_top: RGB = (145, 190, 245)
    sky_bottom: RGB = (215, 238, 255)
    cloud_shadow: RGB = (220, 226, 236)
    cloud_white: RGB = (255, 255, 255)

    name: str = "clouds"

    @classmethod
    def create(cls, *, scene: Scene, args: Any, rng: random.Random) -> "CloudsTheme":
        return cls(seed=rng.randrange(1 << 30))

    def apply_frame(self, scene: Scene, t: float) -> None:
        # Gentle bobbing so the field doesn't look perfectly "slid".
        bob = 0.06 * math.sin(2.0 * math.pi * (0.035 * t))

        for i, poly in enumerate(scene.polys):
            nx = scene.poly_nx[i]
            ny = scene.poly_ny[i]

            sky = mix_rgb(self.sky_top, self.sky_bottom, clamp01(ny))

            # fBm field with a little warp; drift in +x
            x = nx * self.scale + t * self.speed
            y = ny * self.scale * 0.92 + bob

            wx = fbm2(x * 0.65 + 11.3, y * 0.65 + 7.7, self.seed + 17, octaves=3)
            wy = fbm2(x * 0.65 + 31.1, y * 0.65 + 19.9, self.seed + 29, octaves=3)
            xw = x + 0.85 * (wx - 0.5)
            yw = y + 0.85 * (wy - 0.5)

            n1 = fbm2(xw, yw, self.seed, octaves=5)
            n2 = fbm2(xw * 1.6 + 23.0, yw * 1.6 + 41.0, self.seed + 101, octaves=4)
            field = 0.62 * n1 + 0.38 * n2

            # bias: slightly more clouds toward the top
            field = clamp01(field + 0.18 * (0.55 - ny))

            cloud = smoothstep(0.58, 0.78, field)
            core = smoothstep(0.72, 0.92, field)

            shade = clamp01(0.25 + 0.55 * (1.0 - ny))
            cloud_col = mix_rgb(self.cloud_shadow, self.cloud_white, clamp01(0.35 + 0.65 * core))
            cloud_col = mix_rgb(cloud_col, self.cloud_shadow, (1.0 - core) * 0.35 * shade)

            rgb = mix_rgb(sky, cloud_col, cloud)
            poly.set("fill", rgb_to_hex(rgb))
