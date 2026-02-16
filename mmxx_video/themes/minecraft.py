from __future__ import annotations
import io, math, random
from dataclasses import dataclass
from typing import Any, List, Tuple
from urllib.request import Request, urlopen
from pathlib import Path

from ..scene import Scene
from ..anim.easing import clamp01, smoothstep
from ..anim.pulses import whiteness_at, facet_shimmer
from ..anim.color_math import mix_rgb, hsv01_to_rgb255, mix_to_white, rgb_to_hex
from .configs import get_theme_config

DEFAULT_MINECRAFT_TEXTURE_URL = (
    "https://static.wikia.nocookie.net/minecraft_gamepedia/images/b/b2/"
    "Grass_Block_%28carried_side_texture%29_BE1.png/revision/latest?cb=20200928054656"
)

RGB = Tuple[int, int, int]

def load_minecraft_texture_16x16(source: str) -> Tuple[List[RGB], int, int]:
    try:
        from PIL import Image  # type: ignore
    except Exception:
        raise RuntimeError("Minecraft theme requires Pillow. Install with: py -m pip install pillow")

    src = (source or "").strip() or DEFAULT_MINECRAFT_TEXTURE_URL

    if Path(src).is_file():
        data = Path(src).read_bytes()
    else:
        req = Request(src, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=20) as resp:
            data = resp.read()

    img = Image.open(io.BytesIO(data)).convert("RGBA")
    resample = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST
    img = img.resize((16, 16), resample=resample)

    w, h = img.size
    pixels: List[RGB] = []
    for y in range(h):
        for x in range(w):
            r, g, b, _a = img.getpixel((x, y))
            pixels.append((int(r), int(g), int(b)))
    return pixels, w, h

@dataclass
class MinecraftTheme:
    name: str = "minecraft"
    mc_pixels: List[RGB] = None  # type: ignore[assignment]
    mc_w: int = 0
    mc_h: int = 0
    mc_u: List[float] = None  # type: ignore[assignment]
    mc_v: List[float] = None  # type: ignore[assignment]
    mc_freq: List[float] = None  # type: ignore[assignment]
    mc_phase: List[float] = None  # type: ignore[assignment]

    @classmethod
    def create(cls, *, scene: Scene, args: Any, rng: random.Random) -> "MinecraftTheme":
        _cfg = get_theme_config("minecraft")
        mc_pixels, mc_w, mc_h = load_minecraft_texture_16x16(args.minecraft_texture)
        inst = cls()
        inst.mc_pixels = mc_pixels
        inst.mc_w = mc_w
        inst.mc_h = mc_h
        inst.mc_u = []
        inst.mc_v = []
        inst.mc_freq = []
        inst.mc_phase = []
        for _ in scene.polys:
            inst.mc_u.append(rng.random())
            inst.mc_v.append(rng.random())
            inst.mc_freq.append(rng.uniform(0.10, 0.45))
            inst.mc_phase.append(rng.uniform(0.0, 1.0))
        return inst

    def apply_frame(self, scene: Scene, t: float) -> None:
        cfg = get_theme_config("minecraft")
        amb = cfg.amb_base + cfg.amb_amp * (0.5 + 0.5 * math.sin(2.0 * math.pi * (cfg.amb_freq * t)))

        for idx, poly_el in enumerate(scene.polys):
            u = scene.glyph_nx[idx]
            v = scene.glyph_ny[idx]
            px = int(round(clamp01(u) * (self.mc_w - 1)))
            py = int(round(clamp01(v) * (self.mc_h - 1)))
            base = self.mc_pixels[py * self.mc_w + px]

            shim = facet_shimmer(t, self.mc_freq[idx], self.mc_phase[idx])
            pulse = whiteness_at(t, scene.pulses_per_poly[idx]) if scene.pulses_per_poly[idx] else 0.0

            wob = 0.92 + 0.12 * (0.5 + 0.5 * math.sin(2.0 * math.pi * (0.07 * t + self.mc_phase[idx])))
            rgb = (
                max(0, min(255, int(round(base[0] * wob)))),
                max(0, min(255, int(round(base[1] * wob)))),
                max(0, min(255, int(round(base[2] * wob)))),
            )

            glint = max(cfg.gl_shim_w * shim, cfg.gl_pulse_w * pulse)
            spec = smoothstep(cfg.spec_edge0, 1.0, glint) * cfg.spec_scale

            if spec > 0.0:
                sun = hsv01_to_rgb255(45/360.0, 0.18, 1.0)
                rgb = mix_rgb(rgb, sun, 0.12 * spec)
                rgb = mix_rgb(rgb, (255, 255, 255), 0.28 * spec)

            rgb = mix_to_white(rgb, amb * 0.06)
            poly_el.set("fill", rgb_to_hex(rgb))
