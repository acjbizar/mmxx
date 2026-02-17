from __future__ import annotations

"""Minecraft theme.

The original generator sampled a 16×16 texture (grass block side) per polygon and
then applied a tone/specular model with flicker and sparkle. The refactor version
simplified this substantially, which changed the look.

This module restores the original sampling + shading model.
"""

import io
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
from urllib.request import Request, urlopen

from ..anim.color_math import mix_rgb, rgb_to_hex
from ..anim.easing import clamp01, smoothstep
from ..anim.pulses import facet_shimmer, whiteness_at
from ..scene import Scene
from .configs import ThemeConfig, get_theme_config

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None


DEFAULT_MINECRAFT_TEXTURE_URL = (
    "https://static.wikia.nocookie.net/minecraft_gamepedia/images/b/b2/"
    "Grass_Block_%28carried_side_texture%29_BE1.png/revision/latest?cb=20200928054656"
)


def _luma(rgb: Tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _scale_rgb(rgb: Tuple[int, int, int], f: float) -> Tuple[int, int, int]:
    r, g, b = rgb
    return (
        max(0, min(255, int(round(r * f)))),
        max(0, min(255, int(round(g * f)))),
        max(0, min(255, int(round(b * f)))),
    )


def _load_minecraft_texture_16x16(source: str) -> Tuple[List[Tuple[int, int, int]], int, int]:
    if Image is None:
        raise RuntimeError("Minecraft theme requires Pillow. Install with: py -m pip install pillow")

    src = (source or "").strip() or DEFAULT_MINECRAFT_TEXTURE_URL

    if Path(src).is_file():
        data = Path(src).read_bytes()
    else:
        req = Request(src, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=20) as resp:
            data = resp.read()

    img = Image.open(io.BytesIO(data)).convert("RGBA")

    if hasattr(Image, "Resampling"):
        resample = Image.Resampling.NEAREST
    else:  # pragma: no cover
        resample = Image.NEAREST

    img = img.resize((16, 16), resample=resample)

    w, h = img.size
    pixels: List[Tuple[int, int, int]] = []
    for y in range(h):
        for x in range(w):
            r, g, b, _a = img.getpixel((x, y))
            pixels.append((int(r), int(g), int(b)))
    return pixels, w, h


@dataclass
class PolyMC:
    # tone & shimmer
    tone_base: float
    tone_phase: float
    shimmer_freq: float
    shimmer_phase: float

    # sampled texture colors
    base: Tuple[int, int, int]
    hi: Tuple[int, int, int]
    lo: Tuple[int, int, int]
    alt: Tuple[int, int, int]

    # shading randomness
    grain: float
    flicker_freq: float
    flicker_phase: float


@dataclass
class MinecraftTheme:
    cfg: ThemeConfig
    poly: List[PolyMC]
    name: str

    @classmethod
    def create(cls, scene: Scene, args, rng: random.Random) -> "MinecraftTheme":
        cfg = get_theme_config(args.theme)
        assert cfg.kind == "minecraft"

        tex, tw, th = _load_minecraft_texture_16x16(getattr(args, "minecraft_texture", ""))

        poly: List[PolyMC] = []
        for idx in range(scene.n_polys):
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

            # ---- sample texture using centroid-in-glyph normalized coords ----
            nx = clamp01(scene.glyph_nx[idx])
            ny = clamp01(scene.glyph_ny[idx])

            u_px = int(round(nx * (tw - 1)))
            v_px = int(round(ny * (th - 1)))

            if rng.random() < 0.35:
                u_px = max(0, min(tw - 1, u_px + rng.choice([-1, 0, 1])))
                v_px = max(0, min(th - 1, v_px + rng.choice([-1, 0, 1])))

            coords = [
                (u_px, v_px),
                (max(0, u_px - 1), v_px),
                (min(tw - 1, u_px + 1), v_px),
                (u_px, max(0, v_px - 1)),
                (u_px, min(th - 1, v_px + 1)),
            ]

            samples = [(uv, tex[uv[1] * tw + uv[0]]) for uv in coords]
            base = samples[0][1]
            hi = max(samples, key=lambda it: _luma(it[1]))[1]
            lo = min(samples, key=lambda it: _luma(it[1]))[1]
            alt = rng.choice(samples[1:])[1] if len(samples) > 1 else base

            dirtish = 1.0 if v_px >= int(th * 0.30) else 0.0
            grain = rng.uniform(0.84, 1.22) if dirtish else rng.uniform(0.90, 1.14)

            flicker_freq = rng.uniform(0.05, 0.13)
            flicker_phase = rng.uniform(0.0, 1.0)

            poly.append(
                PolyMC(
                    tone_base=tone,
                    tone_phase=tone_phase,
                    shimmer_freq=shimmer_freq,
                    shimmer_phase=shimmer_phase,
                    base=base,
                    hi=hi,
                    lo=lo,
                    alt=alt,
                    grain=grain,
                    flicker_freq=flicker_freq,
                    flicker_phase=flicker_phase,
                )
            )

        return cls(cfg=cfg, poly=poly, name=args.theme)

    def apply_frame(self, scene: Scene, t: float) -> None:
        cfg = self.cfg

        global_amb = cfg.amb_base + cfg.amb_amp * (0.5 + 0.5 * math.sin(2.0 * math.pi * (cfg.amb_freq * t)))

        for idx, poly_el in enumerate(scene.polys):
            pm = self.poly[idx]

            wobble = 0.050 * math.sin(2.0 * math.pi * (0.06 * t + pm.tone_phase))
            tone = clamp01(pm.tone_base + wobble)

            a = whiteness_at(t, scene.pulses_per_poly[idx])
            shim = facet_shimmer(t, pm.shimmer_freq, pm.shimmer_phase)
            gl = clamp01(cfg.gl_pulse_w * a + cfg.gl_shim_w * shim)

            spec = smoothstep(cfg.spec_edge0, 1.00, gl)
            spec_amt = clamp01(spec * cfg.spec_scale)

            # torch-like flicker
            f = 0.5 + 0.5 * math.sin(2.0 * math.pi * (pm.flicker_freq * t + pm.flicker_phase))
            f = f ** 1.8

            shade = (0.62 + 0.78 * tone) * pm.grain
            shade *= (0.90 + 0.22 * f)
            shade *= (0.92 + 0.28 * smoothstep(0.20, 0.95, gl))

            body = _scale_rgb(pm.base, shade)

            shadow_amt = 0.20 + 0.35 * smoothstep(0.00, 0.55, 1.0 - tone)
            body = mix_rgb(body, _scale_rgb(pm.lo, shade * 0.92), shadow_amt * 0.35)
            body = mix_rgb(body, _scale_rgb(pm.hi, shade * 1.02), global_amb)

            sparkle = smoothstep(0.35, 1.00, gl)
            sparkle *= (0.15 + 0.55 * f)
            body = mix_rgb(body, _scale_rgb(pm.alt, shade * 1.04), sparkle * 0.55)

            body = mix_rgb(body, _scale_rgb(pm.hi, shade * 1.22), spec_amt * (0.35 + 0.35 * f))
            if spec_amt > 0.85:
                body = mix_rgb(body, (255, 255, 255), (spec_amt - 0.85) * 0.10)

            poly_el.set("fill", rgb_to_hex(body))
