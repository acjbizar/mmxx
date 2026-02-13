#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/generate-video.py

Generate a video from either:
  - --char  -> src/character-{char}.svg
  - --chars -> combines 4x src/character-{c}.svg into a single SVG (2x2 layout: [0 1; 2 3])

Logo layout (for --chars):
- Always 2 by 2: [0 1; 2 3]
- --gap 0 (default): no extra spacing
- --gap 1: spacing AND outer padding = 1/8th of a character cell size

Also in --chars mode:
- Removes per-glyph full-canvas WHITE background rects (prevents 4 white squares)

Themes:
- classic (default): pulse-to-white using polygon base colors (or --color)
- diamond: high-contrast greys + strong specular + dispersion "fire" in highlights
- silver / gold / bronze: metallic hue drift + colored sheen + glints
- ruby / jade / sapphire / emerald: gem saturation + absorption-like depth + glints
- rainbow: per-facet hue + iridescent drift + colorful highlights
- minecraft: samples the real Grass Block (carried side texture) pixels (16×16) per polygon centroid,
            animated via facet lighting + neighbour "pixel sparkle"
- deidee: cycles polygon colors ONLY between samples of:
            fill(random(0, .5), random(.5, 1), random(0, .75), .5)
          i.e. RGB in that distribution, alpha=0.5. No shading, no whitening, no extra colors.

Output:
  dist/videos/character-{char}.{ext}
  dist/videos/logo-{chars}.{ext}

If output filename already exists:
  - appends a timestamp: ...-YYYYMMDD-HHMMSS.ext
"""

from __future__ import annotations

import argparse
import colorsys
import copy
import io
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import Request, urlopen

from lxml import etree  # py -m pip install lxml

try:
    import cairosvg  # py -m pip install cairosvg
except Exception:
    cairosvg = None

try:
    from PIL import Image, ImageColor  # py -m pip install pillow
except Exception:
    Image = None
    ImageColor = None


SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


DEFAULT_MINECRAFT_TEXTURE_URL = (
    "https://static.wikia.nocookie.net/minecraft_gamepedia/images/b/b2/"
    "Grass_Block_%28carried_side_texture%29_BE1.png/revision/latest?cb=20200928054656"
)


# -------------------- Helpers: SVG parsing / color ----------------------------

def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _parse_viewbox(root: etree._Element) -> Tuple[float, float, float, float]:
    vb = root.get("viewBox") or root.get("viewbox") or "0 0 240 240"
    nums = [float(x) for x in NUM_RE.findall(vb)]
    if len(nums) == 4:
        return nums[0], nums[1], nums[2], nums[3]
    return 0.0, 0.0, 240.0, 240.0


def _style_get(style: str, key: str) -> Optional[str]:
    if not style:
        return None
    parts = [p.strip() for p in style.split(";") if p.strip()]
    for p in parts:
        if ":" not in p:
            continue
        k, v = p.split(":", 1)
        if k.strip().lower() == key.lower():
            return v.strip()
    return None


def _style_set(style: str, key: str, value: str) -> str:
    """
    Set/replace a key in an inline style string (very small helper).
    """
    key_l = key.strip().lower()
    items: List[Tuple[str, str]] = []
    if style:
        parts = [p.strip() for p in style.split(";") if p.strip()]
        for p in parts:
            if ":" not in p:
                continue
            k, v = p.split(":", 1)
            k = k.strip()
            v = v.strip()
            if k.lower() == key_l:
                continue
            items.append((k, v))
    items.append((key.strip(), value.strip()))
    return "; ".join(f"{k}: {v}" for k, v in items)


def _resolve_fill(el: etree._Element) -> Optional[str]:
    cur = el
    while cur is not None and isinstance(cur.tag, str):
        fill = (cur.get("fill") or "").strip()
        if fill:
            return fill
        st = (cur.get("style") or "").strip()
        if st:
            v = _style_get(st, "fill")
            if v:
                return v
        cur = cur.getparent()
    return None


def _parse_css_color_to_rgb(color: str) -> Tuple[int, int, int]:
    if not color:
        raise ValueError("Empty color")

    c = color.strip()
    if c.lower() in {"none", "transparent"}:
        return (0, 0, 0)

    if c.startswith("#"):
        hx = c[1:].strip()
        if len(hx) == 3:
            r = int(hx[0] * 2, 16)
            g = int(hx[1] * 2, 16)
            b = int(hx[2] * 2, 16)
            return (r, g, b)
        if len(hx) == 6:
            r = int(hx[0:2], 16)
            g = int(hx[2:4], 16)
            b = int(hx[4:6], 16)
            return (r, g, b)

    m = re.match(r"rgb\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)$", c, re.I)
    if m:
        r = int(round(float(m.group(1))))
        g = int(round(float(m.group(2))))
        b = int(round(float(m.group(3))))
        return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))

    if ImageColor is not None:
        try:
            r, g, b = ImageColor.getrgb(c)
            return (int(r), int(g), int(b))
        except Exception:
            pass

    raise ValueError(f"Unsupported/unknown color: {color!r}")


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


def _clamp01(x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x


def _mix_rgb(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    t = _clamp01(t)
    ar, ag, ab = a
    br, bg, bb = b
    r = int(round((1.0 - t) * ar + t * br))
    g = int(round((1.0 - t) * ag + t * bg))
    b2 = int(round((1.0 - t) * ab + t * bb))
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b2)))


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge0 == edge1:
        return 0.0
    t = _clamp01((x - edge0) / (edge1 - edge0))
    return t * t * (3.0 - 2.0 * t)


def _cosine_ease(x: float) -> float:
    """
    Smooth 0..1 -> 0..1 (cosine).
    """
    x = _clamp01(x)
    return 0.5 - 0.5 * math.cos(math.pi * x)


def _rgb255_to_hsv01(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    r, g, b = rgb
    return colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)


def _hsv01_to_rgb255(h: float, s: float, v: float) -> Tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, _clamp01(s), _clamp01(v))
    return (int(round(r * 255.0)), int(round(g * 255.0)), int(round(b * 255.0)))


def mix_to_white(base: Tuple[int, int, int], a: float) -> Tuple[int, int, int]:
    r0, g0, b0 = base
    r = int(round((1.0 - a) * r0 + a * 255.0))
    g = int(round((1.0 - a) * g0 + a * 255.0))
    b = int(round((1.0 - a) * b0 + a * 255.0))
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


def _scale_rgb(rgb: Tuple[int, int, int], f: float) -> Tuple[int, int, int]:
    r, g, b = rgb
    return (
        max(0, min(255, int(round(r * f)))),
        max(0, min(255, int(round(g * f)))),
        max(0, min(255, int(round(b * f)))),
    )


def _luma(rgb: Tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _is_whiteish_color_str(s: str) -> bool:
    if not s:
        return False
    c = s.strip().lower()
    if c in {"#fff", "#ffffff", "white"}:
        return True
    if c.startswith("rgb(") and "255" in c:
        return True
    try:
        r, g, b = _parse_css_color_to_rgb(s)
        return (r + g + b) / (3.0 * 255.0) > 0.92
    except Exception:
        return False


def _parse_polygon_points(points: str) -> List[Tuple[float, float]]:
    if not points:
        return []
    nums = [float(x) for x in NUM_RE.findall(points)]
    if len(nums) < 4:
        return []
    return list(zip(nums[0::2], nums[1::2]))


def _poly_centroid_local(poly: etree._Element) -> Optional[Tuple[float, float]]:
    pts = _parse_polygon_points(poly.get("points", ""))
    if not pts:
        return None
    x = sum(p[0] for p in pts) / float(len(pts))
    y = sum(p[1] for p in pts) / float(len(pts))
    return (x, y)


# -------------------- Minecraft texture sampling ------------------------------

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
    else:
        resample = Image.NEAREST
    img = img.resize((16, 16), resample=resample)

    w, h = img.size
    pixels: List[Tuple[int, int, int]] = []
    for y in range(h):
        for x in range(w):
            r, g, b, a = img.getpixel((x, y))
            pixels.append((int(r), int(g), int(b)))
    return pixels, w, h


def _glyph_viewbox_for_element(el: etree._Element, fallback: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    cur = el
    while cur is not None and isinstance(cur.tag, str):
        vbw = cur.get("data-vbw")
        vbh = cur.get("data-vbh")
        if vbw and vbh:
            try:
                minx = float(cur.get("data-minx", "0") or "0")
                miny = float(cur.get("data-miny", "0") or "0")
                vw = float(vbw)
                vh = float(vbh)
                return (minx, miny, vw, vh)
            except Exception:
                break
        cur = cur.getparent()
    return fallback


# -------------------- Logo building (combine 4 character SVGs) ----------------

_URL_ID_RE = re.compile(r"url\(#([^)]+)\)")


def _prefix_svg_ids(svg_fragment: etree._Element, prefix: str) -> None:
    id_map: Dict[str, str] = {}
    for el in svg_fragment.iter():
        if not isinstance(el.tag, str):
            continue
        old = el.get("id")
        if old:
            new = f"{prefix}{old}"
            id_map[old] = new
            el.set("id", new)

    if not id_map:
        return

    def rewrite_value(v: str) -> str:
        if not v:
            return v

        def repl(m: re.Match) -> str:
            old_id = m.group(1)
            new_id = id_map.get(old_id, old_id)
            return f"url(#{new_id})"

        out = _URL_ID_RE.sub(repl, v)

        if out.startswith("#") and out[1:] in id_map:
            out = "#" + id_map[out[1:]]

        for old_id, new_id in id_map.items():
            out = out.replace(f"#{old_id}", f"#{new_id}")

        return out

    for el in svg_fragment.iter():
        if not isinstance(el.tag, str):
            continue
        for attr, val in list(el.attrib.items()):
            if not isinstance(val, str):
                continue
            newv = rewrite_value(val)
            if newv != val:
                el.set(attr, newv)


def _strip_white_full_canvas_rects(svg_root: etree._Element, vb: Tuple[float, float, float, float]) -> None:
    minx, miny, vbw, vbh = vb
    tol = 1e-6
    rects = svg_root.xpath('.//*[local-name()="rect"]')
    for r in rects:
        if not isinstance(r.tag, str):
            continue
        if r.get("transform"):
            continue
        try:
            x = float(r.get("x", "0") or "0")
            y = float(r.get("y", "0") or "0")
            w = float(r.get("width", "0") or "0")
            h = float(r.get("height", "0") or "0")
        except Exception:
            continue

        covers = (
            abs(x - minx) < tol and
            abs(y - miny) < tol and
            abs(w - vbw) < tol and
            abs(h - vbh) < tol
        )
        if not covers:
            continue

        fill = _resolve_fill(r) or ""
        if _is_whiteish_color_str(fill):
            parent = r.getparent()
            if parent is not None:
                parent.remove(r)


def build_logo_svg_from_chars_2x2(char_svgs: List[Path], gap_flag: int) -> etree._Element:
    if len(char_svgs) != 4:
        raise ValueError("Expected exactly 4 character SVG paths for logo mode.")

    parser = etree.XMLParser(remove_blank_text=False, recover=True, remove_comments=False)
    glyph_docs: List[etree._Element] = []
    vbs: List[Tuple[float, float, float, float]] = []

    for p in char_svgs:
        if not p.is_file():
            raise SystemExit(f"Character SVG not found: {p}")
        root = etree.fromstring(p.read_bytes(), parser=parser)
        vb = _parse_viewbox(root)
        _strip_white_full_canvas_rects(root, vb)
        glyph_docs.append(root)
        vbs.append(vb)

    max_w = max(vb[2] for vb in vbs)
    max_h = max(vb[3] for vb in vbs)

    if gap_flag == 1:
        gap_x = max_w / 8.0
        gap_y = max_h / 8.0
        pad_x = gap_x
        pad_y = gap_y
    else:
        gap_x = gap_y = pad_x = pad_y = 0.0

    total_w = 2.0 * max_w + gap_x + 2.0 * pad_x
    total_h = 2.0 * max_h + gap_y + 2.0 * pad_y

    svg = etree.Element(f"{{{SVG_NS}}}svg", nsmap={None: SVG_NS, "xlink": XLINK_NS})
    svg.set("viewBox", f"0 0 {total_w} {total_h}")

    for idx, (groot, (minx, miny, vbw, vbh)) in enumerate(zip(glyph_docs, vbs)):
        row = 0 if idx < 2 else 1
        col = idx % 2

        cell_x0 = pad_x + col * (max_w + gap_x)
        cell_y0 = pad_y + row * (max_h + gap_y)

        tx = cell_x0 + (max_w - vbw) * 0.5 - minx
        ty = cell_y0 + (max_h - vbh) * 0.5 - miny

        group = etree.Element(f"{{{SVG_NS}}}g")
        for child in list(groot):
            group.append(copy.deepcopy(child))

        _prefix_svg_ids(group, f"g{idx}_")

        group.set("data-minx", str(minx))
        group.set("data-miny", str(miny))
        group.set("data-vbw", str(vbw))
        group.set("data-vbh", str(vbh))

        group.set("transform", f"translate({tx},{ty})")
        svg.append(group)

    return svg


# -------------------- Animation model ----------------------------------------

class Pulse:
    __slots__ = ("t0", "half", "amp", "power")

    def __init__(self, t0: float, half: float, amp: float, power: float = 1.0):
        self.t0 = t0
        self.half = half
        self.amp = amp
        self.power = power

    def value(self, t: float) -> float:
        dt = abs(t - self.t0)
        if dt >= self.half:
            return 0.0
        x = dt / self.half
        base = 0.5 * (1.0 + math.cos(math.pi * x))
        if self.power != 1.0:
            base = base ** self.power
        return self.amp * base


def facet_shimmer(t: float, freq: float, phase: float) -> float:
    s1 = 0.5 + 0.5 * math.sin(2.0 * math.pi * (freq * t + phase))
    s2 = 0.5 + 0.5 * math.sin(2.0 * math.pi * ((freq * 0.47) * t + (phase * 1.63)))
    s = s1 * s2
    return _clamp01(s ** 2.0)


def make_pulses(rng: random.Random, duration: float, theme: str) -> List[Pulse]:
    if theme == "deidee":
        return []
    if theme != "classic":
        pulses: List[Pulse] = []
        n_glints = rng.randint(2, 4) if theme == "minecraft" else rng.randint(1, 3)
        for _ in range(n_glints):
            t0 = rng.uniform(0.0, duration)
            half = rng.uniform(1.40, 3.80) if theme == "minecraft" else rng.uniform(2.10, 5.80)
            amp = rng.uniform(0.70, 1.00)
            power = rng.uniform(1.0, 1.30)
            pulses.append(Pulse(t0, half, amp, power=power))

        if rng.random() < 0.60:
            t0 = rng.uniform(0.0, duration)
            half = rng.uniform(3.50, 7.50)
            amp = rng.uniform(0.06, 0.14)
            pulses.append(Pulse(t0, half, amp, power=1.0))

        return pulses

    n = rng.randint(3, 7)
    pulses: List[Pulse] = []
    for _ in range(n):
        t0 = rng.uniform(0.0, duration)
        half = rng.uniform(0.25, 1.10)
        amp = rng.uniform(0.55, 1.00)
        pulses.append(Pulse(t0, half, amp, power=1.0))
    return pulses


def whiteness_at(t: float, pulses: List[Pulse]) -> float:
    a = 0.0
    for p in pulses:
        a = max(a, p.value(t))
    return _clamp01(a)


# -------------------- Themes --------------------------------------------------

@dataclass(frozen=True)
class ThemeConfig:
    kind: str  # "classic", "diamond", "hsv_body", "minecraft", "deidee"
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

    grey_dark: Tuple[int, int, int] = (6, 6, 8)
    grey_light: Tuple[int, int, int] = (255, 255, 255)

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
    fire_sat_mul_punchy_prob: float = 0.25
    fire_sat_mul_punchy_lo: float = 1.15
    fire_sat_mul_punchy_hi: float = 1.60

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
    if theme == "classic":
        return ThemeConfig(kind="classic")

    if theme == "deidee":
        return ThemeConfig(kind="deidee")

    if theme == "diamond":
        return ThemeConfig(
            kind="diamond",
            grey_dark=(6, 6, 8),
            grey_light=(255, 255, 255),
            spec_edge0=0.48,
            spec_scale=0.98,
            fire_prob=0.36,
            fire_hues=_FIRE_HUES_DEFAULT,
            fire_hue_jitter=0.02,
            fire_sat_base_min=0.12, fire_sat_base_max=0.28,
            fire_sat_peak_min=0.45, fire_sat_peak_max=0.85,
            fire_white_mix_min=0.48, fire_white_mix_max=0.72,
        )

    if theme == "minecraft":
        return ThemeConfig(
            kind="minecraft",
            amb_base=0.02,
            amb_amp=0.03,
            amb_freq=0.030,
            gl_pulse_w=0.82,
            gl_shim_w=0.55,
            spec_edge0=0.35,
            spec_scale=0.70,
            sheen_mix=0.0,
            fire_prob=0.0,
        )

    common = dict(
        kind="hsv_body",
        amb_base=0.06,
        amb_amp=0.025,
        spec_edge0=0.54,
        spec_scale=0.86,
        sheen_mix=0.18,
        sheen_sat_boost=0.22,
        fire_white_mix_min=0.10,
        fire_white_mix_max=0.42,
        val_shimmer_amp=0.035,
    )

    if theme == "silver":
        return _cfg_merge(common, dict(
            base_hue=210/360.0, hue_jitter=0.012,
            body_sat_min=0.22, body_sat_max=0.55,
            body_v_min=0.22, body_v_max=0.92, body_v_gamma=1.0,
            body_sat_mul=1.05,
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
            hue_tone_amp=0.0,
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

            # lots of “hot” glints; keep them colorful, not white
            fire_prob=0.92,
            fire_hues=[0/360.0, 12/360.0, 25/360.0, 40/360.0, 55/360.0, 65/360.0, 330/360.0],
            fire_hue_jitter=0.11,
            fire_sat_base_min=0.40, fire_sat_base_max=0.80,
            fire_sat_peak_min=0.90, fire_sat_peak_max=1.00,
            fire_white_mix_min=0.05, fire_white_mix_max=0.22,
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

            # prismatic “ice” glints; allow some white, but not always
            fire_prob=0.62,
            fire_hues=[175/360.0, 195/360.0, 210/360.0, 225/360.0, 245/360.0, 275/360.0],
            fire_hue_jitter=0.10,
            fire_sat_base_min=0.25, fire_sat_base_max=0.55,
            fire_sat_peak_min=0.75, fire_sat_peak_max=1.00,
            fire_white_mix_min=0.12, fire_white_mix_max=0.35,
        ))

    raise ValueError(f"Unknown theme: {theme!r}")


# -------------------- Rendering (SVG -> PNG) ---------------------------------

def _render_with_cairosvg(svg_bytes: bytes, out_png: Path, out_w: int, out_h: int) -> bool:
    if cairosvg is None:
        return False
    cairosvg.svg2png(
        bytestring=svg_bytes,
        write_to=str(out_png),
        output_width=out_w,
        output_height=out_h,
    )
    return True


def _render_with_inkscape(svg_bytes: bytes, out_png: Path, out_w: int, out_h: int) -> bool:
    inkscape = shutil.which("inkscape")
    if not inkscape:
        return False

    fd, tmp_svg = tempfile.mkstemp(suffix=".svg")
    os.close(fd)
    try:
        Path(tmp_svg).write_bytes(svg_bytes)

        cmd = [
            inkscape,
            tmp_svg,
            "--export-type=png",
            f"--export-filename={str(out_png)}",
            "--export-area-page",
            f"--export-width={out_w}",
            f"--export-height={out_h}",
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except Exception:
            cmd_old = [
                inkscape,
                tmp_svg,
                f"--export-png={str(out_png)}",
                "--export-area-page",
                f"--export-width={out_w}",
                f"--export-height={out_h}",
            ]
            subprocess.run(cmd_old, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
    finally:
        try:
            Path(tmp_svg).unlink(missing_ok=True)
        except Exception:
            pass


def _render_with_rsvg(svg_bytes: bytes, out_png: Path, out_w: int, out_h: int) -> bool:
    rsvg = shutil.which("rsvg-convert")
    if not rsvg:
        return False

    fd, tmp_svg = tempfile.mkstemp(suffix=".svg")
    os.close(fd)
    try:
        Path(tmp_svg).write_bytes(svg_bytes)
        cmd = [rsvg, "-o", str(out_png), "-w", str(out_w), "-h", str(out_h), tmp_svg]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    finally:
        try:
            Path(tmp_svg).unlink(missing_ok=True)
        except Exception:
            pass


def render_png(svg_bytes: bytes, out_png: Path, out_w: int, out_h: int) -> str:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if _render_with_cairosvg(svg_bytes, out_png, out_w, out_h):
        return "cairosvg"
    if _render_with_inkscape(svg_bytes, out_png, out_w, out_h):
        return "inkscape"
    if _render_with_rsvg(svg_bytes, out_png, out_w, out_h):
        return "rsvg-convert"

    raise RuntimeError(
        "No SVG renderer available. Install one of:\n"
        "  - CairoSVG:  py -m pip install cairosvg  (recommended)\n"
        "  - Inkscape (ensure `inkscape` is on PATH)\n"
        "  - rsvg-convert (librsvg) (ensure it is on PATH)\n"
    )


# -------------------- FFmpeg --------------------------------------------------

def encode_video_ffmpeg(frames_dir: Path, fps: int, out_file: Path, ext: str) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH. Install ffmpeg and try again.")

    pattern = str(frames_dir / "frame_%05d.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if ext.lower() == "mp4":
        cmd = [
            ffmpeg, "-y",
            "-framerate", str(fps),
            "-start_number", "0",
            "-i", pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-preset", "medium",
            "-movflags", "+faststart",
            str(out_file),
        ]
    elif ext.lower() == "webm":
        cmd = [
            ffmpeg, "-y",
            "-framerate", str(fps),
            "-start_number", "0",
            "-i", pattern,
            "-c:v", "libvpx-vp9",
            "-b:v", "0",
            "-crf", "32",
            "-pix_fmt", "yuv420p",
            str(out_file),
        ]
    else:
        raise ValueError(f"Unsupported --ext {ext!r}. Use mp4 or webm.")

    subprocess.run(cmd, check=True)


def _timestamped_if_exists(path: Path) -> Path:
    if not path.exists():
        return path
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return path.with_name(f"{path.stem}-{ts}{path.suffix}")


# -------------------- MAIN ----------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a polygon-pulse video from a character SVG or a 4-char logo.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--char", type=str, default=None, help="Single character: uses src/character-{char}.svg")
    g.add_argument("--chars", type=str, default=None,
                   help="Four chars key (ignoring spaces). Combines 4 character SVGs into 2x2.")

    ap.add_argument("--gap", type=int, default=0, choices=[0, 1],
                    help="Logo spacing (only for --chars): 0 = no gaps (default), 1 = pad+gap = 1/8 of cell size.")

    ap.add_argument("--color", type=str, default="", help="Override base color (CSS color).")
    ap.add_argument("--bgcolor", type=str, default="", help="Override background color (CSS color).")

    ap.add_argument(
        "--theme",
        type=str,
        default="classic",
        choices=[
            "classic", "diamond",
            "silver", "gold", "bronze",
            "ruby", "jade", "sapphire", "emerald",
            "rainbow",
            "minecraft",
            "deidee",
            "fire", "ice",
        ],
        help="Animation theme.",
    )

    ap.add_argument(
        "--minecraft-texture",
        type=str,
        default="",
        help="Minecraft theme only: path or URL to the Grass Block carried-side texture PNG (defaults to wiki URL).",
    )

    ap.add_argument("--duration", type=float, default=12.0, help="Duration in seconds (default: 12).")
    ap.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30).")
    ap.add_argument("--ext", type=str, default="mp4", help="Output extension: mp4 or webm (default: mp4).")

    ap.add_argument("--max-dim", type=int, default=1080,
                    help="Render so max(viewBox w,h) becomes this size (default: 1080).")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for repeatable animation.")
    ap.add_argument("--keep-frames", action="store_true", help="Keep rendered PNG frames (for debugging).")
    args = ap.parse_args()

    cfg = get_theme_config(args.theme)

    root = Path(__file__).resolve().parent.parent
    src_dir = root / "src"
    out_dir = root / "dist" / "videos"

    parser = etree.XMLParser(remove_blank_text=False, recover=True, remove_comments=False)

    if args.char is not None:
        ch = args.char
        if len(ch) != 1:
            raise SystemExit("--char must be exactly one character.")
        in_svg_path = src_dir / f"character-{ch}.svg"
        if not in_svg_path.is_file():
            raise SystemExit(f"Input SVG not found: {in_svg_path}")
        doc = etree.fromstring(in_svg_path.read_bytes(), parser=parser)
        out_file = out_dir / f"character-{ch}.{args.ext.lower()}"
        label = f"character {ch!r}"
    else:
        key = "".join(c for c in args.chars.strip() if not c.isspace()).lower()
        if len(key) != 4:
            raise SystemExit("--chars must be exactly four characters (ignoring spaces).")
        char_paths = [src_dir / f"character-{c}.svg" for c in key]
        doc = build_logo_svg_from_chars_2x2(char_paths, args.gap)
        out_file = out_dir / f"logo-{key}.{args.ext.lower()}"
        label = f"logo {key!r} (2x2, gap={args.gap})"

    out_file = _timestamped_if_exists(out_file)

    minx, miny, vb_w, vb_h = _parse_viewbox(doc)
    if vb_w <= 0 or vb_h <= 0:
        vb_w, vb_h = 240.0, 240.0

    # Background override
    bgcolor = args.bgcolor.strip() or None
    if bgcolor:
        for el in list(doc):
            if not isinstance(el.tag, str):
                continue
            if _local_name(el.tag) != "rect":
                continue
            try:
                xf = float(el.get("x", "0") or "0")
                yf = float(el.get("y", "0") or "0")
                wf = float(el.get("width", "") or "-1")
                hf = float(el.get("height", "") or "-1")
                if abs(xf - minx) < 1e-6 and abs(yf - miny) < 1e-6 and abs(wf - vb_w) < 1e-6 and abs(hf - vb_h) < 1e-6:
                    doc.remove(el)
                    break
            except Exception:
                pass

        rect = etree.Element(f"{{{SVG_NS}}}rect")
        rect.set("x", str(minx))
        rect.set("y", str(miny))
        rect.set("width", str(vb_w))
        rect.set("height", str(vb_h))
        rect.set("fill", bgcolor)
        doc.insert(0, rect)

    polys = doc.xpath('.//*[local-name()="polygon"]')
    if not polys:
        raise SystemExit("No <polygon> elements found in the input SVG(s).")

    rng = random.Random(args.seed)

    # Base colors for classic mode
    override_color = args.color.strip() or None
    base_rgb_override: Optional[Tuple[int, int, int]] = None
    override_hsv: Optional[Tuple[float, float, float]] = None
    if override_color:
        base_rgb_override = _parse_css_color_to_rgb(override_color)
        override_hsv = _rgb255_to_hsv01(base_rgb_override)

    base_rgbs: List[Tuple[int, int, int]] = []
    for p in polys:
        if base_rgb_override is not None:
            base_rgbs.append(base_rgb_override)
            continue
        fill = _resolve_fill(p) or "#000"
        try:
            base_rgbs.append(_parse_css_color_to_rgb(fill))
        except Exception:
            base_rgbs.append((0, 0, 0))

    pulses_per_poly: List[List[Pulse]] = [make_pulses(rng, float(args.duration), args.theme) for _ in polys]

    # ---------------- deidee data ----------------
    de_alpha = 0.5
    de_colors_per_poly: List[List[Tuple[int, int, int]]] = []
    de_seg_dur: List[float] = []
    de_phase: List[float] = []

    if cfg.kind == "deidee":
        # each polygon cycles through K colors, smoothly
        for _ in polys:
            k = rng.randint(4, 8)
            cols: List[Tuple[int, int, int]] = []
            for _j in range(k):
                r = int(round(rng.uniform(0.0, 0.5) * 255.0))
                g = int(round(rng.uniform(0.5, 1.0) * 255.0))
                b = int(round(rng.uniform(0.0, 0.75) * 255.0))
                cols.append((max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))))
            de_colors_per_poly.append(cols)

            # slower iteration so it feels intentional, not flickery
            de_seg_dur.append(rng.uniform(0.90, 2.60))   # seconds per transition
            de_phase.append(rng.uniform(0.0, 10.0))      # desync polygons a bit

        # force opacity to win even if a polygon has style-based opacity
        for poly in polys:
            poly.set("fill-opacity", f"{de_alpha:.3f}")
            st = (poly.get("style") or "").strip()
            poly.set("style", _style_set(st, "fill-opacity", f"{de_alpha:.3f}"))

    # ---------------- Minecraft per-poly sampled pixels (base + neighbours) ----------------
    mc_tex: Optional[List[Tuple[int, int, int]]] = None
    mc_tw = mc_th = 0
    mc_base_rgb: List[Tuple[int, int, int]] = []
    mc_hi_rgb: List[Tuple[int, int, int]] = []
    mc_lo_rgb: List[Tuple[int, int, int]] = []
    mc_alt_rgb: List[Tuple[int, int, int]] = []
    mc_grain: List[float] = []
    mc_flicker_freq: List[float] = []
    mc_flicker_phase: List[float] = []

    # Shared non-classic arrays (for diamond/hsv_body/minecraft)
    poly_tone_base: List[float] = []
    poly_tone_phase: List[float] = []
    poly_shimmer_freq: List[float] = []
    poly_shimmer_phase: List[float] = []

    poly_body_hue: List[float] = []
    poly_body_sat: List[float] = []

    poly_fire_enabled: List[bool] = []
    poly_fire_hue: List[float] = []
    poly_fire_hue_phase: List[float] = []
    poly_fire_sat_base: List[float] = []
    poly_fire_sat_peak: List[float] = []
    poly_fire_sat_mul: List[float] = []
    poly_fire_white_mix: List[float] = []

    if cfg.kind == "minecraft":
        mc_tex, mc_tw, mc_th = _load_minecraft_texture_16x16(args.minecraft_texture)

    if cfg.kind not in ("classic", "deidee"):
        for idx, poly in enumerate(polys):
            # tone baseline
            if cfg.kind == "diamond":
                if rng.random() < 0.52:
                    tone = (rng.random() ** 2.2) * 0.28
                else:
                    tone = 1.0 - (rng.random() ** 2.2) * 0.28
            else:
                u = rng.random()
                if u < 0.86:
                    tone = rng.uniform(0.38, 0.70)
                elif u < 0.97:
                    tone = rng.uniform(0.30, 0.78)
                else:
                    tone = rng.uniform(0.22, 0.86)

            poly_tone_base.append(tone)
            poly_tone_phase.append(rng.uniform(0.0, 1.0))
            poly_shimmer_freq.append(rng.uniform(0.05, 0.14))
            poly_shimmer_phase.append(rng.uniform(0.0, 1.0))

            # hue/sat body (unused for diamond/minecraft but kept aligned)
            if cfg.kind in ("diamond", "minecraft"):
                poly_body_hue.append(0.0)
                poly_body_sat.append(0.0)
            else:
                if override_hsv is not None:
                    h0, s0, _v0 = override_hsv
                    h = h0
                    s = max(cfg.body_sat_min, min(cfg.body_sat_max, max(s0, rng.uniform(cfg.body_sat_min, cfg.body_sat_max)) * 0.90))
                    poly_body_hue.append(h)
                    poly_body_sat.append(s)
                else:
                    if cfg.base_hue is None:
                        h = rng.random()
                    else:
                        h = (cfg.base_hue + rng.uniform(-cfg.hue_jitter, cfg.hue_jitter)) % 1.0
                    s = rng.uniform(cfg.body_sat_min, cfg.body_sat_max)
                    poly_body_hue.append(h)
                    poly_body_sat.append(s)

            # fire (if enabled by theme)
            poly_fire_enabled.append(rng.random() < cfg.fire_prob)
            if cfg.fire_hues is not None:
                h_fire = rng.choice(cfg.fire_hues)
            else:
                h_fire = poly_body_hue[-1] if poly_body_hue else (cfg.base_hue or 0.0)
            h_fire = (h_fire + rng.uniform(-cfg.fire_hue_jitter, cfg.fire_hue_jitter)) % 1.0
            poly_fire_hue.append(h_fire)
            poly_fire_hue_phase.append(rng.uniform(0.0, 1.0))

            sb = rng.uniform(cfg.fire_sat_base_min, cfg.fire_sat_base_max)
            sp = rng.uniform(cfg.fire_sat_peak_min, cfg.fire_sat_peak_max)
            if rng.random() < cfg.fire_sat_mul_punchy_prob:
                mul = rng.uniform(cfg.fire_sat_mul_punchy_lo, cfg.fire_sat_mul_punchy_hi)
            else:
                mul = rng.uniform(cfg.fire_sat_mul_lo, cfg.fire_sat_mul_hi)

            poly_fire_sat_base.append(sb)
            poly_fire_sat_peak.append(sp)
            poly_fire_sat_mul.append(mul)
            poly_fire_white_mix.append(rng.uniform(cfg.fire_white_mix_min, cfg.fire_white_mix_max))

            # minecraft neighbor sampling
            if cfg.kind == "minecraft":
                c = _poly_centroid_local(poly) or (0.5 * 240.0, 0.5 * 240.0)
                gx, gy = c

                gminx, gminy, gvw, gvh = _glyph_viewbox_for_element(poly, (minx, miny, vb_w, vb_h))

                nx = (gx - gminx) / gvw if gvw > 0 else 0.5
                ny = (gy - gminy) / gvh if gvh > 0 else 0.5
                nx = _clamp01(nx)
                ny = _clamp01(ny)

                u_px = int(round(nx * (mc_tw - 1)))
                v_px = int(round(ny * (mc_th - 1)))

                if rng.random() < 0.35:
                    u_px = max(0, min(mc_tw - 1, u_px + rng.choice([-1, 0, 1])))
                    v_px = max(0, min(mc_th - 1, v_px + rng.choice([-1, 0, 1])))

                coords = [
                    (u_px, v_px),
                    (max(0, u_px - 1), v_px),
                    (min(mc_tw - 1, u_px + 1), v_px),
                    (u_px, max(0, v_px - 1)),
                    (u_px, min(mc_th - 1, v_px + 1)),
                ]
                samples = [(uv, mc_tex[uv[1] * mc_tw + uv[0]]) for uv in coords]  # type: ignore[index]
                base = samples[0][1]
                hi = max(samples, key=lambda it: _luma(it[1]))[1]
                lo = min(samples, key=lambda it: _luma(it[1]))[1]
                alt = rng.choice(samples[1:])[1] if len(samples) > 1 else base

                mc_base_rgb.append(base)
                mc_hi_rgb.append(hi)
                mc_lo_rgb.append(lo)
                mc_alt_rgb.append(alt)

                dirtish = 1.0 if v_px >= int(mc_th * 0.30) else 0.0
                mc_grain.append(rng.uniform(0.84, 1.22) if dirtish else rng.uniform(0.90, 1.14))
                mc_flicker_freq.append(rng.uniform(0.05, 0.13))
                mc_flicker_phase.append(rng.uniform(0.0, 1.0))

    # Output sizing
    max_dim = int(args.max_dim)
    scale = float(max_dim) / float(max(vb_w, vb_h))
    out_w = max(1, int(round(vb_w * scale)))
    out_h = max(1, int(round(vb_h * scale)))

    frames = int(round(float(args.duration) * int(args.fps)))
    fps = int(args.fps)

    tmp_root = Path(tempfile.mkdtemp(prefix="mmxx_video_"))
    frames_dir = tmp_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    try:
        renderer_used = None

        for i in range(frames):
            t = i / float(fps)

            if cfg.kind == "classic":
                for idx, poly in enumerate(polys):
                    a = whiteness_at(t, pulses_per_poly[idx])
                    poly.set("fill", _rgb_to_hex(mix_to_white(base_rgbs[idx], a)))

            elif cfg.kind == "deidee":
                # smooth cycling between random samples; keep alpha constant
                for idx, poly in enumerate(polys):
                    cols = de_colors_per_poly[idx]
                    k = len(cols)
                    seg = de_seg_dur[idx]
                    ph = de_phase[idx]

                    pos = (t + ph) / seg
                    i0 = int(math.floor(pos)) % k
                    i1 = (i0 + 1) % k
                    f = pos - math.floor(pos)
                    u = _cosine_ease(f)

                    rgb = _mix_rgb(cols[i0], cols[i1], u)
                    poly.set("fill", _rgb_to_hex(rgb))

                    # enforce transparency every frame (wins over inherited styles)
                    poly.set("fill-opacity", f"{de_alpha:.3f}")
                    st = (poly.get("style") or "").strip()
                    poly.set("style", _style_set(st, "fill-opacity", f"{de_alpha:.3f}"))

            else:
                # existing non-classic themes
                global_amb = cfg.amb_base + cfg.amb_amp * (0.5 + 0.5 * math.sin(2.0 * math.pi * (cfg.amb_freq * t)))

                for idx, poly in enumerate(polys):
                    tone0 = poly_tone_base[idx]
                    wobble_amp = 0.08 if cfg.kind == "diamond" else 0.050
                    wobble = wobble_amp * math.sin(2.0 * math.pi * (0.06 * t + poly_tone_phase[idx]))
                    tone = _clamp01(tone0 + wobble)

                    a = whiteness_at(t, pulses_per_poly[idx])
                    shim = facet_shimmer(t, poly_shimmer_freq[idx], poly_shimmer_phase[idx])
                    gl = _clamp01(cfg.gl_pulse_w * a + cfg.gl_shim_w * shim)

                    spec = _smoothstep(cfg.spec_edge0, 1.00, gl)
                    spec_amt = spec * cfg.spec_scale

                    if cfg.kind == "minecraft":
                        base = mc_base_rgb[idx]
                        hi = mc_hi_rgb[idx]
                        lo = mc_lo_rgb[idx]
                        alt = mc_alt_rgb[idx]

                        f = 0.5 + 0.5 * math.sin(2.0 * math.pi * (mc_flicker_freq[idx] * t + mc_flicker_phase[idx]))
                        f = f ** 1.8

                        shade = (0.62 + 0.78 * tone) * mc_grain[idx]
                        shade *= (0.90 + 0.22 * f)
                        shade *= (0.92 + 0.28 * _smoothstep(0.20, 0.95, gl))

                        body = _scale_rgb(base, shade)

                        shadow_amt = 0.20 + 0.35 * _smoothstep(0.00, 0.55, 1.0 - tone)
                        body = _mix_rgb(body, _scale_rgb(lo, shade * 0.92), shadow_amt * 0.35)

                        body = _mix_rgb(body, _scale_rgb(hi, shade * 1.02), global_amb)

                        sparkle = _smoothstep(0.35, 1.00, gl)
                        sparkle *= (0.15 + 0.55 * f)
                        body = _mix_rgb(body, _scale_rgb(alt, shade * 1.04), sparkle * 0.55)

                        body = _mix_rgb(body, _scale_rgb(hi, shade * 1.22), spec_amt * (0.35 + 0.35 * f))
                        if spec_amt > 0.85:
                            body = _mix_rgb(body, (255, 255, 255), (spec_amt - 0.85) * 0.10)

                        poly.set("fill", _rgb_to_hex(body))
                        continue

                    if cfg.kind == "diamond":
                        body_grey = _mix_rgb(cfg.grey_dark, cfg.grey_light, tone)
                        body2 = mix_to_white(body_grey, global_amb)

                        if poly_fire_enabled[idx]:
                            fire_gate = _smoothstep(cfg.fire_gate0, 1.00, gl)
                            hue = (
                                poly_fire_hue[idx]
                                + cfg.fire_hue_drift_amp * math.sin(2.0 * math.pi * (cfg.fire_hue_drift_freq * t + poly_fire_hue_phase[idx]))
                            ) % 1.0
                            sat = poly_fire_sat_base[idx] + (poly_fire_sat_peak[idx] - poly_fire_sat_base[idx]) * fire_gate
                            sat *= poly_fire_sat_mul[idx]
                            sat = _clamp01(sat)
                            fire_rgb = _hsv01_to_rgb255(hue, sat, 1.0)
                            tw = _mix_rgb(fire_rgb, (255, 255, 255), poly_fire_white_mix[idx])
                            rgb = _mix_rgb(body2, tw, spec_amt)
                        else:
                            rgb = _mix_rgb(body2, (255, 255, 255), spec_amt)

                        poly.set("fill", _rgb_to_hex(rgb))
                        continue

                    # hsv_body materials
                    h0 = poly_body_hue[idx]
                    h = (h0 + cfg.hue_tone_amp * (tone - 0.5) + cfg.hue_shimmer_amp * (shim - 0.5)) % 1.0

                    v = cfg.body_v_min + (cfg.body_v_max - cfg.body_v_min) * (tone ** cfg.body_v_gamma)
                    if cfg.val_shimmer_amp:
                        v = _clamp01(v + cfg.val_shimmer_amp * (shim - 0.5))

                    s0 = poly_body_sat[idx] * cfg.body_sat_mul
                    s = s0 * (1.0 + cfg.sat_dark_boost * (1.0 - tone))
                    s = _clamp01(s)

                    body3 = _hsv01_to_rgb255(h, s, v)
                    body3 = mix_to_white(body3, global_amb)

                    if poly_fire_enabled[idx]:
                        fire_gate = _smoothstep(cfg.fire_gate0, 1.00, gl)
                        hue = (
                            poly_fire_hue[idx]
                            + cfg.fire_hue_drift_amp * math.sin(2.0 * math.pi * (cfg.fire_hue_drift_freq * t + poly_fire_hue_phase[idx]))
                        ) % 1.0
                        sat = poly_fire_sat_base[idx] + (poly_fire_sat_peak[idx] - poly_fire_sat_base[idx]) * fire_gate
                        sat *= poly_fire_sat_mul[idx]
                        sat = _clamp01(sat)
                        fire_rgb = _hsv01_to_rgb255(hue, sat, 1.0)
                        tw = _mix_rgb(fire_rgb, (255, 255, 255), poly_fire_white_mix[idx])
                        rgb = _mix_rgb(body3, tw, spec_amt)
                    else:
                        hb, sb, vb = _rgb255_to_hsv01(body3)
                        hb2 = (hb + cfg.sheen_hue_shift + 0.010 * (shim - 0.5)) % 1.0
                        sb2 = _clamp01(sb * (1.0 + cfg.sheen_sat_boost))
                        sheen_rgb = _hsv01_to_rgb255(hb2, sb2, 1.0)
                        sheen = _mix_rgb(sheen_rgb, (255, 255, 255), cfg.sheen_mix)
                        rgb = _mix_rgb(body3, sheen, spec_amt)

                    poly.set("fill", _rgb_to_hex(rgb))

            svg_bytes = etree.tostring(doc, encoding="utf-8", xml_declaration=False)
            out_png = frames_dir / f"frame_{i:05d}.png"
            renderer_used = render_png(svg_bytes, out_png, out_w, out_h)

        encode_video_ffmpeg(frames_dir, fps, out_file, args.ext)

        print(f"Output: {out_file}")
        print(f"Theme:  {args.theme}")
        print(f"Input:  {label}")
        print(f"Frames: {frames} @ {fps}fps, size={out_w}x{out_h}, renderer={renderer_used}")

        if args.keep_frames:
            kept = root / "dist" / "videos" / "_frames"
            kept.mkdir(parents=True, exist_ok=True)
            dst = kept / (out_file.stem + "_frames")
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(frames_dir, dst)
            print(f"Kept frames at: {dst}")

    finally:
        if not args.keep_frames:
            try:
                shutil.rmtree(tmp_root)
            except Exception:
                pass


if __name__ == "__main__":
    main()
