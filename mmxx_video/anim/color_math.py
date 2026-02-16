from __future__ import annotations
import colorsys
from typing import Tuple
from .easing import clamp01

RGB = Tuple[int, int, int]

def mix_rgb(a: RGB, b: RGB, t: float) -> RGB:
    t = clamp01(t)
    ar, ag, ab = a
    br, bg, bb = b
    r = int(round((1.0 - t) * ar + t * br))
    g = int(round((1.0 - t) * ag + t * bg))
    b2 = int(round((1.0 - t) * ab + t * bb))
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b2)))

def add_rgb(base: RGB, add: RGB, amt: float) -> RGB:
    amt = clamp01(amt)
    r = int(round(base[0] + add[0] * amt))
    g = int(round(base[1] + add[1] * amt))
    b = int(round(base[2] + add[2] * amt))
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))

def rgb255_to_hsv01(rgb: RGB):
    r, g, b = rgb
    return colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)

def hsv01_to_rgb255(h: float, s: float, v: float) -> RGB:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, clamp01(s), clamp01(v))
    return (int(round(r * 255.0)), int(round(g * 255.0)), int(round(b * 255.0)))

def rgb_to_hex(rgb: RGB) -> str:
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"

def mix_to_white(base: RGB, a: float) -> RGB:
    r0, g0, b0 = base
    r = int(round((1.0 - a) * r0 + a * 255.0))
    g = int(round((1.0 - a) * g0 + a * 255.0))
    b = int(round((1.0 - a) * b0 + a * 255.0))
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))

def mix_to_white2(base: RGB, a: float) -> RGB:
    return mix_to_white(base, a)
