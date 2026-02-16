from __future__ import annotations
import re
from typing import Tuple

try:
    from PIL import ImageColor  # type: ignore
except Exception:
    ImageColor = None

RGB = Tuple[int, int, int]

def parse_css_color_to_rgb(color: str) -> RGB:
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

def is_whiteish_color_str(s: str) -> bool:
    if not s:
        return False
    c = s.strip().lower()
    if c in {"#fff", "#ffffff", "white"}:
        return True
    try:
        r, g, b = parse_css_color_to_rgb(s)
        return (r + g + b) / (3.0 * 255.0) > 0.92
    except Exception:
        return False
