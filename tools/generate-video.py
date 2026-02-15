#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/generate-video.py

Generate a video from either:
  - --char  -> single glyph SVG
  - --chars -> combines N glyph SVGs into a single SVG grid (NxN), where N depends on count(chars):
        4  -> 2x2
        9  -> 3x3
        16 -> 4x4
    Order is left-to-right, top-to-bottom.

Input resolution (backwards-compatible):
- For a glyph token T:
    1) try: src/{prefix}-{T}.svg   (e.g. character-a.svg)
    2) else: src/{prefix}-u{XXXX}.svg (e.g. character-u0061.svg)

  prefix = "character" or "inverse" (if --inverse is set)

Accepted --char formats:
- --char a
- --char u0061 / U+0061 / 0061

Accepted --chars formats:
- --chars abcd              (spaces ignored)
- --chars "u0061 u0062 u0063 u0064" (whitespace-separated codepoint tokens)
- must be 4 / 9 / 16 items after parsing

Grid layout (for --chars):
- --gap 0 (default): no extra spacing/padding
- --gap 1: spacing AND outer padding = 1/8th of a character cell size

Also in --chars mode:
- Removes per-glyph full-canvas WHITE background rects (prevents white squares).

Themes:
- classic (default): pulse-to-white using polygon base colors (or --color)
- diamond: high-contrast greys + strong specular + dispersion "fire" in highlights
- silver / gold / bronze
- ruby / jade / sapphire / emerald
- rainbow
- fire / ice
- valentines
- matrix: "code rain" columns (falling bright heads + trailing tails)
- snow
- minecraft: samples from Minecraft grass block texture (animated glints, texture remains recognizable)
- deidee: interpolates between random RGBA samples from:
    fill(random(0, .5), random(.5, 1), random(0, .75), .5)
- heart: pulsating graphical heart (silhouette) with valentine-like palette + smoother pulse growth
- static: TV-like noise/static (mostly greys with rare color specks)
- champagne: sparkly-wine bubbles rising upward (warm liquid + bright bubble trails)
- camo: Scorpion W2 (OCP-like) procedural camouflage (animated drift + fabric grain)
- fireworks: exploding “fire arrow” rockets + colorful bursts on a dark sky

Output:
  dist/videos/{prefix}-{token}.{ext} (for --char)
  dist/videos/logo-{...}.{ext}       (for --chars)

If output filename already exists:
  - appends a timestamp: ...-YYYYMMDD-HHMMSS.ext

Dependencies (recommended):
  - cairosvg  (fast SVG->PNG):  py -m pip install cairosvg
  - ffmpeg    (on PATH)

Fallback renderers (slower): inkscape, rsvg-convert
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
_CODEPOINT_RE = re.compile(r"^(?:u\+?|U\+?)([0-9a-fA-F]{4,8})$")
_HEX_RE = re.compile(r"^[0-9a-fA-F]{4,8}$")
_TRANSLATE_RE = re.compile(
    r"translate\(\s*(" + NUM_RE.pattern + r")\s*(?:[, ]\s*(" + NUM_RE.pattern + r"))?\s*\)",
    re.I
)
_URL_ID_RE = re.compile(r"url\(#([^)]+)\)")

DEFAULT_MINECRAFT_TEXTURE_URL = (
    "https://static.wikia.nocookie.net/minecraft_gamepedia/images/b/b2/"
    "Grass_Block_%28carried_side_texture%29_BE1.png/revision/latest?cb=20200928054656"
)

# Heart: allow max pulse scale without clipping while still filling the canvas tightly.
_HEART_BEAT_AMP = 0.11  # keep existing “punch” but fit accounts for this


# -------------------- Small utils --------------------------------------------

def _clamp01(x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge0 == edge1:
        return 0.0
    t = _clamp01((x - edge0) / (edge1 - edge0))
    return t * t * (3.0 - 2.0 * t)


def _cosine_ease(x: float) -> float:
    x = _clamp01(x)
    return 0.5 - 0.5 * math.cos(math.pi * x)


def _mix_rgb(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    t = _clamp01(t)
    ar, ag, ab = a
    br, bg, bb = b
    r = int(round((1.0 - t) * ar + t * br))
    g = int(round((1.0 - t) * ag + t * bg))
    b2 = int(round((1.0 - t) * ab + t * bb))
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b2)))


def _add_rgb(base: Tuple[int, int, int], add: Tuple[int, int, int], amt: float) -> Tuple[int, int, int]:
    """Additive blend (glow-like). amt is 0..1 meaning “how much of add to pour in”."""
    amt = _clamp01(amt)
    r = int(round(base[0] + add[0] * amt))
    g = int(round(base[1] + add[1] * amt))
    b = int(round(base[2] + add[2] * amt))
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


def _rgb255_to_hsv01(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    r, g, b = rgb
    return colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)


def _hsv01_to_rgb255(h: float, s: float, v: float) -> Tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, _clamp01(s), _clamp01(v))
    return (int(round(r * 255.0)), int(round(g * 255.0)), int(round(b * 255.0)))


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


def mix_to_white(base: Tuple[int, int, int], a: float) -> Tuple[int, int, int]:
    r0, g0, b0 = base
    r = int(round((1.0 - a) * r0 + a * 255.0))
    g = int(round((1.0 - a) * g0 + a * 255.0))
    b = int(round((1.0 - a) * b0 + a * 255.0))
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


def _timestamped_if_exists(path: Path) -> Path:
    if not path.exists():
        return path
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return path.with_name(f"{path.stem}-{ts}{path.suffix}")


# -------------------- SVG helpers --------------------------------------------

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
        if k.strip().lower() == key.strip().lower():
            return v.strip()
    return None


def _style_set(style: str, key: str, value: str) -> str:
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


def _is_whiteish_color_str(s: str) -> bool:
    if not s:
        return False
    c = s.strip().lower()
    if c in {"#fff", "#ffffff", "white"}:
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


def _parse_translate(transform: str) -> Tuple[float, float]:
    if not transform:
        return (0.0, 0.0)
    m = _TRANSLATE_RE.search(transform)
    if not m:
        return (0.0, 0.0)
    tx = float(m.group(1))
    ty = float(m.group(2)) if m.group(2) is not None else 0.0
    return (tx, ty)


def _accumulate_translate(el: etree._Element) -> Tuple[float, float]:
    tx = 0.0
    ty = 0.0
    cur = el
    while cur is not None and isinstance(cur.tag, str):
        tr = (cur.get("transform") or "").strip()
        if tr:
            dx, dy = _parse_translate(tr)
            tx += dx
            ty += dy
        cur = cur.getparent()
    return (tx, ty)


def _global_centroid_norm(poly: etree._Element, vb: Tuple[float, float, float, float]) -> Tuple[float, float]:
    minx, miny, vbw, vbh = vb
    c = _poly_centroid_local(poly)
    if c is None:
        return (0.5, 0.5)
    tx, ty = _accumulate_translate(poly)
    gx = c[0] + tx
    gy = c[1] + ty
    nx = (gx - minx) / vbw if vbw > 0 else 0.5
    ny = (gy - miny) / vbh if vbh > 0 else 0.5
    return (_clamp01(nx), _clamp01(ny))


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


def _centroid_in_glyph_norm(poly: etree._Element, root_vb: Tuple[float, float, float, float]) -> Tuple[float, float]:
    vb = _glyph_viewbox_for_element(poly, root_vb)
    minx, miny, vbw, vbh = vb
    c = _poly_centroid_local(poly)
    if c is None:
        return (0.5, 0.5)
    nx = (c[0] - minx) / vbw if vbw > 0 else 0.5
    ny = (c[1] - miny) / vbh if vbh > 0 else 0.5
    return (_clamp01(nx), _clamp01(ny))


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
            if isinstance(val, str):
                newv = rewrite_value(val)
                if newv != val:
                    el.set(attr, newv)


# -------------------- Char/codepoint parsing ---------------------------------

def _parse_char_or_codepoint(s: str) -> Tuple[str, int, str]:
    raw = (s or "").strip()
    if not raw:
        raise ValueError("Empty char/codepoint")

    if len(raw) == 1:
        cp = ord(raw)
        return (f"u{cp:04x}".lower(), cp, raw)

    m = _CODEPOINT_RE.match(raw)
    if m:
        cp = int(m.group(1), 16)
        disp = chr(cp) if 0 <= cp <= 0x10FFFF else raw
        return (f"u{cp:04x}".lower(), cp, disp)

    if _HEX_RE.match(raw):
        cp = int(raw, 16)
        disp = chr(cp) if 0 <= cp <= 0x10FFFF else raw
        return (f"u{cp:04x}".lower(), cp, disp)

    raise ValueError(f"Not a single character or codepoint token: {raw!r}")


def _parse_chars_arg(chars: str) -> List[Tuple[str, int, str]]:
    s = (chars or "").strip()
    if not s:
        return []

    toks = [t for t in re.split(r"\s+", s) if t]
    if len(toks) > 1 and all((_CODEPOINT_RE.match(t) or _HEX_RE.match(t)) for t in toks):
        return [_parse_char_or_codepoint(t) for t in toks]

    compact = "".join(c for c in s if not c.isspace())
    return [_parse_char_or_codepoint(c) for c in compact]


def _safe_logo_key(items: List[Tuple[str, int, str]]) -> str:
    out: List[str] = []
    for token, _cp, disp in items:
        if len(disp) == 1 and ord(disp) < 128 and disp.isalnum():
            out.append(disp.lower())
        else:
            out.append(token.lower())
    return "".join(out)


def _resolve_glyph_svg_path(src_dir: Path, prefix: str, token: str, disp: str) -> Path:
    if len(disp) == 1:
        p1 = src_dir / f"{prefix}-{disp}.svg"
        if p1.is_file():
            return p1
    return src_dir / f"{prefix}-{token}.svg"


# -------------------- Logo building (combine glyph SVGs into grid) ------------

def build_logo_svg_from_chars_grid(char_svgs: List[Path], grid_n: int, gap_flag: int) -> etree._Element:
    if grid_n not in (2, 3, 4):
        raise ValueError("grid_n must be 2, 3, or 4.")
    if len(char_svgs) != grid_n * grid_n:
        raise ValueError(f"Expected {grid_n*grid_n} character SVG paths for {grid_n}x{grid_n} grid.")

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

    total_w = grid_n * max_w + (grid_n - 1) * gap_x + 2.0 * pad_x
    total_h = grid_n * max_h + (grid_n - 1) * gap_y + 2.0 * pad_y

    svg = etree.Element(f"{{{SVG_NS}}}svg", nsmap={None: SVG_NS, "xlink": XLINK_NS})
    svg.set("viewBox", f"0 0 {total_w} {total_h}")

    for idx, (groot, (minx, miny, vbw, vbh)) in enumerate(zip(glyph_docs, vbs)):
        row = idx // grid_n
        col = idx % grid_n

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


# -------------------- Animation primitives -----------------------------------

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


def whiteness_at(t: float, pulses: List[Pulse]) -> float:
    a = 0.0
    for p in pulses:
        a = max(a, p.value(t))
    return _clamp01(a)


def facet_shimmer(t: float, freq: float, phase: float) -> float:
    s1 = 0.5 + 0.5 * math.sin(2.0 * math.pi * (freq * t + phase))
    s2 = 0.5 + 0.5 * math.sin(2.0 * math.pi * ((freq * 0.47) * t + (phase * 1.63)))
    s = s1 * s2
    return _clamp01(s ** 2.0)


# -------------------- Themes --------------------------------------------------

@dataclass(frozen=True)
class ThemeConfig:
    kind: str
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

    if theme == "diamond":
        return ThemeConfig(
            kind="diamond",
            amb_base=0.045, amb_amp=0.030, amb_freq=0.022,
            spec_edge0=0.48, spec_scale=0.98,
            fire_prob=0.38,
            fire_hues=_FIRE_HUES_DEFAULT,
            fire_hue_jitter=0.02,
            fire_sat_base_min=0.12, fire_sat_base_max=0.28,
            fire_sat_peak_min=0.45, fire_sat_peak_max=0.85,
            fire_white_mix_min=0.48, fire_white_mix_max=0.72,
        )

    if theme == "minecraft":
        return ThemeConfig(
            kind="minecraft",
            amb_base=0.02, amb_amp=0.03, amb_freq=0.030,
            gl_pulse_w=0.80, gl_shim_w=0.52,
            spec_edge0=0.36, spec_scale=0.72,
            sheen_mix=0.10, sheen_sat_boost=0.10,
            fire_prob=0.0,
        )

    if theme == "deidee":
        return ThemeConfig(kind="deidee")

    if theme == "heart":
        return ThemeConfig(kind="heart")

    if theme == "static":
        return ThemeConfig(kind="static")

    if theme == "champagne":
        return ThemeConfig(kind="champagne")

    if theme == "matrix":
        return ThemeConfig(kind="matrix")

    if theme == "camo":
        return ThemeConfig(kind="camo")

    if theme == "fireworks":
        return ThemeConfig(kind="fireworks")

    common = dict(
        kind="hsv",
        amb_base=0.06,
        amb_amp=0.025,
        spec_edge0=0.54,
        spec_scale=0.86,
        sheen_mix=0.18,
        sheen_sat_boost=0.22,
        val_shimmer_amp=0.035,
        fire_white_mix_min=0.10,
        fire_white_mix_max=0.42,
    )

    if theme == "silver":
        return _cfg_merge(common, dict(
            base_hue=210/360.0, hue_jitter=0.012,
            body_sat_min=0.22, body_sat_max=0.55,
            body_v_min=0.22, body_v_max=0.92, body_v_gamma=1.0,
            body_sat_mul=1.06,
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
            hue_shimmer_amp=0.020,
            val_shimmer_amp=0.030,
            spec_edge0=0.46, spec_scale=0.92,
            sheen_mix=0.10, sheen_sat_boost=0.36, sheen_hue_shift=0.018,
            fire_prob=0.92,
            fire_hues=[0/360.0, 12/360.0, 25/360.0, 40/360.0, 55/360.0, 65/360.0, 330/360.0],
            fire_hue_jitter=0.11,
            fire_sat_base_min=0.40, fire_sat_base_max=0.80,
            fire_sat_peak_min=0.90, fire_sat_peak_max=1.00,
            fire_white_mix_min=0.05,
            fire_white_mix_max=0.22,
        ))

    if theme == "ice":
        return _cfg_merge(common, dict(
            base_hue=205/360.0, hue_jitter=0.055,
            body_sat_min=0.75, body_sat_max=1.00,
            body_v_min=0.18, body_v_max=0.96, body_v_gamma=0.95,
            body_sat_mul=1.08,
            sat_dark_boost=0.16,
            hue_shimmer_amp=0.020,
            val_shimmer_amp=0.040,
            spec_edge0=0.52, spec_scale=0.92,
            sheen_mix=0.22, sheen_sat_boost=0.30, sheen_hue_shift=-0.010,
            fire_prob=0.62,
            fire_hues=[175/360.0, 195/360.0, 210/360.0, 225/360.0, 245/360.0, 275/360.0],
            fire_hue_jitter=0.10,
            fire_sat_base_min=0.25, fire_sat_base_max=0.55,
            fire_sat_peak_min=0.75, fire_sat_peak_max=1.00,
            fire_white_mix_min=0.12, fire_white_mix_max=0.35,
        ))

    if theme == "valentines":
        return _cfg_merge(common, dict(
            base_hue=335/360.0, hue_jitter=0.090,
            body_sat_min=0.78, body_sat_max=1.00,
            body_v_min=0.22, body_v_max=0.98, body_v_gamma=0.95,
            body_sat_mul=1.10,
            sat_dark_boost=0.18,
            hue_shimmer_amp=0.028,
            val_shimmer_amp=0.045,
            spec_edge0=0.50, spec_scale=0.92,
            sheen_mix=0.16, sheen_sat_boost=0.34, sheen_hue_shift=0.010,
            fire_prob=0.78,
            fire_hues=[350/360.0, 0/360.0, 10/360.0, 330/360.0, 315/360.0, 45/360.0],
            fire_hue_jitter=0.10,
            fire_sat_base_min=0.28, fire_sat_base_max=0.62,
            fire_sat_peak_min=0.80, fire_sat_peak_max=1.00,
            fire_white_mix_min=0.10, fire_white_mix_max=0.38,
        ))

    if theme == "snow":
        return _cfg_merge(common, dict(
            base_hue=205/360.0, hue_jitter=0.05,
            body_sat_min=0.04, body_sat_max=0.30,
            body_v_min=0.10, body_v_max=1.00, body_v_gamma=0.90,
            body_sat_mul=1.00,
            sat_dark_boost=0.05,
            hue_shimmer_amp=0.010,
            val_shimmer_amp=0.055,
            spec_edge0=0.52, spec_scale=0.96,
            sheen_mix=0.30, sheen_sat_boost=0.06, sheen_hue_shift=-0.004,
            fire_prob=0.55,
            fire_hues=[190/360.0, 205/360.0, 220/360.0, 235/360.0, 255/360.0],
            fire_hue_jitter=0.08,
            fire_sat_base_min=0.10, fire_sat_base_max=0.30,
            fire_sat_peak_min=0.25, fire_sat_peak_max=0.55,
            fire_white_mix_min=0.60, fire_white_mix_max=0.88,
        ))

    raise ValueError(f"Unknown theme: {theme!r}")


# -------------------- Per-theme precomputed state ----------------------------

@dataclass
class PolyHSV:
    h: float
    s: float
    v: float
    sat_mul: float
    v_mul: float
    freq: float
    phase: float
    fire_enabled: bool
    fire_hue: float
    fire_sat_mul: float
    fire_white_mix: float


@dataclass(frozen=True)
class MatrixDrop:
    speed: float
    phase: float
    tail: float
    head: float
    strength: float
    flicker_freq: float
    flicker_phase: float


@dataclass(frozen=True)
class Bubble:
    x: float
    y0: float
    r: float
    speed: float
    wob_amp: float
    wob_freq: float
    wob_phase: float
    strength: float


@dataclass(frozen=True)
class Firework:
    x: float
    yb: float
    t_launch: float
    t_burst: float
    vel: float
    ring_w: float
    decay: float
    spoke_n: int
    spoke_phase: float
    hue_a: float
    hue_b: float
    glitter_f: float
    glitter_p: float
    trail_len: float
    trail_w: float
    trail_h: float


# -------------------- Pulse schedules ----------------------------------------

def make_pulses(rng: random.Random, duration: float, theme: str) -> List[Pulse]:
    if theme in {"deidee", "matrix", "heart", "static", "champagne", "camo", "fireworks"}:
        return []

    if theme != "classic":
        pulses: List[Pulse] = []
        n_glints = rng.randint(1, 3)
        for _ in range(n_glints):
            t0 = rng.uniform(0.0, duration)
            half = rng.uniform(2.10, 5.80)
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
            r, g, b, _a = img.getpixel((x, y))
            pixels.append((int(r), int(g), int(b)))
    return pixels, w, h


# -------------------- Static helpers -----------------------------------------

def _fract(x: float) -> float:
    return x - math.floor(x)


def _hash01(x: float) -> float:
    return _fract(math.sin(x) * 43758.5453123)


def _noise2(x: float, y: float, seed: float) -> float:
    """2D value noise (0..1)."""
    ix = math.floor(x)
    iy = math.floor(y)
    fx = x - ix
    fy = y - iy

    def h(xx: float, yy: float) -> float:
        return _hash01(xx * 127.1 + yy * 311.7 + seed * 74.7)

    a = h(ix, iy)
    b = h(ix + 1.0, iy)
    c = h(ix, iy + 1.0)
    d = h(ix + 1.0, iy + 1.0)

    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)

    ab = a * (1.0 - ux) + b * ux
    cd = c * (1.0 - ux) + d * ux
    return ab * (1.0 - uy) + cd * uy


def _fbm2(x: float, y: float, seed: float, octaves: int = 4) -> float:
    """Fractal-ish noise (0..1)."""
    amp = 0.55
    freq = 1.0
    s = 0.0
    norm = 0.0
    for i in range(octaves):
        s += amp * _noise2(x * freq, y * freq, seed + i * 9.13)
        norm += amp
        amp *= 0.55
        freq *= 2.0
    if norm <= 1e-9:
        return 0.0
    return _clamp01(s / norm)


# -------------------- Heart silhouette + pulse -------------------------------

def _beat_wave_smooth(t: float) -> float:
    """
    Smooth "heartbeat" envelope: two bumps per cycle, but with extra easing so
    growth/decay feels less steppy on polygon mosaics.
    Returns 0..1.
    """
    cycle = 2.25
    p = (t / cycle) % 1.0

    def bump(phase: float, center: float, width: float) -> float:
        d = abs((phase - center + 0.5) % 1.0 - 0.5)
        if d >= width:
            return 0.0
        x = d / width
        # cosine bell (C1 continuous)
        return 0.5 * (1.0 + math.cos(math.pi * x))

    b1 = bump(p, 0.16, 0.11)
    b2 = bump(p, 0.33, 0.15)
    tail = 0.20 * (1.0 - _smoothstep(0.34, 0.98, p))

    raw = 0.04 + 0.50 * b1 + 1.00 * b2 + tail
    raw = _clamp01(raw)

    # extra easing to make growth/decay smoother visually
    eased = _cosine_ease(raw)
    # blend so it still has "punch", but smoother overall
    pulse = 0.55 * raw + 0.45 * eased
    return _clamp01(pulse)


def _heart_icon_val(x: float, y: float) -> float:
    # Classic implicit heart, slightly tuned (kept compatible with prior look),
    # BUT overall fitting is now anisotropic to fill the canvas like the reference.
    x = abs(x)
    x *= 1.12
    y *= 1.02
    y += 0.10
    a = x * x + y * y - 1.0
    return (a * a * a) - (x * x) * (y * y * y)


def _compute_heart_fit(margin: float = 0.004, headroom: float = (1.0 + _HEART_BEAT_AMP)) -> Dict[str, float]:
    """
    Compute a tight fit for the implicit heart in normalized space.

    CHANGE (requested):
    - Fit is now *anisotropic* (separate sx/sy) so the heart silhouette fills the canvas
      like the reference image, and stretches cleanly to match any output aspect ratio.
    - Fit accounts for max pulse headroom so it won't clip at peak beat.
    """
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
            if _heart_icon_val(x, y) <= 0.0:
                minx = min(minx, x)
                maxx = max(maxx, x)
                miny = min(miny, y)
                maxy = max(maxy, y)

    if not (minx < maxx and miny < maxy):
        # fallback
        return {"sx": 1.20, "sy": 1.20, "cx": 0.0, "cy": 0.0, "margin": margin, "headroom": headroom}

    cx = 0.5 * (minx + maxx)
    cy = 0.5 * (miny + maxy)
    w = (maxx - minx)
    h = (maxy - miny)

    # Target square in normalized coords is [-1,1] -> size 2.0
    # Use margin, and also reserve "headroom" for pulse scale so peak doesn't clip.
    target = 2.0 * (1.0 - margin) / max(1e-6, headroom)

    sx = target / max(1e-6, w)
    sy = target / max(1e-6, h)

    return {"sx": float(sx), "sy": float(sy), "cx": float(cx), "cy": float(cy), "margin": float(margin), "headroom": float(headroom)}


def _heart_mask_icon_single(nx: float, ny: float, pulse: float, fit: Dict[str, float]) -> Tuple[float, float]:
    xs = nx * 2.0 - 1.0
    ys = (1.0 - ny) * 2.0 - 1.0

    beat_scale = 1.0 + _HEART_BEAT_AMP * pulse
    sx = fit["sx"] * beat_scale
    sy = fit["sy"] * beat_scale

    x = (xs / sx) + fit["cx"]
    y = (ys / sy) + fit["cy"]

    val = _heart_icon_val(x, y)

    edge = 0.026  # slightly crisper than before, AA handles smoothness
    mask = 1.0 - _smoothstep(-edge, edge, val)
    glow = math.exp(-abs(val) * 9.5)
    return (_clamp01(mask), _clamp01(glow))


def _heart_mask_icon_aa(nx: float, ny: float, pulse: float, fit: Dict[str, float]) -> Tuple[float, float]:
    """
    Cheap AA for the heart boundary: sample multiple nearby points and average.
    This makes the growth (scale) feel smoother on a polygon-mosaic surface.
    """
    eps = 0.0026  # ~ "subpixel" in normalized coords for 1080p-ish output
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
        m, g = _heart_mask_icon_single(_clamp01(nx + dx), _clamp01(ny + dy), pulse, fit)
        m_sum += m
        g_sum += g

    inv = 1.0 / float(len(samples))
    return (_clamp01(m_sum * inv), _clamp01(g_sum * inv))


# -------------------- Champagne (bubbles) ------------------------------------

def _bubble_influence(nx: float, ny: float, bx: float, by: float, r: float) -> float:
    dx = nx - bx
    dy = ny - by
    d2 = dx * dx + dy * dy
    rr = max(1e-6, r * r)
    return math.exp(-d2 / (2.2 * rr))


# -------------------- MAIN ----------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a polygon animation video from glyph SVGs (single or NxN logo).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--char", type=str, default=None, help="Single character OR codepoint token (uXXXX / U+XXXX / XXXX).")
    g.add_argument(
        "--chars",
        type=str,
        default=None,
        help="Square count of chars: 4->2x2, 9->3x3, 16->4x4. Either literal characters (spaces ignored) or whitespace-separated codepoint tokens.",
    )

    ap.add_argument("--inverse", action="store_true",
                    help="Use src/inverse-*.svg instead of src/character-*.svg")

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
            "fire", "ice",
            "valentines",
            "matrix",
            "snow",
            "minecraft",
            "deidee",
            "heart",
            "static",
            "champagne",
            "camo",
            "fireworks",
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

    prefix = "inverse" if args.inverse else "character"
    parser = etree.XMLParser(remove_blank_text=False, recover=True, remove_comments=False)

    # ---------------- Input SVG(s) ----------------
    if args.char is not None:
        token, _cp, disp = _parse_char_or_codepoint(args.char)
        in_svg_path = _resolve_glyph_svg_path(src_dir, prefix, token, disp)
        if not in_svg_path.is_file():
            raise SystemExit(f"Input SVG not found: {in_svg_path}")

        doc = etree.fromstring(in_svg_path.read_bytes(), parser=parser)

        out_stem = f"{prefix}-{token}"
        out_file = out_dir / f"{out_stem}.{args.ext.lower()}"
        label = f"{prefix} {disp!r} ({token})"

    else:
        items = _parse_chars_arg(args.chars or "")
        if not items:
            raise SystemExit("--chars is empty after parsing.")

        n = len(items)
        grid_n = int(round(math.sqrt(n)))
        if grid_n * grid_n != n or grid_n not in (2, 3, 4):
            raise SystemExit("--chars must contain 4, 9, or 16 items (characters or codepoints) for 2x2 / 3x3 / 4x4 grids.")

        char_paths: List[Path] = []
        for token, _cp, disp in items:
            p = _resolve_glyph_svg_path(src_dir, prefix, token, disp)
            char_paths.append(p)

        doc = build_logo_svg_from_chars_grid(char_paths, grid_n=grid_n, gap_flag=args.gap)

        safe_key = _safe_logo_key(items)
        out_stem = f"logo-{'inv-' if args.inverse else ''}{safe_key}"
        out_file = out_dir / f"{out_stem}.{args.ext.lower()}"

        shown = "".join(d for _t, _cp, d in items)
        label = f"logo {shown!r} ({grid_n}x{grid_n}, gap={args.gap}, {prefix})"

    out_file = _timestamped_if_exists(out_file)

    # ---------------- ViewBox + background override ----------------
    minx, miny, vb_w, vb_h = _parse_viewbox(doc)
    if vb_w <= 0 or vb_h <= 0:
        vb_w, vb_h = 240.0, 240.0
    vb_tuple = (minx, miny, vb_w, vb_h)

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

    # ---------------- Polygons ----------------
    polys = doc.xpath('.//*[local-name()="polygon"]')
    if not polys:
        raise SystemExit("No <polygon> elements found in the input SVG(s).")

    rng = random.Random(args.seed)

    poly_nx: List[float] = []
    poly_ny: List[float] = []
    glyph_nx: List[float] = []
    glyph_ny: List[float] = []
    for p in polys:
        nx, ny = _global_centroid_norm(p, vb_tuple)
        poly_nx.append(nx)
        poly_ny.append(ny)
        gx, gy = _centroid_in_glyph_norm(p, vb_tuple)
        glyph_nx.append(gx)
        glyph_ny.append(gy)

    # Base colors (classic)
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

    # Output sizing
    max_dim = int(args.max_dim)
    scale = float(max_dim) / float(max(vb_w, vb_h))
    out_w = max(1, int(round(vb_w * scale)))
    out_h = max(1, int(round(vb_h * scale)))

    frames = int(round(float(args.duration) * int(args.fps)))
    fps = int(args.fps)

    # ---------------- Theme precompute ----------------
    poly_hsv: List[PolyHSV] = []
    if cfg.kind in {"diamond", "hsv"}:
        for idx in range(len(polys)):
            nx = poly_nx[idx]
            ny = poly_ny[idx]

            freq = rng.uniform(0.18, 0.85)
            phase = rng.uniform(0.0, 1.0)

            u = rng.random()
            if u < 0.18:
                v = (u / 0.18) * 0.10
            elif u > 0.82:
                v = 0.90 + ((u - 0.82) / 0.18) * 0.10
            else:
                v = 0.10 + ((u - 0.18) / 0.64) * 0.80

            v *= 0.90 + 0.20 * (0.5 + 0.5 * math.sin(2 * math.pi * (nx * 1.3 + ny * 0.7)))
            v = _clamp01(v)

            if cfg.kind == "diamond":
                h = (210/360.0) + rng.uniform(-0.01, 0.01)
                s = rng.uniform(0.00, 0.10)
                sat_mul = 1.0
                v_mul = rng.uniform(0.90, 1.10)
            else:
                if cfg.base_hue is None:
                    h = (nx + rng.uniform(-0.05, 0.05)) % 1.0
                else:
                    h = (cfg.base_hue
                         + cfg.hue_jitter * rng.uniform(-1.0, 1.0)
                         + cfg.hue_tone_amp * (nx - 0.5)) % 1.0

                s = rng.uniform(cfg.body_sat_min, cfg.body_sat_max)
                sat_mul = rng.uniform(0.85, 1.25)

                ug = rng.random()
                vg = cfg.body_v_min + (cfg.body_v_max - cfg.body_v_min) * (ug ** cfg.body_v_gamma)
                vg = _clamp01(vg + 0.10 * (0.5 - ny))
                v = vg
                v_mul = rng.uniform(0.92, 1.12)

            fire_enabled = (rng.random() < cfg.fire_prob)
            if cfg.fire_hues:
                fh = rng.choice(cfg.fire_hues)
            else:
                fh = (h + rng.uniform(-0.10, 0.10)) % 1.0

            fire_sat_mul = rng.uniform(cfg.fire_sat_mul_lo, cfg.fire_sat_mul_hi)
            fire_white_mix = rng.uniform(cfg.fire_white_mix_min, cfg.fire_white_mix_max)

            poly_hsv.append(PolyHSV(
                h=h, s=s, v=v,
                sat_mul=sat_mul,
                v_mul=v_mul,
                freq=freq, phase=phase,
                fire_enabled=fire_enabled,
                fire_hue=(fh + rng.uniform(-cfg.fire_hue_jitter, cfg.fire_hue_jitter)) % 1.0,
                fire_sat_mul=fire_sat_mul,
                fire_white_mix=fire_white_mix,
            ))

    # Minecraft
    mc_pixels: List[Tuple[int, int, int]] = []
    mc_w = mc_h = 0
    mc_u: List[float] = []
    mc_v: List[float] = []
    mc_freq: List[float] = []
    mc_phase: List[float] = []
    if cfg.kind == "minecraft":
        mc_pixels, mc_w, mc_h = _load_minecraft_texture_16x16(args.minecraft_texture)
        for idx in range(len(polys)):
            u = glyph_nx[idx]
            v = glyph_ny[idx]
            mc_u.append(u)
            mc_v.append(v)
            mc_freq.append(rng.uniform(0.10, 0.45))
            mc_phase.append(rng.uniform(0.0, 1.0))

    # deidee
    de_alpha = 0.5
    de_colors_per_poly: List[List[Tuple[int, int, int]]] = []
    de_seg_dur: List[float] = []
    de_phase: List[float] = []
    if cfg.kind == "deidee":
        for _ in polys:
            k = rng.randint(4, 8)
            cols: List[Tuple[int, int, int]] = []
            for _j in range(k):
                r = int(round(rng.uniform(0.0, 0.5) * 255.0))
                g = int(round(rng.uniform(0.5, 1.0) * 255.0))
                b = int(round(rng.uniform(0.0, 0.75) * 255.0))
                cols.append((max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))))
            de_colors_per_poly.append(cols)
            de_seg_dur.append(rng.uniform(0.90, 2.60))
            de_phase.append(rng.uniform(0.0, 10.0))
        for poly in polys:
            poly.set("fill-opacity", f"{de_alpha:.3f}")
            st = (poly.get("style") or "").strip()
            poly.set("style", _style_set(st, "fill-opacity", f"{de_alpha:.3f}"))

    # heart (UPDATED FIT)
    heart_fit: Optional[Dict[str, float]] = None
    heart_hj: List[float] = []
    heart_sj: List[float] = []
    heart_tw_f: List[float] = []
    heart_tw_p: List[float] = []
    heart_fire_h: List[float] = []
    if cfg.kind == "heart":
        heart_fit = _compute_heart_fit(margin=0.004, headroom=(1.0 + _HEART_BEAT_AMP))
        fire_hues = [350/360.0, 0/360.0, 10/360.0, 330/360.0, 315/360.0, 45/360.0]  # valentines-like
        for _ in polys:
            heart_hj.append(rng.uniform(-0.030, 0.030))
            heart_sj.append(rng.uniform(0.92, 1.22))
            heart_tw_f.append(rng.uniform(0.22, 0.95))
            heart_tw_p.append(rng.uniform(0.0, 1.0))
            heart_fire_h.append((rng.choice(fire_hues) + rng.uniform(-0.06, 0.06)) % 1.0)

    # static
    static_seg: List[float] = []
    static_phase: List[float] = []
    static_color_prob: List[float] = []
    static_seedf = float(args.seed if args.seed is not None else rng.randint(0, 10_000_000))
    if cfg.kind == "static":
        for _ in polys:
            static_seg.append(rng.uniform(0.05, 0.13))
            static_phase.append(rng.uniform(0.0, 10.0))
            static_color_prob.append(rng.uniform(0.06, 0.14))

    # matrix
    col_count = max(14, min(40, int(round(math.sqrt(len(polys)) * 4))))
    col_drop: List[MatrixDrop] = []
    if cfg.kind == "matrix":
        for _c in range(col_count):
            col_drop.append(MatrixDrop(
                speed=rng.uniform(0.10, 0.32),
                phase=rng.uniform(0.0, 1.0),
                tail=rng.uniform(0.18, 0.55),
                head=rng.uniform(0.02, 0.06),
                strength=rng.uniform(0.65, 1.00),
                flicker_freq=rng.uniform(3.0, 9.0),
                flicker_phase=rng.uniform(0.0, 1.0),
            ))

    # champagne (unchanged)
    bubbles: List[Bubble] = []
    ch_freq: List[float] = []
    ch_phase: List[float] = []
    if cfg.kind == "champagne":
        n_bub = min(90, max(24, int(round(math.sqrt(len(polys)) * 10))))
        stream_x = [rng.uniform(0.12, 0.88) for _ in range(rng.randint(4, 7))]
        for _ in range(n_bub):
            sx = rng.choice(stream_x)
            x = _clamp01(sx + rng.uniform(-0.06, 0.06))
            bubbles.append(Bubble(
                x=x,
                y0=rng.uniform(0.0, 1.0),
                r=rng.uniform(0.010, 0.040) * rng.uniform(0.8, 1.3),
                speed=rng.uniform(0.08, 0.26),
                wob_amp=rng.uniform(0.005, 0.020),
                wob_freq=rng.uniform(0.20, 0.80),
                wob_phase=rng.uniform(0.0, 1.0),
                strength=rng.uniform(0.35, 1.00),
            ))
        for _ in polys:
            ch_freq.append(rng.uniform(0.05, 0.22))
            ch_phase.append(rng.uniform(0.0, 1.0))

    # camo (Scorpion W2) precompute
    camo_seed = float(args.seed if args.seed is not None else rng.randint(0, 10_000_000))
    camo_offx: List[float] = []
    camo_offy: List[float] = []
    camo_phase: List[float] = []
    if cfg.kind == "camo":
        for _ in polys:
            camo_offx.append(rng.uniform(0.0, 1000.0))
            camo_offy.append(rng.uniform(0.0, 1000.0))
            camo_phase.append(rng.uniform(0.0, 1.0))

    # fireworks precompute
    fireworks: List[Firework] = []
    if cfg.kind == "fireworks":
        dur = float(args.duration)
        n_fw = max(8, min(18, int(round(dur * 1.15))))
        for _ in range(n_fw):
            t_burst = rng.uniform(1.0, max(1.2, dur - 0.8))
            launch_lead = rng.uniform(0.55, 1.40)
            t_launch = max(0.0, t_burst - launch_lead)

            x = rng.uniform(0.08, 0.92)
            yb = rng.uniform(0.14, 0.55)

            vel = rng.uniform(0.22, 0.50)
            ring_w = rng.uniform(0.018, 0.050)
            decay = rng.uniform(1.00, 2.20)

            spoke_n = rng.randint(8, 18)
            spoke_phase = rng.uniform(0.0, 2.0 * math.pi)

            hue_a = rng.choice([0/360.0, 20/360.0, 45/360.0, 120/360.0, 190/360.0, 220/360.0, 280/360.0, 315/360.0])
            hue_b = (hue_a + rng.uniform(0.08, 0.22)) % 1.0

            glitter_f = rng.uniform(3.0, 8.5)
            glitter_p = rng.uniform(0.0, 1.0)

            trail_len = rng.uniform(0.20, 0.40)
            trail_w = rng.uniform(0.010, 0.030)
            trail_h = rng.uniform(18/360.0, 55/360.0)  # warm “fire arrow” trail

            fireworks.append(Firework(
                x=x, yb=yb,
                t_launch=t_launch, t_burst=t_burst,
                vel=vel, ring_w=ring_w, decay=decay,
                spoke_n=spoke_n, spoke_phase=spoke_phase,
                hue_a=hue_a, hue_b=hue_b,
                glitter_f=glitter_f, glitter_p=glitter_p,
                trail_len=trail_len, trail_w=trail_w, trail_h=trail_h,
            ))

    # ---------------- Render frames ----------------
    tmp_root = Path(tempfile.mkdtemp(prefix="mmxx_video_"))
    frames_dir = tmp_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Scorpion W2 / OCP-like palette (approx, tuned to “feel right” in motion)
    CAMO_PAL: List[Tuple[int, int, int]] = [
        _parse_css_color_to_rgb("#1b2116"),  # near-black green
        _parse_css_color_to_rgb("#2f3b2a"),  # dark green
        _parse_css_color_to_rgb("#4b5636"),  # olive
        _parse_css_color_to_rgb("#6b6a45"),  # khaki green
        _parse_css_color_to_rgb("#8a7751"),  # tan
        _parse_css_color_to_rgb("#b2a378"),  # light tan
        _parse_css_color_to_rgb("#3c2f20"),  # dark brown
    ]
    # Sorted from darkest-ish to lightest-ish already; we’ll index by thresholds.

    try:
        renderer_used = None

        for i in range(frames):
            t = i / float(fps)

            amb = cfg.amb_base + cfg.amb_amp * (0.5 + 0.5 * math.sin(2.0 * math.pi * (cfg.amb_freq * t)))

            if cfg.kind == "classic":
                for idx, poly in enumerate(polys):
                    a = whiteness_at(t, pulses_per_poly[idx])
                    poly.set("fill", _rgb_to_hex(mix_to_white(base_rgbs[idx], a)))

            elif cfg.kind == "diamond":
                for idx, poly in enumerate(polys):
                    ph = poly_hsv[idx]
                    pulse = whiteness_at(t, pulses_per_poly[idx])
                    shim = facet_shimmer(t, ph.freq, ph.phase)

                    glint = max(cfg.gl_pulse_w * pulse, cfg.gl_shim_w * shim)
                    spec = _smoothstep(cfg.spec_edge0, 1.0, glint)
                    spec = _clamp01(spec * cfg.spec_scale)

                    g0 = _clamp01((ph.v * ph.v_mul))
                    g = _clamp01((g0 - 0.5) * 1.20 + 0.5)
                    grey = int(round(g * 255.0))
                    rgb = (grey, grey, grey)

                    rgb = _mix_rgb(rgb, (255, 255, 255), 0.18 + 0.72 * spec)

                    if ph.fire_enabled:
                        gate = _smoothstep(cfg.fire_gate0, 1.0, glint)
                        if gate > 0.001:
                            drift = cfg.fire_hue_drift_amp * math.sin(2.0 * math.pi * (cfg.fire_hue_drift_freq * t + ph.phase))
                            fh = (ph.fire_hue + drift) % 1.0
                            sat_base = cfg.fire_sat_base_min + (cfg.fire_sat_base_max - cfg.fire_sat_base_min) * (0.5 + 0.5 * math.sin(2*math.pi*(0.15*t + ph.phase)))
                            sat_peak = cfg.fire_sat_peak_min + (cfg.fire_sat_peak_max - cfg.fire_sat_peak_min) * (0.5 + 0.5 * math.sin(2*math.pi*(0.22*t + ph.freq)))
                            s = _clamp01((sat_base + gate * sat_peak) * ph.fire_sat_mul)
                            v = _clamp01(0.55 + 0.45 * gate)
                            fire_rgb = _hsv01_to_rgb255(fh, s, v)
                            mix_amt = _clamp01(gate * (0.20 + 0.55 * ph.fire_white_mix))
                            rgb = _mix_rgb(rgb, fire_rgb, mix_amt)

                    rgb = mix_to_white(rgb, amb * 0.20)
                    poly.set("fill", _rgb_to_hex(rgb))

            elif cfg.kind == "hsv":
                for idx, poly in enumerate(polys):
                    ph = poly_hsv[idx]
                    pulse = whiteness_at(t, pulses_per_poly[idx])
                    shim = facet_shimmer(t, ph.freq, ph.phase)

                    drift_h = cfg.hue_shimmer_amp * math.sin(2.0 * math.pi * (0.10 * t + ph.phase))
                    h = (ph.h + drift_h) % 1.0

                    v = _clamp01((ph.v * ph.v_mul) + cfg.val_shimmer_amp * (shim - 0.5))
                    v = _clamp01(0.06 + 0.90 * v)

                    s = ph.s * cfg.body_sat_mul * ph.sat_mul
                    s = _clamp01(s + cfg.sat_dark_boost * (1.0 - v))

                    rgb = _hsv01_to_rgb255(h, s, v)

                    glint = max(cfg.gl_pulse_w * pulse, cfg.gl_shim_w * shim)
                    spec = _smoothstep(cfg.spec_edge0, 1.0, glint)
                    spec = _clamp01(spec * cfg.spec_scale)

                    sh = (h + cfg.sheen_hue_shift) % 1.0
                    ss = _clamp01(s + cfg.sheen_sat_boost)
                    sheen_rgb = _hsv01_to_rgb255(sh, ss, 1.0)
                    rgb = _mix_rgb(rgb, sheen_rgb, cfg.sheen_mix * spec)

                    rgb = _mix_rgb(rgb, (255, 255, 255), 0.06 * spec)

                    if ph.fire_enabled:
                        gate = _smoothstep(cfg.fire_gate0, 1.0, glint)
                        if gate > 0.001:
                            drift = cfg.fire_hue_drift_amp * math.sin(2.0 * math.pi * (cfg.fire_hue_drift_freq * t + ph.phase))
                            fh = (ph.fire_hue + drift) % 1.0
                            sat_base = cfg.fire_sat_base_min + (cfg.fire_sat_base_max - cfg.fire_sat_base_min) * (0.5 + 0.5 * math.sin(2*math.pi*(0.15*t + ph.phase)))
                            sat_peak = cfg.fire_sat_peak_min + (cfg.fire_sat_peak_max - cfg.fire_sat_peak_min) * (0.5 + 0.5 * math.sin(2*math.pi*(0.22*t + ph.freq)))
                            s2 = _clamp01((sat_base + gate * sat_peak) * ph.fire_sat_mul)
                            v2 = _clamp01(0.55 + 0.45 * gate)
                            fire_rgb = _hsv01_to_rgb255(fh, s2, v2)
                            mix_amt = _clamp01(gate * 0.45)
                            rgb = _mix_rgb(rgb, fire_rgb, mix_amt)

                    rgb = mix_to_white(rgb, amb * 0.10)
                    poly.set("fill", _rgb_to_hex(rgb))

            elif cfg.kind == "minecraft":
                for idx, poly in enumerate(polys):
                    u = mc_u[idx]
                    v = mc_v[idx]
                    px = int(round(u * (mc_w - 1)))
                    py = int(round(v * (mc_h - 1)))
                    base = mc_pixels[py * mc_w + px]

                    shim = facet_shimmer(t, mc_freq[idx], mc_phase[idx])
                    pulse = whiteness_at(t, pulses_per_poly[idx]) if pulses_per_poly[idx] else 0.0

                    wob = 0.92 + 0.12 * (0.5 + 0.5 * math.sin(2.0 * math.pi * (0.07 * t + mc_phase[idx])))
                    rgb = (
                        max(0, min(255, int(round(base[0] * wob)))),
                        max(0, min(255, int(round(base[1] * wob)))),
                        max(0, min(255, int(round(base[2] * wob)))),
                    )

                    glint = max(cfg.gl_shim_w * shim, cfg.gl_pulse_w * pulse)
                    spec = _smoothstep(cfg.spec_edge0, 1.0, glint) * cfg.spec_scale

                    if spec > 0.0:
                        sun = _hsv01_to_rgb255(45/360.0, 0.18, 1.0)
                        rgb = _mix_rgb(rgb, sun, 0.12 * spec)
                        rgb = _mix_rgb(rgb, (255, 255, 255), 0.28 * spec)

                    rgb = mix_to_white(rgb, amb * 0.06)
                    poly.set("fill", _rgb_to_hex(rgb))

            elif cfg.kind == "deidee":
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

                    poly.set("fill-opacity", f"{de_alpha:.3f}")
                    st = (poly.get("style") or "").strip()
                    poly.set("style", _style_set(st, "fill-opacity", f"{de_alpha:.3f}"))

            elif cfg.kind == "static":
                for idx, poly in enumerate(polys):
                    seg = static_seg[idx]
                    ph = static_phase[idx]
                    pos = (t + ph) / seg
                    k0 = int(math.floor(pos))
                    f = pos - math.floor(pos)
                    u = _cosine_ease(f)

                    a0 = _hash01(static_seedf * 0.001 + idx * 12.9898 + k0 * 78.233)
                    a1 = _hash01(static_seedf * 0.001 + idx * 12.9898 + (k0 + 1) * 78.233)
                    a = (1.0 - u) * a0 + u * a1

                    a = _clamp01((a - 0.5) * 1.85 + 0.5)
                    vv = 0.06 + 0.94 * a

                    scan = 0.95 + 0.05 * math.sin(2.0 * math.pi * (poly_ny[idx] * 90.0 + t * 1.25))
                    vv = _clamp01(vv * scan)

                    csel0 = _hash01(static_seedf * 0.002 + idx * 3.11 + k0 * 9.73)
                    csel1 = _hash01(static_seedf * 0.002 + idx * 3.11 + (k0 + 1) * 9.73)
                    csel = (1.0 - u) * csel0 + u * csel1

                    if csel < static_color_prob[idx]:
                        h0 = _hash01(static_seedf * 0.003 + idx * 0.77 + k0 * 2.17)
                        h1 = _hash01(static_seedf * 0.003 + idx * 0.77 + (k0 + 1) * 2.17)
                        h = ((1.0 - u) * h0 + u * h1) % 1.0
                        s = 0.55 + 0.45 * _hash01(static_seedf * 0.004 + idx * 1.33 + k0 * 6.19)
                        rgb = _hsv01_to_rgb255(h, s, 0.25 + 0.75 * vv)
                    else:
                        g = int(round(vv * 255.0))
                        rgb = (g, g, g)

                    poly.set("fill", _rgb_to_hex(rgb))

            elif cfg.kind == "matrix":
                for idx, poly in enumerate(polys):
                    nx = poly_nx[idx]
                    ny = poly_ny[idx]
                    c = int(_clamp01(nx) * (col_count - 1))
                    d = col_drop[c]

                    head = (d.phase + d.speed * t) % 1.0
                    dist = (ny - head) % 1.0

                    if dist <= d.head:
                        inten = 1.0
                    elif dist <= d.tail:
                        z = 1.0 - (dist - d.head) / max(1e-6, (d.tail - d.head))
                        inten = (z * z)
                    else:
                        inten = 0.0

                    fl = 0.72 + 0.28 * math.sin(2.0 * math.pi * (d.flicker_freq * t + d.flicker_phase))
                    inten = _clamp01(inten * d.strength * fl)

                    bg_v = 0.02 + 0.04 * (0.5 + 0.5 * math.sin(2.0 * math.pi * (0.18 * t + nx * 1.7 + ny * 0.9)))
                    bg = _hsv01_to_rgb255(120/360.0, 0.55, bg_v)

                    if inten <= 0.0:
                        rgb = bg
                    else:
                        headness = _smoothstep(0.75, 1.0, inten)
                        g_v = 0.10 + 0.90 * inten
                        green = _hsv01_to_rgb255(120/360.0, 1.0, g_v)
                        rgb = _mix_rgb(bg, green, 0.85)
                        rgb = _mix_rgb(rgb, (255, 255, 255), 0.40 * headness)

                    poly.set("fill", _rgb_to_hex(rgb))

            elif cfg.kind == "champagne":
                for idx, poly in enumerate(polys):
                    nx = poly_nx[idx]
                    ny = poly_ny[idx]

                    h = (45/360.0) + 0.010 * math.sin(2.0 * math.pi * (0.04 * t + ch_phase[idx]))
                    s = 0.18 + 0.12 * (0.5 + 0.5 * math.sin(2.0 * math.pi * (ch_freq[idx] * t + ch_phase[idx])))
                    v = 0.30 + 0.55 * (1.0 - ny)
                    v += 0.06 * (facet_shimmer(t, 0.12 + ch_freq[idx], ch_phase[idx]) - 0.5)
                    v = _clamp01(v)

                    rgb = _hsv01_to_rgb255(h, s, v)

                    bub = 0.0
                    bub_edge = 0.0
                    for b in bubbles:
                        by = (b.y0 - b.speed * t) % 1.0
                        bx = _clamp01(b.x + b.wob_amp * math.sin(2.0 * math.pi * (b.wob_freq * t + b.wob_phase)))
                        infl = _bubble_influence(nx, ny, bx, by, b.r)
                        if infl > 1e-6:
                            bub = max(bub, infl * b.strength)
                            infl2 = _bubble_influence(nx, ny, bx, by, b.r * 0.60)
                            bub_edge = max(bub_edge, infl2 * b.strength)

                    bub = _clamp01(bub)
                    bub_edge = _clamp01(bub_edge)

                    if bub > 0.0:
                        bubble_rgb = _hsv01_to_rgb255(200/360.0, 0.08, 1.0)
                        rgb = _mix_rgb(rgb, bubble_rgb, 0.70 * bub)
                        rgb = _mix_rgb(rgb, (255, 255, 255), 0.55 * bub_edge)

                        tint_gate = _smoothstep(0.30, 0.90, bub_edge)
                        if tint_gate > 0.0:
                            fh = (0.10 + 0.20 * math.sin(2.0 * math.pi * (0.06 * t + nx))) % 1.0
                            tint = _hsv01_to_rgb255(fh, 0.55, 1.0)
                            rgb = _mix_rgb(rgb, tint, 0.10 * tint_gate)

                    rgb = mix_to_white(rgb, amb * 0.08)
                    poly.set("fill", _rgb_to_hex(rgb))

            elif cfg.kind == "camo":
                # Scorpion W2-ish: macro blobs + micro grain + gentle drift.
                for idx, poly in enumerate(polys):
                    nx = poly_nx[idx]
                    ny = poly_ny[idx]

                    # subtle drift so it feels like “fabric” moving
                    drift_x = 0.020 * math.sin(2.0 * math.pi * (0.035 * t + camo_phase[idx]))
                    drift_y = 0.018 * math.cos(2.0 * math.pi * (0.030 * t + camo_phase[idx] * 0.7))

                    # coordinate space (macro + micro)
                    x = (nx * 3.2 + drift_x) + camo_offx[idx] * 0.001
                    y = (ny * 3.2 + drift_y) + camo_offy[idx] * 0.001

                    macro = _fbm2(x * 0.85, y * 0.85, camo_seed + 1.7, octaves=4)
                    mid   = _fbm2(x * 2.10, y * 2.10, camo_seed + 7.9, octaves=3)
                    micro = _fbm2(x * 9.00, y * 9.00, camo_seed + 13.3, octaves=2)

                    n = _clamp01(0.62 * macro + 0.28 * mid + 0.10 * micro)

                    # “fabric” shading (very subtle)
                    shade = 0.90 + 0.10 * math.sin(2.0 * math.pi * (0.045 * t + nx * 2.1 + ny * 1.6))
                    shade *= 0.94 + 0.06 * (0.5 + 0.5 * math.sin(2.0 * math.pi * (0.11 * t + camo_phase[idx])))
                    shade = _clamp01(shade)

                    # thresholds across palette; blend in a small band for organic edges
                    th = [0.12, 0.24, 0.40, 0.56, 0.72, 0.86]
                    bw = 0.045

                    def pick(nv: float) -> Tuple[int, int, int]:
                        if nv <= th[0]:
                            return CAMO_PAL[0]
                        if nv >= th[-1]:
                            return CAMO_PAL[-1]
                        # find interval
                        k = 0
                        while k < len(th) and nv > th[k]:
                            k += 1
                        # palette index mapping (7 colors)
                        # interval k means between palette k and k+1-ish, but keep ends stable
                        a = max(0, min(len(CAMO_PAL) - 2, k))
                        b = a + 1
                        edge0 = th[a] if a < len(th) else th[-1]
                        edge1 = th[b] if b < len(th) else th[-1]
                        # broaden blend a bit
                        t0 = _smoothstep(edge0 - bw, edge0 + bw, nv)
                        t1 = _smoothstep(edge1 - bw, edge1 + bw, nv)
                        # blend around the local boundary; if we're mid-interval use t1, else t0
                        tt = _clamp01((t0 + t1) * 0.5)
                        return _mix_rgb(CAMO_PAL[a], CAMO_PAL[b], tt)

                    rgb = pick(n)

                    # add a tiny high-frequency grain so polygons don’t look too flat
                    g = _noise2(nx * 120.0 + t * 0.35, ny * 120.0 + t * 0.27, camo_seed + 99.1)
                    grain = (g - 0.5) * 0.10  # +/- 0.05
                    rgb = (
                        max(0, min(255, int(round(rgb[0] * (shade + grain))))),
                        max(0, min(255, int(round(rgb[1] * (shade + grain))))),
                        max(0, min(255, int(round(rgb[2] * (shade + grain))))),
                    )

                    # very mild ambient lift (keep matte)
                    rgb = mix_to_white(rgb, amb * 0.04)
                    poly.set("fill", _rgb_to_hex(rgb))

            elif cfg.kind == "fireworks":
                # Dark sky base + repeated “fire arrow” launches + bursts.
                for idx, poly in enumerate(polys):
                    nx = poly_nx[idx]
                    ny = poly_ny[idx]

                    # sky gradient (darker at bottom, slightly brighter toward top)
                    sky_h = 215/360.0 + 0.010 * math.sin(2.0 * math.pi * (0.02 * t + nx * 0.6))
                    sky_s = 0.55
                    sky_v = 0.03 + 0.10 * (1.0 - _smoothstep(0.10, 1.00, ny))**1.4
                    sky_v += 0.015 * math.sin(2.0 * math.pi * (0.05 * t + nx * 1.1 + ny * 0.9))
                    sky_v = _clamp01(sky_v)
                    rgb = _hsv01_to_rgb255(sky_h, sky_s, sky_v)

                    for fw in fireworks:
                        # rocket phase
                        if fw.t_launch <= t < fw.t_burst:
                            u = (t - fw.t_launch) / max(1e-6, (fw.t_burst - fw.t_launch))
                            u = _clamp01(u)

                            # ease upward motion slightly
                            uu = _cosine_ease(u)
                            y0 = 1.05
                            y = y0 + (fw.yb - y0) * uu

                            dx = abs(nx - fw.x)
                            dy = ny - y  # positive below the head

                            # trail behind head
                            if dy >= 0.0 and dy <= fw.trail_len:
                                trail_core = math.exp(-(dx * dx) / (2.0 * fw.trail_w * fw.trail_w))
                                trail_len_gate = math.exp(-(dy * dy) / (2.0 * (fw.trail_len * 0.55) ** 2))
                                trail = trail_core * trail_len_gate
                                trail = _clamp01(trail)

                                # warm “fire arrow” gradient (hotter near head)
                                hot = _clamp01(1.0 - dy / max(1e-6, fw.trail_len))
                                th = (fw.trail_h + 0.02 * math.sin(2.0 * math.pi * (0.20 * t + fw.glitter_p))) % 1.0
                                ts = 0.85
                                tv = 0.35 + 0.65 * hot
                                trail_rgb = _hsv01_to_rgb255(th, ts, tv)

                                # occasional bright spark points along the trail
                                spark = _smoothstep(0.70, 1.00, hot) * (0.65 + 0.35 * math.sin(2.0 * math.pi * (fw.glitter_f * t + fw.glitter_p + dy * 3.0)))
                                trail_rgb = _mix_rgb(trail_rgb, (255, 255, 255), 0.20 * _clamp01(spark))

                                rgb = _add_rgb(rgb, trail_rgb, 0.90 * trail)

                            # head glow
                            head_r = 0.020
                            d2 = (nx - fw.x) ** 2 + (ny - y) ** 2
                            head = math.exp(-d2 / (2.0 * head_r * head_r))
                            if head > 1e-5:
                                head_rgb = _hsv01_to_rgb255((fw.trail_h + 0.01) % 1.0, 0.30, 1.0)
                                rgb = _add_rgb(rgb, head_rgb, 0.95 * _clamp01(head))

                        # burst phase
                        if t >= fw.t_burst:
                            dt = t - fw.t_burst
                            if dt <= 4.0:
                                # expanding ring
                                cx = fw.x
                                cy = fw.yb
                                dx = nx - cx
                                dy = ny - cy
                                d = math.sqrt(dx*dx + dy*dy)

                                r = fw.vel * dt
                                ring = math.exp(-((d - r) * (d - r)) / (2.0 * fw.ring_w * fw.ring_w))
                                ring *= math.exp(-dt / max(1e-6, fw.decay))
                                ring = _clamp01(ring)

                                if ring > 1e-5:
                                    ang = math.atan2(dy, dx)

                                    # spokes (spark “arrows”)
                                    sp = abs(math.sin(fw.spoke_n * ang + fw.spoke_phase))
                                    sp = sp ** 2.2
                                    spoke_gate = 0.35 + 0.65 * sp

                                    # flicker
                                    flick = 0.70 + 0.30 * math.sin(2.0 * math.pi * (fw.glitter_f * t + fw.glitter_p + (ang * 0.07)))
                                    flick = _clamp01(flick)

                                    inten = ring * spoke_gate * flick

                                    # alternate hues per spoke for more “firework” variety
                                    sel = 0.5 + 0.5 * math.sin(fw.spoke_n * ang + fw.spoke_phase)
                                    h = fw.hue_a if sel >= 0.0 else fw.hue_b
                                    # hotter/whiter near the origin early
                                    core = math.exp(-(d*d) / (2.0 * (0.06 + 0.03 * dt) ** 2))
                                    core = _clamp01(core)

                                    s = _clamp01(0.70 + 0.30 * sp)
                                    v = _clamp01(0.25 + 0.75 * inten)
                                    burst_rgb = _hsv01_to_rgb255(h, s, v)

                                    burst_rgb = _mix_rgb(burst_rgb, (255, 255, 255), 0.25 * _clamp01(inten + 0.6 * core))

                                    rgb = _add_rgb(rgb, burst_rgb, 0.95 * inten)

                                    # lingering ember haze
                                    haze = _clamp01(0.22 * ring * (0.5 + 0.5 * math.sin(2.0 * math.pi * (0.45 * t + fw.glitter_p))))
                                    if haze > 1e-5:
                                        ember = _hsv01_to_rgb255((h + 0.02) % 1.0, 0.55, 0.60)
                                        rgb = _add_rgb(rgb, ember, haze)

                    # tiny ambient so sky isn’t crushed
                    rgb = mix_to_white(rgb, amb * 0.05)
                    poly.set("fill", _rgb_to_hex(rgb))

            elif cfg.kind == "heart":
                pulse = _beat_wave_smooth(t)
                assert heart_fit is not None

                # Valentines-like palette bias (pinks/reds) + a little golden sparkle.
                base_h = (override_hsv[0] if override_hsv is not None else (335 / 360.0))

                for idx, poly in enumerate(polys):
                    nx = poly_nx[idx]
                    ny = poly_ny[idx]

                    # AA mask for smoother growth/edges on mosaic polygons
                    mask, glow = _heart_mask_icon_aa(nx, ny, pulse, heart_fit)

                    # Background: deep plum/charcoal with a soft vignette (so heart pops)
                    bg_h = (300/360.0) + 0.010 * math.sin(2.0 * math.pi * (0.03 * t + nx * 0.7))
                    bg_s = 0.35
                    vign = math.sqrt((nx - 0.5) ** 2 + (ny - 0.5) ** 2)
                    bg_v = 0.05 + 0.07 * (1.0 - _smoothstep(0.15, 0.95, vign))
                    bg_v += 0.02 * math.sin(2.0 * math.pi * (0.05 * t + nx * 1.1 + ny * 0.6))
                    bg_v = _clamp01(bg_v)
                    bg_rgb = _hsv01_to_rgb255(bg_h, bg_s, bg_v)

                    # Heart base hue: valentines-ish sweep with jitter
                    h = (base_h
                         + 0.090 * (nx - 0.5)
                         + 0.040 * math.sin(2.0 * math.pi * (0.09 * t + ny * 0.9))
                         + heart_hj[idx]) % 1.0

                    # Saturation: generally high like valentines, but with random breathing so it doesn't feel flat
                    sat_breathe = 0.85 + 0.30 * (0.5 + 0.5 * math.sin(2.0 * math.pi * (0.12 * t + heart_tw_p[idx])))
                    s = (0.72 + 0.30 * mask + 0.22 * glow) * heart_sj[idx] * sat_breathe
                    s = _clamp01(s)

                    # Value: brighter "whitespace" inside heart, especially during the beat; keep edges very bright
                    v = 0.18 + 0.46 * mask
                    v += 0.34 * mask * (0.18 + 0.82 * pulse)
                    v += 0.22 * glow * (0.30 + 0.70 * pulse)
                    v = _clamp01(v)

                    heart_rgb = _hsv01_to_rgb255(h, s, v)
                    rgb = _mix_rgb(bg_rgb, heart_rgb, mask)

                    # Edge + internal sheen: pink-white + occasional warm sparkle (valentines vibe)
                    edge = _clamp01(glow * (0.22 + 0.40 * pulse))
                    rgb = _mix_rgb(rgb, (255, 255, 255), 0.40 * edge)

                    # Add a tinted "fire" sparkle hue on strong edge glow (like valentines dispersion)
                    fire_gate = _smoothstep(0.35, 0.92, edge)
                    if fire_gate > 0.0:
                        fh = (heart_fire_h[idx] + 0.02 * math.sin(2.0 * math.pi * (0.06 * t + nx))) % 1.0
                        fire_rgb = _hsv01_to_rgb255(fh, _clamp01(0.65 + 0.35 * s), 1.0)
                        rgb = _mix_rgb(rgb, fire_rgb, 0.18 * fire_gate)

                    # Twinkles (more valentine-like: whiter heads + colored flare)
                    tw = facet_shimmer(t, heart_tw_f[idx], heart_tw_p[idx])
                    tw = (mask ** 1.55) * (tw ** 1.30)
                    sparkle = _clamp01(tw * (0.06 + 0.22 * pulse) + (mask ** 2.2) * (0.02 + 0.06 * pulse))
                    if sparkle > 0.0:
                        rgb = _mix_rgb(rgb, (255, 255, 255), sparkle)
                        # subtle colored rim on sparkles
                        fh2 = (heart_fire_h[idx] + 0.015 * math.sin(2.0 * math.pi * (0.08 * t + ny))) % 1.0
                        flare = _hsv01_to_rgb255(fh2, 0.85, 1.0)
                        rgb = _mix_rgb(rgb, flare, 0.10 * sparkle)

                    rgb = mix_to_white(rgb, amb * 0.10)
                    poly.set("fill", _rgb_to_hex(rgb))

            else:
                for idx, poly in enumerate(polys):
                    a = whiteness_at(t, pulses_per_poly[idx])
                    poly.set("fill", _rgb_to_hex(mix_to_white(base_rgbs[idx], a)))

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
