#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/generate-video.py

Generate a 12s video from either:
  - --char  -> src/character-{char}.svg
  - --chars -> combines 4x src/character-{c}.svg into a single SVG (2x2 layout: [0 1; 2 3])

Logo layout (for --chars):
- Always 2 by 2: [0 1; 2 3]
- --gap 0 (default): no extra spacing
- --gap 1: spacing AND outer padding = 1/8th of a character cell size
    - i.e. gap_x = max_glyph_width/8, gap_y = max_glyph_height/8

Also in --chars mode:
- Removes per-glyph full-canvas WHITE background rects (so you don’t get 4 white squares)

Animation themes:
- classic (default):
    Every polygon pulses toward white and back to its base color.
- diamond:
    Diamond-like scintillation:
      - strong grey contrast (darker + lighter greys)
      - slow shimmer and slow glints
      - bright specular flashes
      - dispersion color only in highlights (tinted-white), with RANDOM saturation per facet

Output:
  dist/videos/character-{char}.{ext}
  dist/videos/logo-{chars}.{ext}

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
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from lxml import etree  # py -m pip install lxml

try:
    import cairosvg  # py -m pip install cairosvg
except Exception:
    cairosvg = None

try:
    from PIL import ImageColor  # py -m pip install pillow
except Exception:
    ImageColor = None


SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


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


def _resolve_fill(el: etree._Element) -> Optional[str]:
    """
    Attempt to resolve a fill value by walking up the tree:
      - fill attribute
      - style="fill: ..."
    Does not evaluate external CSS classes.
    """
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


def _is_whiteish_color_str(s: str) -> bool:
    if not s:
        return False
    c = s.strip().lower()
    if c in {"#fff", "#ffffff", "white"}:
        return True
    if c.startswith("rgb(") and "255" in c:
        # rough but fine for background strips
        return True
    # try parsing if possible
    try:
        r, g, b = _parse_css_color_to_rgb(s)
        # white-ish threshold
        return (r + g + b) / (3.0 * 255.0) > 0.92
    except Exception:
        return False


# -------------------- Logo building (combine 4 character SVGs) ----------------

_URL_ID_RE = re.compile(r"url\(#([^)]+)\)")


def _prefix_svg_ids(svg_fragment: etree._Element, prefix: str) -> None:
    """
    Prefix all id="" in this fragment and rewrite url(#id) + href="#id" references.
    Best-effort for typical defs usage (gradients/filters/clipPath).
    """
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
    """
    Remove any <rect> that:
      - exactly covers the glyph viewBox
      - AND is filled (directly or via style) with white-ish color
    This prevents 4 "white squares" in the combined 2x2 logo.
    """
    minx, miny, vbw, vbh = vb
    tol = 1e-6

    # remove anywhere in the glyph (common backgrounds are top-level, but be robust)
    rects = svg_root.xpath('.//*[local-name()="rect"]')
    for r in rects:
        if not isinstance(r.tag, str):
            continue
        if r.get("transform"):
            continue  # background rects usually have no transform

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
    """
    Combine 4 character SVGs into a 2x2 logo: [0 1; 2 3].

    gap_flag:
      0 -> no padding, no inter-glyph gap
      1 -> pad + gap = 1/8 of the character cell size
    """
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
        _strip_white_full_canvas_rects(root, vb)  # <-- key fix (removes 4 background squares)
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

        # center glyph in its cell
        tx = cell_x0 + (max_w - vbw) * 0.5 - minx
        ty = cell_y0 + (max_h - vbh) * 0.5 - miny

        group = etree.Element(f"{{{SVG_NS}}}g")
        for child in list(groot):
            group.append(copy.deepcopy(child))

        _prefix_svg_ids(group, f"g{idx}_")
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


# “Fire” hues (dispersion) as hue values.
_FIRE_HUES: List[float] = [
    200 / 360.0,  # cyan
    220 / 360.0,  # blue
    255 / 360.0,  # violet
    300 / 360.0,  # magenta
    330 / 360.0,  # pink-red
    45 / 360.0,   # amber
    70 / 360.0,   # yellow-green
    150 / 360.0,  # green-cyan
]


def facet_shimmer(t: float, freq: float, phase: float) -> float:
    # Slow beat shimmer (very slow changes)
    s1 = 0.5 + 0.5 * math.sin(2.0 * math.pi * (freq * t + phase))
    s2 = 0.5 + 0.5 * math.sin(2.0 * math.pi * ((freq * 0.47) * t + (phase * 1.63)))
    s = s1 * s2
    return _clamp01(s ** 2.0)


def make_pulses(rng: random.Random, duration: float, theme: str) -> List[Pulse]:
    if theme == "diamond":
        pulses: List[Pulse] = []

        # Very slow, long glints (reduce flicker)
        n_glints = rng.randint(1, 3)
        for _ in range(n_glints):
            t0 = rng.uniform(0.0, duration)
            half = rng.uniform(2.10, 5.80)   # long glints
            amp = rng.uniform(0.78, 1.00)
            power = rng.uniform(1.0, 1.20)
            pulses.append(Pulse(t0, half, amp, power=power))

        # Rare gentle "breath"
        if rng.random() < 0.60:
            t0 = rng.uniform(0.0, duration)
            half = rng.uniform(4.20, 8.00)
            amp = rng.uniform(0.05, 0.12)
            pulses.append(Pulse(t0, half, amp, power=1.0))

        return pulses

    # classic
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
    g.add_argument("--chars", type=str, default=None, help="Four chars key (ignoring spaces). Combines 4 character SVGs into 2x2.")

    ap.add_argument("--gap", type=int, default=0, choices=[0, 1],
                    help="Logo spacing (only for --chars): 0 = no gaps (default), 1 = pad+gap = 1/8 of cell size.")

    ap.add_argument("--color", type=str, default="", help="Override polygon base color (CSS color).")
    ap.add_argument("--bgcolor", type=str, default="", help="Override background color (CSS color).")

    ap.add_argument(
        "--theme",
        type=str,
        default="classic",
        choices=["classic", "diamond"],
        help="Animation theme: classic (default) or diamond.",
    )

    ap.add_argument("--duration", type=float, default=12.0, help="Duration in seconds (default: 12).")
    ap.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30).")
    ap.add_argument("--ext", type=str, default="mp4", help="Output extension: mp4 or webm (default: mp4).")

    ap.add_argument("--max-dim", type=int, default=1080,
                    help="Render so max(viewBox w,h) becomes this size (default: 1080).")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for repeatable animation.")
    ap.add_argument("--keep-frames", action="store_true", help="Keep rendered PNG frames (for debugging).")
    args = ap.parse_args()

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

    # ViewBox
    minx, miny, vb_w, vb_h = _parse_viewbox(doc)
    if vb_w <= 0 or vb_h <= 0:
        vb_w, vb_h = 240.0, 240.0

    # Background override
    bgcolor = args.bgcolor.strip() or None
    if bgcolor:
        # Remove an existing full-canvas rect if present (best-effort)
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

    # Determine base colors
    override_color = args.color.strip() or None
    base_rgb_override: Optional[Tuple[int, int, int]] = None
    if override_color:
        base_rgb_override = _parse_css_color_to_rgb(override_color)

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

    rng = random.Random(args.seed)
    pulses_per_poly: List[List[Pulse]] = [make_pulses(rng, float(args.duration), args.theme) for _ in polys]

    # Diamond per-polygon params
    poly_tone_base: List[float] = []
    poly_tone_phase: List[float] = []
    poly_shimmer_freq: List[float] = []
    poly_shimmer_phase: List[float] = []

    poly_fire_enabled: List[bool] = []
    poly_fire_hue: List[float] = []
    poly_fire_hue_phase: List[float] = []

    # random saturation + random “how white” the fire is (so some facets are more colorful than others)
    poly_fire_sat_base: List[float] = []
    poly_fire_sat_peak: List[float] = []
    poly_fire_sat_mul: List[float] = []
    poly_fire_white_mix: List[float] = []

    if args.theme == "diamond":
        for _ in polys:
            # Push tones toward extremes for contrast
            if rng.random() < 0.52:
                tone = (rng.random() ** 2.2) * 0.28            # 0..0.28
            else:
                tone = 1.0 - (rng.random() ** 2.2) * 0.28      # 0.72..1.0
            poly_tone_base.append(tone)
            poly_tone_phase.append(rng.uniform(0.0, 1.0))

            poly_shimmer_freq.append(rng.uniform(0.05, 0.14))  # very slow
            poly_shimmer_phase.append(rng.uniform(0.0, 1.0))

            poly_fire_enabled.append(rng.random() < 0.36)      # enough to be visible
            poly_fire_hue.append(rng.choice(_FIRE_HUES))
            poly_fire_hue_phase.append(rng.uniform(0.0, 1.0))

            # (2) saturation higher but random (and not always)
            # some facets: mild, some: punchier
            sat_base = rng.uniform(0.12, 0.28)
            sat_peak = rng.uniform(0.40, 0.78)
            poly_fire_sat_base.append(sat_base)
            poly_fire_sat_peak.append(sat_peak)

            # per-facet multiplier (occasionally boosts a lot)
            if rng.random() < 0.25:
                poly_fire_sat_mul.append(rng.uniform(1.15, 1.60))
            else:
                poly_fire_sat_mul.append(rng.uniform(0.85, 1.25))

            # how “tinted-white” vs “more colorful” the highlight is (lower = more color)
            poly_fire_white_mix.append(rng.uniform(0.48, 0.72))

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

        # Deep blacks + hot whites for body
        GREY_DARK = (6, 6, 8)
        GREY_LIGHT = (255, 255, 255)

        for i in range(frames):
            t = i / float(fps)

            if args.theme == "classic":
                for idx, poly in enumerate(polys):
                    a = whiteness_at(t, pulses_per_poly[idx])
                    poly.set("fill", _rgb_to_hex(mix_to_white(base_rgbs[idx], a)))

            else:
                # slow global ambient so it never goes dead
                global_amb = 0.05 + 0.03 * (0.5 + 0.5 * math.sin(2.0 * math.pi * (0.025 * t)))

                for idx, poly in enumerate(polys):
                    base = base_rgbs[idx]

                    # body tone with slight wobble
                    tone0 = poly_tone_base[idx]
                    wobble = 0.08 * math.sin(2.0 * math.pi * (0.06 * t + poly_tone_phase[idx]))
                    tone = _clamp01(tone0 + wobble)

                    body_grey = _mix_rgb(GREY_DARK, GREY_LIGHT, tone)
                    body = mix_to_white(body_grey, global_amb)

                    # if user overrides base color, preserve hue but impose facet value contrast
                    if base_rgb_override is not None:
                        h, s, _v = _rgb255_to_hsv01(base)
                        v = _clamp01(0.08 + 0.92 * tone)
                        body = _hsv01_to_rgb255(h, _clamp01(s), v)
                        body = mix_to_white(body, global_amb)

                    # glint strength
                    a = whiteness_at(t, pulses_per_poly[idx])
                    shim = facet_shimmer(t, poly_shimmer_freq[idx], poly_shimmer_phase[idx])
                    gl = _clamp01(0.86 * a + 0.55 * shim)

                    # specular whites
                    spec = _smoothstep(0.48, 1.00, gl)
                    spec_amt = spec * 0.98

                    # dispersion color appears inside specular, with per-facet randomness
                    if poly_fire_enabled[idx]:
                        fire_gate = _smoothstep(0.58, 1.00, gl)

                        hue = (poly_fire_hue[idx] + 0.015 * math.sin(2.0 * math.pi * (0.06 * t + poly_fire_hue_phase[idx]))) % 1.0

                        sat = poly_fire_sat_base[idx] + (poly_fire_sat_peak[idx] - poly_fire_sat_base[idx]) * fire_gate
                        sat *= poly_fire_sat_mul[idx]
                        sat = _clamp01(sat)

                        fire_rgb = _hsv01_to_rgb255(hue, sat, 1.0)

                        # “tinted white” amount varies per facet, so some are punchier than others
                        tw = _mix_rgb(fire_rgb, (255, 255, 255), poly_fire_white_mix[idx])

                        # apply as part of the specular
                        rgb = _mix_rgb(body, tw, spec_amt)
                    else:
                        rgb = _mix_rgb(body, (255, 255, 255), spec_amt)

                    poly.set("fill", _rgb_to_hex(rgb))

            svg_bytes = etree.tostring(doc, encoding="utf-8", xml_declaration=False)
            out_png = frames_dir / f"frame_{i:05d}.png"
            renderer_used = render_png(svg_bytes, out_png, out_w, out_h)

        encode_video_ffmpeg(frames_dir, fps, out_file, args.ext)

        print(f"Output: {out_file}")
        print(f"Input:  {label}")
        print(f"Theme:  {args.theme}")
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
