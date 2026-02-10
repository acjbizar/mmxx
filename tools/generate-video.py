#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/generate-video.py

Generate a 12s video from either:
  - --char  -> src/character-{char}.svg
  - --chars -> src/logo-{chars}.svg

Animation:
- Every polygon "randomly and fluently" pulses toward white and back to its base color.
- Base polygon color:
    - if --color is set: ALL polygons use that as base color
    - else: tries to resolve fill from polygon (and ancestors); falls back to black
- Background:
    - if --bgcolor is set: forces a full-canvas rect with that fill behind everything
    - else: keeps SVG as-is (white/transparent/etc.)

Output:
  dist/videos/character-{char}.{ext}
  dist/videos/logo-{chars}.{ext}

Dependencies (recommended):
  - cairosvg  (fast SVG->PNG):  py -m pip install cairosvg
  - ffmpeg    (on PATH)

Fallback renderers (much slower): inkscape, rsvg-convert
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
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
    # super-simple CSS style parser: "a:b; c:d"
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
    """
    Parse a CSS-ish color to (r,g,b). Accepts:
      - #rgb / #rrggbb
      - rgb(r,g,b)
      - common CSS color names (via Pillow if available)
    """
    if not color:
        raise ValueError("Empty color")

    c = color.strip()
    if c.lower() in {"none", "transparent"}:
        # treat as black base (so we can still animate)
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


# -------------------- Animation model ----------------------------------------

class Pulse:
    __slots__ = ("t0", "half", "amp")

    def __init__(self, t0: float, half: float, amp: float):
        self.t0 = t0
        self.half = half
        self.amp = amp

    def value(self, t: float) -> float:
        dt = abs(t - self.t0)
        if dt >= self.half:
            return 0.0
        # raised cosine bell from 1 at center to 0 at edge (smooth)
        x = dt / self.half  # 0..1
        return self.amp * (0.5 * (1.0 + math.cos(math.pi * x)))


def make_pulses(rng: random.Random, duration: float) -> List[Pulse]:
    # Per-polygon: a handful of random pulses over the 12s
    n = rng.randint(3, 7)
    pulses: List[Pulse] = []
    for _ in range(n):
        t0 = rng.uniform(0.0, duration)
        half = rng.uniform(0.25, 1.10)   # seconds (controls "fluent" speed)
        amp = rng.uniform(0.55, 1.00)    # how close to white it gets
        pulses.append(Pulse(t0, half, amp))
    return pulses


def whiteness_at(t: float, pulses: List[Pulse]) -> float:
    a = 0.0
    for p in pulses:
        a = max(a, p.value(t))
    # clamp
    if a < 0.0:
        return 0.0
    if a > 1.0:
        return 1.0
    return a


def mix_to_white(base: Tuple[int, int, int], a: float) -> Tuple[int, int, int]:
    # a=0 -> base, a=1 -> white
    r0, g0, b0 = base
    r = int(round((1.0 - a) * r0 + a * 255.0))
    g = int(round((1.0 - a) * g0 + a * 255.0))
    b = int(round((1.0 - a) * b0 + a * 255.0))
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


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


# -------------------- MAIN ----------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a 12s polygon-pulse video from a character SVG or a logo SVG.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--char", type=str, default=None, help="Single character: uses src/character-{char}.svg")
    g.add_argument("--chars", type=str, default=None, help="Four chars key: uses src/logo-{chars}.svg (lowercase)")

    ap.add_argument("--color", type=str, default="", help="Override polygon base color (CSS color).")
    ap.add_argument("--bgcolor", type=str, default="", help="Override background color (CSS color).")

    ap.add_argument("--duration", type=float, default=12.0, help="Duration in seconds (default: 12).")
    ap.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30).")
    ap.add_argument("--ext", type=str, default="mp4", help="Output extension: mp4 or webm (default: mp4).")

    ap.add_argument("--max-dim", type=int, default=1080, help="Render so the max(viewBox w,h) becomes this size (default: 1080).")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for repeatable animation.")
    ap.add_argument("--keep-frames", action="store_true", help="Keep rendered PNG frames (for debugging).")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    src_dir = root / "src"
    out_dir = root / "dist" / "videos"

    if args.char is not None:
        ch = args.char
        if len(ch) != 1:
            raise SystemExit("--char must be exactly one character.")
        in_svg = src_dir / f"character-{ch}.svg"
        out_file = out_dir / f"character-{ch}.{args.ext.lower()}"
        label = f"character {ch!r}"
    else:
        key = "".join(c for c in args.chars.strip() if not c.isspace()).lower()
        if len(key) != 4:
            raise SystemExit("--chars must be exactly four characters (ignoring spaces).")
        in_svg = src_dir / f"logo-{key}.svg"
        out_file = out_dir / f"logo-{key}.{args.ext.lower()}"
        label = f"logo {key!r}"

    if not in_svg.is_file():
        raise SystemExit(f"Input SVG not found: {in_svg}")

    # Parse SVG
    parser = etree.XMLParser(remove_blank_text=False, recover=True, remove_comments=False)
    doc = etree.fromstring(in_svg.read_bytes(), parser=parser)

    minx, miny, vb_w, vb_h = _parse_viewbox(doc)
    if vb_w <= 0 or vb_h <= 0:
        vb_w, vb_h = 240.0, 240.0

    # Background override (insert a full-canvas rect first)
    bgcolor = args.bgcolor.strip() or None
    if bgcolor:
        # Remove any existing full-canvas rect if present (best-effort)
        for el in list(doc):
            if not isinstance(el.tag, str):
                continue
            if _local_name(el.tag) != "rect":
                continue
            x = el.get("x", "0")
            y = el.get("y", "0")
            w = el.get("width", "")
            h = el.get("height", "")
            try:
                xf = float(x) if x else 0.0
                yf = float(y) if y else 0.0
                wf = float(w) if w else -1.0
                hf = float(h) if h else -1.0
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

    # Find polygons
    polys = doc.xpath('.//*[local-name()="polygon"]')
    if not polys:
        raise SystemExit(f"No <polygon> elements found in {in_svg}")

    # Determine base colors
    override_color = args.color.strip() or None
    if override_color:
        base_rgb_override = _parse_css_color_to_rgb(override_color)
    else:
        base_rgb_override = None

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

    # Animation params
    rng = random.Random(args.seed)
    pulses_per_poly: List[List[Pulse]] = [make_pulses(rng, float(args.duration)) for _ in polys]

    # Output sizing
    max_dim = int(args.max_dim)
    scale = float(max_dim) / float(max(vb_w, vb_h))
    out_w = max(1, int(round(vb_w * scale)))
    out_h = max(1, int(round(vb_h * scale)))

    frames = int(round(float(args.duration) * int(args.fps)))
    fps = int(args.fps)

    # Render frames
    tmp_root = Path(tempfile.mkdtemp(prefix="mmxx_video_"))
    frames_dir = tmp_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    try:
        renderer_used = None
        for i in range(frames):
            t = i / float(fps)

            # Update polygon fills for this frame
            for idx, poly in enumerate(polys):
                a = whiteness_at(t, pulses_per_poly[idx])
                rgb = mix_to_white(base_rgbs[idx], a)
                poly.set("fill", _rgb_to_hex(rgb))

            svg_bytes = etree.tostring(doc, encoding="utf-8", xml_declaration=False)

            out_png = frames_dir / f"frame_{i:05d}.png"
            renderer_used = render_png(svg_bytes, out_png, out_w, out_h)

        # Encode with ffmpeg
        encode_video_ffmpeg(frames_dir, fps, out_file, args.ext)

        print(f"Input:  {in_svg} ({label})")
        print(f"Output: {out_file}")
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
