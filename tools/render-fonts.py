#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import xml.etree.ElementTree as ET

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont

# -----------------------------
# Fixed config
# -----------------------------
FONT_FAMILY = "mmxx"
FONT_STYLE = "Regular"

DEFAULT_SRC_DIR = Path("tests")   # ✅ source folder
DEFAULT_DIST_DIR = Path("dist")   # ✅ output folder root

UPM = 1000
ADVANCE_WIDTH = UPM
ASCENT = UPM
DESCENT = 0

NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

# -----------------------------
# File resolution: tests/sketch-{letter}.svg
# -----------------------------
def resolve_glyph_svg(src_dir: Path, ch: str) -> Optional[Path]:
    """
    Expected pattern:
      tests/sketch-a.svg ... tests/sketch-z.svg

    We'll try lowercase first, then uppercase, then a few fallbacks.
    """
    lo = ch.lower()
    up = ch.upper()

    candidates = [
        src_dir / f"sketch-{lo}.svg",
        src_dir / f"sketch-{up}.svg",
        src_dir / f"sketch_{lo}.svg",
        src_dir / f"sketch_{up}.svg",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

# -----------------------------
# SVG parsing / cleaning
# -----------------------------
def _local_name(tag) -> str:
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag

def parse_points(points_str: str) -> List[Tuple[float, float]]:
    nums = [float(x) for x in NUM_RE.findall(points_str)]
    if len(nums) < 6 or len(nums) % 2 != 0:
        raise ValueError(f"Bad polygon points: {points_str!r}")
    return list(zip(nums[0::2], nums[1::2]))

def load_svg_polygons(svg_path: Path) -> Tuple[Tuple[float, float, float, float], List[List[Tuple[float, float]]]]:
    """
    Returns:
      viewBox (minx, miny, w, h)
      polys: list of polygons, each polygon = list of (x,y)

    Notes:
    - Commented-out polygons are not part of the parsed DOM, so they’re automatically ignored.
    - We only read <polygon> (ignores any rect background etc.).
    """
    root = ET.parse(svg_path).getroot()

    viewbox = root.get("viewBox", "0 0 240 240")
    vb_nums = [float(x) for x in NUM_RE.findall(viewbox)]
    vb = (vb_nums[0], vb_nums[1], vb_nums[2], vb_nums[3]) if len(vb_nums) == 4 else (0.0, 0.0, 240.0, 240.0)

    polys: List[List[Tuple[float, float]]] = []
    for el in root.iter():
        if _local_name(el.tag) != "polygon":
            continue
        pts = el.get("points")
        if pts:
            polys.append(parse_points(pts))

    return vb, polys

def minify_between_tags(xml_bytes: bytes) -> bytes:
    xml = xml_bytes.strip()
    xml = re.sub(rb">\s+<", rb"><", xml)
    return xml + b"\n"

def write_clean_svg(svg_path: Path, out_path: Path) -> None:
    vb, polys = load_svg_polygons(svg_path)

    # ✅ viewBox only (no width/height)
    svg = ET.Element(
        "svg",
        {"xmlns": "http://www.w3.org/2000/svg", "viewBox": f"{vb[0]:g} {vb[1]:g} {vb[2]:g} {vb[3]:g}"}
    )

    ET.SubElement(svg, "rect", {
        "x": "0", "y": "0",
        "width": f"{vb[2]:g}", "height": f"{vb[3]:g}",
        "fill": "#fff"
    })

    g = ET.SubElement(svg, "g", {"fill": "#000", "shape-rendering": "crispEdges"})

    # Re-emit polygons without ids/extra attrs -> cleaner source
    for pts in polys:
        ET.SubElement(g, "polygon", {"points": " ".join(f"{x:g},{y:g}" for x, y in pts)})

    xml_bytes = ET.tostring(svg, encoding="utf-8", method="xml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(minify_between_tags(xml_bytes))

# -----------------------------
# Geometry -> TrueType glyph
# -----------------------------
def signed_area(points: List[Tuple[int, int]]) -> int:
    s = 0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s

def polys_to_ttglyph(
    vb: Tuple[float, float, float, float],
    polys: List[List[Tuple[float, float]]],
    upm: int = UPM,
) -> object:
    minx, miny, w, h = vb
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid viewBox: {vb}")

    sx = upm / w
    sy = upm / h

    pen = TTGlyphPen(None)

    for pts in polys:
        tpts: List[Tuple[int, int]] = []
        for x, y in pts:
            xx = (x - minx) * sx
            yy = (h - (y - miny)) * sy  # flip Y for font coords (Y-up)
            tpts.append((int(round(xx)), int(round(yy))))

        # Make all contours use consistent winding to avoid accidental “holes”
        if signed_area(tpts) < 0:
            tpts = list(reversed(tpts))

        pen.moveTo(tpts[0])
        for p in tpts[1:]:
            pen.lineTo(p)
        pen.closePath()

    return pen.glyph()

# -----------------------------
# Font build
# -----------------------------
def build_mmxx_font(src_dir: Path, dist_dir: Path) -> None:
    clean_svg_dir = dist_dir / "clean-svg"
    fonts_dir = dist_dir / "fonts"  # ✅ dist/fonts/
    clean_svg_dir.mkdir(parents=True, exist_ok=True)
    fonts_dir.mkdir(parents=True, exist_ok=True)

    glyph_order = [".notdef", "space"] + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    glyphs: Dict[str, object] = {}
    hmtx: Dict[str, Tuple[int, int]] = {}

    # .notdef: simple box
    pen = TTGlyphPen(None)
    m = int(UPM * 0.1)
    pen.moveTo((m, m))
    pen.lineTo((UPM - m, m))
    pen.lineTo((UPM - m, UPM - m))
    pen.lineTo((m, UPM - m))
    pen.closePath()
    glyphs[".notdef"] = pen.glyph()
    hmtx[".notdef"] = (ADVANCE_WIDTH, 0)

    # space
    pen = TTGlyphPen(None)
    glyphs["space"] = pen.glyph()
    hmtx["space"] = (ADVANCE_WIDTH, 0)

    cmap = {32: "space"}

    missing: List[str] = []

    for ch in [chr(c) for c in range(ord("A"), ord("Z") + 1)]:
        svg_path = resolve_glyph_svg(src_dir, ch)

        if svg_path is None:
            # missing glyph -> empty
            pen = TTGlyphPen(None)
            glyphs[ch] = pen.glyph()
            missing.append(ch)
        else:
            # cleaned copy saved as clean-svg/A.svg etc.
            write_clean_svg(svg_path, clean_svg_dir / f"{ch}.svg")
            vb, polys = load_svg_polygons(svg_path)
            glyphs[ch] = polys_to_ttglyph(vb, polys, upm=UPM)

        hmtx[ch] = (ADVANCE_WIDTH, 0)
        cmap[ord(ch)] = ch

    fb = FontBuilder(UPM, isTTF=True)
    fb.setupGlyphOrder(glyph_order)
    fb.setupCharacterMap(cmap)
    fb.setupGlyf(glyphs)
    fb.setupHorizontalMetrics(hmtx)
    fb.setupHorizontalHeader(ascent=ASCENT, descent=DESCENT, lineGap=0)
    fb.setupOS2(
        sTypoAscender=ASCENT,
        sTypoDescender=DESCENT,
        sTypoLineGap=0,
        usWinAscent=max(0, ASCENT),
        usWinDescent=max(0, -DESCENT),
    )
    fb.setupNameTable(
        {
            "familyName": FONT_FAMILY,
            "styleName": FONT_STYLE,
            "uniqueFontIdentifier": f"{FONT_FAMILY}-{FONT_STYLE}",
            "fullName": f"{FONT_FAMILY} {FONT_STYLE}",
            "psName": f"{FONT_FAMILY}-{FONT_STYLE}",
            "version": "Version 1.000",
        }
    )
    fb.setupPost()
    fb.setupMaxp()
    fb.setupHead()

    ttf_path = fonts_dir / f"{FONT_FAMILY}.ttf"
    fb.save(str(ttf_path))

    # WOFF
    font = TTFont(str(ttf_path))
    font.flavor = "woff"
    font.save(str(fonts_dir / f"{FONT_FAMILY}.woff"))

    # WOFF2 (optional; commonly needs brotli)
    try:
        font = TTFont(str(ttf_path))
        font.flavor = "woff2"
        font.save(str(fonts_dir / f"{FONT_FAMILY}.woff2"))
    except Exception as e:
        print(f"[warn] Could not write WOFF2 (often needs 'brotli'): {e}", file=sys.stderr)

    print(f"Source SVGs:  {src_dir.resolve()} (pattern: sketch-{{letter}}.svg)")
    print(f"Clean SVGs:   {clean_svg_dir.resolve()}")
    print(f"Fonts:        {fonts_dir.resolve()}")
    if missing:
        print(f"[warn] Missing glyph SVGs for: {', '.join(missing)}", file=sys.stderr)

def main() -> None:
    # Optional positional overrides:
    #   python build_mmxx_font.py [tests_dir] [dist_dir]
    src_dir = DEFAULT_SRC_DIR
    dist_dir = DEFAULT_DIST_DIR
    if len(sys.argv) >= 2:
        src_dir = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        dist_dir = Path(sys.argv[2])

    build_mmxx_font(src_dir=src_dir, dist_dir=dist_dir)

if __name__ == "__main__":
    main()
