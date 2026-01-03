#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

import xml.etree.ElementTree as ET

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

# -----------------------------
# Fixed config
# -----------------------------
FONT_FAMILY = "mmxx"
FONT_STYLE = "Regular"

DEFAULT_SRC_DIR = Path("src")     # ✅ source folder
DEFAULT_DIST_DIR = Path("dist")   # ✅ output folder root

UPM = 1000
ADVANCE_WIDTH = UPM
ASCENT = UPM
DESCENT = 0

NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

# -----------------------------
# File resolution: src/character-{letter}.svg
# -----------------------------
def resolve_glyph_svg(src_dir: Path, ch: str) -> Optional[Path]:
    lo = ch.lower()
    up = ch.upper()

    candidates = [
        src_dir / f"character-{lo}.svg",
        src_dir / f"character-{up}.svg",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

# -----------------------------
# SVG parsing
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

def load_svg_polygons_raw(svg_path: Path) -> Tuple[Tuple[float, float, float, float], List[List[Tuple[float, float]]]]:
    """
    Returns:
      viewBox (minx, miny, w, h)
      raw polys: list of polygons, each polygon = list of (x,y)
    Reads only <polygon>. Any background rects are ignored automatically.
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

# -----------------------------
# Union polygons like Illustrator "Unite"
# -----------------------------
def union_polygons(polys: List[List[Tuple[float, float]]]):
    shp_polys: List[Polygon] = []
    for pts in polys:
        if len(pts) < 3:
            continue
        try:
            p = Polygon(pts)
            if not p.is_valid:
                p = p.buffer(0)
            if not p.is_empty:
                shp_polys.append(p)
        except Exception:
            continue

    if not shp_polys:
        return None

    geom = unary_union(shp_polys)
    if geom.is_empty:
        return None
    return geom

def iter_polygons(geom) -> Iterable[Polygon]:
    if geom is None:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    # Other geometry collections: best-effort
    out: List[Polygon] = []
    try:
        for g in geom.geoms:
            if isinstance(g, Polygon):
                out.append(g)
            elif isinstance(g, MultiPolygon):
                out.extend(list(g.geoms))
    except Exception:
        pass
    return out

# -----------------------------
# Clean/minify SVG output (single <path>, evenodd)
# -----------------------------
def minify_between_tags(xml_bytes: bytes) -> bytes:
    xml = xml_bytes.strip()
    xml = re.sub(rb">\s+<", rb"><", xml)
    return xml + b"\n"

def _fmt_num(x: float) -> str:
    # keep things compact
    rx = round(x)
    if abs(x - rx) < 1e-6:
        return str(int(rx))
    s = f"{x:.3f}".rstrip("0").rstrip(".")
    return s if s else "0"

def _ring_to_path(coords) -> str:
    coords = list(coords)
    if len(coords) < 4:
        return ""
    if coords[0] == coords[-1]:
        coords = coords[:-1]
    if not coords:
        return ""
    parts = [f"M {_fmt_num(coords[0][0])} {_fmt_num(coords[0][1])}"]
    for (x, y) in coords[1:]:
        parts.append(f"L {_fmt_num(x)} {_fmt_num(y)}")
    parts.append("Z")
    return " ".join(parts)

def geom_to_path_d(geom) -> str:
    if geom is None:
        return ""
    parts: List[str] = []
    for poly in iter_polygons(geom):
        ext = _ring_to_path(poly.exterior.coords)
        if ext:
            parts.append(ext)
        for interior in poly.interiors:
            hole = _ring_to_path(interior.coords)
            if hole:
                parts.append(hole)
    return " ".join(parts)

def write_clean_svg(svg_path: Path, out_path: Path) -> None:
    vb, raw_polys = load_svg_polygons_raw(svg_path)
    geom = union_polygons(raw_polys)

    # ✅ viewBox only (no width/height), single path like template cleaner
    svg = ET.Element(
        "svg",
        {"xmlns": "http://www.w3.org/2000/svg", "viewBox": f"{vb[0]:g} {vb[1]:g} {vb[2]:g} {vb[3]:g}"}
    )

    d = geom_to_path_d(geom)
    if d:
        ET.SubElement(svg, "path", {"d": d, "fill": "#000", "fill-rule": "evenodd"})

    xml_bytes = ET.tostring(svg, encoding="utf-8", method="xml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(minify_between_tags(xml_bytes))

# -----------------------------
# Geometry -> TrueType glyph
#   (non-zero winding: outer CW, holes CCW)
# -----------------------------
def signed_area(points: List[Tuple[int, int]]) -> int:
    s = 0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s

def ensure_winding(points: List[Tuple[int, int]], clockwise: bool) -> List[Tuple[int, int]]:
    if len(points) < 3:
        return points
    area = signed_area(points)
    # In Y-up coords: area > 0 means CCW, area < 0 means CW
    is_ccw = area > 0
    if clockwise and is_ccw:
        return list(reversed(points))
    if (not clockwise) and (not is_ccw):
        return list(reversed(points))
    return points

def geom_to_ttglyph(
    vb: Tuple[float, float, float, float],
    geom,
    upm: int = UPM,
) -> object:
    minx, miny, w, h = vb
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid viewBox: {vb}")

    sx = upm / w
    sy = upm / h

    pen = TTGlyphPen(None)

    if geom is None:
        return pen.glyph()

    for poly in iter_polygons(geom):
        # exterior (outer) ring
        ext_pts: List[Tuple[int, int]] = []
        for x, y in list(poly.exterior.coords)[:-1]:
            xx = (x - minx) * sx
            yy = (h - (y - miny)) * sy  # flip Y (SVG down) -> font up
            ext_pts.append((int(round(xx)), int(round(yy))))
        ext_pts = ensure_winding(ext_pts, clockwise=True)

        if ext_pts:
            pen.moveTo(ext_pts[0])
            for p in ext_pts[1:]:
                pen.lineTo(p)
            pen.closePath()

        # holes (interiors)
        for interior in poly.interiors:
            hole_pts: List[Tuple[int, int]] = []
            for x, y in list(interior.coords)[:-1]:
                xx = (x - minx) * sx
                yy = (h - (y - miny)) * sy
                hole_pts.append((int(round(xx)), int(round(yy))))
            hole_pts = ensure_winding(hole_pts, clockwise=False)

            if hole_pts:
                pen.moveTo(hole_pts[0])
                for p in hole_pts[1:]:
                    pen.lineTo(p)
                pen.closePath()

    return pen.glyph()

# -----------------------------
# Font build
# -----------------------------
def build_mmxx_font(src_dir: Path, dist_dir: Path) -> None:
    clean_svg_dir = dist_dir / "clean-svg"
    fonts_dir = dist_dir / "fonts"
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
            pen = TTGlyphPen(None)
            glyphs[ch] = pen.glyph()
            missing.append(ch)
        else:
            # ✅ Clean & minify FIRST (like templates), and also build glyph from UNIONED geometry
            write_clean_svg(svg_path, clean_svg_dir / f"{ch}.svg")

            vb, raw_polys = load_svg_polygons_raw(svg_path)
            geom = union_polygons(raw_polys)
            glyphs[ch] = geom_to_ttglyph(vb, geom, upm=UPM)

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

    print(f"Source SVGs:  {src_dir.resolve()} (pattern: character-{{letter}}.svg)")
    print(f"Clean SVGs:   {clean_svg_dir.resolve()} (single-path, evenodd)")
    print(f"Fonts:        {fonts_dir.resolve()}")
    if missing:
        print(f"[warn] Missing glyph SVGs for: {', '.join(missing)}", file=sys.stderr)

def main() -> None:
    src_dir = DEFAULT_SRC_DIR
    dist_dir = DEFAULT_DIST_DIR
    if len(sys.argv) >= 2:
        src_dir = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        dist_dir = Path(sys.argv[2])

    build_mmxx_font(src_dir=src_dir, dist_dir=dist_dir)

if __name__ == "__main__":
    main()
