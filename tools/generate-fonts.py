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

from shapely.geometry import Polygon, MultiPolygon, box as shp_box
from shapely.ops import unary_union

# Optional: easiest/robust GSUB builder
try:
    from fontTools.feaLib.builder import addOpenTypeFeaturesFromString
except Exception:
    addOpenTypeFeaturesFromString = None


# -----------------------------
# Fixed config
# -----------------------------
FONT_FAMILY = "mmxx"
FONT_STYLE  = "Regular"

DEFAULT_SRC_DIR  = Path("src")
DEFAULT_DIST_DIR = Path("dist")

UPM = 1000
ADVANCE_WIDTH = UPM
ASCENT = UPM
DESCENT = 0

NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


# -----------------------------
# Helpers
# -----------------------------
def _write_text_lf(path: Path, text: str) -> None:
    """Write UTF-8 text with LF newlines (works on older Python too)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)


# -----------------------------
# Glyph naming (safe)
# -----------------------------
def glyph_name_for_char(ch: str) -> str:
    cp = ord(ch)
    if cp <= 0xFFFF:
        return f"uni{cp:04X}"
    return f"u{cp:06X}"


# -----------------------------
# File resolution
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

def resolve_inverse_svg(src_dir: Path, ch: str) -> Optional[Path]:
    lo = ch.lower()
    up = ch.upper()
    candidates = [
        src_dir / f"inverse-{lo}.svg",
        src_dir / f"inverse-{up}.svg",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def discover_chars(src_dir: Path) -> List[str]:
    chars = set()
    for p in src_dir.glob("character-*.svg"):
        name = p.stem[len("character-"):]
        if len(name) == 1:
            chars.add(name)

    ordered: List[str] = []
    for c in "0123456789":
        if c in chars: ordered.append(c)
    for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if c in chars: ordered.append(c)
    for c in "abcdefghijklmnopqrstuvwxyz":
        if c in chars: ordered.append(c)
    for c in sorted(chars):
        if c not in ordered:
            ordered.append(c)
    return ordered


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
    Reads only <polygon>. Background rects etc. are ignored.
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
    if not geom.is_valid:
        geom = geom.buffer(0)
    if geom.is_empty:
        return None
    return geom

def iter_polygons(geom) -> List[Polygon]:
    if geom is None:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
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
# Exact inverse inside viewBox
# -----------------------------
def inverse_geom_from_base(vb: Tuple[float, float, float, float], base_geom):
    minx, miny, w, h = vb
    clip = shp_box(minx, miny, minx + w, miny + h)

    if base_geom is None:
        return clip

    try:
        g = base_geom.intersection(clip)
        if g.is_empty:
            return clip
    except Exception:
        g = base_geom

    try:
        inv = clip.difference(g)
        if inv.is_empty:
            return None
        if not inv.is_valid:
            inv = inv.buffer(0)
        if inv.is_empty:
            return None
        return inv
    except Exception:
        return None


# -----------------------------
# Geometry -> TrueType glyph (+ correct LSB)
# -----------------------------
def _signed_area(points: List[Tuple[int, int]]) -> int:
    s = 0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s

def _ensure_winding(points: List[Tuple[int, int]], clockwise: bool) -> List[Tuple[int, int]]:
    if len(points) < 3:
        return points
    is_ccw = _signed_area(points) > 0
    if clockwise and is_ccw:
        return list(reversed(points))
    if (not clockwise) and (not is_ccw):
        return list(reversed(points))
    return points

def _dedupe_consecutive(points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not points:
        return points
    out = [points[0]]
    for p in points[1:]:
        if p != out[-1]:
            out.append(p)
    return out

def geom_to_ttglyph_and_lsb(
    vb: Tuple[float, float, float, float],
    geom,
    upm: int = UPM
) -> Tuple[object, int]:
    """
    Returns (glyph, lsb).
    lsb MUST match the outline xMin for correct positioning.
    """
    minx, miny, w, h = vb
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid viewBox: {vb}")

    sx = upm / w
    sy = upm / h

    pen = TTGlyphPen(None)
    if geom is None:
        return pen.glyph(), 0

    polys = iter_polygons(geom)
    polys.sort(key=lambda p: abs(p.area), reverse=True)

    any_point = False
    x_min_outline = None  # type: Optional[int]

    for poly in polys:
        # exterior
        ext_pts: List[Tuple[int, int]] = []
        for x, y in list(poly.exterior.coords)[:-1]:
            xx = (x - minx) * sx
            yy = (h - (y - miny)) * sy  # flip Y
            xi = int(round(xx))
            yi = int(round(yy))
            ext_pts.append((xi, yi))
            any_point = True
            x_min_outline = xi if x_min_outline is None else min(x_min_outline, xi)

        ext_pts = _dedupe_consecutive(ext_pts)
        ext_pts = _ensure_winding(ext_pts, clockwise=True)
        if len(ext_pts) >= 3:
            pen.moveTo(ext_pts[0])
            for p in ext_pts[1:]:
                pen.lineTo(p)
            pen.closePath()

        # holes
        for interior in poly.interiors:
            hole_pts: List[Tuple[int, int]] = []
            for x, y in list(interior.coords)[:-1]:
                xx = (x - minx) * sx
                yy = (h - (y - miny)) * sy
                xi = int(round(xx))
                yi = int(round(yy))
                hole_pts.append((xi, yi))
                any_point = True
                x_min_outline = xi if x_min_outline is None else min(x_min_outline, xi)

            hole_pts = _dedupe_consecutive(hole_pts)
            hole_pts = _ensure_winding(hole_pts, clockwise=False)
            if len(hole_pts) >= 3:
                pen.moveTo(hole_pts[0])
                for p in hole_pts[1:]:
                    pen.lineTo(p)
                pen.closePath()

    if not any_point or x_min_outline is None:
        return pen.glyph(), 0

    # Correct left side bearing:
    return pen.glyph(), int(x_min_outline)


# -----------------------------
# Font build
# -----------------------------
def build_mmxx_font(src_dir: Path, dist_dir: Path) -> None:
    fonts_dir = dist_dir / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)

    chars = discover_chars(src_dir)
    if not chars:
        raise SystemExit(f"No files found in {src_dir} matching character-*.svg")

    # base glyph names
    char_to_gname = {ch: glyph_name_for_char(ch) for ch in chars}

    # If inverse-*.svg exists, we expose a .salt glyph + GSUB 'salt'
    inverse_map: Dict[str, str] = {}
    for ch in chars:
        if resolve_inverse_svg(src_dir, ch) is not None:
            base = char_to_gname[ch]
            alt = f"{base}.salt"
            inverse_map[base] = alt

    glyph_order = [".notdef", "space"] + [char_to_gname[ch] for ch in chars] + list(inverse_map.values())

    glyphs: Dict[str, object] = {}
    hmtx: Dict[str, Tuple[int, int]] = {}

    # .notdef
    pen = TTGlyphPen(None)
    m = int(UPM * 0.1)
    pen.moveTo((m, m))
    pen.lineTo((UPM - m, m))
    pen.lineTo((UPM - m, UPM - m))
    pen.lineTo((m, UPM - m))
    pen.closePath()
    glyphs[".notdef"] = pen.glyph()
    hmtx[".notdef"] = (ADVANCE_WIDTH, m)  # xMin is m

    # space
    pen = TTGlyphPen(None)
    glyphs["space"] = pen.glyph()
    hmtx["space"] = (ADVANCE_WIDTH, 0)

    # cmap (base glyphs only)
    cmap: Dict[int, str] = {32: "space"}
    for ch in chars:
        cmap[ord(ch)] = char_to_gname[ch]

    missing: List[str] = []

    # cache base geom for exact inverse computation
    base_geom_by_gname: Dict[str, Tuple[Tuple[float, float, float, float], object]] = {}

    # build base glyphs
    for ch in chars:
        gname = char_to_gname[ch]
        svg_path = resolve_glyph_svg(src_dir, ch)

        if svg_path is None:
            pen = TTGlyphPen(None)
            glyphs[gname] = pen.glyph()
            hmtx[gname] = (ADVANCE_WIDTH, 0)
            missing.append(ch)
            base_geom_by_gname[gname] = ((0.0, 0.0, 240.0, 240.0), None)
            continue

        vb, raw_polys = load_svg_polygons_raw(svg_path)
        geom = union_polygons(raw_polys)
        base_geom_by_gname[gname] = (vb, geom)

        g, lsb = geom_to_ttglyph_and_lsb(vb, geom, upm=UPM)
        glyphs[gname] = g
        hmtx[gname] = (ADVANCE_WIDTH, lsb)

    # build alt glyphs as exact inverse of base in the same vb
    for base_gname, alt_gname in inverse_map.items():
        vb, base_geom = base_geom_by_gname.get(base_gname, ((0.0, 0.0, 240.0, 240.0), None))
        inv_geom = inverse_geom_from_base(vb, base_geom)
        g, lsb = geom_to_ttglyph_and_lsb(vb, inv_geom, upm=UPM)
        glyphs[alt_gname] = g
        hmtx[alt_gname] = (ADVANCE_WIDTH, lsb)

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
    fb.setupPost(keepGlyphNames=False)
    fb.setupMaxp()
    fb.setupHead()

    # GSUB 'salt' (base -> base.salt)
    if inverse_map and addOpenTypeFeaturesFromString is not None:
        fea_lines = ["feature salt {"]
        for base, alt in inverse_map.items():
            fea_lines.append(f"  sub {base} by {alt};")
        fea_lines.append("} salt;")
        addOpenTypeFeaturesFromString(fb.font, "\n".join(fea_lines))
    elif inverse_map:
        print("[warn] fontTools.feaLib not available; cannot emit GSUB 'salt'.", file=sys.stderr)

    ttf_path = fonts_dir / f"{FONT_FAMILY}.ttf"
    fb.save(str(ttf_path))

    # WOFF
    font = TTFont(str(ttf_path))
    font.flavor = "woff"
    font.save(str(fonts_dir / f"{FONT_FAMILY}.woff"))

    # WOFF2
    try:
        font = TTFont(str(ttf_path))
        font.flavor = "woff2"
        font.save(str(fonts_dir / f"{FONT_FAMILY}.woff2"))
    except Exception as e:
        print(f"[warn] Could not write WOFF2 (often needs 'brotli'): {e}", file=sys.stderr)

    # Copy CSS
    css_src = src_dir / "style" / "main.css"
    css_dst = fonts_dir / f"{FONT_FAMILY}.css"
    if css_src.exists():
        _write_text_lf(css_dst, css_src.read_text(encoding="utf-8"))
    else:
        print(f"[warn] CSS source not found: {css_src}", file=sys.stderr)

    print(f"Source SVGs:  {src_dir.resolve()}  (character-{{char}}.svg)")
    print(f"Inverse flags:{src_dir.resolve()}  (inverse-{{char}}.svg) -> GSUB 'salt'")
    print(f"Fonts:        {fonts_dir.resolve()}")
    print(f"TTF:          {ttf_path.resolve()}")
    print(f"CSS:          {css_dst.resolve()}")
    if missing:
        print(f"[warn] Missing base SVGs for: {', '.join(missing)}", file=sys.stderr)
    if not inverse_map:
        print("[info] No inverse-*.svg found; no 'salt' alternates emitted.")


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
