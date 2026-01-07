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
from fontTools.designspaceLib import (
    DesignSpaceDocument, AxisDescriptor, SourceDescriptor, InstanceDescriptor
)
from fontTools.varLib import build as varlib_build

from shapely.geometry import Polygon, MultiPolygon, box as shp_box
from shapely.ops import unary_union

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
# Whitespace axis config
# -----------------------------
AXIS_TAG   = "GAP "     # 4-char OpenType tag (yes, trailing space)
AXIS_NAME  = "Gap"      # UI name
AXIS_MIN   = 0.5
AXIS_DEF   = 1.0
AXIS_MAX   = 2.0

# How strong the effect is (SVG units, viewBox ~240×240)
GAP_REF_SVG = 10.0

# VF compatibility: fixed number of points per contour
# NOTE: with the new "pad by duplicates" approach, higher numbers are safe.
PTS_PER_CONTOUR = 64

# Keep outer edge from pulling in: preserve default black inside a thin border frame.
# Set to 0.0 to disable anchoring.
ANCHOR_FRAME_SVG = 1.0

# Buffer settings for sharp corners
BUFFER_JOIN_STYLE = 2          # 2 = mitre
BUFFER_MITRE_LIMIT = 10.0      # larger => less beveling of acute corners

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
    Reads only <polygon>.
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

def _poly_sort_key(p: Polygon):
    try:
        c = p.centroid
        return (-abs(p.area), c.y, c.x)
    except Exception:
        return (-abs(p.area), 0.0, 0.0)

# -----------------------------
# Gap factor: offset boundaries, but anchor outer edge strip
# -----------------------------
def build_anchor(vb: Tuple[float, float, float, float], base_geom):
    if base_geom is None or ANCHOR_FRAME_SVG <= 0:
        return None
    minx, miny, w, h = vb
    clip = shp_box(minx, miny, minx + w, miny + h)

    # Border frame = clip minus inset clip
    try:
        inner = clip.buffer(-ANCHOR_FRAME_SVG, join_style=BUFFER_JOIN_STYLE, mitre_limit=BUFFER_MITRE_LIMIT)
        frame = clip.difference(inner)
        anchor = base_geom.intersection(frame)
        if anchor.is_empty:
            return None
        return anchor
    except Exception:
        return None

def geom_with_gap_factor(vb: Tuple[float, float, float, float], base_geom, factor: float, anchor_geom):
    """
    factor: 1.0 = original
            0.5 = less whitespace (black grows)
            2.0 = more whitespace (black shrinks)
    """
    if base_geom is None:
        return None

    minx, miny, w, h = vb
    clip = shp_box(minx, miny, minx + w, miny + h)

    # delta applied to BLACK shape
    delta = -(factor - 1.0) * (GAP_REF_SVG / 2.0)

    g = base_geom
    if abs(delta) > 1e-9:
        try:
            g = g.buffer(delta, join_style=BUFFER_JOIN_STYLE, mitre_limit=BUFFER_MITRE_LIMIT)
            if not g.is_valid:
                g = g.buffer(0)
        except Exception:
            g = base_geom

    # keep within viewBox
    try:
        g = g.intersection(clip)
        if g.is_empty:
            return None
    except Exception:
        pass

    # Anchor outer edge strip from default so the glyph doesn't "pull in" from the cell border
    if anchor_geom is not None:
        try:
            g = unary_union([g, anchor_geom])
            if not g.is_valid:
                g = g.buffer(0)
            g = g.intersection(clip)
        except Exception:
            pass

    if g is not None and getattr(g, "is_empty", False):
        return None
    return g

# -----------------------------
# TrueType contour helpers: preserve straight edges by padding with duplicates
# -----------------------------
def rotate_to_min_xy(pts: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not pts:
        return pts
    min_i = min(range(len(pts)), key=lambda i: (pts[i][0], pts[i][1]))
    return pts[min_i:] + pts[:min_i]

def signed_area_int(points: List[Tuple[int, int]]) -> int:
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
    is_ccw = signed_area_int(points) > 0
    if clockwise and is_ccw:
        return list(reversed(points))
    if (not clockwise) and (not is_ccw):
        return list(reversed(points))
    return points

def _dedupe_consecutive(pts: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not pts:
        return pts
    out = [pts[0]]
    for p in pts[1:]:
        if p != out[-1]:
            out.append(p)
    return out

def _is_collinear(a: Tuple[int, int], b: Tuple[int, int], c: Tuple[int, int]) -> bool:
    # area of triangle == 0
    return (b[0] - a[0]) * (c[1] - a[1]) == (b[1] - a[1]) * (c[0] - a[0])

def _remove_collinear(pts: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if len(pts) < 4:
        return pts
    pts = _dedupe_consecutive(pts)
    changed = True
    while changed and len(pts) >= 4:
        changed = False
        out: List[Tuple[int, int]] = []
        n = len(pts)
        for i in range(n):
            a = pts[(i - 1) % n]
            b = pts[i]
            c = pts[(i + 1) % n]
            if _is_collinear(a, b, c):
                changed = True
                continue
            out.append(b)
        pts = out
        pts = _dedupe_consecutive(pts)
        if len(pts) < 3:
            break
    return pts

def pad_contour_to_n(pts: List[Tuple[int, int]], n: int) -> List[Tuple[int, int]]:
    """
    Keep vertices (edges stay straight), but force exactly n points by duplicating vertices.
    If too many points, remove collinear points first; only decimate as a last resort.
    """
    pts = _dedupe_consecutive(pts)
    pts = _remove_collinear(pts)

    if len(pts) < 3:
        return pts

    if len(pts) > n:
        # try further collinear cleanup (already done), then decimate (rare)
        while len(pts) > n and len(pts) > 3:
            # drop every k-th point conservatively
            step = max(2, len(pts) // (len(pts) - n + 1))
            pts = [p for i, p in enumerate(pts) if (i % step) != 0] or pts
            pts = _dedupe_consecutive(pts)
        return pts[:n]

    if len(pts) == n:
        return pts

    # pad by repeating points evenly
    deficit = n - len(pts)
    reps = [1] * len(pts)
    i = 0
    while deficit > 0:
        reps[i] += 1
        deficit -= 1
        i = (i + 1) % len(pts)

    out: List[Tuple[int, int]] = []
    for p, r in zip(pts, reps):
        out.extend([p] * r)

    return out[:n]

def contours_from_geom_all(vb, geom, upm: int) -> Tuple[List[List[Tuple[int, int]]], Tuple[int, ...]]:
    """
    Build contours from ALL polygons (handles disconnected parts).
    Stable order: polygons by (-area, centroid), holes by centroid.
    Each contour is padded to PTS_PER_CONTOUR by duplicating points (keeps edges sharp).
    """
    if geom is None:
        return ([], ())

    minx, miny, w, h = vb
    if w <= 0 or h <= 0:
        return ([], ())

    sx = upm / w
    sy = upm / h

    polys = iter_polygons(geom)
    if not polys:
        return ([], ())

    polys.sort(key=_poly_sort_key)

    contours: List[List[Tuple[int, int]]] = []

    for poly in polys:
        if poly.is_empty:
            continue

        # Exterior
        ext = list(poly.exterior.coords)[:-1]
        ext_i: List[Tuple[int, int]] = []
        for x, y in ext:
            xx = (x - minx) * sx
            yy = (h - (y - miny)) * sy
            ext_i.append((int(round(xx)), int(round(yy))))
        ext_i = rotate_to_min_xy(ext_i)
        ext_i = ensure_winding(ext_i, clockwise=True)
        ext_i = pad_contour_to_n(ext_i, PTS_PER_CONTOUR)
        if len(ext_i) >= 3:
            contours.append(ext_i)

        # Holes
        holes = []
        for interior in poly.interiors:
            ring = list(interior.coords)[:-1]
            try:
                hp = Polygon(ring)
                c = hp.centroid
                holes.append((c.y, c.x, ring))
            except Exception:
                holes.append((0.0, 0.0, ring))
        holes.sort(key=lambda t: (t[0], t[1]))

        for _, __, ring in holes:
            hole_i: List[Tuple[int, int]] = []
            for x, y in ring:
                xx = (x - minx) * sx
                yy = (h - (y - miny)) * sy
                hole_i.append((int(round(xx)), int(round(yy))))
            hole_i = rotate_to_min_xy(hole_i)
            hole_i = ensure_winding(hole_i, clockwise=False)
            hole_i = pad_contour_to_n(hole_i, PTS_PER_CONTOUR)
            if len(hole_i) >= 3:
                contours.append(hole_i)

    structure = tuple(len(c) for c in contours)
    return contours, structure

def contours_to_ttglyph(contours: List[List[Tuple[int, int]]]) -> object:
    pen = TTGlyphPen(None)
    for pts in contours:
        if len(pts) < 3:
            continue
        pen.moveTo(pts[0])
        for p in pts[1:]:
            pen.lineTo(p)
        pen.closePath()
    return pen.glyph()

# -----------------------------
# Structure-safe factor search (per glyph)
# -----------------------------
def safe_factor_by_structure(vb, base_geom, target_factor: float, default_struct: Tuple[int, ...], anchor_geom):
    if base_geom is None:
        return (None, target_factor)

    # fast path
    g = geom_with_gap_factor(vb, base_geom, target_factor, anchor_geom)
    _, struct = contours_from_geom_all(vb, g, UPM)
    if struct == default_struct:
        return (g, target_factor)

    # binary search between 1.0 and target_factor
    lo = 1.0
    hi = target_factor
    if hi < lo:
        lo, hi = hi, lo

    best_factor = 1.0
    best_geom = geom_with_gap_factor(vb, base_geom, 1.0, anchor_geom)

    for _ in range(28):
        mid = (lo + hi) / 2.0
        gmid = geom_with_gap_factor(vb, base_geom, mid, anchor_geom)
        _, smid = contours_from_geom_all(vb, gmid, UPM)

        if smid == default_struct:
            best_factor = mid
            best_geom = gmid
            if target_factor >= 1.0:
                lo = mid
            else:
                hi = mid
        else:
            if target_factor >= 1.0:
                hi = mid
            else:
                lo = mid

    return (best_geom, best_factor)

# -----------------------------
# Build a static master TTF
# -----------------------------
def build_master_ttf(
    out_ttf: Path,
    glyph_order: List[str],
    cmap: Dict[int, str],
    glyphs: Dict[str, object],
    hmtx: Dict[str, Tuple[int, int]],
    family: str,
    style: str,
) -> None:
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
            "familyName": family,
            "styleName": style,
            "uniqueFontIdentifier": f"{family}-{style}",
            "fullName": f"{family} {style}",
            "psName": f"{family}-{style}",
            "version": "Version 1.000",
        }
    )
    fb.setupPost(keepGlyphNames=False)  # important: avoids latin1 glyph-name issues
    fb.setupMaxp()
    fb.setupHead()

    out_ttf.parent.mkdir(parents=True, exist_ok=True)
    fb.save(str(out_ttf))

# -----------------------------
# Variable font build
# -----------------------------
def build_mmxx_variable_font(src_dir: Path, dist_dir: Path) -> None:
    fonts_dir   = dist_dir / "fonts"
    masters_dir = fonts_dir / "masters"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    masters_dir.mkdir(parents=True, exist_ok=True)

    chars = discover_chars(src_dir)
    if not chars:
        raise SystemExit(f"No files found in {src_dir} matching character-*.svg")

    glyph_order = [".notdef", "space"] + chars

    # .notdef
    pen = TTGlyphPen(None)
    m = int(UPM * 0.1)
    pen.moveTo((m, m))
    pen.lineTo((UPM - m, m))
    pen.lineTo((UPM - m, UPM - m))
    pen.lineTo((m, UPM - m))
    pen.closePath()
    notdef_glyph = pen.glyph()

    # space
    pen = TTGlyphPen(None)
    space_glyph = pen.glyph()

    # shared metrics/cmap
    hmtx: Dict[str, Tuple[int, int]] = {g: (ADVANCE_WIDTH, 0) for g in glyph_order}
    cmap: Dict[int, str] = {32: "space"}
    for ch in chars:
        cmap[ord(ch)] = ch

    # Base geometry per glyph + anchor strip per glyph
    base_data: Dict[str, Tuple[Tuple[float, float, float, float], object]] = {}
    anchors: Dict[str, object] = {}
    missing: List[str] = []

    for ch in chars:
        p = resolve_glyph_svg(src_dir, ch)
        if p is None:
            missing.append(ch)
            base_data[ch] = ((0.0, 0.0, 240.0, 240.0), None)
            anchors[ch] = None
            continue
        vb, raw_polys = load_svg_polygons_raw(p)
        geom = union_polygons(raw_polys)
        base_data[ch] = (vb, geom)
        anchors[ch] = build_anchor(vb, geom)

    # Default structure signature per glyph at factor=1.0
    default_struct: Dict[str, Tuple[int, ...]] = {}
    for ch in chars:
        vb, geom = base_data[ch]
        gdef = geom_with_gap_factor(vb, geom, 1.0, anchors[ch])
        _, struct = contours_from_geom_all(vb, gdef, UPM)
        default_struct[ch] = struct

    # Three masters: min/default/max
    factors = [AXIS_MIN, AXIS_DEF, AXIS_MAX]
    master_paths: List[Path] = []

    for f in factors:
        glyphs: Dict[str, object] = {
            ".notdef": notdef_glyph,
            "space": space_glyph,
        }

        clamped: List[str] = []

        for ch in chars:
            vb, geom = base_data[ch]
            if geom is None:
                pen = TTGlyphPen(None)
                glyphs[ch] = pen.glyph()
                continue

            g, achieved = safe_factor_by_structure(vb, geom, f, default_struct[ch], anchors[ch])
            if abs(achieved - f) > 1e-4:
                clamped.append(f"{ch}:{achieved:.3f}")

            contours, _ = contours_from_geom_all(vb, g, UPM)
            glyphs[ch] = contours_to_ttglyph(contours)

        master_name = f"{FONT_FAMILY}-master-{f:.3f}".replace(".", "_")
        out_ttf = masters_dir / f"{master_name}.ttf"
        build_master_ttf(
            out_ttf=out_ttf,
            glyph_order=glyph_order,
            cmap=cmap,
            glyphs=glyphs,
            hmtx=hmtx,
            family=FONT_FAMILY,
            style=FONT_STYLE,
        )
        master_paths.append(out_ttf)

        if clamped:
            print(f"[info] Master {f:.3f}: clamped -> " + ", ".join(clamped))

    # DesignSpace
    ds = DesignSpaceDocument()
    axis = AxisDescriptor()
    axis.tag = AXIS_TAG
    axis.name = AXIS_NAME
    axis.minimum = AXIS_MIN
    axis.default = AXIS_DEF
    axis.maximum = AXIS_MAX
    ds.addAxis(axis)

    # Sources
    for f, path in zip(factors, master_paths):
        s = SourceDescriptor()
        s.name = f"master-{f:.3f}"
        s.filename = path.name
        s.path = str(path)
        s.location = {AXIS_NAME: f}
        s.copyLib = True
        s.copyInfo = True
        s.copyGroups = True
        s.copyFeatures = True
        ds.addSource(s)

    # Instances
    inst = InstanceDescriptor()
    inst.familyName = FONT_FAMILY
    inst.styleName = "Regular"
    inst.location = {AXIS_NAME: AXIS_DEF}
    ds.addInstance(inst)

    inst = InstanceDescriptor()
    inst.familyName = FONT_FAMILY
    inst.styleName = "Gap 50%"
    inst.location = {AXIS_NAME: AXIS_MIN}
    ds.addInstance(inst)

    inst = InstanceDescriptor()
    inst.familyName = FONT_FAMILY
    inst.styleName = "Gap 200%"
    inst.location = {AXIS_NAME: AXIS_MAX}
    ds.addInstance(inst)

    designspace_path = masters_dir / f"{FONT_FAMILY}.designspace"
    ds.write(str(designspace_path))

    # Build VF
    varfont, _, _ = varlib_build(str(designspace_path))
    vf_ttf = fonts_dir / f"{FONT_FAMILY}-vf.ttf"
    varfont.save(str(vf_ttf))

    # WOFF2
    try:
        vf = TTFont(str(vf_ttf))
        vf.flavor = "woff2"
        vf.save(str(fonts_dir / f"{FONT_FAMILY}-vf.woff2"))
    except Exception as e:
        print(f"[warn] Could not write VF WOFF2 (often needs 'brotli'): {e}", file=sys.stderr)

    # WOFF
    try:
        vf = TTFont(str(vf_ttf))
        vf.flavor = "woff"
        vf.save(str(fonts_dir / f"{FONT_FAMILY}-vf.woff"))
    except Exception as e:
        print(f"[warn] Could not write VF WOFF: {e}", file=sys.stderr)

    print(f"Source SVGs: {src_dir.resolve()}  (pattern: character-{{letter}}.svg)")
    print(f"Masters:     {masters_dir.resolve()}")
    print(f"Var font:    {vf_ttf.resolve()}")
    if missing:
        print(f"[warn] Missing SVGs for: {', '.join(missing)}", file=sys.stderr)

def main() -> None:
    src_dir = DEFAULT_SRC_DIR
    dist_dir = DEFAULT_DIST_DIR
    if len(sys.argv) >= 2:
        src_dir = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        dist_dir = Path(sys.argv[2])

    build_mmxx_variable_font(src_dir=src_dir, dist_dir=dist_dir)

if __name__ == "__main__":
    main()
