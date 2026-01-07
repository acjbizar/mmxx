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

# Variable-font compiler
from fontTools.designspaceLib import DesignSpaceDocument, AxisDescriptor, SourceDescriptor
from fontTools.varLib import build as varlib_build

# -----------------------------
# Fixed config
# -----------------------------
FONT_FAMILY = "mmxx"
FONT_STYLE = "Regular"

DEFAULT_SRC_DIR = Path("src")
DEFAULT_DIST_DIR = Path("dist")

UPM = 1000
ADVANCE_WIDTH = UPM
ASCENT = UPM
DESCENT = 0

NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

# -----------------------------
# VF axis: “Gap” (bigger value => bigger whitespace)
# -----------------------------
AXIS_TAG = "GAPS"     # 4 chars
AXIS_NAME = "Gap"
AXIS_MIN = 0
AXIS_DEFAULT = 0
AXIS_MAX = 1000

# At GAP=AXIS_MAX, polygons are scaled down to this factor around their centroid.
# Smaller factor => more whitespace/gaps. Keep this conservative to avoid degeneracy.
GAP_SCALE_AT_MAX = 0.82  # 0.82 is a good starting point; tweak later if desired.

# -----------------------------
# File resolution
# Supports BOTH patterns:
#  - src/character-{letter}.svg
#  - tests/sketch-{letter}.svg
# -----------------------------
def resolve_glyph_svg(src_dir: Path, ch: str) -> Optional[Path]:
    lo = ch.lower()
    up = ch.upper()

    candidates = [
        src_dir / f"character-{lo}.svg",
        src_dir / f"character-{up}.svg",
        src_dir / f"sketch-{lo}.svg",
        src_dir / f"sketch-{up}.svg",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

# -----------------------------
# SVG parsing (polygons only)
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

def load_svg_polygons_raw(svg_path: Path) -> Tuple[Tuple[float, float, float, float], List[Tuple[str, List[Tuple[float, float]]]]]:
    """
    Returns:
      viewBox (minx, miny, w, h)
      polygons: list of (stable_key, pts)
    stable_key is used to keep contour order identical across masters.
    """
    root = ET.parse(svg_path).getroot()

    viewbox = root.get("viewBox", "0 0 240 240")
    vb_nums = [float(x) for x in NUM_RE.findall(viewbox)]
    vb = (vb_nums[0], vb_nums[1], vb_nums[2], vb_nums[3]) if len(vb_nums) == 4 else (0.0, 0.0, 240.0, 240.0)

    polys: List[Tuple[str, List[Tuple[float, float]]]] = []
    auto_i = 0

    for el in root.iter():
        if _local_name(el.tag) != "polygon":
            continue
        pts_str = el.get("points")
        if not pts_str:
            continue

        pts = parse_points(pts_str)
        # Stable contour ordering: prefer id if present
        key = el.get("id") or f"__poly{auto_i:06d}"
        auto_i += 1
        polys.append((key, pts))

    # sort by key to ensure deterministic order
    polys.sort(key=lambda t: t[0])
    return vb, polys

# -----------------------------
# Polygon scaling (gap control)
# -----------------------------
def gap_to_scale(gap_value: float) -> float:
    """
    gap_value in [AXIS_MIN..AXIS_MAX].
    Returns a scale factor in [GAP_SCALE_AT_MAX..1.0].
    """
    t = 0.0
    if AXIS_MAX > AXIS_MIN:
        t = (gap_value - AXIS_MIN) / (AXIS_MAX - AXIS_MIN)
    t = max(0.0, min(1.0, t))
    return 1.0 + t * (GAP_SCALE_AT_MAX - 1.0)

def scale_points_about_centroid(pts: List[Tuple[float, float]], scale: float) -> List[Tuple[float, float]]:
    if not pts:
        return pts
    cx = sum(x for x, _ in pts) / len(pts)
    cy = sum(y for _, y in pts) / len(pts)
    out = []
    for x, y in pts:
        out.append((cx + (x - cx) * scale, cy + (y - cy) * scale))
    return out

# -----------------------------
# Geometry -> TrueType glyph
# (we keep EACH polygon as its own contour so masters remain compatible)
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
    is_ccw = area > 0
    if clockwise and is_ccw:
        return list(reversed(points))
    if (not clockwise) and (not is_ccw):
        return list(reversed(points))
    return points

def polygons_to_ttglyph(
    vb: Tuple[float, float, float, float],
    keyed_polys: List[Tuple[str, List[Tuple[float, float]]]],
    gap_value: float,
    upm: int = UPM,
) -> object:
    minx, miny, w, h = vb
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid viewBox: {vb}")

    sx = upm / w
    sy = upm / h

    scale = gap_to_scale(gap_value)

    pen = TTGlyphPen(None)

    for _key, pts in keyed_polys:
        if len(pts) < 3:
            continue

        pts2 = scale_points_about_centroid(pts, scale)

        ipts: List[Tuple[int, int]] = []
        for x, y in pts2:
            xx = (x - minx) * sx
            yy = (h - (y - miny)) * sy  # flip Y (SVG down) -> font up
            ipts.append((int(round(xx)), int(round(yy))))

        # Each polygon is a filled contour (clockwise outer)
        ipts = ensure_winding(ipts, clockwise=True)
        if not ipts:
            continue

        pen.moveTo(ipts[0])
        for p in ipts[1:]:
            pen.lineTo(p)
        pen.closePath()

    return pen.glyph()

# -----------------------------
# Static master build
# -----------------------------
def build_static_master(
    src_dir: Path,
    out_ttf: Path,
    gap_value: float,
    glyph_chars: List[str],
) -> None:
    glyph_order = [".notdef", "space"] + glyph_chars
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

    cmap: Dict[int, str] # codepoint -> glyph name
    cmap = {32: "space"}

    missing: List[str] = []

    for ch in glyph_chars:
        svg_path = resolve_glyph_svg(src_dir, ch)

        if svg_path is None:
            pen = TTGlyphPen(None)
            glyphs[ch] = pen.glyph()
            missing.append(ch)
        else:
            vb, keyed_polys = load_svg_polygons_raw(svg_path)
            glyphs[ch] = polygons_to_ttglyph(vb, keyed_polys, gap_value=gap_value, upm=UPM)

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

    # Give each master a distinct unique ID, but same family/style
    fb.setupNameTable(
        {
            "familyName": FONT_FAMILY,
            "styleName": FONT_STYLE,
            "uniqueFontIdentifier": f"{FONT_FAMILY}-{FONT_STYLE}-GAP{int(gap_value)}",
            "fullName": f"{FONT_FAMILY} {FONT_STYLE}",
            "psName": f"{FONT_FAMILY}-{FONT_STYLE}",
            "version": "Version 1.000",
        }
    )
    fb.setupPost()
    fb.setupMaxp()
    fb.setupHead()

    out_ttf.parent.mkdir(parents=True, exist_ok=True)
    fb.save(str(out_ttf))

    if missing:
        print(f"[warn] Missing glyph SVGs for: {', '.join(missing)}", file=sys.stderr)

# -----------------------------
# Variable font build (varLib)
# -----------------------------
def build_variable_font_from_masters(master_min: Path, master_max: Path, out_var_ttf: Path) -> None:
    """
    Create a small designspace on disk and let varLib compile the VF.
    """
    ds = DesignSpaceDocument()

    axis = AxisDescriptor()
    axis.tag = AXIS_TAG
    axis.name = AXIS_NAME
    axis.minimum = AXIS_MIN
    axis.default = AXIS_DEFAULT
    axis.maximum = AXIS_MAX
    ds.addAxis(axis)

    s0 = SourceDescriptor()
    s0.path = str(master_min)
    s0.name = "master.gap0"
    s0.location = {AXIS_NAME: AXIS_MIN}
    s0.familyName = FONT_FAMILY
    s0.styleName = "MasterGap0"
    ds.addSource(s0)

    s1 = SourceDescriptor()
    s1.path = str(master_max)
    s1.name = "master.gap1000"
    s1.location = {AXIS_NAME: AXIS_MAX}
    s1.familyName = FONT_FAMILY
    s1.styleName = "MasterGap1000"
    ds.addSource(s1)

    out_var_ttf.parent.mkdir(parents=True, exist_ok=True)
    designspace_path = out_var_ttf.with_suffix(".designspace")
    ds.write(str(designspace_path))

    # Build VF
    varfont = varlib_build(str(designspace_path))
    # fontTools has had different return shapes across versions; normalize:
    if isinstance(varfont, tuple):
        varfont = varfont[0]
    varfont.save(str(out_var_ttf))

# -----------------------------
# Build mmxx VF (and optional web formats)
# -----------------------------
def build_mmxx_variable_font(src_dir: Path, dist_dir: Path) -> None:
    fonts_dir = dist_dir / "fonts"
    masters_dir = fonts_dir / "masters"
    masters_dir.mkdir(parents=True, exist_ok=True)
    fonts_dir.mkdir(parents=True, exist_ok=True)

    # You can expand this later; keeping it aligned with your current generator intent.
    glyph_chars = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [chr(c) for c in range(ord("a"), ord("z") + 1)]

    master0 = masters_dir / f"{FONT_FAMILY}-GAP{AXIS_MIN}.ttf"
    master1 = masters_dir / f"{FONT_FAMILY}-GAP{AXIS_MAX}.ttf"

    print(f"Building masters:")
    print(f"  {master0.name} (gap={AXIS_MIN}, scale={gap_to_scale(AXIS_MIN):.4f})")
    build_static_master(src_dir=src_dir, out_ttf=master0, gap_value=AXIS_MIN, glyph_chars=glyph_chars)

    print(f"  {master1.name} (gap={AXIS_MAX}, scale={gap_to_scale(AXIS_MAX):.4f})")
    build_static_master(src_dir=src_dir, out_ttf=master1, gap_value=AXIS_MAX, glyph_chars=glyph_chars)

    out_var_ttf = fonts_dir / f"{FONT_FAMILY}.ttf"
    print(f"\nCompiling variable font -> {out_var_ttf}")
    build_variable_font_from_masters(master0, master1, out_var_ttf)

    # WOFF
    font = TTFont(str(out_var_ttf))
    font.flavor = "woff"
    font.save(str(fonts_dir / f"{FONT_FAMILY}.woff"))

    # WOFF2 (optional; commonly needs brotli)
    try:
        font = TTFont(str(out_var_ttf))
        font.flavor = "woff2"
        font.save(str(fonts_dir / f"{FONT_FAMILY}.woff2"))
    except Exception as e:
        print(f"[warn] Could not write WOFF2 (often needs 'brotli'): {e}", file=sys.stderr)

    print(f"\nDone.")
    print(f"Source SVGs:  {src_dir.resolve()} (character-{{ch}}.svg OR sketch-{{ch}}.svg)")
    print(f"Fonts:        {fonts_dir.resolve()}")
    print(f"Masters:      {masters_dir.resolve()}")
    print(f"Axis:         {AXIS_NAME} ({AXIS_TAG}) {AXIS_MIN}..{AXIS_MAX} default {AXIS_DEFAULT}")
    print(f"GAP_SCALE_AT_MAX = {GAP_SCALE_AT_MAX}")

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
