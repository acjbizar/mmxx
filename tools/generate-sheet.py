#!/usr/bin/env python3
"""
tools/generate-sheet.py

Generate a character sheet SVG at dist/sheet.svg that displays all glyphs found in
src/character-{char}.svg (single-character {char} only).

- Reads only <polygon> elements (same as the font generator input)
- Unites polygons per glyph (Illustrator "Unite"-style) via Shapely
- Writes each glyph as a single <path> (evenodd) in the sheet
- Lays out glyphs in a near-square grid with a 30px gap between cells
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import xml.etree.ElementTree as ET

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


# -----------------------------
# Discover characters
# -----------------------------
def discover_chars(src_dir: Path) -> List[str]:
    chars = set()
    for p in src_dir.glob("character-*.svg"):
        name = p.stem[len("character-") :]
        if len(name) == 1:
            chars.add(name)

    ordered: List[str] = []
    for c in "0123456789":
        if c in chars:
            ordered.append(c)
    for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if c in chars:
            ordered.append(c)
    for c in "abcdefghijklmnopqrstuvwxyz":
        if c in chars:
            ordered.append(c)
    for c in sorted(chars):
        if c not in ordered:
            ordered.append(c)
    return ordered


def _local_name(tag) -> str:
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


# -----------------------------
# SVG parsing (polygons only)
# -----------------------------
def _to_float(x: str, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def parse_points(points_str: str) -> List[Tuple[float, float]]:
    nums = [float(x) for x in NUM_RE.findall(points_str or "")]
    if len(nums) < 6 or (len(nums) % 2) != 0:
        return []
    return list(zip(nums[0::2], nums[1::2]))


def load_svg_polygons_raw(svg_path: Path) -> Tuple[Tuple[float, float, float, float], List[List[Tuple[float, float]]]]:
    """
    Returns:
      viewBox (minx, miny, w, h)
      raw polys: list of polygons, each polygon = list of (x,y)
    Reads only <polygon>. Other elements are ignored.
    """
    root = ET.parse(svg_path).getroot()

    viewbox = root.get("viewBox", "0 0 240 240")
    vb_nums = [float(x) for x in NUM_RE.findall(viewbox)]
    if len(vb_nums) == 4:
        vb = (vb_nums[0], vb_nums[1], vb_nums[2], vb_nums[3])
    else:
        vb = (0.0, 0.0, 240.0, 240.0)

    polys: List[List[Tuple[float, float]]] = []
    for el in root.iter():
        if _local_name(el.tag) != "polygon":
            continue
        pts = el.get("points")
        if pts:
            ring = parse_points(pts)
            if len(ring) >= 3:
                polys.append(ring)

    return vb, polys


# -----------------------------
# Unite polygons (Illustrator "Unite")
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


# -----------------------------
# Geometry -> single SVG path d
# -----------------------------
def _fmt_num(x: float, snap_eps: float = 1e-6, max_decimals: int = 3) -> str:
    rx = round(x)
    if abs(x - rx) <= snap_eps:
        return str(int(rx))
    s = f"{x:.{max_decimals}f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _ring_to_path(coords, snap_eps: float = 1e-6, max_decimals: int = 3) -> str:
    coords = list(coords)
    if len(coords) < 4:
        return ""
    if coords[0] == coords[-1]:
        coords = coords[:-1]
    if not coords:
        return ""
    parts = [f"M {_fmt_num(coords[0][0], snap_eps, max_decimals)} {_fmt_num(coords[0][1], snap_eps, max_decimals)}"]
    for (x, y) in coords[1:]:
        parts.append(f"L {_fmt_num(x, snap_eps, max_decimals)} {_fmt_num(y, snap_eps, max_decimals)}")
    parts.append("Z")
    return " ".join(parts)


def geom_to_path_d(geom, snap_eps: float = 1e-6, max_decimals: int = 3) -> str:
    if geom is None or geom.is_empty:
        return ""
    parts: List[str] = []
    polys = iter_polygons(geom)
    # stable order: bigger first
    polys.sort(key=lambda p: abs(p.area), reverse=True)

    for poly in polys:
        ext = _ring_to_path(poly.exterior.coords, snap_eps, max_decimals)
        if ext:
            parts.append(ext)
        for interior in poly.interiors:
            hole = _ring_to_path(interior.coords, snap_eps, max_decimals)
            if hole:
                parts.append(hole)
    return " ".join(parts)


# -----------------------------
# Sheet generation
# -----------------------------
def main() -> None:
    root = Path(__file__).resolve().parent.parent
    src_dir = root / "src"
    out_svg = root / "dist" / "sheet.svg"

    if not src_dir.is_dir():
        raise SystemExit(f"Source folder not found: {src_dir}")

    chars = discover_chars(src_dir)
    if not chars:
        raise SystemExit(f"No files found matching: {src_dir / 'character-*.svg'} (single-char names only)")

    gap = 30.0  # requested 30px gap between characters

    # Load + simplify all glyphs first, and compute max viewBox size for cell sizing
    items: List[Dict] = []
    max_w = 0.0
    max_h = 0.0

    for ch in chars:
        p = src_dir / f"character-{ch}.svg"
        if not p.exists():
            # also allow case variants like character-A.svg when ch is "A"
            # (discover_chars already uses stems, so this is mostly for safety)
            alt = src_dir / f"character-{ch.lower()}.svg"
            p = alt if alt.exists() else p

        vb, polys = load_svg_polygons_raw(p)
        geom = union_polygons(polys)
        d = geom_to_path_d(geom, snap_eps=1e-6, max_decimals=3)

        minx, miny, w, h = vb
        max_w = max(max_w, w)
        max_h = max(max_h, h)

        items.append(
            {
                "ch": ch,
                "path_d": d,
                "vb": vb,
            }
        )

    cell_w = max_w
    cell_h = max_h

    n = len(items)
    cols = max(1, int(math.ceil(math.sqrt(n))))
    rows = int(math.ceil(n / cols))

    sheet_w = cols * cell_w + (cols - 1) * gap
    sheet_h = rows * cell_h + (rows - 1) * gap

    svg = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "viewBox": f"0 0 {_fmt_num(sheet_w)} {_fmt_num(sheet_h)}",
        },
    )

    # Optional: keep edges crisp
    # (stroke-free shapes; crispEdges mainly matters for rect/lines, but harmless)
    svg.set("shape-rendering", "crispEdges")

    for i, it in enumerate(items):
        r = i // cols
        c = i % cols
        x0 = c * (cell_w + gap)
        y0 = r * (cell_h + gap)

        minx, miny, w, h = it["vb"]

        # Center within the cell (if some glyphs have smaller viewBox)
        dx = (cell_w - w) / 2.0
        dy = (cell_h - h) / 2.0

        # Translate so glyph's viewBox min corner aligns, then position in the grid
        tx = x0 + dx - minx
        ty = y0 + dy - miny

        g = ET.SubElement(svg, "g", {"transform": f"translate({_fmt_num(tx)} {_fmt_num(ty)})"})
        if it["path_d"]:
            ET.SubElement(
                g,
                "path",
                {
                    "d": it["path_d"],
                    "fill": "#000",
                    "fill-rule": "evenodd",
                },
            )

    out_svg.parent.mkdir(parents=True, exist_ok=True)
    xml = ET.tostring(svg, encoding="unicode", method="xml")
    # minify whitespace between tags a bit
    xml = re.sub(r">\s+<", "><", xml).strip() + "\n"
    with out_svg.open("w", encoding="utf-8", newline="\n") as f:
        f.write(xml)

    print(f"Found {n} character(s) in {src_dir}")
    print(f"Wrote: {out_svg}  ({cols} cols × {rows} rows, gap={int(gap)}px)")

if __name__ == "__main__":
    main()
