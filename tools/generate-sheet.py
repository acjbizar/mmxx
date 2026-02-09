#!/usr/bin/env python3
"""
tools/generate-sheet.py

Generate a character sheet SVG at src/sheet.svg that displays ONLY:
- digits 0-9
- lowercase a-z

That’s 36 characters, laid out in a fixed 6×6 grid.

- Reads only <polygon> elements (same as the font generator input)
- Unites polygons per glyph (Illustrator "Unite"-style) via Shapely
- Writes each glyph as a single <path> (evenodd) in the sheet
- Keeps a 30px gap between cells
- If a glyph SVG is missing, its cell is left empty (and a warning is printed)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import xml.etree.ElementTree as ET

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


# -----------------------------
# Fixed character set (36)
# -----------------------------
def target_chars() -> List[str]:
    # Order: 0-9 then a-z (36 total)
    return list("0123456789abcdefghijklmnopqrstuvwxyz")


def _local_name(tag) -> str:
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


# -----------------------------
# SVG parsing (polygons only)
# -----------------------------
def parse_points(points_str: str) -> List[Tuple[float, float]]:
    nums = [float(x) for x in NUM_RE.findall(points_str or "")]
    if len(nums) < 6 or (len(nums) % 2) != 0:
        return []
    return list(zip(nums[0::2], nums[1::2]))


def load_svg_polygons_raw(
    svg_path: Path,
) -> Tuple[Tuple[float, float, float, float], List[List[Tuple[float, float]]]]:
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
    parts = [
        f"M {_fmt_num(coords[0][0], snap_eps, max_decimals)} {_fmt_num(coords[0][1], snap_eps, max_decimals)}"
    ]
    for (x, y) in coords[1:]:
        parts.append(f"L {_fmt_num(x, snap_eps, max_decimals)} {_fmt_num(y, snap_eps, max_decimals)}")
    parts.append("Z")
    return " ".join(parts)


def geom_to_path_d(geom, snap_eps: float = 1e-6, max_decimals: int = 3) -> str:
    if geom is None or geom.is_empty:
        return ""
    parts: List[str] = []
    polys = iter_polygons(geom)
    polys.sort(key=lambda p: abs(p.area), reverse=True)  # stable: bigger first

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
    out_svg = src_dir / "sheet.svg"  # (1) write to src

    if not src_dir.is_dir():
        raise SystemExit(f"Source folder not found: {src_dir}")

    chars = target_chars()
    assert len(chars) == 36

    gap = 30.0
    cols = 6  # (2) fixed 6×6
    rows = 6

    # Load + simplify glyphs, and compute max viewBox size for cell sizing
    items: List[Dict] = []
    max_w = 0.0
    max_h = 0.0
    missing: List[str] = []

    for ch in chars:
        p = src_dir / f"character-{ch}.svg"

        if not p.exists():
            missing.append(ch)
            # Keep placeholder so the grid stays 36 cells
            items.append({"ch": ch, "path_d": "", "vb": (0.0, 0.0, 240.0, 240.0), "missing": True})
            max_w = max(max_w, 240.0)
            max_h = max(max_h, 240.0)
            continue

        vb, polys = load_svg_polygons_raw(p)
        geom = union_polygons(polys)
        d = geom_to_path_d(geom, snap_eps=1e-6, max_decimals=3)

        minx, miny, w, h = vb
        max_w = max(max_w, w)
        max_h = max(max_h, h)

        items.append({"ch": ch, "path_d": d, "vb": vb, "missing": False})

    cell_w = max_w
    cell_h = max_h

    sheet_w = cols * cell_w + (cols - 1) * gap
    sheet_h = rows * cell_h + (rows - 1) * gap

    svg = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "viewBox": f"0 0 {_fmt_num(sheet_w)} {_fmt_num(sheet_h)}",
        },
    )
    svg.set("shape-rendering", "crispEdges")

    for i, it in enumerate(items):
        r = i // cols
        c = i % cols
        x0 = c * (cell_w + gap)
        y0 = r * (cell_h + gap)

        minx, miny, w, h = it["vb"]
        dx = (cell_w - w) / 2.0
        dy = (cell_h - h) / 2.0

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
    xml = re.sub(r">\s+<", "><", xml).strip() + "\n"

    with out_svg.open("w", encoding="utf-8", newline="\n") as f:
        f.write(xml)

    print("Target set: 0-9 and a-z (36 glyphs)")
    if missing:
        print(f"WARNING: Missing {len(missing)} glyph file(s): " + ", ".join(missing))
        print("         Those cells were left empty.")
    print(f"Wrote: {out_svg}  ({cols} cols × {rows} rows, gap={int(gap)}px)")


if __name__ == "__main__":
    main()
