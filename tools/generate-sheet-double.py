#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
import math
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict

# ----------------------------
# Config: your grid
# ----------------------------
CELL = 30
GRID = 8
GLYPH = CELL * GRID  # 240


Point = Tuple[float, float]
Poly = List[Point]


# ----------------------------
# Geometry: point-in-polygon (evenodd)
# ----------------------------
def _is_point_on_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float, eps: float = 1e-9) -> bool:
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > eps:
        return False
    dot = (px - ax) * (bx - ax) + (py - ay) * (by - ay)
    if dot < -eps:
        return False
    sq_len = (bx - ax) ** 2 + (by - ay) ** 2
    if dot - sq_len > eps:
        return False
    return True


def point_in_poly_evenodd(x: float, y: float, polys: List[Poly]) -> bool:
    """
    Even-odd fill across multiple closed polygons.
    Treat boundary points as inside.
    """
    inside = False
    for poly in polys:
        n = len(poly)
        if n < 3:
            continue

        # boundary check
        for i in range(n):
            ax, ay = poly[i]
            bx, by = poly[(i + 1) % n]
            if _is_point_on_segment(x, y, ax, ay, bx, by):
                return True

        # ray casting
        j = n - 1
        c = False
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi):
                c = not c
            j = i

        if c:
            inside = not inside

    return inside


# ----------------------------
# SVG path parsing: M/L/Z only
# ----------------------------
TOKEN_RE = re.compile(r"[MLZ]|-?\d+(?:\.\d+)?")

def parse_path_to_polys(d: str) -> List[Poly]:
    tokens = TOKEN_RE.findall(d)
    polys: List[Poly] = []
    cur: List[Point] = []
    i = 0

    def flush():
        nonlocal cur
        if len(cur) >= 3:
            polys.append(cur)
        cur = []

    while i < len(tokens):
        t = tokens[i]
        i += 1

        if t in ("M", "L"):
            if i + 1 >= len(tokens):
                break
            x = float(tokens[i]); y = float(tokens[i + 1])
            i += 2
            if t == "M":
                flush()
                cur = [(x, y)]
            else:
                cur.append((x, y))
        elif t == "Z":
            flush()

    flush()
    return polys


# ----------------------------
# Masks + components
# ----------------------------
def empty_mask_from_path(d: str) -> List[List[bool]]:
    """
    Return GRID×GRID mask where True means "whitespace" (empty).
    We sample 4 points in each cell to be robust near edges.
    """
    polys = parse_path_to_polys(d)

    mask_empty = [[False] * GRID for _ in range(GRID)]
    offsets = (0.25, 0.75)

    for gy in range(GRID):
        for gx in range(GRID):
            filled_hits = 0
            total = 0
            for oy in offsets:
                for ox in offsets:
                    total += 1
                    sx = (gx + ox) * CELL
                    sy = (gy + oy) * CELL
                    if point_in_poly_evenodd(sx, sy, polys):
                        filled_hits += 1

            # majority vote
            filled = (filled_hits >= (total / 2))
            mask_empty[gy][gx] = (not filled)

    return mask_empty


def connected_components(mask: List[List[bool]]) -> List[List[Tuple[int, int]]]:
    h = len(mask)
    w = len(mask[0]) if h else 0
    seen = [[False] * w for _ in range(h)]
    comps: List[List[Tuple[int, int]]] = []

    for y in range(h):
        for x in range(w):
            if not mask[y][x] or seen[y][x]:
                continue
            stack = [(x, y)]
            seen[y][x] = True
            comp: List[Tuple[int, int]] = []

            while stack:
                cx, cy = stack.pop()
                comp.append((cx, cy))
                for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and mask[ny][nx] and not seen[ny][nx]:
                        seen[ny][nx] = True
                        stack.append((nx, ny))

            comps.append(comp)

    return comps


def scale_whitespace(mask_empty: List[List[bool]]) -> List[List[bool]]:
    """
    IMPORTANT FIX:
    Scale each whitespace component SHAPE by 2× (nearest-neighbor), not its bbox.
    Center scaling around the component centroid.
    """
    comps = connected_components(mask_empty)
    out = [[False] * GRID for _ in range(GRID)]

    for comp in comps:
        xs = [p[0] for p in comp]
        ys = [p[1] for p in comp]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        w = maxx - minx + 1
        h = maxy - miny + 1

        # Build tight crop of the component (shape-accurate)
        crop = [[False] * w for _ in range(h)]
        sum_cx = 0.0
        sum_cy = 0.0
        for x, y in comp:
            crop[y - miny][x - minx] = True
            # centroid in cell-center coordinates
            sum_cx += (x + 0.5)
            sum_cy += (y + 0.5)

        n = len(comp)
        gcx = sum_cx / n  # global centroid (cell units)
        gcy = sum_cy / n

        # Scale crop by 2× using nearest-neighbor (cell replication)
        scaled_h = h * 2
        scaled_w = w * 2
        scaled = [[False] * scaled_w for _ in range(scaled_h)]
        for yy in range(h):
            row = crop[yy]
            for xx in range(w):
                if not row[xx]:
                    continue
                sy = yy * 2
                sx = xx * 2
                scaled[sy][sx] = True
                scaled[sy][sx + 1] = True
                scaled[sy + 1][sx] = True
                scaled[sy + 1][sx + 1] = True

        # Local centroid inside crop (in cell units, relative to crop origin)
        lcx = gcx - minx
        lcy = gcy - miny

        # After 2× scaling, centroid doubles in the scaled crop coordinate frame
        slcx = 2.0 * lcx
        slcy = 2.0 * lcy

        # Place scaled crop so its centroid matches the original centroid
        ox = int(round(gcx - slcx))
        oy = int(round(gcy - slcy))

        # Clamp placement so it stays inside GRID
        ox = max(0, min(ox, GRID - scaled_w))
        oy = max(0, min(oy, GRID - scaled_h))

        # OR into output mask
        for yy in range(scaled_h):
            ty = oy + yy
            if not (0 <= ty < GRID):
                continue
            srow = scaled[yy]
            orow = out[ty]
            for xx in range(scaled_w):
                if srow[xx]:
                    tx = ox + xx
                    if 0 <= tx < GRID:
                        orow[tx] = True

    return out


# ----------------------------
# Mask -> merged rectangles -> SVG path
# ----------------------------
def mask_to_rects(mask: List[List[bool]]) -> List[Tuple[int, int, int, int]]:
    """
    Convert True-mask (whitespace) to merged rectangles (cell coords),
    returned as (x0, y0, x1, y1) inclusive.
    """
    segs_by_row: List[List[Tuple[int, int]]] = []
    for y in range(GRID):
        segs = []
        x = 0
        while x < GRID:
            if not mask[y][x]:
                x += 1
                continue
            x0 = x
            while x < GRID and mask[y][x]:
                x += 1
            segs.append((x0, x - 1))
        segs_by_row.append(segs)

    rects: List[Tuple[int, int, int, int]] = []
    active: Dict[Tuple[int, int], Tuple[int, int, int, int]] = {}

    for y in range(GRID):
        new_active: Dict[Tuple[int, int], Tuple[int, int, int, int]] = {}
        for x0, x1 in segs_by_row[y]:
            key = (x0, x1)
            if key in active:
                rx0, ry0, rx1, _ = active[key]
                new_active[key] = (rx0, ry0, rx1, y)
            else:
                new_active[key] = (x0, y, x1, y)

        for key, rect in active.items():
            if key not in new_active:
                rects.append(rect)

        active = new_active

    rects.extend(active.values())
    return rects


def build_glyph_path_d(d_in: str) -> str:
    empty = empty_mask_from_path(d_in)
    scaled_empty = scale_whitespace(empty)
    rects = mask_to_rects(scaled_empty)

    parts = []
    # Base square stays the same
    parts.append(f"M 0 0 L {GLYPH} 0 L {GLYPH} {GLYPH} L 0 {GLYPH} Z")

    # Subtract scaled whitespace rectangles (evenodd)
    for x0, y0, x1, y1 in rects:
        rx = x0 * CELL
        ry = y0 * CELL
        rw = (x1 - x0 + 1) * CELL
        rh = (y1 - y0 + 1) * CELL
        parts.append(f"M {rx} {ry} L {rx+rw} {ry} L {rx+rw} {ry+rh} L {rx} {ry+rh} Z")

    return " ".join(parts)


# ----------------------------
# Main: rewrite sheet
# ----------------------------
def main():
    if len(sys.argv) != 3:
        print("Usage: python double-holes-sheet.py INPUT.svg OUTPUT.svg", file=sys.stderr)
        sys.exit(2)

    in_path, out_path = sys.argv[1], sys.argv[2]

    tree = ET.parse(in_path)
    root = tree.getroot()

    # Prevent ns0 prefixes by registering the default SVG namespace
    if root.tag.startswith("{"):
        SVG_NS = root.tag.split("}", 1)[0][1:]
    else:
        SVG_NS = "http://www.w3.org/2000/svg"
    ET.register_namespace("", SVG_NS)

    def local(tag: str) -> str:
        return tag.split("}", 1)[-1] if "}" in tag else tag

    out_svg = ET.Element(root.tag, dict(root.attrib))
    out_svg.set("shape-rendering", root.attrib.get("shape-rendering", "crispEdges"))

    for child in list(root):
        if local(child.tag) != "g":
            continue

        g = ET.SubElement(out_svg, child.tag, dict(child.attrib))

        path_el = None
        for sub in list(child):
            if local(sub.tag) == "path":
                path_el = sub
                break
        if path_el is None:
            continue

        d_in = path_el.attrib.get("d", "")
        d_out = build_glyph_path_d(d_in)

        ET.SubElement(g, path_el.tag, {
            "d": d_out,
            "fill": "#000",
            "fill-rule": "evenodd",
        })

    xml = ET.tostring(out_svg, encoding="unicode")
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(xml)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
