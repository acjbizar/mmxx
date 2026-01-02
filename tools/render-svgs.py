#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from collections import deque
from typing import Dict, List, Set, Tuple

# ---------------------------------------------
# SVG / grid settings
# ---------------------------------------------
OUT_DIR = Path("src")

GRID = 8
CELL = 30
SIZE = GRID * CELL

INCLUDE_BLUE_GRID = True
GRID_COLOR = "#1e90ff"
GRID_STROKE = 2

PARTS = ["top", "right", "bottom", "left"]
BIT_FOR = {"top": 1, "right": 2, "bottom": 4, "left": 8}

# ---------------------------------------------
# 8×8 bitmap font used ONLY as an "auto-carve hint"
# (Manual carve overrides are the authoritative shapes.)
# ---------------------------------------------
FONT: Dict[str, List[str]] = {
    "A": ["00111100","01100110","11000011","11000011","11111111","11000011","11000011","00000000"],
    "B": ["11111100","11000110","11000110","11111100","11000110","11000110","11111100","00000000"],
    "C": ["00111110","01100011","11000000","11000000","11000000","01100011","00111110","00000000"],
    "D": ["11111100","11000110","11000011","11000011","11000011","11000110","11111100","00000000"],
    "E": ["11111111","11000000","11000000","11111110","11000000","11000000","11111111","00000000"],
    "F": ["11111111","11000000","11000000","11111110","11000000","11000000","11000000","00000000"],
    "G": ["00111110","01100011","11000000","11011111","11000011","01100011","00111110","00000000"],
    "H": ["11000011","11000011","11000011","11111111","11000011","11000011","11000011","00000000"],
    "I": ["00111100","00011000","00011000","00011000","00011000","00011000","00111100","00000000"],
    "J": ["00011111","00000110","00000110","00000110","11000110","11000110","01111100","00000000"],
    "K": ["11000110","11001100","11011000","11110000","11011000","11001100","11000110","00000000"],
    "L": ["11000000","11000000","11000000","11000000","11000000","11000000","11111111","00000000"],
    "M": ["11000011","11100111","11111111","11011011","11000011","11000011","11000011","00000000"],
    "N": ["11000011","11100011","11110011","11011011","11001111","11000111","11000011","00000000"],
    "O": ["00111100","01100110","11000011","11000011","11000011","01100110","00111100","00000000"],
    "P": ["11111100","11000110","11000110","11111100","11000000","11000000","11000000","00000000"],
    "Q": ["00111100","01100110","11000011","11000011","11011011","01100110","00111101","00000000"],
    "R": ["11111100","11000110","11000110","11111100","11011000","11001100","11000110","00000000"],
    "S": ["01111110","11000011","11000000","01111100","00000011","11000011","01111110","00000000"],
    "T": ["11111111","00011000","00011000","00011000","00011000","00011000","00011000","00000000"],
    "U": ["11000011","11000011","11000011","11000011","11000011","11000011","01111110","00000000"],
    "V": ["11000011","11000011","11000011","11000011","01100110","00111100","00011000","00000000"],
    "W": ["11000011","11000011","11000011","11011011","11111111","11100111","11000011","00000000"],
    "X": ["11000011","01100110","00111100","00011000","00111100","01100110","11000011","00000000"],
    "Y": ["11000011","01100110","00111100","00011000","00011000","00011000","00011000","00000000"],
    "Z": ["11111111","00000110","00001100","00011000","00110000","01100000","11111111","00000000"],
}

def bitmap_to_bool(rows: List[str]) -> List[List[bool]]:
    return [[ch == "1" for ch in row] for row in rows]

# ---------------------------------------------
# Manual carve helpers (THIS is how your examples work)
# ---------------------------------------------
def cell_triangles(r: int, c: int) -> Set[str]:
    return {f"r{r}c{c}-{p}" for p in PARTS}

def tri(r: int, c: int, part: str) -> str:
    return f"r{r}c{c}-{part}"

# EXACT matches from your examples:
MANUAL_CARVE: Dict[str, Set[str]] = {
    # G from your screenshot: white blocks at r3c4..r3c7 plus r4c4 (full cells)
    "G": set().union(
        *[cell_triangles(3, c) for c in (4,5,6,7)],
        cell_triangles(4, 4),
    ),

    # R from your pasted markup (exact):
    # - remove full cell r2c4
    # - remove r4c7 top+right
    # - remove full cells r6c4 and r7c4
    "R": set().union(
        cell_triangles(2, 4),
        {tri(4, 7, "top"), tri(4, 7, "right")},
        cell_triangles(6, 4),
        cell_triangles(7, 4),
    ),

    # NOTE: Add "J" here to match your exact example for J.
    # "J": {...},
}

# ---------------------------------------------
# Auto-carve (fallback) — minimalist, same principle:
# carve only a few triangles to suggest letter edges + carve counters fully.
# ---------------------------------------------
def find_enclosed_holes(ink: List[List[bool]]) -> List[List[bool]]:
    H = W = 8
    bg = [[not ink[r][c] for c in range(W)] for r in range(H)]
    seen = [[False] * W for _ in range(H)]
    q: deque[Tuple[int,int]] = deque()

    def push(r: int, c: int) -> None:
        seen[r][c] = True
        q.append((r, c))

    # flood-fill background from border
    for r in range(H):
        for c in (0, W - 1):
            if bg[r][c] and not seen[r][c]:
                push(r, c)
    for c in range(W):
        for r in (0, H - 1):
            if bg[r][c] and not seen[r][c]:
                push(r, c)

    while q:
        r, c = q.popleft()
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and bg[rr][cc] and not seen[rr][cc]:
                push(rr, cc)

    return [[bg[r][c] and not seen[r][c] for c in range(W)] for r in range(H)]

def auto_carve_ids(letter: str) -> Set[str]:
    ink = bitmap_to_bool(FONT[letter])
    holes = find_enclosed_holes(ink)

    carve: Set[str] = set()

    for r in range(8):
        for c in range(8):
            if holes[r][c]:
                # carve counters as full blocks (your R example does this)
                carve |= cell_triangles(r, c)
                continue

            if ink[r][c]:
                continue

            # Minimal “edge hint” only:
            # If this background cell is immediately to the RIGHT of ink,
            # carve ONLY its LEFT triangle (thin cut).
            if c > 0 and ink[r][c - 1]:
                carve.add(tri(r, c, "left"))

            # If this background cell is immediately BELOW ink,
            # carve ONLY its TOP triangle (thin cut).
            if r > 0 and ink[r - 1][c]:
                carve.add(tri(r, c, "top"))

    return carve

# ---------------------------------------------
# SVG rendering
# ---------------------------------------------
def triangle_points(r: int, c: int, part: str) -> str:
    x0, y0 = c * CELL, r * CELL
    x1, y1 = x0 + CELL, y0 + CELL
    cx, cy = x0 + CELL // 2, y0 + CELL // 2

    if part == "top":
        return f"{x0},{y0} {x1},{y0} {cx},{cy}"
    if part == "right":
        return f"{x1},{y0} {x1},{y1} {cx},{cy}"
    if part == "bottom":
        return f"{x1},{y1} {x0},{y1} {cx},{cy}"
    if part == "left":
        return f"{x0},{y1} {x0},{y0} {cx},{cy}"
    raise ValueError(part)

def render_svg(carve_ids: Set[str], include_grid: bool = True) -> str:
    out: List[str] = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{SIZE}" height="{SIZE}" viewBox="0 0 {SIZE} {SIZE}">')
    out.append("    <!-- Background (initial square) -->")
    out.append(f'    <rect x="0" y="0" width="{SIZE}" height="{SIZE}" fill="#fff"/>')
    out.append("")
    out.append("    <!-- 8×8 cells, each cell is 4 editable triangles (ids: r{row}c{col}-{top|right|bottom|left}) -->")
    out.append('    <g fill="#000" shape-rendering="crispEdges">')

    for r in range(GRID):
        for c in range(GRID):
            in_comment = False
            for part in PARTS:
                pid = f"r{r}c{c}-{part}"
                line = f'        <polygon id="{pid}" points="{triangle_points(r, c, part)}"/>'

                if pid in carve_ids:
                    if not in_comment:
                        out.append("        <!--")
                        in_comment = True
                    out.append(line)
                else:
                    if in_comment:
                        out.append("        -->")
                        in_comment = False
                    out.append(line)

            if in_comment:
                out.append("        -->")

    out.append("    </g>")

    if include_grid:
        out.append("")
        out.append("    <!-- Blue grid lines on top -->")
        out.append(f'    <g fill="none" stroke="{GRID_COLOR}" stroke-width="{GRID_STROKE}" shape-rendering="crispEdges">')
        for x in range(0, SIZE + 1, CELL):
            out.append(f'        <line x1="{x}" y1="0" x2="{x}" y2="{SIZE}"/>')
        for y in range(0, SIZE + 1, CELL):
            out.append(f'        <line x1="0" y1="{y}" x2="{SIZE}" y2="{y}"/>')
        out.append("    </g>")

    out.append("</svg>")
    return "\n".join(out) + "\n"

# ---------------------------------------------
# Main
# ---------------------------------------------
def carve_for_letter(ch: str) -> Set[str]:
    if ch in MANUAL_CARVE:
        return MANUAL_CARVE[ch]
    return auto_carve_ids(ch)

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        carve = carve_for_letter(ch)
        svg = render_svg(carve, include_grid=INCLUDE_BLUE_GRID)
        (OUT_DIR / f"{ch}.svg").write_text(svg, encoding="utf-8")

    print(f"Wrote 26 SVGs to {OUT_DIR.resolve()}")
    print("Manual exact-match letters:", ", ".join(sorted(MANUAL_CARVE.keys())))

if __name__ == "__main__":
    main()
