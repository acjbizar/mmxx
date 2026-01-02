#!/usr/bin/env python3
"""
Generate 8×8 triangle-cell SVGs for ALL capital letters A–Z into ./src/

Method matches your examples:
- Start with ALL triangles present (black field on white).
- "Whitespace" is created by COMMENTING OUT specific polygons.
- We carve a minimal triangle-outline ("trench") around a black letter shape,
  plus fully carved enclosed counters (holes) where they exist.

Edit-friendly:
- Every triangle has a stable id: r{row}c{col}-{top|right|bottom|left}
- Comment blocks are emitted per-cell in runs (like your manual approach).
"""

from __future__ import annotations

from pathlib import Path
from collections import deque
from typing import Dict, List

# --- Output / geometry --------------------------------------------------------

OUT_DIR = Path("src")

GRID = 8
CELL = 30
SIZE = GRID * CELL

INCLUDE_BLUE_GRID = True
GRID_COLOR = "#1e90ff"
GRID_STROKE = 2

# Triangle order in each cell (matches your markup style)
PARTS = ["top", "right", "bottom", "left"]

# 4-bit mask per cell: which triangles to "carve" (comment out)
BIT_FOR = {"top": 1, "right": 2, "bottom": 4, "left": 8}

# If True, enclosed counters (holes) are carved as full cells (all 4 triangles)
CARVE_HOLES_AS_FULL_CELLS = True

# --- 8×8 bitmap font (black letter guide) ------------------------------------
# 1 = "letter stroke should be black"; we'll carve whitespace around it.
FONT: Dict[str, List[str]] = {
    "A": [
        "00111100",
        "01100110",
        "11000011",
        "11000011",
        "11111111",
        "11000011",
        "11000011",
        "00000000",
    ],
    "B": [
        "11111100",
        "11000110",
        "11000110",
        "11111100",
        "11000110",
        "11000110",
        "11111100",
        "00000000",
    ],
    "C": [
        "00111110",
        "01100011",
        "11000000",
        "11000000",
        "11000000",
        "01100011",
        "00111110",
        "00000000",
    ],
    "D": [
        "11111100",
        "11000110",
        "11000011",
        "11000011",
        "11000011",
        "11000110",
        "11111100",
        "00000000",
    ],
    "E": [
        "11111111",
        "11000000",
        "11000000",
        "11111110",
        "11000000",
        "11000000",
        "11111111",
        "00000000",
    ],
    "F": [
        "11111111",
        "11000000",
        "11000000",
        "11111110",
        "11000000",
        "11000000",
        "11000000",
        "00000000",
    ],
    "G": [
        "00111110",
        "01100011",
        "11000000",
        "11011111",
        "11000011",
        "01100011",
        "00111110",
        "00000000",
    ],
    "H": [
        "11000011",
        "11000011",
        "11000011",
        "11111111",
        "11000011",
        "11000011",
        "11000011",
        "00000000",
    ],
    "I": [
        "00111100",
        "00011000",
        "00011000",
        "00011000",
        "00011000",
        "00011000",
        "00111100",
        "00000000",
    ],
    "J": [
        "00011111",
        "00000110",
        "00000110",
        "00000110",
        "11000110",
        "11000110",
        "01111100",
        "00000000",
    ],
    "K": [
        "11000110",
        "11001100",
        "11011000",
        "11110000",
        "11011000",
        "11001100",
        "11000110",
        "00000000",
    ],
    "L": [
        "11000000",
        "11000000",
        "11000000",
        "11000000",
        "11000000",
        "11000000",
        "11111111",
        "00000000",
    ],
    "M": [
        "11000011",
        "11100111",
        "11111111",
        "11011011",
        "11000011",
        "11000011",
        "11000011",
        "00000000",
    ],
    "N": [
        "11000011",
        "11100011",
        "11110011",
        "11011011",
        "11001111",
        "11000111",
        "11000011",
        "00000000",
    ],
    "O": [
        "00111100",
        "01100110",
        "11000011",
        "11000011",
        "11000011",
        "01100110",
        "00111100",
        "00000000",
    ],
    "P": [
        "11111100",
        "11000110",
        "11000110",
        "11111100",
        "11000000",
        "11000000",
        "11000000",
        "00000000",
    ],
    "Q": [
        "00111100",
        "01100110",
        "11000011",
        "11000011",
        "11011011",
        "01100110",
        "00111101",
        "00000000",
    ],
    "R": [
        "11111100",
        "11000110",
        "11000110",
        "11111100",
        "11011000",
        "11001100",
        "11000110",
        "00000000",
    ],
    "S": [
        "01111110",
        "11000011",
        "11000000",
        "01111100",
        "00000011",
        "11000011",
        "01111110",
        "00000000",
    ],
    "T": [
        "11111111",
        "00011000",
        "00011000",
        "00011000",
        "00011000",
        "00011000",
        "00011000",
        "00000000",
    ],
    "U": [
        "11000011",
        "11000011",
        "11000011",
        "11000011",
        "11000011",
        "11000011",
        "01111110",
        "00000000",
    ],
    "V": [
        "11000011",
        "11000011",
        "11000011",
        "11000011",
        "01100110",
        "00111100",
        "00011000",
        "00000000",
    ],
    "W": [
        "11000011",
        "11000011",
        "11000011",
        "11011011",
        "11111111",
        "11100111",
        "11000011",
        "00000000",
    ],
    "X": [
        "11000011",
        "01100110",
        "00111100",
        "00011000",
        "00111100",
        "01100110",
        "11000011",
        "00000000",
    ],
    "Y": [
        "11000011",
        "01100110",
        "00111100",
        "00011000",
        "00011000",
        "00011000",
        "00011000",
        "00000000",
    ],
    "Z": [
        "11111111",
        "00000110",
        "00001100",
        "00011000",
        "00110000",
        "01100000",
        "11111111",
        "00000000",
    ],
}

# --- Core logic ---------------------------------------------------------------

def bitmap_to_bool(rows: List[str]) -> List[List[bool]]:
    if len(rows) != 8 or any(len(r) != 8 for r in rows):
        raise ValueError("Each glyph must be 8 rows of 8 chars.")
    return [[ch == "1" for ch in row] for row in rows]


def find_enclosed_holes(ink: List[List[bool]]) -> List[List[bool]]:
    """
    Background pixels NOT connected to the border (4-neighborhood) are holes.
    These correspond to counters inside letters (A/B/D/O/P/Q/R, etc.).
    """
    H = W = 8
    bg = [[not ink[r][c] for c in range(W)] for r in range(H)]
    seen = [[False] * W for _ in range(H)]
    q = deque()

    def push(r: int, c: int) -> None:
        seen[r][c] = True
        q.append((r, c))

    # Start flood-fill from border background
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
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and bg[rr][cc] and not seen[rr][cc]:
                push(rr, cc)

    holes = [[bg[r][c] and not seen[r][c] for c in range(W)] for r in range(H)]
    return holes


def build_carve_mask(ink: List[List[bool]]) -> List[List[int]]:
    """
    Returns an 8×8 int grid. Each int is a 4-bit mask for which triangles
    to comment out in that cell to create whitespace.

    Outline carving:
    - For any BACKGROUND cell adjacent (N/E/S/W) to an INK cell,
      carve the triangle on the side facing that ink neighbor.
    Hole carving:
    - For enclosed background holes, optionally carve the full cell (all triangles).
    """
    H = W = 8
    holes = find_enclosed_holes(ink)
    mask = [[0] * W for _ in range(H)]

    for r in range(H):
        for c in range(W):
            if holes[r][c] and CARVE_HOLES_AS_FULL_CELLS:
                mask[r][c] = 15  # carve whole cell (top|right|bottom|left)
                continue

            if ink[r][c]:
                continue  # keep ink cells fully black (no carving inside strokes)

            # Carve minimal "trenches" around ink using triangles
            if r > 0 and ink[r - 1][c]:
                mask[r][c] |= BIT_FOR["top"]
            if c < W - 1 and ink[r][c + 1]:
                mask[r][c] |= BIT_FOR["right"]
            if r < H - 1 and ink[r + 1][c]:
                mask[r][c] |= BIT_FOR["bottom"]
            if c > 0 and ink[r][c - 1]:
                mask[r][c] |= BIT_FOR["left"]

    return mask


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
    raise ValueError(f"Unknown part: {part}")


def render_svg(letter: str, carve_mask: List[List[int]]) -> str:
    out: List[str] = []

    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{SIZE}" height="{SIZE}" viewBox="0 0 {SIZE} {SIZE}">')
    out.append("    <!-- Background (initial square) -->")
    out.append(f'    <rect x="0" y="0" width="{SIZE}" height="{SIZE}" fill="#fff"/>')
    out.append("")
    out.append("    <!-- 8×8 cells, each cell is 4 editable triangles (ids: r{row}c{col}-{top|right|bottom|left}) -->")
    out.append('    <g fill="#000" shape-rendering="crispEdges">')

    for r in range(GRID):
        for c in range(GRID):
            m = carve_mask[r][c]

            # Determine which parts to carve in this cell
            carve_parts = {p for p in PARTS if (m & BIT_FOR[p])}

            # Emit polygons in fixed order, using comment *runs* (like your hand-edits)
            in_comment = False
            for p in PARTS:
                poly = f'        <polygon id="r{r}c{c}-{p}" points="{triangle_points(r, c, p)}"/>'

                if p in carve_parts:
                    if not in_comment:
                        out.append("        <!--")
                        in_comment = True
                    out.append(poly)
                else:
                    if in_comment:
                        out.append("        -->")
                        in_comment = False
                    out.append(poly)

            if in_comment:
                out.append("        -->")

    out.append("    </g>")

    if INCLUDE_BLUE_GRID:
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        ink = bitmap_to_bool(FONT[ch])
        mask = build_carve_mask(ink)
        svg = render_svg(ch, mask)
        (OUT_DIR / f"{ch}.svg").write_text(svg, encoding="utf-8")

    print(f"Wrote 26 SVG files to: {OUT_DIR.resolve()}")
    print(f"Grid: {INCLUDE_BLUE_GRID}, Holes as full cells: {CARVE_HOLES_AS_FULL_CELLS}")


if __name__ == "__main__":
    main()
