#!/usr/bin/env python3
"""
Generate 8x8 tile-letter SVGs (240x240), where the tile is fully black
(4 triangles per cell) and the letter is formed by COMMENTING OUT
(selected) triangles to reveal the white background.

- Writes: src/A.svg ... src/Z.svg
- A and R are EXACTLY matched to the user-provided carve examples.
- Other letters are generated with a minimal-carve heuristic so the result
  stays "carved" (sparse removals), not a normal bitmap font.

Edit strategy:
- Each triangle has an id like: r{row}c{col}-{top|right|bottom|left}
- Full-cell removal = comment out all 4 triangles of that cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

SIZE = 240
GRID = 8
CELL = SIZE // GRID  # 30
HALF = CELL // 2     # 15


# ----------------------------
# Exact manual overrides
# ----------------------------

# Remove whole cells (all 4 triangles) for these letters
MANUAL_REMOVE_CELLS: Dict[str, Set[Tuple[int, int]]] = {
    # EXACT A from your latest example:
    # comment out r3c3, r3c4, r7c3, r7c4 (whole cells)
    "A": {(3, 3), (3, 4), (7, 3), (7, 4)},

    # EXACT R from your earlier example:
    # whole cells: r2c4, r6c4, r7c4
    "R": {(2, 4), (6, 4), (7, 4)},
}

# Remove individual triangles for these letters
MANUAL_REMOVE_TRIS: Dict[str, Set[str]] = {
    # EXACT R from your earlier example:
    # comment out only these two triangles (keep the other two)
    "R": {"r4c7-top", "r4c7-right"},
}


# ----------------------------
# Helper geometry
# ----------------------------

def tri_points(r: int, c: int) -> Dict[str, str]:
    x0 = c * CELL
    y0 = r * CELL
    x1 = x0 + CELL
    y1 = y0 + CELL
    cx = x0 + HALF
    cy = y0 + HALF
    return {
        "top":    f"{x0},{y0} {x1},{y0} {cx},{cy}",
        "right":  f"{x1},{y0} {x1},{y1} {cx},{cy}",
        "bottom": f"{x1},{y1} {x0},{y1} {cx},{cy}",
        "left":   f"{x0},{y1} {x0},{y0} {cx},{cy}",
    }


def cell_tri_ids(r: int, c: int) -> List[str]:
    return [f"r{r}c{c}-top", f"r{r}c{c}-right", f"r{r}c{c}-bottom", f"r{r}c{c}-left"]


# ----------------------------
# Minimal-carve heuristic
# ----------------------------
# The goal here is NOT a normal font. It’s a sparse “carved” mask:
# start fully black, then remove only a thin set of background pixels
# that still hints the letter structure.

# A tiny 5x7 uppercase bitmap (ink=1) used only to infer where to carve.
# (Centered into 8x8; then we carve a thin subset of its background.)
FONT_5x7: Dict[str, List[str]] = {
    "A": [
        "01110",
        "10001",
        "10001",
        "11111",
        "10001",
        "10001",
        "10001",
    ],
    "B": [
        "11110",
        "10001",
        "10001",
        "11110",
        "10001",
        "10001",
        "11110",
    ],
    "C": [
        "01111",
        "10000",
        "10000",
        "10000",
        "10000",
        "10000",
        "01111",
    ],
    "D": [
        "11110",
        "10001",
        "10001",
        "10001",
        "10001",
        "10001",
        "11110",
    ],
    "E": [
        "11111",
        "10000",
        "10000",
        "11110",
        "10000",
        "10000",
        "11111",
    ],
    "F": [
        "11111",
        "10000",
        "10000",
        "11110",
        "10000",
        "10000",
        "10000",
    ],
    "G": [
        "01111",
        "10000",
        "10000",
        "10011",
        "10001",
        "10001",
        "01111",
    ],
    "H": [
        "10001",
        "10001",
        "10001",
        "11111",
        "10001",
        "10001",
        "10001",
    ],
    "I": [
        "11111",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
        "11111",
    ],
    "J": [
        "11111",
        "00010",
        "00010",
        "00010",
        "00010",
        "10010",
        "01100",
    ],
    "K": [
        "10001",
        "10010",
        "10100",
        "11000",
        "10100",
        "10010",
        "10001",
    ],
    "L": [
        "10000",
        "10000",
        "10000",
        "10000",
        "10000",
        "10000",
        "11111",
    ],
    "M": [
        "10001",
        "11011",
        "10101",
        "10101",
        "10001",
        "10001",
        "10001",
    ],
    "N": [
        "10001",
        "11001",
        "10101",
        "10011",
        "10001",
        "10001",
        "10001",
    ],
    "O": [
        "01110",
        "10001",
        "10001",
        "10001",
        "10001",
        "10001",
        "01110",
    ],
    "P": [
        "11110",
        "10001",
        "10001",
        "11110",
        "10000",
        "10000",
        "10000",
    ],
    "Q": [
        "01110",
        "10001",
        "10001",
        "10001",
        "10101",
        "10010",
        "01101",
    ],
    "R": [
        "11110",
        "10001",
        "10001",
        "11110",
        "10100",
        "10010",
        "10001",
    ],
    "S": [
        "01111",
        "10000",
        "10000",
        "01110",
        "00001",
        "00001",
        "11110",
    ],
    "T": [
        "11111",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
    ],
    "U": [
        "10001",
        "10001",
        "10001",
        "10001",
        "10001",
        "10001",
        "01110",
    ],
    "V": [
        "10001",
        "10001",
        "10001",
        "10001",
        "01010",
        "01010",
        "00100",
    ],
    "W": [
        "10001",
        "10001",
        "10001",
        "10101",
        "10101",
        "11011",
        "10001",
    ],
    "X": [
        "10001",
        "01010",
        "00100",
        "00100",
        "00100",
        "01010",
        "10001",
    ],
    "Y": [
        "10001",
        "01010",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
    ],
    "Z": [
        "11111",
        "00001",
        "00010",
        "00100",
        "01000",
        "10000",
        "11111",
    ],
}


def embed_5x7_into_8x8(letter: str) -> List[List[int]]:
    """Return 8x8 ink mask (1=ink, 0=background), centered."""
    pat = FONT_5x7[letter]
    ink = [[0 for _ in range(8)] for _ in range(8)]
    # center 5x7 into 8x8: left=1, top=0 (7 fits), add 1 col margin on both sides
    off_r = 0
    off_c = 1
    for r in range(7):
        for c in range(5):
            if pat[r][c] == "1":
                ink[off_r + r][off_c + c] = 1
    return ink


def flood_outside(bg: List[List[int]]) -> List[List[int]]:
    """Mark outside-connected background pixels. bg is 1 for background."""
    H, W = 8, 8
    outside = [[0 for _ in range(W)] for _ in range(H)]
    stack: List[Tuple[int, int]] = []
    # seed from borders where bg=1
    for x in range(W):
        if bg[0][x]: stack.append((0, x))
        if bg[H - 1][x]: stack.append((H - 1, x))
    for y in range(H):
        if bg[y][0]: stack.append((y, 0))
        if bg[y][W - 1]: stack.append((y, W - 1))

    while stack:
        y, x = stack.pop()
        if outside[y][x]:
            continue
        if not bg[y][x]:
            continue
        outside[y][x] = 1
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and not outside[ny][nx] and bg[ny][nx]:
                stack.append((ny, nx))
    return outside


def manhattan_dist_to_ink(ink: List[List[int]]) -> List[List[int]]:
    """Distance from each cell to nearest ink (Manhattan)."""
    H, W = 8, 8
    INF = 10**9
    dist = [[INF for _ in range(W)] for _ in range(H)]
    q: List[Tuple[int, int]] = []
    for y in range(H):
        for x in range(W):
            if ink[y][x]:
                dist[y][x] = 0
                q.append((y, x))
    # simple BFS since all edges weight 1
    head = 0
    while head < len(q):
        y, x = q[head]
        head += 1
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and dist[ny][nx] > dist[y][x] + 1:
                dist[ny][nx] = dist[y][x] + 1
                q.append((ny, nx))
    return dist


def thin_background_points(letter: str, budget: int) -> Set[Tuple[int, int]]:
    """
    Pick a sparse set of background cells to carve:
    - Always include enclosed holes (counters)
    - For outside background, carve only the 'deepest' cells (far from ink),
      up to a small budget. This yields a carved/engraved look.
    """
    ink = embed_5x7_into_8x8(letter)
    bg = [[1 - ink[r][c] for c in range(8)] for r in range(8)]
    outside = flood_outside(bg)
    holes = {(r, c) for r in range(8) for c in range(8) if bg[r][c] and not outside[r][c]}

    # candidate outside background points, scored by distance to ink
    dist = manhattan_dist_to_ink(ink)
    candidates: List[Tuple[int, int, int]] = []
    for r in range(8):
        for c in range(8):
            if outside[r][c]:
                candidates.append((dist[r][c], r, c))
    candidates.sort(reverse=True)  # deepest first

    carve: Set[Tuple[int, int]] = set(holes)

    # reserve some budget for holes; carve holes fully, then add outside points
    remaining = max(0, budget - len(carve))
    for d, r, c in candidates:
        if remaining <= 0:
            break
        # skip borders to keep the tile's frame heavy (matches your "use the square" vibe)
        if r in (0, 7) or c in (0, 7):
            continue
        if (r, c) in carve:
            continue
        # only carve meaningful background, not tiny noise
        if d <= 1:
            continue
        carve.add((r, c))
        remaining -= 1

    return carve


# ----------------------------
# SVG construction
# ----------------------------

@dataclass(frozen=True)
class RemovalSpec:
    remove_cells: Set[Tuple[int, int]]
    remove_tris: Set[str]


def removal_for_letter(letter: str) -> RemovalSpec:
    if letter in MANUAL_REMOVE_CELLS or letter in MANUAL_REMOVE_TRIS:
        return RemovalSpec(
            remove_cells=set(MANUAL_REMOVE_CELLS.get(letter, set())),
            remove_tris=set(MANUAL_REMOVE_TRIS.get(letter, set())),
        )

    # Heuristic for others: small budget that scales mildly with glyph complexity
    # (still intentionally sparse)
    # You can tune this number if you want more/less carving.
    budget = 6
    if letter in "BDEOPQR":
        budget = 7
    elif letter in "MW":
        budget = 8
    elif letter in "IJLT":
        budget = 5

    carve_cells = thin_background_points(letter, budget=budget)
    return RemovalSpec(remove_cells=carve_cells, remove_tris=set())


def svg_letter(letter: str) -> str:
    spec = removal_for_letter(letter)

    # Build a set of triangle ids to comment out
    remove_ids: Set[str] = set(spec.remove_tris)
    for (r, c) in spec.remove_cells:
        remove_ids.update(cell_tri_ids(r, c))

    lines: List[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{SIZE}" height="{SIZE}" viewBox="0 0 {SIZE} {SIZE}">')
    lines.append('    <!-- Background (initial square) -->')
    lines.append(f'    <rect x="0" y="0" width="{SIZE}" height="{SIZE}" fill="#fff"/>')
    lines.append('')
    lines.append('    <!-- 8×8 cells, each cell is 4 editable triangles (ids: r{row}c{col}-{top|right|bottom|left}) -->')
    lines.append('    <g fill="#000" shape-rendering="crispEdges">')

    # Emit polygons in row-major order; comment out removed ones.
    for r in range(GRID):
        for c in range(GRID):
            pts = tri_points(r, c)
            for part in ("top", "right", "bottom", "left"):
                pid = f"r{r}c{c}-{part}"
                poly = f'        <polygon id="{pid}" points="{pts[part]}"/>'
                if pid in remove_ids:
                    lines.append("<!--")
                    lines.append(poly)
                    lines.append("-->")
                else:
                    lines.append(poly)

    lines.append('    </g>')
    lines.append('</svg>')
    return "\n".join(lines)


def write_all(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(ord("A"), ord("Z") + 1):
        ch = chr(i)
        svg = svg_letter(ch)
        (out_dir / f"{ch}.svg").write_text(svg, encoding="utf-8")

    # Quick index page so you can eyeball everything at once
    index = [
        "<!doctype html>",
        "<meta charset='utf-8'/>",
        "<title>Tile Letters</title>",
        "<style>",
        "  body{font-family:system-ui, sans-serif; padding:16px}",
        "  .grid{display:grid; grid-template-columns:repeat(6, 240px); gap:16px}",
        "  figure{margin:0}",
        "  figcaption{margin-top:6px; text-align:center}",
        "  object{width:240px; height:240px; border:1px solid #ddd}",
        "</style>",
        "<div class='grid'>",
    ]
    for i in range(ord("A"), ord("Z") + 1):
        ch = chr(i)
        index += [
            "<figure>",
            f"  <object data='{ch}.svg' type='image/svg+xml'></object>",
            f"  <figcaption>{ch}</figcaption>",
            "</figure>",
        ]
    index += ["</div>"]
    (out_dir / "_index.html").write_text("\n".join(index), encoding="utf-8")


if __name__ == "__main__":
    write_all(Path("src"))
    print("Wrote src/A.svg ... src/Z.svg and src/_index.html")
