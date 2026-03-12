#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/generate-data.py

Reverse-engineer polygon-grid glyph SVGs into binary glyph data (0/1) and
optionally export that data back to structured SVG files.

Expected source filenames:
  src/character-u{codepoint}.svg   (hex codepoint, e.g. character-u0032.svg)

Expected polygon selectors inside the SVG (either style is supported):
  - id="r{row}c{col}-{top|right|bottom|left}"      (legacy)
  - class="r{row}c{col}-{top|right|bottom|left}"   (new)
    (class may contain multiple tokens; the matching token is used)

Each matching polygon selector that exists in the SVG is treated as an enabled
triangle (1). Missing triangles are treated as disabled (0). This naturally
supports hand-edited SVGs where disabled polygons were commented out.

Outputs (default):
  - data/glyphs.json        (portable data dump, includes bitstring)
  - data/glyphs_data.py     (Python module with GLYPHS dict, includes bitstring)
  - data/glyphs_bits.py     (compact Python module with codepoint -> bitstring)
  - config/mmxx.php         (runtime-friendly CakePHP config export)

Optional:
  - regenerated SVGs        (structured output from the binary data)

Examples:
  # Import existing SVGs and write JSON + Python files + CakePHP config
  py tools/generate-data.py

  # Also regenerate SVGs into src/generated-glyphs
  py tools/generate-data.py --export-svgs --svg-out-dir src/generated-glyphs

  # Round-trip back into src (overwrite originals carefully)
  py tools/generate-data.py --export-svgs --svg-out-dir src --overwrite

  # Only process selected glyphs
  py tools/generate-data.py --only 2 A U+03A9
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------------------
# Defaults / geometry
# ---------------------------------------------------------------------------

TRI_ORDER = ("top", "right", "bottom", "left")
TRI_INDEX = {name: i for i, name in enumerate(TRI_ORDER)}

DEFAULT_GRID_ROWS = 8
DEFAULT_GRID_COLS = 8
DEFAULT_CELL_SIZE = 30

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC_DIR = ROOT / "src"
DEFAULT_DATA_DIR = ROOT / "data"

DEFAULT_JSON_OUT = DEFAULT_DATA_DIR / "glyphs.json"
DEFAULT_PY_OUT = DEFAULT_DATA_DIR / "glyphs_data.py"
DEFAULT_BITS_OUT = DEFAULT_DATA_DIR / "glyphs_bits.py"
DEFAULT_CAKE_CONFIG_OUT = ROOT / "config" / "mmxx.php"

FILENAME_RE = re.compile(r"^character-u([0-9A-Fa-f]+)\.svg$")
POLYGON_ID_RE = re.compile(r"^r(\d+)c(\d+)-(top|right|bottom|left)$")


@dataclass
class GlyphRecord:
    codepoint: int
    source_hex: str
    source_file: Path
    grid: List[List[List[int]]]  # [row][col][tri] -> 0/1

    @property
    def char(self) -> str:
        try:
            return chr(self.codepoint)
        except ValueError:
            return ""

    def flattened_bits(self) -> List[int]:
        out: List[int] = []
        for row in self.grid:
            for cell in row:
                out.extend(cell)
        return out

    def bitstring(self) -> str:
        # For 8x8x4, this is exactly 256 chars long.
        return "".join(str(b) for b in self.flattened_bits())

    def enabled_count(self) -> int:
        return sum(self.flattened_bits())


def make_empty_grid(rows: int, cols: int) -> List[List[List[int]]]:
    return [[[0, 0, 0, 0] for _ in range(cols)] for _ in range(rows)]


def parse_codepoint_from_filename(path: Path) -> Tuple[int, str]:
    m = FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Not a glyph filename: {path.name}")
    hex_token = m.group(1)
    return int(hex_token, 16), hex_token.lower()


def _extract_polygon_selector(elem: ET.Element) -> str:
    """
    Accept both:
      - legacy: <polygon id="r0c0-top" .../>
      - new:    <polygon class="r0c0-top" .../>
    If class contains multiple tokens, return the first matching one.
    """
    # Legacy format
    selector = (elem.attrib.get("id") or "").strip()
    if selector:
        return selector

    # New format (class-based). class may contain multiple classes.
    class_attr = (elem.attrib.get("class") or "").strip()
    if class_attr:
        for token in class_attr.split():
            if POLYGON_ID_RE.match(token):
                return token

    return ""


def parse_glyph_svg(svg_path: Path, rows: int, cols: int) -> GlyphRecord:
    codepoint, source_hex = parse_codepoint_from_filename(svg_path)
    grid = make_empty_grid(rows, cols)

    try:
        tree = ET.parse(svg_path)
    except ET.ParseError as e:
        raise ValueError(f"{svg_path}: XML parse error: {e}") from e

    root = tree.getroot()
    seen: set[str] = set()
    warnings: List[str] = []

    for elem in root.iter():
        tag = elem.tag.split("}", 1)[-1]
        if tag != "polygon":
            continue

        selector = _extract_polygon_selector(elem)
        if not selector:
            continue

        m = POLYGON_ID_RE.match(selector)
        if not m:
            # Ignore unrelated polygon ids/classes
            continue

        if selector in seen:
            warnings.append(f"duplicate polygon selector '{selector}'")
            continue
        seen.add(selector)

        r = int(m.group(1))
        c = int(m.group(2))
        tri_name = m.group(3)

        if not (0 <= r < rows and 0 <= c < cols):
            warnings.append(f"polygon selector '{selector}' out of grid bounds ({rows}x{cols})")
            continue

        # If a polygon is present but explicitly hidden/none, treat it as disabled.
        display = (elem.attrib.get("display") or "").strip().lower()
        visibility = (elem.attrib.get("visibility") or "").strip().lower()
        opacity = (elem.attrib.get("opacity") or "").strip()
        fill = (elem.attrib.get("fill") or "").strip().lower()

        if display == "none" or visibility == "hidden" or opacity == "0":
            continue
        if fill in {"none", "white", "#fff", "#ffffff"}:
            continue

        grid[r][c][TRI_INDEX[tri_name]] = 1

    if warnings:
        print(f"[warn] {svg_path.name}: " + "; ".join(warnings), file=sys.stderr)

    return GlyphRecord(codepoint=codepoint, source_hex=source_hex, source_file=svg_path, grid=grid)


def iter_source_svgs(src_dir: Path) -> Iterable[Path]:
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src_dir}")
    for path in sorted(src_dir.glob("character-u*.svg")):
        if FILENAME_RE.match(path.name):
            yield path


def _is_printable_char(ch: str) -> bool:
    return ch.isprintable() and ch not in {"\n", "\r", "\t", "\x0b", "\x0c"}


def _parse_only_token(token: str) -> int:
    token = token.strip()
    if not token:
        raise ValueError("Empty token in --only")

    if token.upper().startswith("U+"):
        return int(token[2:], 16)
    if token.lower().startswith("0x"):
        return int(token[2:], 16)
    if token.isdigit():
        return int(token, 10)
    if len(token) == 1:
        return ord(token)
    if re.fullmatch(r"[0-9A-Fa-f]+", token):
        return int(token, 16)

    raise ValueError(f"Cannot parse glyph selector token: {token!r}")


def load_all_glyphs(src_dir: Path, rows: int, cols: int, only: set[int] | None = None) -> Dict[int, GlyphRecord]:
    glyphs: Dict[int, GlyphRecord] = {}
    for svg_path in iter_source_svgs(src_dir):
        rec = parse_glyph_svg(svg_path, rows=rows, cols=cols)
        if only is not None and rec.codepoint not in only:
            continue
        glyphs[rec.codepoint] = rec
    return dict(sorted(glyphs.items(), key=lambda kv: kv[0]))


def _char_repr(cp: int) -> str:
    try:
        ch = chr(cp)
    except ValueError:
        return ""
    return ch if _is_printable_char(ch) else ""


def _group_glyph_chars(glyphs: Dict[int, GlyphRecord]) -> dict:
    upper: List[str] = []
    lower: List[str] = []
    digits: List[str] = []
    punct: List[str] = []
    other: List[str] = []

    for cp in sorted(glyphs):
        ch = _char_repr(cp)
        if not ch:
            continue

        if "A" <= ch <= "Z":
            upper.append(ch)
        elif "a" <= ch <= "z":
            lower.append(ch)
        elif "0" <= ch <= "9":
            digits.append(ch)
        elif ch.isprintable() and not ch.isalnum() and not ch.isspace():
            punct.append(ch)
        else:
            other.append(ch)

    all_chars = upper + lower + digits + punct + other

    return {
        "all": "".join(all_chars),
        "uppercase": "".join(upper),
        "lowercase": "".join(lower),
        "digits": "".join(digits),
        "punct": "".join(punct),
        "other": "".join(other),
        "codepoints": sorted(glyphs.keys()),
    }


def _codepoints_to_ranges(codepoints: List[int]) -> List[List[int]]:
    if not codepoints:
        return []

    cps = sorted(set(codepoints))
    ranges: List[List[int]] = []

    start = prev = cps[0]
    for cp in cps[1:]:
        if cp == prev + 1:
            prev = cp
            continue
        ranges.append([start, prev])
        start = prev = cp
    ranges.append([start, prev])

    return ranges


def _php_scalar(value, indent: int = 0) -> str:
    """
    Render Python scalars/lists/dicts into readable PHP array syntax.
    """
    pad = " " * indent
    next_pad = " " * (indent + 4)

    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        escaped = (
            value.replace("\\", "\\\\")
                 .replace("'", "\\'")
                 .replace("\r", "\\r")
                 .replace("\n", "\\n")
                 .replace("\t", "\\t")
        )
        return f"'{escaped}'"

    if isinstance(value, list):
        if not value:
            return "[]"
        lines = ["["]
        for item in value:
            lines.append(f"{next_pad}{_php_scalar(item, indent + 4)},")
        lines.append(f"{pad}]")
        return "\n".join(lines)

    if isinstance(value, dict):
        if not value:
            return "[]"
        lines = ["["]
        for k, v in value.items():
            lines.append(f"{next_pad}{_php_scalar(str(k))} => {_php_scalar(v, indent + 4)},")
        lines.append(f"{pad}]")
        return "\n".join(lines)

    raise TypeError(f"Unsupported value for PHP export: {type(value)!r}")


def build_cake_config_payload(
    glyphs: Dict[int, GlyphRecord],
    rows: int,
    cols: int,
    cell_size: int,
) -> dict:
    grouped = _group_glyph_chars(glyphs)
    codepoints = grouped["codepoints"]
    unicode_ranges = _codepoints_to_ranges(codepoints)

    return {
        "MMXX": {
            "fontFamily": "mmxx",
            "gridRows": rows,
            "gridCols": cols,
            "cellSize": cell_size,
            "triangleOrder": list(TRI_ORDER),
            "glyphCount": len(glyphs),

            # Most useful runtime strings
            "chars": grouped["all"],
            "uppercase": grouped["uppercase"],
            "lowercase": grouped["lowercase"],
            "digits": grouped["digits"],
            "punct": grouped["punct"],
            "other": grouped["other"],

            # Helpful for demos, validators, front-end helpers
            "codepoints": codepoints,
            "unicodeRanges": unicode_ranges,

            # Convenience flags
            "hasUppercase": bool(grouped["uppercase"]),
            "hasLowercase": bool(grouped["lowercase"]),
            "hasDigits": bool(grouped["digits"]),
            "hasPunct": bool(grouped["punct"]),
        }
    }


def build_json_payload(
    glyphs: Dict[int, GlyphRecord],
    rows: int,
    cols: int,
    cell_size: int,
) -> dict:
    bit_count = rows * cols * len(TRI_ORDER)

    payload = {
        "meta": {
            "grid_rows": rows,
            "grid_cols": cols,
            "cell_size": cell_size,
            "triangle_order": list(TRI_ORDER),
            "bit_order": "row-major, cell-major, triangle-order(top,right,bottom,left)",
            "bit_count": bit_count,
            "filename_pattern": "character-u{hex}.svg",
        },
        "glyphs": {},
    }

    for cp, rec in glyphs.items():
        key = f"u{cp:04x}"
        payload["glyphs"][key] = {
            "codepoint": cp,
            "char": _char_repr(cp),
            "source_file": rec.source_file.name,
            "source_hex": rec.source_hex,
            "enabled_count": rec.enabled_count(),
            "bitstring": rec.bitstring(),
            "bits": rec.flattened_bits(),
            "grid": rec.grid,
        }

    return payload


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _py_repr_grid(grid: List[List[List[int]]], base_indent: str = " " * 8) -> str:
    """Pretty repr for nested [row][col][tri] list."""
    lines: List[str] = ["["]
    row_indent = base_indent + "    "
    for i, row in enumerate(grid):
        row_cells = ", ".join("[" + ", ".join(str(v) for v in cell) + "]" for cell in row)
        comma = "," if i < len(grid) - 1 else ""
        lines.append(f"{row_indent}[{row_cells}]{comma}")
    lines.append(base_indent + "]")
    return "\n".join(lines)


def write_python_module(
    glyphs: Dict[int, GlyphRecord],
    rows: int,
    cols: int,
    cell_size: int,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bit_count = rows * cols * len(TRI_ORDER)

    lines: List[str] = []
    lines.append("# Auto-generated by tools/generate-data.py")
    lines.append("# Do not edit by hand unless you know what you're doing.")
    lines.append("")
    lines.append(f"GRID_ROWS = {rows}")
    lines.append(f"GRID_COLS = {cols}")
    lines.append(f"CELL_SIZE = {cell_size}")
    lines.append(f"TRIANGLE_ORDER = {TRI_ORDER!r}")
    lines.append(f"BIT_COUNT = {bit_count}")
    lines.append("")
    lines.append("GLYPHS = {")
    for cp, rec in glyphs.items():
        ch = _char_repr(cp)
        lines.append(f"    0x{cp:04X}: {{")
        lines.append(f"        'char': {ch!r},")
        lines.append(f"        'source_file': {rec.source_file.name!r},")
        lines.append(f"        'source_hex': {rec.source_hex!r},")
        lines.append(f"        'enabled_count': {rec.enabled_count()},")
        lines.append(f"        'bitstring': {rec.bitstring()!r},")
        lines.append("        'grid': " + _py_repr_grid(rec.grid, base_indent=" " * 8))
        lines.append("    },")
    lines.append("}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_bits_module(
    glyphs: Dict[int, GlyphRecord],
    rows: int,
    cols: int,
    path: Path,
) -> None:
    """
    Write a compact codepoint -> bitstring module.
    For the default 8x8x4 layout, each string is 256 chars long.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    bit_count = rows * cols * len(TRI_ORDER)

    lines: List[str] = []
    lines.append("# Auto-generated by tools/generate-data.py")
    lines.append("# Compact glyph data: codepoint -> bitstring")
    lines.append("")
    lines.append(f"GRID_ROWS = {rows}")
    lines.append(f"GRID_COLS = {cols}")
    lines.append(f"TRIANGLE_ORDER = {TRI_ORDER!r}")
    lines.append(f"BIT_COUNT = {bit_count}")
    lines.append("")
    lines.append("GLYPH_BITS = {")
    for cp, rec in glyphs.items():
        ch = _char_repr(cp)
        comment = f"  # {ch}" if ch else ""
        lines.append(f"    0x{cp:04X}: {rec.bitstring()!r},{comment}")
    lines.append("}")
    lines.append("")
    lines.append("GLYPH_BITS_BY_KEY = {")
    for cp, rec in glyphs.items():
        key = f"u{cp:04x}"
        lines.append(f"    {key!r}: GLYPH_BITS[0x{cp:04X}],")
    lines.append("}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_cake_config(
    glyphs: Dict[int, GlyphRecord],
    rows: int,
    cols: int,
    cell_size: int,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = build_cake_config_payload(
        glyphs,
        rows=rows,
        cols=cols,
        cell_size=cell_size,
    )

    lines: List[str] = []
    lines.append("<?php")
    lines.append("declare(strict_types=1);")
    lines.append("")
    lines.append("// Auto-generated by tools/generate-data.py")
    lines.append("// Runtime-friendly config for CakePHP/plugin usage.")
    lines.append("")
    lines.append("return " + _php_scalar(payload, 0) + ";")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt_num(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.4f}".rstrip("0").rstrip(".")


def _points_for_triangle(row: int, col: int, tri: str, cell_size: int) -> str:
    x0 = col * cell_size
    y0 = row * cell_size
    x1 = x0 + cell_size
    y1 = y0 + cell_size
    cx = x0 + (cell_size / 2.0)
    cy = y0 + (cell_size / 2.0)

    if tri == "top":
        pts = [(x0, y0), (x1, y0), (cx, cy)]
    elif tri == "right":
        pts = [(x1, y0), (x1, y1), (cx, cy)]
    elif tri == "bottom":
        pts = [(x1, y1), (x0, y1), (cx, cy)]
    elif tri == "left":
        pts = [(x0, y1), (x0, y0), (cx, cy)]
    else:
        raise ValueError(f"Unknown triangle: {tri}")

    return " ".join(f"{_fmt_num(x)},{_fmt_num(y)}" for x, y in pts)


def _glyph_filename_for_export(rec: GlyphRecord, hex_width: int, preserve_imported_names: bool) -> str:
    if preserve_imported_names and rec.source_hex:
        return f"character-u{rec.source_hex}.svg"
    if hex_width > 0:
        return f"character-u{rec.codepoint:0{hex_width}x}.svg"
    return f"character-u{rec.codepoint:x}.svg"


def render_glyph_svg(
    rec: GlyphRecord,
    rows: int,
    cols: int,
    cell_size: int,
    fg_fill: str = "#000",
    bg_fill: str = "#fff",
) -> str:
    w = cols * cell_size
    h = rows * cell_size

    out: List[str] = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
    out.append("    <!-- Background (initial square) -->")
    out.append(f'    <rect x="0" y="0" width="{w}" height="{h}" fill="{bg_fill}"/>')
    out.append("")
    out.append("    <!-- Grid cells, each cell is 4 editable triangles (top/right/bottom/left) -->")
    out.append(f'    <g fill="{fg_fill}" shape-rendering="crispEdges">')

    for r in range(rows):
        for c in range(cols):
            cell = rec.grid[r][c]
            for tri_name, bit in zip(TRI_ORDER, cell):
                if not bit:
                    continue
                pts = _points_for_triangle(r, c, tri_name, cell_size)
                out.append(f'        <polygon id="r{r}c{c}-{tri_name}" points="{pts}"/>')
        if r != rows - 1:
            out.append("")

    out.append("    </g>")
    out.append("</svg>")
    out.append("")
    return "\n".join(out)


def export_svgs(
    glyphs: Dict[int, GlyphRecord],
    out_dir: Path,
    rows: int,
    cols: int,
    cell_size: int,
    fg_fill: str,
    bg_fill: str,
    overwrite: bool,
    hex_width: int,
    preserve_imported_names: bool,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for rec in glyphs.values():
        filename = _glyph_filename_for_export(
            rec,
            hex_width=hex_width,
            preserve_imported_names=preserve_imported_names,
        )
        out_path = out_dir / filename
        if out_path.exists() and not overwrite:
            print(f"[skip] {out_path} exists (use --overwrite to replace)")
            continue

        svg = render_glyph_svg(
            rec,
            rows=rows,
            cols=cols,
            cell_size=cell_size,
            fg_fill=fg_fill,
            bg_fill=bg_fill,
        )
        out_path.write_text(svg, encoding="utf-8")
        written += 1

    return written


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reverse-engineer polygon-grid glyph SVGs into binary data and export SVGs."
    )

    p.add_argument(
        "--src-dir",
        type=Path,
        default=DEFAULT_SRC_DIR,
        help=f"Directory containing character-u*.svg (default: {DEFAULT_SRC_DIR})",
    )
    p.add_argument("--rows", type=int, default=DEFAULT_GRID_ROWS, help="Grid rows (default: 8)")
    p.add_argument("--cols", type=int, default=DEFAULT_GRID_COLS, help="Grid cols (default: 8)")
    p.add_argument("--cell-size", type=int, default=DEFAULT_CELL_SIZE, help="Cell size in pixels for SVG export (default: 30)")

    p.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional glyph selectors (e.g. A 2 U+03A9 0x41 65). If omitted, import all glyphs.",
    )

    p.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT, help=f"Write JSON glyph dump (default: {DEFAULT_JSON_OUT})")
    p.add_argument("--no-json", action="store_true", help="Do not write JSON output")

    p.add_argument("--py-out", type=Path, default=DEFAULT_PY_OUT, help=f"Write Python glyph module (default: {DEFAULT_PY_OUT})")
    p.add_argument("--no-py", action="store_true", help="Do not write Python module output")

    p.add_argument("--bits-out", type=Path, default=DEFAULT_BITS_OUT, help=f"Write compact bitstring Python module (default: {DEFAULT_BITS_OUT})")
    p.add_argument("--no-bits", action="store_true", help="Do not write compact bitstring output")

    p.add_argument(
        "--cake-config-out",
        type=Path,
        default=DEFAULT_CAKE_CONFIG_OUT,
        help=f"Write CakePHP runtime config (default: {DEFAULT_CAKE_CONFIG_OUT})",
    )
    p.add_argument(
        "--no-cake-config",
        action="store_true",
        help="Do not write CakePHP config output",
    )

    p.add_argument("--export-svgs", action="store_true", help="Export SVGs from the parsed binary glyph data")
    p.add_argument("--svg-out-dir", type=Path, default=DEFAULT_SRC_DIR / "generated-glyphs", help="Output directory for exported SVGs")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing exported SVGs")
    p.add_argument("--hex-width", type=int, default=4, help="Hex padding width for export filenames if not preserving imported names (default: 4)")
    p.add_argument(
        "--no-preserve-imported-names",
        action="store_true",
        help="Do not reuse imported filename hex tokens when exporting",
    )
    p.add_argument("--fg-fill", default="#000", help="Foreground fill color for exported polygons (default: #000)")
    p.add_argument("--bg-fill", default="#fff", help="Background fill color for exported SVGs (default: #fff)")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.rows <= 0 or args.cols <= 0 or args.cell_size <= 0:
        print("rows, cols, and cell-size must be positive integers", file=sys.stderr)
        return 2

    only_set: set[int] | None = None
    if args.only:
        only_set = set()
        for token in args.only:
            try:
                only_set.add(_parse_only_token(token))
            except ValueError as e:
                print(f"Invalid --only token {token!r}: {e}", file=sys.stderr)
                return 2

    glyphs = load_all_glyphs(args.src_dir, rows=args.rows, cols=args.cols, only=only_set)

    if not glyphs:
        print(f"No glyph SVGs found in {args.src_dir} matching character-u*.svg", file=sys.stderr)
        return 1

    bit_count = args.rows * args.cols * len(TRI_ORDER)
    total_enabled = sum(rec.enabled_count() for rec in glyphs.values())

    print(f"Imported {len(glyphs)} glyph(s) from {args.src_dir}")
    print(f"Grid: {args.rows}x{args.cols} cells -> {bit_count} bits per glyph")
    print(f"Total enabled triangles across all glyphs: {total_enabled}")

    if not args.no_json:
        payload = build_json_payload(glyphs, rows=args.rows, cols=args.cols, cell_size=args.cell_size)
        write_json(payload, args.json_out)
        print(f"Wrote JSON data: {args.json_out}")

    if not args.no_py:
        write_python_module(glyphs, rows=args.rows, cols=args.cols, cell_size=args.cell_size, path=args.py_out)
        print(f"Wrote Python module: {args.py_out}")

    if not args.no_bits:
        write_bits_module(glyphs, rows=args.rows, cols=args.cols, path=args.bits_out)
        print(f"Wrote compact bitstrings: {args.bits_out}")

    if not args.no_cake_config:
        write_cake_config(
            glyphs,
            rows=args.rows,
            cols=args.cols,
            cell_size=args.cell_size,
            path=args.cake_config_out,
        )
        print(f"Wrote CakePHP config: {args.cake_config_out}")

    if args.export_svgs:
        written = export_svgs(
            glyphs,
            out_dir=args.svg_out_dir,
            rows=args.rows,
            cols=args.cols,
            cell_size=args.cell_size,
            fg_fill=args.fg_fill,
            bg_fill=args.bg_fill,
            overwrite=args.overwrite,
            hex_width=max(0, args.hex_width),
            preserve_imported_names=not args.no_preserve_imported_names,
        )
        print(f"Exported {written} SVG(s) to {args.svg_out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())