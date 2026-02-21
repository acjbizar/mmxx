#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/generate-svgs.py

Generate glyph SVGs from binary glyph data (the reverse of generate-data.py).

Supported input sources:
- data/glyphs_bits.py   (default, compact bitstrings)
- data/glyphs.json
- data/glyphs_data.py   (full Python module)

Output SVGs:
- src/character-u{codepoint}.svg
- src/character-u{codepoint}-alt.svg   (inverse occupancy: 0=active, 1=inactive)

Polygon selector format in output:
- class="r{row}c{col}-{top|right|bottom|left}"
  (uses class instead of id)

The SVG also includes character metadata in sensible places:
- <title> and <desc>
- root attributes: data-codepoint, data-char, data-variant

Examples:
  # Default: read data/glyphs_bits.py and write into src/ (normal + alt)
  py tools/generate-svgs.py

  # Read from JSON instead
  py tools/generate-svgs.py --source data/glyphs.json

  # Export only a few glyphs
  py tools/generate-svgs.py --only A 2 U+03A9

  # Export into another directory
  py tools/generate-svgs.py --out-dir src/generated-glyphs

  # Skip the -alt inverse files
  py tools/generate-svgs.py --no-alt
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from xml.sax.saxutils import escape as xml_escape

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_GRID_ROWS = 8
DEFAULT_GRID_COLS = 8
DEFAULT_CELL_SIZE = 30
DEFAULT_TRI_ORDER = ("top", "right", "bottom", "left")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_SRC_DIR = ROOT / "src"

DEFAULT_SOURCE_CANDIDATES = [
    DEFAULT_DATA_DIR / "glyphs_bits.py",
    DEFAULT_DATA_DIR / "glyphs.json",
    DEFAULT_DATA_DIR / "glyphs_data.py",
]

FILENAME_HEX_RE = re.compile(r"^u([0-9A-Fa-f]+)$")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class GlyphRecord:
    codepoint: int
    bitstring: str
    source_hex: str | None = None
    char_hint: str | None = None


@dataclass
class GlyphMeta:
    rows: int
    cols: int
    cell_size: int
    triangle_order: Tuple[str, ...]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_printable_char(ch: str) -> bool:
    return ch.isprintable() and ch not in {"\n", "\r", "\t", "\x0b", "\x0c"}


def _safe_char_for_attr(ch: str) -> str:
    return ch if _is_printable_char(ch) else ""


def _unicode_name(cp: int) -> str:
    try:
        ch = chr(cp)
    except ValueError:
        return ""
    try:
        return unicodedata.name(ch)
    except ValueError:
        return ""


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
        raise ValueError(f"Unknown triangle side: {tri!r}")

    return " ".join(f"{_fmt_num(x)},{_fmt_num(y)}" for x, y in pts)


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


def _bit_count(meta: GlyphMeta) -> int:
    return meta.rows * meta.cols * len(meta.triangle_order)


def _validate_bitstring(bitstring: str, expected_len: int, label: str) -> None:
    if len(bitstring) != expected_len:
        raise ValueError(f"{label}: bitstring length {len(bitstring)} != expected {expected_len}")
    bad = set(bitstring) - {"0", "1"}
    if bad:
        raise ValueError(f"{label}: bitstring contains non-binary chars: {sorted(bad)}")


def _grid_to_bitstring(grid: Sequence[Sequence[Sequence[int]]], triangle_order_len: int) -> str:
    bits: List[str] = []
    for row in grid:
        for cell in row:
            if len(cell) != triangle_order_len:
                raise ValueError(f"Grid cell has {len(cell)} entries, expected {triangle_order_len}")
            for v in cell:
                bits.append("1" if int(v) else "0")
    return "".join(bits)


def _choose_default_source() -> Path:
    for p in DEFAULT_SOURCE_CANDIDATES:
        if p.exists():
            return p
    return DEFAULT_SOURCE_CANDIDATES[0]


# ---------------------------------------------------------------------------
# Loading sources
# ---------------------------------------------------------------------------

def _load_python_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Python module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_from_bits_py(path: Path) -> Tuple[GlyphMeta, Dict[int, GlyphRecord]]:
    mod = _load_python_module(path)

    rows = int(getattr(mod, "GRID_ROWS", DEFAULT_GRID_ROWS))
    cols = int(getattr(mod, "GRID_COLS", DEFAULT_GRID_COLS))
    cell_size = int(getattr(mod, "CELL_SIZE", DEFAULT_CELL_SIZE)) if hasattr(mod, "CELL_SIZE") else DEFAULT_CELL_SIZE
    triangle_order = tuple(getattr(mod, "TRIANGLE_ORDER", DEFAULT_TRI_ORDER))
    meta = GlyphMeta(rows=rows, cols=cols, cell_size=cell_size, triangle_order=triangle_order)

    if hasattr(mod, "GLYPH_BITS"):
        raw = getattr(mod, "GLYPH_BITS")
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: GLYPH_BITS is not a dict")
        glyphs: Dict[int, GlyphRecord] = {}
        for cp, bitstring in raw.items():
            cp_int = int(cp)
            bs = str(bitstring)
            _validate_bitstring(bs, _bit_count(meta), f"{path.name} cp=0x{cp_int:04X}")
            glyphs[cp_int] = GlyphRecord(codepoint=cp_int, bitstring=bs)
        return meta, dict(sorted(glyphs.items()))

    if hasattr(mod, "GLYPHS"):
        raw = getattr(mod, "GLYPHS")
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: GLYPHS is not a dict")
        glyphs = {}
        for cp, payload in raw.items():
            cp_int = int(cp)
            if isinstance(payload, dict) and "bitstring" in payload:
                bs = str(payload["bitstring"])
            elif isinstance(payload, dict) and "grid" in payload:
                bs = _grid_to_bitstring(payload["grid"], len(meta.triangle_order))
            else:
                raise ValueError(f"{path}: glyph 0x{cp_int:04X} missing bitstring/grid")
            _validate_bitstring(bs, _bit_count(meta), f"{path.name} cp=0x{cp_int:04X}")
            source_hex = payload.get("source_hex") if isinstance(payload, dict) else None
            char_hint = payload.get("char") if isinstance(payload, dict) else None
            glyphs[cp_int] = GlyphRecord(
                codepoint=cp_int,
                bitstring=bs,
                source_hex=str(source_hex) if source_hex else None,
                char_hint=str(char_hint) if char_hint is not None else None,
            )
        return meta, dict(sorted(glyphs.items()))

    raise ValueError(f"{path}: expected GLYPH_BITS or GLYPHS in module")


def load_from_json(path: Path) -> Tuple[GlyphMeta, Dict[int, GlyphRecord]]:
    data = json.loads(path.read_text(encoding="utf-8"))

    meta_raw = data.get("meta", {})
    rows = int(meta_raw.get("grid_rows", DEFAULT_GRID_ROWS))
    cols = int(meta_raw.get("grid_cols", DEFAULT_GRID_COLS))
    cell_size = int(meta_raw.get("cell_size", DEFAULT_CELL_SIZE))
    triangle_order = tuple(meta_raw.get("triangle_order", list(DEFAULT_TRI_ORDER)))
    meta = GlyphMeta(rows=rows, cols=cols, cell_size=cell_size, triangle_order=triangle_order)

    glyphs_raw = data.get("glyphs")
    if not isinstance(glyphs_raw, dict):
        raise ValueError(f"{path}: JSON missing object 'glyphs'")

    glyphs: Dict[int, GlyphRecord] = {}
    expected_len = _bit_count(meta)

    for key, payload in glyphs_raw.items():
        if not isinstance(payload, dict):
            continue

        cp = payload.get("codepoint")
        if cp is None:
            m = FILENAME_HEX_RE.match(str(key))
            if not m:
                raise ValueError(f"{path}: glyph entry {key!r} missing codepoint and invalid key")
            cp = int(m.group(1), 16)
        cp_int = int(cp)

        if "bitstring" in payload:
            bs = str(payload["bitstring"])
        elif "bits" in payload:
            bits = payload["bits"]
            if not isinstance(bits, list):
                raise ValueError(f"{path}: glyph 0x{cp_int:04X} has non-list 'bits'")
            bs = "".join("1" if int(v) else "0" for v in bits)
        elif "grid" in payload:
            bs = _grid_to_bitstring(payload["grid"], len(meta.triangle_order))
        else:
            raise ValueError(f"{path}: glyph 0x{cp_int:04X} missing bitstring/bits/grid")

        _validate_bitstring(bs, expected_len, f"{path.name} cp=0x{cp_int:04X}")

        glyphs[cp_int] = GlyphRecord(
            codepoint=cp_int,
            bitstring=bs,
            source_hex=str(payload.get("source_hex")) if payload.get("source_hex") else None,
            char_hint=str(payload.get("char")) if payload.get("char") is not None else None,
        )

    return meta, dict(sorted(glyphs.items()))


def load_glyph_source(source_path: Path) -> Tuple[GlyphMeta, Dict[int, GlyphRecord]]:
    if not source_path.exists():
        raise FileNotFoundError(f"Source file does not exist: {source_path}")

    suffix = source_path.suffix.lower()
    if suffix == ".json":
        return load_from_json(source_path)
    if suffix == ".py":
        return load_from_bits_py(source_path)

    raise ValueError(f"Unsupported source type: {source_path} (expected .py or .json)")


# ---------------------------------------------------------------------------
# SVG rendering
# ---------------------------------------------------------------------------

def _glyph_filename(rec: GlyphRecord, hex_width: int, preserve_source_hex: bool, alt: bool = False) -> str:
    if preserve_source_hex and rec.source_hex:
        base_hex = rec.source_hex
    else:
        base_hex = f"{rec.codepoint:0{hex_width}x}"
    suffix = "-alt" if alt else ""
    return f"character-u{base_hex}{suffix}.svg"


def render_svg(
    rec: GlyphRecord,
    meta: GlyphMeta,
    fg_fill: str = "#000",
    bg_fill: str = "#fff",
    *,
    alt: bool = False,
) -> str:
    rows, cols, cell_size = meta.rows, meta.cols, meta.cell_size
    tri_order = meta.triangle_order
    w = cols * cell_size
    h = rows * cell_size

    ch = rec.char_hint if (rec.char_hint and len(rec.char_hint) == 1) else ""
    if not ch:
        try:
            candidate = chr(rec.codepoint)
            ch = candidate if _is_printable_char(candidate) else ""
        except ValueError:
            ch = ""

    cp_hex = f"U+{rec.codepoint:04X}"
    title_txt = f"Glyph {cp_hex}"
    if ch:
      title_txt += f" ({ch})"
    if alt:
      title_txt += " alt"

    uname = _unicode_name(rec.codepoint)
    desc_parts = [f"Generated glyph for {cp_hex}"]
    if uname:
        desc_parts.append(uname)
    desc_parts.append(f"grid {rows}x{cols}")
    desc_parts.append(f"{len(rec.bitstring)} bits")
    desc_parts.append("inverse occupancy (0=active,1=inactive)" if alt else "normal occupancy (1=active,0=inactive)")
    desc_txt = " | ".join(desc_parts)

    variant = "alt" if alt else "normal"

    out: List[str] = []
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        f'data-codepoint="{xml_escape(cp_hex)}" '
        f'data-char="{xml_escape(_safe_char_for_attr(ch))}" '
        f'data-variant="{variant}">'
    )
    out.append(f"    <title>{xml_escape(title_txt)}</title>")
    out.append(f"    <desc>{xml_escape(desc_txt)}</desc>")
    out.append("")
    out.append("    <!-- Background -->")
    out.append(f'    <rect x="0" y="0" width="{w}" height="{h}" fill="{xml_escape(bg_fill)}"/>')
    out.append("")
    out.append("    <!-- Grid cells, each cell has 4 editable triangles (class-based selectors) -->")
    out.append(f'    <g fill="{xml_escape(fg_fill)}" shape-rendering="crispEdges">')

    idx = 0
    for r in range(rows):
        for c in range(cols):
            for tri in tri_order:
                bit = rec.bitstring[idx]
                idx += 1
                is_active = (bit == "0") if alt else (bit == "1")
                if not is_active:
                    continue
                pts = _points_for_triangle(r, c, tri, cell_size)
                out.append(f'        <polygon class="r{r}c{c}-{tri}" points="{pts}"/>')
        if r != rows - 1:
            out.append("")

    out.append("    </g>")
    out.append("</svg>")
    out.append("")
    return "\n".join(out)


def write_svgs(
    glyphs: Dict[int, GlyphRecord],
    meta: GlyphMeta,
    out_dir: Path,
    fg_fill: str,
    bg_fill: str,
    overwrite: bool,
    hex_width: int,
    preserve_source_hex: bool,
    only: set[int] | None = None,
    write_alt: bool = True,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for cp, rec in glyphs.items():
        if only is not None and cp not in only:
            continue

        # Normal file
        normal_name = _glyph_filename(rec, hex_width=hex_width, preserve_source_hex=preserve_source_hex, alt=False)
        normal_path = out_dir / normal_name

        if normal_path.exists() and not overwrite:
            print(f"[skip] {normal_path} exists (use --overwrite to replace)")
        else:
            svg = render_svg(rec, meta=meta, fg_fill=fg_fill, bg_fill=bg_fill, alt=False)
            normal_path.write_text(svg, encoding="utf-8")
            written += 1

        # Alt file (inverse occupancy)
        if write_alt:
            alt_name = _glyph_filename(rec, hex_width=hex_width, preserve_source_hex=preserve_source_hex, alt=True)
            alt_path = out_dir / alt_name

            if alt_path.exists() and not overwrite:
                print(f"[skip] {alt_path} exists (use --overwrite to replace)")
            else:
                alt_svg = render_svg(rec, meta=meta, fg_fill=fg_fill, bg_fill=bg_fill, alt=True)
                alt_path.write_text(alt_svg, encoding="utf-8")
                written += 1

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate glyph SVGs from binary glyph data.")

    p.add_argument(
        "--source",
        type=Path,
        default=_choose_default_source(),
        help="Input data file (.py or .json). Defaults to first existing of: "
             "data/glyphs_bits.py, data/glyphs.json, data/glyphs_data.py",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_SRC_DIR,
        help=f"Output directory for SVGs (default: {DEFAULT_SRC_DIR})",
    )

    p.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional glyph selectors (e.g. A 2 U+03A9 0x41 65). If omitted, export all.",
    )

    p.add_argument("--overwrite", action="store_true", help="Overwrite existing SVG files")
    p.add_argument("--hex-width", type=int, default=4, help="Hex padding width in filenames (default: 4)")
    p.add_argument(
        "--no-preserve-source-hex",
        action="store_true",
        help="Do not preserve original source_hex from the data; always format filenames with --hex-width",
    )

    p.add_argument("--fg-fill", default="#000", help="Foreground polygon fill (default: #000)")
    p.add_argument("--bg-fill", default="#fff", help="Background fill (default: #fff)")
    p.add_argument("--no-alt", action="store_true", help="Do not generate -alt inverse SVGs")

    # Optional overrides
    p.add_argument("--rows", type=int, default=None, help="Override rows from source metadata")
    p.add_argument("--cols", type=int, default=None, help="Override cols from source metadata")
    p.add_argument("--cell-size", type=int, default=None, help="Override cell size from source metadata")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        meta, glyphs = load_glyph_source(args.source)
    except Exception as e:
        print(f"Error loading source: {e}", file=sys.stderr)
        return 1

    if args.rows is not None:
        meta.rows = int(args.rows)
    if args.cols is not None:
        meta.cols = int(args.cols)
    if args.cell_size is not None:
        meta.cell_size = int(args.cell_size)

    if meta.rows <= 0 or meta.cols <= 0 or meta.cell_size <= 0:
        print("rows, cols, and cell-size must be positive integers", file=sys.stderr)
        return 2

    expected_len = _bit_count(meta)
    try:
        for cp, rec in glyphs.items():
            _validate_bitstring(rec.bitstring, expected_len, f"cp=0x{cp:04X}")
    except Exception as e:
        print(f"Invalid glyph data: {e}", file=sys.stderr)
        return 1

    only_set: set[int] | None = None
    if args.only:
        only_set = set()
        for token in args.only:
            try:
                only_set.add(_parse_only_token(token))
            except ValueError as e:
                print(f"Invalid --only token {token!r}: {e}", file=sys.stderr)
                return 2

    glyph_total = len(glyphs) if only_set is None else sum(1 for cp in glyphs if cp in only_set)
    variant_count = 1 if args.no_alt else 2

    print(f"Loaded {len(glyphs)} glyph(s) from {args.source}")
    print(f"Geometry: {meta.rows}x{meta.cols} cells, cell={meta.cell_size}px, triangles={meta.triangle_order}")
    print(f"Bits per glyph: {expected_len}")
    print(f"About to export: {glyph_total} glyph(s) × {variant_count} variant(s)")

    try:
        written = write_svgs(
            glyphs=glyphs,
            meta=meta,
            out_dir=args.out_dir,
            fg_fill=args.fg_fill,
            bg_fill=args.bg_fill,
            overwrite=args.overwrite,
            hex_width=max(1, int(args.hex_width)),
            preserve_source_hex=not args.no_preserve_source_hex,
            only=only_set,
            write_alt=not args.no_alt,
        )
    except Exception as e:
        print(f"Error writing SVGs: {e}", file=sys.stderr)
        return 1

    print(f"Exported {written} SVG file(s) to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())