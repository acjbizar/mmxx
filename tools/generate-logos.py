#!/usr/bin/env python3
"""
tools/generate-logos.py

Generate square SVG "logos" by combining 4 glyph SVGs from src/.

Outputs:
  src/logo-{chars}.svg   (filename always lowercase)

Usage:
  # One logo:
  python tools/generate-logos.py --chars mmxx

  # Batch (default):
  python tools/generate-logos.py
  # -> reads data/logos.txt, one logo per non-empty line

Options:
  --chars    Four characters to use (e.g. "abcd" or "a,b,c,d"). If omitted, read data/logos.txt.
  --color    CSS color for glyphs. If omitted, glyph fill is left UNDEFINED (UA default black).
  --bgcolor  CSS color for ONE background square (full logo). If omitted, no background (transparent).

Notes:
- Output SVG is 480×480 (width/height attrs) with viewBox "0 0 480 480".
- No padding/gap between the 2×2 cells.
- Removes per-glyph full-canvas <rect> backgrounds inside character SVGs (prevents black squares).
- Safely ignores comment/PI nodes in SVGs (fixes the cython_function_or_method error).
"""

from __future__ import annotations

import argparse
import sys
import re
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple

from lxml import etree  # py -m pip install lxml


SVG_NS = "http://www.w3.org/2000/svg"
NSMAP = {None: SVG_NS}

NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


# ---------- helpers: chars parsing ----------

def normalize_chars_spec(s: str) -> str:
    """Remove comments (#...), whitespace and commas."""
    s = s.strip()
    if "#" in s:
        s = s.split("#", 1)[0]
    s = "".join(ch for ch in s if (not ch.isspace()) and ch != ",")
    return s


def require_4_chars(spec: str, context: str) -> str:
    spec = normalize_chars_spec(spec)
    if len(spec) < 4:
        raise ValueError(f"{context}: expected 4 characters, got {len(spec)}: {spec!r}")
    if len(spec) > 4:
        print(f"WARNING: {context}: got {len(spec)} characters; using first 4: {spec[:4]!r}", file=sys.stderr)
        spec = spec[:4]
    return spec


def filename_key(chars4: str) -> str:
    """
    Produce a lowercase filesystem-friendly key.
    If a char is not [a-z0-9], encode as uXXXX.
    """
    out = []
    for ch in chars4:
        if "A" <= ch <= "Z":
            out.append(ch.lower())
        elif "a" <= ch <= "z" or "0" <= ch <= "9":
            out.append(ch)
        else:
            out.append(f"u{ord(ch):04x}")
    return "".join(out).lower()


# ---------- helpers: SVG loading / cleanup ----------

def is_element_node(node) -> bool:
    # In lxml, comments/processing-instructions have node.tag as a Cython callable (not a str).
    return isinstance(getattr(node, "tag", None), str)


def _local_name(tag) -> str:
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    v = value.strip().lower().replace("px", "").strip()
    if not v or "%" in v:
        return None
    try:
        return float(v)
    except Exception:
        return None


def parse_viewbox(vb: Optional[str]) -> Tuple[float, float, float, float]:
    vb = vb or "0 0 240 240"
    nums = [float(x) for x in NUM_RE.findall(vb)]
    if len(nums) == 4:
        return nums[0], nums[1], nums[2], nums[3]
    return 0.0, 0.0, 240.0, 240.0


def strip_fill_from_style(style: str) -> str:
    """Remove 'fill: ...' from a style attribute while keeping other properties."""
    if not style:
        return style
    style2 = re.sub(r"(?:(?:^|;)\s*fill\s*:\s*[^;]+)", "", style, flags=re.IGNORECASE)
    style2 = re.sub(r";{2,}", ";", style2).strip().strip(";").strip()
    return style2


def strip_fill_attrs(el: etree._Element) -> None:
    """
    Remove explicit 'fill' attributes (and fill in style=...) so that:
      - if --color is omitted, fill is undefined (UA default black)
      - if --color is set on wrapper group, children inherit it
    """
    for node in el.iter():
        if not is_element_node(node):
            continue
        if "fill" in node.attrib:
            del node.attrib["fill"]
        if "style" in node.attrib:
            new_style = strip_fill_from_style(node.attrib.get("style", ""))
            if new_style:
                node.attrib["style"] = new_style
            else:
                del node.attrib["style"]


def remove_full_background_rects(el: etree._Element, vb: Tuple[float, float, float, float]) -> int:
    """
    Remove <rect> elements that appear to be full-background rectangles for the glyph SVG.
    This prevents them turning black when fills are stripped.

    Matches:
      - x/y defaulting to 0, width/height numeric matching viewBox w/h (or 100%/100%)
      - also allows x/y matching viewBox minx/miny
    """
    minx, miny, w, h = vb
    eps = 1e-6
    removed = 0

    rects: List[etree._Element] = []
    for n in el.iter():
        if not is_element_node(n):
            continue
        if _local_name(n.tag) == "rect":
            rects.append(n)

    for r in rects:
        rx = _to_float(r.get("x")) or 0.0
        ry = _to_float(r.get("y")) or 0.0
        rw_raw = (r.get("width") or "").strip()
        rh_raw = (r.get("height") or "").strip()

        is_percent = (rw_raw == "100%" and rh_raw == "100%")
        rw = _to_float(rw_raw)
        rh = _to_float(rh_raw)

        matches_numeric = (
            rw is not None and rh is not None and
            (
                (abs(rx - 0.0) < eps and abs(ry - 0.0) < eps and abs(rw - w) < eps and abs(rh - h) < eps) or
                (abs(rx - minx) < eps and abs(ry - miny) < eps and abs(rw - w) < eps and abs(rh - h) < eps)
            )
        )

        matches_percent = (abs(rx - 0.0) < eps and abs(ry - 0.0) < eps and is_percent)

        if matches_numeric or matches_percent:
            parent = r.getparent()
            if parent is not None:
                parent.remove(r)
                removed += 1

    return removed


def find_glyph_svg(src_dir: Path, ch: str) -> Optional[Path]:
    direct = src_dir / f"character-{ch}.svg"
    if direct.is_file():
        return direct

    if len(ch) == 1 and ch.isalpha():
        p = src_dir / f"character-{ch.lower()}.svg"
        if p.is_file():
            return p
        p = src_dir / f"character-{ch.upper()}.svg"
        if p.is_file():
            return p

    cp = src_dir / f"character-u{ord(ch):04x}.svg"
    if cp.is_file():
        return cp

    return None


def load_glyph(src_dir: Path, ch: str) -> Tuple[Tuple[float, float, float, float], List[etree._Element]]:
    """
    Returns:
      viewBox (minx, miny, w, h)
      element children (deep-copied) of glyph <svg>, with background rects removed
      and fills stripped for proper inheritance/undefined behavior.
    """
    p = find_glyph_svg(src_dir, ch)
    if p is None:
        raise FileNotFoundError(f"No SVG found for {ch!r} in {src_dir}")

    root = etree.fromstring(p.read_bytes())
    vb = parse_viewbox(root.get("viewBox") or root.get("viewbox"))

    # Only element children (skip comments/PI nodes)
    children = [deepcopy(child) for child in list(root) if is_element_node(child)]

    # Work in a container so parent pointers exist for removals
    tmp_container = etree.Element("container")
    for child in children:
        tmp_container.append(child)

    # Remove per-glyph full background rects
    remove_full_background_rects(tmp_container, vb)

    # Strip fills for inheritance / undefined fill
    for child in list(tmp_container):
        strip_fill_attrs(child)

    return vb, [deepcopy(child) for child in list(tmp_container)]


# ---------- logo building ----------

def build_logo_svg(chars4: str, src_dir: Path, color: Optional[str], bgcolor: Optional[str]) -> etree._Element:
    CANVAS = 480.0
    CELL = CANVAS / 2.0
    PAD = 0.0  # requested: no spacing/padding

    svg = etree.Element(f"{{{SVG_NS}}}svg", nsmap=NSMAP)
    svg.set("width", "480")
    svg.set("height", "480")
    svg.set("viewBox", "0 0 480 480")

    # Optional single background for the whole logo
    if bgcolor:
        bg = etree.SubElement(svg, f"{{{SVG_NS}}}rect")
        bg.set("x", "0")
        bg.set("y", "0")
        bg.set("width", "480")
        bg.set("height", "480")
        bg.set("fill", bgcolor)

    # Order: TL, TR, BL, BR
    for i, ch in enumerate(chars4):
        vb, nodes = load_glyph(src_dir, ch)
        minx, miny, w, h = vb

        r = 0 if i < 2 else 1
        c = 0 if (i % 2) == 0 else 1

        cell_x = c * CELL
        cell_y = r * CELL

        avail_w = CELL - 2.0 * PAD
        avail_h = CELL - 2.0 * PAD

        if w <= 0 or h <= 0 or avail_w <= 0 or avail_h <= 0:
            scale = 1.0
            scaled_w = w
            scaled_h = h
        else:
            scale = min(avail_w / w, avail_h / h)
            scaled_w = w * scale
            scaled_h = h * scale

        tx = cell_x + (CELL - scaled_w) / 2.0
        ty = cell_y + (CELL - scaled_h) / 2.0

        g_outer = etree.SubElement(svg, f"{{{SVG_NS}}}g")
        g_outer.set("transform", f"translate({tx:.6f} {ty:.6f})")

        if color:
            g_outer.set("fill", color)

        g_scale = etree.SubElement(g_outer, f"{{{SVG_NS}}}g")
        g_scale.set("transform", f"scale({scale:.12f})")

        g_shift = etree.SubElement(g_scale, f"{{{SVG_NS}}}g")
        g_shift.set("transform", f"translate({-minx:.6f} {-miny:.6f})")

        for node in nodes:
            g_shift.append(node)

    return svg


def write_logo(chars4: str, src_dir: Path, color: Optional[str], bgcolor: Optional[str]) -> Path:
    key = filename_key(chars4)
    out_path = src_dir / f"logo-{key}.svg"

    svg = build_logo_svg(chars4, src_dir, color=color, bgcolor=bgcolor)
    xml_bytes = etree.tostring(svg, encoding="utf-8", xml_declaration=False, pretty_print=False)
    out_path.write_bytes(xml_bytes + b"\n")
    return out_path


def read_logos_file(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"logos file not found: {path}")
    specs: List[str] = []
    for idx, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        s = normalize_chars_spec(raw)
        if not s:
            continue
        try:
            specs.append(require_4_chars(s, context=f"{path.name}:{idx}"))
        except ValueError as e:
            print(f"WARNING: {e}", file=sys.stderr)
    return specs


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate 2×2 logo SVGs (480×480) from 4 glyph SVGs in src/.")
    ap.add_argument("--chars", type=str, default="", help="Exactly 4 characters (e.g. mmxx). If omitted, read data/logos.txt.")
    ap.add_argument("--color", type=str, default="", help="CSS color for glyphs. If empty, glyph fill stays undefined (UA default black).")
    ap.add_argument("--bgcolor", type=str, default="", help="CSS color for one background rect. If empty, no background (transparent).")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    src_dir = root / "src"
    logos_txt = root / "data" / "logos.txt"

    if not src_dir.is_dir():
        raise SystemExit(f"Source folder not found: {src_dir}")

    color = args.color.strip() or None
    bgcolor = args.bgcolor.strip() or None

    if args.chars.strip():
        try:
            chars4 = require_4_chars(args.chars, context="--chars")
        except ValueError as e:
            raise SystemExit(str(e))

        try:
            out = write_logo(chars4, src_dir, color=color, bgcolor=bgcolor)
            print(f"Wrote: {out}")
        except Exception as e:
            raise SystemExit(f"ERROR generating logo for {chars4!r}: {e}")
        return

    # Batch mode
    specs = read_logos_file(logos_txt)
    if not specs:
        raise SystemExit(f"No valid 4-char lines found in {logos_txt}")

    errors = 0
    for chars4 in specs:
        try:
            out = write_logo(chars4, src_dir, color=color, bgcolor=bgcolor)
            print(f"Wrote: {out}")
        except Exception as e:
            errors += 1
            print(f"ERROR generating logo for {chars4!r}: {e}", file=sys.stderr)

    if errors:
        raise SystemExit(f"Done with {errors} error(s).")
    print("Done.")


if __name__ == "__main__":
    main()
