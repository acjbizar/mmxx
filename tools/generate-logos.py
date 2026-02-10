#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/generate-logos.py

Generate square 480×480 logo SVGs by combining four glyph SVGs from src/.

- Input glyph SVGs: src/character-{ch}.svg
- Uses ONLY the top-level <g> from the glyph SVG (or synthesizes one if missing)
- Output: src/logo-{chars}.svg (filename lowercased)
- Output contains exactly 4 <g> elements (one per character), plus optional background <rect>
- Each glyph group:
    - class="u{codepoint}"  (lowercase hex, at least 4 digits)
    - transform=...         (positions the glyph into its quadrant)
    - optional fill if --color provided
"""

from __future__ import annotations

import argparse
import re
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple

from lxml import etree  # py -m pip install lxml


SVG_NS = "http://www.w3.org/2000/svg"
NSMAP = {None: SVG_NS}

NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


OUT_SIZE = 480.0
GRID = 2
CELL = OUT_SIZE / GRID  # 240
PAD = 0.0               # no spacing/padding between the 4 characters


def _parse_viewbox(root: etree._Element) -> Tuple[float, float, float, float]:
    vb = root.get("viewBox") or root.get("viewbox") or "0 0 240 240"
    nums = [float(x) for x in NUM_RE.findall(vb)]
    if len(nums) == 4:
        return nums[0], nums[1], nums[2], nums[3]
    return 0.0, 0.0, 240.0, 240.0


def _read_logo_keys(root: Path, chars_arg: Optional[str]) -> List[str]:
    if chars_arg:
        return [chars_arg]

    path = root / "data" / "logos.txt"
    if not path.is_file():
        raise SystemExit(f"Missing {path}. Provide --chars or create data/logos.txt")

    keys: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        keys.append(raw)
    return keys


def _normalize_4chars(s: str) -> str:
    cleaned = "".join(ch for ch in s.strip() if not ch.isspace())
    if len(cleaned) != 4:
        raise ValueError(f"Expected exactly 4 characters, got {len(cleaned)} from: {s!r}")
    return cleaned.lower()


def _find_glyph_file(src_dir: Path, ch: str) -> Path:
    candidates = [
        src_dir / f"character-{ch}.svg",
        src_dir / f"character-{ch.lower()}.svg",
        src_dir / f"character-{ch.upper()}.svg",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"No glyph SVG found for {ch!r} (tried: {', '.join(str(p.name) for p in candidates)})"
    )


def load_top_level_group(svg_path: Path) -> Tuple[Tuple[float, float, float, float], etree._Element]:
    """
    Returns (viewBox, group_element_copy)

    Uses ONLY the top-level <g> child. If none exists, synthesizes one by
    wrapping all top-level children (except defs/metadata/title/desc) into a <g>.
    """
    parser = etree.XMLParser(remove_blank_text=False, recover=True, remove_comments=False)
    root = etree.fromstring(svg_path.read_bytes(), parser=parser)

    vb = _parse_viewbox(root)

    groups = root.xpath("./*[local-name()='g']")
    if groups:
        g = groups[0]
    else:
        g = etree.Element(f"{{{SVG_NS}}}g")
        to_move = []
        for child in list(root):
            ln = etree.QName(child).localname
            if ln in {"defs", "metadata", "title", "desc"}:
                continue
            to_move.append(child)
        for child in to_move:
            root.remove(child)
            g.append(child)

    return vb, deepcopy(g)


def _fmt(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.6f}".rstrip("0").rstrip(".")


def build_logo_svg(src_dir: Path, chars4: str, color: Optional[str], bgcolor: Optional[str]) -> etree._Element:
    svg = etree.Element(
        f"{{{SVG_NS}}}svg",
        nsmap=NSMAP,
        width=str(int(OUT_SIZE)),
        height=str(int(OUT_SIZE)),
        viewBox=f"0 0 {int(OUT_SIZE)} {int(OUT_SIZE)}",
    )

    if bgcolor:
        bg = etree.SubElement(svg, f"{{{SVG_NS}}}rect")
        bg.set("x", "0")
        bg.set("y", "0")
        bg.set("width", str(int(OUT_SIZE)))
        bg.set("height", str(int(OUT_SIZE)))
        bg.set("fill", bgcolor)

    # Quadrants: TL, TR, BL, BR
    for i, ch in enumerate(chars4):
        glyph_path = _find_glyph_file(src_dir, ch)
        (minx, miny, w, h), g = load_top_level_group(glyph_path)

        r = 0 if i < 2 else 1
        c = 0 if (i % 2) == 0 else 1

        avail_w = CELL - 2.0 * PAD
        avail_h = CELL - 2.0 * PAD

        scale = 1.0
        if w > 0 and h > 0 and avail_w > 0 and avail_h > 0:
            scale = min(avail_w / w, avail_h / h)

        scaled_w = w * scale
        scaled_h = h * scale

        x0 = c * CELL + PAD + (avail_w - scaled_w) / 2.0
        y0 = r * CELL + PAD + (avail_h - scaled_h) / 2.0

        tx = x0
        ty = y0

        # Required class on THIS group (and ONLY 4 groups in output)
        g.set("class", f"u{ord(ch):04x}")

        if color:
            g.set("fill", color)

        # Build ONE transform string on THIS group (no wrappers).
        ops: List[str] = []

        # Position in quadrant
        if abs(tx) > 1e-9 or abs(ty) > 1e-9:
            ops.append(f"translate({_fmt(tx)} {_fmt(ty)})")

        # Scale if needed
        if abs(scale - 1.0) > 1e-12:
            ops.append(f"scale({scale:.12f}".rstrip("0").rstrip(".") + ")")

        # Shift viewBox min corner to origin
        if abs(minx) > 1e-9 or abs(miny) > 1e-9:
            ops.append(f"translate({_fmt(-minx)} {_fmt(-miny)})")

        # Preserve any existing transform from the source group (applied inside)
        old_t = (g.get("transform") or "").strip()
        if old_t:
            ops.append(old_t)

        if ops:
            g.set("transform", " ".join(ops))
        else:
            g.attrib.pop("transform", None)

        svg.append(g)

    return svg


def write_svg(path: Path, root_el: etree._Element) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xml = etree.tostring(root_el, encoding="unicode", pretty_print=False)
    xml = re.sub(r">\s+<", "><", xml).strip() + "\n"

    # Python 3.9 Path.write_text() has no newline= argument -> use open()
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(xml)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate 480x480 logo SVG(s) from four src/character-*.svg glyphs.")
    ap.add_argument("--chars", type=str, default=None, help="Exactly 4 characters (e.g. 'mmxx'). If omitted, reads data/logos.txt")
    ap.add_argument("--color", type=str, default="", help="CSS color for glyph fill. If empty, leave fill undefined.")
    ap.add_argument("--bgcolor", type=str, default="", help="CSS color for background square. If empty, no background rect.")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    src_dir = root / "src"

    keys = _read_logo_keys(root, args.chars)
    color = args.color.strip() or None
    bgcolor = args.bgcolor.strip() or None

    errors = 0
    for raw_key in keys:
        try:
            key = _normalize_4chars(raw_key)
            svg = build_logo_svg(src_dir, key, color=color, bgcolor=bgcolor)
            out_path = src_dir / f"logo-{key}.svg"
            write_svg(out_path, svg)
            print(f"Wrote {out_path}")
        except Exception as e:
            errors += 1
            print(f"ERROR generating logo for {raw_key!r}: {e}")

    if errors:
        raise SystemExit(f"Done with {errors} error(s).")
    print("Done.")


if __name__ == "__main__":
    main()
