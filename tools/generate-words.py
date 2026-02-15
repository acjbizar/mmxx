#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/generate-words.py

Generate static PNG "logo" images from 4-character words, arranged as:

- Each word becomes a 2x2 tile of 4 glyphs:
    word[0] word[1]
    word[2] word[3]
  (No gap between letters inside the word.)

- Words are placed in a square grid:
    1 word  -> 1x1
    4 words -> 2x2
    9 words -> 3x3
    16      -> 4x4
    25      -> 5x5

Gap behavior (matches your video generator semantics, but applied BETWEEN WORD TILES):
- --gap 0: no extra spacing/padding around/between word tiles
- --gap 1: spacing AND outer padding = 1/8th of a *character cell* size

Inputs (codepoint-based filenames):
- Glyph SVGs are expected at:
    src/character-u{codepoint}.svg
  If --inverse is set, we try:
    src/inverse-u{codepoint}.svg
  falling back to character-u{codepoint}.svg if inverse file not found.

Where {codepoint} is lowercase hex, zero-padded:
- min 4 digits (e.g. u0061)
- 6 digits for > 0xFFFF (e.g. u01f600)

Words source:
- If positional words are provided, those are used.
- If no positional words are provided, it loads from:
    data/words.txt
  (relative to project root).
  The file may contain words separated by whitespace and/or commas.
  Blank lines and lines starting with '#' are ignored.

Outputs:
- dist/images/logos-{words}.png
- dist/images/instagram/logos-{words}.png

Dependencies:
- Pillow
- CairoSVG
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from PIL import Image, ImageColor
except ImportError as e:
    raise SystemExit("Missing dependency: Pillow. Install with: pip install pillow") from e

try:
    import cairosvg  # type: ignore
except ImportError as e:
    raise SystemExit("Missing dependency: CairoSVG. Install with: pip install cairosvg") from e

import xml.etree.ElementTree as ET

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"

ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _parse_words(tokens: List[str]) -> List[str]:
    # allow comma-separated in a single argument, plus normal list
    words: List[str] = []
    for token in tokens:
        parts = [p for p in re.split(r"[,\s]+", token.strip()) if p]
        words.extend(parts)
    return words


def _load_words_file(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Words file not found: {path.as_posix()}")

    raw: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        raw.append(s)

    # join lines then split by commas/whitespace using the same parser
    return _parse_words([" ".join(raw)])


def _is_square_count(n: int) -> bool:
    r = int(math.isqrt(n))
    return r * r == n and r in (1, 2, 3, 4, 5)


def _codepoint_tag(ch: str) -> str:
    cp = ord(ch)
    if cp <= 0xFFFF:
        return f"u{cp:04x}"
    return f"u{cp:06x}"


def _slugify_word(word: str) -> str:
    out = []
    for ch in word:
        if ch.isalnum():
            out.append(ch.lower())
        else:
            out.append(_codepoint_tag(ch))
    return "".join(out)


def _words_slug(words: List[str]) -> str:
    return "-".join(_slugify_word(w) for w in words)


def _read_svg_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _get_viewbox_wh(root: ET.Element) -> Tuple[float, float]:
    vb = root.get("viewBox")
    if vb:
        parts = [p for p in re.split(r"[,\s]+", vb.strip()) if p]
        if len(parts) == 4:
            return float(parts[2]), float(parts[3])

    w_attr = root.get("width", "0")
    h_attr = root.get("height", "0")

    def num(s: str) -> float:
        m = re.match(r"^\s*([0-9]*\.?[0-9]+)", s)
        return float(m.group(1)) if m else 0.0

    return num(w_attr), num(h_attr)


def _remove_full_white_bg_rect(root: ET.Element) -> None:
    vb_w, vb_h = _get_viewbox_wh(root)
    if vb_w <= 0 or vb_h <= 0:
        return

    def is_white(fill: str) -> bool:
        f = fill.strip().lower()
        return f in ("#fff", "#ffffff", "white", "rgb(255,255,255)", "rgb(255, 255, 255)")

    to_remove: List[Tuple[ET.Element, ET.Element]] = []
    for parent in root.iter():
        for child in list(parent):
            if _strip_ns(child.tag) != "rect":
                continue
            fill = child.get("fill") or ""
            if not fill or not is_white(fill):
                continue

            x = float(child.get("x", "0") or "0")
            y = float(child.get("y", "0") or "0")

            def parse_len(v: str) -> float:
                m = re.match(r"^\s*([0-9]*\.?[0-9]+)", (v or "").strip())
                return float(m.group(1)) if m else 0.0

            w = parse_len(child.get("width", "0"))
            h = parse_len(child.get("height", "0"))

            if abs(x) < 1e-6 and abs(y) < 1e-6 and abs(w - vb_w) < 1e-3 and abs(h - vb_h) < 1e-3:
                to_remove.append((parent, child))

    for parent, child in to_remove:
        parent.remove(child)


def _override_fill_styles(elem: ET.Element, color: str) -> None:
    tag = _strip_ns(elem.tag)

    if tag in ("polygon", "path", "rect", "circle", "ellipse", "polyline"):
        fill = elem.get("fill")
        if fill is None or fill.strip().lower() not in ("none", "transparent"):
            elem.set("fill", color)

    style = elem.get("style")
    if style and "fill" in style:
        def repl(m: re.Match) -> str:
            val = (m.group(1) or "").strip().lower()
            if val in ("none", "transparent"):
                return m.group(0)
            return f"fill:{color}"

        elem.set("style", re.sub(r"fill\s*:\s*([^;]+)", repl, style, flags=re.IGNORECASE))


def _prepare_glyph_svg(svg_text: str, color: Optional[str]) -> bytes:
    """
    Modify a glyph SVG safely:
    - remove per-glyph full-canvas white background rects
    - apply fill override if color is set

    IMPORTANT: Avoid manually forcing xmlns on already-namespaced documents
    (that can create duplicate xmlns attributes when re-serialized).
    """
    root = ET.fromstring(svg_text.encode("utf-8"))

    _remove_full_white_bg_rect(root)

    if color:
        for el in root.iter():
            _override_fill_styles(el, color)

    # Only add xmlns if this SVG is NOT already namespaced.
    if not root.tag.startswith("{"):
        root.set("xmlns", SVG_NS)

    return ET.tostring(root, encoding="utf-8", method="xml")


def _render_svg_to_rgba(svg_bytes: bytes, px: int) -> Image.Image:
    png_bytes = cairosvg.svg2png(bytestring=svg_bytes, output_width=px, output_height=px)
    return Image.open(BytesIO(png_bytes)).convert("RGBA")


@dataclass
class Layout:
    g: int
    cell: int
    word: int
    pad: int
    spacing: int
    offset: int


def _layout_for(size_px: int, n_words: int, gap: int) -> Layout:
    g = int(math.isqrt(n_words))
    if gap not in (0, 1):
        raise ValueError("--gap must be 0 or 1")

    if gap == 0:
        cell = max(1, size_px // (2 * g))
        spacing = 0
        pad = 0
    else:
        # total = 2*g*cell + (g-1)*spacing + 2*pad, with spacing=pad=cell/8
        cell = max(1, int((size_px * 8) / (17 * g + 1)))
        spacing = max(1, int(round(cell / 8)))
        pad = spacing

    word = 2 * cell
    content = g * word + (g - 1) * spacing + 2 * pad
    offset = max(0, (size_px - content) // 2)

    return Layout(g=g, cell=cell, word=word, pad=pad, spacing=spacing, offset=offset)


def _find_glyph_path(src_dir: Path, ch: str, inverse: bool) -> Path:
    tag = _codepoint_tag(ch)

    if inverse:
        p_inv = src_dir / f"inverse-{tag}.svg"
        if p_inv.exists():
            return p_inv

    p_chr = src_dir / f"character-{tag}.svg"
    if p_chr.exists():
        return p_chr

    tried = [src_dir / f"inverse-{tag}.svg", p_chr]
    raise FileNotFoundError(
        "Glyph SVG not found. Tried:\n" + "\n".join(f"  - {p.as_posix()}" for p in tried)
    )


def _parse_bgcolor(bgcolor: Optional[str]) -> Tuple[int, int, int, int]:
    if not bgcolor:
        return (0, 0, 0, 0)
    b = bgcolor.strip().lower()
    if b in ("none", "transparent", "rgba(0,0,0,0)"):
        return (0, 0, 0, 0)
    return ImageColor.getcolor(bgcolor, "RGBA")


def _compose(
    *,
    size_px: int,
    words: List[str],
    src_dir: Path,
    color: Optional[str],
    bgcolor: Optional[str],
    inverse: bool,
    gap: int,
) -> Image.Image:
    n = len(words)
    layout = _layout_for(size_px, n, gap=gap)
    bg_rgba = _parse_bgcolor(bgcolor)
    canvas = Image.new("RGBA", (size_px, size_px), bg_rgba)

    glyph_cache: Dict[Tuple[str, int, Optional[str], bool], Image.Image] = {}

    def get_glyph_img(ch: str) -> Image.Image:
        tag = _codepoint_tag(ch)
        key = (tag, layout.cell, color, inverse)
        if key in glyph_cache:
            return glyph_cache[key]

        path = _find_glyph_path(src_dir, ch, inverse=inverse)
        svg_text = _read_svg_text(path)
        svg_bytes = _prepare_glyph_svg(svg_text, color=color)
        img = _render_svg_to_rgba(svg_bytes, px=layout.cell)
        glyph_cache[key] = img
        return img

    for i, word in enumerate(words):
        row = i // layout.g
        col = i % layout.g

        wx = layout.offset + layout.pad + col * (layout.word + layout.spacing)
        wy = layout.offset + layout.pad + row * (layout.word + layout.spacing)

        letters = list(word)
        for j, ch in enumerate(letters):
            lx = wx + (j % 2) * layout.cell
            ly = wy + (j // 2) * layout.cell
            canvas.alpha_composite(get_glyph_img(ch), (lx, ly))

    return canvas


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="generate-words.py",
        description="Generate static PNGs from 4-character words arranged into 1x1..5x5 grids.",
    )
    ap.add_argument(
        "words",
        nargs="*",
        help="Four-character words. Space-separated or comma-separated (e.g. love hate OR love,hate). "
             "If omitted, words are loaded from data/words.txt",
    )
    ap.add_argument("--gap", type=int, default=0, choices=(0, 1), help="0=no spacing/padding; 1=spacing+padding=cell/8")
    ap.add_argument("--color", default=None, help="Override glyph fill color (e.g. '#000', '#ff00aa', 'black').")
    ap.add_argument("--bgcolor", default=None, help="Background color (e.g. '#fff') or 'transparent' for alpha.")
    ap.add_argument(
        "--inverse",
        action="store_true",
        help="Use src/inverse-u{codepoint}.svg when available (falls back to character-u{codepoint}.svg).",
    )
    ap.add_argument("--size", type=int, default=2048, help="Output size (square) for dist/images (default 2048).")
    ap.add_argument(
        "--instagram-size",
        type=int,
        default=1080,
        help="Output size (square) for dist/images/instagram (default 1080).",
    )

    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    src_dir = root / "src"

    if args.words:
        words = _parse_words(args.words)
    else:
        words_file = root / "data" / "words.txt"
        words = _load_words_file(words_file)

    if not _is_square_count(len(words)):
        raise SystemExit("Number of words must be 1, 4, 9, 16, or 25 (for 1x1 .. 5x5).")

    for w in words:
        if len(w) != 4:
            raise SystemExit(f"Each word must be exactly 4 characters long. Got '{w}' (len={len(w)}).")

    dist_dir = root / "dist" / "images"
    insta_dir = dist_dir / "instagram"
    _ensure_dir(dist_dir)
    _ensure_dir(insta_dir)

    slug = _words_slug(words)
    out_main = dist_dir / f"logos-{slug}.png"
    out_insta = insta_dir / f"logos-{slug}.png"

    img_main = _compose(
        size_px=int(args.size),
        words=words,
        src_dir=src_dir,
        color=args.color,
        bgcolor=args.bgcolor,
        inverse=bool(args.inverse),
        gap=int(args.gap),
    )
    img_main.save(out_main, format="PNG")

    img_insta = _compose(
        size_px=int(args.instagram_size),
        words=words,
        src_dir=src_dir,
        color=args.color,
        bgcolor=args.bgcolor,
        inverse=bool(args.inverse),
        gap=int(args.gap),
    )
    img_insta.save(out_insta, format="PNG")

    print(f"Wrote:\n  - {out_main}\n  - {out_insta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
