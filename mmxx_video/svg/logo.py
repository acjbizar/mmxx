from __future__ import annotations
import copy
from pathlib import Path
from typing import List, Tuple, Optional
from lxml import etree

from ..constants import SVG_NS, XLINK_NS
from .parse import parse_viewbox
from .style import strip_white_full_canvas_rects
from .color import parse_css_color_to_rgb, rgb_to_hex
from .ids import prefix_svg_ids

def _flatten_color_tokens(tokens: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    for t in (tokens or []):
        if t is None:
            continue
        for part in str(t).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out

def _is_skip_bg_token(tok: str) -> bool:
    t = (tok or "").strip().lower()
    return t in {"", "-", "none", "null", "skip", "transparent"}


def build_logo_svg_from_chars_grid(
    char_svgs: List[Path],
    grid_n: int,
    gap_flag: int,
    bgcolors: Optional[List[str]] = None,
) -> etree._Element:
    if grid_n not in (2, 3, 4):
        raise ValueError("grid_n must be 2, 3, or 4.")
    if len(char_svgs) != grid_n * grid_n:
        raise ValueError(f"Expected {grid_n*grid_n} character SVG paths for {grid_n}x{grid_n} grid.")

    parser = etree.XMLParser(remove_blank_text=False, recover=True, remove_comments=False)

    glyph_docs: List[etree._Element] = []
    vbs: List[Tuple[float, float, float, float]] = []

    for p in char_svgs:
        if not p.is_file():
            raise SystemExit(f"Character SVG not found: {p}")
        root = etree.fromstring(p.read_bytes(), parser=parser)
        vb = parse_viewbox(root)
        strip_white_full_canvas_rects(root, vb)
        glyph_docs.append(root)
        vbs.append(vb)

    max_w = max(vb[2] for vb in vbs)
    max_h = max(vb[3] for vb in vbs)

    if gap_flag == 1:
        gap_x = max_w / 8.0
        gap_y = max_h / 8.0
        pad_x = gap_x
        pad_y = gap_y
    else:
        gap_x = gap_y = pad_x = pad_y = 0.0

    total_w = grid_n * max_w + (grid_n - 1) * gap_x + 2.0 * pad_x
    total_h = grid_n * max_h + (grid_n - 1) * gap_y + 2.0 * pad_y

    svg = etree.Element(f"{{{SVG_NS}}}svg", nsmap={None: SVG_NS, "xlink": XLINK_NS})
    svg.set("viewBox", f"0 0 {total_w} {total_h}")

    # Optional per-glyph background colors (for --chars mode).
    bg_tokens = _flatten_color_tokens(bgcolors)
    bg_fills: List[Optional[str]] = [None] * len(char_svgs)
    if bg_tokens:
        if len(bg_tokens) == 1:
            bg_tokens = bg_tokens * len(char_svgs)
        if len(bg_tokens) != len(char_svgs):
            raise ValueError(
                f"--bgcolors expects 1 or {len(char_svgs)} colors for this logo; got {len(bg_tokens)}"
            )
        for i, tok in enumerate(bg_tokens):
            if _is_skip_bg_token(tok):
                bg_fills[i] = None
                continue
            try:
                bg_fills[i] = rgb_to_hex(parse_css_color_to_rgb(tok))
            except Exception as e:
                raise ValueError(f"Invalid --bgcolors color: {tok!r}") from e

    for idx, (groot, (minx, miny, vbw, vbh)) in enumerate(zip(glyph_docs, vbs)):
        row = idx // grid_n
        col = idx % grid_n

        cell_x0 = pad_x + col * (max_w + gap_x)
        cell_y0 = pad_y + row * (max_h + gap_y)

        # Background rect for this glyph cell (optional)
        if 0 <= idx < len(bg_fills) and bg_fills[idx] is not None:
            rect = etree.Element(f"{{{SVG_NS}}}rect")
            rect.set("x", str(cell_x0))
            rect.set("y", str(cell_y0))
            rect.set("width", str(max_w))
            rect.set("height", str(max_h))
            rect.set("fill", str(bg_fills[idx]))
            rect.set("data-bg-glyph-index", str(idx + 1))
            svg.append(rect)

        tx = cell_x0 + (max_w - vbw) * 0.5 - minx
        ty = cell_y0 + (max_h - vbh) * 0.5 - miny

        group = etree.Element(f"{{{SVG_NS}}}g")
        for child in list(groot):
            group.append(copy.deepcopy(child))

        prefix_svg_ids(group, f"g{idx}_")

        group.set("data-minx", str(minx))
        group.set("data-miny", str(miny))
        group.set("data-vbw", str(vbw))
        group.set("data-vbh", str(vbh))
        group.set("data-glyph-index", str(idx + 1))
        group.set("transform", f"translate({tx},{ty})")
        svg.append(group)

    return svg
