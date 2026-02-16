from __future__ import annotations
import copy
from pathlib import Path
from typing import List, Tuple
from lxml import etree

from ..constants import SVG_NS, XLINK_NS
from .parse import parse_viewbox
from .style import strip_white_full_canvas_rects
from .ids import prefix_svg_ids

def build_logo_svg_from_chars_grid(char_svgs: List[Path], grid_n: int, gap_flag: int) -> etree._Element:
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

    for idx, (groot, (minx, miny, vbw, vbh)) in enumerate(zip(glyph_docs, vbs)):
        row = idx // grid_n
        col = idx % grid_n

        cell_x0 = pad_x + col * (max_w + gap_x)
        cell_y0 = pad_y + row * (max_h + gap_y)

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
        group.set("transform", f"translate({tx},{ty})")
        svg.append(group)

    return svg
