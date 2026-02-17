from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from lxml import etree

from .constants import SVG_NS
from .svg.parse import parse_viewbox, local_name
from .svg.style import resolve_fill
from .svg.color import parse_css_color_to_rgb
from .svg.geom import global_centroid_norm, centroid_in_glyph_norm
from .anim.color_math import rgb255_to_hsv01
from .anim.pulses import make_pulses, Pulse

RGB = Tuple[int, int, int]

@dataclass
class Scene:
    doc: etree._Element
    polys: List[etree._Element]

    vb_tuple: Tuple[float, float, float, float]
    out_w: int
    out_h: int
    fps: int
    duration: float
    frames: int

    poly_nx: List[float]
    poly_ny: List[float]
    glyph_nx: List[float]
    glyph_ny: List[float]

    base_rgbs: List[RGB]
    override_hsv: Optional[Tuple[float, float, float]]
    bgcolor: Optional[str]

    pulses_per_poly: List[List[Pulse]]

    @property
    def n_polys(self) -> int:
        return len(self.polys)

def apply_bgcolor_override(doc: etree._Element, vb_tuple: Tuple[float, float, float, float], bgcolor: str) -> None:
    minx, miny, vb_w, vb_h = vb_tuple
    # remove a full-canvas <rect> if present near the top-level
    for el in list(doc):
        if not isinstance(el.tag, str):
            continue
        if local_name(el.tag) != "rect":
            continue
        if el.get("transform"):
            continue
        try:
            xf = float(el.get("x", "0") or "0")
            yf = float(el.get("y", "0") or "0")
            wf = float(el.get("width", "") or "-1")
            hf = float(el.get("height", "") or "-1")
            if abs(xf - minx) < 1e-6 and abs(yf - miny) < 1e-6 and abs(wf - vb_w) < 1e-6 and abs(hf - vb_h) < 1e-6:
                doc.remove(el)
                break
        except Exception:
            pass

    rect = etree.Element(f"{{{SVG_NS}}}rect")
    rect.set("x", str(minx))
    rect.set("y", str(miny))
    rect.set("width", str(vb_w))
    rect.set("height", str(vb_h))
    rect.set("fill", bgcolor)
    doc.insert(0, rect)

def build_scene_from_args(*, args, rng: random.Random, svg_doc: etree._Element, duration: float, fps: int) -> Scene:
    minx, miny, vb_w, vb_h = parse_viewbox(svg_doc)
    if vb_w <= 0 or vb_h <= 0:
        vb_w, vb_h = 240.0, 240.0
    vb_tuple = (minx, miny, vb_w, vb_h)

    bgcolor = (args.bgcolor.strip() or None)
    if bgcolor:
        apply_bgcolor_override(svg_doc, vb_tuple, bgcolor)

    polys = svg_doc.xpath('.//*[local-name()="polygon"]')
    if not polys:
        raise SystemExit("No <polygon> elements found in the input SVG(s).")

    poly_nx: List[float] = []
    poly_ny: List[float] = []
    glyph_nx: List[float] = []
    glyph_ny: List[float] = []
    for p in polys:
        nx, ny = global_centroid_norm(p, vb_tuple)
        poly_nx.append(nx)
        poly_ny.append(ny)
        gx, gy = centroid_in_glyph_norm(p, vb_tuple)
        glyph_nx.append(gx)
        glyph_ny.append(gy)

    override_color = args.color.strip() or None
    base_rgb_override: Optional[RGB] = None
    override_hsv: Optional[Tuple[float, float, float]] = None
    if override_color:
        base_rgb_override = parse_css_color_to_rgb(override_color)
        override_hsv = rgb255_to_hsv01(base_rgb_override)

    base_rgbs: List[RGB] = []
    for p in polys:
        if base_rgb_override is not None:
            base_rgbs.append(base_rgb_override)
            continue
        fill = resolve_fill(p) or "#000"
        try:
            base_rgbs.append(parse_css_color_to_rgb(fill))
        except Exception:
            base_rgbs.append((0, 0, 0))

    pulses_theme_name = "gif" if bool((args.gif or "").strip()) else args.theme
    pulses_per_poly: List[List[Pulse]] = [
        make_pulses(rng, float(duration), pulses_theme_name) for _ in polys
    ]

    max_dim = int(args.max_dim)
    scale = float(max_dim) / float(max(vb_w, vb_h))
    out_w = max(1, int(round(vb_w * scale)))
    out_h = max(1, int(round(vb_h * scale)))
    frames = int(round(float(duration) * int(fps)))

    return Scene(
        doc=svg_doc,
        polys=polys,
        vb_tuple=vb_tuple,
        out_w=out_w,
        out_h=out_h,
        fps=int(fps),
        duration=float(duration),
        frames=frames,
        poly_nx=poly_nx,
        poly_ny=poly_ny,
        glyph_nx=glyph_nx,
        glyph_ny=glyph_ny,
        base_rgbs=base_rgbs,
        override_hsv=override_hsv,
        bgcolor=bgcolor,
        pulses_per_poly=pulses_per_poly,
    )
