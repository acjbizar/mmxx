from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from lxml import etree

from .constants import SVG_NS
from .svg.parse import parse_viewbox, local_name
from .svg.style import resolve_fill
from .svg.color import parse_css_color_to_rgb
from .svg.geom import global_centroid_norm, centroid_in_glyph_norm, glyph_index_for_element
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

    # Centroids in full SVG viewBox
    poly_nx: List[float]
    poly_ny: List[float]

    # Centroids normalized per glyph group (logo grids). Falls back to full viewBox for single glyph.
    glyph_nx: List[float]
    glyph_ny: List[float]

    # 1-based glyph index for logo grids; -1 when not available.
    poly_glyph_idx: List[int]

    base_rgbs: List[RGB]
    override_hsv: Optional[Tuple[float, float, float]]
    override_hsv_per_poly: List[Optional[Tuple[float, float, float]]]
    bgcolor: Optional[str]

    pulses_per_poly: List[List[Pulse]]

    @property
    def n_polys(self) -> int:
        return len(self.polys)

    def subset(self, poly_indices: List[int]) -> "Scene":
        """Return a Scene that references a subset of polygons.

        Useful for applying a theme to only part of a logo (e.g. one glyph in a grid).
        The returned Scene shares the same underlying SVG document.
        """
        idxs = [i for i in poly_indices if 0 <= i < len(self.polys)]
        return Scene(
            doc=self.doc,
            polys=[self.polys[i] for i in idxs],
            vb_tuple=self.vb_tuple,
            out_w=self.out_w,
            out_h=self.out_h,
            fps=self.fps,
            duration=self.duration,
            frames=self.frames,
            poly_nx=[self.poly_nx[i] for i in idxs],
            poly_ny=[self.poly_ny[i] for i in idxs],
            glyph_nx=[self.glyph_nx[i] for i in idxs],
            glyph_ny=[self.glyph_ny[i] for i in idxs],
            poly_glyph_idx=[self.poly_glyph_idx[i] for i in idxs],
            base_rgbs=[self.base_rgbs[i] for i in idxs],
            override_hsv=self.override_hsv,
            override_hsv_per_poly=[self.override_hsv_per_poly[i] for i in idxs],
            bgcolor=self.bgcolor,
            pulses_per_poly=[self.pulses_per_poly[i] for i in idxs],
        )


def _flatten_colors_args(values: Optional[List[str]]) -> List[str]:
    """Accept either space-separated values (argparse nargs='+') or comma-separated lists."""
    if not values:
        return []
    out: List[str] = []
    for v in values:
        v = (v or "").strip()
        if not v:
            continue
        out.extend([p.strip() for p in v.split(",") if p.strip()])
    return out


def _build_glyph_color_overrides(colors: List[str], n_glyphs: int) -> List[Optional[RGB]]:
    """Return list where index is 1-based glyph index; index 0 unused."""

    def _is_skip(tok: str) -> bool:
        t = (tok or "").strip().lower()
        return t in {"-", "none", "null", "skip", ""}

    if not colors:
        return [None] * (n_glyphs + 1)

    if len(colors) == 1:
        tok = colors[0]
        if _is_skip(tok):
            return [None] * (n_glyphs + 1)
        rgb = parse_css_color_to_rgb(tok)
        return [None] + [rgb for _ in range(n_glyphs)]

    if len(colors) != n_glyphs:
        raise SystemExit(
            f"--colors must provide either 1 color (apply to all glyphs) or exactly {n_glyphs} colors (one per glyph). "
            f"Got {len(colors)}."
        )

    out: List[Optional[RGB]] = [None] * (n_glyphs + 1)
    for i, tok in enumerate(colors, start=1):
        out[i] = None if _is_skip(tok) else parse_css_color_to_rgb(tok)
    return out


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
    poly_glyph_idx: List[int] = []

    for p in polys:
        nx, ny = global_centroid_norm(p, vb_tuple)
        poly_nx.append(nx)
        poly_ny.append(ny)

        gx, gy = centroid_in_glyph_norm(p, vb_tuple)
        glyph_nx.append(gx)
        glyph_ny.append(gy)

        poly_glyph_idx.append(glyph_index_for_element(p, default=-1))

    override_color = args.color.strip() or None
    base_rgb_override: Optional[RGB] = None
    override_hsv: Optional[Tuple[float, float, float]] = None
    if override_color:
        base_rgb_override = parse_css_color_to_rgb(override_color)
        override_hsv = rgb255_to_hsv01(base_rgb_override)

    # Per-glyph color overrides (primarily useful for --chars)
    n_glyphs = max([g for g in poly_glyph_idx if g > 0], default=1)
    colors_tokens = _flatten_colors_args(getattr(args, "colors", None))
    glyph_overrides = _build_glyph_color_overrides(colors_tokens, n_glyphs)

    base_rgbs: List[RGB] = []
    override_hsv_per_poly: List[Optional[Tuple[float, float, float]]] = []
    for idx, p in enumerate(polys):
        gi = poly_glyph_idx[idx]
        ov_rgb: Optional[RGB] = glyph_overrides[gi] if (gi > 0 and gi < len(glyph_overrides)) else None

        if ov_rgb is not None:
            base_rgbs.append(ov_rgb)
            override_hsv_per_poly.append(rgb255_to_hsv01(ov_rgb))
            continue

        override_hsv_per_poly.append(None)

        if base_rgb_override is not None:
            base_rgbs.append(base_rgb_override)
            continue

        fill = resolve_fill(p) or "#000"
        try:
            base_rgbs.append(parse_css_color_to_rgb(fill))
        except Exception:
            base_rgbs.append((0, 0, 0))

    # Which pulse timing to use. "--to" uses the classic pulses.
    if bool((getattr(args, "to", None) or "")):
        pulses_theme_name = "classic"
    elif bool((args.gif or "").strip()):
        pulses_theme_name = "gif"
    else:
        pulses_theme_name = args.theme

    pulses_per_poly: List[List[Pulse]] = [
        make_pulses(rng, float(duration), str(pulses_theme_name)) for _ in polys
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
        poly_glyph_idx=poly_glyph_idx,
        base_rgbs=base_rgbs,
        override_hsv=override_hsv,
        override_hsv_per_poly=override_hsv_per_poly,
        bgcolor=bgcolor,
        pulses_per_poly=pulses_per_poly,
    )
