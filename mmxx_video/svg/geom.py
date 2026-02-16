from __future__ import annotations
from lxml import etree
from typing import List, Tuple, Optional
from ..constants import NUM_RE, _TRANSLATE_RE
from ..anim.easing import clamp01

def parse_polygon_points(points: str) -> List[Tuple[float, float]]:
    if not points:
        return []
    nums = [float(x) for x in NUM_RE.findall(points)]
    if len(nums) < 4:
        return []
    return list(zip(nums[0::2], nums[1::2]))

def poly_centroid_local(poly: etree._Element) -> Optional[Tuple[float, float]]:
    pts = parse_polygon_points(poly.get("points", ""))
    if not pts:
        return None
    x = sum(p[0] for p in pts) / float(len(pts))
    y = sum(p[1] for p in pts) / float(len(pts))
    return (x, y)

def parse_translate(transform: str) -> Tuple[float, float]:
    if not transform:
        return (0.0, 0.0)
    m = _TRANSLATE_RE.search(transform)
    if not m:
        return (0.0, 0.0)
    tx = float(m.group(1))
    ty = float(m.group(2)) if m.group(2) is not None else 0.0
    return (tx, ty)

def accumulate_translate(el: etree._Element) -> Tuple[float, float]:
    tx = 0.0
    ty = 0.0
    cur = el
    while cur is not None and isinstance(cur.tag, str):
        tr = (cur.get("transform") or "").strip()
        if tr:
            dx, dy = parse_translate(tr)
            tx += dx
            ty += dy
        cur = cur.getparent()
    return (tx, ty)

def global_centroid_norm(poly: etree._Element, vb: Tuple[float, float, float, float]) -> Tuple[float, float]:
    minx, miny, vbw, vbh = vb
    c = poly_centroid_local(poly)
    if c is None:
        return (0.5, 0.5)
    tx, ty = accumulate_translate(poly)
    gx = c[0] + tx
    gy = c[1] + ty
    nx = (gx - minx) / vbw if vbw > 0 else 0.5
    ny = (gy - miny) / vbh if vbh > 0 else 0.5
    return (clamp01(nx), clamp01(ny))

def glyph_viewbox_for_element(el: etree._Element, fallback: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    cur = el
    while cur is not None and isinstance(cur.tag, str):
        vbw = cur.get("data-vbw")
        vbh = cur.get("data-vbh")
        if vbw and vbh:
            try:
                minx = float(cur.get("data-minx", "0") or "0")
                miny = float(cur.get("data-miny", "0") or "0")
                vw = float(vbw)
                vh = float(vbh)
                return (minx, miny, vw, vh)
            except Exception:
                break
        cur = cur.getparent()
    return fallback

def centroid_in_glyph_norm(poly: etree._Element, root_vb: Tuple[float, float, float, float]) -> Tuple[float, float]:
    minx, miny, vbw, vbh = glyph_viewbox_for_element(poly, root_vb)
    c = poly_centroid_local(poly)
    if c is None:
        return (0.5, 0.5)
    nx = (c[0] - minx) / vbw if vbw > 0 else 0.5
    ny = (c[1] - miny) / vbh if vbh > 0 else 0.5
    return (clamp01(nx), clamp01(ny))
