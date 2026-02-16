from __future__ import annotations
from lxml import etree
from typing import Optional, Tuple, List
from .parse import local_name
from .color import is_whiteish_color_str

def style_get(style: str, key: str) -> Optional[str]:
    if not style:
        return None
    parts = [p.strip() for p in style.split(";") if p.strip()]
    for p in parts:
        if ":" not in p:
            continue
        k, v = p.split(":", 1)
        if k.strip().lower() == key.strip().lower():
            return v.strip()
    return None

def style_set(style: str, key: str, value: str) -> str:
    key_l = key.strip().lower()
    items: List[tuple[str, str]] = []
    if style:
        parts = [p.strip() for p in style.split(";") if p.strip()]
        for p in parts:
            if ":" not in p:
                continue
            k, v = p.split(":", 1)
            k = k.strip()
            v = v.strip()
            if k.lower() == key_l:
                continue
            items.append((k, v))
    items.append((key.strip(), value.strip()))
    return "; ".join(f"{k}: {v}" for k, v in items)

def resolve_fill(el: etree._Element) -> Optional[str]:
    cur = el
    while cur is not None and isinstance(cur.tag, str):
        fill = (cur.get("fill") or "").strip()
        if fill:
            return fill
        st = (cur.get("style") or "").strip()
        if st:
            v = style_get(st, "fill")
            if v:
                return v
        cur = cur.getparent()
    return None

def strip_white_full_canvas_rects(svg_root: etree._Element, vb: Tuple[float, float, float, float]) -> None:
    minx, miny, vbw, vbh = vb
    tol = 1e-6
    rects = svg_root.xpath('.//*[local-name()="rect"]')
    for r in rects:
        if not isinstance(r.tag, str):
            continue
        if r.get("transform"):
            continue
        try:
            x = float(r.get("x", "0") or "0")
            y = float(r.get("y", "0") or "0")
            w = float(r.get("width", "0") or "0")
            h = float(r.get("height", "0") or "0")
        except Exception:
            continue

        covers = (
            abs(x - minx) < tol and
            abs(y - miny) < tol and
            abs(w - vbw) < tol and
            abs(h - vbh) < tol
        )
        if not covers:
            continue

        fill = resolve_fill(r) or ""
        if is_whiteish_color_str(fill):
            parent = r.getparent()
            if parent is not None:
                parent.remove(r)
