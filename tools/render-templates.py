#!/usr/bin/env python3
from pathlib import Path
import re
from typing import Optional, Tuple, Dict, List

from lxml import etree  # py -m pip install lxml
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union  # py -m pip install shapely

# --- Background-rect stripping (same idea as before) --------------------------

RECT_SELF_CLOSING_RE = re.compile(r"<rect\b[^>]*?/>", re.IGNORECASE | re.DOTALL)
RECT_OPEN_CLOSE_RE   = re.compile(r"<rect\b[^>]*?>.*?</rect\s*>", re.IGNORECASE | re.DOTALL)
SVG_TAG_RE           = re.compile(r"<svg\b[^>]*>", re.IGNORECASE | re.DOTALL)
ATTR_RE              = re.compile(r'(\w+)\s*=\s*(["\'])(.*?)\2', re.DOTALL)

def _to_float(value: str) -> Optional[float]:
    if value is None:
        return None
    v = value.strip().lower().replace("px", "").strip()
    if not v or "%" in v:
        return None
    try:
        return float(v)
    except ValueError:
        return None

def _parse_attrs(tag_text: str) -> Dict[str, str]:
    return {k: v for (k, _, v) in ATTR_RE.findall(tag_text)}

def _extract_canvas_size(svg_text: str) -> Tuple[Optional[float], Optional[float]]:
    m = SVG_TAG_RE.search(svg_text)
    if not m:
        return (None, None)

    svg_tag = m.group(0)
    attrs = _parse_attrs(svg_tag)

    w = _to_float(attrs.get("width", ""))
    h = _to_float(attrs.get("height", ""))

    if w is not None and h is not None:
        return (w, h)

    vb = attrs.get("viewBox") or attrs.get("viewbox")
    if vb:
        parts = re.split(r"[,\s]+", vb.strip())
        if len(parts) == 4:
            vb_w = _to_float(parts[2])
            vb_h = _to_float(parts[3])
            return (vb_w, vb_h)

    return (None, None)

def _is_white_fill(attrs: Dict[str, str]) -> bool:
    fill = (attrs.get("fill") or "").strip().lower()

    if not fill:
        style = (attrs.get("style") or "").lower()
        m = re.search(r"fill\s*:\s*([^;]+)", style)
        if m:
            fill = m.group(1).strip()

    fill = fill.replace(" ", "")

    if fill in {"#fff", "#ffffff", "white"}:
        return True

    if fill.startswith("rgb(") and fill.endswith(")"):
        inside = fill[4:-1]
        nums = [p.strip() for p in inside.split(",")]
        if len(nums) == 3 and all(n == "255" for n in nums):
            return True

    return False

def _is_full_background_rect(tag_text: str, canvas_w: Optional[float], canvas_h: Optional[float]) -> bool:
    attrs = _parse_attrs(tag_text)

    if not _is_white_fill(attrs):
        return False
    if canvas_w is None or canvas_h is None:
        return False

    x = _to_float(attrs.get("x", "0")) or 0.0
    y = _to_float(attrs.get("y", "0")) or 0.0
    w = _to_float(attrs.get("width", ""))
    h = _to_float(attrs.get("height", ""))

    if w is None or h is None:
        return False

    eps = 1e-6
    return (
        abs(x - 0.0) < eps and
        abs(y - 0.0) < eps and
        abs(w - canvas_w) < eps and
        abs(h - canvas_h) < eps
    )

def strip_large_white_square(svg_text: str) -> Tuple[str, int]:
    canvas_w, canvas_h = _extract_canvas_size(svg_text)
    removed = 0

    def _remove_matches(pattern: re.Pattern, text: str) -> str:
        nonlocal removed
        out = []
        last = 0
        for m in pattern.finditer(text):
            tag = m.group(0)
            if _is_full_background_rect(tag, canvas_w, canvas_h):
                out.append(text[last:m.start()])
                last = m.end()
                removed += 1
        out.append(text[last:])
        return "".join(out)

    svg_text = _remove_matches(RECT_OPEN_CLOSE_RE, svg_text)
    svg_text = _remove_matches(RECT_SELF_CLOSING_RE, svg_text)
    return svg_text, removed

# --- Polygon union ("Illustrator Unite") --------------------------------------

def _parse_points(points_str: str) -> List[Tuple[float, float]]:
    # robust: works with commas/spaces/newlines
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", points_str or "")
    if len(nums) < 6 or (len(nums) % 2) != 0:
        return []
    coords = []
    it = iter(nums)
    for x in it:
        y = next(it)
        coords.append((float(x), float(y)))
    return coords

def _fmt_num(x: float, snap_eps: float = 1e-6, max_decimals: int = 3) -> str:
    # snap near-integers to ints (keeps your grid crisp)
    rx = round(x)
    if abs(x - rx) <= snap_eps:
        return str(int(rx))
    s = f"{x:.{max_decimals}f}".rstrip("0").rstrip(".")
    return s if s else "0"

def _ring_to_path(coords, snap_eps: float, max_decimals: int) -> str:
    coords = list(coords)
    if len(coords) < 4:
        return ""
    # shapely rings are closed: last == first; drop the last
    if coords[0] == coords[-1]:
        coords = coords[:-1]
    if not coords:
        return ""
    parts = [f"M {_fmt_num(coords[0][0], snap_eps, max_decimals)} {_fmt_num(coords[0][1], snap_eps, max_decimals)}"]
    for (x, y) in coords[1:]:
        parts.append(f"L {_fmt_num(x, snap_eps, max_decimals)} {_fmt_num(y, snap_eps, max_decimals)}")
    parts.append("Z")
    return " ".join(parts)

def _geom_to_single_path_d(geom, snap_eps: float = 1e-6, max_decimals: int = 3) -> str:
    # Returns one "d" string, potentially with multiple subpaths (still one <path> element)
    paths = []

    def add_polygon(poly: Polygon):
        # exterior
        ext = _ring_to_path(poly.exterior.coords, snap_eps, max_decimals)
        if ext:
            paths.append(ext)
        # holes
        for interior in poly.interiors:
            hole = _ring_to_path(interior.coords, snap_eps, max_decimals)
            if hole:
                paths.append(hole)

    if geom.is_empty:
        return ""

    if isinstance(geom, Polygon):
        add_polygon(geom)
    elif isinstance(geom, MultiPolygon):
        for p in geom.geoms:
            add_polygon(p)
    else:
        # GeometryCollection etc: keep polygonal parts
        try:
            for g in geom.geoms:
                if isinstance(g, Polygon):
                    add_polygon(g)
                elif isinstance(g, MultiPolygon):
                    for p in g.geoms:
                        add_polygon(p)
        except Exception:
            pass

    return " ".join(paths)

def unite_svg_polygons_to_one_path(svg_text: str, simplify_tolerance: float = 0.0) -> Tuple[str, int]:
    """
    Finds all <polygon> elements, unions them, replaces them with a single <path>.
    Returns (new_svg_text, polygons_consumed_count)
    """
    parser = etree.XMLParser(remove_blank_text=False, recover=True)
    root = etree.fromstring(svg_text.encode("utf-8"), parser=parser)

    # find all polygons anywhere
    polygons = root.xpath('.//*[local-name()="polygon"]')
    if not polygons:
        return (svg_text, 0)

    shp_polys = []
    for p in polygons:
        pts = _parse_points(p.get("points", ""))
        if len(pts) < 3:
            continue
        try:
            poly = Polygon(pts)
            if not poly.is_valid:
                # typical fix for minor self-intersections
                poly = poly.buffer(0)
            if not poly.is_empty:
                shp_polys.append(poly)
        except Exception:
            continue

    if not shp_polys:
        return (svg_text, 0)

    unioned = unary_union(shp_polys)

    if simplify_tolerance and simplify_tolerance > 0:
        unioned = unioned.simplify(simplify_tolerance, preserve_topology=True)

    d = _geom_to_single_path_d(unioned, snap_eps=1e-6, max_decimals=3)
    if not d:
        return (svg_text, 0)

    # Remove all original polygons
    for p in polygons:
        parent = p.getparent()
        if parent is not None:
            parent.remove(p)

    # Add ONE path element.
    # Put it into the first polygon's former parent if possible, otherwise append to root.
    parent_for_path = None
    try:
        parent_for_path = polygons[0].getparent()
    except Exception:
        parent_for_path = None

    path_el = etree.Element("path")
    path_el.set("d", d)
    path_el.set("fill-rule", "evenodd")  # makes holes reliable
    # No explicit fill: it will inherit from <g fill="#000"> if that exists.

    if parent_for_path is None:
        root.append(path_el)
    else:
        parent_for_path.append(path_el)

    out = etree.tostring(root, encoding="unicode")
    return (out, len(polygons))

# --- File writing helper (LF newlines even on Windows) ------------------------

def _write_text_lf(path: Path, text: str) -> None:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)

# --- Main --------------------------------------------------------------------

def main() -> None:
    root = Path(__file__).resolve().parent.parent

    src_dir = root / "tests"
    dst_twig_dir = root / "templates"
    dst_php_dir = root / "templates" / "elements"

    if not src_dir.is_dir():
        raise SystemExit(f"Source folder not found: {src_dir}")

    dst_twig_dir.mkdir(parents=True, exist_ok=True)
    dst_php_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("sketch-*.svg"))
    if not files:
        print(f"No files found matching {src_dir / 'sketch-*.svg'}")
        return

    processed = 0
    total_bg_removed = 0
    total_polys_united = 0

    for src in files:
        name = src.name
        if not (name.startswith("sketch-") and name.endswith(".svg")):
            continue

        letter = name[len("sketch-") : -len(".svg")]
        if not letter:
            continue

        svg_text = src.read_text(encoding="utf-8")

        cleaned, bg_removed = strip_large_white_square(svg_text)

        # Unite polygons into one path (Illustrator “Unite”)
        cleaned, poly_count = unite_svg_polygons_to_one_path(
            cleaned,
            simplify_tolerance=0.0,  # set e.g. 0.25 if you want fewer points
        )

        dst_twig = dst_twig_dir / f"mmxx-{letter}.svg.twig"
        dst_php = dst_php_dir / f"mmxx-{letter}.php"

        _write_text_lf(dst_twig, cleaned)
        _write_text_lf(dst_php, cleaned)

        processed += 1
        total_bg_removed += bg_removed
        total_polys_united += poly_count

        print(f"{src} -> {dst_twig} (bg rect removed {bg_removed}, polygons united {poly_count})")
        print(f"{src} -> {dst_php} (bg rect removed {bg_removed}, polygons united {poly_count})")

    print(f"\nDone. Processed {processed} file(s).")
    print(f"Removed {total_bg_removed} background rect(s) total.")
    print(f"United {total_polys_united} polygon(s) total into single-path shapes.")

if __name__ == "__main__":
    main()
