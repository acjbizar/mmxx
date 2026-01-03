#!/usr/bin/env python3
from pathlib import Path
import re
from typing import Optional, Tuple, Dict, List

from lxml import etree  # py -m pip install lxml
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union  # py -m pip install shapely

# --- Comment stripping --------------------------------------------------------

COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

def strip_svg_comments(svg_text: str) -> Tuple[str, int]:
    new_text, n = COMMENT_RE.subn("", svg_text)
    return new_text, n

# --- Background-rect stripping ------------------------------------------------

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
    rx = round(x)
    if abs(x - rx) <= snap_eps:
        return str(int(rx))
    s = f"{x:.{max_decimals}f}".rstrip("0").rstrip(".")
    return s if s else "0"

def _ring_to_path(coords, snap_eps: float, max_decimals: int) -> str:
    coords = list(coords)
    if len(coords) < 4:
        return ""
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
    paths = []

    def add_polygon(poly: Polygon):
        ext = _ring_to_path(poly.exterior.coords, snap_eps, max_decimals)
        if ext:
            paths.append(ext)
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
    parser = etree.XMLParser(remove_blank_text=False, recover=True, remove_comments=True)
    root = etree.fromstring(svg_text.encode("utf-8"), parser=parser)

    svg_ns = root.nsmap.get(None)
    def tag(local: str) -> str:
        return f"{{{svg_ns}}}{local}" if svg_ns else local

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

    for p in polygons:
        parent = p.getparent()
        if parent is not None:
            parent.remove(p)

    parent_for_path = polygons[0].getparent() if polygons else None

    path_el = etree.Element(tag("path"))
    path_el.set("d", d)
    path_el.set("fill-rule", "evenodd")

    if parent_for_path is None:
        root.append(path_el)
    else:
        parent_for_path.append(path_el)

    return etree.tostring(root, encoding="unicode"), len(polygons)

# --- Remove empty <g> elements ------------------------------------------------

def remove_empty_groups(svg_text: str) -> Tuple[str, int]:
    parser = etree.XMLParser(remove_blank_text=False, recover=True, remove_comments=True)
    root = etree.fromstring(svg_text.encode("utf-8"), parser=parser)

    removed = 0
    while True:
        to_remove = []
        for g in root.xpath('.//*[local-name()="g"]'):
            if len(g) != 0:
                continue
            if (g.text or "").strip():
                continue
            to_remove.append(g)

        if not to_remove:
            break

        for g in to_remove:
            parent = g.getparent()
            if parent is None:
                continue
            parent.remove(g)
            removed += 1

    return etree.tostring(root, encoding="unicode"), removed

# --- Final normalize: remove root width/height + remove whitespace ------------

def normalize_svg(svg_text: str) -> Tuple[str, int, int]:
    parser = etree.XMLParser(remove_blank_text=True, recover=True, remove_comments=True)
    root = etree.fromstring(svg_text.encode("utf-8"), parser=parser)

    removed_dims = 0
    for attr in ("width", "height"):
        if attr in root.attrib:
            del root.attrib[attr]
            removed_dims += 1

    removed_ws = 0
    for el in root.iter():
        if el.text is not None and el.text.strip() == "":
            el.text = None
            removed_ws += 1
        if el.tail is not None and el.tail.strip() == "":
            el.tail = None
            removed_ws += 1

    out = etree.tostring(root, encoding="unicode", pretty_print=False)
    return out, removed_dims, removed_ws

# --- File writing helper ------------------------------------------------------

def _write_text_lf(path: Path, text: str) -> None:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)

# --- Main --------------------------------------------------------------------

def main() -> None:
    root = Path(__file__).resolve().parent.parent

    src_dir = root / "src"
    dst_twig_dir = root / "templates"
    dst_element_dir = root / "templates" / "element"  # <-- changed here

    if not src_dir.is_dir():
        raise SystemExit(f"Source folder not found: {src_dir}")

    dst_twig_dir.mkdir(parents=True, exist_ok=True)
    dst_element_dir.mkdir(parents=True, exist_ok=True)

    char_files = sorted(src_dir.glob("character-*.svg"))
    logo_file = src_dir / "logo.svg"

    if not char_files and not logo_file.is_file():
        print(f"No files found matching {src_dir / 'character-*.svg'} and no {logo_file}")
        return

    processed = 0
    total_bg_removed = 0
    total_comments_removed = 0
    total_polys_united = 0
    total_groups_removed = 0
    total_dims_removed = 0
    total_ws_removed = 0

    def process_one(src: Path, key: str) -> None:
        nonlocal processed
        nonlocal total_bg_removed, total_comments_removed, total_polys_united
        nonlocal total_groups_removed, total_dims_removed, total_ws_removed

        svg_text = src.read_text(encoding="utf-8")

        cleaned, bg_removed = strip_large_white_square(svg_text)
        cleaned, comments_removed = strip_svg_comments(cleaned)
        cleaned, poly_count = unite_svg_polygons_to_one_path(cleaned, simplify_tolerance=0.0)
        cleaned, groups_removed = remove_empty_groups(cleaned)
        cleaned, dims_removed, ws_removed = normalize_svg(cleaned)

        dst_twig = dst_twig_dir / f"_mmxx-{key}.svg.twig"
        dst_php  = dst_element_dir / f"mmxx-{key}.php"

        _write_text_lf(dst_twig, cleaned)
        _write_text_lf(dst_php, cleaned)

        processed += 1
        total_bg_removed += bg_removed
        total_comments_removed += comments_removed
        total_polys_united += poly_count
        total_groups_removed += groups_removed
        total_dims_removed += dims_removed
        total_ws_removed += ws_removed

        print(
            f"{src} -> {dst_twig} / {dst_php} "
            f"(bg {bg_removed}, comments {comments_removed}, polys {poly_count}, "
            f"empty-g {groups_removed}, dims {dims_removed}, ws {ws_removed})"
        )

    # Characters
    for src in char_files:
        name = src.name
        if not (name.startswith("character-") and name.endswith(".svg")):
            continue

        letter = name[len("character-") : -len(".svg")]
        if not letter:
            continue

        process_one(src, letter)

    # Logo (same treatment)
    if logo_file.is_file():
        process_one(logo_file, "logo")

    print(f"\nDone. Processed {processed} file(s).")
    print(f"Removed {total_bg_removed} background rect(s) total.")
    print(f"Removed {total_comments_removed} comment(s) total.")
    print(f"United {total_polys_united} polygon(s) total into single-path shapes.")
    print(f"Removed {total_groups_removed} empty group(s) total.")
    print(f"Removed {total_dims_removed} root width/height attribute(s) total.")
    print(f"Removed {total_ws_removed} whitespace node(s) total.")

if __name__ == "__main__":
    main()
