#!/usr/bin/env python3
from pathlib import Path
import argparse
import re
import shutil
import subprocess
import tempfile
from typing import Optional, Tuple, Dict, List

from lxml import etree  # py -m pip install lxml
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union  # py -m pip install shapely

# Optional CairoSVG backend
try:
    import cairosvg  # py -m pip install cairosvg
except Exception:
    cairosvg = None


# -------------------- SVG CLEANUP (same as template script) -------------------

COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

RECT_SELF_CLOSING_RE = re.compile(r"<rect\b[^>]*?/>", re.IGNORECASE | re.DOTALL)
RECT_OPEN_CLOSE_RE   = re.compile(r"<rect\b[^>]*?>.*?</rect\s*>", re.IGNORECASE | re.DOTALL)
SVG_TAG_RE           = re.compile(r"<svg\b[^>]*>", re.IGNORECASE | re.DOTALL)
ATTR_RE              = re.compile(r'(\w+)\s*=\s*(["\'])(.*?)\2', re.DOTALL)

VIEWBOX_RE = re.compile(r'\bviewBox\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
WIDTH_RE   = re.compile(r'\bwidth\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
HEIGHT_RE  = re.compile(r'\bheight\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)


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


def strip_svg_comments(svg_text: str) -> Tuple[str, int]:
    new_text, n = COMMENT_RE.subn("", svg_text)
    return new_text, n


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
    paths: List[str] = []

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


def normalize_svg(svg_text: str) -> str:
    parser = etree.XMLParser(remove_blank_text=True, recover=True, remove_comments=True)
    root = etree.fromstring(svg_text.encode("utf-8"), parser=parser)

    for attr in ("width", "height"):
        if attr in root.attrib:
            del root.attrib[attr]

    for el in root.iter():
        if el.text is not None and el.text.strip() == "":
            el.text = None
        if el.tail is not None and el.tail.strip() == "":
            el.tail = None

    return etree.tostring(root, encoding="unicode", pretty_print=False)


def cleanup_svg(svg_text: str) -> str:
    cleaned, _ = strip_large_white_square(svg_text)
    cleaned, _ = strip_svg_comments(cleaned)
    cleaned, _ = unite_svg_polygons_to_one_path(cleaned, simplify_tolerance=0.0)
    cleaned, _ = remove_empty_groups(cleaned)
    cleaned = normalize_svg(cleaned)
    return cleaned


# -------------------- SVG SIZE / PNG RENDER ----------------------------------

def _get_svg_size(svg_text: str) -> Tuple[int, int]:
    m = VIEWBOX_RE.search(svg_text)
    if m:
        parts = re.split(r"[,\s]+", m.group(1).strip())
        if len(parts) == 4:
            w = _to_float(parts[2])
            h = _to_float(parts[3])
            if w and h and w > 0 and h > 0:
                return int(round(w)), int(round(h))

    mw = WIDTH_RE.search(svg_text)
    mh = HEIGHT_RE.search(svg_text)
    if mw and mh:
        w = _to_float(mw.group(1))
        h = _to_float(mh.group(1))
        if w and h and w > 0 and h > 0:
            return int(round(w)), int(round(h))

    return 240, 240


def _render_with_cairosvg(svg_text: str, out_png: Path, out_w: int, out_h: int) -> bool:
    if cairosvg is None:
        return False
    cairosvg.svg2png(
        bytestring=svg_text.encode("utf-8"),
        write_to=str(out_png),
        output_width=out_w,
        output_height=out_h,
    )
    return True


def _render_with_inkscape(svg_text: str, out_png: Path, out_w: int, out_h: int) -> bool:
    inkscape = shutil.which("inkscape")
    if not inkscape:
        return False

    fd, tmp_svg = tempfile.mkstemp(suffix=".svg")
    try:
        with open(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(svg_text)

        cmd = [
            inkscape,
            tmp_svg,
            "--export-type=png",
            f"--export-filename={str(out_png)}",
            "--export-area-page",
            f"--export-width={out_w}",
            f"--export-height={out_h}",
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except Exception:
            cmd_old = [
                inkscape,
                tmp_svg,
                f"--export-png={str(out_png)}",
                "--export-area-page",
                f"--export-width={out_w}",
                f"--export-height={out_h}",
            ]
            subprocess.run(cmd_old, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
    finally:
        try:
            Path(tmp_svg).unlink(missing_ok=True)
        except Exception:
            pass


def _render_with_rsvg(svg_text: str, out_png: Path, out_w: int, out_h: int) -> bool:
    rsvg = shutil.which("rsvg-convert")
    if not rsvg:
        return False

    fd, tmp_svg = tempfile.mkstemp(suffix=".svg")
    try:
        with open(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(svg_text)

        cmd = [rsvg, "-o", str(out_png), "-w", str(out_w), "-h", str(out_h), tmp_svg]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    finally:
        try:
            Path(tmp_svg).unlink(missing_ok=True)
        except Exception:
            pass


def render_png(svg_text: str, out_png: Path, out_w: int, out_h: int) -> str:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if _render_with_cairosvg(svg_text, out_png, out_w, out_h):
        return "cairosvg"
    if _render_with_inkscape(svg_text, out_png, out_w, out_h):
        return "inkscape"
    if _render_with_rsvg(svg_text, out_png, out_w, out_h):
        return "rsvg-convert"

    raise RuntimeError(
        "No renderer available. Install one of:\n"
        "  - CairoSVG:  py -m pip install cairosvg\n"
        "  - Inkscape (and ensure `inkscape` is on PATH)\n"
        "  - rsvg-convert (librsvg) (and ensure it is on PATH)\n"
    )


# --------------------------------- MAIN --------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Render src/character-*.svg plus src/logo.svg and src/sheet.svg "
            "to dist/images/mmxx-*.png and dist/images/instagram/mmxx-*.png (1080x1080)"
        )
    )
    ap.add_argument("--scale", type=float, default=1.0, help="Scale factor applied to viewBox size for dist/ (default: 1.0)")
    ap.add_argument("--size", type=int, default=0, help="Force square size for dist/ (e.g. 512). Overrides --scale if set.")
    ap.add_argument("--no-clean", action="store_true", help="Render raw SVGs without cleanup/simplification.")
    ap.add_argument("--instagram", action="store_true", default=True, help="Also render 1080x1080 to dist/images/instagram (default: on).")
    ap.add_argument("--no-instagram", dest="instagram", action="store_false", help="Disable dist/images/instagram output.")
    ap.add_argument("--ig-size", type=int, default=1080, help="Instagram output size (default: 1080).")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent  # tools/ -> project root
    src_dir = root / "src"
    dist_dir = root / "dist/images"
    ig_dir = dist_dir / "instagram"

    char_files = sorted(src_dir.glob("character-*.svg"))

    logo_file = src_dir / "logo.svg"
    sheet_file = src_dir / "sheet.svg"  # NEW: include sheet.svg

    inputs: List[Tuple[Path, str]] = []
    for f in char_files:
        name = f.name
        letter = name[len("character-") : -len(".svg")] if name.startswith("character-") and name.endswith(".svg") else None
        if letter:
            inputs.append((f, letter))

    if logo_file.is_file():
        inputs.append((logo_file, "logo"))

    if sheet_file.is_file():
        inputs.append((sheet_file, "sheet"))  # NEW

    if not inputs:
        print(f"No inputs found matching: {src_dir / 'character-*.svg'} (and no {logo_file} / {sheet_file})")
        return

    dist_dir.mkdir(parents=True, exist_ok=True)
    if args.instagram:
        ig_dir.mkdir(parents=True, exist_ok=True)

    rendered_main = 0
    rendered_ig = 0

    for f, key in inputs:
        svg_text = f.read_text(encoding="utf-8")
        if not args.no_clean:
            svg_text = cleanup_svg(svg_text)

        base_w, base_h = _get_svg_size(svg_text)

        # Main dist output
        if args.size and args.size > 0:
            out_w = out_h = int(args.size)
        else:
            out_w = max(1, int(round(base_w * args.scale)))
            out_h = max(1, int(round(base_h * args.scale)))

        out_png = dist_dir / f"mmxx-{key}.png"
        method = render_png(svg_text, out_png, out_w, out_h)
        rendered_main += 1

        msg = f"{f} -> {out_png} ({out_w}x{out_h}, {method})"

        # Instagram output (forced square)
        if args.instagram:
            ig_png = ig_dir / f"mmxx-{key}.png"
            ig_method = render_png(svg_text, ig_png, int(args.ig_size), int(args.ig_size))
            rendered_ig += 1
            msg += f" | IG -> {ig_png} ({args.ig_size}x{args.ig_size}, {ig_method})"

        print(msg)

    print(f"\nDone.")
    print(f"Rendered {rendered_main} PNG(s) to {dist_dir}.")
    if args.instagram:
        print(f"Rendered {rendered_ig} PNG(s) to {ig_dir}.")


if __name__ == "__main__":
    main()
