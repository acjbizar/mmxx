#!/usr/bin/env python3
from pathlib import Path
import random
import re
import shutil
import subprocess
import tempfile
from typing import Optional, Tuple, Dict, List

from lxml import etree  # py -m pip install lxml
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union  # py -m pip install shapely

# Optional CairoSVG backend (often fails on Windows without cairo DLLs)
CAIROSVG_IMPORT_ERROR = None
try:
    import cairosvg  # py -m pip install cairosvg
except Exception as e:
    cairosvg = None
    CAIROSVG_IMPORT_ERROR = str(e)


# -------------------- deJade color --------------------

def random_dejade(rng: random.Random) -> Tuple[Tuple[int, int, int], float, str]:
    """
    fill(random(0, .5), random(.5, 1), random(0, .75), .5)
    returns (rgb 0-255, alpha, hex rrggbb)
    """
    r = rng.uniform(0.0, 0.5)
    g = rng.uniform(0.5, 1.0)
    b = rng.uniform(0.0, 0.75)
    a = 0.5

    rgb = (
        max(0, min(255, int(round(r * 255)))),
        max(0, min(255, int(round(g * 255)))),
        max(0, min(255, int(round(b * 255)))),
    )
    hexcolor = f"{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    return rgb, a, hexcolor


# -------------------- cleanup pipeline (same idea as your template script) --------------------

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


def strip_svg_comments(svg_text: str) -> str:
    return COMMENT_RE.sub("", svg_text)


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


def strip_large_white_square(svg_text: str) -> str:
    canvas_w, canvas_h = _extract_canvas_size(svg_text)

    def _remove_matches(pattern: re.Pattern, text: str) -> str:
        out = []
        last = 0
        for m in pattern.finditer(text):
            tag = m.group(0)
            if _is_full_background_rect(tag, canvas_w, canvas_h):
                out.append(text[last:m.start()])
                last = m.end()
        out.append(text[last:])
        return "".join(out)

    svg_text = _remove_matches(RECT_OPEN_CLOSE_RE, svg_text)
    svg_text = _remove_matches(RECT_SELF_CLOSING_RE, svg_text)
    return svg_text


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


def unite_svg_polygons_to_one_path(svg_text: str) -> str:
    parser = etree.XMLParser(remove_blank_text=False, recover=True, remove_comments=True)
    root = etree.fromstring(svg_text.encode("utf-8"), parser=parser)

    svg_ns = root.nsmap.get(None)

    def tag(local: str) -> str:
        return f"{{{svg_ns}}}{local}" if svg_ns else local

    polygons = root.xpath('.//*[local-name()="polygon"]')
    if not polygons:
        return svg_text

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
        return svg_text

    unioned = unary_union(shp_polys)
    d = _geom_to_single_path_d(unioned, snap_eps=1e-6, max_decimals=3)
    if not d:
        return svg_text

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

    return etree.tostring(root, encoding="unicode")


def remove_empty_groups(svg_text: str) -> str:
    parser = etree.XMLParser(remove_blank_text=False, recover=True, remove_comments=True)
    root = etree.fromstring(svg_text.encode("utf-8"), parser=parser)

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
            if parent is not None:
                parent.remove(g)

    return etree.tostring(root, encoding="unicode")


def normalize_svg(svg_text: str) -> str:
    parser = etree.XMLParser(remove_blank_text=True, recover=True, remove_comments=True)
    root = etree.fromstring(svg_text.encode("utf-8"), parser=parser)

    # Remove width/height if present; we’ll set them for export later anyway
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
    svg_text = strip_large_white_square(svg_text)
    svg_text = strip_svg_comments(svg_text)
    svg_text = unite_svg_polygons_to_one_path(svg_text)
    svg_text = remove_empty_groups(svg_text)
    svg_text = normalize_svg(svg_text)
    return svg_text


# -------------------- recolor + white background --------------------

def _viewbox_size(svg_text: str) -> Tuple[float, float]:
    m = VIEWBOX_RE.search(svg_text)
    if m:
        parts = re.split(r"[,\s]+", m.group(1).strip())
        if len(parts) == 4:
            w = _to_float(parts[2])
            h = _to_float(parts[3])
            if w and h and w > 0 and h > 0:
                return float(w), float(h)
    # fallback
    mw = WIDTH_RE.search(svg_text)
    mh = HEIGHT_RE.search(svg_text)
    if mw and mh:
        w = _to_float(mw.group(1))
        h = _to_float(mh.group(1))
        if w and h and w > 0 and h > 0:
            return float(w), float(h)
    return 240.0, 240.0


def apply_white_bg_and_dejade(svg_text: str, rgb: Tuple[int, int, int], alpha: float) -> str:
    parser = etree.XMLParser(remove_blank_text=True, recover=True, remove_comments=True)
    root = etree.fromstring(svg_text.encode("utf-8"), parser=parser)

    svg_ns = root.nsmap.get(None)
    def tag(local: str) -> str:
        return f"{{{svg_ns}}}{local}" if svg_ns else local

    # Ensure there is a page size for Inkscape's --export-area-page
    vb_w, vb_h = _viewbox_size(svg_text)
    root.set("width", str(int(round(vb_w))))
    root.set("height", str(int(round(vb_h))))

    # Background rect (white)
    bg = etree.Element(tag("rect"))
    bg.set("id", "mmxx-bg")
    bg.set("x", "0")
    bg.set("y", "0")
    bg.set("width", "100%")
    bg.set("height", "100%")
    bg.set("fill", "#fff")

    # insert after defs if present
    insert_idx = 0
    for i, ch in enumerate(list(root)):
        if etree.QName(ch).localname == "defs":
            insert_idx = i + 1
    root.insert(insert_idx, bg)

    fill_hex = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    # Recolor shapes (avoid defs + bg)
    for el in root.xpath(".//*[not(ancestor::*[local-name()='defs'])]"):
        ln = etree.QName(el).localname
        if ln == "rect" and el.get("id") == "mmxx-bg":
            continue
        if ln in {"path", "polygon", "rect", "circle", "ellipse", "polyline", "line"}:
            el.set("fill", fill_hex)
            el.set("fill-opacity", str(alpha))
            el.set("stroke", "none")

    return etree.tostring(root, encoding="unicode", pretty_print=False)


# -------------------- rendering (same fallback chain as earlier script) --------------------

def find_inkscape() -> Optional[str]:
    # First: PATH
    p = shutil.which("inkscape") or shutil.which("inkscape.exe")
    if p:
        return p

    # Common Windows install locations
    candidates = [
        r"C:\Program Files\Inkscape\bin\inkscape.exe",
        r"C:\Program Files\Inkscape\inkscape.exe",
        r"C:\Program Files (x86)\Inkscape\bin\inkscape.exe",
        r"C:\Program Files (x86)\Inkscape\inkscape.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def _render_with_cairosvg(svg_text: str, out_png: Path, out_w: int, out_h: int) -> bool:
    if cairosvg is None:
        return False
    try:
        cairosvg.svg2png(
            bytestring=svg_text.encode("utf-8"),
            write_to=str(out_png),
            output_width=out_w,
            output_height=out_h,
            background_color="white",  # extra safety; we also add a rect
        )
        return True
    except Exception:
        return False


def _render_with_inkscape(svg_text: str, out_png: Path, out_w: int, out_h: int) -> bool:
    inkscape = find_inkscape()
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
            # older inkscape fallback
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

    # Same order as earlier script (CairoSVG -> Inkscape -> rsvg)
    if _render_with_cairosvg(svg_text, out_png, out_w, out_h):
        return "cairosvg"
    if _render_with_inkscape(svg_text, out_png, out_w, out_h):
        return "inkscape"
    if _render_with_rsvg(svg_text, out_png, out_w, out_h):
        return "rsvg-convert"

    msg = (
        "No renderer available.\n"
        f"Inkscape on PATH: {bool(shutil.which('inkscape') or shutil.which('inkscape.exe'))}\n"
        f"Inkscape found via common paths: {bool(find_inkscape())}\n"
        f"rsvg-convert on PATH: {bool(shutil.which('rsvg-convert'))}\n"
    )
    if CAIROSVG_IMPORT_ERROR:
        msg += f"\nCairoSVG import issue (expected on Windows without cairo DLLs): {CAIROSVG_IMPORT_ERROR}\n"
    raise RuntimeError(msg)


# -------------------- main --------------------

def main() -> None:
    root = Path(__file__).resolve().parent.parent  # tools/ -> project root
    src_dir = root / "src"
    out_dir = root / "dist" / "images" / "instagram" / "2026"
    out_dir.mkdir(parents=True, exist_ok=True)

    letters = ["M", "X", "V", "I"]

    # If you want stable colors across runs, set a seed here:
    rng = random.Random()  # or random.Random(2026)

    for ch in letters:
        src = src_dir / f"character-{ch}.svg"
        if not src.exists():
            print(f"Skipping {ch}: missing {src}")
            continue

        svg_text = src.read_text(encoding="utf-8")
        svg_text = cleanup_svg(svg_text)

        rgb, alpha, hexcolor = random_dejade(rng)

        svg_colored = apply_white_bg_and_dejade(svg_text, rgb, alpha)

        out_png = out_dir / f"mmxx-{ch}-{hexcolor}.png"
        method = render_png(svg_colored, out_png, 1080, 1080)

        print(f"{src} -> {out_png} ({method}, fill #{hexcolor} @ {alpha})")

    print("\nDone.")


if __name__ == "__main__":
    main()
