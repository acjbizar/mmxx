#!/usr/bin/env python3
from pathlib import Path
import re
from typing import Optional, Tuple, Dict

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

    # x/y default to 0 if omitted
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

def _write_text_lf(path: Path, text: str) -> None:
    # Normalize any CRLF/CR to LF, then write with explicit newline handling
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)

def main() -> None:
    # Script lives in /tools, project root is one level up
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
    total_removed = 0

    for src in files:
        name = src.name  # e.g. sketch-A.svg
        if not (name.startswith("sketch-") and name.endswith(".svg")):
            continue

        letter = name[len("sketch-") : -len(".svg")]
        if not letter:
            continue

        svg_text = src.read_text(encoding="utf-8")
        cleaned, removed = strip_large_white_square(svg_text)

        dst_twig = dst_twig_dir / f"mmxx-{letter}.svg.twig"
        dst_php = dst_php_dir / f"mmxx-{letter}.php"

        _write_text_lf(dst_twig, cleaned)
        _write_text_lf(dst_php, cleaned)

        processed += 1
        total_removed += removed

        print(f"{src} -> {dst_twig} (removed {removed} background rect(s))")
        print(f"{src} -> {dst_php} (removed {removed} background rect(s))")

    print(f"\nDone. Processed {processed} file(s). Removed {total_removed} background rect(s) total.")

if __name__ == "__main__":
    main()
