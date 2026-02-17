from __future__ import annotations
import argparse
import math
import random
import re
from pathlib import Path
from typing import List, Tuple, Optional

from lxml import etree

from .constants import _CODEPOINT_RE, _HEX_RE
from .svg.logo import build_logo_svg_from_chars_grid
from .scene import build_scene_from_args
from .themes.registry import THEME_CHOICES, create_theme
from .pipeline import render_and_encode
from .util import timestamped_if_exists
from .export.svgjs import export_svgjs

def parse_char_or_codepoint(s: str) -> Tuple[str, int, str]:
    raw = (s or "").strip()
    if not raw:
        raise ValueError("Empty char/codepoint")

    if len(raw) == 1:
        cp = ord(raw)
        return (f"u{cp:04x}".lower(), cp, raw)

    m = _CODEPOINT_RE.match(raw)
    if m:
        cp = int(m.group(1), 16)
        disp = chr(cp) if 0 <= cp <= 0x10FFFF else raw
        return (f"u{cp:04x}".lower(), cp, disp)

    if _HEX_RE.match(raw):
        cp = int(raw, 16)
        disp = chr(cp) if 0 <= cp <= 0x10FFFF else raw
        return (f"u{cp:04x}".lower(), cp, disp)

    raise ValueError(f"Not a single character or codepoint token: {raw!r}")

def parse_chars_arg(chars: str) -> List[Tuple[str, int, str]]:
    s = (chars or "").strip()
    if not s:
        return []
    toks = [t for t in re.split(r"\s+", s) if t]
    if len(toks) > 1 and all((_CODEPOINT_RE.match(t) or _HEX_RE.match(t)) for t in toks):
        return [parse_char_or_codepoint(t) for t in toks]
    compact = "".join(c for c in s if not c.isspace())
    return [parse_char_or_codepoint(c) for c in compact]

def safe_logo_key(items: List[Tuple[str, int, str]]) -> str:
    out: List[str] = []
    for token, _cp, disp in items:
        if len(disp) == 1 and ord(disp) < 128 and disp.isalnum():
            out.append(disp.lower())
        else:
            out.append(token.lower())
    return "".join(out)

def resolve_glyph_svg_path(src_dir: Path, prefix: str, token: str, disp: str) -> Path:
    if len(disp) == 1:
        p1 = src_dir / f"{prefix}-{disp}.svg"
        if p1.is_file():
            return p1
    return src_dir / f"{prefix}-{token}.svg"

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate a polygon animation video from glyph SVGs (single or NxN logo).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--char", type=str, default=None, help="Single character OR codepoint token (uXXXX / U+XXXX / XXXX).")
    g.add_argument("--chars", type=str, default=None,
                  help="Square count of chars: 4->2x2, 9->3x3, 16->4x4. Either literal characters (spaces ignored) or whitespace-separated codepoint tokens.")
    ap.add_argument("--inverse", action="store_true", help="Use src/inverse-*.svg instead of src/character-*.svg")
    ap.add_argument("--gap", type=int, default=0, choices=[0, 1],
                    help="Logo spacing (only for --chars): 0 = no gaps (default), 1 = pad+gap = 1/8 of cell size.")
    ap.add_argument("--color", type=str, default="", help="Override base color (CSS color).")
    ap.add_argument(
        "--colors",
        nargs="+",
        default=None,
        help=(
            "Per-glyph base color overrides (mainly for --chars). Provide either 1 color (applies to all glyphs) "
            "or exactly N colors where N is the number of glyphs. You can pass space-separated values or a single "
            "comma-separated string. Examples: --colors red green blue OR --colors=red,green,blue"
        ),
    )
    ap.add_argument("--bgcolor", type=str, default="", help="Override background color (CSS color).")
    ap.add_argument("--gif", type=str, default="", help="Use an image from data/ as a theme source (colors + animation).")
    ap.add_argument("--theme", type=str, default="none", choices=THEME_CHOICES, help="Theme (default: none = no animation).")
    ap.add_argument("--to", nargs="?", const="white", default=None, help="Animate polygons toward a target color using the default pulse animation (e.g. --to or --to=yellow).")
    ap.add_argument("--only", type=str, default=None, help="In --chars mode, apply the chosen theme only to 1-based glyph index(es): --only=2 or --only=2,4")
    ap.add_argument("--minecraft-texture", type=str, default="",
                    help="Minecraft theme only: path or URL to texture PNG (defaults to wiki URL).")
    ap.add_argument("--duration", type=float, default=12.0, help="Duration in seconds (default: 12).")
    ap.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30).")
    ap.add_argument("--ext", type=str, default="mp4", help="Output extension: mp4 or webm (default: mp4).")
    ap.add_argument("--export", type=str, default="video", choices=["video", "svgjs"], help="Export type: video (default) or self-contained SVG+JS.")
    ap.add_argument("--max-dim", type=int, default=1080, help="Render so max(viewBox w,h) becomes this size (default: 1080).")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for repeatable animation.")
    ap.add_argument("--keep-frames", action="store_true", help="Keep rendered PNG frames (for debugging).")
    return ap

def main(argv: Optional[List[str]] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    src_dir = root / "src"
    out_dir = root / "dist" / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = "inverse" if args.inverse else "character"
    parser = etree.XMLParser(remove_blank_text=False, recover=True, remove_comments=False)

    if args.char is not None:
        token, _cp, disp = parse_char_or_codepoint(args.char)
        in_svg_path = resolve_glyph_svg_path(src_dir, prefix, token, disp)
        if not in_svg_path.is_file():
            raise SystemExit(f"Input SVG not found: {in_svg_path}")

        doc = etree.fromstring(in_svg_path.read_bytes(), parser=parser)
        out_stem = f"{prefix}-{token}"
        out_file = out_dir / f"{out_stem}.{args.ext.lower()}"
        label = f"{prefix} {disp!r} ({token})"
        is_logo = False
    else:
        items = parse_chars_arg(args.chars or "")
        if not items:
            raise SystemExit("--chars is empty after parsing.")

        n = len(items)
        grid_n = int(round(math.sqrt(n)))
        if grid_n * grid_n != n or grid_n not in (2, 3, 4):
            raise SystemExit("--chars must contain 4, 9, or 16 items for 2x2 / 3x3 / 4x4 grids.")

        char_paths: List[Path] = []
        for token, _cp, disp in items:
            char_paths.append(resolve_glyph_svg_path(src_dir, prefix, token, disp))

        doc = build_logo_svg_from_chars_grid(char_paths, grid_n=grid_n, gap_flag=args.gap)

        safe_key = safe_logo_key(items)
        out_stem = f"logo-{'inv-' if args.inverse else ''}{safe_key}"
        out_file = out_dir / f"{out_stem}.{args.ext.lower()}"
        shown = "".join(d for _t, _cp, d in items)
        label = f"logo {shown!r} ({grid_n}x{grid_n}, gap={args.gap}, {prefix})"
        is_logo = True

    if args.export == "svgjs":
        out_file = out_file.with_suffix(".svg")

    out_file = timestamped_if_exists(out_file)

    rng = random.Random(args.seed)

    scene = build_scene_from_args(
        args=args,
        rng=rng,
        svg_doc=doc,
        duration=float(args.duration),
        fps=int(args.fps),
    )

    theme = create_theme(scene=scene, args=args, rng=rng)
    if args.export == "svgjs":
        export_svgjs(scene, theme, out_file, duration=float(args.duration), fps_hint=int(args.fps))
        renderer = "svgjs"
    else:
        out_file, renderer = render_and_encode(scene=scene, theme=theme, out_file=out_file, ext=args.ext.lower(), keep_frames=bool(args.keep_frames))
    print(f"Output: {out_file}")
    print(f"Theme:  {'gif:' + (args.gif.strip() or '') if bool((args.gif or '').strip()) else args.theme}")
    print(f"Input:  {label}")
    print(f"Frames: {scene.frames} @ {scene.fps}fps, size={scene.out_w}x{scene.out_h}, renderer={renderer}")
    return 0
