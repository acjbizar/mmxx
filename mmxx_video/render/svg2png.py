from __future__ import annotations
import os, shutil, subprocess, tempfile
from pathlib import Path

try:
    import cairosvg  # type: ignore
except Exception:
    cairosvg = None

def _render_with_cairosvg(svg_bytes: bytes, out_png: Path, out_w: int, out_h: int) -> bool:
    if cairosvg is None:
        return False
    cairosvg.svg2png(bytestring=svg_bytes, write_to=str(out_png), output_width=out_w, output_height=out_h)
    return True

def _render_with_inkscape(svg_bytes: bytes, out_png: Path, out_w: int, out_h: int) -> bool:
    inkscape = shutil.which("inkscape")
    if not inkscape:
        return False

    fd, tmp_svg = tempfile.mkstemp(suffix=".svg")
    os.close(fd)
    try:
        Path(tmp_svg).write_bytes(svg_bytes)
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

def _render_with_rsvg(svg_bytes: bytes, out_png: Path, out_w: int, out_h: int) -> bool:
    rsvg = shutil.which("rsvg-convert")
    if not rsvg:
        return False

    fd, tmp_svg = tempfile.mkstemp(suffix=".svg")
    os.close(fd)
    try:
        Path(tmp_svg).write_bytes(svg_bytes)
        cmd = [rsvg, "-o", str(out_png), "-w", str(out_w), "-h", str(out_h), tmp_svg]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    finally:
        try:
            Path(tmp_svg).unlink(missing_ok=True)
        except Exception:
            pass

def render_png(svg_bytes: bytes, out_png: Path, out_w: int, out_h: int) -> str:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if _render_with_cairosvg(svg_bytes, out_png, out_w, out_h):
        return "cairosvg"
    if _render_with_inkscape(svg_bytes, out_png, out_w, out_h):
        return "inkscape"
    if _render_with_rsvg(svg_bytes, out_png, out_w, out_h):
        return "rsvg-convert"

    raise RuntimeError(
        "No SVG renderer available. Install one of:\n"
        "  - CairoSVG:  py -m pip install cairosvg  (recommended)\n"
        "  - Inkscape (ensure `inkscape` is on PATH)\n"
        "  - rsvg-convert (librsvg) (ensure it is on PATH)\n"
    )
