from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple

from lxml import etree

from .scene import Scene
from .themes.base import Theme
from .render.svg2png import render_png
from .render.ffmpeg import encode_video_ffmpeg

def render_and_encode(
    *,
    scene: Scene,
    theme: Theme,
    out_file: Path,
    ext: str,
    keep_frames: bool,
    export_png_indices: Optional[Set[int]] = None,
    export_png_dir: Optional[Path] = None,
) -> Tuple[Path, str]:
    tmp_root = Path(tempfile.mkdtemp(prefix="mmxx_video_"))
    frames_dir = tmp_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    renderer_used = None
    try:
        for i in range(scene.frames):
            t = i / float(scene.fps)
            theme.apply_frame(scene, t)
            svg_bytes = etree.tostring(scene.doc, encoding="utf-8", xml_declaration=False)
            out_png = frames_dir / f"frame_{i:05d}.png"
            renderer_used = render_png(svg_bytes, out_png, scene.out_w, scene.out_h)

            if export_png_indices and i in export_png_indices:
                dst_dir = export_png_dir or out_file.parent
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / f"{out_file.stem}_frame_{i:05d}.png"
                shutil.copyfile(out_png, dst)

        encode_video_ffmpeg(frames_dir, scene.fps, out_file, ext)
        return out_file, (renderer_used or "?")
    finally:
        if keep_frames:
            try:
                root = Path(__file__).resolve().parent.parent
                kept = root / "dist" / "videos" / "_frames"
                kept.mkdir(parents=True, exist_ok=True)
                dst = kept / (out_file.stem + "_frames")
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(frames_dir, dst)
            except Exception:
                pass
        try:
            shutil.rmtree(tmp_root)
        except Exception:
            pass


def export_png_frames(
    *,
    scene: Scene,
    theme: Theme,
    out_stem: str,
    out_dir: Path,
    frames: Iterable[int],
) -> Tuple[Path, str]:
    """Render a selected set of frames directly to PNG.

    Useful for quick previews or thumbnails without generating a video.
    Returns (out_dir, renderer_used).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    renderer_used = None
    for i in sorted(set(int(x) for x in frames)):
        if i < 0 or i >= scene.frames:
            continue
        t = i / float(scene.fps)
        theme.apply_frame(scene, t)
        svg_bytes = etree.tostring(scene.doc, encoding="utf-8", xml_declaration=False)
        out_png = out_dir / f"{out_stem}_frame_{i:05d}.png"
        renderer_used = render_png(svg_bytes, out_png, scene.out_w, scene.out_h)
    return out_dir, (renderer_used or "?")
