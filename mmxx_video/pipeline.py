from __future__ import annotations
import shutil, tempfile
from pathlib import Path
from typing import Tuple
from lxml import etree
from .scene import Scene
from .render.svg2png import render_png
from .render.ffmpeg import encode_video_ffmpeg

def render_and_encode(*, scene: Scene, theme, out_file: Path, ext: str, keep_frames: bool) -> Tuple[Path, str]:
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
