from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Any, List, Tuple
from pathlib import Path
from ..scene import Scene
from ..anim.easing import clamp01
from ..anim.color_math import rgb_to_hex
from ..svg.color import parse_css_color_to_rgb

@dataclass
class GifTheme:
    name: str = "gif"
    frames_rgb: List[bytes] = None  # type: ignore[assignment]
    w: int = 0
    h: int = 0
    cum: List[float] = None  # type: ignore[assignment]
    total: float = 1.0

    @classmethod
    def create(cls, *, scene: Scene, args: Any, rng: random.Random) -> "GifTheme":
        try:
            from PIL import Image  # type: ignore
        except Exception:
            raise RuntimeError("GIF theme requires Pillow. Install with: py -m pip install pillow")

        root = Path(__file__).resolve().parent.parent.parent
        data_dir = root / "data"
        gif_name = (args.gif or "").strip()
        gif_path = (data_dir / gif_name).resolve()
        if not gif_path.is_file():
            raise SystemExit(f"--gif file not found in data/: {gif_path}")

        im = Image.open(str(gif_path))
        bg_rgb = parse_css_color_to_rgb(scene.bgcolor) if scene.bgcolor else (0, 0, 0)

        durations: List[float] = []
        frames: List[bytes] = []

        n_frames = int(getattr(im, "n_frames", 1) or 1)
        for fi in range(n_frames):
            try:
                im.seek(fi)
            except Exception:
                break
            dur_ms = im.info.get("duration", 100)
            try:
                dur_ms = float(dur_ms)
            except Exception:
                dur_ms = 100.0
            durations.append(max(1.0, dur_ms) / 1000.0)

            fr = im.convert("RGBA")
            resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
            if fr.size != (scene.out_w, scene.out_h):
                fr = fr.resize((scene.out_w, scene.out_h), resample=resample)

            bg = Image.new("RGBA", fr.size, (bg_rgb[0], bg_rgb[1], bg_rgb[2], 255))
            fr = Image.alpha_composite(bg, fr)
            frames.append(fr.convert("RGB").tobytes())

        if not frames:
            raise SystemExit(f"--gif could not be decoded into frames: {gif_path}")

        total = float(sum(durations)) if durations else 1.0
        if total <= 1e-9:
            total = 1.0

        cum: List[float] = []
        acc = 0.0
        for d in durations:
            acc += d
            cum.append(acc)

        inst = cls()
        inst.frames_rgb = frames
        inst.w = scene.out_w
        inst.h = scene.out_h
        inst.cum = cum if cum else [total]
        inst.total = total
        return inst

    def _frame_index_at_time(self, tsec: float) -> int:
        if len(self.frames_rgb) == 1:
            return 0
        pos = tsec % self.total
        for j, c in enumerate(self.cum):
            if pos < c:
                return j
        return len(self.frames_rgb) - 1

    def _sample_rgb(self, frame_bytes: bytes, nx: float, ny: float) -> Tuple[int, int, int]:
        x = int(round(clamp01(nx) * (self.w - 1)))
        y = int(round(clamp01(ny) * (self.h - 1)))
        off = (y * self.w + x) * 3
        return (frame_bytes[off], frame_bytes[off + 1], frame_bytes[off + 2])

    def apply_frame(self, scene: Scene, t: float) -> None:
        fi = self._frame_index_at_time(t)
        fb = self.frames_rgb[fi]
        for idx, poly in enumerate(scene.polys):
            rgb = self._sample_rgb(fb, scene.poly_nx[idx], scene.poly_ny[idx])
            poly.set("fill", rgb_to_hex(rgb))
