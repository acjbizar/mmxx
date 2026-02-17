from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, List

from ..anim.color_math import hsv01_to_rgb255, mix_rgb, rgb_to_hex
from ..anim.easing import clamp01, smoothstep
from ..scene import Scene


@dataclass(frozen=True)
class MatrixDrop:
    """One falling stream in a single column (normalized 0..1 Y space)."""

    speed: float
    phase: float
    tail: float
    head: float
    strength: float
    flicker_freq: float
    flicker_phase: float


@dataclass
class MatrixTheme:
    """"Code rain" columns: bright heads + trailing tails.

    This operates per polygon by centroid (scene.poly_nx/poly_ny), but quantizes into
    a virtual grid so it reads as columns of characters.
    """

    name: str = "matrix"

    cols: int = 48
    rows: int = 72

    # Per-poly precomputed mapping
    poly_col: List[int] | None = None
    poly_row: List[int] | None = None
    sat_jitter: List[float] | None = None
    hue_jitter: List[float] | None = None

    # Per-column precomputed motion
    col_hue_off: List[float] | None = None
    col_flicker_freq: List[float] | None = None
    col_flicker_phase: List[float] | None = None
    drops_by_col: List[List[MatrixDrop]] | None = None

    @classmethod
    def create(cls, *, scene: Scene, args: Any, rng: random.Random) -> "MatrixTheme":
        _minx, _miny, vbw, vbh = scene.vb_tuple

        # Choose a stable number of columns/rows based on the SVG viewBox.
        # (The original single-file generator used the same idea; it makes matrix feel consistent
        # across differently-sized glyphs/logos.)
        cols = max(24, min(64, int(round(vbw / 8.0))))
        rows = max(24, min(96, int(round(vbh / 6.0))))

        col_hue_off = [rng.uniform(-0.012, 0.012) for _ in range(cols)]
        col_flicker_freq = [rng.uniform(0.15, 0.45) for _ in range(cols)]
        col_flicker_phase = [rng.uniform(0.0, 1.0) for _ in range(cols)]

        drops_by_col: List[List[MatrixDrop]] = [[] for _ in range(cols)]
        for c in range(cols):
            # Leave some columns empty to avoid an overly uniform screen.
            if rng.random() < 0.18:
                continue

            # Multiple independent drops per column keeps the rain lively.
            n_drops = 1 + (1 if rng.random() < 0.28 else 0) + (1 if rng.random() < 0.07 else 0)
            for _ in range(n_drops):
                drops_by_col[c].append(
                    MatrixDrop(
                        speed=rng.uniform(0.16, 0.46),
                        phase=rng.random(),
                        tail=rng.uniform(0.10, 0.28),
                        head=rng.uniform(0.020, 0.060),
                        strength=rng.uniform(0.55, 1.00) * (0.70 if rng.random() < 0.35 else 1.0),
                        flicker_freq=rng.uniform(0.8, 2.4),
                        flicker_phase=rng.random(),
                    )
                )

        poly_col: List[int] = []
        poly_row: List[int] = []
        sat_jitter: List[float] = []
        hue_jitter: List[float] = []
        for i in range(scene.n_polys):
            nx = clamp01(scene.poly_nx[i])
            ny = clamp01(scene.poly_ny[i])
            poly_col.append(min(cols - 1, int(nx * cols)))
            poly_row.append(min(rows - 1, int(ny * rows)))
            sat_jitter.append(rng.uniform(0.92, 1.12))
            hue_jitter.append(rng.uniform(-0.010, 0.010))

        return cls(
            cols=cols,
            rows=rows,
            poly_col=poly_col,
            poly_row=poly_row,
            sat_jitter=sat_jitter,
            hue_jitter=hue_jitter,
            col_hue_off=col_hue_off,
            col_flicker_freq=col_flicker_freq,
            col_flicker_phase=col_flicker_phase,
            drops_by_col=drops_by_col,
        )

    def _matrix_rain(self, *, ny: float, col: int, row: int, t: float) -> tuple[float, float]:
        """Return (rain_intensity, head_intensity) in [0..1]."""
        if not self.drops_by_col or col < 0 or col >= len(self.drops_by_col):
            return (0.0, 0.0)
        drops = self.drops_by_col[col]
        if not drops:
            return (0.0, 0.0)

        # Per-column flicker keeps columns from feeling locked.
        col_f = 1.0
        if self.col_flicker_freq and self.col_flicker_phase:
            col_f = 0.90 + 0.10 * math.sin(
                2.0 * math.pi * (self.col_flicker_freq[col] * t + self.col_flicker_phase[col])
            )

        best = 0.0
        best_head = 0.0
        for d in drops:
            head_pos = (d.phase + d.speed * t) % 1.0
            # Distance from the head downwards, with wrap-around so drops loop forever.
            dist = (ny - head_pos) % 1.0
            if dist > d.tail:
                continue

            head_int = 1.0 - smoothstep(0.0, d.head, dist)
            tail_int = math.exp(-dist / (d.tail * 0.35 + 1e-6))

            flick = 0.78 + 0.22 * math.sin(2.0 * math.pi * (d.flicker_freq * t + d.flicker_phase + row * 0.031))
            stream = d.strength * max(head_int, 0.70 * tail_int) * flick * col_f

            if stream > best:
                best = stream
            if head_int * d.strength > best_head:
                best_head = head_int * d.strength

        return (clamp01(best), clamp01(best_head))

    def apply_frame(self, scene: Scene, t: float) -> None:
        assert self.poly_col is not None and self.poly_row is not None
        assert self.sat_jitter is not None and self.hue_jitter is not None
        assert self.col_hue_off is not None

        base_h = 120 / 360.0

        for idx, poly in enumerate(scene.polys):
            nx = scene.poly_nx[idx]
            ny = scene.poly_ny[idx]
            col = self.poly_col[idx]
            row = self.poly_row[idx]

            rain, head = self._matrix_rain(ny=ny, col=col, row=row, t=t)

            # Dark background with a subtle animated drift (so it doesn't look dead when rain is absent).
            bg_v = 0.02 + 0.04 * (0.5 + 0.5 * math.sin(2.0 * math.pi * (0.18 * t + nx * 1.7 + ny * 0.9)))
            bg = hsv01_to_rgb255(base_h, 0.55, clamp01(bg_v))

            if rain <= 1e-6:
                poly.set("fill", rgb_to_hex(bg))
                continue

            # Discrete row stepping helps it read as "characters" within the streams.
            row_step = (row % 6) / 5.0

            v = clamp01((0.02 + 0.10 * (0.10 + 0.90 * row_step)) + 0.96 * (rain ** 1.0))
            s = clamp01((0.78 + 0.22 * rain) * self.sat_jitter[idx])
            h = (base_h + self.col_hue_off[col] + self.hue_jitter[idx] - 0.018 * head) % 1.0

            # Extra shimmer so the rain isn't too smooth.
            flick = 0.88 + 0.12 * math.sin(2.0 * math.pi * (0.9 * t + col * 0.037 + row * 0.011))
            v = clamp01(v * flick)

            fg = hsv01_to_rgb255(h, s, v)
            rgb = mix_rgb(bg, fg, 0.90)

            # Heads bloom toward white.
            head_mix = clamp01(0.10 * v + 0.42 * head)
            rgb = mix_rgb(rgb, (255, 255, 255), head_mix * 0.55)

            # Slight tint for very bright heads (CRT-ish).
            if head > 0.55:
                rgb = mix_rgb(rgb, hsv01_to_rgb255((h + 0.08) % 1.0, 0.55, 1.0), (head - 0.55) * 0.25)

            poly.set("fill", rgb_to_hex(rgb))
