from __future__ import annotations
import math
from .easing import clamp01

def fract(x: float) -> float:
    return x - math.floor(x)

def hash01(x: float) -> float:
    return fract(math.sin(x) * 43758.5453123)

def noise2(x: float, y: float, seed: float) -> float:
    # 2D value noise (0..1)
    ix = math.floor(x)
    iy = math.floor(y)
    fx = x - ix
    fy = y - iy

    def h(xx: float, yy: float) -> float:
        return hash01(xx * 127.1 + yy * 311.7 + seed * 74.7)

    a = h(ix, iy)
    b = h(ix + 1.0, iy)
    c = h(ix, iy + 1.0)
    d = h(ix + 1.0, iy + 1.0)

    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)

    ab = a * (1.0 - ux) + b * ux
    cd = c * (1.0 - ux) + d * ux
    return ab * (1.0 - uy) + cd * uy

def fbm2(x: float, y: float, seed: float, octaves: int = 4) -> float:
    # Fractal-ish noise (0..1)
    amp = 0.55
    freq = 1.0
    s = 0.0
    norm = 0.0
    for i in range(octaves):
        s += amp * noise2(x * freq, y * freq, seed + i * 9.13)
        norm += amp
        amp *= 0.55
        freq *= 2.0
    if norm <= 1e-9:
        return 0.0
    return clamp01(s / norm)
