from __future__ import annotations
import math

def clamp01(x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x

def smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge0 == edge1:
        return 0.0
    t = clamp01((x - edge0) / (edge1 - edge0))
    return t * t * (3.0 - 2.0 * t)

def cosine_ease(x: float) -> float:
    x = clamp01(x)
    return 0.5 - 0.5 * math.cos(math.pi * x)
