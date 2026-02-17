from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List
from .easing import clamp01

@dataclass
class Pulse:
    t0: float
    half: float
    amp: float
    power: float = 1.0

    def value(self, t: float) -> float:
        dt = abs(t - self.t0)
        if dt >= self.half:
            return 0.0
        x = dt / self.half
        base = 0.5 * (1.0 + math.cos(math.pi * x))
        if self.power != 1.0:
            base = base ** self.power
        return self.amp * base

def whiteness_at(t: float, pulses: List[Pulse]) -> float:
    a = 0.0
    for p in pulses:
        a = max(a, p.value(t))
    return clamp01(a)

def facet_shimmer(t: float, freq: float, phase: float) -> float:
    s1 = 0.5 + 0.5 * math.sin(2.0 * math.pi * (freq * t + phase))
    s2 = 0.5 + 0.5 * math.sin(2.0 * math.pi * ((freq * 0.47) * t + (phase * 1.63)))
    s = s1 * s2
    return clamp01(s ** 2.0)

def make_pulses(rng: random.Random, duration: float, theme: str) -> List[Pulse]:
    if theme in {"deidee", "matrix", "heart", "static", "champagne", "camo", "fireworks", "nuke", "gif", "none", "clouds"}:
        return []

    if theme != "classic":
        pulses: List[Pulse] = []

        # Legacy behavior: minecraft had slightly more frequent + shorter glints.
        n_glints = rng.randint(2, 4) if theme == "minecraft" else rng.randint(1, 3)
        for _ in range(n_glints):
            t0 = rng.uniform(0.0, duration)
            half = rng.uniform(1.40, 3.80) if theme == "minecraft" else rng.uniform(2.10, 5.80)
            amp = rng.uniform(0.70, 1.00)
            power = rng.uniform(1.0, 1.30)
            pulses.append(Pulse(t0, half, amp, power=power))

        if rng.random() < 0.60:
            t0 = rng.uniform(0.0, duration)
            half = rng.uniform(3.50, 7.50)
            amp = rng.uniform(0.06, 0.14)
            pulses.append(Pulse(t0, half, amp, power=1.0))
        return pulses

    n = rng.randint(3, 7)
    pulses: List[Pulse] = []
    for _ in range(n):
        t0 = rng.uniform(0.0, duration)
        half = rng.uniform(0.25, 1.10)
        amp = rng.uniform(0.55, 1.00)
        pulses.append(Pulse(t0, half, amp, power=1.0))
    return pulses
