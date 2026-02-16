from __future__ import annotations
from typing import Any
import random
from ..scene import Scene
from .base import Theme
from .configs import get_theme_config
from .classic import ClassicTheme
from .diamond import DiamondTheme
from .hsv_materials import HSVTheme
from .minecraft import MinecraftTheme
from .deidee import DeideeTheme
from .heart import HeartTheme
from .static import StaticTheme
from .matrix import MatrixTheme
from .champagne import ChampagneTheme
from .camo import CamoTheme
from .fireworks import FireworksTheme
from .gif import GifTheme

THEME_CHOICES = [
    "classic", "diamond",
    "silver", "gold", "bronze",
    "ruby", "jade", "sapphire", "emerald",
    "rainbow",
    "fire", "ice",
    "valentines",
    "matrix",
    "snow",
    "minecraft",
    "deidee",
    "heart",
    "static",
    "champagne",
    "camo",
    "fireworks",
]

def create_theme(*, scene: Scene, args: Any, rng: random.Random) -> Theme:
    if bool((args.gif or "").strip()):
        return GifTheme.create(scene=scene, args=args, rng=rng)

    cfg = get_theme_config(args.theme)

    if cfg.kind == "classic":
        return ClassicTheme.create(scene=scene, args=args, rng=rng)
    if cfg.kind == "diamond":
        return DiamondTheme.create(scene=scene, args=args, rng=rng)
    if cfg.kind == "hsv":
        return HSVTheme.create(scene=scene, args=args, rng=rng)
    if cfg.kind == "minecraft":
        return MinecraftTheme.create(scene=scene, args=args, rng=rng)
    if cfg.kind == "deidee":
        return DeideeTheme.create(scene=scene, args=args, rng=rng)
    if cfg.kind == "heart":
        return HeartTheme.create(scene=scene, args=args, rng=rng)
    if cfg.kind == "static":
        return StaticTheme.create(scene=scene, args=args, rng=rng)
    if cfg.kind == "matrix":
        return MatrixTheme.create(scene=scene, args=args, rng=rng)
    if cfg.kind == "champagne":
        return ChampagneTheme.create(scene=scene, args=args, rng=rng)
    if cfg.kind == "camo":
        return CamoTheme.create(scene=scene, args=args, rng=rng)
    if cfg.kind == "fireworks":
        return FireworksTheme.create(scene=scene, args=args, rng=rng)

    raise ValueError(f"Theme kind not wired in registry: {cfg.kind!r}")
