from __future__ import annotations

from typing import Any, List
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
from .nuke import NukeTheme
from .gif import GifTheme

from .none import NoneTheme
from .to import ToTheme
from .clouds import CloudsTheme
from .scoped import ScopedTheme, StackTheme


THEME_CHOICES = [
    # Default (no animation)
    "none",

    "classic", "diamond",
    "silver", "gold", "bronze",
    "ruby", "jade", "sapphire", "emerald",
    "rainbow",
    "fire", "ice",
    "valentines",
    "matrix",
    "snow",
    "clouds",
    "minecraft",
    "deidee",
    "heart",
    "static",
    "champagne",
    "camo",
    "fireworks",
    "nuke",
]


def _create_unscoped(*, scene: Scene, args: Any, rng: random.Random) -> Theme:
    # --gif always overrides the theme selection.
    if bool((args.gif or "").strip()):
        return GifTheme.create(scene=scene, args=args, rng=rng)

    # --to is a dedicated "animate toward a color" mode (defaults to white)
    if getattr(args, "to", None) is not None:
        if (args.theme or "none") != "none":
            raise SystemExit("--to can not be combined with --theme (use one or the other)")
        return ToTheme.create(scene=scene, args=args, rng=rng)

    cfg = get_theme_config(args.theme)

    if cfg.kind == "none":
        return NoneTheme.create(scene=scene, args=args, rng=rng)
    if cfg.kind == "classic":
        return ClassicTheme.create(scene=scene, args=args, rng=rng)
    if cfg.kind == "diamond":
        return DiamondTheme.create(scene=scene, args=args, rng=rng)
    if cfg.kind == "hsv":
        return HSVTheme.create(scene=scene, args=args, rng=rng)
    if cfg.kind == "clouds":
        return CloudsTheme.create(scene=scene, args=args, rng=rng)
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
    if cfg.kind == "nuke":
        return NukeTheme.create(scene=scene, args=args, rng=rng)

    raise ValueError(f"Theme kind not wired in registry: {cfg.kind!r}")


def _parse_only(only: Any) -> List[int]:
    if only is None:
        return []
    if isinstance(only, list):
        return [int(x) for x in only]
    s = str(only).strip()
    if not s:
        return []
    out: List[int] = []
    for part in s.replace(" ", "").split(","):
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            raise SystemExit(f"Invalid --only value {only!r}. Use e.g. --only=2 or --only=2,4")
    return out


def create_theme(*, scene: Scene, args: Any, rng: random.Random) -> Theme:
    """Create a theme; optionally scope it to a glyph index in --chars mode via --only."""

    only_idxs = _parse_only(getattr(args, "only", None))
    if only_idxs:
        targets = set(only_idxs)
        poly_idxs = [i for i, g in enumerate(scene.poly_glyph_idx) if g in targets]
        if poly_idxs:
            subscene = scene.subset(poly_idxs)
            inner = _create_unscoped(scene=subscene, args=args, rng=rng)
            # Always reset base colors for the full scene first, so un-themed glyphs remain stable
            base = NoneTheme.create(scene=scene, args=args, rng=rng)
            return StackTheme(themes=[base, ScopedTheme(inner=inner, scoped_scene=subscene)])

    return _create_unscoped(scene=scene, args=args, rng=rng)
