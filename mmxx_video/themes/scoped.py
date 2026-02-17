from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..scene import Scene


@dataclass
class ScopedTheme:
    """Apply an inner theme to a pre-selected Scene subset."""

    inner: object
    scoped_scene: Scene

    def apply_frame(self, scene: Scene, t: float) -> None:
        # ignore the passed scene; operate on the subset (holds references to original SVG elements)
        self.inner.apply_frame(self.scoped_scene, t)


@dataclass
class StackTheme:
    """Apply multiple themes in order."""

    themes: List[object]

    def apply_frame(self, scene: Scene, t: float) -> None:
        for th in self.themes:
            th.apply_frame(scene, t)
