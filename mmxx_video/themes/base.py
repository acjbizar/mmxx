from __future__ import annotations
from typing import Protocol, Any
import random
from ..scene import Scene

class Theme(Protocol):
    name: str
    def apply_frame(self, scene: Scene, t: float) -> None: ...
    @classmethod
    def create(cls, *, scene: Scene, args: Any, rng: random.Random) -> "Theme": ...
