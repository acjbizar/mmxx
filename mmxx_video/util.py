from __future__ import annotations
from datetime import datetime
from pathlib import Path

def timestamped_if_exists(path: Path) -> Path:
    if not path.exists():
        return path
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return path.with_name(f"{path.stem}-{ts}{path.suffix}")
