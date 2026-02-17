from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import List

def timestamped_if_exists(path: Path) -> Path:
    if not path.exists():
        return path
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return path.with_name(f"{path.stem}-{ts}{path.suffix}")


_FRAME_SPEC_TOKEN_RE = re.compile(r"[^,\s]+")


def parse_frame_spec(spec: str, total_frames: int) -> List[int]:
    """Parse a frame selection spec into sorted, unique, in-range frame indices.

    Supported examples:
      - "0" (first frame)
      - "last" / "first" / "mid"
      - "0,10,25" (commas or whitespace)
      - "0-120" (inclusive range)
      - "0-300:10" (inclusive range with step)
      - "-1" (last frame), "-2" (second last)

    Out-of-range values are ignored.
    """
    if total_frames <= 0:
        return []
    s = (spec or "").strip()
    if not s:
        return []

    out = set()
    for tok in _FRAME_SPEC_TOKEN_RE.findall(s):
        t = tok.strip()
        if not t:
            continue
        tl = t.lower()
        if tl in {"first", "start"}:
            out.add(0)
            continue
        if tl in {"last", "end"}:
            out.add(total_frames - 1)
            continue
        if tl in {"mid", "middle", "center", "centre"}:
            out.add(total_frames // 2)
            continue

        m = re.match(r"^(-?\d+)\s*-\s*(-?\d+)(?::(\d+))?$", t)
        if m:
            a = int(m.group(1))
            b = int(m.group(2))
            step = int(m.group(3) or "1")
            step = max(1, step)

            if a < 0:
                a = total_frames + a
            if b < 0:
                b = total_frames + b

            if a <= b:
                for i in range(a, b + 1, step):
                    if 0 <= i < total_frames:
                        out.add(i)
            else:
                for i in range(a, b - 1, -step):
                    if 0 <= i < total_frames:
                        out.add(i)
            continue

        try:
            i = int(t)
        except Exception:
            continue
        if i < 0:
            i = total_frames + i
        if 0 <= i < total_frames:
            out.add(i)

    return sorted(out)
