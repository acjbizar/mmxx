#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thin wrapper for the refactored video generator.
Keeps the original entrypoint path: tools/generate-video.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmxx_video.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
