#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/convert-filenames.py

Renames:
  src/character-{character}.svg  ->  src/character-u{codepoint}.svg

Examples:
  src/character-a.svg        -> src/character-u0061.svg
  src/character-🙂.svg       -> src/character-u1f642.svg
  src/character-🇳🇱.svg      -> src/character-u1f1f3_u1f1f1.svg  (multi-codepoint case)

By default this runs from the project root (one directory above /tools) and targets ./src.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote


RE_ALREADY = re.compile(r"^character-u[0-9a-fA-F]{4,}(?:_u[0-9a-fA-F]{4,})*\.svg$")
RE_TARGET = re.compile(r"^character-(.+)\.svg$", re.IGNORECASE)
RE_PCT = re.compile(r"%[0-9A-Fa-f]{2}")


def codepoints_slug(s: str) -> str:
    cps = [format(ord(ch), "x").rjust(4, "0") for ch in s]
    # Prefix each codepoint with 'u' so multi-codepoint filenames remain unambiguous
    return "u" + "_u".join(cps)


def unique_path(path: Path) -> Path:
    """Return a non-existing path by appending -{n} before suffix."""
    if not path.exists():
        return path
    base = path.with_suffix("")
    suffix = path.suffix
    for i in range(1, 10_000):
        candidate = Path(f"{base}-{i}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find free filename for {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rename src/character-{character}.svg to src/character-u{codepoint}.svg"
    )
    parser.add_argument(
        "--src-dir",
        default=None,
        help="Directory containing character-*.svg (default: <repo>/src)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be renamed without changing anything",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If target exists, overwrite it (DANGEROUS). Default: skip conflicts.",
    )
    parser.add_argument(
        "--suffix-on-conflict",
        action="store_true",
        help="If target exists, append -1, -2, ... to create a unique filename.",
    )
    parser.add_argument(
        "--url-decode",
        action="store_true",
        help="URL-decode stems that contain percent-escapes (e.g. %20).",
    )

    args = parser.parse_args()

    tools_dir = Path(__file__).resolve().parent
    repo_root = tools_dir.parent
    src_dir = Path(args.src_dir).resolve() if args.src_dir else (repo_root / "src")

    if not src_dir.exists() or not src_dir.is_dir():
        print(f"ERROR: src dir not found: {src_dir}", file=sys.stderr)
        return 2

    files = sorted(src_dir.glob("character-*.svg"))

    renamed = 0
    skipped = 0
    conflicts = 0
    errors = 0

    for f in files:
        name = f.name

        # Skip already-converted files
        if RE_ALREADY.match(name):
            skipped += 1
            continue

        m = RE_TARGET.match(name)
        if not m:
            skipped += 1
            continue

        stem = m.group(1)

        # Optional: URL decode for filenames like character-%20.svg
        if args.url_decode and RE_PCT.search(stem):
            stem = unquote(stem)

        if stem == "":
            print(f"SKIP (empty character): {f}", file=sys.stderr)
            skipped += 1
            continue

        slug = codepoints_slug(stem)
        target = f.with_name(f"character-{slug}.svg")

        # Handle conflicts
        if target.exists():
            conflicts += 1
            if args.overwrite:
                action_target = target
            elif args.suffix_on_conflict:
                action_target = unique_path(target)
            else:
                print(f"CONFLICT (exists, skipping): {f.name} -> {target.name}", file=sys.stderr)
                continue
        else:
            action_target = target

        if args.dry_run:
            print(f"DRY-RUN: {f.name} -> {action_target.name}")
            renamed += 1
            continue

        try:
            if action_target.exists() and args.overwrite:
                os.replace(str(f), str(action_target))  # overwrite atomically where possible
            else:
                f.rename(action_target)
            print(f"RENAMED: {f.name} -> {action_target.name}")
            renamed += 1
        except Exception as e:
            errors += 1
            print(f"ERROR: {f.name} -> {action_target.name}: {e}", file=sys.stderr)

    print(
        f"\nDone. renamed={renamed}, skipped={skipped}, conflicts={conflicts}, errors={errors}\n"
        f"Dir: {src_dir}"
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
