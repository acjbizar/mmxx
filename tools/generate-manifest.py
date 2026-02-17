#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/generate-manifest.py

Generate a universal, JSON font manifest from dist/fonts/{name}.{ext} files.

Outputs:
  - data/manifest.json
  - dist/fonts/{name}.manifest.json

The manifest is derived from:
  - cmap (Unicode coverage)
  - GSUB/GPOS feature tags (when present)
  - basic font metadata (name table, unitsPerEm, glyph count)
  - per-file hashes/sizes

Usage:
  python tools/generate-manifest.py --name mmxx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from fontTools.ttLib import TTFont


EXTS = ["woff2", "woff", "ttf", "otf"]  # look for these in dist/fonts
PREFERRED_PARSE_ORDER = ["ttf", "otf", "woff2", "woff"]  # for metadata extraction


@dataclass
class FontFileInfo:
    path: Path
    ext: str
    size: int
    sha256: str
    parse_error: Optional[str] = None


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def compress_to_ranges(codepoints: List[int]) -> List[List[int]]:
    """
    Compress sorted codepoints into inclusive [start, end] ranges.
    """
    if not codepoints:
        return []
    cps = sorted(set(codepoints))
    ranges: List[List[int]] = []
    start = prev = cps[0]
    for cp in cps[1:]:
        if cp == prev + 1:
            prev = cp
            continue
        ranges.append([start, prev])
        start = prev = cp
    ranges.append([start, prev])
    return ranges


def safe_get_name(tt: TTFont, name_id: int, platform_id: int = 3, enc_id: int = 1, lang_id: int = 0x0409) -> Optional[str]:
    """
    Try a few reasonable fallbacks to get a readable string from the 'name' table.
    """
    if "name" not in tt:
        return None

    name_tbl = tt["name"]

    # Try Windows Unicode English (US)
    n = name_tbl.getName(name_id, platform_id, enc_id, lang_id)
    if n:
        try:
            return str(n)
        except Exception:
            pass

    # Try any record with this nameID
    for rec in name_tbl.names:
        if rec.nameID == name_id:
            try:
                return rec.toUnicode()
            except Exception:
                try:
                    return str(rec)
                except Exception:
                    continue
    return None


def extract_feature_tags(tt: TTFont, table_tag: str) -> List[str]:
    """
    Extract Feature tags from GSUB/GPOS.
    """
    if table_tag not in tt:
        return []
    try:
        table = tt[table_tag].table
        fl = getattr(table, "FeatureList", None)
        if not fl or not getattr(fl, "FeatureRecord", None):
            return []
        tags = [fr.FeatureTag for fr in fl.FeatureRecord if getattr(fr, "FeatureTag", None)]
        return sorted(set(tags))
    except Exception:
        return []


def parse_font(path: Path) -> TTFont:
    # Keep parsing as "light" as possible (fast & consistent)
    return TTFont(
        str(path),
        recalcBBoxes=False,
        recalcTimestamp=False,
        lazy=True,
    )


def find_font_files(dist_fonts_dir: Path, name: str) -> List[FontFileInfo]:
    found: List[FontFileInfo] = []
    for ext in EXTS:
        p = dist_fonts_dir / f"{name}.{ext}"
        if p.exists() and p.is_file():
            size = p.stat().st_size
            digest = sha256_file(p)
            found.append(FontFileInfo(path=p, ext=ext, size=size, sha256=digest))
    return found


def choose_primary_parse_file(files: List[FontFileInfo]) -> Optional[FontFileInfo]:
    by_ext = {f.ext: f for f in files}
    for ext in PREFERRED_PARSE_ORDER:
        if ext in by_ext:
            return by_ext[ext]
    return files[0] if files else None


def build_manifest(name: str, files: List[FontFileInfo], repo_root: Path) -> Dict:
    # Try parsing each file; union cmap coverage; union feature tags.
    cmap_codepoints: Set[int] = set()
    gsub_tags: Set[str] = set()
    gpos_tags: Set[str] = set()

    primary_info = choose_primary_parse_file(files)

    meta: Dict[str, Optional[object]] = {
        "family": None,
        "subfamily": None,
        "fullName": None,
        "postScriptName": None,
        "unitsPerEm": None,
        "glyphCount": None,
    }

    # Parse primary first for metadata
    if primary_info:
        try:
            tt = parse_font(primary_info.path)

            # Metadata (best-effort)
            meta["family"] = safe_get_name(tt, 1)  # Font Family
            meta["subfamily"] = safe_get_name(tt, 2)  # Subfamily
            meta["fullName"] = safe_get_name(tt, 4)  # Full font name
            meta["postScriptName"] = safe_get_name(tt, 6)  # PostScript name

            try:
                meta["unitsPerEm"] = int(tt["head"].unitsPerEm) if "head" in tt else None
            except Exception:
                meta["unitsPerEm"] = None

            try:
                meta["glyphCount"] = len(tt.getGlyphOrder())
            except Exception:
                meta["glyphCount"] = None

            # cmap union from primary
            if "cmap" in tt:
                best = tt["cmap"].getBestCmap() or {}
                cmap_codepoints.update(best.keys())

            # features from primary
            gsub_tags.update(extract_feature_tags(tt, "GSUB"))
            gpos_tags.update(extract_feature_tags(tt, "GPOS"))

            tt.close()
        except Exception as e:
            primary_info.parse_error = f"{type(e).__name__}: {e}"

    # Parse the rest for coverage/features union (best-effort)
    for info in files:
        if primary_info and info.path == primary_info.path:
            continue
        try:
            tt = parse_font(info.path)

            if "cmap" in tt:
                best = tt["cmap"].getBestCmap() or {}
                cmap_codepoints.update(best.keys())

            gsub_tags.update(extract_feature_tags(tt, "GSUB"))
            gpos_tags.update(extract_feature_tags(tt, "GPOS"))

            tt.close()
        except Exception as e:
            info.parse_error = f"{type(e).__name__}: {e}"

    # Build file map for CDN-relative paths (dist/fonts/...)
    dist_fonts_dir = repo_root / "dist" / "fonts"
    file_entries = []
    for info in sorted(files, key=lambda x: EXTS.index(x.ext)):
        rel = info.path.relative_to(dist_fonts_dir) if info.path.is_relative_to(dist_fonts_dir) else info.path.name
        file_entries.append(
            {
                "ext": info.ext,
                "file": str(rel).replace("\\", "/"),
                "bytes": info.size,
                "sha256": info.sha256,
                **({"parseError": info.parse_error} if info.parse_error else {}),
            }
        )

    ranges = compress_to_ranges(sorted(cmap_codepoints))

    # Minimal but useful summary stats
    total_codepoints = len(set(cmap_codepoints))
    bmp_codepoints = sum(1 for cp in cmap_codepoints if 0x0000 <= cp <= 0xFFFF)
    astral_codepoints = total_codepoints - bmp_codepoints

    # Also provide a few convenience "well-known" blocks (optional)
    # This is intentionally tiny; the authoritative coverage is unicodeRanges.
    # You can expand this later if you want.
    common_blocks = {
        "basicLatin": _range_intersects(ranges, 0x0020, 0x007E),
        "latin1Supplement": _range_intersects(ranges, 0x00A0, 0x00FF),
        "latinExtendedA": _range_intersects(ranges, 0x0100, 0x017F),
        "latinExtendedB": _range_intersects(ranges, 0x0180, 0x024F),
    }

    manifest = {
        "manifestVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "name": name,
        "metadata": meta,
        "files": file_entries,
        "unicodeRanges": ranges,  # inclusive [start,end] codepoint ranges
        "counts": {
            "codepointsTotal": total_codepoints,
            "codepointsBMP": bmp_codepoints,
            "codepointsAstral": astral_codepoints,
        },
        "features": {
            "GSUB": sorted(gsub_tags),
            "GPOS": sorted(gpos_tags),
        },
        "hints": {
            "commonBlocks": common_blocks,
            "notes": [
                "unicodeRanges is derived from cmap.getBestCmap() across all parseable font files.",
                "GSUB/GPOS feature tags are extracted when those tables exist.",
                "Ligature contents are not enumerated here (by design); use shaping in your sample to demonstrate them.",
            ],
        },
    }

    return manifest


def _range_intersects(ranges: List[List[int]], lo: int, hi: int) -> bool:
    # ranges are inclusive [a,b]
    for a, b in ranges:
        if b < lo:
            continue
        if a > hi:
            return False  # because sorted
        return True
    return False


def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a JSON manifest for a font family/style name in dist/fonts.")
    ap.add_argument("--name", required=True, help="Base filename (without extension) in dist/fonts/, e.g. 'mmxx'.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]  # tools/.. -> repo root
    dist_fonts_dir = repo_root / "dist" / "fonts"
    data_dir = repo_root / "data"

    name = args.name.strip()
    if not name:
        raise SystemExit("ERROR: --name cannot be empty")

    files = find_font_files(dist_fonts_dir, name)
    if not files:
        # Still write a minimal manifest so downstream code can fail gracefully.
        minimal = {
            "manifestVersion": 1,
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "name": name,
            "error": f"No font files found at dist/fonts/{name}.{{{','.join(EXTS)}}}",
            "files": [],
            "unicodeRanges": [],
            "counts": {"codepointsTotal": 0, "codepointsBMP": 0, "codepointsAstral": 0},
            "features": {"GSUB": [], "GPOS": []},
        }
        write_json(data_dir / "manifest.json", minimal)
        write_json(dist_fonts_dir / f"{name}.manifest.json", minimal)
        print(minimal["error"])
        return 2

    manifest = build_manifest(name, files, repo_root)

    out_data = data_dir / "manifest.json"
    out_cdn = dist_fonts_dir / f"{name}.manifest.json"

    write_json(out_data, manifest)
    write_json(out_cdn, manifest)

    # Small console summary
    print(f"Wrote: {out_data}")
    print(f"Wrote: {out_cdn}")
    print(f"Files: {', '.join([f'{x.ext}' for x in files])}")
    print(f"Codepoints: {manifest['counts']['codepointsTotal']}")
    if any(f.parse_error for f in files):
        print("Note: Some files could not be parsed; see files[].parseError in the manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
