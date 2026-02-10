#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/generate-inverse.py

Traverse all src/character-*.svg and generate src/inverse-*.svg where the
*enabled* polygons are the exact opposite of the originals.

Definition used:
- "Enabled polygons" = actual <polygon ...> elements in the SVG DOM
- "Disabled polygons" = <polygon ...> tags that are inside XML comments <!-- ... -->

Inversion rule (per file):
- Every enabled <polygon> becomes commented-out
- Every commented-out <polygon> becomes enabled again (inserted back in place)

Notes:
- Only polygons are inverted; other elements (rect/background/defs/etc.) are preserved.
- Requires: lxml  (py -m pip install lxml)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple
from xml.sax.saxutils import escape

from lxml import etree  # py -m pip install lxml

SVG_NS = "http://www.w3.org/2000/svg"

POLYGON_TAG_RE = re.compile(
    r"<polygon\b[^>]*?(?:/>|>.*?</polygon\s*>)",
    re.IGNORECASE | re.DOTALL,
)
XML_DECL_RE = re.compile(r"^\s*(<\?xml[^?]*\?>)", re.DOTALL)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _attr_local(k: str) -> str:
    # "{ns}foo" -> "foo"
    return k.split("}", 1)[1] if k.startswith("{") and "}" in k else k


def _polygon_to_comment_text(poly: etree._Element) -> str:
    # Build a clean, namespace-free <polygon .../> tag string
    attrs = []
    for k, v in poly.attrib.items():
        kk = _attr_local(k)
        vv = escape(v, {'"': "&quot;"})
        attrs.append((kk, vv))

    # keep stable ordering (nice diffs)
    attrs.sort(key=lambda kv: kv[0])

    if attrs:
        attr_str = " " + " ".join(f'{k}="{v}"' for k, v in attrs)
    else:
        attr_str = ""

    return f"<polygon{attr_str}/>"


def _parse_polygons_from_comment_text(comment_text: str) -> List[etree._Element]:
    tags = POLYGON_TAG_RE.findall(comment_text or "")
    if not tags:
        return []

    # Wrap so the polygons get the SVG namespace
    wrapper = f'<svg xmlns="{SVG_NS}">' + "".join(tags) + "</svg>"
    try:
        tmp = etree.fromstring(wrapper.encode("utf-8"), parser=etree.XMLParser(recover=True))
    except Exception:
        return []

    out: List[etree._Element] = []
    for child in tmp:
        if isinstance(child.tag, str) and _local_name(child.tag).lower() == "polygon":
            out.append(child)
    return out


def invert_polygons(svg_text: str) -> str:
    parser = etree.XMLParser(remove_blank_text=False, recover=True, remove_comments=False)
    root = etree.fromstring(svg_text.encode("utf-8"), parser=parser)

    # Snapshot: enabled polygons and comment nodes with polygons
    enabled_polys: List[etree._Element] = root.xpath('.//*[local-name()="polygon"]')
    comments: List[etree._Element] = root.xpath("//comment()")

    comment_infos: List[Tuple[etree._Element, List[etree._Element], str]] = []
    for c in comments:
        txt = c.text or ""
        polys = _parse_polygons_from_comment_text(txt)
        if polys:
            # also compute leftover comment text (non-polygon content)
            leftover = POLYGON_TAG_RE.sub("", txt)
            comment_infos.append((c, polys, leftover))

    # 1) Disable currently enabled polygons by replacing each with a comment containing its tag
    for poly in enabled_polys:
        parent = poly.getparent()
        if parent is None:
            continue
        comment_text = _polygon_to_comment_text(poly)
        parent.replace(poly, etree.Comment(comment_text))

    # 2) Enable polygons that were inside comments (insert them back where the comment was)
    for comment_node, polys_to_enable, leftover in comment_infos:
        parent = comment_node.getparent()
        if parent is None:
            continue
        try:
            idx = parent.index(comment_node)
        except Exception:
            continue

        # Insert enabled polygons in the same spot
        for p in polys_to_enable:
            parent.insert(idx, p)
            idx += 1

        # Keep leftover comment content only if it contains non-whitespace
        if leftover.strip():
            comment_node.text = leftover
        else:
            parent.remove(comment_node)

    out = etree.tostring(root, encoding="unicode", pretty_print=False)
    # normalize newlines
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    return out


def main() -> None:
    root = Path(__file__).resolve().parent.parent  # tools/ -> project root
    src_dir = root / "src"
    in_files = sorted(src_dir.glob("character-*.svg"))

    if not in_files:
        print(f"No inputs found in {src_dir} matching character-*.svg")
        return

    errors = 0
    made = 0

    for f in in_files:
        try:
            original = f.read_text(encoding="utf-8")
            xml_decl = None
            m = XML_DECL_RE.match(original)
            if m:
                xml_decl = m.group(1)

            inverted = invert_polygons(original)

            if xml_decl:
                # ensure declaration stays first line
                inverted = xml_decl.strip() + "\n" + inverted.lstrip()

            key = f.stem[len("character-") :] if f.stem.startswith("character-") else f.stem
            out_path = src_dir / f"inverse-{key}.svg"
            out_path.write_text(inverted + ("\n" if not inverted.endswith("\n") else ""), encoding="utf-8")
            made += 1
            print(f"{f.name} -> {out_path.name}")
        except Exception as e:
            errors += 1
            print(f"ERROR {f.name}: {e}")

    print(f"\nDone. Wrote {made} inverse SVG(s).")
    if errors:
        print(f"With {errors} error(s).")


if __name__ == "__main__":
    main()
