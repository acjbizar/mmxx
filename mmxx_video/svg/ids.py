from __future__ import annotations
from lxml import etree
from typing import Dict
from ..constants import _URL_ID_RE

def prefix_svg_ids(svg_fragment: etree._Element, prefix: str) -> None:
    id_map: Dict[str, str] = {}
    for el in svg_fragment.iter():
        if not isinstance(el.tag, str):
            continue
        old = el.get("id")
        if old:
            new = f"{prefix}{old}"
            id_map[old] = new
            el.set("id", new)

    if not id_map:
        return

    def rewrite_value(v: str) -> str:
        if not v:
            return v

        def repl(m):
            old_id = m.group(1)
            new_id = id_map.get(old_id, old_id)
            return f"url(#{new_id})"

        out = _URL_ID_RE.sub(repl, v)

        if out.startswith("#") and out[1:] in id_map:
            out = "#" + id_map[out[1:]]

        for old_id, new_id in id_map.items():
            out = out.replace(f"#{old_id}", f"#{new_id}")

        return out

    for el in svg_fragment.iter():
        if not isinstance(el.tag, str):
            continue
        for attr, val in list(el.attrib.items()):
            if isinstance(val, str):
                newv = rewrite_value(val)
                if newv != val:
                    el.set(attr, newv)
