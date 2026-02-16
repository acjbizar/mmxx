from __future__ import annotations
from lxml import etree
from typing import Tuple
from ..constants import NUM_RE

def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag

def parse_viewbox(root: etree._Element) -> Tuple[float, float, float, float]:
    vb = root.get("viewBox") or root.get("viewbox") or "0 0 240 240"
    nums = [float(x) for x in NUM_RE.findall(vb)]
    if len(nums) == 4:
        return nums[0], nums[1], nums[2], nums[3]
    return 0.0, 0.0, 240.0, 240.0
