from __future__ import annotations
import re

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"

NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_CODEPOINT_RE = re.compile(r"^(?:u\+?|U\+?)([0-9a-fA-F]{4,8})$")
_HEX_RE = re.compile(r"^[0-9a-fA-F]{4,8}$")
_TRANSLATE_RE = re.compile(
    r"translate\(\s*(" + NUM_RE.pattern + r")\s*(?:[, ]\s*(" + NUM_RE.pattern + r"))?\s*\)",
    re.I
)
_URL_ID_RE = re.compile(r"url\(#([^)]+)\)")
