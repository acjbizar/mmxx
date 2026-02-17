from __future__ import annotations

import base64
from pathlib import Path
from lxml import etree
from ..constants import SVG_NS
from ..scene import Scene
from ..themes.base import Theme
import dataclasses
import json
from typing import Any, Dict, Iterable, Optional, Sequence

_MISSING = object()

def _to_jsonable(x: Any) -> Any:
    """Recursively convert dataclasses/tuples/etc into JSON-safe primitives."""
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if dataclasses.is_dataclass(x):
        return _to_jsonable(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    # lxml elements etc should never be in state; fall back to string to avoid crashing
    return str(x)

def _pick(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    """Pick the first non-None attribute (or dict key) from a list of candidate names."""
    for n in names:
        if not n:
            continue
        if isinstance(obj, dict):
            v = obj.get(n, _MISSING)
        else:
            v = getattr(obj, n, _MISSING)
        if v is not _MISSING and v is not None:
            return v
    return default

def _alias(state: Dict[str, Any], primary: str, *alts: str, default: Any = None) -> None:
    """
    Ensure state[primary] exists. If missing/None, try alternates.
    Also mirrors primary -> alternates when alternates missing.
    """
    if state.get(primary) is None:
        v = _pick(state, [primary, *alts], default=default)
        state[primary] = v
    for a in alts:
        if state.get(a) is None and state.get(primary) is not None:
            state[a] = state[primary]

def _theme_state(theme_name: str, theme: Any) -> Dict[str, Any]:
    """
    Export a JSON-safe state dict for JS runtime.
    Uses aliases so internal theme attribute naming differences don't break export.
    """
    # Start from theme.__dict__ (works for normal classes and dataclasses)
    raw = getattr(theme, "__dict__", {}) or {}
    state: Dict[str, Any] = _to_jsonable(raw)

    # Make sure we always have a numeric-ish seed value available.
    _alias(state, "seed", "seedf", "static_seedf", "camo_seed", default=0.0)
    _alias(state, "seedf", "seed", "static_seedf", "camo_seed", default=state.get("seed", 0.0))

    # Common “poly parameter arrays” for hsv/diamond materials
    _alias(state, "poly_hsv", "polyHSV", "poly_params", default=[])

    # Minecraft (different refactors name these differently)
    _alias(state, "mc_pixels", "pixels", "tex_pixels", default=[])
    _alias(state, "mc_w", "w", "tex_w", default=0)
    _alias(state, "mc_h", "h", "tex_h", default=0)
    _alias(state, "mc_u", "u", "tex_u", default=[])
    _alias(state, "mc_v", "v", "tex_v", default=[])
    _alias(state, "mc_freq", "freq", default=[])
    _alias(state, "mc_phase", "phase", default=[])

    # Deidee
    _alias(state, "de_alpha", "alpha", default=0.5)
    _alias(state, "de_colors_per_poly", "colors_per_poly", "colors", default=[])
    _alias(state, "de_seg_dur", "seg_dur", "seg", default=[])
    _alias(state, "de_phase", "phase", default=[])

    # Static
    _alias(state, "static_seedf", "seedf", "seed", default=state.get("seed", 0.0))
    _alias(state, "static_seg", "seg", default=[])
    _alias(state, "static_phase", "phase", default=[])
    _alias(state, "static_color_prob", "color_prob", default=[])

    # Matrix
    _alias(state, "col_count", "columns", default=0)
    _alias(state, "col_drop", "drops", default=[])

    # Champagne
    _alias(state, "bubbles", "bubble_list", default=[])
    _alias(state, "ch_freq", "freq", default=[])
    _alias(state, "ch_phase", "phase", default=[])

    # Camo
    _alias(state, "camo_seed", "seed", "seedf", default=state.get("seed", 0.0))
    _alias(state, "camo_offx", "offx", default=[])
    _alias(state, "camo_offy", "offy", default=[])
    _alias(state, "camo_phase", "phase", default=[])
    _alias(state, "CAMO_PAL", "camo_pal", "palette", default=[])

    # Heart
    _alias(state, "heart_fit", "fit", default=None)
    _alias(state, "heart_hj", "hj", default=[])
    _alias(state, "heart_sj", "sj", default=[])
    _alias(state, "heart_tw_f", "tw_f", default=[])
    _alias(state, "heart_tw_p", "tw_p", default=[])
    _alias(state, "heart_fire_h", "fire_h", default=[])

    # Nuke
    _alias(state, "offx", "nuke_offx", default=state.get("offx", []))
    _alias(state, "offy", "nuke_offy", default=state.get("offy", []))
    _alias(state, "freq", "nuke_freq", default=state.get("freq", []))
    _alias(state, "phase", "nuke_phase", default=state.get("phase", []))
    # tint_rgb can be null; keep as-is

    # Fireworks: this is the big one — ensure we export a plain list of dicts.
    fw = _pick(theme, ["fireworks", "shots", "fw", "bursts"], default=None)
    if fw is None:
        fw = state.get("fireworks", None)
    if fw is None:
        fw = []
    fw = _to_jsonable(fw)
    state["fireworks"] = fw
    state.setdefault("shots", fw)  # alias for JS if it expects another name

    # Finally, tag the theme name for JS side (useful for debugging)
    state["__theme__"] = theme_name

    return state


def _round_floats(obj: Any, ndigits: int = 6) -> Any:
    """Recursively round floats to reduce embedded JS size."""
    if isinstance(obj, float):
        return float(round(obj, ndigits))
    if isinstance(obj, (list, tuple)):
        return [ _round_floats(x, ndigits) for x in obj ]
    if isinstance(obj, dict):
        return { k: _round_floats(v, ndigits) for k, v in obj.items() }
    return obj


def _jd(obj: Any) -> str:
    return json.dumps(_round_floats(obj), separators=(",", ":"), ensure_ascii=False)


def _pack_pulses(pulses_per_poly: Sequence[Sequence[Any]]) -> Dict[str, Any]:
    """
    Pack list-of-list Pulse objects into flat arrays for JS:
      - p_t0, p_half, p_amp, p_pow
      - p_start[i], p_count[i]
    """
    t0: List[float] = []
    half: List[float] = []
    amp: List[float] = []
    pow_: List[float] = []
    start: List[int] = []
    count: List[int] = []

    cur = 0
    for pulses in pulses_per_poly:
        start.append(cur)
        n = 0
        for p in pulses:
            # Pulse has slots: t0, half, amp, power
            t0.append(float(p.t0))
            half.append(float(p.half))
            amp.append(float(p.amp))
            pow_.append(float(getattr(p, "power", 1.0)))
            cur += 1
            n += 1
        count.append(n)

    return {
        "t0": t0,
        "half": half,
        "amp": amp,
        "pow": pow_,
        "start": start,
        "count": count,
    }


def export_svgjs(scene: Scene, theme: Theme, out_svg: Path, *, duration: float, fps_hint: int) -> None:
    """
    Export a self-contained SVG with embedded JS that animates polygon fills.

    Notes:
    - This keeps the SVG geometry (polygons) as-is and uses JS to update their `fill`.
    - For most themes, the JS re-implements the same procedural math as Python using
      per-polygon precomputed parameters embedded as JSON.
    - For the GIF theme, we embed precomputed per-frame per-polygon RGB samples to
      avoid embedding full raster frames.
    """
    # Ensure stable polygon ordering
    for i, poly in enumerate(scene.polys):
        poly.set("data-i", str(i))

    # Set initial frame to match Python output at t=0 (nice first paint).
    theme.apply_frame(scene, 0.0)

    js = _build_js(scene, theme, duration=float(duration), fps_hint=int(fps_hint))

    script = etree.Element(f"{{{SVG_NS}}}script")
    script.set("type", "application/ecmascript")
    script.text = etree.CDATA(js)

    scene.doc.append(script)

    out_svg.parent.mkdir(parents=True, exist_ok=True)
    out_svg.write_bytes(etree.tostring(scene.doc, encoding="utf-8", xml_declaration=True))


def _build_js(scene: Scene, theme: Theme, *, duration: float, fps_hint: int) -> str:
    """Build the self-contained JS runtime used by the SVG+JS export.

    This function must be tolerant of small differences between Theme implementations.
    """
    from dataclasses import asdict, is_dataclass
    from ..themes.configs import get_theme_config

    n = len(scene.polys)

    # Theme name (what JS switches on)
    theme_name = getattr(theme, "name", None) or getattr(theme, "kind", None) or theme.__class__.__name__.lower()

    poly_nx = scene.poly_nx
    poly_ny = scene.poly_ny
    glyph_nx = scene.glyph_nx
    glyph_ny = scene.glyph_ny

    # Optional: compact pulse schedules (if present)
    pulses = None
    if hasattr(scene, "pulses_compact_for_js"):
        try:
            pulses = scene.pulses_compact_for_js()
        except Exception:
            pulses = None

    def _get(obj: Any, name: str, default: Any = None) -> Any:
        return getattr(obj, name, default)

    def _num(obj: Any, *names: str, default: float = 0.0) -> float:
        for nm in names:
            v = _get(obj, nm, None)
            if v is None:
                continue
            try:
                return float(v)
            except Exception:
                continue
        return float(default)

    def _cfg(theme_kind: str) -> dict:
        # Prefer an explicit cfg on the theme, else fall back to the canonical config.
        cfg_obj = _get(theme, "cfg", None)
        if cfg_obj is None:
            try:
                cfg_obj = get_theme_config(theme_kind)
            except Exception:
                cfg_obj = None

        if cfg_obj is None:
            return {}

        if is_dataclass(cfg_obj):
            return asdict(cfg_obj)
        if isinstance(cfg_obj, dict):
            return dict(cfg_obj)
        return dict(getattr(cfg_obj, "__dict__", {}))

    state: Dict[str, Any] = {
        "__theme__": theme_name,
        "n": n,
        "duration": float(duration),
        "fps_hint": int(fps_hint),
        "poly_nx": poly_nx,
        "poly_ny": poly_ny,
        "glyph_nx": glyph_nx,
        "glyph_ny": glyph_ny,
        "base_rgbs": scene.base_rgbs,
        "override_hsv": scene.override_hsv,
        "bgcolor": scene.bgcolor,
    }
    if pulses is not None:
        state["pulses"] = pulses

    # ---- theme-specific state (mirrors what the JS runtime expects) ----
    HSV_MATERIAL_THEMES = {
        "diamond",
        "silver", "gold", "bronze",
        "ruby", "jade", "sapphire", "emerald",
        "rainbow", "fire", "ice",
        "valentine", "valentines",
        "snow",
    }

    if theme_name in HSV_MATERIAL_THEMES:
        ph = _get(theme, "poly_hsv", None)
        if ph is not None:
            state["poly_hsv"] = [asdict(h) for h in ph]
        d = _cfg(theme_name)
        # The runtime only checks for 'diamond' special-casing.
        d["kind"] = "diamond" if theme_name == "diamond" else "hsv"
        state["cfg"] = d

    elif theme_name == "minecraft":
        # Theme doesn't carry cfg; use canonical config
        state["cfg"] = _cfg("minecraft")
        state["mc_pixels"] = [list(px) for px in (_get(theme, "mc_pixels") or [])]
        state["mc_w"] = int(_get(theme, "mc_w", 16))
        state["mc_h"] = int(_get(theme, "mc_h", 16))
        state["mc_u_amp"] = list(_get(theme, "mc_u_amp", [0.0] * n))
        state["mc_u_flow"] = list(_get(theme, "mc_u_flow", [0.0] * n))
        state["mc_v_amp"] = list(_get(theme, "mc_v_amp", [0.0] * n))
        state["mc_v_flow"] = list(_get(theme, "mc_v_flow", [0.0] * n))

    elif theme_name == "deidee":
        # DeideeTheme stores: colors_per_poly, seg_dur, phase, alpha
        state["colors"] = [list(map(list, poly)) for poly in (_get(theme, "colors_per_poly") or [])]
        state["seg_dur"] = float(_get(theme, "seg_dur", 1.0))
        state["phase"] = float(_get(theme, "phase", 0.0))
        state["alpha"] = float(_get(theme, "alpha", 1.0))

    elif theme_name == "static":
        state["seedf"] = _num(theme, "seedf", "seed", default=0.0)
        state["s_freq"] = list(_get(theme, "s_freq", [0.0] * n))
        state["s_phase"] = list(_get(theme, "s_phase", [0.0] * n))
        state["s_u"] = list(_get(theme, "s_u", [0.0] * n))
        state["s_v"] = list(_get(theme, "s_v", [0.0] * n))
        state["static_yfade"] = float(_get(theme, "static_yfade", 0.0))

    elif theme_name == "matrix":
        state["col_count"] = int(_get(theme, "col_count", 0))
        drops = _get(theme, "drops", None) or []
        state["drops"] = [asdict(d) for d in drops]

    elif theme_name == "champagne":
        state["cfg"] = _cfg("champagne")
        bubbles = _get(theme, "bubbles", None) or []
        state["bubbles"] = [asdict(b) for b in bubbles]
        state["ch_freq"] = list(_get(theme, "ch_freq", [0.0] * n))
        state["ch_phase"] = list(_get(theme, "ch_phase", [0.0] * n))

    elif theme_name == "camo":
        state["cfg"] = _cfg("camo")
        state["camo_seed"] = _num(theme, "seedf", "seed", default=0.0)
        state["offx"] = list(_get(theme, "offx", [0.0] * n))
        state["offy"] = list(_get(theme, "offy", [0.0] * n))

    elif theme_name == "fireworks":
        state["cfg"] = _cfg("fireworks")
        fws = _get(theme, "fireworks", None) or []
        state["fireworks"] = [asdict(fw) for fw in fws]

    elif theme_name == "nuke":
        state["seed"] = _num(theme, "seedf", "seed", default=0.0)
        state["offx"] = float(_get(theme, "offx", 0.0))
        state["offy"] = float(_get(theme, "offy", 0.0))
        state["plume_dens"] = float(_get(theme, "plume_dens", 0.0))
        state["halo_phase"] = float(_get(theme, "halo_phase", 0.0))
        state["uv_phase"] = float(_get(theme, "uv_phase", 0.0))
        state["bt_phi"] = float(_get(theme, "bt_phi", 0.0))

    elif theme_name == "heart":
        fit = _get(theme, "fit", None)
        if fit is not None:
            state["fit"] = asdict(fit) if is_dataclass(fit) else dict(fit)
        state["u"] = list(_get(theme, "u", [0.0] * n))
        state["v"] = list(_get(theme, "v", [0.0] * n))
        state["phase"] = list(_get(theme, "phase", [0.0] * n))
        state["freq"] = list(_get(theme, "freq", [0.0] * n))
        # Optional "extra spice" fields (runtime has defaults if missing)
        for k in ("hj", "sj", "tw_f", "tw_p", "fire_h", "beat_amp"):
            v = _get(theme, k, None)
            if v is not None:
                state[k] = v

    elif theme_name == "gif":
        # GifTheme provides pre-sampled per-poly colors for each frame.
        samples_b64 = _get(theme, "samples_b64", None)
        if samples_b64 is not None:
            state["samples_b64"] = samples_b64
        for k in ("frames", "fw", "fh", "fps", "o_w", "o_h", "frames_rgb", "frame_sec"):
            v = _get(theme, k, None)
            if v is not None:
                state[k] = v

    # NOTE: if a theme doesn't need extra state, 'state' stays minimal and the
    # JS runtime will still render something sensible.

    # JS prelude: math helpers + runtime
    js = f"""
(() => {{
  'use strict';

  const state = {_jd(state)};
  const THEME = String(state.__theme__ || 'classic');
  const N = state.n|0;

  // Polygons are indexed by data-i (exporter writes it). If missing, fallback to all polygons.
  const polys = [];
  for (let i=0;i<N;i++) {{
    const el = document.querySelector(`polygon[data-i="${{i}}"]`);
    if (el) polys.push(el);
  }}
  if (polys.length !== N) {{
    polys.length = 0;
    const all = Array.from(document.querySelectorAll('polygon'));
    for (let i=0;i<Math.min(all.length, N);i++) polys.push(all[i]);
  }}

  // Base RGBs: either embedded, or computed from computed style on load.
  let baseRGB = state.base_rgbs;
  if (!baseRGB || baseRGB.length !== N*3) {{
    baseRGB = new Array(N*3);
    const parse = (s) => {{
      if (!s) return [0,0,0];
      if (s.startsWith('rgb')) {{
        const m = s.match(/\\d+/g);
        if (!m || m.length < 3) return [0,0,0];
        return [m[0]|0, m[1]|0, m[2]|0];
      }}
      if (s[0] === '#') {{
        const h = s.slice(1);
        if (h.length === 3) {{
          return [
            parseInt(h[0]+h[0],16),
            parseInt(h[1]+h[1],16),
            parseInt(h[2]+h[2],16),
          ];
        }}
        if (h.length === 6) {{
          return [
            parseInt(h.slice(0,2),16),
            parseInt(h.slice(2,4),16),
            parseInt(h.slice(4,6),16),
          ];
        }}
      }}
      return [0,0,0];
    }};
    for (let i=0;i<N;i++) {{
      const s = getComputedStyle(polys[i]).fill;
      const [r,g,b] = parse(s);
      baseRGB[i*3] = r; baseRGB[i*3+1] = g; baseRGB[i*3+2] = b;
    }}
  }}

  // Coords
  const polyNx = state.poly_nx || [];
  const polyNy = state.poly_ny || [];
  const glyphNx = state.glyph_nx || [];
  const glyphNy = state.glyph_ny || [];

  // Timing
  const DURATION = Number(state.duration || 12.0);
  const FPS_HINT = Number(state.fps_hint || 30);

  // ---- Math helpers ----
  const TAU = Math.PI * 2.0;
  const clamp01 = (x) => x < 0 ? 0 : (x > 1 ? 1 : x);
  const lerp = (a,b,t) => a + (b-a)*t;
  const smoothstep = (a,b,x) => {{
    const t = clamp01((x-a)/(b-a));
    return t*t*(3-2*t);
  }};
  const fract = (x) => x - Math.floor(x);
  const hash11 = (x) => fract(Math.sin(x*127.1 + 311.7)*43758.5453);
  const hash21 = (x,y) => fract(Math.sin(x*127.1 + y*311.7 + 74.7)*43758.5453);

  function hsv2rgb(h,s,v) {{
    h = ((h % 1) + 1) % 1;
    s = clamp01(s); v = clamp01(v);
    const i = Math.floor(h*6);
    const f = h*6 - i;
    const p = v*(1-s);
    const q = v*(1-f*s);
    const t = v*(1-(1-f)*s);
    let r,g,b;
    switch (i % 6) {{
      case 0: r=v; g=t; b=p; break;
      case 1: r=q; g=v; b=p; break;
      case 2: r=p; g=v; b=t; break;
      case 3: r=p; g=q; b=v; break;
      case 4: r=t; g=p; b=v; break;
      case 5: r=v; g=p; b=q; break;
    }}
    return [r,g,b];
  }}

  function rgbToHex(rgb) {{
    const r = Math.max(0, Math.min(255, rgb[0]|0)).toString(16).padStart(2,'0');
    const g = Math.max(0, Math.min(255, rgb[1]|0)).toString(16).padStart(2,'0');
    const b = Math.max(0, Math.min(255, rgb[2]|0)).toString(16).padStart(2,'0');
    return `#${{r}}${{g}}${{b}}`;
  }}

  function mixToWhite(rgb, a) {{
    const t = clamp01(a);
    return [
      (rgb[0] + (255 - rgb[0]) * t)|0,
      (rgb[1] + (255 - rgb[1]) * t)|0,
      (rgb[2] + (255 - rgb[2]) * t)|0
    ];
  }}

  // Per-poly whiteness pulses (deterministic, plus optional embedded pulses)
  const pulses = state.pulses || null;
  function whitenessAt(t,i) {{
    if (pulses && pulses[i] && pulses[i].length) {{
      let acc = 0.0;
      for (const p of pulses[i]) {{
        const t0 = p[0], half = p[1], amp = p[2], pow = p[3];
        const x = Math.abs(t - t0) / half;
        if (x < 1.0) acc += amp * Math.pow(1.0 - x, pow);
      }}
      return clamp01(acc);
    }}
    // fallback "good enough" pulse feel
    const s = 0.35 + 0.65*hash11(i*19.17);
    const ph = hash11(i*3.7)*TAU;
    const w = 0.5 + 0.5*Math.sin(TAU*t*s + ph);
    return Math.pow(clamp01(w), 1.6);
  }}

  // ---- Theme renderers ----

  function renderHSV(t) {{
    const cfg = state.cfg || {{}};
    const kind = cfg.kind || 'hsv';
    const ph = state.poly_hsv || [];
    if (!ph.length) return false;

    // Ambient / shimmer
    const ambBase = cfg.amb_base ?? 0.06;
    const ambAmp  = cfg.amb_amp  ?? 0.05;
    const ambFreq = cfg.amb_freq ?? 0.25;

    const glFreq = cfg.gl_pulse_freq ?? 0.33;
    const glWid  = cfg.gl_pulse_w    ?? 0.20;
    const glAmp  = cfg.gl_pulse_amp  ?? 0.70;

    // Diamond spec extras
    const specStrength = cfg.spec_strength ?? 1.0;
    const disp = cfg.dispersion ?? 0.25;

    const amb = ambBase + ambAmp*(0.5 + 0.5*Math.sin(TAU*ambFreq*t));

    for (let i=0;i<N;i++) {{
      const h = ph[i]?.h ?? 0.0;
      const s = ph[i]?.s ?? 0.0;
      const v = ph[i]?.v ?? 0.0;
      const sm = ph[i]?.sat_mul ?? 1.0;
      const vm = ph[i]?.v_mul ?? 1.0;
      const fq = ph[i]?.freq ?? 0.0;
      const pp = ph[i]?.phase ?? 0.0;

      // Base HSV with gentle per-poly movement
      const wob = 0.5 + 0.5*Math.sin(TAU*(fq*t + pp));
      const hh = h + 0.03*(wob - 0.5);
      const ss = clamp01(s * sm);
      const vv = clamp01(v * vm);

      let rgb = hsv2rgb(hh, ss, vv);
      rgb = [ (rgb[0]*255)|0, (rgb[1]*255)|0, (rgb[2]*255)|0 ];

      // Ambient
      rgb = [ (rgb[0]*(1-amb) + 255*amb)|0,
              (rgb[1]*(1-amb) + 255*amb)|0,
              (rgb[2]*(1-amb) + 255*amb)|0 ];

      // Glint
      const nx = polyNx[i] ?? 0.5;
      const ny = polyNy[i] ?? 0.5;
      const gl = glAmp * Math.exp(-((fract(nx + glFreq*t) - 0.5)**2)/(2*glWid*glWid));
      rgb = mixToWhite(rgb, gl);

      if (kind === 'diamond') {{
        // Cheap "dispersion" tint: split channels slightly by a sin curve
        const tt = Math.sin(TAU*(t*0.4 + nx*0.8 + ny*0.6));
        const dr = disp*tt, db = -disp*tt;
        rgb = [
          (rgb[0] * (1.0 + dr*specStrength))|0,
          rgb[1]|0,
          (rgb[2] * (1.0 + db*specStrength))|0
        ];
      }}

      polys[i].setAttribute('fill', rgbToHex(rgb));
    }}
    return true;
  }}

  function renderMinecraft(t) {{
    const px = state.mc_pixels;
    if (!px || !px.length) return false;

    const cfg = state.cfg || {{}};
    const ambBase = cfg.amb_base ?? 0.06;
    const ambAmp  = cfg.amb_amp  ?? 0.05;
    const ambFreq = cfg.amb_freq ?? 0.25;
    const amb = ambBase + ambAmp*(0.5 + 0.5*Math.sin(TAU*ambFreq*t));

    const W = state.mc_w|0, H = state.mc_h|0;
    const uA = state.mc_u_amp || [];
    const uF = state.mc_u_flow || [];
    const vA = state.mc_v_amp || [];
    const vF = state.mc_v_flow || [];

    for (let i=0;i<N;i++) {{
      const nx = polyNx[i] ?? 0.5;
      const ny = polyNy[i] ?? 0.5;

      const du = (uA[i] ?? 0.0) * Math.sin(TAU*((uF[i] ?? 0.0)*t + ny));
      const dv = (vA[i] ?? 0.0) * Math.sin(TAU*((vF[i] ?? 0.0)*t + nx));

      const u = clamp01(nx + du);
      const v = clamp01(ny + dv);

      const x = Math.max(0, Math.min(W-1, (u*(W-1))|0));
      const y = Math.max(0, Math.min(H-1, (v*(H-1))|0));
      const idx = y*W + x;

      const c = px[idx] || [0,0,0];
      let rgb = [c[0]|0, c[1]|0, c[2]|0];

      // Ambient lift
      rgb = [ (rgb[0]*(1-amb) + 255*amb)|0,
              (rgb[1]*(1-amb) + 255*amb)|0,
              (rgb[2]*(1-amb) + 255*amb)|0 ];

      // Add pulse-to-white
      const a = whitenessAt(t, i) * (cfg.gl_pulse_amp ?? 0.6);
      rgb = mixToWhite(rgb, a);

      polys[i].setAttribute('fill', rgbToHex(rgb));
    }}
    return true;
  }}

  function renderDeidee(t) {{
    const cols = state.colors;
    if (!cols || !cols.length) return false;
    const segDur = Number(state.seg_dur || 1.0);
    const phase = Number(state.phase || 0.0);
    const alpha = Number(state.alpha ?? 0.55);

    const tt = t + phase;
    const k0 = Math.floor(tt / segDur);
    const k1 = k0 + 1;
    const mix = (tt - k0*segDur) / segDur;

    for (let i=0;i<N;i++) {{
      const pal = cols[i] || [[0,0,0]];
      const c0 = pal[k0 % pal.length];
      const c1 = pal[k1 % pal.length];
      const rgb = [
        lerp(c0[0], c1[0], mix)|0,
        lerp(c0[1], c1[1], mix)|0,
        lerp(c0[2], c1[2], mix)|0,
      ];
      polys[i].setAttribute('fill', rgbToHex(rgb));
      polys[i].setAttribute('fill-opacity', String(alpha));
    }}
    return true;
  }}

  function renderStatic(t) {{
    const seed = Number(state.seedf || 0.0);
    const sFreq = state.s_freq || [];
    const sPhase = state.s_phase || [];
    const sU = state.s_u || [];
    const sV = state.s_v || [];
    const yfade = Number(state.static_yfade || 0.0);

    for (let i=0;i<N;i++) {{
      const nx = polyNx[i] ?? 0.5;
      const ny = polyNy[i] ?? 0.5;

      const f = sFreq[i] ?? 0.0;
      const p = sPhase[i] ?? 0.0;
      const u = sU[i] ?? 0.0;
      const v = sV[i] ?? 0.0;

      const n0 = hash21(nx*37.0 + seed + u, ny*91.0 + seed + v);
      const n1 = 0.5 + 0.5*Math.sin(TAU*(f*t + p));
      const g = clamp01(0.5*n0 + 0.5*n1);

      const fade = (yfade > 0.0) ? (1.0 - yfade*ny) : 1.0;

      const c = (255 * g * fade)|0;
      polys[i].setAttribute('fill', rgbToHex([c,c,c]));
    }}
    return true;
  }}

  function renderMatrix(t) {{
    const drops = state.drops;
    const colCount = state.col_count|0;
    if (!drops || !drops.length || colCount <= 0) return false;

    // A simple green palette
    const green0 = [0, 18, 0];
    const green1 = [70, 255, 70];

    // Column width in normalized glyph-space
    for (let i=0;i<N;i++) {{
      const gx = glyphNx[i] ?? 0.5;
      const gy = glyphNy[i] ?? 0.5;
      const col = Math.max(0, Math.min(colCount-1, Math.floor(gx * colCount)));

      const d = drops[col % drops.length];
      const speed = d.speed ?? 0.4;
      const phase = d.phase ?? 0.0;
      const head = d.head ?? 0.18;
      const tail = d.tail ?? 0.36;

      const y = fract(1.0 - (gy + speed*t + phase));
      const inHead = smoothstep(head, 0.0, y);
      const inTail = smoothstep(tail, head, y);

      const a = clamp01(inHead + inTail*0.8);
      const rgb = [
        lerp(green0[0], green1[0], a)|0,
        lerp(green0[1], green1[1], a)|0,
        lerp(green0[2], green1[2], a)|0,
      ];
      polys[i].setAttribute('fill', rgbToHex(rgb));
    }}
    return true;
  }}

  function renderChampagne(t) {{
    const bubbles = state.bubbles;
    if (!bubbles || !bubbles.length) return false;

    const cfg = state.cfg || {{}};
    const ambBase = cfg.amb_base ?? 0.06;
    const ambAmp  = cfg.amb_amp  ?? 0.05;
    const ambFreq = cfg.amb_freq ?? 0.25;
    const amb = ambBase + ambAmp*(0.5 + 0.5*Math.sin(TAU*ambFreq*t));

    const chF = state.ch_freq || [];
    const chP = state.ch_phase || [];

    for (let i=0;i<N;i++) {{
      const nx = polyNx[i] ?? 0.5;
      const ny = polyNy[i] ?? 0.5;

      // base from original
      let rgb = [baseRGB[i*3], baseRGB[i*3+1], baseRGB[i*3+2]];

      // champagne shimmer: slight warm pulse
      const sf = chF[i] ?? 0.0;
      const sp = chP[i] ?? 0.0;
      const sh = 0.5 + 0.5*Math.sin(TAU*(sf*t + sp));
      rgb = [
        (rgb[0] + 22*sh)|0,
        (rgb[1] + 18*sh)|0,
        (rgb[2] +  8*sh)|0,
      ];

      // bubble highlights
      let bub = 0.0;
      for (let k=0;k<bubbles.length;k++) {{
        const b = bubbles[k];
        const x = b.x ?? 0.5;
        const y0 = b.y0 ?? 1.0;
        const r = b.r ?? 0.06;
        const spd = b.speed ?? 0.25;
        const wob = b.wob ?? 0.02;
        const wf  = b.wob_f ?? 0.3;
        const wp  = b.wob_phase ?? 0.0;
        const a   = b.alpha ?? 0.6;

        const yy = fract(y0 - spd*t);
        const xx = x + wob*Math.sin(TAU*(wf*t + wp));
        const dx = nx - xx;
        const dy = ny - yy;
        const d2 = dx*dx + dy*dy;
        const w = Math.exp(-d2/(2*r*r)) * a;
        bub += w;
      }}

      // ambient lift + bubble to white
      rgb = [ (rgb[0]*(1-amb) + 255*amb)|0,
              (rgb[1]*(1-amb) + 255*amb)|0,
              (rgb[2]*(1-amb) + 255*amb)|0 ];
      rgb = mixToWhite(rgb, bub);

      polys[i].setAttribute('fill', rgbToHex(rgb));
    }}
    return true;
  }}

  function renderCamo(t) {{
    const cfg = state.cfg || {{}};
    const seed = Number(state.camo_seed || 0.0);
    const offx = state.offx || [];
    const offy = state.offy || [];

    const pal = [
      [ 70,  73,  56],
      [ 92,  97,  76],
      [114, 112,  86],
      [ 56,  58,  44],
      [133, 131, 101],
    ];

    for (let i=0;i<N;i++) {{
      const nx = (polyNx[i] ?? 0.5) + (offx[i] ?? 0.0);
      const ny = (polyNy[i] ?? 0.5) + (offy[i] ?? 0.0);

      // Worley-ish camo blobs
      const g = hash21(nx*6.0 + seed, ny*6.0 + seed);
      const idx = Math.max(0, Math.min(pal.length-1, Math.floor(g*pal.length)));
      const rgb0 = pal[idx];

      // pulse-to-white a bit
      const a = 0.35*whitenessAt(t,i);
      const rgb = mixToWhite(rgb0, a);
      polys[i].setAttribute('fill', rgbToHex(rgb));
    }}
    return true;
  }}

  function renderFireworks(t) {{
    const fws = state.fireworks;
    if (!fws || !fws.length) return false;

    const cfg = state.cfg || {{}};
    const ambBase = cfg.amb_base ?? 0.06;
    const ambAmp  = cfg.amb_amp  ?? 0.05;
    const ambFreq = cfg.amb_freq ?? 0.25;
    const amb = ambBase + ambAmp*(0.5 + 0.5*Math.sin(TAU*ambFreq*t));

    for (let i=0;i<N;i++) {{
      const nx = polyNx[i] ?? 0.5;
      const ny = polyNy[i] ?? 0.5;

      let rgb = [baseRGB[i*3], baseRGB[i*3+1], baseRGB[i*3+2]];
      rgb = [ (rgb[0]*(1-amb) + 255*amb)|0,
              (rgb[1]*(1-amb) + 255*amb)|0,
              (rgb[2]*(1-amb) + 255*amb)|0 ];

      let w = 0.0;

      for (const fw of fws) {{
        const x = fw.x ?? 0.5;
        const yb = fw.yb ?? 0.92;
        const tl = fw.t_launch ?? 0.0;
        const tb = fw.t_burst ?? 1.2;
        const vel = fw.vel ?? 0.6;

        const ringW = fw.ring_w ?? 0.04;
        const spread = fw.ring_spread ?? 0.35;
        const decay = fw.decay ?? 1.6;

        const spokeN = fw.spoke_n ?? 12;
        const spokeW = fw.spoke_w ?? 0.03;
        const spokeJ = fw.spoke_jit ?? 0.2;

        // rocket rise
        const tr = clamp01((t - tl) / Math.max(1e-6, (tb - tl)));
        const ry = yb - vel*tr;
        const dx0 = nx - x, dy0 = ny - ry;
        const d0 = Math.sqrt(dx0*dx0 + dy0*dy0);
        const rocket = Math.exp(-(d0*d0)/(2*0.012*0.012)) * (1.0 - tr);
        w += rocket * 0.9;

        // burst ring
        if (t >= tb) {{
          const dt = t - tb;
          const r = spread*dt;
          const dd = Math.abs(d0 - r);
          const ring = Math.exp(-(dd*dd)/(2*ringW*ringW)) * Math.exp(-decay*dt);
          w += ring * 1.1;

          // spokes
          const ang = Math.atan2(dy0, dx0);
          const spokeIdx = Math.floor(((ang + Math.PI) / (TAU)) * spokeN);
          const spokeAng = (spokeIdx / spokeN) * TAU - Math.PI + spokeJ*(hash11(spokeIdx*13.7 + tb*9.1)-0.5);
          const da = Math.abs(ang - spokeAng);
          const spoke = Math.exp(-(da*da)/(2*spokeW*spokeW)) * ring;
          w += spoke * 0.9;
        }}
      }}

      rgb = mixToWhite(rgb, w);
      polys[i].setAttribute('fill', rgbToHex(rgb));
    }}
    return true;
  }}

  function renderNuke(t) {{
    // Procedural mushroom cloud approximation
    const seed = Number(state.seed || 0.0);
    const offx = Number(state.offx || 0.0);
    const offy = Number(state.offy || 0.0);
    const dens = Number(state.plume_dens || 1.0);
    const haloP = Number(state.halo_phase || 0.0);
    const uvP = Number(state.uv_phase || 0.0);
    const btP = Number(state.bt_phi || 0.0);

    const cx = 0.5 + 0.08*Math.sin(offx + seed*0.01);
    const baseY = 0.86 + 0.02*Math.sin(offy + seed*0.02);

    for (let i=0;i<N;i++) {{
      const x = polyNx[i] ?? 0.5;
      const y = polyNy[i] ?? 0.5;

      // normalized to mushroom coordinates
      const dx = (x - cx);
      const dy = (baseY - y);

      // timeline
      const tt = t / Math.max(1e-6, DURATION);
      const rise = smoothstep(0.02, 0.35, tt);
      const bloom = smoothstep(0.15, 0.55, tt);
      const fade = 1.0 - smoothstep(0.75, 0.98, tt);

      // stem + cap fields
      const stemW = 0.06 + 0.03*bloom;
      const stem = Math.exp(-(dx*dx)/(2*stemW*stemW)) * smoothstep(0.0, 0.55, dy);

      const capR = 0.18 + 0.22*bloom;
      const capY = 0.32 + 0.18*rise;
      const cap = Math.exp(-((dx*dx + (dy-capY)*(dy-capY)))/(2*capR*capR));

      // turbulence
      const turb = 0.7 + 0.3*Math.sin(TAU*(dx*3.0 + dy*2.0 + uvP + t*0.25));
      const f = dens*(0.55*stem + 0.9*cap) * turb * fade;

      // heat tint
      const hot = smoothstep(0.15, 0.85, cap) * bloom;
      const core = smoothstep(0.65, 1.0, cap);

      let rgb = [baseRGB[i*3], baseRGB[i*3+1], baseRGB[i*3+2]];

      // smoke to white/gray
      const smoke = clamp01(f);
      rgb = mixToWhite(rgb, smoke);

      // warm core glow
      const warm = clamp01(0.85*hot + 1.1*core*fade);
      rgb = [
        (rgb[0] + 120*warm)|0,
        (rgb[1] +  60*warm)|0,
        (rgb[2] +  10*warm)|0,
      ];

      // halo ring
      const haloR = 0.26 + 0.16*bloom;
      const d = Math.sqrt(dx*dx + (dy-capY)*(dy-capY));
      const halo = Math.exp(-((d-haloR)*(d-haloR))/(2*(0.02+0.03*bloom)*(0.02+0.03*bloom))) * fade;
      rgb = mixToWhite(rgb, 0.9*halo);

      // blast flash early
      const flash = smoothstep(0.02, 0.08, tt) * (1.0 - smoothstep(0.10, 0.18, tt));
      if (flash > 0.0) {{
        rgb = mixToWhite(rgb, flash*1.2);
      }}

      // slight blue/orange shimmer
      const sh = 0.5 + 0.5*Math.sin(TAU*(t*0.35 + btP + dx*1.7));
      rgb = [
        (rgb[0] + 12*sh)|0,
        rgb[1]|0,
        (rgb[2] + 18*(1.0-sh))|0
      ];

      polys[i].setAttribute('fill', rgbToHex(rgb));
    }}
    return true;
  }}

  function renderHeart(t) {{
    const fit = state.fit;
    if (!fit) return false;

    const uArr = state.u || [];
    const vArr = state.v || [];
    const phArr = state.phase || [];
    const frArr = state.freq || [];

    const hj = Number(state.hj || 0.0);
    const sj = Number(state.sj || 0.0);

    const twF = state.tw_f || null;
    const twP = state.tw_p || null;
    const fireH = state.fire_h || null;
    const beatA = state.beat_amp || null;

    // Heart field in fitted coords
    function heartField(u,v) {{
      // classic implicit heart curve
      const x = u*1.2;
      const y = v*1.1;
      const a = x*x + y*y - 1;
      return a*a*a - x*x*y*y*y;
    }}

    for (let i=0;i<N;i++) {{
      const nx = polyNx[i] ?? 0.5;
      const ny = polyNy[i] ?? 0.5;

      // map to heart local coords (inverse of fit)
      const u0 = (nx - fit.cx) / Math.max(1e-6, fit.sx);
      const v0 = (ny - fit.cy) / Math.max(1e-6, fit.sy);

      // per-poly animation controls
      const uu = (uArr[i] ?? 0.0);
      const vv = (vArr[i] ?? 0.0);
      const pp = (phArr[i] ?? 0.0);
      const ff = (frArr[i] ?? 0.0);

      const beat = 0.5 + 0.5*Math.sin(TAU*(ff*t + pp));
      const pulse = 0.06 + 0.08*beat + (beatA ? (beatA[i] ?? 0.0) : 0.0);

      const u = u0*(1.0 - pulse) + uu*0.02*Math.sin(TAU*(t*0.6 + pp));
      const v = v0*(1.0 - pulse) + vv*0.02*Math.cos(TAU*(t*0.6 + pp));

      const f = heartField(u,v);

      // inside if f <= 0
      const inside = smoothstep(0.06, -0.06, f);

      // edge glow
      const edge = Math.exp(-Math.abs(f)/0.06) * inside;

      // base from original
      let rgb = [baseRGB[i*3], baseRGB[i*3+1], baseRGB[i*3+2]];

      // tint towards valentines-y red
      const tw = twF ? (0.5 + 0.5*Math.sin(TAU*((twF[i] ?? 0.0)*t + (twP ? (twP[i] ?? 0.0) : 0.0)))) : 0.6;
      rgb = [
        (rgb[0] + 140*inside*tw)|0,
        (rgb[1] +  10*inside)|0,
        (rgb[2] +  25*inside*(1.0-tw))|0,
      ];

      // small "spark" in highlights
      const fh = fireH ? (fireH[i] ?? 0.0) : 0.0;
      const spark = edge * (0.35 + 0.65*hash21(i*9.1, t*2.1 + fh));
      rgb = mixToWhite(rgb, spark);

      // soften background slightly (less dark whitespace)
      const bgLift = 0.10 + 0.12*inside;
      rgb = [ (rgb[0] + 255*bgLift)|0,
              (rgb[1] + 255*bgLift)|0,
              (rgb[2] + 255*bgLift)|0 ];

      // jitter (optional)
      if (hj !== 0.0 || sj !== 0.0) {{
        const j = (hash21(i*11.7, t*3.1) - 0.5);
        rgb = [
          (rgb[0] + 32*hj*j)|0,
          (rgb[1] + 32*sj*j)|0,
          rgb[2]|0,
        ];
      }}

      polys[i].setAttribute('fill', rgbToHex(rgb));
    }}
    return true;
  }}

  function renderGif(t) {{
    const b64 = state.samples_b64;
    const frames = state.frames|0;
    if (!b64 || !frames) return false;

    // samples are stored as base64 of bytes: frames * N * 3
    function b64ToBytes(s) {{
      const bin = atob(s);
      const out = new Uint8Array(bin.length);
      for (let i=0;i<bin.length;i++) out[i] = bin.charCodeAt(i);
      return out;
    }}

    if (!renderGif._bytes) {{
      renderGif._bytes = b64ToBytes(b64);
    }}
    const bytes = renderGif._bytes;
    const fps = Number(state.fps || 12);
    const frameSec = Number(state.frame_sec || (1.0 / Math.max(1e-6, fps)));
    const fi = Math.floor((t / frameSec)) % frames;

    const base = fi * N * 3;
    for (let i=0;i<N;i++) {{
      const o = base + i*3;
      const rgb = [bytes[o], bytes[o+1], bytes[o+2]];
      polys[i].setAttribute('fill', rgbToHex(rgb));
    }}
    return true;
  }}

  function applyFrame(t) {{
    // Theme priority order:
    if (renderHSV(t)) return;
    if (THEME === 'minecraft' && renderMinecraft(t)) return;
    if (THEME === 'deidee' && renderDeidee(t)) return;
    if (THEME === 'static' && renderStatic(t)) return;
    if (THEME === 'matrix' && renderMatrix(t)) return;
    if (THEME === 'champagne' && renderChampagne(t)) return;
    if (THEME === 'camo' && renderCamo(t)) return;
    if (THEME === 'fireworks' && renderFireworks(t)) return;
    if (THEME === 'nuke' && renderNuke(t)) return;
    if (THEME === 'heart' && renderHeart(t)) return;
    if (THEME === 'gif' && renderGif(t)) return;

    // Default: classic pulse-to-white
    for (let i=0;i<N;i++) {{
      const a = whitenessAt(t,i);
      const r = baseRGB[i*3], g = baseRGB[i*3+1], b = baseRGB[i*3+2];
      polys[i].setAttribute('fill', rgbToHex(mixToWhite([r,g,b], a)));
    }}
  }}

  // ---- Playback ----
  const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let t0 = null;
  function tick(now){{
    if (t0 === null) t0 = now;
    const t = (((now - t0) / 1000.0) % DURATION + DURATION) % DURATION;
    if (!reduce) {{
      applyFrame(t);
      requestAnimationFrame(tick);
    }}
  }}

  // First paint
  applyFrame(0.0);
  if (!reduce) requestAnimationFrame(tick);
}})();
"""
    return js.strip() + "\n"
