from __future__ import annotations

import base64
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Sequence

from lxml import etree

from ..constants import SVG_NS
from ..scene import Scene
from ..themes.base import Theme


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
    n = len(scene.polys)
    theme_name = getattr(theme, "name", "classic")

    # Always provide these
    poly_nx = list(map(float, scene.poly_nx))
    poly_ny = list(map(float, scene.poly_ny))

    base_rgbs = [int(c) for rgb in scene.base_rgbs for c in rgb]  # flat
    override_hsv = list(scene.override_hsv) if scene.override_hsv is not None else None

    pulses = None
    if theme_name in {"classic", "diamond", "silver", "gold", "bronze", "ruby", "jade", "sapphire", "emerald", "rainbow", "fire", "ice", "valentines", "snow"}:
        pulses = _pack_pulses(scene.pulses_per_poly)

    # Theme-specific exported state
    state: Dict[str, Any] = {}

    # Helper to safely read attrs (themes store their state as attributes)
    def _get(obj: Any, name: str, default: Any = None) -> Any:
        return getattr(obj, name, default)

    # Materials (diamond / hsv)
    if theme_name == "diamond":
        ph = _get(theme, "poly_hsv")
        cfg = _get(theme, "cfg")
        if cfg is not None:
            d = asdict(cfg)
            d["kind"] = "diamond"
            state["cfg"] = d
        else:
            state["cfg"] = {"kind": "diamond"}
        state["poly_hsv"] = [asdict(x) for x in ph] if ph is not None else []
    elif theme_name in {"silver", "gold", "bronze", "ruby", "jade", "sapphire", "emerald", "rainbow", "fire", "ice", "valentines", "snow"}:
        ph = _get(theme, "poly_hsv")
        cfg = _get(theme, "cfg")
        if cfg is not None:
            d = asdict(cfg)
            d["kind"] = "hsv"
            state["cfg"] = d
        else:
            state["cfg"] = {"kind": "hsv"}
        state["poly_hsv"] = [asdict(x) for x in ph] if ph is not None else []
    elif theme_name == "minecraft":
        state["mc_pixels"] = [int(v) for rgb in _get(theme, "mc_pixels") for v in rgb]
        state["mc_w"] = int(_get(theme, "mc_w"))
        state["mc_h"] = int(_get(theme, "mc_h"))
        state["mc_u"] = list(map(float, _get(theme, "mc_u")))
        state["mc_v"] = list(map(float, _get(theme, "mc_v")))
        state["mc_freq"] = list(map(float, _get(theme, "mc_freq")))
        state["mc_phase"] = list(map(float, _get(theme, "mc_phase")))
        state["cfg"] = asdict(_get(theme, "cfg"))
    elif theme_name == "deidee":
        de_cols = _get(theme, "colors_per_poly")
        state["alpha"] = float(_get(theme, "alpha"))
        # flatten colors + start/count per poly
        flat: List[int] = []
        start: List[int] = []
        count: List[int] = []
        cur = 0
        for cols in de_cols:
            start.append(cur)
            count.append(len(cols))
            for (r,g,b) in cols:
                flat.extend([int(r), int(g), int(b)])
                cur += 1
        state["colors_flat"] = flat
        state["colors_start"] = start
        state["colors_count"] = count
        state["seg_dur"] = list(map(float, _get(theme, "seg_dur")))
        state["phase"] = list(map(float, _get(theme, "phase")))
    elif theme_name == "heart":
        state["fit"] = dict(_get(theme, "fit"))
        state["hj"] = list(map(float, _get(theme, "hj")))
        state["sj"] = list(map(float, _get(theme, "sj")))
        state["tw_f"] = list(map(float, _get(theme, "tw_f")))
        state["tw_p"] = list(map(float, _get(theme, "tw_p")))
        state["fire_h"] = list(map(float, _get(theme, "fire_h")))
        state["beat_amp"] = float(_get(theme, "beat_amp"))
    elif theme_name == "static":
        state["seg"] = list(map(float, _get(theme, "seg")))
        state["phase"] = list(map(float, _get(theme, "phase")))
        state["color_prob"] = list(map(float, _get(theme, "color_prob")))
        state["seedf"] = float(_get(theme, "seedf"))
    elif theme_name == "matrix":
        state["col_count"] = int(_get(theme, "col_count"))
        state["drops"] = [asdict(d) for d in _get(theme, "drops")]
    elif theme_name == "champagne":
        state["bubbles"] = [asdict(b) for b in _get(theme, "bubbles")]
        state["ch_freq"] = list(map(float, _get(theme, "ch_freq")))
        state["ch_phase"] = list(map(float, _get(theme, "ch_phase")))
        state["cfg"] = asdict(_get(theme, "cfg"))
    elif theme_name == "camo":
        state["camo_seed"] = float(_get(theme, "seedf"))
        state["offx"] = list(map(float, _get(theme, "offx")))
        state["offy"] = list(map(float, _get(theme, "offy")))
        state["phase"] = list(map(float, _get(theme, "phase")))
        state["pal"] = [int(v) for rgb in _get(theme, "pal") for v in rgb]
        state["cfg"] = asdict(_get(theme, "cfg"))
    elif theme_name == "fireworks":
        state["fireworks"] = [asdict(fw) for fw in _get(theme, "fireworks")]
        state["cfg"] = asdict(_get(theme, "cfg"))
    elif theme_name == "gif":
        # Precompute per-video-frame per-poly RGB samples (compact).
        # This avoids embedding full raster frames.
        frames = int(round(duration * max(1, fps_hint)))
        if frames <= 0:
            frames = 1
        buf = bytearray(frames * n * 3)
        # theme provides frame bytes + timing; we sample via its internal method.
        frame_index_at_time = _get(theme, "_frame_index_at_time")
        sample_rgb = _get(theme, "_sample_rgb")
        frames_rgb = _get(theme, "frames_rgb")
        if frame_index_at_time is None or sample_rgb is None or frames_rgb is None:
            raise RuntimeError("GIF theme export missing internals; update mmxx_video.themes.gif")
        for fi in range(frames):
            tsec = fi / float(max(1, fps_hint))
            gi = int(frame_index_at_time(tsec))
            fb = frames_rgb[gi]
            for pi in range(n):
                r, g, b = sample_rgb(fb, poly_nx[pi], poly_ny[pi])
                o = (fi * n + pi) * 3
                buf[o] = int(r) & 255
                buf[o + 1] = int(g) & 255
                buf[o + 2] = int(b) & 255
        state["frames"] = frames
        state["fps"] = int(max(1, fps_hint))
        state["samples_b64"] = base64.b64encode(bytes(buf)).decode("ascii")
    elif theme_name == "nuke":
        # nuke theme uses `seed` (not `seedf`)
        seed = _get(theme, "seed", None)
        if seed is None:
            seed = _get(theme, "seedf", 0.0)
        state["seed"] = float(seed)

        state["offx"] = list(map(float, _get(theme, "offx")))
        state["offy"] = list(map(float, _get(theme, "offy")))
        state["phase"] = list(map(float, _get(theme, "phase")))
        state["freq"] = list(map(float, _get(theme, "freq")))

        tint = _get(theme, "tint_rgb", None)
        state["tint_rgb"] = list(map(int, tint)) if tint is not None else None

        # nuke doesn't have `cfg` (and the JS doesn't require it)
    else:
        # classic and any other unsupported theme name: treat as classic-ish
        pass

    # JS prelude: math helpers + runtime
    js = f"""
(function(){{
  'use strict';
  const script = document.currentScript;
  const svg = script && script.ownerSVGElement ? script.ownerSVGElement : document.documentElement;
  const polys = Array.from(svg.querySelectorAll('polygon[data-i]'));
  polys.sort((a,b)=> (parseInt(a.getAttribute('data-i')||'0',10) - parseInt(b.getAttribute('data-i')||'0',10)));

  const N = {n};
  const DURATION = {float(duration):.6f};
  const THEME = {json.dumps(theme_name)};

  const polyNx = {_jd(poly_nx)};
  const polyNy = {_jd(poly_ny)};
  const baseRGB = {_jd(base_rgbs)}; // flat [r,g,b,...]
  const overrideHSV = {_jd(override_hsv) if override_hsv is not None else "null"};

  const state = {_jd(state)};

  function clamp01(x){{ return x<=0?0:(x>=1?1:x); }}
  function smoothstep(e0,e1,x){{
    if (e0===e1) return 0;
    const t = clamp01((x-e0)/(e1-e0));
    return t*t*(3-2*t);
  }}
  function cosineEase(x){{
    x = clamp01(x);
    return 0.5 - 0.5*Math.cos(Math.PI*x);
  }}
  function mixRGB(a,b,t){{
    t = clamp01(t);
    const r = Math.round((1-t)*a[0] + t*b[0]);
    const g = Math.round((1-t)*a[1] + t*b[1]);
    const bb = Math.round((1-t)*a[2] + t*b[2]);
    return [r<0?0:(r>255?255:r), g<0?0:(g>255?255:g), bb<0?0:(bb>255?255:bb)];
  }}
  function addRGB(base, add, amt){{
    amt = clamp01(amt);
    const r = Math.round(base[0] + add[0]*amt);
    const g = Math.round(base[1] + add[1]*amt);
    const b = Math.round(base[2] + add[2]*amt);
    return [r<0?0:(r>255?255:r), g<0?0:(g>255?255:g), b<0?0:(b>255?255:b)];
  }}
  function mixToWhite(base, a){{
    a = clamp01(a);
    return [
      Math.round((1-a)*base[0] + a*255),
      Math.round((1-a)*base[1] + a*255),
      Math.round((1-a)*base[2] + a*255),
    ];
  }}
  function hsvToRgb(h,s,v){{
    h = ((h%1)+1)%1;
    s = clamp01(s); v = clamp01(v);
    const i = Math.floor(h*6);
    const f = h*6 - i;
    const p = v*(1-s);
    const q = v*(1-f*s);
    const t = v*(1-(1-f)*s);
    let r,g,b;
    switch(i%6){{
      case 0: r=v; g=t; b=p; break;
      case 1: r=q; g=v; b=p; break;
      case 2: r=p; g=v; b=t; break;
      case 3: r=p; g=q; b=v; break;
      case 4: r=t; g=p; b=v; break;
      case 5: r=v; g=p; b=q; break;
    }}
    return [Math.round(r*255), Math.round(g*255), Math.round(b*255)];
  }}
  function hex2(x){{
    const s = x.toString(16);
    return s.length===1 ? '0'+s : s;
  }}
  function rgbToHex(rgb){{
    return '#'+hex2(rgb[0]) + hex2(rgb[1]) + hex2(rgb[2]);
  }}

  // ---- Noise helpers (matches Python) ----
  function fract(x){{ return x - Math.floor(x); }}
  function hash01(x){{ return fract(Math.sin(x) * 43758.5453123); }}
  function noise2(x,y,seed){{
    const ix = Math.floor(x), iy = Math.floor(y);
    const fx = x - ix, fy = y - iy;
    function h(xx,yy){{ return hash01(xx*127.1 + yy*311.7 + seed*74.7); }}
    const a = h(ix,iy);
    const b = h(ix+1,iy);
    const c = h(ix,iy+1);
    const d = h(ix+1,iy+1);
    const ux = fx*fx*(3-2*fx);
    const uy = fy*fy*(3-2*fy);
    const ab = a*(1-ux) + b*ux;
    const cd = c*(1-ux) + d*ux;
    return ab*(1-uy) + cd*uy;
  }}
  function fbm2(x,y,seed,octaves){{
    let amp=0.55, freq=1.0, s=0.0, norm=0.0;
    const oct = octaves||4;
    for(let i=0;i<oct;i++){{ s += amp*noise2(x*freq,y*freq,seed + i*9.13); norm += amp; amp*=0.55; freq*=2.0; }}
    if (norm<=1e-9) return 0.0;
    return clamp01(s/norm);
  }}

  // ---- Pulse helpers (optional) ----
  const pulses = { _jd(pulses) if pulses is not None else "null" };
  function whitenessAt(t, idx){{
    if (!pulses) return 0.0;
    const s = pulses.start[idx];
    const c = pulses.count[idx];
    let a = 0.0;
    for(let j=0;j<c;j++) {{
      const k = s + j;
      const t0 = pulses.t0[k], half = pulses.half[k], amp = pulses.amp[k], pw = pulses.pow[k];
      const dt = Math.abs(t - t0);
      if (dt >= half) continue;
      let x = dt / half;
      let base = 0.5*(1 + Math.cos(Math.PI * x));
      if (pw !== 1.0) base = Math.pow(base, pw);
      const v = amp * base;
      if (v > a) a = v;
    }}
    return clamp01(a);
  }}

  function facetShimmer(t, freq, phase){{
    const s1 = 0.5 + 0.5*Math.sin(2*Math.PI*(freq*t + phase));
    const s2 = 0.5 + 0.5*Math.sin(2*Math.PI*((freq*0.47)*t + (phase*1.63)));
    let s = s1*s2;
    s = clamp01(Math.pow(s,2.0));
    return s;
  }}

  function applyFrame(t){{
    // Ambient (when cfg exists, it's embedded per-theme).
    let ambBase = 0.05, ambAmp = 0.03, ambFreq = 0.025;
    if (state.cfg && typeof state.cfg.amb_base === 'number') {{
      ambBase = state.cfg.amb_base;
      ambAmp = state.cfg.amb_amp;
      ambFreq = state.cfg.amb_freq;
    }}
    const amb = ambBase + ambAmp*(0.5 + 0.5*Math.sin(2*Math.PI*(ambFreq*t)));

    if (THEME === 'classic' || !THEME) {{
      for (let i=0;i<N;i++) {{
        const a = whitenessAt(t,i);
        const r = baseRGB[i*3], g = baseRGB[i*3+1], b = baseRGB[i*3+2];
        polys[i].setAttribute('fill', rgbToHex(mixToWhite([r,g,b], a)));
      }}
      return;
    }}

    if (THEME === 'deidee') {{
      const alpha = state.alpha;
      const flat = state.colors_flat;
      const st = state.colors_start;
      const ct = state.colors_count;
      const seg = state.seg_dur;
      const ph = state.phase;
      for (let i=0;i<N;i++) {{
        const k = ct[i];
        const pos = (t + ph[i]) / seg[i];
        const i0 = (Math.floor(pos) % k + k) % k;
        const i1 = (i0 + 1) % k;
        const f = pos - Math.floor(pos);
        const u = cosineEase(f);
        const o0 = (st[i] + i0)*3;
        const o1 = (st[i] + i1)*3;
        const c0 = [flat[o0], flat[o0+1], flat[o0+2]];
        const c1 = [flat[o1], flat[o1+1], flat[o1+2]];
        polys[i].setAttribute('fill', rgbToHex(mixRGB(c0,c1,u)));
        polys[i].setAttribute('fill-opacity', alpha.toFixed(3));
      }}
      return;
    }}

    if (THEME === 'static') {{
      const seg = state.seg;
      const ph = state.phase;
      const prob = state.color_prob;
      const seedf = state.seedf;
      for (let i=0;i<N;i++) {{
        const pos = (t + ph[i]) / seg[i];
        const k0 = Math.floor(pos);
        const f = pos - k0;
        const u = cosineEase(f);

        const a0 = hash01(seedf*0.001 + i*12.9898 + k0*78.233);
        const a1 = hash01(seedf*0.001 + i*12.9898 + (k0+1)*78.233);
        let a = (1-u)*a0 + u*a1;
        a = clamp01((a - 0.5)*1.85 + 0.5);
        let vv = 0.06 + 0.94*a;

        const scan = 0.95 + 0.05*Math.sin(2*Math.PI*(polyNy[i]*90.0 + t*1.25));
        vv = clamp01(vv*scan);

        const csel0 = hash01(seedf*0.002 + i*3.11 + k0*9.73);
        const csel1 = hash01(seedf*0.002 + i*3.11 + (k0+1)*9.73);
        const csel = (1-u)*csel0 + u*csel1;

        let rgb;
        if (csel < prob[i]) {{
          const h0 = hash01(seedf*0.003 + i*0.77 + k0*2.17);
          const h1 = hash01(seedf*0.003 + i*0.77 + (k0+1)*2.17);
          const h = ((1-u)*h0 + u*h1) % 1.0;
          const s = 0.55 + 0.45*hash01(seedf*0.004 + i*1.33 + k0*6.19);
          rgb = hsvToRgb(h, s, 0.25 + 0.75*vv);
        }} else {{
          const g = Math.round(vv*255);
          rgb = [g,g,g];
        }}
        polys[i].setAttribute('fill', rgbToHex(rgb));
      }}
      return;
    }}

    if (THEME === 'matrix') {{
      const colCount = state.col_count;
      const drops = state.drops;
      for (let i=0;i<N;i++) {{
        const nx = polyNx[i], ny = polyNy[i];
        const c = Math.max(0, Math.min(colCount-1, Math.floor(clamp01(nx)*(colCount-1))));
        const d = drops[c];
        const head = (d.phase + d.speed*t) % 1.0;
        const dist = (ny - head + 1.0) % 1.0;
        let inten=0.0;
        if (dist <= d.head) inten = 1.0;
        else if (dist <= d.tail) {{
          const z = 1.0 - (dist - d.head)/Math.max(1e-6, (d.tail - d.head));
          inten = z*z;
        }}
        const fl = 0.72 + 0.28*Math.sin(2*Math.PI*(d.flicker_freq*t + d.flicker_phase));
        inten = clamp01(inten * d.strength * fl);

        const bgV = 0.02 + 0.04*(0.5 + 0.5*Math.sin(2*Math.PI*(0.18*t + nx*1.7 + ny*0.9)));
        const bg = hsvToRgb(120/360.0, 0.55, bgV);

        let rgb = bg;
        if (inten > 0.0) {{
          const headness = smoothstep(0.75, 1.0, inten);
          const gv = 0.10 + 0.90*inten;
          const green = hsvToRgb(120/360.0, 1.0, gv);
          rgb = mixRGB(bg, green, 0.85);
          rgb = mixRGB(rgb, [255,255,255], 0.40*headness);
        }}
        polys[i].setAttribute('fill', rgbToHex(rgb));
      }}
      return;
    }}

    if (THEME === 'champagne') {{
      const bubbles = state.bubbles;
      const chFreq = state.ch_freq;
      const chPhase = state.ch_phase;
      for (let i=0;i<N;i++) {{
        const nx = polyNx[i], ny = polyNy[i];
        const h = (45/360.0) + 0.010*Math.sin(2*Math.PI*(0.04*t + chPhase[i]));
        const s = 0.18 + 0.12*(0.5 + 0.5*Math.sin(2*Math.PI*(chFreq[i]*t + chPhase[i])));
        let v = 0.30 + 0.55*(1.0 - ny);
        v += 0.06*(facetShimmer(t, 0.12 + chFreq[i], chPhase[i]) - 0.5);
        v = clamp01(v);
        let rgb = hsvToRgb(h,s,v);

        let bub=0.0, bubEdge=0.0;
        for (let j=0;j<bubbles.length;j++) {{
          const b = bubbles[j];
          const by = (b.y0 - b.speed*t) % 1.0;
          const bx = clamp01(b.x + b.wob_amp*Math.sin(2*Math.PI*(b.wob_freq*t + b.wob_phase)));
          const dx = nx-bx, dy = ny-by;
          const rr = Math.max(1e-6, b.r*b.r);
          const infl = Math.exp(-(dx*dx + dy*dy)/(2.2*rr));
          if (infl > 1e-6) {{
            bub = Math.max(bub, infl*b.strength);
            const rr2 = Math.max(1e-6, (b.r*0.60)*(b.r*0.60));
            const infl2 = Math.exp(-(dx*dx + dy*dy)/(2.2*rr2));
            bubEdge = Math.max(bubEdge, infl2*b.strength);
          }}
        }}
        bub = clamp01(bub); bubEdge = clamp01(bubEdge);
        if (bub>0.0) {{
          const bubbleRgb = hsvToRgb(200/360.0, 0.08, 1.0);
          rgb = mixRGB(rgb, bubbleRgb, 0.70*bub);
          rgb = mixRGB(rgb, [255,255,255], 0.55*bubEdge);
          const tintGate = smoothstep(0.30, 0.90, bubEdge);
          if (tintGate>0.0) {{
            const fh = (0.10 + 0.20*Math.sin(2*Math.PI*(0.06*t + nx))) % 1.0;
            const tint = hsvToRgb(fh, 0.55, 1.0);
            rgb = mixRGB(rgb, tint, 0.10*tintGate);
          }}
        }}
        rgb = mixToWhite(rgb, amb*0.08);
        polys[i].setAttribute('fill', rgbToHex(rgb));
      }}
      return;
    }}

    if (THEME === 'camo') {{
      const seed = state.camo_seed;
      const offx = state.offx, offy = state.offy, phase = state.phase;
      const palFlat = state.pal;
      function pal(i){{ const o=i*3; return [palFlat[o], palFlat[o+1], palFlat[o+2]]; }}
      const th = [0.12,0.24,0.40,0.56,0.72,0.86];
      const bw = 0.045;
      function pick(nv){{
        if (nv<=th[0]) return pal(0);
        if (nv>=th[th.length-1]) return pal(palFlat.length/3 - 1);
        let k=0;
        while(k<th.length && nv>th[k]) k++;
        const a = Math.max(0, Math.min((palFlat.length/3)-2, k));
        const b = a+1;
        const e0 = th[a]!==undefined?th[a]:th[th.length-1];
        const e1 = th[b]!==undefined?th[b]:th[th.length-1];
        const t0 = smoothstep(e0-bw, e0+bw, nv);
        const t1 = smoothstep(e1-bw, e1+bw, nv);
        const tt = clamp01((t0+t1)*0.5);
        return mixRGB(pal(a), pal(b), tt);
      }}
      for(let i=0;i<N;i++) {{
        const nx=polyNx[i], ny=polyNy[i];
        const driftX = 0.020*Math.sin(2*Math.PI*(0.035*t + phase[i]));
        const driftY = 0.018*Math.cos(2*Math.PI*(0.030*t + phase[i]*0.7));
        const x = (nx*3.2 + driftX) + offx[i]*0.001;
        const y = (ny*3.2 + driftY) + offy[i]*0.001;
        const macro = fbm2(x*0.85, y*0.85, seed+1.7, 4);
        const mid = fbm2(x*2.10, y*2.10, seed+7.9, 3);
        const micro = fbm2(x*9.00, y*9.00, seed+13.3, 2);
        const nn = clamp01(0.62*macro + 0.28*mid + 0.10*micro);

        let shade = 0.90 + 0.10*Math.sin(2*Math.PI*(0.045*t + nx*2.1 + ny*1.6));
        shade *= 0.94 + 0.06*(0.5 + 0.5*Math.sin(2*Math.PI*(0.11*t + phase[i])));
        shade = clamp01(shade);

        let rgb = pick(nn);
        const g = noise2(nx*120.0 + t*0.35, ny*120.0 + t*0.27, seed+99.1);
        const grain = (g-0.5)*0.10;
        rgb = [
          Math.max(0, Math.min(255, Math.round(rgb[0]*(shade+grain)))),
          Math.max(0, Math.min(255, Math.round(rgb[1]*(shade+grain)))),
          Math.max(0, Math.min(255, Math.round(rgb[2]*(shade+grain)))),
        ];
        rgb = mixToWhite(rgb, amb*0.04);
        polys[i].setAttribute('fill', rgbToHex(rgb));
      }}
      return;
    }}

    if (THEME === 'fireworks') {{
      const fireworks = state.fireworks;
      for (let i=0;i<N;i++) {{
        const nx=polyNx[i], ny=polyNy[i];
        let skyH = 215/360.0 + 0.010*Math.sin(2*Math.PI*(0.02*t + nx*0.6));
        const skyS = 0.55;
        let skyV = 0.03 + 0.10*Math.pow(1.0 - smoothstep(0.10, 1.00, ny), 1.4);
        skyV += 0.015*Math.sin(2*Math.PI*(0.05*t + nx*1.1 + ny*0.9));
        skyV = clamp01(skyV);
        let rgb = hsvToRgb(skyH, skyS, skyV);

        for (let j=0;j<fireworks.length;j++) {{
          const fw = fireworks[j];
          if (fw.t_launch <= t && t < fw.t_burst) {{
            let u = (t - fw.t_launch) / Math.max(1e-6, (fw.t_burst - fw.t_launch));
            u = clamp01(u);
            const uu = cosineEase(u);
            const y0 = 1.05;
            const y = y0 + (fw.yb - y0)*uu;
            const dx = Math.abs(nx - fw.x);
            const dy = ny - y;

            if (dy>=0.0 && dy<=fw.trail_len) {{
              const trailCore = Math.exp(-(dx*dx)/(2.0*fw.trail_w*fw.trail_w));
              const trailLenGate = Math.exp(-(dy*dy)/(2.0*Math.pow(fw.trail_len*0.55,2)));
              let trail = clamp01(trailCore*trailLenGate);
              const hot = clamp01(1.0 - dy/Math.max(1e-6, fw.trail_len));
              const th = (fw.trail_h + 0.02*Math.sin(2*Math.PI*(0.20*t + fw.glitter_p))) % 1.0;
              const trailRgb0 = hsvToRgb(th, 0.85, 0.35 + 0.65*hot);
              const spark = smoothstep(0.70, 1.00, hot) * (0.65 + 0.35*Math.sin(2*Math.PI*(fw.glitter_f*t + fw.glitter_p + dy*3.0)));
              let trailRgb = mixRGB(trailRgb0, [255,255,255], 0.20*clamp01(spark));
              rgb = addRGB(rgb, trailRgb, 0.90*trail);
            }}

            const headR = 0.020;
            const d2 = (nx - fw.x)*(nx - fw.x) + (ny - y)*(ny - y);
            const head = Math.exp(-d2/(2.0*headR*headR));
            if (head>1e-5) {{
              const headRgb = hsvToRgb((fw.trail_h + 0.01)%1.0, 0.30, 1.0);
              rgb = addRGB(rgb, headRgb, 0.95*clamp01(head));
            }}
          }}

          if (t >= fw.t_burst) {{
            const dt = t - fw.t_burst;
            if (dt <= 4.0) {{
              const cx = fw.x, cy = fw.yb;
              const dx = nx - cx, dy = ny - cy;
              const d = Math.sqrt(dx*dx + dy*dy);
              const rr = fw.vel * dt;
              let ring = Math.exp(-Math.pow(d-rr,2)/(2.0*fw.ring_w*fw.ring_w));
              ring *= Math.exp(-dt/Math.max(1e-6, fw.decay));
              ring = clamp01(ring);
              if (ring>1e-5) {{
                const ang = Math.atan2(dy, dx);
                let sp = Math.abs(Math.sin(fw.spoke_n*ang + fw.spoke_phase));
                sp = Math.pow(sp, 2.2);
                const spokeGate = 0.35 + 0.65*sp;
                let flick = 0.70 + 0.30*Math.sin(2*Math.PI*(fw.glitter_f*t + fw.glitter_p + (ang*0.07)));
                flick = clamp01(flick);
                const inten = ring*spokeGate*flick;

                const sel = 0.5 + 0.5*Math.sin(fw.spoke_n*ang + fw.spoke_phase);
                const h = (sel>=0.0) ? fw.hue_a : fw.hue_b;
                let core = Math.exp(-(d*d)/(2.0*Math.pow(0.06 + 0.03*dt,2)));
                core = clamp01(core);

                const s = clamp01(0.70 + 0.30*sp);
                const v = clamp01(0.25 + 0.75*inten);
                let burstRgb = hsvToRgb(h, s, v);
                burstRgb = mixRGB(burstRgb, [255,255,255], 0.25*clamp01(inten + 0.6*core));
                rgb = addRGB(rgb, burstRgb, 0.95*inten);

                const haze = clamp01(0.22*ring*(0.5 + 0.5*Math.sin(2*Math.PI*(0.45*t + fw.glitter_p))));
                if (haze>1e-5) {{
                  const ember = hsvToRgb((h + 0.02)%1.0, 0.55, 0.60);
                  rgb = addRGB(rgb, ember, haze);
                }}
              }}
            }}
          }}
        }}

        rgb = mixToWhite(rgb, amb*0.05);
        polys[i].setAttribute('fill', rgbToHex(rgb));
      }}
      return;
    }}

    if (THEME === 'minecraft') {{
      const pix = state.mc_pixels;
      const W = state.mc_w, H = state.mc_h;
      const uarr = state.mc_u, varr = state.mc_v, farr = state.mc_freq, parr = state.mc_phase;
      const cfg = state.cfg || {{}};
      for(let i=0;i<N;i++) {{
        const u=uarr[i], v=varr[i];
        const px = Math.round(u*(W-1));
        const py = Math.round(v*(H-1));
        const o = (py*W + px)*3;
        const base = [pix[o], pix[o+1], pix[o+2]];

        const shim = facetShimmer(t, farr[i], parr[i]);
        const pulse = pulses ? whitenessAt(t,i) : 0.0;
        const wob = 0.92 + 0.12*(0.5 + 0.5*Math.sin(2*Math.PI*(0.07*t + parr[i])));
        let rgb = [
          Math.max(0, Math.min(255, Math.round(base[0]*wob))),
          Math.max(0, Math.min(255, Math.round(base[1]*wob))),
          Math.max(0, Math.min(255, Math.round(base[2]*wob))),
        ];

        const glPulseW = cfg.gl_pulse_w ?? 0.80;
        const glShimW = cfg.gl_shim_w ?? 0.52;
        const specEdge0 = cfg.spec_edge0 ?? 0.36;
        const specScale = cfg.spec_scale ?? 0.72;

        const glint = Math.max(glShimW*shim, glPulseW*pulse);
        let spec = smoothstep(specEdge0, 1.0, glint) * specScale;

        if (spec>0.0) {{
          const sun = hsvToRgb(45/360.0, 0.18, 1.0);
          rgb = mixRGB(rgb, sun, 0.12*spec);
          rgb = mixRGB(rgb, [255,255,255], 0.28*spec);
        }}
        rgb = mixToWhite(rgb, amb*0.06);
        polys[i].setAttribute('fill', rgbToHex(rgb));
      }}
      return;
    }}

    if (THEME === 'heart') {{
      const fit = state.fit;
      const hj = state.hj, sj = state.sj, twf = state.tw_f, twp = state.tw_p, fireh = state.fire_h;
      const beatAmp = state.beat_amp;
      const baseH = (overrideHSV ? overrideHSV[0] : (335/360.0));
      function beatWave(t){{
        const cycle=2.25;
        const p = (t/cycle) % 1.0;
        function bump(phase, center, width){{
          const d = Math.abs(((phase-center+0.5)%1.0) - 0.5);
          if (d>=width) return 0.0;
          const x = d/width;
          return 0.5*(1 + Math.cos(Math.PI*x));
        }}
        const b1 = bump(p, 0.16, 0.11);
        const b2 = bump(p, 0.33, 0.15);
        const tail = 0.20*(1.0 - smoothstep(0.34, 0.98, p));
        let raw = 0.04 + 0.50*b1 + 1.00*b2 + tail;
        raw = clamp01(raw);
        const eased = cosineEase(raw);
        return clamp01(0.55*raw + 0.45*eased);
      }}
      function heartVal(x,y){{
        x = Math.abs(x);
        x *= 1.12;
        y *= 1.02;
        y += 0.10;
        const a = x*x + y*y - 1.0;
        return (a*a*a) - (x*x)*(y*y*y);
      }}
      function heartMaskSingle(nx,ny,pulse){{
        const xs = nx*2.0 - 1.0;
        const ys = (1.0-ny)*2.0 - 1.0;
        const beatScale = 1.0 + beatAmp*pulse;
        const sx = fit.sx * beatScale;
        const sy = fit.sy * beatScale;
        const x = (xs/sx) + fit.cx;
        const y = (ys/sy) + fit.cy;
        const val = heartVal(x,y);
        const edge = 0.026;
        const mask = 1.0 - smoothstep(-edge, edge, val);
        const glow = Math.exp(-Math.abs(val)*9.5);
        return [clamp01(mask), clamp01(glow)];
      }}
      function heartMaskAA(nx,ny,pulse){{
        const eps=0.0026;
        const samples = [
          [0,0],[+eps,0],[-eps,0],[0,+eps],[0,-eps],
          [eps*0.7,eps*0.7],[-eps*0.7,eps*0.7],[eps*0.7,-eps*0.7],[-eps*0.7,-eps*0.7],
        ];
        let ms=0.0, gs=0.0;
        for(let k=0;k<samples.length;k++) {{
          const dx=samples[k][0], dy=samples[k][1];
          const a = heartMaskSingle(clamp01(nx+dx), clamp01(ny+dy), pulse);
          ms += a[0]; gs += a[1];
        }}
        const inv = 1.0/samples.length;
        return [clamp01(ms*inv), clamp01(gs*inv)];
      }}

      const pulse = beatWave(t);
      for(let i=0;i<N;i++) {{
        const nx=polyNx[i], ny=polyNy[i];
        const mg = heartMaskAA(nx,ny,pulse);
        const mask=mg[0], glow=mg[1];

        const bgH = (300/360.0) + 0.010*Math.sin(2*Math.PI*(0.03*t + nx*0.7));
        const bgS = 0.35;
        const vign = Math.sqrt(Math.pow(nx-0.5,2) + Math.pow(ny-0.5,2));
        let bgV = 0.05 + 0.07*(1.0 - smoothstep(0.15, 0.95, vign));
        bgV += 0.02*Math.sin(2*Math.PI*(0.05*t + nx*1.1 + ny*0.6));
        bgV = clamp01(bgV);
        const bg = hsvToRgb(bgH,bgS,bgV);

        const h = (baseH + 0.090*(nx-0.5) + 0.040*Math.sin(2*Math.PI*(0.09*t + ny*0.9)) + hj[i]) % 1.0;
        const satBreathe = 0.85 + 0.30*(0.5 + 0.5*Math.sin(2*Math.PI*(0.12*t + twp[i])));
        let s = (0.72 + 0.30*mask + 0.22*glow) * sj[i] * satBreathe;
        s = clamp01(s);
        let v = 0.18 + 0.46*mask;
        v += 0.34*mask*(0.18 + 0.82*pulse);
        v += 0.22*glow*(0.30 + 0.70*pulse);
        v = clamp01(v);

        let heart = hsvToRgb(h,s,v);
        let rgb = mixRGB(bg, heart, mask);

        const edge = clamp01(glow*(0.22 + 0.40*pulse));
        rgb = mixRGB(rgb, [255,255,255], 0.40*edge);

        const fireGate = smoothstep(0.35, 0.92, edge);
        if (fireGate>0.0) {{
          const fh = (fireh[i] + 0.02*Math.sin(2*Math.PI*(0.06*t + nx))) % 1.0;
          const fire = hsvToRgb(fh, clamp01(0.65 + 0.35*s), 1.0);
          rgb = mixRGB(rgb, fire, 0.18*fireGate);
        }}

        let tw = facetShimmer(t, twf[i], twp[i]);
        tw = Math.pow(mask,1.55) * Math.pow(tw,1.30);
        const sparkle = clamp01(tw*(0.06 + 0.22*pulse) + Math.pow(mask,2.2)*(0.02 + 0.06*pulse));
        if (sparkle>0.0) {{
          rgb = mixRGB(rgb, [255,255,255], sparkle);
          const fh2 = (fireh[i] + 0.015*Math.sin(2*Math.PI*(0.08*t + ny))) % 1.0;
          const flare = hsvToRgb(fh2, 0.85, 1.0);
          rgb = mixRGB(rgb, flare, 0.10*sparkle);
        }}

        rgb = mixToWhite(rgb, amb*0.10);
        polys[i].setAttribute('fill', rgbToHex(rgb));
      }}
      return;
    }}

    if (THEME === 'gif') {{
      const frames = state.frames;
      const fps = state.fps;
      const b64 = state.samples_b64;
      // decode once
      if (!applyFrame._gifDecoded) {{
        const bin = atob(b64);
        const arr = new Uint8Array(bin.length);
        for(let i=0;i<bin.length;i++) arr[i] = bin.charCodeAt(i) & 255;
        applyFrame._gifSamples = arr;
        applyFrame._gifDecoded = true;
      }}
      const arr = applyFrame._gifSamples;
      const fi = (Math.floor(t*fps) % frames + frames) % frames;
      const base = fi*N*3;
      for(let i=0;i<N;i++) {{
        const o = base + i*3;
        polys[i].setAttribute('fill', rgbToHex([arr[o],arr[o+1],arr[o+2]]));
      }}
      return;
    }}

    if (THEME === 'nuke') {{
      const tn = clamp01(t / Math.max(1e-6, DURATION));
      const ox=0.50, oy=0.86;

      const flash = clamp01(Math.exp(-Math.pow((tn-0.035)/0.035,2.0)));

      const coreGrow = smoothstep(0.02, 0.22, tn);
      const coreR = 0.055 + 0.28*coreGrow;
      const coreY = oy - (0.03 + 0.33*smoothstep(0.06, 0.55, tn));

      const capRise = smoothstep(0.12, 0.80, tn);
      const capY = oy - (0.14 + 0.58*capRise);
      const capRx = 0.10 + 0.50*smoothstep(0.15, 0.78, tn);
      const capRy = 0.07 + 0.22*smoothstep(0.15, 0.70, tn);

      const stemBuild = smoothstep(0.10, 0.55, tn);
      const stemR = 0.030 + 0.080*stemBuild;

      let heatDecay = 1.0 - smoothstep(0.10, 0.82, tn);
      heatDecay = clamp01(heatDecay);

      const smokeBuild = smoothstep(0.10, 0.42, tn);
      const smokeFade = 1.0 - 0.55*smoothstep(0.72, 1.00, tn);
      const smokeMul = clamp01(smokeBuild*smokeFade);

      const waveT = smoothstep(0.03, 0.30, tn) * (1.0 - smoothstep(0.30, 0.62, tn));
      const waveR = 0.02 + 1.10*smoothstep(0.03, 0.42, tn);
      const waveW = 0.010 + 0.010*(1.0 - smoothstep(0.03, 0.42, tn));

      const hotWhite=[255,255,255], hotYellow=[255,240,190], hotOrange=[255,155,55], hotRed=[205,65,25];
      const smokeDark=[18,17,16], smokeLight=[96,84,70], dustBrown=[90,60,35];
      const warmGlow=[255,190,95];

      function ellipseMask(dx,dy,rx,ry,edge){{
        rx = Math.max(1e-6, rx); ry = Math.max(1e-6, ry);
        const q = Math.sqrt((dx*dx)/(rx*rx) + (dy*dy)/(ry*ry));
        return clamp01(1.0 - smoothstep(1.0, 1.0 + edge, q));
      }}
      function expRing(d,r,w){{
        w = Math.max(1e-6,w);
        return Math.exp(-Math.pow(d-r,2)/(2.0*w*w));
      }}

      const seed = state.seed;
      const offx=state.offx, offy=state.offy, phase=state.phase, freq=state.freq;
      const tint = state.tint_rgb;

      for(let i=0;i<N;i++) {{
        const nx=polyNx[i], ny=polyNy[i];
        let skyV = 0.02 + 0.16*Math.pow((1.0-ny),2.0);
        skyV += 0.02*Math.sin(2*Math.PI*(0.03*t + nx*0.7 + ny*0.23));
        let sky = hsvToRgb(215/360.0, 0.70, clamp01(skyV));

        const dx0 = nx-ox, dy0=ny-oy;
        const d0 = Math.sqrt(dx0*dx0 + dy0*dy0);

        const driftX = 0.06*Math.sin(2*Math.PI*(0.030*t + phase[i]));
        const driftY = -0.08*(0.50 + 0.50*Math.sin(2*Math.PI*(0.022*t + phase[i]*0.7)));

        const n0 = fbm2(nx*3.0 + offx[i]*0.001 + driftX,
                       ny*3.0 + offy[i]*0.001 + driftY,
                       seed+7.1, 4);
        const n2 = clamp01((n0 - 0.5)*1.35 + 0.5);

        const coreMask = ellipseMask(nx-ox, ny-coreY, coreR*(1.05 + 0.22*(n2-0.5)), coreR*(0.92 + 0.22*(0.5 - (n2-0.5))), 0.12);
        const capMask = ellipseMask(nx-ox, ny-capY, capRx*(0.92 + 0.24*n2), capRy*(0.92 + 0.28*(1.0-n2)), 0.14);

        const xnorm = Math.abs(nx-ox)/Math.max(1e-6, stemR*(0.85 + 0.40*n2));
        const xmask = clamp01(1.0 - smoothstep(1.0, 1.22, xnorm));
        const yGateTop = smoothstep(capY-0.06, capY+0.02, ny);
        const yGateBot = 1.0 - smoothstep(oy+0.02, oy+0.14, ny);
        const stemMask = clamp01(xmask*yGateTop*yGateBot);

        const plume = Math.max(coreMask, Math.max(capMask, stemMask));

        const flick = 0.85 + 0.15*Math.sin(2*Math.PI*(freq[i]*t + phase[i]));
        const heatLocal = (1.15*coreMask + 0.85*stemMask + 0.75*capMask);
        const heat = clamp01(heatLocal*heatDecay*flick);

        let smoke = smokeMul*(1.10*capMask + 0.75*stemMask + 0.25*(plume-coreMask));
        smoke *= (0.55 + 0.95*n2);
        smoke = clamp01(smoke);

        let dust = smoothstep(oy-0.02, oy+0.20, ny) * smoothstep(0.08, 0.30, tn);
        dust *= smoothstep(0.00, 0.45, Math.abs(nx-ox));
        dust *= (0.55 + 0.75*n2);
        dust = clamp01(dust);

        const glowR = 0.18 + 0.36*smoothstep(0.02, 0.25, tn);
        let glow = Math.exp(-(d0*d0)/(2.0*glowR*glowR));
        glow *= clamp01(0.25 + 0.90*heatDecay + 0.85*flash);
        sky = addRGB(sky, warmGlow, 0.35*glow);

        let fire = mixRGB(hotRed, hotOrange, smoothstep(0.10, 0.35, heat));
        fire = mixRGB(fire, hotYellow, smoothstep(0.35, 0.70, heat));
        fire = mixRGB(fire, hotWhite, smoothstep(0.78, 1.00, heat));
        if (tint && heat>0.01) {{
          fire = mixRGB(fire, tint, 0.22*heat);
        }}

        const smix = clamp01(0.35 + 0.45*(1.0-ny) + 0.25*n2);
        let smokeRgb = mixRGB(smokeDark, smokeLight, smix);
        smokeRgb = mixRGB(smokeRgb, dustBrown, 0.45*dust);

        let rgb = sky;
        rgb = addRGB(rgb, fire, 0.95*heat);
        if (smoke>0.0) rgb = mixRGB(rgb, smokeRgb, 0.90*smoke);
        if (dust>0.0) rgb = mixRGB(rgb, smokeRgb, 0.55*dust);

        if (flash>1e-4) {{
          const bloom = flash*Math.exp(-(d0*d0)/(2.0*(0.20*0.20)));
          rgb = mixRGB(rgb, hotWhite, 0.65*clamp01(bloom));
        }}

        if (waveT>1e-4) {{
          let ring = expRing(d0, waveR, waveW) * waveT;
          ring *= smoothstep(oy-0.08, oy+0.12, ny);
          ring = clamp01(ring);
          if (ring>0.0) {{
            rgb = addRGB(rgb, hotYellow, 0.35*ring);
            rgb = mixRGB(rgb, hotWhite, 0.55*ring);
          }}
        }}

        polys[i].setAttribute('fill', rgbToHex(rgb));
      }}
      return;
    }}

    // HSV-material themes (silver/gold/etc + diamond handled separately)
    if (state.poly_hsv && state.poly_hsv.length === N && state.cfg) {{
      const cfg = state.cfg;
      const phs = state.poly_hsv;

      const kind = cfg.kind || 'hsv';
      if (kind === 'diamond') {{
        for(let i=0;i<N;i++) {{
          const ph = phs[i];
          const pulse = whitenessAt(t,i);
          const shim = facetShimmer(t, ph.freq, ph.phase);
          const glint = Math.max(cfg.gl_pulse_w*pulse, cfg.gl_shim_w*shim);
          let spec = smoothstep(cfg.spec_edge0, 1.0, glint);
          spec = clamp01(spec*cfg.spec_scale);

          let g0 = clamp01(ph.v * ph.v_mul);
          let g = clamp01((g0 - 0.5)*1.20 + 0.5);
          const grey = Math.round(g*255);
          let rgb = [grey,grey,grey];
          rgb = mixRGB(rgb, [255,255,255], 0.18 + 0.72*spec);

          if (ph.fire_enabled) {{
            const gate = smoothstep(cfg.fire_gate0, 1.0, glint);
            if (gate > 0.001) {{
              const drift = cfg.fire_hue_drift_amp*Math.sin(2*Math.PI*(cfg.fire_hue_drift_freq*t + ph.phase));
              const fh = (ph.fire_hue + drift) % 1.0;
              const satBase = cfg.fire_sat_base_min + (cfg.fire_sat_base_max - cfg.fire_sat_base_min)*(0.5 + 0.5*Math.sin(2*Math.PI*(0.15*t + ph.phase)));
              const satPeak = cfg.fire_sat_peak_min + (cfg.fire_sat_peak_max - cfg.fire_sat_peak_min)*(0.5 + 0.5*Math.sin(2*Math.PI*(0.22*t + ph.freq)));
              const s = clamp01((satBase + gate*satPeak) * ph.fire_sat_mul);
              const vv = clamp01(0.55 + 0.45*gate);
              const fireRgb = hsvToRgb(fh, s, vv);
              const mixAmt = clamp01(gate*(0.20 + 0.55*ph.fire_white_mix));
              rgb = mixRGB(rgb, fireRgb, mixAmt);
            }}
          }}
          rgb = mixToWhite(rgb, amb*0.20);
          polys[i].setAttribute('fill', rgbToHex(rgb));
        }}
        return;
      }}

      // Regular HSV material
      for(let i=0;i<N;i++) {{
        const ph = phs[i];
        const pulse = whitenessAt(t,i);
        const shim = facetShimmer(t, ph.freq, ph.phase);

        const driftH = cfg.hue_shimmer_amp*Math.sin(2*Math.PI*(0.10*t + ph.phase));
        const h = (ph.h + driftH) % 1.0;

        let v = clamp01((ph.v * ph.v_mul) + cfg.val_shimmer_amp*(shim - 0.5));
        v = clamp01(0.06 + 0.90*v);

        let s = ph.s * cfg.body_sat_mul * ph.sat_mul;
        s = clamp01(s + cfg.sat_dark_boost*(1.0 - v));

        let rgb = hsvToRgb(h,s,v);

        const glint = Math.max(cfg.gl_pulse_w*pulse, cfg.gl_shim_w*shim);
        let spec = smoothstep(cfg.spec_edge0, 1.0, glint);
        spec = clamp01(spec*cfg.spec_scale);

        const sh = (h + cfg.sheen_hue_shift) % 1.0;
        const ss = clamp01(s + cfg.sheen_sat_boost);
        const sheen = hsvToRgb(sh, ss, 1.0);
        rgb = mixRGB(rgb, sheen, cfg.sheen_mix*spec);
        rgb = mixRGB(rgb, [255,255,255], 0.06*spec);

        if (ph.fire_enabled) {{
          const gate = smoothstep(cfg.fire_gate0, 1.0, glint);
          if (gate > 0.001) {{
            const drift = cfg.fire_hue_drift_amp*Math.sin(2*Math.PI*(cfg.fire_hue_drift_freq*t + ph.phase));
            const fh = (ph.fire_hue + drift) % 1.0;
            const satBase = cfg.fire_sat_base_min + (cfg.fire_sat_base_max - cfg.fire_sat_base_min)*(0.5 + 0.5*Math.sin(2*Math.PI*(0.15*t + ph.phase)));
            const satPeak = cfg.fire_sat_peak_min + (cfg.fire_sat_peak_max - cfg.fire_sat_peak_min)*(0.5 + 0.5*Math.sin(2*Math.PI*(0.22*t + ph.freq)));
            const s2 = clamp01((satBase + gate*satPeak) * ph.fire_sat_mul);
            const v2 = clamp01(0.55 + 0.45*gate);
            const fireRgb = hsvToRgb(fh, s2, v2);
            rgb = mixRGB(rgb, fireRgb, clamp01(gate*0.45));
          }}
        }}

        rgb = mixToWhite(rgb, amb*0.10);
        polys[i].setAttribute('fill', rgbToHex(rgb));
      }}
      return;
    }}

    // Fallback: behave like classic
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
