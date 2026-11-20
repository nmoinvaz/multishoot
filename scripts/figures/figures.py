"""The README's figures, drawn from the JSON that figure-data writes (cargo run --release -- ../../docs/figures/data).

    python3 -B figures.py [name ...]

Each figure prints its numbers as a table on stdout. STATIC=1 draws every animated figure at its last frame, for
checking the layout.
"""

import json
import math
import os
import sys

from data_graphs import (GRID, INK, INK_SOFT, SERIES, SURFACE, Svg, axes, better_hint, callout, caption, esc,
                         fmt_pct, footnote, legend, marker, segment, text_width)
from diagrams import Diagram

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir))
DATA = os.path.join(ROOT, "docs", "figures", "data")
OUT = os.path.join(ROOT, "docs", "figures")
REPO = "https://github.com/nmoinvaz/multishoot"
STATIC = os.environ.get("STATIC") == "1"
NEUTRAL = SERIES[-1]
W = 760


def load(name):
    with open(os.path.join(DATA, name + ".json"), encoding="utf-8") as f:
        return json.load(f)


def source(t):
    """the engine that produced a figure's data, as figure-data saved it with the data"""
    return f"multishoot {t['source']['version']} ({t['source']['commit']})"


def write(name, svg_text):
    path = os.path.join(OUT, name + ".svg")
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg_text)
    print(f"wrote {os.path.relpath(path, ROOT)}")


def table(title, header, rows):
    print(f"\n{title}")
    widths = [max(len(str(c)) for c in col) for col in zip(header, *rows)]
    print("  ".join(str(h).ljust(w) for h, w in zip(header, widths)))
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


# ---------------------------------------------------------------------------------------------------------------
# animation, SMIL so that it plays in an <img> on GitHub


def shown(a, b, period):
    """An opacity animation that shows its element from a to b seconds of each period (b None: to the end)."""
    if STATIC:
        return ""
    fa = a / period
    fb = 1.0 if b is None else b / period
    if fa <= 0 and fb >= 1:
        return ""
    if fa <= 0:
        vals, keys = "1;0", f"0;{fb:.4f}"
    elif fb >= 1:
        vals, keys = "0;1", f"0;{fa:.4f}"
    else:
        vals, keys = "0;1;0", f"0;{fa:.4f};{fb:.4f}"
    return (f'<animate attributeName="opacity" values="{vals}" keyTimes="{keys}" dur="{period}s" '
            f'calcMode="discrete" repeatCount="indefinite"/>')


def drawn(a, b, period):
    """A dash animation that draws a path of pathLength 1 from a to b seconds, held to the end of the period."""
    if STATIC:
        return ""
    fa, fb = a / period, b / period
    return (f'<animate attributeName="stroke-dashoffset" values="1;1;0;0" keyTimes="0;{fa:.4f};{fb:.4f};1" '
            f'dur="{period}s" repeatCount="indefinite"/>')


def group(svg, inner, anim="", opacity=None):
    op = f' opacity="{opacity}"' if opacity is not None else ""
    svg.add(f"<g{op}>{anim}{''.join(inner)}</g>")


def path_d(pts):
    return "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts)


def polyline(pts, color, width=1.5, dash=None, extra=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="{path_d(pts)}" fill="none" stroke="{color}" stroke-width="{width}" '
            f'stroke-linejoin="round"{d}{extra}/>')


def drawing(pts, color, a, b, period, width=2.0):
    """A polyline drawn from a to b seconds of each period; drawn whole when static"""
    if STATIC:
        return polyline(pts, color, width)
    return (f'<path d="{path_d(pts)}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" '
            f'pathLength="1" stroke-dasharray="1" stroke-dashoffset="0">{drawn(a, b, period)}</path>')


def mover(pts, color, period):
    """a 6px marker moving through pts over each period; at the last point when static"""
    if STATIC:
        x, y = pts[-1]
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" stroke="{SURFACE}" stroke-width="2"/>'
    xs = ";".join(f"{x:.1f}" for x, _ in pts)
    ys = ";".join(f"{y:.1f}" for _, y in pts)
    return (f'<circle r="6" fill="{color}" stroke="{SURFACE}" stroke-width="2">'
            f'<animate attributeName="cx" values="{xs}" dur="{period}s" repeatCount="indefinite"/>'
            f'<animate attributeName="cy" values="{ys}" dur="{period}s" repeatCount="indefinite"/></circle>')


def dot(x, y, color, r=3.2, title=None):
    t = f"<title>{esc(title)}</title>" if title else ""
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}" stroke="{SURFACE}" stroke-width="1">{t}</circle>'


def clip(svg, cid, x, y, w, h):
    svg.add(f'<clipPath id="{cid}"><rect x="{x}" y="{y}" width="{w}" height="{h}"/></clipPath>')
    return f'clip-path="url(#{cid})"'


def sci(v):
    return f"{v:.1e}".replace("e-0", "e-").replace("e+0", "e")


def exp_fmt(v):
    return f"1e{int(round(v))}"


# ---------------------------------------------------------------------------------------------------------------
# diagrams


def lanes():
    d = Diagram()
    d.group("lane", "lane types")
    d.group("out", "what Shooting forms")
    d.node("eq", "Equations<B>", col=0, row=2,
           note="taylor_coeffs, left_state, right_state: written once over a generic lane T")
    rows = [
        ("val", "DV · QV · OV", "values in double-, quad- and octo-double", "res", "residual F(z)"),
        ("dual", "Du · Qu · Ou · Oun<N>", "one or N forward-mode tangents", "jac", "Jacobian, J v"),
        ("jet", "Jet<N> · QJet<N> · OJet<N>", "power series in the parameter", "ser", "residual_jet, JetSession"),
        ("cplx", "Dc", "a complex tangent, the rate of exp(λ t)", "mode", "mode_matrix, mode_logdet"),
        ("f64", "Fu · Fun<N> · Dtn<N>", "f64 tangents", "fast", "jacobian_f64, staircase"),
    ]
    for r, (lid, lab, note, oid, olab) in enumerate(rows):
        d.node(lid, lab, col=1, row=r, group="lane", note=note)
        d.node(oid, olab, col=2, row=r, group="out")
        d.edge("eq", lid)
        d.edge(lid, oid)
    write("lanes", d.svg())
    return d.ascii()


def solvers():
    d = Diagram()
    d.group("ms", "multishoot")
    d.group("tr", "multisolver traits")
    d.group("sv", "multisolver solvers")
    d.node("sh", "Shooting<E, B>", col=0, row=1, group="ms")
    d.node("ig", "InGlobal", col=0, row=3, group="ms", note="Shooting::in_global(g, pin): global g as the parameter")
    d.node("pr", "Problem<P>", col=1, row=0, group="tr", note="len, residual, jacobian")
    d.node("jv", "Jvp<P>", col=1, row=1, group="tr", note="J v without the Jacobian")
    d.node("rj", "ResidualJet<P>", col=1, row=2, group="tr", note="the residual's series in the parameter")
    d.node("fa", "Family<P>", col=1, row=3, group="tr", note="residual_at, jacobian_at, dp_at")
    d.node("nw", "Newton · chord · QNERR · LM · deflation", col=2, row=0, group="sv")
    d.node("kr", "Krylov-refined Newton", col=2, row=1, group="sv")
    d.node("an", "ANM · exact tangent predictor", col=2, row=2, group="sv")
    d.node("ar", "arclength continuation · locate_fold", col=2, row=3, group="sv")
    d.edge("sh", "pr")
    d.edge("sh", "jv")
    d.edge("sh", "rj")
    d.edge("sh", "ig", "in_global")
    d.edge("ig", "fa")
    d.edge("pr", "nw")
    d.edge("jv", "kr")
    d.edge("rj", "an")
    d.edge("fa", "ar")
    write("solvers", d.svg())
    return d.ascii()


def shared_hash(hashes):
    """the one fingerprint of every backend measured; a ValueError when two differ or none was measured"""
    distinct = set(hashes.values())
    if len(distinct) != 1:
        raise ValueError(f"the backends' fingerprints are not one: {hashes}")
    return distinct.pop()


def backends():
    # the control of the check: two backends with different fingerprints must fail it
    try:
        shared_hash({"NEON": "983b7216fe9b7bb1", "portable": "983b7216fe9b7bb0"})
    except ValueError:
        pass
    else:
        raise AssertionError("shared_hash passed two different fingerprints")
    hashes = {}
    for f in sorted(os.listdir(DATA)):
        if f.startswith("backend-") and f.endswith(".json"):
            b = load(f[:-len(".json")])
            hashes[b["backend"]] = b["hash"]
    d = Diagram()
    d.group("bk", "multidouble backends")
    d.group("ck", "fingerprints")
    d.node("eq", "the same source", col=0, row=1, note="Equations<B> and the engine, no per-backend code")
    order = ["NEON", "portable", "SSE2", "AVX2"]
    order += sorted(n for n in hashes if n not in order)
    labels = {"SSE2": "SSE2 · MSVC x64"}
    for r, name in enumerate(order):
        d.node(name, labels.get(name, name), col=1, row=r, group="bk")
        d.edge("eq", name)
    measured = [n for n in order if n in hashes]
    h = shared_hash(hashes)
    d.node("fp", f"roots: {h}", col=2, row=0, group="ck",
           note="FNV-1a of every part of the README example's double- and quad-double roots, run here")
    d.node("ci", "the tests' pinned hashes", col=2, row=3, group="ck",
           note="CI runs the suite on each backend and compares bit for bit")
    for name in measured:
        d.edge(name, "fp", "here")
    for name in order:
        if name not in measured:
            d.edge(name, "ci", "in CI", dashed=True)
    write("backends", d.svg())
    table("backend fingerprints", ["backend", "hash"], [(k, v) for k, v in hashes.items()])
    return d.ascii()


# ---------------------------------------------------------------------------------------------------------------
# Taylor steps


def taylor():
    t = load("taylor")
    svg = Svg(W)
    xs = t["path_x"]
    hs = [b - a for a, b in zip(xs, xs[1:])]
    y = caption(svg, 16, 22, "Each step keeps its last two terms below tol; one leaving a sliver takes half the rest",
                f"Rotor (tests/common), {len(hs)} steps from x = {xs[0]} to {xs[-1]:g} at order {t['path_kt']}, "
                f"tol {sci(t['path_tol'])}: the pole at x = −1 limits the steps, then the solution's oscillation")
    period = 7.0
    px, pw = 84, W - 84 - 56
    # the steps as a strip, one bar per step
    sh = 14
    xm, ym = axes(svg, px, y + sh + 10, pw, 150, (xs[0], xs[-1]), (-3, -1.4), "x",
                  "step length h", yfmt=lambda v: f"{10 ** v:.0e}".replace("e-0", "e-"))
    for i, (a, b) in enumerate(zip(xs, xs[1:])):
        x0, x1 = xm(a), xm(b)
        anim = shown(0.3 + 5.0 * i / len(hs), None, period)
        fo = "1" if i % 2 == 0 else "0.45"
        svg.add(f'<rect x="{x0:.2f}" y="{y:.1f}" width="{max(x1 - x0, 0.4):.2f}" height="{sh}" fill="{SERIES[0]}" '
                f'fill-opacity="{fo}">'
                f'{anim}<title>step {i + 1}: [{a:.4f}, {b:.4f}], h = {b - a:.3e}</title></rect>')
    svg.text(px - 6, y + 11, "steps", size=10, anchor="end")
    pts = [(xm((a + b) / 2), ym(math.log10(b - a))) for a, b in zip(xs, xs[1:])]
    svg.add(drawing(pts, SERIES[0], 0.3, 5.3, period))
    callout(svg, *pts[0], f"h = {sci(hs[0])} near the pole", dx=36, dy=-6)
    callout(svg, *pts[-1], f"h = {sci(hs[-1])}", dx=-30, dy=28)
    svg.text(xm(-0.55), ym(-2.15), "radius 1 + x", size=10, anchor="start")
    svg.text(xm(1.6), ym(-1.62), "harmonics up to 5ω = 6.5 in x", size=10, anchor="middle")
    better_hint(svg, px, y + sh + 10, pw, 150, "right", label="longer")
    y += sh + 10 + 150 + 56

    ks, cost = t["orders"], t["cost"]
    y = caption(svg, 16, y, "The order choice keeps the order of least cost",
                f"Rotor's 4 segments on [0, 2] at tol {sci(t['tol'])}: the steps planned at order k times the step "
                f"cost k² + 20k; diamonds are the first pass's candidates")
    lc = [math.log10(c) for c in cost]
    xm, ym = axes(svg, px, y, pw, 160, (ks[0], ks[-1]), (math.floor(min(lc) * 2) / 2, math.ceil(max(lc))),
                  "Taylor order k", "cost (steps × step cost)", yfmt=exp_fmt)
    svg.add(polyline([(xm(k), ym(v)) for k, v in zip(ks, lc)], SERIES[0], 2))
    for k, v, s in zip(ks, lc, t["steps"]):
        if k in t["candidates"]:
            marker(svg, "diamond", xm(k), ym(v), SERIES[1], f"order {k}: {s} steps, cost {10 ** v:.3g}")
    kept = t["second"]
    kv = lc[ks.index(kept)]
    callout(svg, xm(kept), ym(kv), f"kept: k = {kept}", dx=-20, dy=-40)
    ap = t["a_priori"]
    callout(svg, xm(ap), ym(lc[ks.index(ap)]), f"a priori: k = {ap}", dx=28, dy=-30)
    better_hint(svg, px, y, pw, 160, "right", label="cheaper", reverse=True)
    legend(svg, px + pw, y - 10, [("cost at every order", SERIES[0]), ("candidate", SERIES[1])])
    h = footnote(svg, [source(t), "model: Rotor (tests/common)"], "", [], y + 160 + 40, REPO)
    write("taylor", svg.finish(h))
    table("Taylor order choice", ["k", "steps", "cost"],
          [(k, s, f"{c:.4g}") for k, s, c in zip(ks, t["steps"], cost) if k in t["candidates"] or k == kept])


# ---------------------------------------------------------------------------------------------------------------
# The layout


def layout():
    t = load("layout")
    svg = Svg(W)
    frames, fnorm = t["frames"], t["fnorm"]
    y = caption(svg, 16, 22, "Newton's method closes the gaps between the segments' Taylor paths",
                "y'' = 3/2 y² on [0, 1], y = 4 / (a + x)², 8 lanes a = 1 … 1.875, 5 nodes: each frame is one "
                "Newton iterate in double-double")
    step, hold = 1.4, 2.8
    period = step * len(frames) + hold
    px, pw, ph = 70, 420, 250
    xm, ym = axes(svg, px, y, pw, ph, (0, 1), (0.4, 4.2), "x", "y")
    cp = clip(svg, "plot", px, y, pw, ph)
    ex = t["exact"]
    for j in range(8):
        pts = [(xm(i / 40), ym(v[j])) for i, v in enumerate(ex)]
        svg.add(polyline(pts, NEUTRAL, 1, dash="3 3"))
    for it, fr in enumerate(frames):
        a = step * it
        b = None if it == len(frames) - 1 else step * (it + 1)
        inner = [f"<g {cp}>"]
        for s, pth in enumerate(fr["paths"]):
            for j in range(8):
                pts = [(xm(x), ym(v)) for x, v in zip(pth["x"], pth["y"][j])]
                inner.append(polyline(pts, SERIES[s], 1.6))
        inner.append("</g>")
        for i, x in enumerate(t["node_x"]):
            for j in range(8):
                inner.append(dot(xm(x), ym(fr["nodes"][i][j]), INK, 2.6,
                                 f"iterate {it}, node {i}, lane {j}: y = {fr['nodes'][i][j]:.6f}"))
        label = f"iterate {it}: max |F| = {sci(fnorm[it])}"
        inner.append(f'<text x="{px + pw - 8}" y="{y + 18}" font-size="12" fill="{INK}" text-anchor="end" '
                     f'font-weight="bold" font-variant-numeric="tabular-nums">{esc(label)}</text>')
        base = 1 if it == len(frames) - 1 else 0
        group(svg, inner, shown(a, b, period), opacity=None if not STATIC else base)
    legend(svg, px + pw, y - 10, [(f"segment {s + 1}", SERIES[s]) for s in range(len(frames[0]["paths"]))])
    svg.text(px + 8, y + ph - 10, "dashed: the exact solution · dots: the node values", size=10)

    # max |F| per iterate
    qx, qw = px + pw + 80, W - (px + pw + 80) - 40
    lf = [math.log10(v) for v in fnorm] + [math.log10(t["fnorm_qd"])]
    xm2, ym2 = axes(svg, qx, y, qw, ph, (0, len(lf) - 1), (-64, 1), "Newton iterate", "max |F|", yfmt=exp_fmt,
                    xfmt=lambda v: "qd" if v == len(lf) - 1 else str(int(v)))
    pts = [(xm2(i), ym2(v)) for i, v in enumerate(lf)]
    svg.add(polyline(pts[:-1], SERIES[0], 2))
    svg.add(polyline(pts[-2:], SERIES[0], 1.5, dash="4 3"))
    for i, (x, yy) in enumerate(pts):
        shape = "diamond" if i == len(pts) - 1 else "circle"
        label = "quad-double refinement" if i == len(pts) - 1 else f"iterate {i}"
        marker(svg, shape, x, yy, SERIES[0], f"{label}: max |F| = {sci(10 ** lf[i])}")
    for it in range(len(frames)):
        a = step * it
        b = None if it == len(frames) - 1 else step * (it + 1)
        ring = (f'<circle cx="{pts[it][0]:.1f}" cy="{pts[it][1]:.1f}" r="10" fill="none" stroke="{INK}" '
                f'stroke-width="1.5"/>')
        group(svg, [ring], shown(a, b, period), opacity=None if not STATIC else (1 if it == len(frames) - 1 else 0))
    callout(svg, *pts[-1], f"qd {sci(t['fnorm_qd'])}", dx=-14, dy=-34)
    better_hint(svg, qx, y, qw, ph, "right", label="smaller", reverse=True)
    y += ph + 62

    # the unknowns and the residual's rows
    y = caption(svg, 16, y, "The unknowns and the residual's blocks",
                "z holds the left data, every node's fields and the right data; block b matches segment b's path "
                "to its target node, blocks 0 and K the end states")
    k = len(t["node_x"])
    zparts = [("left data", 8, NEUTRAL)] + [(f"node {i}", 16, SERIES[i] if i < k - 1 else NEUTRAL)
                                             for i in range(k)] + [("right data", 8, NEUTRAL)]
    fparts = [("block 0", 16, NEUTRAL)] + [(f"block {b}", 16, SERIES[b - 1]) for b in range(1, k)] + \
        [(f"block {k}", 16, NEUTRAL)]
    bx, bw = 110, W - 110 - 24
    for row, (name, parts) in enumerate([(f"z ({t['len']} unknowns)", zparts),
                                         (f"F ({t['rows']} rows)", fparts)]):
        yy = y + row * 34
        svg.text(bx - 8, yy + 15, name, size=11, fill=INK, anchor="end")
        tot = sum(p[1] for p in parts)
        x0 = bx
        for lab, n, col in parts:
            x1 = x0 + bw * n / tot
            segment(svg, x0, x1, yy, 22, col, f"{lab}: {n} rows" if row else f"{lab}: {n} unknowns",
                    label=[f"{lab} · {n}", lab, str(n)], ink="#ffffff")
            x0 = x1
    svg.text(bx, y + 84, "a node's color is the segment that starts from it; that segment's block carries the "
             "same color", size=10)
    h = footnote(svg, [source(t), "the README's usage example"], "", [], y + 92, REPO)
    write("layout", svg.finish(h))
    table("Newton on the README example", ["iterate", "max |F|"],
          [(i, sci(v)) for i, v in enumerate(fnorm)] + [("qd", sci(t["fnorm_qd"]))])


# ---------------------------------------------------------------------------------------------------------------
# Residual and derivatives


def jacobian():
    t = load("jacobian")
    n, rows, k = t["n"], t["rows"], t["k"]
    svg = Svg(W)
    y = caption(svg, 16, 22, "The exact Jacobian, one dual evaluation per column of each block",
                f"Planted (tests/common), {k} nodes: {n} unknowns, {rows} rows; a column of block b costs one "
                "evaluation of b on dual numbers, while J v is one pass per block")
    cell = 4
    px, py = 104, y + 26
    size = n * cell
    blocks = t["row_block"]
    color = lambda b: NEUTRAL if b > k else SERIES[b]
    # the evaluation order: block by block, the columns it reads in order (its inputs, then the global); the
    # target node's -I and the extra row are linear and formed without one
    ents = t["entries"]
    order = [(b, j) for b in range(k + 1) for j in range(n) if _reads(t, b, j)]
    nev = len(order)
    step = 5.6 / nev
    period = 5.6 + 0.6 + 3.0
    t_of = {bj: 0.4 + step * q for q, bj in enumerate(order)}
    lvs = [lv for _, _, lv in ents]
    lo, hi = min(lvs), max(lvs)
    op = lambda lv: 0.3 + 0.7 * (lv - lo) / (hi - lo or 1)
    svg.add(f'<rect x="{px}" y="{py}" width="{size}" height="{size}" fill="none" stroke="{GRID}"/>')
    groups = {}
    for i, j, lv in ents:
        b = blocks[i]
        key = (b, j) if b <= k and _reads(t, b, j) else None
        groups.setdefault(key, []).append(
            f'<rect x="{px + j * cell}" y="{py + i * cell}" width="{cell}" height="{cell}" fill="{color(b)}" '
            f'fill-opacity="{op(lv):.2f}"><title>J[{i}, {j}] = ±{10 ** lv:.3e}</title></rect>')
    group(svg, groups.pop(None, []))
    for key, rects in groups.items():
        group(svg, rects, shown(t_of[key], None, period))
    # tick labels: the unknowns' groups on top, the row blocks on the left
    cg = t["col_group"]
    names = ["left"] + [f"node {i}" for i in range(k)] + ["right", "ω"]
    for g, name in enumerate(names):
        js = [j for j in range(n) if cg[j] == g]
        if not js:
            continue
        x0, x1 = px + js[0] * cell, px + (js[-1] + 1) * cell
        svg.line(x0, py - 4, x0, py, INK_SOFT)
        if name != "ω" or True:
            svg.text((x0 + x1) / 2, py - 7, name, size=9 if len(js) < 9 else 10, anchor="middle")
    svg.text(px + size / 2, py - 22, "unknown (column of J)", size=12, fill=INK, anchor="middle")
    for b in range(k + 2):
        rs = [r for r in range(rows) if blocks[r] == b]
        y0, y1 = py + rs[0] * cell, py + (rs[-1] + 1) * cell
        lab = "extra row" if b > k else f"block {b}"
        svg.text(px - 6, (y0 + y1) / 2 + 4, lab, size=10, fill=color(b) if b <= k else INK_SOFT, anchor="end",
                 weight="bold" if b <= k else "normal")
    cy = py + size / 2
    lx = px - 70
    svg.add(f'<text x="{lx}" y="{cy}" font-size="12" fill="{INK}" text-anchor="middle" '
            f'transform="rotate(-90 {lx} {cy})">residual row</text>')

    # J v: a column filled block by block
    jx = px + size + 30
    svg.text(jx + cell * 2, py - 7, "J v", size=11, fill=INK, anchor="middle", weight="bold")
    for b in range(k + 1):
        rs = [r for r in range(rows) if blocks[r] == b]
        rect = (f'<rect x="{jx}" y="{py + rs[0] * cell}" width="{cell * 4}" height="{len(rs) * cell}" '
                f'fill="{color(b)}" fill-opacity="0.8"><title>J v, block {b}: one dual evaluation</title></rect>')
        group(svg, [rect], shown(0.4 + step * b, None, period))
    rs = [r for r in range(rows) if blocks[r] > k]
    svg.add(f'<rect x="{jx}" y="{py + rs[0] * cell}" width="{cell * 4}" height="{len(rs) * cell}" fill="{NEUTRAL}" '
            f'fill-opacity="0.6"/>')

    # the counter
    cx = jx + cell * 4 + 22
    # evaluation q (from 1) shows at 0.4 + step (q - 1), when its entries appear
    for q in range(nev + 1):
        a = 0.4 + step * (q - 1) if q else 0
        b = None if q == nev else 0.4 + step * q
        if STATIC and q != nev:
            continue
        txt = (f'<text x="{cx}" y="{py + 20}" font-size="12" fill="{INK}" font-variant-numeric="tabular-nums">'
               f'Jacobian: {q} of {nev}</text>')
        group(svg, [txt], shown(a, b, period))
    jt = (f'<text x="{cx}" y="{py + 40}" font-size="12" fill="{INK}" font-variant-numeric="tabular-nums">'
          f'J v: {k + 1} of {k + 1}</text>')
    group(svg, [jt], shown(0.4 + step * k, None, period))
    for q, line in enumerate(["block evaluations so far", "", "shading: log |J| by row block", "",
                              "the target nodes' −I and the", "extra row are linear: they", "show from the start"]):
        svg.text(cx, py + 64 + 14 * q, line, size=10)

    # bars: evaluations
    by = py + size + 30
    bx, bw = 150, W - 150 - 200
    mx = nev
    for r, (lab, v, col) in enumerate([("Jacobian", nev, SERIES[0]), ("J v", k + 1, SERIES[2])]):
        yy = by + r * 26
        svg.text(bx - 8, yy + 13, lab, size=11, fill=INK, anchor="end")
        segment(svg, bx, bx + bw * v / mx + 2, yy, 18, col, f"{lab}: {v} block evaluations")
        svg.text(bx + bw * v / mx + 8, yy + 13, f"{v} block evaluations", size=11, fill=INK, tabular=True)
    better_hint(svg, bx, by, bw, 50, "below", label="fewer", reverse=True)
    h = footnote(svg, [source(t), "model: Planted (tests/common), B = 1"], "", [], by + 80, REPO)
    write("jacobian", svg.finish(h))
    table("Jacobian evaluations", ["what", "block evaluations"], [("Jacobian", nev), ("J v", k + 1)])


def _reads(t, b, j):
    """whether block b reads unknown j: its inputs per block_inputs, from the column groups"""
    k = t["k"]
    g = t["col_group"][j]
    if g == k + 2:
        return True  # the global enters every block
    if b == 0:
        return g == 0
    if b == k:
        return g == k + 1
    return g == b  # segment b's source node is node b - 1, column group b


# ---------------------------------------------------------------------------------------------------------------
# Series in a parameter and continuation: Bratu's problem, lane 0 (a = 1)


def bratu_branch(theta):
    """(lambda, u'(0)) on Bratu's branch at theta: lambda = theta^2 / (2 cosh^2(theta / 4)), u'(0) = theta
    tanh(theta / 4)"""
    return theta * theta / (2 * math.cosh(theta / 4) ** 2), theta * math.tanh(theta / 4)


THETA_C = None


def theta_c():
    global THETA_C
    if THETA_C is None:
        lo, hi = 1.0, 10.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if mid * math.tanh(mid / 4) < 4:
                lo = mid
            else:
                hi = mid
        THETA_C = lo
    return THETA_C


def lower_dudx(lam):
    """u'(0) on the lower branch at lambda < lambda_c, by bisection in theta on [0, theta_c]"""
    lo, hi = 0.0, theta_c()
    for _ in range(200):
        mid = (lo + hi) / 2
        if bratu_branch(mid)[0] < lam:
            lo = mid
        else:
            hi = mid
    return bratu_branch(lo)[1]


def branch_pts(xm, ym, tmax):
    pts = []
    n = 400
    for i in range(n + 1):
        th = tmax * i / n
        lam, du = bratu_branch(th)
        pts.append((xm(lam), ym(du)))
    return pts


def series_fig():
    t = load("series")
    c, lam0 = t["coef"], t["lam0"]
    lam_c = bratu_branch(theta_c())[0]
    svg = Svg(W)
    y = caption(svg, 16, 22, "The branch's series in the parameter, from the residual's series",
                f"Bratu's problem u'' + λ e^u = 0 (lane 0) about λ = {lam0:g}, by the asymptotic numerical method "
                f"on residual_jet; shaded: within the fold's distance of λ = {lam0:g}")
    period = 8.5
    px, pw, ph = 70, 420, 260
    xm, ym = axes(svg, px, y, pw, ph, (0, 4), (0, 8), "λ", "u'(0)")
    r = lam_c - lam0
    svg.add(f'<rect x="{xm(lam0 - r):.1f}" y="{y}" width="{xm(lam0 + r) - xm(lam0 - r):.1f}" height="{ph}" '
            f'fill="{GRID}" fill-opacity="0.6"><title>|λ − {lam0:g}| below the distance to the fold, '
            f'{r:.4f}</title></rect>')
    cp = clip(svg, "sp", px, y, pw, ph)
    svg.add(f"<g {cp}>{polyline(branch_pts(xm, ym, 12), NEUTRAL, 2)}</g>")
    orders = [1, 2, 4, 8, 16, 30]
    for q, nmax in enumerate(orders):
        pts = []
        for i in range(321):
            lam = 4 * i / 320
            hh = lam - lam0
            s = sum(c[k] * hh ** k for k in range(nmax + 1))
            pts.append((xm(lam), ym(max(min(s, 40), -40))))
        a = 0.4 + 1.1 * q
        svg.add(f"<g {cp}>" + "".join([drawing(pts, SERIES[q], a, a + 0.9, period, 1.8)]) + "</g>")
    callout(svg, xm(lam_c), ym(4), f"fold λc = {lam_c:.4f}", dx=-30, dy=-50)
    callout(svg, xm(lam0), ym(c[0]), f"λ = {lam0:g}", dx=-34, dy=-34)
    legend(svg, px + pw, y - 10, [(f"N = {n}", SERIES[q]) for q, n in enumerate(orders)] + [("exact", NEUTRAL)])

    # the error of the partial sums inside the radius
    qx, qw = px + pw + 80, W - (px + pw + 80) - 40
    probes = [2.5, 3.2]
    rows = []
    xm2, ym2 = axes(svg, qx, y, qw, ph, (0, 30), (-16, 1), "order N", "|S_N − u'(0)|", yfmt=exp_fmt)
    for q, lam in enumerate(probes):
        ex = lower_dudx(lam)
        hh = lam - lam0
        pts = []
        s = 0.0
        for k in range(31):
            s += c[k] * hh ** k
            e = abs(s - ex)
            rows.append((lam, k, f"{e:.2e}")) if k in (1, 2, 4, 8, 16, 30) else None
            pts.append((xm2(k), ym2(math.log10(max(e, 1e-16)))))
        svg.add(polyline(pts, SERIES[q + 6], 2))
        for k in (1, 4, 8, 16, 30):
            marker(svg, "circle", pts[k][0], pts[k][1], SERIES[q + 6], f"λ = {lam}, N = {k}")
    legend(svg, qx + qw, y - 10, [(f"λ = {lam}", SERIES[q + 6]) for q, lam in enumerate(probes)])
    svg.text(qx + 6, y + ph - 10, "f64 floor at 1e-16", size=10)
    better_hint(svg, qx, y, qw, ph, "right", label="smaller", reverse=True)
    h = footnote(svg, [source(t), "Bratu's problem with λ the equations' parameter"],
                 "", [], y + ph + 44, REPO)
    write("series", svg.finish(h))
    table("partial sums against the closed form", ["λ", "N", "error"], rows)


def continuation_fig():
    t = load("continuation")
    svg = Svg(W)
    lam, du, tp = t["lam"], t["dudx"], t["tp"]
    n = len(lam)
    y = caption(svg, 16, 22, "Pseudo-arclength continuation through the fold",
                f"Bratu's problem (lane 0) with λ the layout's global, through Shooting::in_global: {n} points from "
                f"u = 0, then locate_fold")
    period = 9.0
    px, pw, ph = 70, W - 70 - 40, 270
    xm, ym = axes(svg, px, y, pw, ph, (0, 4), (0, 12.5), "λ", "u'(0)")
    cp = clip(svg, "cont", px, y, pw, ph)
    svg.add(f"<g {cp}>{polyline(branch_pts(xm, ym, 14.5), NEUTRAL, 2)}</g>")
    span = 6.5
    for i in range(n):
        col = SERIES[0] if tp[i] > 0 else SERIES[1]
        d = dot(xm(lam[i]), ym(du[i]), col, 3.4,
                f"point {i}: λ = {lam[i]:.6f}, u'(0) = {du[i]:.6f}, t_p = {tp[i]:+.3f}, {t['newton'][i]} corrections")
        group(svg, [d], shown(0.3 + span * i / n, None, period))
    fl, fu = t["fold"]
    fm = (f'<g>{_diamond(xm(fl), ym(fu))}<title>fold: λ = {fl:.16f}, u\'(0) = {fu:.15f}</title></g>')
    group(svg, [fm], shown(0.3 + span + 0.3, None, period))
    callout(svg, xm(fl), ym(fu), f"fold λ = {fl:.15f}", dx=-110, dy=-6)
    svg.text(xm(fl) - 114, ym(fu) + 12, f"|λ − λc| = {sci(t['fold_err'])}, {t['fold_its']} corrections", size=10,
             anchor="end")
    legend(svg, px + pw, y - 10, [("t_p > 0", SERIES[0]), ("t_p < 0, past the fold", SERIES[1]),
                                  ("closed form", NEUTRAL)])
    h = footnote(svg, [source(t), "Bratu's problem (tests/family.rs)", "exact bordered LU in double-double"], "", [],
                 y + ph + 44, REPO)
    write("continuation", svg.finish(h))
    i = next(i for i in range(n) if tp[i] < 0)
    table("continuation", ["what", "value"],
          [("points", n), ("first point past the fold", i), ("fold λ", f"{fl:.16f}"), ("|λ − λc|", sci(t["fold_err"])),
           ("fold corrections", t["fold_its"])])


def _diamond(x, y):
    r = 7
    return (f'<polygon points="{x:.1f},{y - r:.1f} {x + r:.1f},{y:.1f} {x:.1f},{y + r:.1f} {x - r:.1f},{y:.1f}" '
            f'fill="{INK}" stroke="{SURFACE}" stroke-width="2"/>')


# ---------------------------------------------------------------------------------------------------------------
# Mode matrix


def mode():
    t = load("mode")
    svg = Svg(W)
    L, om = t["L"], t["omega"]
    y = caption(svg, 16, 22, "Modes as the zeros of det M(λ), counted by the argument principle",
                f"The linear wave (tests/mode.rs): det M(λ) = ±∏ cos(L(λ ± iωk)), k = 1, 3, 5, 7, L = {L:g}, "
                f"ω = {om:g}; M is {t['n']}×{t['n']} in complex double-double")
    period = 8.0
    px, pw, ph = 70, 300, 300
    (re0, re1), (im0, im1) = t["re"], (-2.0, 2.0)
    xm, ym = axes(svg, px, y, pw, ph, (re0, re1), (im0, im1), "Re λ", "Im λ")
    zeros = []
    for n_ in range(-3, 3):
        zr = math.pi * (2 * n_ + 1) / (2 * L)
        for k in (1, 3, 5, 7):
            for s in (1, -1):
                zi = s * om * k
                if re0 <= zr <= re1 and im0 <= zi <= im1:
                    zeros.append((zr, zi))
    for zr, zi in zeros:
        marker(svg, "cross", xm(zr), ym(zi), SERIES[0], f"mode λ = {zr:.6f} {'+' if zi > 0 else '−'} {abs(zi):g} i")
    svg.line(xm(re0), ym(t["im"]), xm(re1), ym(t["im"]), SERIES[1], 1)
    rect = t["rect"] + [t["rect"][0]]
    svg.add(polyline([(xm(a), ym(b)) for a, b in rect], INK, 1.5))
    smp = t["samples"]
    for q, (a, b, arg) in enumerate(smp):
        svg.add(dot(xm(a), ym(b), INK, 2.2, f"sample {q}: λ = {a:.4f} + {b:.4f} i, arg det M = {arg:+.4f}"))
    # the moving sample
    svg.add(mover([(xm(a), ym(b)) for a, b, _ in smp + smp[:1]], SERIES[3], period))
    zr, zi = t["zero"]
    callout(svg, xm(zr), ym(zi), "Muller", dx=34, dy=-34)
    legend(svg, px + pw, y - 10, [("mode", SERIES[0]), ("contour samples", INK), ("Im λ = 0.5", SERIES[1])])

    # arg det M along the contour, unwrapped
    qx, qw = px + pw + 90, W - (px + pw + 90) - 30
    args = [s[2] for s in smp] + [smp[0][2]]
    un = [0.0]
    for a, b in zip(args, args[1:]):
        d = (b - a + math.pi) % (2 * math.pi) - math.pi
        un.append(un[-1] + d)
    top = math.ceil(max(un) / math.pi)
    pifmt = lambda v: "0" if abs(v) < 1e-9 else ("π" if v == 1 else f"{v:g}π")
    xm2, ym2 = axes(svg, qx, y, qw, ph, (0, len(un) - 1), (0, top), "contour sample",
                    "arg det M, unwrapped", yfmt=pifmt)
    pts = [(xm2(i), ym2(v / math.pi)) for i, v in enumerate(un)]
    svg.add(polyline(pts, SERIES[3], 2))
    svg.add(mover(pts, SERIES[3], period))
    callout(svg, *pts[-1], f"winding {t['turns']:.9f}", dx=-30, dy=36)
    svg.text(qx + 8, y + 16, f"{t['evals']} adaptive samples", size=10)
    y += ph + 62

    # ln|det M| along the line Im = 0.5
    y = caption(svg, 16, y, f"ln|det M| along Im λ = {t['im']:g}, against the closed form",
                "mode_logdets at 801 rates on rayon tasks, each rate's bits; it falls toward −∞ at each mode")
    lw = W - 70 - 40
    ld = t["logdet"]
    nr = t["nr"]
    xs = [re0 + (re1 - re0) * a / (nr - 1) for a in range(nr)]

    def closed(x, yv):
        s = 0.0
        for k in (1, 3, 5, 7):
            for sg in (1, -1):
                a, b = L * x, L * (yv + sg * om * k)
                s += 0.5 * math.log(math.cos(a) ** 2 + math.sinh(b) ** 2)
        return s

    cf = [closed(x, t["im"]) for x in xs]
    err = max(abs(a - b) for a, b in zip(ld, cf))
    lo = math.floor(min(ld) / 5) * 5
    hi = math.ceil(max(ld) / 5) * 5
    xm3, ym3 = axes(svg, px, y, lw, 150, (re0, re1), (lo, hi), "Re λ", "ln|det M|")
    svg.add(polyline([(xm3(x), ym3(v)) for x, v in zip(xs, ld)], SERIES[1], 2.2))
    svg.add(polyline([(xm3(x), ym3(v)) for x, v in zip(xs, cf)], NEUTRAL, 1.2, dash="4 3"))
    for n_ in (-1, 0, 1):
        zr = math.pi * (2 * n_ + 1) / (2 * L)
        i = min(range(nr), key=lambda a: abs(xs[a] - zr))
        if n_ == 0:
            callout(svg, xm3(xs[i]), ym3(ld[i]), "π/5 + 0.5i", dx=26, dy=-4)
    legend(svg, px + lw, y - 10, [("mode_logdets", SERIES[1]), ("closed form", NEUTRAL)])
    svg.text(px + 8, y + 16, f"largest difference {sci(err)}", size=10)
    h = footnote(svg, [source(t), "the linear wave (tests/mode.rs)"], "", [], y + 150 + 44, REPO)
    write("mode", svg.finish(h))
    table("mode matrix", ["what", "value"],
          [("winding", f"{t['turns']:.9f}"), ("count", t["count"]), ("evaluations", t["evals"]),
           ("Muller zero", f"{t['zero'][0]:.16f} + {t['zero'][1]:g} i"), ("max |ln|det| − closed form|", sci(err))])


# ---------------------------------------------------------------------------------------------------------------
# Periodic direction and relayout


def periodic():
    t = load("periodic")
    svg = Svg(W)
    y = caption(svg, 16, 22, "A field on the half grid, moved to more points",
                "f = sin θ + 0.3 cos 3θ − 0.2 sin 5θ, odd under θ → θ + π: 8 points on the half period hold it; "
                "the Resampler moves it to 64")
    period = 6.0
    px, pw, ph = 70, W - 70 - 40, 200
    tau = 2 * math.pi
    pi = math.pi
    xpi, ym = axes(svg, px, y, pw, ph, (0, 2), (-1.6, 1.6), "θ", "f(θ)",
                   xfmt=lambda v: "0" if v == 0 else ("π" if v == 1 else f"{v:g}π"))
    xm = lambda th: xpi(th / pi)
    f = lambda th: math.sin(th) + 0.3 * math.cos(3 * th) - 0.2 * math.sin(5 * th)
    exact = [(xm(tau * i / 400), ym(f(tau * i / 400))) for i in range(401)]
    svg.add(drawing(exact, NEUTRAL, 0.8, 2.6, period, 1.2))
    big = []
    for th, v in zip(t["theta64"], t["f64"]):
        for s in (0, 1):
            x = th + s * math.pi
            big.append(dot(xm(x), ym(v if s == 0 else -v), SERIES[1], 2.4,
                           f"θ = {x:.4f}: {v if s == 0 else -v:+.6f}" + (" (by symmetry)" if s else "")))
    group(svg, big, shown(2.8, None, period))
    for th, v in zip(t["theta8"], t["f8"]):
        marker(svg, "circle", xm(th), ym(v), SERIES[0], f"θ = {th:.4f}: {v:+.6f}")
        svg.add(f'<g opacity="0.4">{dot(xm(th + math.pi), ym(-v), SERIES[0], 4.5, "by symmetry f(θ + π) = −f(θ)")}'
                f"</g>")
    svg.line(xm(math.pi), y, xm(math.pi), y + ph, INK_SOFT)
    svg.text(xm(math.pi) + 6, y + 14, "θ = π: the half grid ends", size=10)
    callout(svg, xm(t["theta64"][40]), ym(t["f64"][40]), f"max error {sci(t['resample_err'])}", dx=30, dy=-26)
    legend(svg, px + pw, y - 10, [("the 8 points", SERIES[0]), ("resampled to 64", SERIES[1]), ("f", NEUTRAL)])
    y += ph + 62

    y = caption(svg, 16, y, "Long segments split into blocks of their own",
                "Planted on [0, 2] at order 16: split_segments at 20 steps cuts each segment at its step points; "
                "every old row keeps its bits")
    from data_graphs import Axis
    xm2 = Axis(0, 2, px, px + pw)
    for v in (0, 0.5, 1, 1.5, 2):
        svg.line(xm2(v), y, xm2(v), y + 64, GRID)
        svg.text(xm2(v), y + 64 + 15, f"{v:g}", size=11, anchor="middle", tabular=True)
    svg.line(px, y + 64, px + pw, y + 64, INK_SOFT)
    svg.text(px + pw / 2, y + 64 + 34, "x", size=12, fill=INK, anchor="middle")
    for row, (key, ekey, lab) in enumerate([("old_steps", "old_ends", "before"), ("new_steps", "new_ends", "after")]):
        yy = y + 6 + row * 30
        svg.text(px - 8, yy + 14, lab, size=11, fill=INK, anchor="end")
        for (a, b), s in zip(t[ekey], t[key]):
            col = SERIES[0] if a < 1.0 else SERIES[2]
            segment(svg, xm2(a), xm2(b), yy, 22, col, f"[{a:.3f}, {b:.3f}]: {s} steps",
                    label=[f"{s} steps", str(s)], ink="#ffffff")
    lo, ln_ = max(t["old_steps"]), max(t["new_steps"])
    svg.text(px, y + 64 + 58, f"largest block {lo} → {ln_} steps ({fmt_pct((ln_ - lo) / lo * 100)}); the new nodes "
             f"add rows, {t['old_rows']} → {t['new_rows']}", size=11, fill=INK)
    h = footnote(svg, [source(t), "model: Planted (tests/common)"], "", [], y + 64 + 64, REPO)
    write("periodic", svg.finish(h))
    table("split", ["layout", "steps per segment"], [("before", t["old_steps"]), ("after", t["new_steps"])])


# ---------------------------------------------------------------------------------------------------------------
# Fast path


def fast():
    t = load("fast")
    runs = t["runs"]
    # the exact Newton is the baseline: the neutral slot
    colors = {runs[0]["name"]: NEUTRAL}
    for q, r in enumerate(runs[1:]):
        colors[r["name"]] = SERIES[q]
    shapes = {r["name"]: s for r, s in zip(runs, ["square", "circle", "diamond", "triangle"])}
    svg = Svg(W)
    y = caption(svg, 16, 22, "The f64 Jacobian takes the same steps for a fraction of the time",
                f"Planted (B = 2, {t['n']} unknowns) from z* perturbed by 1e-3 to max |F| ≤ {sci(t['tol'])} in "
                "double-double, one thread")
    px, pw, ph = 70, W - 70 - 50, 230
    nmax = max(len(r["fnorm"]) for r in runs) - 1
    xm, ym = axes(svg, px, y, pw, ph, (0, nmax), (-32, -2), "Newton step", "max |F|", yfmt=exp_fmt)
    period = 6.0
    for q, r in enumerate(runs):
        pts = [(xm(i), ym(math.log10(v))) for i, v in enumerate(r["fnorm"])]
        a = 0.3 + 1.2 * q
        svg.add(drawing(pts, colors[r["name"]], a, a + 1.0, period, 2.0))
    from data_graphs import dodge_markers
    sp = {q: {i: (xm(i), ym(math.log10(v))) for i, v in enumerate(r["fnorm"])} for q, r in enumerate(runs)}
    sp = dodge_markers(sp, step=6.0, tol=4.0)
    for q, r in enumerate(runs):
        for i, (x, yy) in sp[q].items():
            marker(svg, shapes[r["name"]], x, yy, colors[r["name"]], f"{r['name']}, step {i}: {sci(r['fnorm'][i])}")
    quasi = runs[-1]
    callout(svg, *sp[len(runs) - 1][len(quasi["fnorm"]) - 1],
            f"{quasi['factorizations']} factorization, Broyden updates", dx=-40, dy=-40)
    legend(svg, px + pw, y - 10, [(r["name"], colors[r["name"]]) for r in runs])
    better_hint(svg, px, y, pw, ph, "right", label="smaller", reverse=True)
    y += ph + 62

    y = caption(svg, 16, y, "Time to the root", "median of 5 solves, the change against newton")
    bx, bw = 200, W - 200 - 150
    mx = max(r["ms"] for r in runs)
    base = runs[0]["ms"]
    for q, r in enumerate(runs):
        yy = y + q * 28
        svg.text(bx - 8, yy + 14, r["name"], size=11, fill=INK, anchor="end")
        segment(svg, bx, bx + bw * r["ms"] / mx + 2, yy, 20, colors[r["name"]], f"{r['name']}: {r['ms']:.1f} ms")
        note = "" if q == 0 else f" ({fmt_pct((r['ms'] - base) / base * 100)})"
        facts = r["factorizations"]
        svg.text(bx + bw * r["ms"] / mx + 8, yy + 14,
                 f"{r['ms']:.1f} ms{note} · {facts} Jacobian{'s' if facts != 1 else ''}", size=11, fill=INK,
                 tabular=True)
    hb = len(runs) * 28
    better_hint(svg, bx, y, bw, hb, "below", label="faster", reverse=True)
    h = footnote(svg, [source(t), "model: Planted (tests/common)"], t["source"]["machine"], [], y + hb + 36, REPO)
    write("fast", svg.finish(h))
    table("fast path", ["solver", "steps", "Jacobians", "ms", "change"],
          [(r["name"], len(r["fnorm"]) - 1, r["factorizations"], f"{r['ms']:.1f}",
            "" if q == 0 else fmt_pct((r["ms"] - base) / base * 100)) for q, r in enumerate(runs)])


FIGURES = {
    "lanes": lanes,
    "solvers": solvers,
    "backends": backends,
    "taylor": taylor,
    "layout": layout,
    "jacobian": jacobian,
    "series": series_fig,
    "continuation": continuation_fig,
    "mode": mode,
    "periodic": periodic,
    "fast": fast,
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(FIGURES)
    for name in names:
        out = FIGURES[name]()
        if isinstance(out, str):
            print(out)
