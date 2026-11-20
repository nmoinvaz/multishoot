"""Grid-placed SVG diagrams in the data_graphs house style, with a matching ASCII rendering.

Canonical copy: skills/diagram/diagrams.py in github.com/arriba-tools/speedy-gonzales. It draws
with data_graphs.py from the graph-data skill, so projects vendor both files side by side and edit
the canonical copies rather than a vendored one. Standard library only.

    from diagrams import Diagram
"""

import math
import os
import sys

# A vendored copy finds data_graphs.py beside it. The canonical copy finds it in graph-data.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _dir in (_HERE, os.path.join(_HERE, os.pardir, "graph-data")):
    if os.path.isfile(os.path.join(_dir, "data_graphs.py")):
        if _dir not in sys.path:
            sys.path.insert(0, _dir)
        break

from data_graphs import GRID, INK, INK_SOFT, SERIES, SURFACE, Svg, esc, text_width  # noqa: E402

FONT = 12  # Node label size.
LABEL = 11  # Edge and group label size.
PAD_X = 14  # Node padding beside its label.
BOX_H = 34  # Node height.
CPAD = 16  # Group padding on the sides and bottom.
CTOP = 34  # Group padding on top, room for its label.
MARGIN = 20
MIN_GAP = 48  # Narrowest gap between columns.
ROW_GAP = 50  # Gap between rows.
MIN_COL = 40  # Width of a column that holds no node.
LANE = 8  # Spacing between edges that share a gap.
ARROW = 8  # Arrowhead length.
ASCII_WIDTH = 100  # Widest ASCII rendering that fits a terminal without wrapping.

# Sides an edge leaves and enters by for each direction of travel.
_H = {1: ("R", "L"), -1: ("L", "R")}
_V = {1: ("B", "T"), -1: ("T", "B")}


def node_w(label):
    return round(text_width(label, FONT) + 2 * PAD_X)


def label_w(label):
    return round(text_width(label, LABEL))


class Diagram:
    """Nodes in grid cells, optional groups around them, and labeled edges between nodes."""

    def __init__(self):
        self.groups = {}
        self.nodes = {}
        self.cells = {}
        self.edges = []

    def group(self, gid, label):
        """Groups take SERIES slots in the order they're declared."""
        if gid in self.groups:
            raise ValueError(f"duplicate group {gid!r}")
        if len(self.groups) == len(SERIES) - 1:
            raise ValueError(f"more than {len(SERIES) - 1} groups, split the diagram")
        self.groups[gid] = label

    def node(self, nid, label, col, row, group=None, note=None):
        """A node in cell (col, row). The note shows on hover in the SVG."""
        if nid in self.nodes:
            raise ValueError(f"duplicate node {nid!r}")
        if col < 0 or row < 0:
            raise ValueError(f"node {nid!r} has a negative cell")
        if (col, row) in self.cells:
            raise ValueError(f"cell ({col}, {row}) already holds {self.cells[col, row]!r}")
        if group is not None and group not in self.groups:
            raise ValueError(f"node {nid!r} names unknown group {group!r}")
        self.nodes[nid] = {"label": label, "col": col, "row": row, "group": group, "note": note}
        self.cells[col, row] = nid

    def edge(self, src, dst, label=None, dashed=False, both=False):
        """Dashed for optional or runtime-only links, both for two-way ones."""
        for n in (src, dst):
            if n not in self.nodes:
                raise ValueError(f"edge names unknown node {n!r}")
        if src == dst:
            raise ValueError(f"node {src!r} can't connect to itself")
        self.edges.append({"src": src, "dst": dst, "label": label, "dashed": dashed,
                           "both": both})

    def svg(self):
        lay = _Layout(self)
        out = lay.draw()
        _warn("svg", lay.warnings)
        return out

    def ascii(self):
        """The same diagram in box-drawing characters, for a terminal or a code block."""
        lay = _Ascii(self)
        out = lay.draw()
        _warn("ascii", lay.warnings)
        return out

    def save(self, path):
        with open(path, "w") as f:
            f.write(self.svg())


def _warn(kind, msgs):
    # Warnings go to stderr so stdout carries only what the script prints.
    for m in msgs:
        print(f"diagram {kind}: {m}", file=sys.stderr)


def _sign(v):
    return (v > 0) - (v < 0)


def _arrow(e):
    return f"{e['src']} -> {e['dst']}"


class _Layout:
    """Grid validation and route shapes shared by both renderers, measured in SVG pixels."""

    def __init__(self, d):
        self._grid(d)
        self._size()
        self._place()
        self._ports()
        for r in self.routes:
            self._points(r)
        self._lanes()
        self._labels()
        self._merges(3, 4)

    def _grid(self, d):
        if not d.nodes:
            raise ValueError("the diagram has no nodes")
        self.d = d
        self.warnings = []
        self.ncols = max(n["col"] for n in d.nodes.values()) + 1
        self.nrows = max(n["row"] for n in d.nodes.values()) + 1
        self._spans()
        self.routes = [self._route(e) for e in d.edges]
        # Only a one-row or one-column diagram routes through the margin past its last cell.
        self.outer_c = sum(("col", self.ncols - 1) in r["chans"] for r in self.routes)
        self.outer_r = sum(("row", self.nrows - 1) in r["chans"] for r in self.routes)

    def _spans(self):
        d = self.d
        self.spans = {}
        for gid in d.groups:
            cells = [(n["col"], n["row"]) for n in d.nodes.values() if n["group"] == gid]
            if cells:
                cs, rs = [c for c, _ in cells], [r for _, r in cells]
                self.spans[gid] = (min(cs), max(cs), min(rs), max(rs))
        # A group's rectangle of cells may hold only its own members.
        for gid, (c0, c1, r0, r1) in self.spans.items():
            for (c, r), nid in d.cells.items():
                if c0 <= c <= c1 and r0 <= r <= r1 and d.nodes[nid]["group"] != gid:
                    raise ValueError(f"node {nid!r} sits inside group {gid!r}")
        spans = list(self.spans.items())
        for i, (ga, (a0, a1, a2, a3)) in enumerate(spans):
            for gb, (b0, b1, b2, b3) in spans[i + 1:]:
                if a0 <= b1 and b0 <= a1 and a2 <= b3 and b2 <= a3:
                    raise ValueError(f"groups {ga!r} and {gb!r} overlap")

    def _bounds(self):
        """Columns where groups start and end, then rows where they start and end."""
        s = self.spans.values()
        return ({x[0] for x in s}, {x[1] for x in s}, {x[2] for x in s}, {x[3] for x in s})

    def _fit_labels(self, base, need):
        """Widens gaps so each straight edge along a row has room for its label."""
        for r in self.routes:
            e = r["e"]
            if r["kind"] == "h" and e["label"]:
                lo, hi = sorted((self.d.nodes[e["src"]]["col"], self.d.nodes[e["dst"]]["col"]))
                have = sum(base[lo:hi]) + sum(self.col_w[lo + 1:hi])
                base[lo] += max(0, need(e) - have)

    def _size(self):
        d = self.d
        self.col_w = [MIN_COL] * self.ncols
        for n in d.nodes.values():
            self.col_w[n["col"]] = max(self.col_w[n["col"]], node_w(n["label"]))
        # A one-column group widens its column to fit the group label.
        for gid, (c0, c1, _, _) in self.spans.items():
            if c0 == c1:
                self.col_w[c0] = max(self.col_w[c0], label_w(d.groups[gid]) + 24 - 2 * CPAD)
        sc, ec, sr, er = self._bounds()
        base = [MIN_GAP] * (self.ncols - 1)
        self._fit_labels(base, lambda e: label_w(e["label"]) + 24)
        self.gaps = [base[c] + CPAD * (c in ec) + CPAD * (c + 1 in sc)
                     for c in range(self.ncols - 1)]
        self.rgaps = [ROW_GAP + CPAD * (r in er) + CTOP * (r + 1 in sr)
                      for r in range(self.nrows - 1)]
        self.rpad = CPAD * (self.ncols - 1 in ec)
        self.bpad = CPAD * (self.nrows - 1 in er)
        self.left = MARGIN + CPAD * (0 in sc)
        self.right = MARGIN + self.rpad + (16 + LANE * self.outer_c if self.outer_c else 0)
        self.top = MARGIN + CTOP * (0 in sr)
        self.bottom = MARGIN + self.bpad + (16 + LANE * self.outer_r if self.outer_r else 0)

    def _place(self):
        self.col_x = [self.left]
        for c in range(self.ncols - 1):
            self.col_x.append(self.col_x[c] + self.col_w[c] + self.gaps[c])
        self.row_y = [self.top]
        for r in range(self.nrows - 1):
            self.row_y.append(self.row_y[r] + BOX_H + self.rgaps[r])
        self.width = self.col_x[-1] + self.col_w[-1] + self.right
        self.height = self.row_y[-1] + BOX_H + self.bottom
        self.rect = {}
        for nid, n in self.d.nodes.items():
            w, c = node_w(n["label"]), n["col"]
            self.rect[nid] = (self.col_x[c] + (self.col_w[c] - w) / 2, self.row_y[n["row"]], w,
                              BOX_H)

    def _col_chan(self, k):
        """x of the gap between columns k and k + 1."""
        if k >= self.ncols - 1:
            return self.col_x[-1] + self.col_w[-1] + self.rpad + 12 + LANE * (self.outer_c - 1) / 2
        return (self.col_x[k] + self.col_w[k] + self.col_x[k + 1]) / 2

    def _row_chan(self, k):
        """y of the gap between rows k and k + 1."""
        if k >= self.nrows - 1:
            return self.row_y[-1] + BOX_H + self.bpad + 12 + LANE * (self.outer_r - 1) / 2
        return (self.row_y[k] + BOX_H + self.row_y[k + 1]) / 2

    def _route(self, e):
        """Picks the route shape with the fewest bends that crosses no node."""
        cells = self.d.cells
        s, t = self.d.nodes[e["src"]], self.d.nodes[e["dst"]]
        c1, r1, c2, r2 = s["col"], s["row"], t["col"], t["row"]
        dc, dr = _sign(c2 - c1), _sign(r2 - r1)

        def row_clear(r, ca, cb):
            return all((c, r) not in cells for c in range(min(ca, cb) + 1, max(ca, cb)))

        def col_clear(c, ra, rb):
            return all((c, r) not in cells for r in range(min(ra, rb) + 1, max(ra, rb)))

        def route(kind, ss, ds, chans=()):
            return {"e": e, "kind": kind, "ss": ss, "ds": ds, "chans": list(chans)}

        if dr == 0:
            if row_clear(r1, c1, c2):
                return route("h", *_H[dc])
            # Hop over the nodes between through the gap above, or below on the top row.
            side, k = ("T", r1 - 1) if r1 > 0 else ("B", r1)
            return route("zrow", side, side, [("row", k)])
        if dc == 0:
            if col_clear(c1, r1, r2):
                return route("v", *_V[dr])
            # A one-column diagram routes through the right margin.
            side, k = ("R", c1) if c1 < self.ncols - 1 or c1 == 0 else ("L", c1 - 1)
            return route("zcol", side, side, [("col", k)])
        if row_clear(r1, c1, c2) and (c2, r1) not in cells and col_clear(c2, r1, r2):
            return route("hv", _H[dc][0], _V[dr][1])
        if col_clear(c1, r1, r2) and (c1, r2) not in cells and row_clear(r2, c1, c2):
            return route("vh", _V[dr][0], _H[dc][1])
        # Z shapes turn in the column gap beside the target, then beside the source.
        if row_clear(r1, c1, c2):
            return route("zcol", _H[dc][0], _H[dc][1], [("col", min(c2 - dc, c2))])
        if row_clear(r2, c1, c2):
            return route("zcol", _H[dc][0], _H[dc][1], [("col", min(c1, c1 + dc))])
        if col_clear(c1, r1, r2):
            return route("zrow", _V[dr][0], _V[dr][1], [("row", min(r2 - dr, r2))])
        if col_clear(c2, r1, r2):
            return route("zrow", _V[dr][0], _V[dr][1], [("row", min(r1, r1 + dr))])
        # Gaps never hold nodes, so a route along a row gap then a column gap is always clear.
        return route("gap", _V[dr][0], _H[dc][1],
                     [("row", min(r1, r1 + dr)), ("col", min(c2 - dc, c2))])

    def _ports(self):
        """Spreads the edges that share a node side evenly along it."""
        nodes = self.d.nodes
        sides = {}
        for i, r in enumerate(self.routes):
            src, dst = r["e"]["src"], r["e"]["dst"]
            sides.setdefault((src, r["ss"]), []).append((i, "sp", dst))
            sides.setdefault((dst, r["ds"]), []).append((i, "dp", src))
        for (nid, side), ends in sides.items():
            # Order ends by where the other node sits so neighboring edges don't cross.
            axis = "row" if side in "LR" else "col"
            ends.sort(key=lambda end: (nodes[end[2]][axis], end[0]))
            x, y, w, h = self.rect[nid]
            for k, (i, which, _) in enumerate(ends):
                f = (k + 1) / (len(ends) + 1)
                self.routes[i][which] = {"L": (x, y + f * h), "R": (x + w, y + f * h),
                                         "T": (x + f * w, y), "B": (x + f * w, y + h)}[side]

    def _points(self, r):
        (sx, sy), (dx, dy) = r["sp"], r["dp"]
        kind = r["kind"]
        if kind == "h":
            y = (sy + dy) / 2
            r["pts"] = [(sx, y), (dx, y)]
        elif kind == "v":
            a, b = self.rect[r["e"]["src"]], self.rect[r["e"]["dst"]]
            lo, hi = max(a[0], b[0]) + 6, min(a[0] + a[2], b[0] + b[2]) - 6
            x = min(max((sx + dx) / 2, lo), hi)
            r["pts"] = [(x, sy), (x, dy)]
        elif kind == "hv":
            r["pts"] = [(sx, sy), (dx, sy), (dx, dy)]
        elif kind == "vh":
            r["pts"] = [(sx, sy), (sx, dy), (dx, dy)]
        elif kind == "zcol":
            gx = self._col_chan(r["chans"][0][1])
            r["pts"] = [(sx, sy), (gx, sy), (gx, dy), (dx, dy)]
        elif kind == "zrow":
            gy = self._row_chan(r["chans"][0][1])
            r["pts"] = [(sx, sy), (sx, gy), (dx, gy), (dx, dy)]
        else:
            gy, gx = self._row_chan(r["chans"][0][1]), self._col_chan(r["chans"][1][1])
            r["pts"] = [(sx, sy), (sx, gy), (gx, gy), (gx, dy), (dx, dy)]
        # Each channel entry runs along segment j, from point j to point j + 1.
        r["segs"] = [(c, j) for c, j in zip(r["chans"], {"zcol": [1], "zrow": [1],
                                                          "gap": [1, 2]}.get(kind, []))]

    def _lanes(self):
        """Gives edges that run along the same gap and overlap their own lane."""
        runs = {}
        for i, r in enumerate(self.routes):
            for chan, j in r["segs"]:
                (ax, ay), (bx, by) = r["pts"][j], r["pts"][j + 1]
                lo, hi = (min(ay, by), max(ay, by)) if chan[0] == "col" else (min(ax, bx),
                                                                              max(ax, bx))
                runs.setdefault(chan, []).append((lo, hi, i, j))
        for chan, segs in runs.items():
            segs.sort()
            ends, lane_of = [], []
            for lo, hi, _, _ in segs:
                lane = next((k for k, end in enumerate(ends) if end + LANE <= lo), len(ends))
                if lane == len(ends):
                    ends.append(hi)
                else:
                    ends[lane] = hi
                lane_of.append(lane)
            for (_, _, i, j), lane in zip(segs, lane_of):
                off = (lane - (len(ends) - 1) / 2) * LANE
                pts = self.routes[i]["pts"]
                for p in (j, j + 1):
                    x, y = pts[p]
                    pts[p] = (x + off, y) if chan[0] == "col" else (x, y + off)

    def _labels(self):
        """Places each edge label at the candidate spot with the fewest collisions."""
        d = self.d
        # Nodes and group labels must never sit under an edge label.
        solid = [(x - 2, y - 2, w + 4, h + 4) for x, y, w, h in self.rect.values()]
        for gid, (c0, _, r0, _) in self.spans.items():
            solid.append((self.col_x[c0] - CPAD + 12, self.row_y[r0] - CTOP + 9,
                          label_w(d.groups[gid]) * 1.05, LABEL + 2))
        segs = [(r["pts"][j], r["pts"][j + 1]) for r in self.routes
                for j in range(len(r["pts"]) - 1)]
        self.labels = []
        for r in self.routes:
            text = r["e"]["label"]
            if not text:
                continue
            scored = [(c[0] + self._cost(c[1], solid, segs), c)
                      for c in self._candidates(r["pts"], label_w(text))]
            cost, (_, box, x, y, anchor) = min(scored, key=lambda s: s[0])
            if cost >= 100:
                self.warnings.append(f'label "{text}" on {_arrow(r["e"])} overlaps a node or '
                                     f"another label. Move a node or shorten the label.")
            solid.append(box)
            self.labels.append((x, y, anchor, text))
            # A label that spills past the right or bottom edge grows the canvas.
            self.width = max(self.width, box[0] + box[2] + 8)
            self.height = max(self.height, box[1] + box[3] + 8)

    def _candidates(self, pts, lw):
        """(penalty, box, x, y, anchor) spots above, below, or beside each run of an edge."""
        out = []
        runs = sorted(((pts[j], pts[j + 1]) for j in range(len(pts) - 1)),
                      key=lambda s: -abs(s[1][0] - s[0][0]) - abs(s[1][1] - s[0][1]))
        for a, b in runs:
            if abs(a[1] - b[1]) < 0.5:
                lo, hi = sorted((a[0], b[0]))
                short = 4 if hi - lo < lw + 8 else 0
                for t in (0.5, 0.3, 0.7, 0.15, 0.85):
                    x = min(max(lo + t * (hi - lo), lo + lw / 2 + 4), hi - lw / 2 - 4)
                    if short:
                        x = (lo + hi) / 2
                    for y in (a[1] - 5, a[1] + 14):
                        out.append((short, (x - lw / 2, y - 9, lw, LABEL + 1), x, y, "middle"))
            else:
                lo, hi = sorted((a[1], b[1]))
                short = 4 if hi - lo < 24 else 0
                for t in (0.5, 0.3, 0.7):
                    y = lo + t * (hi - lo) + 4
                    out.append((short + 1, (a[0] + 6, y - 9, lw, LABEL + 1), a[0] + 6, y, "start"))
                    out.append((short + 1, (a[0] - 6 - lw, y - 9, lw, LABEL + 1), a[0] - 6, y,
                                "end"))
        # Earlier spots win ties, which keeps labels central on their longest run.
        return [(p + k * 0.01, *rest) for k, (p, *rest) in enumerate(out)]

    def _cost(self, box, solid, segs):
        bx, by, bw, bh = box
        if bx < 2 or by < 2:
            return 100
        cost = 5 * (bx + bw > self.width - 2 or by + bh > self.height - 2)
        hits = sum(bx < x + w and x < bx + bw and by < y + h and y < by + bh
                   for x, y, w, h in solid)
        cross = 0
        for (ax, ay), (cx, cy) in segs:
            if abs(ay - cy) < 0.5:
                cross += by <= ay <= by + bh and min(ax, cx) <= bx + bw and max(ax, cx) >= bx
            else:
                cross += bx <= ax <= bx + bw and min(ay, cy) <= by + bh and max(ay, cy) >= by
        return cost + 100 * hits + 3 * cross

    def _merges(self, tol, reach):
        """Warns about two edges whose runs line up end to end, which read as one edge."""
        runs = []
        for i, r in enumerate(self.routes):
            for (ax, ay), (bx, by) in zip(r["pts"], r["pts"][1:]):
                if (ax, ay) == (bx, by):
                    continue
                if abs(ay - by) < 0.5:
                    runs.append((i, 0, ay, min(ax, bx), max(ax, bx)))
                else:
                    runs.append((i, 1, ax, min(ay, by), max(ay, by)))
        pairs = set()
        for p, (i, o, c, lo, hi) in enumerate(runs):
            for j, o2, c2, lo2, hi2 in runs[p + 1:]:
                if (i != j and o == o2 and abs(c - c2) <= tol and lo <= hi2 + reach
                        and lo2 <= hi + reach):
                    pairs.add((min(i, j), max(i, j)))
        for i, j in sorted(pairs):
            self.warnings.append(f"edges {_arrow(self.routes[i]['e'])} and "
                                 f"{_arrow(self.routes[j]['e'])} line up and read as one. "
                                 f"Move a node so they don't.")

    def draw(self):
        d = self.d
        svg = Svg(round(self.width))
        slot = {gid: i for i, gid in enumerate(d.groups)}
        for gid, (c0, c1, r0, r1) in self.spans.items():
            x0, y0 = self.col_x[c0] - CPAD, self.row_y[r0] - CTOP
            x1, y1 = self.col_x[c1] + self.col_w[c1] + CPAD, self.row_y[r1] + BOX_H + CPAD
            svg.add(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" '
                    f'height="{y1 - y0:.1f}" rx="10" fill="{GRID}" fill-opacity="0.45"/>')
            svg.text(x0 + 12, y0 + 20, d.groups[gid], size=LABEL, fill=INK_SOFT, weight="600")
        for r in self.routes:
            _edge(svg, r["pts"], r["e"]["dashed"], r["e"]["both"])
        for nid, n in d.nodes.items():
            x, y, w, h = self.rect[nid]
            color = SERIES[slot[n["group"]]] if n["group"] else SERIES[-1]
            if n["note"]:
                svg.add(f"<g><title>{esc(n['note'])}</title>")
            svg.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="6" '
                    f'fill="{SURFACE}" stroke="{color}" stroke-width="1.5"/>')
            svg.text(x + w / 2, y + h / 2 + 4, n["label"], size=FONT, fill=INK, anchor="middle")
            if n["note"]:
                svg.add("</g>")
        # Labels go last so no line or node covers them.
        for x, y, anchor, text in self.labels:
            svg.text(x, y, text, size=LABEL, fill=INK_SOFT, anchor=anchor)
        return svg.finish(round(self.height))


def _base(a, b):
    """Arrowhead base for a tip at b on a run from a, and the run's unit direction."""
    d = math.hypot(b[0] - a[0], b[1] - a[1]) or 1
    ux, uy = (b[0] - a[0]) / d, (b[1] - a[1]) / d
    return (b[0] - ux * ARROW, b[1] - uy * ARROW), (ux, uy)


def _edge(svg, pts, dashed=False, both=False):
    heads = [(pts[-2], pts[-1])] + ([(pts[1], pts[0])] if both else [])
    line = list(pts)
    # The line stops at each arrowhead base so its cap can't poke past the tip.
    line[-1] = _base(pts[-2], pts[-1])[0]
    if both:
        line[0] = _base(pts[1], pts[0])[0]
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in line)
    svg.add(f'<polyline points="{coords}" fill="none" stroke="{INK_SOFT}" stroke-width="1.25" '
            f'stroke-linejoin="round"{dash}/>')
    for a, b in heads:
        (bx, by), (ux, uy) = _base(a, b)
        px, py = -uy * 4, ux * 4
        svg.add(f'<polygon points="{b[0]:.1f},{b[1]:.1f} {bx + px:.1f},{by + py:.1f} '
                f'{bx - px:.1f},{by - py:.1f}" fill="{INK_SOFT}"/>')


# Line cells hold direction bits toward their neighbors.
_U, _D, _L, _R = 1, 2, 4, 8
_GLYPH = {_L | _R: "─", _U | _D: "│", _D | _R: "╭", _D | _L: "╮", _U | _R: "╰", _U | _L: "╯",
          _U | _D | _R: "├", _U | _D | _L: "┤", _L | _R | _D: "┬", _L | _R | _U: "┴",
          _U | _D | _L | _R: "┼", _L: "─", _R: "─", _U: "│", _D: "│"}
_DASHED = {_L | _R: "┄", _U | _D: "┆", _L: "┄", _R: "┄", _U: "┆", _D: "┆"}
_TEE = {"R": "├", "L": "┤", "T": "┴", "B": "┬"}
# Arrowheads point into the node, so their direction comes from the side they enter.
_HEAD = {"L": "▶", "R": "◀", "T": "▼", "B": "▲"}


class _Ascii(_Layout):
    """The SVG layout's grid and route shapes, measured in character cells."""

    GSIDE, GTOP, GBOT = 2, 2, 2  # Group border plus one blank cell inside it.
    MIN_GAP, MIN_RGAP = 4, 3
    SLACK = 40  # Room around the drawing for labels, trimmed away at the end.

    def __init__(self, d):
        self._grid(d)
        self._grid_lanes()
        self._size()
        self._place()
        self._ports()
        for r in self.routes:
            self._points(r)
        self._merges(0, 1)

    def _grid_lanes(self):
        """Lanes from grid positions, where 2i is row or column i and 2i + 1 the gap after."""
        nodes = self.d.nodes
        runs = {}
        for i, r in enumerate(self.routes):
            s, t = nodes[r["e"]["src"]], nodes[r["e"]["dst"]]
            if r["kind"] == "zcol":
                spans = [(r["chans"][0], 2 * s["row"], 2 * t["row"])]
            elif r["kind"] == "zrow":
                spans = [(r["chans"][0], 2 * s["col"], 2 * t["col"])]
            elif r["kind"] == "gap":
                rc, cc = r["chans"]
                spans = [(rc, 2 * s["col"], 2 * cc[1] + 1), (cc, 2 * rc[1] + 1, 2 * t["row"])]
            else:
                spans = []
            r["lanes"] = [0] * len(spans)
            for k, (chan, a, b) in enumerate(spans):
                runs.setdefault(chan, []).append((min(a, b), max(a, b), i, k))
        self.nlanes = {}
        for chan, segs in runs.items():
            ends = []
            # Runs that touch end to end still share a cell, so they take separate lanes.
            for lo, hi, i, k in sorted(segs):
                lane = next((j for j, end in enumerate(ends) if end < lo), len(ends))
                if lane == len(ends):
                    ends.append(hi)
                else:
                    ends[lane] = hi
                self.routes[i]["lanes"][k] = lane
            self.nlanes[chan] = len(ends)

    def _size(self):
        d = self.d
        side_n = {}
        for r in self.routes:
            for nid, side in ((r["e"]["src"], r["ss"]), (r["e"]["dst"], r["ds"])):
                side_n[nid, side] = side_n.get((nid, side), 0) + 1
        # A node grows taller so each edge on its left or right side gets its own line.
        self.box = {nid: (len(n["label"]) + 4,
                          2 + max(1, side_n.get((nid, "L"), 0), side_n.get((nid, "R"), 0)))
                    for nid, n in d.nodes.items()}
        self.col_w = [2] * self.ncols
        self.row_h = [3] * self.nrows
        for nid, n in d.nodes.items():
            w, h = self.box[nid]
            self.col_w[n["col"]] = max(self.col_w[n["col"]], w)
            self.row_h[n["row"]] = max(self.row_h[n["row"]], h)
        for gid, (c0, c1, _, _) in self.spans.items():
            if c0 == c1:
                self.col_w[c0] = max(self.col_w[c0], len(d.groups[gid]) + 6 - 2 * self.GSIDE)
        sc, ec, sr, er = self._bounds()
        # A label beside a vertical edge would hit the border of a group ending at its column.
        for r in self.routes:
            c = d.nodes[r["e"]["src"]]["col"]
            if r["kind"] == "v" and r["e"]["label"] and c in ec:
                self.col_w[c] = max(self.col_w[c], 2 * (len(r["e"]["label"]) + 2))
        base = [max(self.MIN_GAP, 2 * self.nlanes.get(("col", c), 0) + 3)
                for c in range(self.ncols - 1)]
        self._fit_labels(base, lambda e: len(e["label"]) + 4)
        self.gaps = [base[c] + self.GSIDE * (c in ec) + self.GSIDE * (c + 1 in sc)
                     for c in range(self.ncols - 1)]
        self.rgaps = [max(self.MIN_RGAP, self.nlanes.get(("row", r), 0) + 2)
                      + self.GBOT * (r in er) + self.GTOP * (r + 1 in sr)
                      for r in range(self.nrows - 1)]
        self.rpad = self.GSIDE * (self.ncols - 1 in ec)
        self.bpad = self.GBOT * (self.nrows - 1 in er)
        nc = self.nlanes.get(("col", self.ncols - 1), 0)
        nr = self.nlanes.get(("row", self.nrows - 1), 0)
        self.left = self.GSIDE * (0 in sc)
        self.right = self.rpad + (2 * nc + 1 if nc else 0)
        self.top = self.GTOP * (0 in sr)
        self.bottom = self.bpad + (nr + 1 if nr else 0)

    def _place(self):
        self.col_x = [self.left]
        for c in range(self.ncols - 1):
            self.col_x.append(self.col_x[c] + self.col_w[c] + self.gaps[c])
        self.row_y = [self.top]
        for r in range(self.nrows - 1):
            self.row_y.append(self.row_y[r] + self.row_h[r] + self.rgaps[r])
        self.width = self.col_x[-1] + self.col_w[-1] + self.right
        self.height = self.row_y[-1] + self.row_h[-1] + self.bottom
        self.rect = {}
        for nid, n in self.d.nodes.items():
            (w, h), c, r = self.box[nid], n["col"], n["row"]
            self.rect[nid] = (self.col_x[c] + (self.col_w[c] - w) // 2,
                              self.row_y[r] + (self.row_h[r] - h) // 2, w, h)

    def _col_chan(self, k, lane=0):
        if k >= self.ncols - 1:
            # One cell past any group border, so the edge turns visibly before it runs.
            return self.col_x[-1] + self.col_w[-1] + self.rpad + 1 + 2 * lane
        n = self.nlanes.get(("col", k), 1)
        mid = (self.col_x[k] + self.col_w[k] + self.col_x[k + 1] - 1) // 2
        return mid - (n - 1) + 2 * lane

    def _row_chan(self, k, lane=0):
        if k >= self.nrows - 1:
            return self.row_y[-1] + self.row_h[-1] + self.bpad + 1 + lane
        n = self.nlanes.get(("row", k), 1)
        mid = (self.row_y[k] + self.row_h[k] + self.row_y[k + 1] - 1) // 2
        return mid - (n - 1) // 2 + lane

    def _ports(self):
        """Gives each edge end its own border cell, spread along the side like the SVG."""
        nodes = self.d.nodes
        sides = {}
        for i, r in enumerate(self.routes):
            src, dst = r["e"]["src"], r["e"]["dst"]
            sides.setdefault((src, r["ss"]), []).append((i, "sp", dst))
            sides.setdefault((dst, r["ds"]), []).append((i, "dp", src))
        for (nid, side), ends in sides.items():
            axis = "row" if side in "LR" else "col"
            ends.sort(key=lambda end: (nodes[end[2]][axis], end[0]))
            x, y, w, h = self.rect[nid]
            for k, (i, which, _) in enumerate(ends):
                f = (k + 0.5) / len(ends)
                py, px = y + 1 + int(f * (h - 2)), x + 1 + int(f * (w - 2))
                # The border cell and the first cell outside it.
                self.routes[i][which] = {"L": ((x, py), (x - 1, py)),
                                         "R": ((x + w - 1, py), (x + w, py)),
                                         "T": ((px, y), (px, y - 1)),
                                         "B": ((px, y + h - 1), (px, y + h))}[side]

    def _points(self, r):
        (sb, (sx, sy)), (db, (dx, dy)) = r["sp"], r["dp"]
        kind, lanes = r["kind"], r["lanes"]
        if kind in ("h", "v"):
            # Both ends of a straight edge share one line, inside the target's side.
            x, y, w, h = self.rect[r["e"]["dst"]]
            if kind == "h":
                sy = dy = min(max(sy, y + 1), y + h - 2)
                r["sp"], r["dp"] = ((sb[0], sy), (sx, sy)), ((db[0], dy), (dx, dy))
            else:
                sx = dx = min(max(sx, x + 1), x + w - 2)
                r["sp"], r["dp"] = ((sx, sb[1]), (sx, sy)), ((dx, db[1]), (dx, dy))
            r["pts"] = [(sx, sy), (dx, dy)]
        elif kind == "hv":
            r["pts"] = [(sx, sy), (dx, sy), (dx, dy)]
        elif kind == "vh":
            r["pts"] = [(sx, sy), (sx, dy), (dx, dy)]
        elif kind == "zcol":
            gx = self._col_chan(r["chans"][0][1], lanes[0])
            r["pts"] = [(sx, sy), (gx, sy), (gx, dy), (dx, dy)]
        elif kind == "zrow":
            gy = self._row_chan(r["chans"][0][1], lanes[0])
            r["pts"] = [(sx, sy), (sx, gy), (dx, gy), (dx, dy)]
        else:
            gy = self._row_chan(r["chans"][0][1], lanes[0])
            gx = self._col_chan(r["chans"][1][1], lanes[1])
            r["pts"] = [(sx, sy), (sx, gy), (gx, gy), (gx, dy), (dx, dy)]

    def draw(self):
        d, P = self.d, self.SLACK
        W, H = self.width + 2 * P, self.height + 4
        chars = [[" "] * W for _ in range(H)]
        bits = [[0] * W for _ in range(H)]
        solid = [[False] * W for _ in range(H)]

        def put(x, y, s):
            for k, ch in enumerate(s):
                if 0 <= x + k + P < W and 0 <= y + 2 < H:
                    chars[y + 2][x + k + P] = ch

        def taken(x, y, n):
            # Cells off the canvas count as taken, so labels stay on it.
            return sum(not (0 <= xx + P < W and 0 <= y + 2 < H) or chars[y + 2][xx + P] != " "
                       for xx in range(x, x + n))

        for gid, (c0, c1, r0, r1) in self.spans.items():
            x0, y0 = self.col_x[c0] - self.GSIDE, self.row_y[r0] - self.GTOP
            x1 = self.col_x[c1] + self.col_w[c1] + self.GSIDE - 1
            y1 = self.row_y[r1] + self.row_h[r1] + self.GBOT - 1
            put(x0, y0, "┌" + "╌" * (x1 - x0 - 1) + "┐")
            put(x0, y1, "└" + "╌" * (x1 - x0 - 1) + "┘")
            for y in range(y0 + 1, y1):
                put(x0, y, "╎")
                put(x1, y, "╎")
            put(x0 + 2, y0, f" {d.groups[gid]} ")
        for nid, n in d.nodes.items():
            x, y, w, h = self.rect[nid]
            put(x, y, "┌" + "─" * (w - 2) + "┐")
            put(x, y + h - 1, "└" + "─" * (w - 2) + "┘")
            for yy in range(y + 1, y + h - 1):
                put(x, yy, "│")
                put(x + w - 1, yy, "│")
            put(x + 2, y + h // 2, n["label"])

        # A cell any solid edge crosses draws solid, so dashes show only where an edge runs alone.
        for r in self.routes:
            dashed = r["e"]["dashed"]
            for (ax, ay), (bx, by) in zip(r["pts"], r["pts"][1:]):
                horiz = ay == by
                a, b = (ax, bx) if horiz else (ay, by)
                step = _sign(b - a) or 1
                fwd, back = (_R, _L) if horiz else (_D, _U)
                if step < 0:
                    fwd, back = back, fwd
                for v in range(a, b + step, step):
                    x, y = (v, ay) if horiz else (ax, v)
                    cell = (y + 2, x + P)
                    bits[cell[0]][cell[1]] |= (fwd if v != b else 0) | (back if v != a else 0)
                    solid[cell[0]][cell[1]] |= not dashed
        for y in range(H):
            for x in range(W):
                if bits[y][x]:
                    glyphs = _GLYPH if solid[y][x] else {**_GLYPH, **_DASHED}
                    chars[y][x] = glyphs[bits[y][x]]
        for r in self.routes:
            (sb, so), (_, do) = r["sp"], r["dp"]
            put(*do, _HEAD[r["ds"]])
            if r["e"]["both"]:
                put(*so, _HEAD[r["ss"]])
            else:
                put(*sb, _TEE[r["ss"]])

        for r in self.routes:
            text = r["e"]["label"]
            if not text:
                continue
            n = len(text)

            def cost(spot):
                # Overlaps cost the most. Touching something at either end costs a little.
                p, x, y = spot
                return p + 10 * taken(x, y, n) + taken(x - 1, y, 1) + taken(x + n, y, 1)

            _, x, y = min(self._spots(r["pts"], n), key=cost)
            if taken(x, y, n):
                self.warnings.append(f'label "{text}" on {_arrow(r["e"])} overwrites part of the '
                                     f"drawing. Move a node or shorten the label.")
            put(x, y, text)

        lines = ["".join(row).rstrip() for row in chars]
        while lines and not lines[-1]:
            lines.pop()
        while lines and not lines[0]:
            lines.pop(0)
        indent = min(len(s) - len(s.lstrip(" ")) for s in lines if s)
        lines = [s[indent:] for s in lines]
        width = max(map(len, lines))
        if width > ASCII_WIDTH:
            self.warnings.append(f"{width} columns wide, over {ASCII_WIDTH}. Move nodes into "
                                 f"more rows or shorten labels.")
        return "\n".join(lines) + "\n"

    def _spots(self, pts, n):
        """(penalty, x, y) spots above, below, or beside each run, longest run first."""
        out = []
        runs = sorted(zip(pts, pts[1:]),
                      key=lambda s: -abs(s[1][0] - s[0][0]) - abs(s[1][1] - s[0][1]))
        for (ax, ay), (bx, by) in runs:
            if ay == by:
                lo, hi = sorted((ax, bx))
                for t in (0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8):
                    x = min(max(round(lo + t * (hi - lo) - n / 2), lo + 1), hi - n)
                    out += [(0, x, ay - 1), (0, x, ay + 1)]
            else:
                lo, hi = sorted((ay, by))
                for t in (0.5, 0.3, 0.7):
                    y = round(lo + t * (hi - lo))
                    out += [(1, ax + 2, y), (1, ax - 1 - n, y)]
        # Earlier spots win ties, which keeps labels central on their longest run.
        return [(p + k * 0.01, x, y) for k, (p, x, y) in enumerate(out)]
