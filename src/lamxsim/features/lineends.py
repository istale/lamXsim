"""Line-end detection (spec section 4D): candidate definitions.

Merged, flattened geometry carries no notion of "a routing line", so a line
end has to be defined from shape alone. It also cannot be inferred from
perimeter: chopping lines into segments moves perimeter density about 3 %
while the termination count rises tenfold, because the long-edge length lost
to the cuts almost exactly cancels the end-cap length gained.

Four candidate definitions are implemented so they can be scored against
patterns whose termination count is known by construction. Each is written to
map onto an SVRF primitive, since the intent is to run these full-chip in
Calibre and only prototype them here:

    cap        -> CONVEX EDGE METAL WITH LENGTH <= w_max
    aspect     -> ... plus a flank-length ratio
    flanked    -> ... plus a requirement that the flanks run parallel
    protrusion -> METAL NOT (METAL SIZED BY -w/2 SIZED BY +w/2), area-based

Scored against eight patterns whose termination count follows from their
construction (continuous lines, segmented lines, solid plate, dummy fill,
closed ring, comb, staircase, T junctions):

    D1 cap         144 wrong   -- every side of a fill square is a "line end"
    D2 aspect        0 wrong
    D3 flanked       0 wrong
    D4 protrusion    area, not comparable

**D2 is the recommendation.** D3 costs an extra SVRF condition and buys
nothing: on 300 random Manhattan layouts carrying 7,579 terminations the two
agreed every time, because on Manhattan rings the antiparallel-flank
condition is already implied by convex-convex.

Parameter behaviour, measured:

* ``aspect`` is the knob that matters. Safe between 1.2 and 2.0. At 1.0 dummy
  fill floods the result (a square's flanks equal its cap, so its aspect is
  exactly 1); from 3.0 upward genuine short stubs start being dropped.
* ``w_max`` is flat over a wide plateau -- identical results from 1 to 20 um
  on 1-um lines -- and then flips as a step once it reaches the width of a
  wide structure. It is what separates "a routing line terminated" from "a
  power strap edge", so set it between the routing width and the strap width
  for that layer. It does not need tuning.

All operate on database units; the caller converts.
"""
from __future__ import annotations

from dataclasses import dataclass

import klayout.db as db

from .corners import _orientation, _rings


@dataclass(frozen=True)
class LineEnd:
    x: int
    y: int
    length_dbu: int
    definition: str


def _edge_walk(region: db.Region):
    """Yield per-edge geometry with convexity judged from the metal side.

    Hole rings are flipped, as in corner classification, so that convexity
    means the same thing everywhere. The consequence is deliberate: the end
    of a slot cut into metal is bounded by re-entrant corners and is therefore
    not reported as a line end. A terminated routing tip and a slot end are
    different mechanical objects, and Tan (2008) observed the former.
    """
    for poly in region.each():
        for pts, is_hole in _rings(poly):
            n = len(pts)
            if n < 4:
                continue
            s = _orientation(pts) * (-1 if is_hole else 1)

            def convex(i):
                a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
                return s * ((b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x)) > 0

            def elen(i):
                a, b = pts[i], pts[(i + 1) % n]
                return abs(b.x - a.x) + abs(b.y - a.y)   # Manhattan edges

            def evec(i):
                a, b = pts[i], pts[(i + 1) % n]
                return (b.x - a.x, b.y - a.y)

            for i in range(n):
                yield {
                    "pts": pts, "i": i, "n": n,
                    "len": elen(i),
                    "prev_len": elen((i - 1) % n),
                    "next_len": elen((i + 1) % n),
                    "prev_vec": evec((i - 1) % n),
                    "next_vec": evec((i + 1) % n),
                    "convex_start": convex(i),
                    "convex_end": convex((i + 1) % n),
                    "mid": ((pts[i].x + pts[(i + 1) % n].x) // 2,
                            (pts[i].y + pts[(i + 1) % n].y) // 2),
                }


def detect_cap(region: db.Region, w_max_dbu: int) -> list[LineEnd]:
    """D1. A short edge with a convex corner at each end.

    The direct reading of "terminated tip". Cheapest, and the closest match to
    a single SVRF primitive, but it cannot tell a line tip from any other
    short convex-convex face -- every side of an isolated fill square
    qualifies.
    """
    out = []
    for e in _edge_walk(region):
        if e["convex_start"] and e["convex_end"] and 0 < e["len"] <= w_max_dbu:
            out.append(LineEnd(*e["mid"], e["len"], "cap"))
    return out


def detect_aspect(region: db.Region, w_max_dbu: int, aspect: float = 2.0
                  ) -> list[LineEnd]:
    """D2. A cap whose two flanking edges are at least *aspect* times as long.

    Adds the elongation a "line" implies, which is what separates a routing
    tip from a dummy-fill square: on a square the flanks equal the cap, giving
    an aspect of exactly 1.
    """
    out = []
    for e in _edge_walk(region):
        if not (e["convex_start"] and e["convex_end"]):
            continue
        if not (0 < e["len"] <= w_max_dbu):
            continue
        if min(e["prev_len"], e["next_len"]) >= aspect * e["len"]:
            out.append(LineEnd(*e["mid"], e["len"], "aspect"))
    return out


def detect_flanked(region: db.Region, w_max_dbu: int, aspect: float = 2.0
                   ) -> list[LineEnd]:
    """D3. An aspect-guarded cap whose flanks run antiparallel.

    A genuine tip has its two flanks leaving in opposite directions, forming
    the sides of the conductor. Requiring that rejects short faces on a
    staircase jog, where the flanks are collinear rather than opposed.
    """
    out = []
    for e in _edge_walk(region):
        if not (e["convex_start"] and e["convex_end"]):
            continue
        if not (0 < e["len"] <= w_max_dbu):
            continue
        if min(e["prev_len"], e["next_len"]) < aspect * e["len"]:
            continue
        px, py = e["prev_vec"]
        nx, ny = e["next_vec"]
        # Antiparallel flanks: the incoming and outgoing directions oppose.
        if px * nx + py * ny < 0:
            out.append(LineEnd(*e["mid"], e["len"], "flanked"))
    return out


def detect_protrusion(region: db.Region, w_dbu: int) -> db.Region:
    """D4. Area-based: what a morphological opening removes.

    ``METAL NOT (METAL SIZED BY -w/2 SIZED BY +w/2)`` leaves the parts an
    opening cannot reconstruct -- tips, thin spurs and sharp corners. It maps
    to two SVRF SIZE operations with no edge walking at all, but it returns
    area rather than a count, and it responds to corners as well as to ends.
    """
    h = max(w_dbu // 2, 1)
    opened = region.sized(-h).sized(h)
    return region - opened


def markers(ends: list[LineEnd], size_dbu: int) -> db.Region:
    """Fixed-size markers, so a count density rides the same DENSITY scanner."""
    h = max(size_dbu // 2, 1)
    r = db.Region()
    for e in ends:
        r.insert(db.Box(e.x - h, e.y - h, e.x + h, e.y + h))
    return r


DETECTORS = {
    "cap": detect_cap,
    "aspect": detect_aspect,
    "flanked": detect_flanked,
}

#: Recommended definition and defaults (see module docstring for the scoring).
RECOMMENDED = "aspect"
DEFAULT_ASPECT = 1.5      # mid-plateau of the safe 1.2-2.0 window
DEFAULT_WMAX_RATIO = 4.0  # x minimum drawn width, unless the layer carries straps


def detect(region: db.Region, w_max_dbu: int, *, definition: str = RECOMMENDED,
           aspect: float = DEFAULT_ASPECT) -> list[LineEnd]:
    """Run the named definition with the shared signature."""
    fn = DETECTORS[definition]
    if definition == "cap":
        return fn(region, w_max_dbu)
    return fn(region, w_max_dbu, aspect)
