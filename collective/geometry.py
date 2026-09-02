"""Window features: the analysis grid and everything measured on it.

Consolidated from ``features/grid.py``, ``features/corners.py``, ``features/lineends.py``, ``features/geometry.py``, ``features/orientation.py``, ``features/structures.py``, ``features/vias.py``, ``features/gradient.py``, ``features/crosslayer.py``, ``features/bump_relative.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import klayout.db as db
import numpy as np
from .foundation import EvidenceClass
from .layout import BBox, LayerSpec, LayoutReader


# ----------------------------------------------------------------------
# features/grid.py
# ----------------------------------------------------------------------
"""Multi-scale analysis grids in physical units (spec section 6).

Grids are defined in micrometres, never in pixels, and every cell knows the
scale it came from. Non-overlapping grids are the default: overlapping
windows inflate the apparent sample count without adding independent
information, which is exactly the failure mode the spatial null model
(spec section 15) exists to catch.
"""
@dataclass(frozen=True)
class GridCell:
    cell_id: int
    x_center: float
    y_center: float
    x0: float
    y0: float
    x1: float
    y1: float
    scale_um: float
    row: int
    col: int

    @property
    def area_um2(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)


@dataclass(frozen=True)
class Grid:
    scale_um: float
    stride_um: float
    n_rows: int
    n_cols: int
    cells: tuple[GridCell, ...]
    bbox: BBox

    def __len__(self) -> int:
        return len(self.cells)

    @property
    def overlapping(self) -> bool:
        return self.stride_um < self.scale_um

    def centers(self) -> np.ndarray:
        return np.array([(c.x_center, c.y_center) for c in self.cells], dtype=float)

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "cell_id": np.array([c.cell_id for c in self.cells], dtype=np.int64),
            "x_um": np.array([c.x_center for c in self.cells], dtype=float),
            "y_um": np.array([c.y_center for c in self.cells], dtype=float),
            "row": np.array([c.row for c in self.cells], dtype=np.int32),
            "col": np.array([c.col for c in self.cells], dtype=np.int32),
            "scale_um": np.full(len(self.cells), self.scale_um, dtype=float),
        }


def build_grid(bbox: BBox, scale_um: float, stride_um: float | None = None,
               align: str = "origin") -> Grid:
    """Regular analysis grid covering *bbox*.

    ``stride_um`` defaults to ``scale_um`` (non-overlapping). Partial cells at
    the far edge are dropped rather than analysed with a smaller area, so that
    every cell of a given scale represents the same physical footprint and
    densities stay comparable across the die.
    """
    if scale_um <= 0:
        raise ValueError("scale_um must be positive")
    stride = float(scale_um if stride_um is None else stride_um)
    if stride <= 0:
        raise ValueError("stride_um must be positive")

    if align == "origin":
        x_start, y_start = bbox.xmin, bbox.ymin
    elif align == "center":
        nx = int((bbox.width - scale_um) // stride) + 1
        ny = int((bbox.height - scale_um) // stride) + 1
        x_start = bbox.xmin + (bbox.width - ((nx - 1) * stride + scale_um)) / 2
        y_start = bbox.ymin + (bbox.height - ((ny - 1) * stride + scale_um)) / 2
    else:
        raise ValueError(f"unknown align={align!r}")

    n_cols = int((bbox.width - scale_um) // stride) + 1
    n_rows = int((bbox.height - scale_um) // stride) + 1
    if n_cols < 1 or n_rows < 1:
        raise ValueError(
            f"scale {scale_um}um does not fit in bbox {bbox.width}x{bbox.height}um"
        )

    cells = []
    cid = 0
    for r in range(n_rows):
        y0 = y_start + r * stride
        for c in range(n_cols):
            x0 = x_start + c * stride
            cells.append(GridCell(
                cell_id=cid, x_center=x0 + scale_um / 2, y_center=y0 + scale_um / 2,
                x0=x0, y0=y0, x1=x0 + scale_um, y1=y0 + scale_um,
                scale_um=scale_um, row=r, col=c,
            ))
            cid += 1
    return Grid(scale_um=scale_um, stride_um=stride, n_rows=n_rows,
                n_cols=n_cols, cells=tuple(cells), bbox=bbox)


def build_multiscale(bbox: BBox, scales_um, stride_ratio: float = 1.0) -> dict[float, Grid]:
    """One grid per requested scale, keyed by scale."""
    out = {}
    for s in scales_um:
        s = float(s)
        if s > min(bbox.width, bbox.height):
            continue  # scale larger than the die: nothing to measure
        out[s] = build_grid(bbox, s, stride_um=s * stride_ratio)
    if not out:
        raise ValueError("no requested scale fits inside the layout bbox")
    return out


def point_counts(grid: Grid, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """How many of the given points fall in each window."""
    return point_accumulate(grid, x, y)


def point_accumulate(grid: Grid, x: np.ndarray, y: np.ndarray,
                     weights: "np.ndarray | None" = None) -> np.ndarray:
    """Sum *weights* (or count, with none) over the points in each window.

    By index arithmetic, not by scanning the grid for every point. The
    obvious nested version -- for each cell, mask the whole point array --
    is O(points x cells), and on a layout both grow with die area, so it is
    quadratic in area. Measured on synthetic dies it was the fastest-growing
    term in the whole extractor: a fourfold rise in polygon count made it
    fifteen times slower while every other stage stayed linear. At a hundred
    million points and a million windows it would never finish.

    The grid is regular by construction -- every window is ``scale_um`` across
    and starts at ``stride_um`` intervals from a common origin -- so the
    windows containing a point are found directly. With overlapping windows a
    point belongs to several, at most ``ceil(scale/stride)`` in each axis, and
    those are enumerated rather than searched.

    The half-open convention matches the rest of the package: a point on a
    window's lower or left edge belongs to it, one on the upper or right edge
    belongs to the neighbour.
    """
    out = np.zeros(len(grid), dtype=float)
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size == 0:
        return out
    w = (np.ones(x.size) if weights is None
         else np.asarray(weights, dtype=float).ravel())

    first = grid.cells[0]
    x_start, y_start = first.x0, first.y0
    stride, scale = grid.stride_um, grid.scale_um
    span = int(np.ceil(scale / stride))

    # The highest-indexed window whose left edge is at or below the point.
    col_hi = np.floor((x - x_start) / stride).astype(np.int64)
    row_hi = np.floor((y - y_start) / stride).astype(np.int64)

    for dc in range(span):
        cols = col_hi - dc
        ok_c = (cols >= 0) & (cols < grid.n_cols)
        # Inside the window, not merely to the right of its left edge: with
        # overlapping windows the two differ, and with abutting ones the
        # check costs nothing and guards the floating-point boundary.
        ok_c &= x < (x_start + cols * stride + scale)
        ok_c &= x >= (x_start + cols * stride)
        for dr in range(span):
            rows = row_hi - dr
            ok = (ok_c & (rows >= 0) & (rows < grid.n_rows)
                  & (y < (y_start + rows * stride + scale))
                  & (y >= (y_start + rows * stride)))
            if not ok.any():
                continue
            np.add.at(out, rows[ok] * grid.n_cols + cols[ok], w[ok])
    return out

# ----------------------------------------------------------------------
# features/corners.py
# ----------------------------------------------------------------------
"""Corner classification and corner markers (spec section 4E).

Tan et al. (2008) observed delamination preferentially at terminated tips and
corners rather than along parallel comb lines, so corners are a feature in
their own right. A raw vertex count is not enough: it gives the same answer
for a re-entrant corner, where stress concentrates, and an ordinary convex
one.

Corner markers double as the exact correction for the Calibre perimeter band,
which loses eps^2 of area at each convex corner and gains it at each concave
one.
"""
def _rings(polygon):
    yield list(polygon.each_point_hull()), False
    for h in range(polygon.holes()):
        yield list(polygon.each_point_hole(h)), True


def _orientation(pts) -> int:
    """+1 if the ring is counter-clockwise, -1 if clockwise."""
    n = len(pts)
    s = sum(pts[i].x * pts[(i + 1) % n].y - pts[(i + 1) % n].x * pts[i].y
            for i in range(n))
    return 1 if s > 0 else -1


def classify(region: db.Region) -> tuple[list, list]:
    """Return (convex_points, concave_points) in database units.

    Convexity is judged **with respect to the metal**, not with respect to the
    ring being walked. Two normalisations are needed for that: the ring's own
    winding, taken from its signed area, and then a flip for hole rings.
    A hole's corner that is convex in its own winding is re-entrant seen from
    the material around it -- and re-entrant corners are the
    stress-concentrating case, so getting this backwards would report a seal
    ring or any enclosing metal as having no concave corners at all.
    """
    convex, concave = [], []
    for poly in region.each():
        for pts, is_hole in _rings(poly):
            n = len(pts)
            if n < 3:
                continue
            s = _orientation(pts) * (-1 if is_hole else 1)
            for i in range(n):
                a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
                cross = s * ((b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x))
                if cross > 0:
                    convex.append(b)
                elif cross < 0:
                    concave.append(b)
    return convex, concave


def counts(region: db.Region) -> tuple[int, int]:
    convex, concave = classify(region)
    return len(convex), len(concave)


def corner_markers(region: db.Region, size_dbu: int, *, kind: str = "convex") -> db.Region:
    """Fixed-size square markers centred on each corner of the given kind.

    Turning corners into markers converts a count density into an area
    density, so the same moving-window machinery -- and the same Calibre
    DENSITY primitive -- serves every feature.
    """
    convex, concave = classify(region)
    pts = {"convex": convex, "concave": concave,
           "all": convex + concave}[kind]
    h = max(size_dbu // 2, 1)
    r = db.Region()
    for p in pts:
        r.insert(db.Box(p.x - h, p.y - h, p.x + h, p.y + h))
    return r


def corrected_band_perimeter(region: db.Region, eps_dbu: int, dbu: float) -> float:
    """True boundary length in um, from the inside band plus the corner term.

    Exact on Manhattan geometry: verified to 0.0000 % against edge lengths on
    line arrays, staircases and segmented patterns, including a case where the
    uncorrected band was 5.7 % low.
    """
    eps = eps_dbu * dbu
    band_area = (region - region.sized(-eps_dbu)).area() * dbu * dbu
    n_cv, n_cc = counts(region)
    return band_area / eps + eps * (n_cv - n_cc)

# ----------------------------------------------------------------------
# features/lineends.py
# ----------------------------------------------------------------------
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


def line_end_markers(ends: list[LineEnd], size_dbu: int) -> db.Region:
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

# ----------------------------------------------------------------------
# features/geometry.py
# ----------------------------------------------------------------------
"""Deterministic geometry features (spec section 4), thin-slice subset.

Thin slice carries metal_density (4A) and perimeter_density (4C) together
rather than density alone. One feature cannot demonstrate that the pipeline
is more than a metal-density detector, and that demonstration -- spec section 26 --
is what the whole platform rests on. The two are computed from the same
merged Region so the "same density, different perimeter" comparison is exact.
"""
@dataclass(frozen=True)
class FeatureSpec:
    name: str
    evidence_class: EvidenceClass
    unit: str
    description: str


FEATURES: dict[str, FeatureSpec] = {
    "metal_density": FeatureSpec(
        "metal_density", EvidenceClass.GDS_GEOMETRY, "dimensionless",
        "metal area / analysis area (spec 4A)"),
    "perimeter_density": FeatureSpec(
        "perimeter_density", EvidenceClass.GDS_GEOMETRY, "um^-1",
        "metal/dielectric boundary length / analysis area (spec 4C); "
        "Yoo 2004 found this more decisive than density"),
    "corner_density": FeatureSpec(
        "corner_density", EvidenceClass.GDS_GEOMETRY, "um^-2",
        "polygon corners / analysis area (spec 4E); Tan 2008 observed "
        "delamination at corners as well as tips"),
    "convex_corner_density": FeatureSpec(
        "convex_corner_density", EvidenceClass.GDS_GEOMETRY, "um^-2",
        "90-degree corners / analysis area (spec 4E)"),
    "concave_corner_density": FeatureSpec(
        "concave_corner_density", EvidenceClass.GDS_GEOMETRY, "um^-2",
        "re-entrant corners / analysis area (spec 4E); these are the "
        "stress-concentrating case, and a raw vertex count conflates them "
        "with ordinary convex corners"),
    "line_end_density": FeatureSpec(
        "line_end_density", EvidenceClass.GDS_GEOMETRY, "um^-2",
        "terminated line tips / analysis area (spec 4D); Tan 2008 observed "
        "delamination at tips rather than along comb lines. Cannot be derived "
        "from perimeter: segmenting lines moves perimeter ~3% while the "
        "termination count rises tenfold"),
}


def _window(cell) -> db.Region:
    return db.Region(db.Box(
        int(round(cell.x0 * 1000)), int(round(cell.y0 * 1000)),
        int(round(cell.x1 * 1000)), int(round(cell.y1 * 1000)),
    ))


class GeometryExtractor:
    """Extracts window-local geometry features for one layer."""

    def __init__(self, reader: LayoutReader, *,
                 line_end_w_max_um: float | None = None,
                 line_rules: "dict[str, tuple[float, float]] | None" = None,
                 line_end_aspect: float = DEFAULT_ASPECT):
        """``line_rules`` maps a layer name to (min_width_um, line_max_width_um).

        Per layer, because the rules differ per layer: applying the widest
        cutoff in the stack to every layer lets a wide line on a finer layer be
        read as a terminated tip. ``line_end_w_max_um`` remains as a single
        fallback for callers with one layer and no manifest.
        """
        self.reader = reader
        self.u = reader.units
        self.line_end_w_max_um = line_end_w_max_um
        self.line_rules = line_rules or {}
        self.line_end_aspect = line_end_aspect
        self._line_end_cache: dict[tuple[int, int], np.ndarray] = {}
        self._corner_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def corners(self, spec: LayerSpec) -> tuple[np.ndarray, np.ndarray]:
        """Convex and concave corner positions in um, each shape (n, 2).

        Classified once per layer. Orientation is resolved per polygon ring
        from its signed area, so a hole -- which winds the opposite way --
        does not have its corner types inverted.
        """
        if spec.key in self._corner_cache:
            return self._corner_cache[spec.key]
        convex, concave = classify(self.reader.region(spec))
        d = self.u.dbu_to_um
        out = (np.array([[d(p.x), d(p.y)] for p in convex], float).reshape(-1, 2),
               np.array([[d(p.x), d(p.y)] for p in concave], float).reshape(-1, 2))
        self._corner_cache[spec.key] = out
        return out

    def line_ends(self, spec: LayerSpec) -> np.ndarray:
        """Line-end positions in um, shape (n, 2). Computed once per layer."""
        if spec.key in self._line_end_cache:
            return self._line_end_cache[spec.key]
        region = self.reader.region(spec)
        rule = self.line_rules.get(spec.name)
        if rule is not None:
            min_width_um, w_max = rule
        elif self.line_end_w_max_um is not None:
            min_width_um, w_max = None, self.line_end_w_max_um
        else:
            # Nothing declared: fall back to the shortest edge present, which
            # on a layout carrying dummy fill is the fill edge rather than a
            # routing width. The manifest exists to avoid this.
            edges = region.edges()
            shortest = min((e.length() for e in edges.each()), default=0)
            min_width_um = None
            w_max = self.u.dbu_to_um(shortest) * DEFAULT_WMAX_RATIO

        if min_width_um is not None:
            # Below the drawn minimum width nothing is a routing line, so a
            # cap shorter than it is an artefact of merging or of off-grid
            # geometry rather than a terminated tip.
            region = region.sized(-self.u.um_to_dbu(min_width_um / 2)).sized(
                self.u.um_to_dbu(min_width_um / 2))

        ends = detect(region, self.u.um_to_dbu(w_max),
                               aspect=self.line_end_aspect)
        pts = np.array([[self.u.dbu_to_um(e.x), self.u.dbu_to_um(e.y)]
                        for e in ends], dtype=float).reshape(-1, 2)
        self._line_end_cache[spec.key] = pts
        return pts

    def _win_region(self, cell) -> db.Region:
        d = self.u.um_to_dbu
        return db.Region(db.Box(d(cell.x0), d(cell.y0), d(cell.x1), d(cell.y1)))

    def _win_edges_region(self, cell, grid=None) -> db.Region:
        """The window as a half-open box, for clipping *edges*.

        A closed box counts an edge lying exactly on the shared border of two
        tiles in both of them. Area does not care -- a border strip has no
        area -- but perimeter does, and layouts put edges on round
        coordinates, which is exactly where an analysis grid puts its borders.
        Measured on the golden die at a 100um non-overlapping grid, the tiled
        windows summed to 4.7 % (M8) and 7.0 % (M7) more perimeter than the
        layer actually has, all of it edge lying on a grid line. The inflation
        depends on where the grid falls, so it is an artefact of the analysis
        rather than a property of the layout.

        Excluding the top and right borders makes the tiles partition the
        plane, matching the half-open rule the point-count features (corners,
        line ends, vias) already use. The cost is one database unit of length
        at each end of a crossing edge -- 0.002um against a window perimeter
        measured in thousands, against the 5-7 % it removes.

        The grid's own outermost border is closed, for the same reason a
        histogram closes its last bin: there is no neighbouring window to
        receive the edge, so leaving it open drops the outer boundary of the
        layout -- on a die whose geometry runs to the bbox, that is the whole
        seal ring.
        """
        d = self.u.um_to_dbu
        x1, y1 = d(cell.x1) - 1, d(cell.y1) - 1
        if grid is not None:
            if cell.x1 >= grid.bbox.xmax:
                x1 = d(cell.x1)
            if cell.y1 >= grid.bbox.ymax:
                y1 = d(cell.y1)
        return db.Region(db.Box(d(cell.x0), d(cell.y0), x1, y1))

    def extract(self, spec: LayerSpec, grid: Grid) -> dict[str, np.ndarray]:
        """Return {feature_name: array over grid cells} for one layer/grid.

        Windows are processed a grid row at a time against a pre-clipped
        horizontal strip. Intersecting every window against the full-layer
        Region instead costs ~35x more on a 2mm test die and scales with die
        area, which is what makes full-chip extraction impractical.
        """
        region = self.reader.region(spec)
        edges = self.reader.edges(spec)
        u = self.u

        n = len(grid)
        metal = np.zeros(n, dtype=float)
        perim = np.zeros(n, dtype=float)
        if region.is_empty():
            zero = np.zeros(n, dtype=float)
            return {"metal_density": metal, "perimeter_density": perim,
                    "line_end_density": zero.copy(),
                    "corner_density": zero.copy(),
                    "convex_corner_density": zero.copy(),
                    "concave_corner_density": zero.copy()}

        rb = region.bbox()
        rows: dict[int, list] = {}
        for cell in grid.cells:
            rows.setdefault(cell.row, []).append(cell)

        x_lo = min(rb.left, u.um_to_dbu(grid.bbox.xmin)) - 1
        x_hi = max(rb.right, u.um_to_dbu(grid.bbox.xmax)) + 1

        for cells in rows.values():
            y0, y1 = u.um_to_dbu(cells[0].y0), u.um_to_dbu(cells[0].y1)
            if y1 <= rb.bottom or y0 >= rb.top:
                continue
            strip = db.Region(db.Box(x_lo, y0, x_hi, y1))
            s_region = region & strip
            if s_region.is_empty():
                continue
            s_edges = edges & strip
            for cell in cells:
                if cell.x1 * 1000 <= rb.left or cell.x0 * 1000 >= rb.right:
                    continue
                win = self._win_region(cell)
                i = cell.cell_id
                metal[i] = u.area_dbu2_to_um2((s_region & win).area()) / cell.area_um2
                # Clip the *edges*, not the Region: clipping the Region and
                # taking its perimeter counts the window cut as metal boundary
                # (a 30x10um bar cut at x=15um reports 50um, not the true 40um).
                perim[i] = (u.length_dbu_to_um(
                    (s_edges & self._win_edges_region(cell, grid)).length())
                    / cell.area_um2)

        out = {"metal_density": metal, "perimeter_density": perim,
               "line_end_density": self._point_density(
                   self.line_ends(spec), grid)}
        convex, concave = self.corners(spec)
        out["convex_corner_density"] = self._point_density(convex, grid)
        out["concave_corner_density"] = self._point_density(concave, grid)
        out["corner_density"] = (out["convex_corner_density"]
                                 + out["concave_corner_density"])
        return out

    def _point_density(self, pts: np.ndarray, grid: Grid) -> np.ndarray:
        """Count of point features per unit area, per window.

        Line ends and corners are points, so they are counted directly rather
        than routed through an area proxy. The Calibre path turns them into
        fixed-size markers instead, because there the moving window is the
        DENSITY primitive and only understands area.
        """
        if len(pts) == 0:
            return np.zeros(len(grid), dtype=float)
        counts = point_accumulate(grid, pts[:, 0], pts[:, 1])
        area = np.array([c.area_um2 for c in grid.cells], dtype=float)
        return counts / area

    def extract_roi(self, spec: LayerSpec, x0: float, y0: float,
                    x1: float, y1: float) -> dict[str, float]:
        """Single-window extraction, used by the section 26 pair tests."""
        region = self.reader.region(spec)
        edges = self.reader.edges(spec)
        u = self.u
        win = db.Region(db.Box(u.um_to_dbu(x0), u.um_to_dbu(y0),
                               u.um_to_dbu(x1), u.um_to_dbu(y1)))
        area = (x1 - x0) * (y1 - y0)
        def count(pts):
            if not len(pts):
                return 0
            return int(((pts[:, 0] >= x0) & (pts[:, 0] < x1)
                        & (pts[:, 1] >= y0) & (pts[:, 1] < y1)).sum())

        convex, concave = self.corners(spec)
        n_convex, n_concave = count(convex), count(concave)
        return {
            "metal_density": u.area_dbu2_to_um2((region & win).area()) / area,
            "perimeter_density": u.length_dbu_to_um((edges & win).length()) / area,
            "line_end_density": count(self.line_ends(spec)) / area,
            "convex_corner_density": n_convex / area,
            "concave_corner_density": n_concave / area,
            "corner_density": (n_convex + n_concave) / area,
        }

# ----------------------------------------------------------------------
# features/orientation.py
# ----------------------------------------------------------------------
"""Orientation features (spec section 4F).

Tan et al. (2008) report an orientation dependence in observed delamination,
and Rabie et al. (2018) list diagonal final-metal routing under corner bumps
as a CPI mitigation, so orientation is a tier-1 feature rather than a
descriptor.

Measurements are **length weighted**, not polygon-count weighted: one long
line and one short stub in the same direction are not equally strong evidence
of that direction, and a count-based statistic would treat them as such.

A non-Manhattan edge is decomposed into its |dx| and |dy| components, so a
45-degree edge contributes equally to both axes and the Manhattan case falls
out exactly.
"""
ORIENTATION_FEATURES = ("horizontal_fraction", "vertical_fraction", "orientation_anisotropy",
            "routing_direction_rad", "orientation_coherence")
EVIDENCE_CLASS = EvidenceClass.GDS_GEOMETRY


def _axial_moments(other: db.Edges, dbu: float) -> tuple[float, float, float]:
    """Length-weighted (cos 2t, sin 2t, length) for non-axis-aligned edges.

    Only these need a Python loop. Horizontal and vertical edges have known
    doubled angles, so their moments follow from the lengths KLayout already
    computes in C++ -- and on Manhattan geometry that is every edge. Walking
    every edge of every window instead cost 8 ms per cell, thirteen times the
    whole geometry extractor.
    """
    import math

    cos2 = sin2 = total = 0.0
    for e in other.each():
        dx, dy = e.dx() * dbu, e.dy() * dbu
        length = math.hypot(dx, dy)
        if length <= 0:
            continue
        theta2 = 2.0 * math.atan2(dy, dx)
        cos2 += length * math.cos(theta2)
        sin2 += length * math.sin(theta2)
        total += length
    return cos2, sin2, total


def _direction_and_coherence(h_len: float, v_len: float,
                             other_moments: tuple[float, float, float]
                             ) -> tuple[float, float]:
    """Axial orientation from the horizontal, vertical and oblique moments.

    Orientation is axial rather than directional -- a line at theta is the
    same line as one at theta + 180 -- so angles are doubled before averaging.
    Averaging raw angles would place the mean of 170 and 10 degrees at 90,
    perpendicular to both. A horizontal edge has 2t = 0 and a vertical one
    2t = pi, which is why they enter as +length and -length on the cosine axis
    and contribute nothing to the sine.

    ``coherence`` is the resultant over the total length: 0 for an isotropic
    window, 1 where every edge runs the same way. It is what separates a
    deliberately diagonal routing direction from no direction at all, since
    both sit at the same place on a radial-versus-tangential axis.
    """
    import math

    o_cos, o_sin, o_len = other_moments
    cos2 = h_len - v_len + o_cos
    sin2 = o_sin
    total = h_len + v_len + o_len
    if total <= 0:
        return float("nan"), 0.0
    direction = (0.5 * math.atan2(sin2, cos2)) % math.pi
    # 0 and pi are the same axis; a rounding error just below zero would
    # otherwise be reported as 180 degrees for horizontal routing.
    if direction >= math.pi - 1e-9:
        direction = 0.0
    return direction, math.hypot(cos2, sin2) / total


def _split(edges: db.Edges) -> tuple[db.Edges, db.Edges, db.Edges]:
    """Horizontal, vertical, and everything else."""
    h = edges.with_angle(0, False)
    v = edges.with_angle(90, False)
    other = edges - h - v
    return h, v, other


def _projected_lengths(other: db.Edges, dbu: float) -> tuple[float, float]:
    """Sum |dx| and |dy| over non-axis-aligned edges."""
    dx = dy = 0.0
    for e in other.each():
        dx += abs(e.dx())
        dy += abs(e.dy())
    return dx * dbu, dy * dbu


def anisotropy(h_len: float, v_len: float) -> float:
    """Signed, in [-1, 1]. +1 all horizontal, -1 all vertical, 0 balanced.

    Kept signed rather than absolute so that "mostly horizontal" and "mostly
    vertical" stay distinguishable -- a cross-layer orientation mismatch is
    the difference between two of these, which an absolute value would erase.
    """
    total = h_len + v_len
    return 0.0 if total <= 0 else (h_len - v_len) / total


class OrientationExtractor:
    def __init__(self, reader: LayoutReader):
        self.reader = reader
        self.u = reader.units
        self._split_cache: dict[tuple[int, int], tuple] = {}
        self._edge_cache: dict[tuple[int, int], db.Edges] = {}

    def _all_edges(self, spec: LayerSpec) -> db.Edges:
        if spec.key not in self._edge_cache:
            self._edge_cache[spec.key] = self.reader.edges(spec)
        return self._edge_cache[spec.key]

    def _edges(self, spec: LayerSpec):
        if spec.key not in self._split_cache:
            self._split_cache[spec.key] = _split(self.reader.edges(spec))
        return self._split_cache[spec.key]

    def _measure(self, h, v, other, win: db.Region) -> dict[str, float]:
        u = self.u
        h_len = u.length_dbu_to_um((h & win).length())
        v_len = u.length_dbu_to_um((v & win).length())

        oblique = other & win
        moments = ((0.0, 0.0, 0.0) if oblique.is_empty()
                   else _axial_moments(oblique, u.dbu))
        direction, coherence = _direction_and_coherence(h_len, v_len, moments)

        if not oblique.is_empty():
            ox, oy = _projected_lengths(oblique, u.dbu)
            h_len += ox
            v_len += oy
        total = h_len + v_len
        if total <= 0:
            return {"horizontal_fraction": 0.0, "vertical_fraction": 0.0,
                    "orientation_anisotropy": 0.0,
                    "routing_direction_rad": float("nan"),
                    "orientation_coherence": 0.0}
        return {"horizontal_fraction": h_len / total,
                "vertical_fraction": v_len / total,
                "orientation_anisotropy": anisotropy(h_len, v_len),
                "routing_direction_rad": direction,
                "orientation_coherence": coherence}

    def extract(self, spec: LayerSpec, grid: Grid) -> dict[str, np.ndarray]:
        h, v, other = self._edges(spec)
        n = len(grid)
        out = {k: np.zeros(n) for k in ORIENTATION_FEATURES}
        out["routing_direction_rad"] = np.full(n, np.nan)
        if h.is_empty() and v.is_empty() and other.is_empty():
            return out

        u = self.u
        rows: dict[int, list] = {}
        for c in grid.cells:
            rows.setdefault(c.row, []).append(c)

        x_lo = u.um_to_dbu(grid.bbox.xmin) - 1
        x_hi = u.um_to_dbu(grid.bbox.xmax) + 1
        for cells in rows.values():
            y0, y1 = u.um_to_dbu(cells[0].y0), u.um_to_dbu(cells[0].y1)
            strip = db.Region(db.Box(x_lo, y0, x_hi, y1))
            sh, sv, so = h & strip, v & strip, other & strip
            if sh.is_empty() and sv.is_empty() and so.is_empty():
                continue
            for c in cells:
                win = db.Region(db.Box(u.um_to_dbu(c.x0), u.um_to_dbu(c.y0),
                                       u.um_to_dbu(c.x1), u.um_to_dbu(c.y1)))
                for k, val in self._measure(sh, sv, so, win).items():
                    out[k][c.cell_id] = val
        return out

    def extract_roi(self, spec: LayerSpec, x0, y0, x1, y1) -> dict[str, float]:
        h, v, other = self._edges(spec)
        u = self.u
        win = db.Region(db.Box(u.um_to_dbu(x0), u.um_to_dbu(y0),
                               u.um_to_dbu(x1), u.um_to_dbu(y1)))
        return self._measure(h, v, other, win)

# ----------------------------------------------------------------------
# features/structures.py
# ----------------------------------------------------------------------
"""Large-structure families (spec sections 4H-4I, extended).

Rabie et al. (2018) list wide-metal slotting among the layout levers, and both
slotting and dummy fill change what the other features mean: fill sets the
shortest edge on a layer, and a slot's end is bounded by re-entrant corners
rather than by a routing tip.

Four things are separated here that a single density cannot distinguish:

* how much of the metal is *wide* metal rather than routing;
* how much of that wide metal is slotted;
* how much of the layer is dummy fill rather than functional geometry;
* the boundary length of the wide metal alone, which is where an abrupt
  stiffness change sits.

Wide metal is found by morphological opening: a shape survives an opening by
w/2 only if it is at least w across. Fill is not inferred -- it is a declared
layer, because separating fill from routing by shape alone is guesswork and
getting it wrong is what makes a fill edge look like a minimum-width line.
"""
STRUCTURE_FEATURES = ("wide_metal_fraction", "wide_metal_perimeter_density",
            "slot_density", "slotted_metal_fraction",
            "unslotted_wide_metal_fraction",
            "fill_density", "fill_fraction")


class StructureExtractor:
    """Wide metal, slots and declared dummy fill for one metal layer."""

    def __init__(self, reader: LayoutReader, *, wide_width_um: float = 3.0,
                 fill_layers: "dict[str, LayerSpec] | None" = None):
        self.reader = reader
        self.u = reader.units
        self.wide_width_um = wide_width_um
        self.fill_layers = fill_layers or {}
        self._cache: dict[tuple[int, int], tuple] = {}

    def _derived(self, spec: LayerSpec):
        """Merged region, its wide-metal opening, and the slot markers."""
        if spec.key in self._cache:
            return self._cache[spec.key]
        region = self.reader.region(spec)
        h = max(self.u.um_to_dbu(self.wide_width_um / 2), 1)
        wide = region.sized(-h).sized(h)

        # A slot is a hole in the metal. Its area is the natural measure; its
        # count needs a marker, because a hole has no centroid the boolean
        # engine will hand back.
        slots = db.Region()
        slot_points = []
        for poly in region.each():
            for i in range(poly.holes()):
                pts = list(poly.each_point_hole(i))
                xs = [p.x for p in pts]
                ys = [p.y for p in pts]
                slots.insert(db.Box(min(xs), min(ys), max(xs), max(ys)))
                slot_points.append(((min(xs) + max(xs)) / 2 * self.u.dbu,
                                    (min(ys) + max(ys)) / 2 * self.u.dbu))
        # Wide metal belonging to shapes that carry no slot at all. Rabie's
        # lever is slotting, so the recommended state has to lower the score:
        # ranking wide-metal fraction alone flags a correctly slotted plate as
        # hard as an unbroken one, which inverts the lever.
        #
        # The split is per polygon, not per window. A window-level version --
        # zero the cell if any slot centroid lands in it -- takes only two
        # values per layer, so every cell ties and the channel can never
        # report anything. Per polygon it is an area fraction that varies
        # across windows, which is what a percentile rank needs. A shape
        # slotted at one end is excluded along its whole length, so the
        # measure under-reports rather than over-reports.
        unslotted = db.Region()
        for poly in region.each():
            if poly.holes() == 0:
                unslotted.insert(poly)
        unslotted.merge()

        out = (region, wide, wide.edges(),
               np.array(slot_points, dtype=float).reshape(-1, 2),
               unslotted.sized(-h).sized(h))
        self._cache[spec.key] = out
        return out

    def extract(self, spec: LayerSpec, grid: Grid) -> dict[str, np.ndarray]:
        region, wide, wide_edges, slot_points, unslotted = self._derived(spec)
        n = len(grid)
        out = {k: np.zeros(n) for k in STRUCTURE_FEATURES}
        if region.is_empty():
            return out

        fill_spec = self.fill_layers.get(spec.name)
        fill_region = (self.reader.region(fill_spec)
                       if fill_spec is not None else db.Region())
        u = self.u

        rows: dict[int, list] = {}
        for c in grid.cells:
            rows.setdefault(c.row, []).append(c)
        rb = region.bbox()
        x_lo = min(rb.left, u.um_to_dbu(grid.bbox.xmin)) - 1
        x_hi = max(rb.right, u.um_to_dbu(grid.bbox.xmax)) + 1

        for cells in rows.values():
            y0, y1 = u.um_to_dbu(cells[0].y0), u.um_to_dbu(cells[0].y1)
            strip_box = db.Region(db.Box(x_lo, y0, x_hi, y1))
            s_region = region & strip_box
            if s_region.is_empty():
                continue
            s_wide = wide & strip_box
            s_unslotted = unslotted & strip_box
            s_edges = wide_edges & strip_box
            s_fill = fill_region & strip_box

            for c in cells:
                win = db.Region(db.Box(u.um_to_dbu(c.x0), u.um_to_dbu(c.y0),
                                       u.um_to_dbu(c.x1), u.um_to_dbu(c.y1)))
                i = c.cell_id
                metal_area = u.area_dbu2_to_um2((s_region & win).area())
                wide_area = u.area_dbu2_to_um2((s_wide & win).area())
                fill_area = u.area_dbu2_to_um2((s_fill & win).area())

                out["wide_metal_fraction"][i] = (
                    wide_area / metal_area if metal_area > 0 else 0.0)
                out["unslotted_wide_metal_fraction"][i] = (
                    u.area_dbu2_to_um2((s_unslotted & win).area()) / metal_area
                    if metal_area > 0 else 0.0)
                out["wide_metal_perimeter_density"][i] = (
                    u.length_dbu_to_um((s_edges & win).length()) / c.area_um2)
                out["fill_density"][i] = fill_area / c.area_um2
                total = metal_area + fill_area
                out["fill_fraction"][i] = fill_area / total if total > 0 else 0.0

                if len(slot_points):
                    inside = ((slot_points[:, 0] >= c.x0)
                              & (slot_points[:, 0] < c.x1)
                              & (slot_points[:, 1] >= c.y0)
                              & (slot_points[:, 1] < c.y1))
                    count = int(inside.sum())
                    out["slot_density"][i] = count / c.area_um2
                    out["slotted_metal_fraction"][i] = (
                        wide_area / metal_area if count and metal_area > 0 else 0.0)
        return out

    def extract_roi(self, spec: LayerSpec, x0, y0, x1, y1) -> dict[str, float]:
        region, wide, wide_edges, slot_points, unslotted = self._derived(spec)
        u = self.u
        win = db.Region(db.Box(u.um_to_dbu(x0), u.um_to_dbu(y0),
                               u.um_to_dbu(x1), u.um_to_dbu(y1)))
        area = (x1 - x0) * (y1 - y0)
        metal_area = u.area_dbu2_to_um2((region & win).area())
        wide_area = u.area_dbu2_to_um2((wide & win).area())
        fill_spec = self.fill_layers.get(spec.name)
        fill_area = (u.area_dbu2_to_um2((self.reader.region(fill_spec) & win).area())
                     if fill_spec is not None else 0.0)
        count = 0
        if len(slot_points):
            count = int(((slot_points[:, 0] >= x0) & (slot_points[:, 0] < x1)
                         & (slot_points[:, 1] >= y0)
                         & (slot_points[:, 1] < y1)).sum())
        total = metal_area + fill_area
        return {
            "wide_metal_fraction": wide_area / metal_area if metal_area else 0.0,
            "wide_metal_perimeter_density":
                u.length_dbu_to_um((wide_edges & win).length()) / area,
            "slot_density": count / area,
            "slotted_metal_fraction":
                (wide_area / metal_area if count and metal_area else 0.0),
            "unslotted_wide_metal_fraction":
                (u.area_dbu2_to_um2((unslotted & win).area()) / metal_area
                 if metal_area else 0.0),
            "fill_density": fill_area / area,
            "fill_fraction": fill_area / total if total > 0 else 0.0,
        }

# ----------------------------------------------------------------------
# features/vias.py
# ----------------------------------------------------------------------
"""Via features (spec section 4B).

Vanstreels et al. (2020) correlate via density with observed BEOL fracture,
and Zahedmanesh & Vanstreels (2019) show intermediate-ULK via density and top
Z-group via density acting differently -- so via features are tier-1 evidence
and their layer identity has to survive into the association table.

Both an area density and a count density are produced, because they answer
different questions and a layer can hold one constant while moving the other.
Many small vias and few large ones at equal area density represent different
amounts of interface, and Vanstreels' fracture counts are per via rather than
per unit of via area.

Vias are discrete objects, so counting is by centroid: a via straddling a
window boundary belongs to exactly one window, and the sum of counts over a
non-overlapping grid equals the number of vias on the die.
"""
VIA_FEATURES = ("via_density", "via_count_density", "mean_via_area")



class ViaExtractor:
    """Window-local via area and count densities for one via layer."""

    def __init__(self, reader: LayoutReader):
        self.reader = reader
        self.u = reader.units
        self._centroid_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def centroids(self, spec: LayerSpec) -> tuple[np.ndarray, np.ndarray]:
        """Via centroids in um, shape (n, 2), and their areas in um^2.

        Computed once per layer. Merging first means a stack of coincident
        shapes counts as one via, which is what the drawn layer means.
        """
        if spec.key in self._centroid_cache:
            return self._centroid_cache[spec.key]
        region = self.reader.region(spec)
        pts, areas = [], []
        u = self.u
        for poly in region.each():
            b = poly.bbox()
            pts.append(((b.left + b.right) / 2 * u.dbu,
                        (b.bottom + b.top) / 2 * u.dbu))
            areas.append(poly.area() * u.dbu * u.dbu)
        out = (np.array(pts, dtype=float).reshape(-1, 2),
               np.array(areas, dtype=float))
        self._centroid_cache[spec.key] = out
        return out

    def extract(self, spec: LayerSpec, grid: Grid) -> dict[str, np.ndarray]:
        n = len(grid)
        out = {k: np.zeros(n) for k in VIA_FEATURES}
        pts, areas = self.centroids(spec)
        if len(pts) == 0:
            return out

        region = self.reader.region(spec)
        u = self.u
        px, py = pts[:, 0], pts[:, 1]

        # Counts and mean area by index arithmetic, once, rather than by
        # masking the whole via array inside the per-window loop: that is
        # O(vias x windows), and on a layout both grow with die area.
        cell_area = np.array([c.area_um2 for c in grid.cells], dtype=float)
        counts = point_accumulate(grid, px, py)
        summed = point_accumulate(grid, px, py, areas)
        out["via_count_density"] = counts / cell_area
        out["mean_via_area"] = np.where(counts > 0,
                                        summed / np.maximum(counts, 1.0), 0.0)

        rows: dict[int, list] = {}
        for c in grid.cells:
            rows.setdefault(c.row, []).append(c)
        rb = region.bbox()
        x_lo = min(rb.left, u.um_to_dbu(grid.bbox.xmin)) - 1
        x_hi = max(rb.right, u.um_to_dbu(grid.bbox.xmax)) + 1

        for cells in rows.values():
            y0, y1 = u.um_to_dbu(cells[0].y0), u.um_to_dbu(cells[0].y1)
            if y1 <= rb.bottom or y0 >= rb.top:
                continue
            strip = region & db.Region(db.Box(x_lo, y0, x_hi, y1))
            if strip.is_empty():
                continue
            for c in cells:
                win = db.Region(db.Box(u.um_to_dbu(c.x0), u.um_to_dbu(c.y0),
                                       u.um_to_dbu(c.x1), u.um_to_dbu(c.y1)))
                area = u.area_dbu2_to_um2((strip & win).area())
                out["via_density"][c.cell_id] = area / c.area_um2
        return out

    def extract_roi(self, spec: LayerSpec, x0, y0, x1, y1) -> dict[str, float]:
        region = self.reader.region(spec)
        u = self.u
        win = db.Region(db.Box(u.um_to_dbu(x0), u.um_to_dbu(y0),
                               u.um_to_dbu(x1), u.um_to_dbu(y1)))
        area = (x1 - x0) * (y1 - y0)
        pts, areas = self.centroids(spec)
        if len(pts) == 0:
            return dict.fromkeys(VIA_FEATURES, 0.0)
        inside = ((pts[:, 0] >= x0) & (pts[:, 0] < x1)
                  & (pts[:, 1] >= y0) & (pts[:, 1] < y1))
        count = int(inside.sum())
        return {
            "via_density": u.area_dbu2_to_um2((region & win).area()) / area,
            "via_count_density": count / area,
            "mean_via_area": float(areas[inside].mean()) if count else 0.0,
        }

# ----------------------------------------------------------------------
# features/gradient.py
# ----------------------------------------------------------------------
"""Gradient features (spec section 5).

The spec is explicit that a feature's spatial gradient must not be assumed
less important than its absolute value, so gradients are first-class outputs
rather than diagnostics.

Two things are handled carefully.

**Physical units.** ``dQ_dx`` is per micrometre, differenced over the grid
stride. Differencing over cell indices would make the same layout produce
gradients differing by orders of magnitude between scales, and "which scale
shows the strongest association" would then be answering a question about the
grid rather than about the layout.

**Die-boundary cells.** Interior cells get a centred difference; boundary
cells can only get a one-sided one. One-sided differences are biased
differently from centred ones, and the cells carrying them form a ring at the
die edge -- which is exactly the shape of ``distance_to_die_edge``, a
PACKAGE_POSITION confounder. Left unmarked, a gradient feature acquires a
spurious association with die position through nothing but its own numerics.
Every gradient therefore ships with an ``*_interior`` mask, and the extractor
can drop the boundary ring outright.
"""
def to_field(values: np.ndarray, grid: Grid) -> np.ndarray:
    """Flat per-cell array -> (n_rows, n_cols) field."""
    field = np.full((grid.n_rows, grid.n_cols), np.nan)
    for i, c in enumerate(grid.cells):
        field[c.row, c.col] = values[i]
    return field


def to_flat(field: np.ndarray, grid: Grid) -> np.ndarray:
    out = np.zeros(len(grid))
    for i, c in enumerate(grid.cells):
        out[i] = field[c.row, c.col]
    return out


def interior_mask(grid: Grid) -> np.ndarray:
    """True for cells whose gradient came from a centred difference."""
    mask = np.zeros(len(grid), dtype=bool)
    for i, c in enumerate(grid.cells):
        mask[i] = (0 < c.row < grid.n_rows - 1) and (0 < c.col < grid.n_cols - 1)
    return mask


def gradients(values: np.ndarray, grid: Grid, name: str, *,
              drop_boundary: bool = True) -> dict[str, np.ndarray]:
    """Return dQ_dx, dQ_dy and |grad Q| for one scalar feature.

    Spacing is the grid stride in micrometres, so results are per-um and
    comparable across scales. With ``drop_boundary`` the die-edge ring is set
    to NaN rather than filled with a one-sided estimate; downstream statistics
    skip NaN, which keeps the numerical artifact out of the association tables
    instead of letting it masquerade as a die-position effect.
    """
    if grid.n_rows < 3 or grid.n_cols < 3:
        nan = np.full(len(values), np.nan)
        return {f"{name}_dx": nan.copy(), f"{name}_dy": nan.copy(),
                f"{name}_grad_mag": nan.copy()}

    field = to_field(values, grid)
    h = grid.stride_um
    dy_field, dx_field = np.gradient(field, h, h, edge_order=1)

    dx = to_flat(dx_field, grid)
    dy = to_flat(dy_field, grid)
    if drop_boundary:
        edge = ~interior_mask(grid)
        dx[edge] = np.nan
        dy[edge] = np.nan
    return {f"{name}_dx": dx, f"{name}_dy": dy,
            f"{name}_grad_mag": np.hypot(dx, dy)}


def gradient_set(features: dict[str, np.ndarray], grid: Grid, *,
                 only: tuple[str, ...] | None = None,
                 drop_boundary: bool = True) -> dict[str, np.ndarray]:
    """Gradients for every (or a chosen subset of) scalar feature."""
    out = {}
    for name, vals in features.items():
        if only is not None and name not in only:
            continue
        out.update(gradients(vals, grid, name, drop_boundary=drop_boundary))
    return out

# ----------------------------------------------------------------------
# features/crosslayer.py
# ----------------------------------------------------------------------
"""Cross-layer architecture features (spec section 7).

Vanstreels et al. (2020) correlate BEOL architecture with observed fracture
counts, and Zahedmanesh & Vanstreels (2019) show a stiff top metal group can
*lower* the crack driving force in the layer beneath it. Two consequences are
built in here.

**Layer identity is preserved.** Features are named ``density_difference_M8_M7``,
never ``generic_density_difference``. A pooled cross-layer index would average
a shielding pair against a loading pair and report neither.

**Differences are signed, in a fixed order.** ``density_difference_A_B`` is
``density(A) - density(B)``, upper layer first. Taking an absolute value would
erase precisely the distinction the shielding result rests on.

**The magnitude is emitted alongside the signed value, not instead of it.**
They answer different questions and neither substitutes for the other: a
signed difference cannot detect an effect driven by how *much* two layers
disagree, because both directions of disagreement sit at opposite ends of the
scale and the association collapses to chance. Measured on a die whose driver
is orientation mismatch, the signed feature scores AUC 0.50 while its own
absolute value scores 0.78 -- from identical inputs.

The pair set is the main lever on the hypothesis budget: all pairs of 12
layers is 66 combinations, while the pairs the literature actually motivates
-- adjacent layers, and the top layer against each underlying one -- is 21.
Choose the pair set before extraction, not after seeing results.
"""
#: Features computed for each selected layer pair. Each appears twice: signed
#: (``*_difference_A_B``) and as a magnitude (``*_mismatch_A_B``).
PAIR_FEATURES = ("density_difference", "perimeter_density_difference",
                 "orientation_difference", "line_end_density_difference",
                 "density_mismatch", "perimeter_density_mismatch",
                 "orientation_mismatch", "line_end_density_mismatch")

#: Features computed once across the whole stack.
STACK_FEATURES = ("density_variance_across_layers", "stacked_dense_layer_count",
                  "stacked_sparse_layer_count", "cross_layer_transition_index")


@dataclass(frozen=True)
class LayerStack:
    """Ordered layer names, topmost first."""
    names: tuple[str, ...]

    @property
    def top(self) -> str:
        return self.names[0]

    def pairs(self, selection: str = "adjacent_and_top") -> list[tuple[str, str]]:
        """Layer pairs to compute, upper layer first.

        ``adjacent`` is the mechanically local relationship; ``top_vs_all``
        is the chip-package one. Together they cover what the literature
        motivates at a fraction of the hypothesis count of ``all``.
        """
        n = self.names
        if selection == "all":
            return list(combinations(n, 2))
        adjacent = [(n[i], n[i + 1]) for i in range(len(n) - 1)]
        if selection == "adjacent":
            return adjacent
        top = [(n[0], m) for m in n[1:]]
        if selection == "top_vs_all":
            return top
        if selection == "adjacent_and_top":
            seen, out = set(), []
            for p in adjacent + top:
                if p not in seen:
                    seen.add(p)
                    out.append(p)
            return out
        raise ValueError(f"unknown pair selection {selection!r}")

    def hypothesis_count(self, selection: str, n_scales: int,
                         n_pair_features: int = len(PAIR_FEATURES)) -> int:
        return len(self.pairs(selection)) * n_pair_features * n_scales


def pair_features(per_layer: dict[str, dict[str, np.ndarray]],
                  upper: str, lower: str) -> dict[str, np.ndarray]:
    """Signed differences between two layers on the same grid."""
    a, b = per_layer[upper], per_layer[lower]
    out = {}
    mapping = {
        "density_difference": "metal_density",
        "perimeter_density_difference": "perimeter_density",
        "line_end_density_difference": "line_end_density",
        "orientation_difference": "orientation_anisotropy",
    }
    for feature, source in mapping.items():
        if source in a and source in b:
            diff = a[source] - b[source]
            out[f"{feature}_{upper}_{lower}"] = diff
            magnitude = feature.replace("_difference", "_mismatch")
            out[f"{magnitude}_{upper}_{lower}"] = np.abs(diff)
    return out


def stack_features(per_layer: dict[str, dict[str, np.ndarray]],
                   stack: LayerStack, *, dense_threshold: float = 0.5,
                   sparse_threshold: float = 0.2) -> dict[str, np.ndarray]:
    """Whole-stack summaries at each location.

    The dense/sparse counts are the layout analogue of the stiff-group idea:
    how many layers are heavily metallised above a given point, rather than
    how much metal any one of them carries.
    """
    names = [n for n in stack.names if n in per_layer]
    if not names:
        return {}
    dens = np.vstack([per_layer[n]["metal_density"] for n in names])

    out = {
        "density_variance_across_layers": dens.var(axis=0),
        "stacked_dense_layer_count": (dens >= dense_threshold).sum(axis=0).astype(float),
        "stacked_sparse_layer_count": (dens <= sparse_threshold).sum(axis=0).astype(float),
    }
    # Transition index: how much the stack changes from layer to layer at this
    # point. A uniformly dense stack and a uniformly sparse one both score 0;
    # a dense-over-sparse interface scores high.
    if len(names) > 1:
        out["cross_layer_transition_index"] = np.abs(np.diff(dens, axis=0)).mean(axis=0)
    return out


def top_vs_underlying(per_layer: dict[str, dict[str, np.ndarray]],
                      stack: LayerStack) -> dict[str, np.ndarray]:
    """Top layer against the mean of everything beneath it (spec section 8)."""
    under = [n for n in stack.names[1:] if n in per_layer]
    if stack.top not in per_layer or not under:
        return {}
    top = per_layer[stack.top]
    out = {}
    for feature, source in (("top_to_underlying_density_mismatch", "metal_density"),
                            ("top_to_underlying_orientation_mismatch",
                             "orientation_anisotropy")):
        if source not in top:
            continue
        mean_under = np.mean([per_layer[n][source] for n in under
                              if source in per_layer[n]], axis=0)
        out[feature] = top[source] - mean_under
    return out


def crosslayer_extract(per_layer: dict[str, dict[str, np.ndarray]], stack: LayerStack, *,
            selection: str = "adjacent_and_top") -> dict[str, np.ndarray]:
    """All cross-layer features for one grid."""
    out: dict[str, np.ndarray] = {}
    for upper, lower in stack.pairs(selection):
        if upper in per_layer and lower in per_layer:
            out.update(pair_features(per_layer, upper, lower))
    out.update(stack_features(per_layer, stack))
    out.update(top_vs_underlying(per_layer, stack))
    return out

# ----------------------------------------------------------------------
# features/bump_relative.py
# ----------------------------------------------------------------------
"""Routing orientation resolved against the package loading direction.

Rabie et al. (2018) recommend running the final metal *diagonally* under the
corner bumps. That is a directional statement about the layout relative to the
package, and no scalar distance to a bump can express it: two cells the same
distance from the same bump, one routed radially and one diagonally, are the
same in every feature the engine had until now.

These are GDS_GEOMETRY features -- they describe how the layout is drawn, and
they are the thing a designer can change -- but they cannot be computed
without a bump map. The bump-position confounders stay in the
PACKAGE_POSITION baseline, so an association found here has to beat "how far
from a bump this is" before it means anything.

Orientation is axial, so every difference is taken on doubled angles: a line
at 179 degrees and one at 1 degree differ by 2, not by 178.
"""
BUMP_RELATIVE_FEATURES = ("routing_vs_radial_angle_rad", "routing_radial_alignment",
            "routing_diagonality")

#: Below this the window has no dominant routing direction, so an angle
#: measured from it describes rounding rather than layout.
MIN_COHERENCE = 0.15


def bump_relative_extract(routing_direction_rad: np.ndarray, orientation_coherence: np.ndarray,
            bump_radial_direction_rad: np.ndarray, *,
            min_coherence: float = MIN_COHERENCE) -> dict[str, np.ndarray]:
    """Resolve the routing direction against the bump radial direction.

    ``routing_radial_alignment`` is ``cos(2*delta)``: +1 when routing runs
    radially, -1 when it runs tangentially, 0 at 45 degrees.
    ``routing_diagonality`` is ``|sin(2*delta)|``, which peaks at exactly the
    45 degrees Rabie recommends and is zero at both of the other two -- so the
    lever has a feature of its own rather than being the midpoint of one.

    A window whose edges have no dominant direction gets NaN rather than an
    angle: an isotropic window and a deliberately diagonal one sit at the same
    point on the alignment axis, and only coherence separates them.
    """
    routing = np.asarray(routing_direction_rad, dtype=float)
    coherence = np.asarray(orientation_coherence, dtype=float)
    radial = np.asarray(bump_radial_direction_rad, dtype=float)

    delta = routing - radial
    acute = np.abs(np.arctan2(np.sin(2 * delta), np.cos(2 * delta))) / 2.0

    usable = (np.isfinite(routing) & np.isfinite(radial)
              & (coherence >= min_coherence))
    nan = np.full(len(routing), np.nan)

    return {
        "routing_vs_radial_angle_rad": np.where(usable, acute, nan),
        "routing_radial_alignment": np.where(usable, np.cos(2 * acute), nan),
        "routing_diagonality": np.where(usable, np.abs(np.sin(2 * acute)), nan),
    }
