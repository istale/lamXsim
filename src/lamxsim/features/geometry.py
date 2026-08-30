"""Deterministic geometry features (spec section 4), thin-slice subset.

Thin slice carries metal_density (4A) and perimeter_density (4C) together
rather than density alone. One feature cannot demonstrate that the pipeline
is more than a metal-density detector, and that demonstration -- spec section 26 --
is what the whole platform rests on. The two are computed from the same
merged Region so the "same density, different perimeter" comparison is exact.
"""
from __future__ import annotations

from dataclasses import dataclass

import klayout.db as db
import numpy as np

from ..evidence import EvidenceClass
from ..layout.reader import LayerSpec, LayoutReader
from . import corners, lineends
from .grid import Grid


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
                 line_end_aspect: float = lineends.DEFAULT_ASPECT):
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
        convex, concave = corners.classify(self.reader.region(spec))
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
            w_max = self.u.dbu_to_um(shortest) * lineends.DEFAULT_WMAX_RATIO

        if min_width_um is not None:
            # Below the drawn minimum width nothing is a routing line, so a
            # cap shorter than it is an artefact of merging or of off-grid
            # geometry rather than a terminated tip.
            region = region.sized(-self.u.um_to_dbu(min_width_um / 2)).sized(
                self.u.um_to_dbu(min_width_um / 2))

        ends = lineends.detect(region, self.u.um_to_dbu(w_max),
                               aspect=self.line_end_aspect)
        pts = np.array([[self.u.dbu_to_um(e.x), self.u.dbu_to_um(e.y)]
                        for e in ends], dtype=float).reshape(-1, 2)
        self._line_end_cache[spec.key] = pts
        return pts

    def _win_region(self, cell) -> db.Region:
        d = self.u.um_to_dbu
        return db.Region(db.Box(d(cell.x0), d(cell.y0), d(cell.x1), d(cell.y1)))

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
                perim[i] = u.length_dbu_to_um((s_edges & win).length()) / cell.area_um2

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
        out = np.zeros(len(grid), dtype=float)
        if len(pts) == 0:
            return out
        px, py = pts[:, 0], pts[:, 1]
        for cell in grid.cells:
            inside = ((px >= cell.x0) & (px < cell.x1)
                      & (py >= cell.y0) & (py < cell.y1))
            out[cell.cell_id] = inside.sum() / cell.area_um2
        return out

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
