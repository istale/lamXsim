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
from . import lineends
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
                 line_end_aspect: float = lineends.DEFAULT_ASPECT):
        self.reader = reader
        self.u = reader.units
        self.line_end_w_max_um = line_end_w_max_um
        self.line_end_aspect = line_end_aspect
        self._line_end_cache: dict[tuple[int, int], np.ndarray] = {}

    def line_ends(self, spec: LayerSpec) -> np.ndarray:
        """Line-end positions in um, shape (n, 2). Computed once per layer."""
        if spec.key in self._line_end_cache:
            return self._line_end_cache[spec.key]
        region = self.reader.region(spec)
        if self.line_end_w_max_um is None:
            # No layer width configured: fall back to the shortest edge present,
            # scaled by the recommended ratio.
            edges = region.edges()
            shortest = min((e.length() for e in edges.each()), default=0)
            w_max = self.u.dbu_to_um(shortest) * lineends.DEFAULT_WMAX_RATIO
        else:
            w_max = self.line_end_w_max_um
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
            return {"metal_density": metal, "perimeter_density": perim,
                    "line_end_density": np.zeros(n, dtype=float)}

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

        return {"metal_density": metal, "perimeter_density": perim,
                "line_end_density": self._line_end_density(spec, grid)}

    def _line_end_density(self, spec: LayerSpec, grid: Grid) -> np.ndarray:
        """Line-end count per unit area, counted per window.

        Line ends are points, so they are counted directly rather than routed
        through an area proxy. The Calibre path converts them to fixed-size
        markers instead, because there the moving window is the DENSITY
        primitive and only understands area.
        """
        pts = self.line_ends(spec)
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
        pts = self.line_ends(spec)
        n_ends = 0
        if len(pts):
            n_ends = int(((pts[:, 0] >= x0) & (pts[:, 0] < x1)
                          & (pts[:, 1] >= y0) & (pts[:, 1] < y1)).sum())
        return {
            "metal_density": u.area_dbu2_to_um2((region & win).area()) / area,
            "perimeter_density": u.length_dbu_to_um((edges & win).length()) / area,
            "line_end_density": n_ends / area,
        }
