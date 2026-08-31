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
from __future__ import annotations

import klayout.db as db
import numpy as np

from ..evidence import EvidenceClass
from ..layout.reader import LayerSpec, LayoutReader
from .grid import Grid, point_accumulate

FEATURES = ("via_density", "via_count_density", "mean_via_area")
EVIDENCE_CLASS = EvidenceClass.GDS_GEOMETRY


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
        out = {k: np.zeros(n) for k in FEATURES}
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
            return dict.fromkeys(FEATURES, 0.0)
        inside = ((pts[:, 0] >= x0) & (pts[:, 0] < x1)
                  & (pts[:, 1] >= y0) & (pts[:, 1] < y1))
        count = int(inside.sum())
        return {
            "via_density": u.area_dbu2_to_um2((region & win).area()) / area,
            "via_count_density": count / area,
            "mean_via_area": float(areas[inside].mean()) if count else 0.0,
        }
