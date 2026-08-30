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
from __future__ import annotations

import klayout.db as db
import numpy as np

from ..evidence import EvidenceClass
from ..layout.reader import LayerSpec, LayoutReader
from .grid import Grid

FEATURES = ("horizontal_fraction", "vertical_fraction", "orientation_anisotropy")
EVIDENCE_CLASS = EvidenceClass.GDS_GEOMETRY


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

    def _edges(self, spec: LayerSpec):
        if spec.key not in self._split_cache:
            self._split_cache[spec.key] = _split(self.reader.edges(spec))
        return self._split_cache[spec.key]

    def _measure(self, h, v, other, win: db.Region) -> dict[str, float]:
        u = self.u
        h_len = u.length_dbu_to_um((h & win).length())
        v_len = u.length_dbu_to_um((v & win).length())
        if not (other & win).is_empty():
            ox, oy = _projected_lengths(other & win, u.dbu)
            h_len += ox
            v_len += oy
        total = h_len + v_len
        if total <= 0:
            return {"horizontal_fraction": 0.0, "vertical_fraction": 0.0,
                    "orientation_anisotropy": 0.0}
        return {"horizontal_fraction": h_len / total,
                "vertical_fraction": v_len / total,
                "orientation_anisotropy": anisotropy(h_len, v_len)}

    def extract(self, spec: LayerSpec, grid: Grid) -> dict[str, np.ndarray]:
        h, v, other = self._edges(spec)
        n = len(grid)
        out = {k: np.zeros(n) for k in FEATURES}
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
