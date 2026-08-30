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
from __future__ import annotations

import klayout.db as db
import numpy as np

from ..evidence import EvidenceClass
from ..layout.reader import LayerSpec, LayoutReader
from .grid import Grid

EVIDENCE_CLASS = EvidenceClass.GDS_GEOMETRY

FEATURES = ("wide_metal_fraction", "wide_metal_perimeter_density",
            "slot_density", "slotted_metal_fraction",
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
        out = (region, wide, wide.edges(),
               np.array(slot_points, dtype=float).reshape(-1, 2))
        self._cache[spec.key] = out
        return out

    def extract(self, spec: LayerSpec, grid: Grid) -> dict[str, np.ndarray]:
        region, wide, wide_edges, slot_points = self._derived(spec)
        n = len(grid)
        out = {k: np.zeros(n) for k in FEATURES}
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
        region, wide, wide_edges, slot_points = self._derived(spec)
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
            "fill_density": fill_area / area,
            "fill_fraction": fill_area / total if total > 0 else 0.0,
        }
