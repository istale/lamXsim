"""GDS/OASIS reader (spec section 3), KLayout backend.

Physical coordinates and layer/datatype identity are preserved; the caller
sees micrometres only. Regions are cached per layer because merging a
full-chip layer is the expensive step and every scale re-uses it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import klayout.db as db

from ..units import Units


@dataclass(frozen=True)
class LayerSpec:
    """A named layer/datatype pair from the config's layer map."""
    name: str          # e.g. "M8"
    layer: int
    datatype: int = 0

    @property
    def key(self) -> tuple[int, int]:
        return (self.layer, self.datatype)

    def __str__(self) -> str:
        return f"{self.name}({self.layer}/{self.datatype})"


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box in micrometres."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        return self.width * self.height


class LayoutReader:
    """Reads a layout and hands out merged, flattened Regions per layer.

    Flattening is deliberate for V1: every feature in spec section 4 is defined on
    the as-drawn union of shapes, and hierarchy would make window clipping
    ambiguous. Hierarchy preservation is a Phase-2 concern and the layer
    Regions are the only thing downstream code depends on, so it can be
    swapped for a DeepShapeStore without touching the feature code.
    """

    def __init__(self, path: str | Path, top_cell: str | None = None):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.layout = db.Layout()
        self.layout.read(str(self.path))
        self.units = Units(self.layout.dbu)

        if top_cell is not None:
            cell = self.layout.cell(top_cell)
            if cell is None:
                raise ValueError(f"top cell {top_cell!r} not found in {self.path}")
            self.top = cell
        else:
            tops = self.layout.top_cells()
            if len(tops) != 1:
                names = [c.name for c in tops]
                raise ValueError(
                    f"{self.path} has {len(tops)} top cells {names}; "
                    "specify top_cell explicitly"
                )
            self.top = tops[0]

        self._region_cache: dict[tuple[int, int], db.Region] = {}
        self._edge_cache: dict[tuple[int, int], db.Edges] = {}

    # -- introspection ------------------------------------------------
    def available_layers(self) -> list[tuple[int, int]]:
        out = []
        for idx in self.layout.layer_indexes():
            info = self.layout.get_info(idx)
            out.append((info.layer, info.datatype))
        return sorted(out)

    def bbox(self) -> BBox:
        b = self.top.bbox()
        u = self.units
        return BBox(
            u.dbu_to_um(b.left), u.dbu_to_um(b.bottom),
            u.dbu_to_um(b.right), u.dbu_to_um(b.top),
        )

    # -- geometry -----------------------------------------------------
    def region(self, spec: LayerSpec) -> db.Region:
        """Merged Region for *spec*. Empty Region if the layer is absent."""
        if spec.key in self._region_cache:
            return self._region_cache[spec.key]
        idx = self.layout.find_layer(spec.layer, spec.datatype)
        region = db.Region()
        if idx is not None:
            # insert() copies the shapes in. Constructing the Region directly
            # from the iterator leaves it lazily bound to this Layout, so it
            # silently empties if the reader is garbage collected -- which
            # happens whenever a caller writes LayoutReader(p).region(...) as
            # a one-liner. The failure mode is all-zero features, not an error.
            region.insert(self.top.begin_shapes_rec(idx))
            region.merge()
        self._region_cache[spec.key] = region
        return region

    def edges(self, spec: LayerSpec) -> db.Edges:
        """True metal/dielectric boundary edges of the merged layer.

        These are cached separately from the Region because window-local
        perimeter must be computed by clipping *edges*, never by clipping the
        Region and taking its perimeter -- the latter counts the window cut
        itself as metal boundary and inflates perimeter_density (a 30x10um bar
        cut at x=15um reports 50um instead of the true 40um).
        """
        if spec.key in self._edge_cache:
            return self._edge_cache[spec.key]
        e = self.region(spec).edges()
        self._edge_cache[spec.key] = e
        return e
