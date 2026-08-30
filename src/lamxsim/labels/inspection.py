"""Inspection footprint and control opportunity.

A case-control design needs a denominator. The pipeline's default -- every
cell without a recorded failure is a control -- is only valid where inspection
actually had a chance to find one. Acoustic scans cover regions; FIB
cross-sections are selected, and selected towards die corners and known-risk
structures. So uninspected area silently becomes control area, and any layout
feature correlated with where engineers chose to look acquires an association
that has nothing to do with mechanics.

That bias is not fixable downstream. Block permutation, FDR and the
position-only baseline all operate on whatever population they are handed; if
the population is wrong they are wrong together.

Cells outside the footprint are therefore excluded from **both** cases and
controls. Full-die coverage remains available but has to be asserted
deliberately, with a justification recorded in the run metadata, rather than
being what happens when nobody supplies a footprint.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import klayout.db as db
import numpy as np

from ..layout.reader import BBox, LayerSpec, LayoutReader


@dataclass
class InspectionFootprint:
    """Where inspection had a real opportunity to find a failure."""
    region: db.Region
    source: str
    assumed_full_coverage: bool = False
    justification: str = ""
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_gds_layer(cls, reader: LayoutReader, spec: LayerSpec
                       ) -> "InspectionFootprint":
        """Footprint drawn as polygons on a GDS layer."""
        region = reader.region(spec)
        if region.is_empty():
            raise ValueError(
                f"inspection footprint layer {spec} is empty; supply the "
                "inspected area or assert full coverage explicitly")
        return cls(region=region, source=f"gds_layer:{spec}")

    @classmethod
    def from_rectangles(cls, rects, *, dbu: float = 0.001, source: str = "rectangles"
                        ) -> "InspectionFootprint":
        """Footprint as (x0, y0, x1, y1) boxes in um -- e.g. scan frames."""
        region = db.Region()
        for x0, y0, x1, y1 in rects:
            region.insert(db.Box(int(round(x0 / dbu)), int(round(y0 / dbu)),
                                 int(round(x1 / dbu)), int(round(y1 / dbu))))
        region.merge()
        if region.is_empty():
            raise ValueError("no inspection rectangles supplied")
        return cls(region=region, source=source)

    @classmethod
    def full_die(cls, bbox: BBox, justification: str, *, dbu: float = 0.001
                 ) -> "InspectionFootprint":
        """Assert that the whole die was inspected.

        The justification is required and travels into the run metadata,
        because "we inspected everything" is a claim about the measurement
        campaign that someone has to own.
        """
        if not justification.strip():
            raise ValueError(
                "full-die coverage must be justified: state how the whole die "
                "was inspected and called, or supply the real footprint")
        region = db.Region(db.Box(int(round(bbox.xmin / dbu)),
                                  int(round(bbox.ymin / dbu)),
                                  int(round(bbox.xmax / dbu)),
                                  int(round(bbox.ymax / dbu))))
        return cls(region=region, source="full_die",
                   assumed_full_coverage=True, justification=justification)

    def area_um2(self, dbu: float = 0.001) -> float:
        return self.region.area() * dbu * dbu

    def report(self, dbu: float = 0.001) -> dict:
        return {"source": self.source,
                "assumed_full_coverage": self.assumed_full_coverage,
                "justification": self.justification,
                "area_um2": round(self.area_um2(dbu), 3),
                "notes": self.notes}


def coverage(footprint: InspectionFootprint, grid, *, dbu: float = 0.001
             ) -> np.ndarray:
    """Fraction of each cell that lies inside the inspected footprint."""
    out = np.zeros(len(grid))
    rb = footprint.region.bbox()
    rows: dict[int, list] = {}
    for c in grid.cells:
        rows.setdefault(c.row, []).append(c)

    def u(v):
        return int(round(v / dbu))

    for cells in rows.values():
        y0, y1 = u(cells[0].y0), u(cells[0].y1)
        if y1 <= rb.bottom or y0 >= rb.top:
            continue
        strip = footprint.region & db.Region(
            db.Box(rb.left - 1, y0, rb.right + 1, y1))
        if strip.is_empty():
            continue
        for c in cells:
            win = db.Region(db.Box(u(c.x0), u(c.y0), u(c.x1), u(c.y1)))
            out[c.cell_id] = ((strip & win).area() * dbu * dbu) / c.area_um2
    return np.clip(out, 0.0, 1.0)


def eligibility(footprint: InspectionFootprint, grid, *,
                min_coverage: float = 0.5, dbu: float = 0.001
                ) -> tuple[np.ndarray, np.ndarray]:
    """Which cells may take part in the analysis, and their coverage.

    A cell only partly inspected is a weaker control than a fully inspected
    one, and there is no way to express "half a control" in a binary label, so
    the threshold excludes it rather than pretending.
    """
    frac = coverage(footprint, grid, dbu=dbu)
    return frac >= min_coverage, frac


def audit_failures(footprint: InspectionFootprint, failures, *,
                   dbu: float = 0.001) -> dict:
    """Check that every recorded failure lies inside the inspected footprint.

    A failure outside it is a contradiction -- something was found where
    nothing was looked at -- and means the footprint, the registration or the
    coordinate frame is wrong. It is reported rather than quietly tolerated,
    because each of those three causes invalidates a different part of the
    analysis.
    """
    x = failures.table["x_um"].to_numpy(float)
    y = failures.table["y_um"].to_numpy(float)
    inside = np.zeros(len(x), dtype=bool)
    for i, (xi, yi) in enumerate(zip(x, y)):
        px, py = int(round(xi / dbu)), int(round(yi / dbu))
        probe = db.Region(db.Box(px, py, px + 1, py + 1))
        inside[i] = not (footprint.region & probe).is_empty()

    outside = np.where(~inside)[0]
    ids = failures.table["sample_id"].astype(str).to_numpy()
    return {
        "n_failures": len(x),
        "n_inside_footprint": int(inside.sum()),
        "n_outside_footprint": int((~inside).sum()),
        "outside_sample_ids": list(ids[outside][:10]),
        "consistent": bool(inside.all()),
    }
