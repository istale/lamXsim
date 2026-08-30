"""Reproduce the generated deck's operator sequence with KLayout.

Why this exists: nothing in the repository ever ran the Calibre path
end-to-end. The band correction was measured to 0.0000 % on isolated
patterns, but the deck -> RDB -> grid -> feature chain had no test at all, so
a wrong ``eps``, a window/grid offset, or a marker read as a density would
have produced a plausible map that no test would question.

What this is: an executable statement of what each generated SVRF rule
*means*, written in KLayout region algebra, emitting the same RDB files the
deck names. It lets the ingest path, the conversions and the grid alignment
be tested, and it lets a user dry-run the whole flow without a Calibre
licence.

What this is **not**: Calibre. It cannot detect a difference between our
reading of SVRF and Mentor's implementation of it -- if the deck's
``SIZE ... BY -eps`` does something other than what ``Region.sized`` does,
both sides here are wrong together and agree. Only a run against the real
tool settles that, and until one happens the deck's rules stay marked
unverified in the header it prints.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import klayout.db as db
import numpy as np

from ..features import corners as corners_mod
from ..features.grid import Grid, build_grid
from ..layout.reader import LayerSpec, LayoutReader
from . import svrf as svrf_mod
from .svrf import CalibreLayer

#: Side of the square marker dropped at each counted point, in um. Only the
#: marker *list* is used downstream, so this affects nothing but readability
#: of the RDB; a density-based count would have to divide by its area.
MARKER_SIDE_UM = 0.02


def _window_area_density(region: db.Region, grid: Grid, u) -> np.ndarray:
    """Area of *region* inside each window, over window area.

    This is what ``DENSITY <layer> WINDOW w w STEP s s`` reports. Windows are
    taken a grid row at a time against a pre-clipped strip, for the same
    reason the KLayout extractor does it: intersecting each window against the
    whole layer costs ~35x more and scales with die area.
    """
    out = np.zeros(len(grid), dtype=float)
    if region.is_empty():
        return out
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
        strip = region & db.Region(db.Box(x_lo, y0, x_hi, y1))
        if strip.is_empty():
            continue
        for cell in cells:
            win = db.Region(db.Box(u.um_to_dbu(cell.x0), u.um_to_dbu(cell.y0),
                                   u.um_to_dbu(cell.x1), u.um_to_dbu(cell.y1)))
            out[cell.cell_id] = (u.area_dbu2_to_um2((strip & win).area())
                                 / cell.area_um2)
    return out


def _write_density_rdb(path: Path, grid: Grid, values: np.ndarray,
                       check: str) -> None:
    """Write the rectangle-plus-value records ``read_density_rdb`` parses.

    Windows whose value is zero are omitted, because Calibre omits them; the
    ingest side turns an absent window back into 0.0. Round-tripping through
    the omission is part of what needs testing.
    """
    lines = [f"// {check}"]
    for cell in grid.cells:
        v = float(values[cell.cell_id])
        if v == 0.0:
            continue
        lines.append(f"{cell.x0:.6f} {cell.y0:.6f} {cell.x1:.6f} "
                     f"{cell.y1:.6f} {v:.10g}")
    path.write_text("\n".join(lines) + "\n")


def _write_marker_rdb(path: Path, points_um: np.ndarray, check: str,
                      side_um: float = MARKER_SIDE_UM) -> None:
    """Write one square record per counted point."""
    h = side_um / 2
    lines = [f"// {check}"]
    for x, y in np.asarray(points_um, dtype=float).reshape(-1, 2):
        lines.append(f"{x - h:.6f} {y - h:.6f} {x + h:.6f} {y + h:.6f}")
    path.write_text("\n".join(lines) + "\n")


@dataclass
class EmulatedRun:
    """Where each emulated output landed, and what it stands for."""
    outdir: Path
    density: dict[tuple[str, float, str], Path] = field(default_factory=dict)
    markers: dict[tuple[str, str], Path] = field(default_factory=dict)
    eps_um: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def density_files(self, layer: str, scale_um: float) -> dict[str, str]:
        return {kind: str(p) for (l, s, kind), p in self.density.items()
                if l == layer and s == scale_um}


def run(gds_path: str, layers: list[CalibreLayer], *,
        scales_um=(100.0,), step_ratio: float = 0.5,
        outdir: str | Path = "calibre_out",
        min_width_um: dict[str, float] | None = None,
        top_cell: str | None = None) -> EmulatedRun:
    """Emulate the deck over *gds_path* and write its outputs to *outdir*."""
    reader = LayoutReader(gds_path, top_cell=top_cell)
    u = reader.units
    bbox = reader.bbox()
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    result = EmulatedRun(outdir=out)

    for layer in layers:
        spec = LayerSpec(layer.name, layer.layer, layer.datatype)
        region = reader.region(spec)
        if layer.is_via:
            # via_count_density counts one point per via, so the emulated
            # output is the via list rather than an area fraction. The
            # KLayout extractor counts centroids on a half-open cell, and the
            # marker reader reproduces that rule.
            # Same centroid rule as ViaExtractor.centroids: the bbox centre
            # in database units, scaled. Rounding to the dbu grid first would
            # move a via off a cell boundary and change its count.
            pts = [[(b.left + b.right) / 2 * u.dbu, (b.bottom + b.top) / 2 * u.dbu]
                   for b in (poly.bbox() for poly in region.each())]
            path = out / f"via_marker_{layer.name}.rdb"
            _write_marker_rdb(path, np.array(pts, float).reshape(-1, 2),
                              f"VIA_MARKER_{layer.name}")
            result.markers[(layer.name, "via_marker")] = path
        else:
            eps = layer.eps_um
            result.eps_um[layer.name] = eps
            band = region - region.sized(-u.um_to_dbu(eps))
            convex, concave = corners_mod.classify(region)
            for kind, pts in (("convex_corner", convex),
                              ("concave_corner", concave)):
                arr = np.array([[u.dbu_to_um(p.x), u.dbu_to_um(p.y)]
                                for p in pts], float).reshape(-1, 2)
                path = out / f"{kind}_{layer.name}.rdb"
                _write_marker_rdb(path, arr, f"{kind.upper()}_{layer.name}")
                result.markers[(layer.name, kind)] = path

            # Geometry narrower than w, by morphological opening. Not a
            # line-end proxy: opening by w/2 erases a line of width w, so on
            # an array of 1um lines it returns the whole array. Written under
            # the name of what it measures.
            w = (min_width_um or {}).get(layer.name, layer.min_width_um)
            h = max(u.um_to_dbu(w / 2), 1)
            narrow = region - region.sized(-h).sized(h)

        for scale in scales_um:
            grid = build_grid(bbox, float(scale), stride_um=float(scale) * step_ratio)
            tag = f"{layer.name}_{scale:g}um"
            todo = [("via_density" if layer.is_via else "metal_density", region)]
            if not layer.is_via:
                todo += [("perimeter_band", band),
                         ("narrow_structure", narrow)]
            for kind, r in todo:
                path = out / f"{kind}_{tag}.rdb"
                _write_density_rdb(path, grid, _window_area_density(r, grid, u),
                                   f"DENSITY_{kind.upper()}_{tag}")
                result.density[(layer.name, float(scale), kind)] = path

    # The same sidecar the real deck writes, so the ingest side reads one
    # format and cannot tell an emulated run from a real one by accident --
    # it is told, by the "emulated" flag, which is the honest way round.
    side = svrf_mod.extraction_manifest(layers, scales_um, step_ratio, str(out))
    side["generator"] = "lamxsim.calibre.emulate"
    side["emulated"] = True
    (out / "extraction_manifest.json").write_text(json.dumps(side, indent=2))

    result.notes.append(
        "emulated with KLayout region algebra, not run through Calibre; "
        "this checks the ingest and conversion path, not the tool")
    return result
