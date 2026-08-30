"""Read Calibre density output back into the analysis grid.

Calibre reports an area fraction per window. Marker-layer densities have to be
converted back into the physical quantity they stand for before anything
downstream sees them, and the conversion factor has to travel with the data --
a band density silently treated as a perimeter density is wrong by 1/eps,
which is a factor of 20 to 40 in practice.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..evidence import EvidenceClass
from ..features.grid import Grid, build_grid
from ..layout.reader import BBox


@dataclass(frozen=True)
class MarkerConversion:
    """How a Calibre marker-layer density maps to a physical feature."""
    feature: str
    unit: str
    #: value = calibre_density * factor. For a perimeter band, factor = 1/eps.
    factor: float
    note: str = ""


def perimeter_conversion(eps_um: float) -> MarkerConversion:
    return MarkerConversion(
        feature="perimeter_density", unit="um^-1", factor=1.0 / eps_um,
        note=f"inside band, eps={eps_um:g}um; boundary_length = band_area/eps")


def area_conversion(feature: str) -> MarkerConversion:
    return MarkerConversion(feature=feature, unit="dimensionless", factor=1.0,
                            note="native area density")


def count_conversion(feature: str, marker_area_um2: float) -> MarkerConversion:
    """Fixed-size corner/line-end markers: area density -> count per area."""
    return MarkerConversion(
        feature=feature, unit="um^-2", factor=1.0 / marker_area_um2,
        note=f"marker area {marker_area_um2:g}um^2; count = marker_area_total/marker_area")


_RDB_VALUE = re.compile(
    r"^\s*(?P<x0>-?[\d.]+)\s+(?P<y0>-?[\d.]+)\s+(?P<x1>-?[\d.]+)\s+(?P<y1>-?[\d.]+)"
    r"(?:\s+(?P<val>-?[\d.eE+]+))?\s*$")


def read_density_csv(path: str | Path, *, x_col="x_um", y_col="y_um",
                     value_col="value") -> pd.DataFrame:
    """Read a simple x,y,value density dump."""
    df = pd.read_csv(path)
    missing = [c for c in (x_col, y_col, value_col) if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; got {list(df.columns)}")
    return df.rename(columns={x_col: "x_um", y_col: "y_um", value_col: "value"})


def read_density_rdb(path: str | Path) -> pd.DataFrame:
    """Read the rectangle-plus-value subset of a Calibre ASCII RDB.

    Only the geometry and value lines are interpreted; check names and cell
    headers are ignored. Window centres come from the rectangle, so a deck
    whose STEP differs from its WINDOW still lands on the right coordinates.
    """
    xs, ys, vs = [], [], []
    for line in Path(path).read_text().splitlines():
        m = _RDB_VALUE.match(line)
        if not m:
            continue
        x0, y0, x1, y1 = (float(m.group(k)) for k in ("x0", "y0", "x1", "y1"))
        if x1 <= x0 or y1 <= y0:
            continue
        xs.append((x0 + x1) / 2)
        ys.append((y0 + y1) / 2)
        v = m.group("val")
        vs.append(float(v) if v is not None else np.nan)
    if not xs:
        raise ValueError(f"{path}: no rectangle records recognised")
    return pd.DataFrame({"x_um": xs, "y_um": ys, "value": vs})


def to_grid(df: pd.DataFrame, grid: Grid, conversion: MarkerConversion,
            *, tol_um: float | None = None) -> np.ndarray:
    """Snap Calibre window values onto *grid*, applying the conversion.

    Unreported windows become 0.0, which is what Calibre's omission of empty
    windows means -- but the count of them is checked against the grid so a
    coordinate-frame mismatch shows up as a near-empty map rather than as a
    map of zeros that looks plausible.
    """
    tol = (grid.stride_um / 2) if tol_um is None else tol_um
    out = np.zeros(len(grid), dtype=float)

    cx = np.array([c.x_center for c in grid.cells])
    cy = np.array([c.y_center for c in grid.cells])
    order = np.lexsort((cx, cy))
    key = {(round(cx[i] / tol), round(cy[i] / tol)): i for i in order}

    hit = 0
    for x, y, v in zip(df["x_um"], df["y_um"], df["value"]):
        i = key.get((round(x / tol), round(y / tol)))
        if i is None:
            continue
        out[i] = v * conversion.factor
        hit += 1

    if hit == 0:
        raise ValueError(
            "no Calibre window matched a grid cell; the layout origin or the "
            "WINDOW/STEP in the deck does not match this grid")
    return out


@dataclass
class CalibreFeatureSet:
    """Calibre-derived features on one grid, with provenance."""
    grid: Grid
    layer: str
    scale_um: float
    values: dict[str, np.ndarray]
    conversions: dict[str, MarkerConversion]
    evidence_class: EvidenceClass = EvidenceClass.GDS_GEOMETRY
    source_files: dict[str, str] | None = None

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.grid.to_arrays())
        df["layer"] = self.layer
        for k, v in self.values.items():
            df[k] = v
        return df

    def provenance(self) -> dict:
        return {
            "layer": self.layer,
            "scale_um": self.scale_um,
            "n_windows": len(self.grid),
            "overlapping_windows": self.grid.overlapping,
            "stride_um": self.grid.stride_um,
            "conversions": {k: {"unit": c.unit, "factor": c.factor, "note": c.note}
                            for k, c in self.conversions.items()},
            "source_files": self.source_files or {},
        }


def load_scale(bbox: BBox, scale_um: float, layer: str, files: dict[str, str],
               *, eps_um: float, step_ratio: float = 0.5,
               reader=read_density_rdb) -> CalibreFeatureSet:
    """Load one layer x scale from a set of Calibre output files.

    ``files`` maps a marker kind to a path, e.g.
    ``{"metal_density": ..., "perimeter_band": ..., "convex_corner": ...}``.
    """
    grid = build_grid(bbox, scale_um, stride_um=scale_um * step_ratio)
    conv = {
        "metal_density": area_conversion("metal_density"),
        "perimeter_band": perimeter_conversion(eps_um),
        "convex_corner": area_conversion("convex_corner_density"),
        "concave_corner": area_conversion("concave_corner_density"),
    }
    values, used = {}, {}
    for kind, path in files.items():
        if kind not in conv:
            raise ValueError(f"unknown marker kind {kind!r}; known: {sorted(conv)}")
        c = conv[kind]
        values[c.feature] = to_grid(reader(path), grid, c)
        used[c.feature] = c
    return CalibreFeatureSet(grid=grid, layer=layer, scale_um=scale_um,
                             values=values, conversions=used,
                             source_files=dict(files))


def perimeter_from_band_and_corners(band_density: np.ndarray,
                                    convex_density: np.ndarray,
                                    concave_density: np.ndarray,
                                    *, eps_um: float,
                                    corner_marker_area_um2: float) -> np.ndarray:
    """Exact perimeter density from Calibre's band and corner marker densities.

    The inside band under-reports by eps^2 at each convex corner and
    over-reports by the same at each concave one, so

        P = band_area/eps + eps * (n_convex - n_concave)

    recovers the true boundary length exactly on Manhattan geometry. Dividing
    through by window area turns every term into the density Calibre reports:

        PD = band_density/eps + eps * (convex_density - concave_density) / a_marker

    ``eps_um`` must be the database-unit-snapped value the deck actually used,
    not the nominal one.
    """
    return (band_density / eps_um
            + eps_um * (convex_density - concave_density) / corner_marker_area_um2)
