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


def _data_lines(path: str | Path) -> list[str]:
    """Record lines only -- comments, blanks and check/cell headers dropped.

    A scan that matched nothing produces a file with no record lines, and that
    is a real answer: every window is empty, so every value is zero. A file
    whose record lines do not parse is a different thing entirely, and the two
    used to be indistinguishable -- both surfaced as "no records recognised",
    so a format change would have read as an empty layer.
    """
    out = []
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("//", "#")):
            continue
        if not stripped[0].isdigit() and not stripped.startswith("-"):
            continue  # check name / cell header
        out.append(line)
    return out


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
    for line in _data_lines(path):
        m = _RDB_VALUE.match(line)
        if not m:
            raise ValueError(f"{path}: unparseable record {line.strip()!r}")
        x0, y0, x1, y1 = (float(m.group(k)) for k in ("x0", "y0", "x1", "y1"))
        if x1 <= x0 or y1 <= y0:
            continue
        xs.append((x0 + x1) / 2)
        ys.append((y0 + y1) / 2)
        v = m.group("val")
        vs.append(float(v) if v is not None else np.nan)
    return pd.DataFrame({"x_um": xs, "y_um": ys, "value": vs},
                        columns=["x_um", "y_um", "value"])


def to_grid(df: pd.DataFrame, grid: Grid, conversion: MarkerConversion,
            *, tol_um: float | None = None) -> np.ndarray:
    """Snap Calibre window values onto *grid*, applying the conversion.

    Matching is nearest-centre, then a distance check against the tolerance.
    Bucketing the coordinates instead -- rounding both sides to a common grid
    and comparing keys -- rejects pairs that are well inside the tolerance
    whenever they straddle a bucket edge: at a 50 um tolerance a window 26 um
    from its cell centre lands in the neighbouring bucket and is dropped.

    Unreported windows become 0.0, which is what Calibre's omission of empty
    windows means. Individual records further than the tolerance from any cell
    are dropped, so a grid may cover a sub-region of what the deck reported;
    a *complete* failure to match raises, so a coordinate-frame mismatch
    surfaces as an error rather than as a map of zeros that looks plausible.
    """
    from scipy.spatial import cKDTree

    tol = (grid.stride_um / 2) if tol_um is None else float(tol_um)
    out = np.zeros(len(grid), dtype=float)

    centres = np.column_stack([[c.x_center for c in grid.cells],
                               [c.y_center for c in grid.cells]])
    points = df[["x_um", "y_um"]].to_numpy(float).reshape(-1, 2)
    if len(points) == 0:
        # An empty scan means every window was empty, which is a map of
        # zeros. This is only safe to assume because the reader now
        # distinguishes an empty file from an unparseable one.
        return out

    distance, index = cKDTree(centres).query(points, k=1)
    matched = distance <= tol

    values = df["value"].to_numpy(float) * conversion.factor

    # Calibre emits one record per window, so two records claiming the same
    # cell means the deck and the grid disagree about WINDOW/STEP. Letting the
    # later record win would leave the other cell reading zero -- a value that
    # is indistinguishable from an empty window.
    hit = index[matched]
    if len(hit) != len(np.unique(hit)):
        cells, counts = np.unique(hit, return_counts=True)
        clashing = cells[counts > 1]
        example = grid.cells[int(clashing[0])]
        raise ValueError(
            f"{len(clashing)} grid cell(s) were claimed by more than one "
            f"Calibre record; for example the cell centred at "
            f"({example.x_center:g}, {example.y_center:g})um was claimed "
            f"{int(counts[counts > 1][0])} times. Calibre emits one record per "
            "window, so this means the deck's WINDOW/STEP does not match this "
            f"grid (scale {grid.scale_um:g}um, stride {grid.stride_um:g}um).")

    out[hit] = values[matched]

    if not matched.any():
        raise ValueError(
            f"no Calibre window matched a grid cell within {tol:g}um "
            f"(closest was {distance.min():.3g}um); the layout origin or the "
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
               reader=read_density_rdb, marker_files: dict[str, str] | None = None,
               grid: Grid | None = None) -> CalibreFeatureSet:
    """Load one layer x scale from a set of Calibre output files.

    ``files`` maps a density-scan kind to a path, e.g.
    ``{"metal_density": ..., "perimeter_band": ..., "via_density": ...}``.
    ``marker_files`` maps a count kind -- ``convex_corner``, ``concave_corner``,
    ``via_marker`` -- to a marker RDB, which is counted rather than scaled.

    When both corner marker files and a perimeter band are present the exact
    corner correction is applied, so ``perimeter_density`` is the corrected
    value and the uncorrected band is kept beside it under
    ``perimeter_density_band_only``. Without the corners the band value is
    still returned, but it is low wherever convex corners outnumber concave
    ones -- 5.7 % on one of the benchmark patterns -- so the two are never
    given the same name.
    """
    if grid is None:
        grid = build_grid(bbox, scale_um, stride_um=scale_um * step_ratio)
    elif abs(grid.stride_um - scale_um * step_ratio) > 1e-9:
        raise ValueError(
            f"the deck stepped {scale_um * step_ratio:g}um at a {scale_um:g}um "
            f"window and this grid strides {grid.stride_um:g}um. The windows "
            "do not line up, so every value would be attributed to the wrong "
            "cell or rejected as a duplicate claim. Regenerate the deck with "
            f"step_ratio={grid.stride_um / scale_um:g}.")
    density_conv = {
        "metal_density": area_conversion("metal_density"),
        "via_density": area_conversion("via_density"),
        "perimeter_band": perimeter_conversion(eps_um),
        "narrow_structure": area_conversion("narrow_structure_density"),
    }
    count_features = {
        "convex_corner": "convex_corner_density",
        "concave_corner": "concave_corner_density",
        "via_marker": "via_count_density",
    }

    values, used = {}, {}
    for kind, path in files.items():
        if kind not in density_conv:
            raise ValueError(f"unknown density kind {kind!r}; "
                             f"known: {sorted(density_conv)}")
        c = density_conv[kind]
        values[c.feature] = to_grid(reader(path), grid, c)
        used[c.feature] = c

    for kind, path in (marker_files or {}).items():
        if kind not in count_features:
            raise ValueError(f"unknown marker kind {kind!r}; "
                             f"known: {sorted(count_features)}")
        feature = count_features[kind]
        values[feature] = to_count_grid(read_marker_rdb(path), grid)
        used[feature] = MarkerConversion(
            feature=feature, unit="um^-2", factor=1.0,
            note="counted from the marker list; no marker-area assumption")

    if {"convex_corner_density", "concave_corner_density"} <= values.keys() \
            and "perimeter_density" in values:
        band_only = values["perimeter_density"]
        values["perimeter_density_band_only"] = band_only
        values["perimeter_density"] = band_only + eps_um * (
            values["convex_corner_density"] - values["concave_corner_density"])
        used["perimeter_density"] = MarkerConversion(
            feature="perimeter_density", unit="um^-1", factor=1.0 / eps_um,
            note=f"inside band (eps={eps_um:g}um) plus the exact corner "
                 "correction eps*(convex - concave)")

    if "convex_corner_density" in values and "concave_corner_density" in values:
        values["corner_density"] = (values["convex_corner_density"]
                                    + values["concave_corner_density"])
        used["corner_density"] = MarkerConversion(
            feature="corner_density", unit="um^-2", factor=1.0,
            note="convex + concave marker counts")

    return CalibreFeatureSet(grid=grid, layer=layer, scale_um=scale_um,
                             values=values, conversions=used,
                             source_files={**dict(files),
                                           **dict(marker_files or {})})


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


def read_marker_rdb(path: str | Path) -> pd.DataFrame:
    """Read a marker RDB as a list of individual markers, not as a density.

    A count feature -- vias, corners, line ends -- is a point count, and the
    density scanner cannot report one: it reports the area fraction of the
    marker layer, so recovering a count needs the markers to be uniform in
    area and needs that area to be known. Neither is checkable from the
    density output itself, and an error in the assumed marker area scales
    every count by a constant that no downstream test would notice.

    Reporting the markers themselves removes the assumption. It costs one
    record per marker instead of one per window, which is why it is the right
    trade for line ends and vias and the wrong one for a full-chip corner
    layer -- see ``to_count_grid`` for the density-based alternative.
    """
    xs, ys, areas = [], [], []
    for line in _data_lines(path):
        m = _RDB_VALUE.match(line)
        if not m:
            raise ValueError(f"{path}: unparseable record {line.strip()!r}")
        x0, y0, x1, y1 = (float(m.group(k)) for k in ("x0", "y0", "x1", "y1"))
        if x1 <= x0 or y1 <= y0:
            continue
        xs.append((x0 + x1) / 2)
        ys.append((y0 + y1) / 2)
        areas.append((x1 - x0) * (y1 - y0))
    return pd.DataFrame({"x_um": xs, "y_um": ys, "area_um2": areas})


def to_count_grid(markers: pd.DataFrame, grid: Grid) -> np.ndarray:
    """Marker count per unit area on *grid*, from a marker list.

    A marker is assigned to the cell containing its centroid, on the same
    half-open convention the KLayout via extractor uses, so a marker sitting
    exactly on a cell boundary is counted once rather than twice or not at
    all. With an overlapping grid a marker belongs to every window that
    contains it, which is what the moving-window density would also report.
    """
    out = np.zeros(len(grid), dtype=float)
    if markers.empty:
        return out
    px = markers["x_um"].to_numpy(float)
    py = markers["y_um"].to_numpy(float)
    for c in grid.cells:
        inside = ((px >= c.x0) & (px < c.x1) & (py >= c.y0) & (py < c.y1))
        out[c.cell_id] = int(inside.sum()) / c.area_um2
    return out


#: Features the deck can supply. Everything else in the atlas -- orientation,
#: gradients, cross-layer terms, position, package context -- is computed in
#: Python either way, so a Calibre run replaces part of the extraction, never
#: all of it.
CALIBRE_SUPPLIED = (
    "metal_density", "perimeter_density", "convex_corner_density",
    "concave_corner_density", "corner_density", "via_density",
    "via_count_density", "narrow_structure_density",
)


@dataclass
class CalibreSource:
    """A directory of deck output, addressable by (layer, scale)."""
    directory: Path
    manifest: dict
    density: dict
    markers: dict

    @property
    def emulated(self) -> bool:
        return bool(self.manifest.get("emulated"))

    def eps_um(self, layer: str) -> float:
        for entry in self.manifest["layers"]:
            if entry["name"] == layer:
                return float(entry["eps_um"])
        raise KeyError(f"{layer!r} is not in the extraction manifest; "
                       f"it has {[e['name'] for e in self.manifest['layers']]}")

    def layers(self) -> list[str]:
        return [e["name"] for e in self.manifest["layers"]]

    def scales_um(self) -> list[float]:
        return [float(s) for s in self.manifest["scales_um"]]

    def features_for(self, layer: str, scale_um: float, bbox: BBox,
                     grid: Grid | None = None) -> dict[str, np.ndarray]:
        """Every feature the deck supplied for this layer and scale.

        Absent kinds are simply absent -- the caller keeps its own value for
        them. Returning zeros instead would be indistinguishable from a layer
        the deck really did find empty.
        """
        files = {k: str(p) for (l, s, k), p in self.density.items()
                 if l == layer and s == float(scale_um)}
        marker_files = {k: str(p) for (l, k), p in self.markers.items()
                        if l == layer}
        if not files and not marker_files:
            return {}
        return load_scale(bbox, float(scale_um), layer, files,
                          eps_um=self.eps_um(layer) or 1.0,
                          step_ratio=float(self.manifest["step_ratio"]),
                          marker_files=marker_files, grid=grid).values


def discover(directory: str | Path) -> CalibreSource:
    """Read a deck output directory, by the naming the deck itself writes.

    The extraction manifest is required rather than optional. Without it the
    reader would have to guess ``eps``, and a band density divided by the
    wrong one is off by that ratio while looking entirely ordinary -- the one
    error in this path that no downstream check could catch.
    """
    import json

    d = Path(directory)
    side = d / "extraction_manifest.json"
    if not side.exists():
        raise FileNotFoundError(
            f"{side} is missing. It is written beside the deck by "
            "lamxsim.calibre.svrf.write_deck (or by the emulator) and carries "
            "the snapped eps per layer, which cannot be recovered from the "
            "RDB files. Regenerate the deck rather than assuming a value.")
    manifest = json.loads(side.read_text())
    names = {e["name"] for e in manifest["layers"]}

    density, markers = {}, {}
    for path in sorted(d.glob("*.rdb")):
        stem = path.stem
        layer = next((n for n in names
                      if stem.endswith(f"_{n}") or f"_{n}_" in stem), None)
        if layer is None:
            continue
        if stem.endswith(f"_{layer}"):
            kind = stem[: -len(layer) - 1]
            markers[(layer, kind)] = path
        else:
            head, _, tail = stem.partition(f"_{layer}_")
            if not tail.endswith("um"):
                continue
            density[(layer, float(tail[:-2]), head)] = path
    if not density and not markers:
        raise ValueError(f"{d} holds no RDB file matching a layer in "
                         f"{sorted(names)}")
    return CalibreSource(directory=d, manifest=manifest,
                         density=density, markers=markers)
