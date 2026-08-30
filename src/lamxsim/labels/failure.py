"""Measured failure import and mapping (spec section 10).

Failure data is MEASURED_FAILURE evidence and is kept in its own table with
its own coordinate frame and its own uncertainty. It is joined to grid cells
only through :func:`map_to_grid`, which never converts an uncertain location
into an exact one -- the reported ``position_sigma_um`` travels with every
record and gates which analysis scales are admissible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..evidence import EvidenceClass

#: Columns required by spec section 2, extended for the grouped-split needs of
#: spec section 17. lot/wafer/die identity cannot be recovered later, so it is
#: required at import rather than optional.
REQUIRED_COLUMNS = ("sample_id", "x_um", "y_um", "failure_type")
GROUPING_COLUMNS = ("lot_id", "wafer_id", "die_x", "die_y")
OPTIONAL_COLUMNS = ("confidence", "position_sigma_um", "extent_um", "coord_frame",
                    "failed_layer", "failed_interface")

#: Columns whose distinct values define separate failure populations. Li et al.
#: (2023) found the largest energy release rate at one particular upper BEOL
#: interface, with bottom interconnect interfaces more critical than sidewalls,
#: so two failures at the same (x, y) on different interfaces are not two
#: observations of the same thing.
MODE_COLUMNS = ("failure_type", "failed_layer", "failed_interface")


@dataclass
class FailureSet:
    """A set of measured (or simulated) failure locations."""
    table: pd.DataFrame
    evidence_class: EvidenceClass = EvidenceClass.MEASURED_FAILURE
    simulated: bool = False
    source: str = ""
    notes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.table)

    @property
    def position_sigma_um(self) -> float:
        """Worst-case reported positional uncertainty, in um."""
        if "position_sigma_um" not in self.table:
            return float("nan")
        sigma = float(self.table["position_sigma_um"].max())
        if np.isfinite(sigma) and sigma < 0:
            raise ValueError(
                f"position_sigma_um is negative ({sigma}); a negative "
                "uncertainty would certify every analysis scale")
        return sigma

    def die_keys(self) -> "pd.Series":
        """One identifier per physical die, from lot/wafer/die coordinates."""
        cols = [c for c in GROUPING_COLUMNS if c in self.table]
        if not cols:
            return pd.Series(["<unknown>"] * len(self.table),
                             index=self.table.index)
        return (self.table[cols].astype(str)
                .agg("|".join, axis=1).rename("die_key"))

    def n_dies(self) -> int:
        return int(self.die_keys().nunique())

    def modes(self) -> dict[str, list]:
        """Distinct values of every column that defines a failure population."""
        out = {}
        for col in MODE_COLUMNS:
            if col in self.table:
                vals = sorted(self.table[col].dropna().astype(str).unique())
                if vals:
                    out[col] = vals
        return out

    def assert_single_mode(self, *, allow_pooling: bool = False) -> list[str]:
        """Refuse to pool failure modes that were not declared poolable.

        A mode column with more than one value means the file mixes
        populations. They may share a mechanism, but that is an engineering
        judgement about the physics, not something the counts can settle, so
        it has to be asserted rather than assumed.
        """
        mixed = {k: v for k, v in self.modes().items() if len(v) > 1}
        if not mixed:
            return []
        summary = "; ".join(f"{k}: {v}" for k, v in mixed.items())
        if not allow_pooling:
            raise ValueError(
                f"the failure set mixes populations ({summary}). Analyse them "
                "separately, or pass allow_pooling=True to assert that these "
                "modes share a defensible mechanism -- the assertion is "
                "recorded in the run metadata.")
        return [f"pooling failure modes across {summary}, asserted by the "
                "operator rather than established by the data"]

    def min_trustworthy_scale_um(self, factor: float = 3.0) -> float:
        """Smallest analysis scale the registration accuracy can support.

        Below roughly 3x the positional uncertainty a window no longer
        reliably contains the failure it is credited with, so association at
        that scale measures registration noise rather than layout.
        """
        s = self.position_sigma_um
        return float("nan") if np.isnan(s) else factor * s


def _validate_values(df: pd.DataFrame, path) -> None:
    """Reject values that would corrupt the analysis rather than fail loudly.

    Each of these has a specific downstream consequence, so none is tolerated
    and none is silently dropped -- discarding a measured failure is exactly
    the kind of quiet loss this module exists to prevent.
    """
    problems: list[str] = []

    for col in ("x_um", "y_um"):
        bad = ~np.isfinite(pd.to_numeric(df[col], errors="coerce").to_numpy(float))
        if bad.any():
            rows = list(np.where(bad)[0][:5])
            problems.append(
                f"{col}: {int(bad.sum())} non-finite value(s) at row(s) {rows}. "
                "A single NaN coordinate makes distance_to_nearest_failure NaN "
                "for every cell on the die")

    if "position_sigma_um" in df:
        sigma = pd.to_numeric(df["position_sigma_um"], errors="coerce").to_numpy(float)
        bad = np.isfinite(sigma) & (sigma < 0)
        if bad.any():
            problems.append(
                f"position_sigma_um: {int(bad.sum())} negative value(s) at "
                f"row(s) {list(np.where(bad)[0][:5])}. A negative uncertainty "
                "produces a negative scale floor, which certifies every "
                "analysis scale instead of rejecting the small ones")

    if "confidence" in df:
        conf = pd.to_numeric(df["confidence"], errors="coerce").to_numpy(float)
        bad = np.isfinite(conf) & ((conf < 0) | (conf > 1))
        if bad.any():
            problems.append(
                f"confidence: {int(bad.sum())} value(s) outside [0, 1] at "
                f"row(s) {list(np.where(bad)[0][:5])}")

    ids = df["sample_id"].astype("string")
    bad = ids.isna() | (ids.str.strip() == "")
    if bad.any():
        problems.append(
            f"sample_id: {int(bad.sum())} empty value(s) at row(s) "
            f"{list(np.where(bad.to_numpy())[0][:5])}; a failure that cannot be "
            "named cannot be traced back to its measurement")

    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))


def load_failures(path: str | Path, *, require_grouping: bool = True) -> FailureSet:
    """Read a failure CSV, validating both the schema and the values."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")

    notes = []
    if require_grouping:
        miss_g = [c for c in GROUPING_COLUMNS if c not in df.columns]
        if miss_g:
            raise ValueError(
                f"{path}: missing grouping columns {miss_g}. Without lot/wafer/die "
                "identity the held-out-die validation of spec section 17 cannot be "
                "performed, and any reported AUC would be un-generalisable."
            )
    if "confidence" not in df:
        df["confidence"] = 1.0
        notes.append("confidence absent; defaulted to 1.0")
    if "position_sigma_um" not in df:
        df["position_sigma_um"] = np.nan
        notes.append(
            "position_sigma_um absent: registration accuracy unknown, so no "
            "analysis scale can be certified trustworthy"
        )
    for col in ("failed_layer", "failed_interface"):
        if col not in df:
            notes.append(
                f"{col} absent: the failed layer/interface is not recorded, so "
                "failures on mechanically different interfaces cannot be "
                "separated and are being analysed as one population")

    _validate_values(df, path)
    return FailureSet(table=df, source=str(path), notes=notes)


def map_to_grid(failures: FailureSet, grid, *, radius_um: float | None = None
                ) -> dict[str, np.ndarray]:
    """Derive per-cell failure labels (spec section 10).

    By default a failure belongs to the cell whose *bounds* contain it. Cells
    are squares, so testing a radius against the cell centre instead inscribes
    a circle in each one and silently discards everything in the corners --
    21% of the die (1 - pi/4), arranged on a regular lattice rather than
    scattered, which biases the labels wherever the layout has structure on
    the grid pitch.

    ``radius_um`` switches to a circular test around the cell centre, for the
    case where a failure is deliberately being credited to every cell within
    some distance of it -- an uncertainty-aware assignment, not containment.

    ``distance_to_nearest_failure`` is Euclidean in both modes.

    Returns failure_present, failure_count and distance_to_nearest_failure for
    every cell. A cell may legitimately hold more than one failure, and with
    an overlapping grid one failure legitimately lands in several cells.
    """
    cx = np.array([c.x_center for c in grid.cells])
    cy = np.array([c.y_center for c in grid.cells])
    fx = failures.table["x_um"].to_numpy(float)
    fy = failures.table["y_um"].to_numpy(float)

    n = len(grid)
    if len(fx) == 0:
        return {"failure_present": np.zeros(n, np.int8),
                "failure_count": np.zeros(n, np.int32),
                "distance_to_nearest_failure": np.full(n, np.inf)}

    x0 = np.array([c.x0 for c in grid.cells])
    y0 = np.array([c.y0 for c in grid.cells])
    x1 = np.array([c.x1 for c in grid.cells])
    y1 = np.array([c.y1 for c in grid.cells])
    # Bounds are half-open so that a failure on a shared edge is counted once,
    # except at the outer edge of the grid, where closing the interval is what
    # keeps a failure exactly on the die boundary from belonging to no cell.
    close_x = x1 >= grid.bbox.xmax - 1e-9
    close_y = y1 >= grid.bbox.ymax - 1e-9

    nearest = np.empty(n)
    count = np.zeros(n, np.int32)
    block = 4096
    for s in range(0, n, block):
        e = min(s + block, n)
        d = np.hypot(cx[s:e, None] - fx[None, :], cy[s:e, None] - fy[None, :])
        nearest[s:e] = d.min(axis=1)
        if radius_um is None:
            inside = (
                (fx[None, :] >= x0[s:e, None])
                & (np.where(close_x[s:e, None], fx[None, :] <= x1[s:e, None],
                            fx[None, :] < x1[s:e, None]))
                & (fy[None, :] >= y0[s:e, None])
                & (np.where(close_y[s:e, None], fy[None, :] <= y1[s:e, None],
                            fy[None, :] < y1[s:e, None]))
            )
        else:
            inside = d <= float(radius_um)
        count[s:e] = inside.sum(axis=1)

    return {"failure_present": (count > 0).astype(np.int8),
            "failure_count": count,
            "distance_to_nearest_failure": nearest}


def map_to_grid_per_die(failures: FailureSet, grid, *,
                        radius_um: float | None = None
                        ) -> dict[str, dict[str, np.ndarray]]:
    """Labels for each die separately, keyed by die identity.

    Pooling several dies of the same design onto one grid and asking "did
    anything ever fail here" is not a rescaling of the single-die case, it is
    a different and wrong question: prevalence grows with the number of dies
    (0.24 for one die becomes 0.98 for ten in a uniform simulation), a cell
    that failed on one die of ten becomes indistinguishable from one that
    failed on all ten, and die identity -- the thing spec section 17 wants to
    hold out -- is gone before any fold can be built from it.

    The observation unit is therefore (cell, die), which is also what spec
    section 11 means by a case: a location on a piece of silicon.
    """
    from dataclasses import replace

    keys = failures.die_keys()
    out = {}
    for key in keys.unique():
        subset = replace(failures,
                         table=failures.table[keys == key].reset_index(drop=True))
        out[str(key)] = map_to_grid(subset, grid, radius_um=radius_um)
    return out

