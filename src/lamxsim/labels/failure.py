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
OPTIONAL_COLUMNS = ("confidence", "position_sigma_um", "extent_um", "coord_frame")


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
        return float(self.table["position_sigma_um"].max())

    def min_trustworthy_scale_um(self, factor: float = 3.0) -> float:
        """Smallest analysis scale the registration accuracy can support.

        Below roughly 3x the positional uncertainty a window no longer
        reliably contains the failure it is credited with, so association at
        that scale measures registration noise rather than layout.
        """
        s = self.position_sigma_um
        return float("nan") if np.isnan(s) else factor * s


def load_failures(path: str | Path, *, require_grouping: bool = True) -> FailureSet:
    """Read a failure CSV, validating the schema up front."""
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
