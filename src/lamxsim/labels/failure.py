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

    ``radius_um`` defaults to half the grid scale, i.e. a failure marks the
    cell it falls in. Returns failure_present, failure_count and
    distance_to_nearest_failure for every cell.
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

    # Blocked distance computation keeps memory bounded on real grids.
    nearest = np.empty(n)
    count = np.zeros(n, np.int32)
    r = (grid.scale_um / 2) if radius_um is None else float(radius_um)
    block = 4096
    for s in range(0, n, block):
        e = min(s + block, n)
        d = np.hypot(cx[s:e, None] - fx[None, :], cy[s:e, None] - fy[None, :])
        nearest[s:e] = d.min(axis=1)
        count[s:e] = (d <= r).sum(axis=1)

    return {"failure_present": (count > 0).astype(np.int8),
            "failure_count": count,
            "distance_to_nearest_failure": nearest}
