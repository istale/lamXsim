"""Simulated failure labels, for pipeline validation only.

Every FailureSet produced here carries ``simulated=True``. Simulated labels
exist to prove the statistical pipeline recovers a driver it was given and
reports nothing when there is no driver; they are never evidence about a real
process and must not reach a report that claims measured association.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .failure import FailureSet


def failures_from_driver(driver: np.ndarray, grid, *, n_failures: int = 120,
                         strength: float = 2.5, seed: int = 0,
                         position_sigma_um: float = 0.0,
                         lot_id: str = "SIM", wafer_id: str = "W01",
                         die_x: int = 0, die_y: int = 0) -> FailureSet:
    """Draw failure sites with probability driven by *driver*.

    ``strength`` is the logit coefficient on the z-scored driver: 0 gives a
    spatially uniform pattern (the negative-control case), larger values give
    a stronger association.
    """
    rng = np.random.default_rng(seed)
    z = (driver - driver.mean()) / (driver.std() + 1e-12)
    w = 1.0 / (1.0 + np.exp(-strength * z))
    w = w / w.sum()

    pick = rng.choice(len(grid), size=n_failures, replace=True, p=w)
    half = grid.scale_um / 2
    xs, ys = [], []
    for i in pick:
        c = grid.cells[i]
        xs.append(c.x_center + rng.uniform(-half, half))
        ys.append(c.y_center + rng.uniform(-half, half))
    xs = np.array(xs)
    ys = np.array(ys)

    if position_sigma_um > 0:
        # Registration error: the recorded position is not the true one.
        xs = xs + rng.normal(0, position_sigma_um, size=len(xs))
        ys = ys + rng.normal(0, position_sigma_um, size=len(ys))
        # A failure is on the die, so a simulated measurement of one stays on
        # it. Letting the jitter carry a near-edge failure off the die would
        # manufacture the very contradiction the footprint audit exists to
        # detect on real data, and mask it as ordinary simulation noise.
        b = grid.bbox
        xs = np.clip(xs, b.xmin, b.xmax)
        ys = np.clip(ys, b.ymin, b.ymax)

    df = pd.DataFrame({
        "sample_id": [f"S{i:04d}" for i in range(len(xs))],
        "lot_id": lot_id, "wafer_id": wafer_id, "die_x": die_x, "die_y": die_y,
        "x_um": xs, "y_um": ys,
        "failure_type": "delamination",
        "confidence": 1.0,
        "position_sigma_um": position_sigma_um,
        "coord_frame": "die_local",
    })
    return FailureSet(table=df, simulated=True,
                      source=f"simulated(strength={strength}, seed={seed})",
                      notes=["SIMULATED LABELS - not measured evidence"])


def uniform_failures(grid, *, n_failures: int = 120, seed: int = 0,
                     **kwargs) -> FailureSet:
    """Spatially uniform failures: the null case, no feature drives them."""
    return failures_from_driver(np.zeros(len(grid)), grid,
                                n_failures=n_failures, strength=0.0,
                                seed=seed, **kwargs)
