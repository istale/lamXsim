"""Gradient features (spec section 5).

The spec is explicit that a feature's spatial gradient must not be assumed
less important than its absolute value, so gradients are first-class outputs
rather than diagnostics.

Two things are handled carefully.

**Physical units.** ``dQ_dx`` is per micrometre, differenced over the grid
stride. Differencing over cell indices would make the same layout produce
gradients differing by orders of magnitude between scales, and "which scale
shows the strongest association" would then be answering a question about the
grid rather than about the layout.

**Die-boundary cells.** Interior cells get a centred difference; boundary
cells can only get a one-sided one. One-sided differences are biased
differently from centred ones, and the cells carrying them form a ring at the
die edge -- which is exactly the shape of ``distance_to_die_edge``, a
PACKAGE_POSITION confounder. Left unmarked, a gradient feature acquires a
spurious association with die position through nothing but its own numerics.
Every gradient therefore ships with an ``*_interior`` mask, and the extractor
can drop the boundary ring outright.
"""
from __future__ import annotations

import numpy as np

from ..evidence import EvidenceClass
from .grid import Grid

EVIDENCE_CLASS = EvidenceClass.GDS_GEOMETRY


def to_field(values: np.ndarray, grid: Grid) -> np.ndarray:
    """Flat per-cell array -> (n_rows, n_cols) field."""
    field = np.full((grid.n_rows, grid.n_cols), np.nan)
    for i, c in enumerate(grid.cells):
        field[c.row, c.col] = values[i]
    return field


def to_flat(field: np.ndarray, grid: Grid) -> np.ndarray:
    out = np.zeros(len(grid))
    for i, c in enumerate(grid.cells):
        out[i] = field[c.row, c.col]
    return out


def interior_mask(grid: Grid) -> np.ndarray:
    """True for cells whose gradient came from a centred difference."""
    mask = np.zeros(len(grid), dtype=bool)
    for i, c in enumerate(grid.cells):
        mask[i] = (0 < c.row < grid.n_rows - 1) and (0 < c.col < grid.n_cols - 1)
    return mask


def gradients(values: np.ndarray, grid: Grid, name: str, *,
              drop_boundary: bool = True) -> dict[str, np.ndarray]:
    """Return dQ_dx, dQ_dy and |grad Q| for one scalar feature.

    Spacing is the grid stride in micrometres, so results are per-um and
    comparable across scales. With ``drop_boundary`` the die-edge ring is set
    to NaN rather than filled with a one-sided estimate; downstream statistics
    skip NaN, which keeps the numerical artifact out of the association tables
    instead of letting it masquerade as a die-position effect.
    """
    if grid.n_rows < 3 or grid.n_cols < 3:
        nan = np.full(len(values), np.nan)
        return {f"{name}_dx": nan.copy(), f"{name}_dy": nan.copy(),
                f"{name}_grad_mag": nan.copy()}

    field = to_field(values, grid)
    h = grid.stride_um
    dy_field, dx_field = np.gradient(field, h, h, edge_order=1)

    dx = to_flat(dx_field, grid)
    dy = to_flat(dy_field, grid)
    if drop_boundary:
        edge = ~interior_mask(grid)
        dx[edge] = np.nan
        dy[edge] = np.nan
    return {f"{name}_dx": dx, f"{name}_dy": dy,
            f"{name}_grad_mag": np.hypot(dx, dy)}


def gradient_set(features: dict[str, np.ndarray], grid: Grid, *,
                 only: tuple[str, ...] | None = None,
                 drop_boundary: bool = True) -> dict[str, np.ndarray]:
    """Gradients for every (or a chosen subset of) scalar feature."""
    out = {}
    for name, vals in features.items():
        if only is not None and name not in only:
            continue
        out.update(gradients(vals, grid, name, drop_boundary=drop_boundary))
    return out
