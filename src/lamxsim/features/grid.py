"""Multi-scale analysis grids in physical units (spec section 6).

Grids are defined in micrometres, never in pixels, and every cell knows the
scale it came from. Non-overlapping grids are the default: overlapping
windows inflate the apparent sample count without adding independent
information, which is exactly the failure mode the spatial null model
(spec section 15) exists to catch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..layout.reader import BBox


@dataclass(frozen=True)
class GridCell:
    cell_id: int
    x_center: float
    y_center: float
    x0: float
    y0: float
    x1: float
    y1: float
    scale_um: float
    row: int
    col: int

    @property
    def area_um2(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)


@dataclass(frozen=True)
class Grid:
    scale_um: float
    stride_um: float
    n_rows: int
    n_cols: int
    cells: tuple[GridCell, ...]
    bbox: BBox

    def __len__(self) -> int:
        return len(self.cells)

    @property
    def overlapping(self) -> bool:
        return self.stride_um < self.scale_um

    def centers(self) -> np.ndarray:
        return np.array([(c.x_center, c.y_center) for c in self.cells], dtype=float)

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "cell_id": np.array([c.cell_id for c in self.cells], dtype=np.int64),
            "x_um": np.array([c.x_center for c in self.cells], dtype=float),
            "y_um": np.array([c.y_center for c in self.cells], dtype=float),
            "row": np.array([c.row for c in self.cells], dtype=np.int32),
            "col": np.array([c.col for c in self.cells], dtype=np.int32),
            "scale_um": np.full(len(self.cells), self.scale_um, dtype=float),
        }


def build_grid(bbox: BBox, scale_um: float, stride_um: float | None = None,
               align: str = "origin") -> Grid:
    """Regular analysis grid covering *bbox*.

    ``stride_um`` defaults to ``scale_um`` (non-overlapping). Partial cells at
    the far edge are dropped rather than analysed with a smaller area, so that
    every cell of a given scale represents the same physical footprint and
    densities stay comparable across the die.
    """
    if scale_um <= 0:
        raise ValueError("scale_um must be positive")
    stride = float(scale_um if stride_um is None else stride_um)
    if stride <= 0:
        raise ValueError("stride_um must be positive")

    if align == "origin":
        x_start, y_start = bbox.xmin, bbox.ymin
    elif align == "center":
        nx = int((bbox.width - scale_um) // stride) + 1
        ny = int((bbox.height - scale_um) // stride) + 1
        x_start = bbox.xmin + (bbox.width - ((nx - 1) * stride + scale_um)) / 2
        y_start = bbox.ymin + (bbox.height - ((ny - 1) * stride + scale_um)) / 2
    else:
        raise ValueError(f"unknown align={align!r}")

    n_cols = int((bbox.width - scale_um) // stride) + 1
    n_rows = int((bbox.height - scale_um) // stride) + 1
    if n_cols < 1 or n_rows < 1:
        raise ValueError(
            f"scale {scale_um}um does not fit in bbox {bbox.width}x{bbox.height}um"
        )

    cells = []
    cid = 0
    for r in range(n_rows):
        y0 = y_start + r * stride
        for c in range(n_cols):
            x0 = x_start + c * stride
            cells.append(GridCell(
                cell_id=cid, x_center=x0 + scale_um / 2, y_center=y0 + scale_um / 2,
                x0=x0, y0=y0, x1=x0 + scale_um, y1=y0 + scale_um,
                scale_um=scale_um, row=r, col=c,
            ))
            cid += 1
    return Grid(scale_um=scale_um, stride_um=stride, n_rows=n_rows,
                n_cols=n_cols, cells=tuple(cells), bbox=bbox)


def build_multiscale(bbox: BBox, scales_um, stride_ratio: float = 1.0) -> dict[float, Grid]:
    """One grid per requested scale, keyed by scale."""
    out = {}
    for s in scales_um:
        s = float(s)
        if s > min(bbox.width, bbox.height):
            continue  # scale larger than the die: nothing to measure
        out[s] = build_grid(bbox, s, stride_um=s * stride_ratio)
    if not out:
        raise ValueError("no requested scale fits inside the layout bbox")
    return out


def point_counts(grid: Grid, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """How many of the given points fall in each window."""
    return point_accumulate(grid, x, y)


def point_accumulate(grid: Grid, x: np.ndarray, y: np.ndarray,
                     weights: "np.ndarray | None" = None) -> np.ndarray:
    """Sum *weights* (or count, with none) over the points in each window.

    By index arithmetic, not by scanning the grid for every point. The
    obvious nested version -- for each cell, mask the whole point array --
    is O(points x cells), and on a layout both grow with die area, so it is
    quadratic in area. Measured on synthetic dies it was the fastest-growing
    term in the whole extractor: a fourfold rise in polygon count made it
    fifteen times slower while every other stage stayed linear. At a hundred
    million points and a million windows it would never finish.

    The grid is regular by construction -- every window is ``scale_um`` across
    and starts at ``stride_um`` intervals from a common origin -- so the
    windows containing a point are found directly. With overlapping windows a
    point belongs to several, at most ``ceil(scale/stride)`` in each axis, and
    those are enumerated rather than searched.

    The half-open convention matches the rest of the package: a point on a
    window's lower or left edge belongs to it, one on the upper or right edge
    belongs to the neighbour.
    """
    out = np.zeros(len(grid), dtype=float)
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size == 0:
        return out
    w = (np.ones(x.size) if weights is None
         else np.asarray(weights, dtype=float).ravel())

    first = grid.cells[0]
    x_start, y_start = first.x0, first.y0
    stride, scale = grid.stride_um, grid.scale_um
    span = int(np.ceil(scale / stride))

    # The highest-indexed window whose left edge is at or below the point.
    col_hi = np.floor((x - x_start) / stride).astype(np.int64)
    row_hi = np.floor((y - y_start) / stride).astype(np.int64)

    for dc in range(span):
        cols = col_hi - dc
        ok_c = (cols >= 0) & (cols < grid.n_cols)
        # Inside the window, not merely to the right of its left edge: with
        # overlapping windows the two differ, and with abutting ones the
        # check costs nothing and guards the floating-point boundary.
        ok_c &= x < (x_start + cols * stride + scale)
        ok_c &= x >= (x_start + cols * stride)
        for dr in range(span):
            rows = row_hi - dr
            ok = (ok_c & (rows >= 0) & (rows < grid.n_rows)
                  & (y < (y_start + rows * stride + scale))
                  & (y >= (y_start + rows * stride)))
            if not ok.any():
                continue
            np.add.at(out, rows[ok] * grid.n_cols + cols[ok], w[ok])
    return out
