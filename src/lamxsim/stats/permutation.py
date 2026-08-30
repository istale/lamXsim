"""Spatial null model (spec section 15).

Grid cells adjacent in space are correlated, so shuffling individual cell
labels destroys that structure and produces a null distribution far narrower
than reality -- which makes ordinary features look significant. Labels are
therefore permuted in contiguous blocks whose size is chosen from the
measured spatial autocorrelation rather than picked by hand.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PermutationResult:
    observed: float
    null_mean: float
    null_sd: float
    p_value: float
    n_permutations: int
    block_cells: int
    block_um: float

    def as_row(self) -> dict:
        return self.__dict__.copy()


def morans_i(values: np.ndarray, grid) -> float:
    """Rook-neighbour Moran's I of a feature on the grid."""
    idx = {(c.row, c.col): i for i, c in enumerate(grid.cells)}
    z = values - values.mean()
    denom = float((z ** 2).sum())
    if denom <= 0:
        return 0.0
    num, w = 0.0, 0
    for (r, c), i in idx.items():
        for dr, dc in ((0, 1), (1, 0)):
            j = idx.get((r + dr, c + dc))
            if j is not None:
                num += z[i] * z[j]
                w += 1
    if w == 0:
        return 0.0
    return float((len(values) / w) * (num / denom))


def autocorrelation_range_cells(values: np.ndarray, grid, threshold: float = 0.2,
                                max_lag: int | None = None) -> int:
    """Lag (in cells) at which spatial autocorrelation drops below *threshold*.

    This is the empirical basis for the permutation block size: blocks smaller
    than the correlation range leave correlated cells free to be split apart,
    which is the naive-shuffle failure the spec forbids.
    """
    n_rows, n_cols = grid.n_rows, grid.n_cols
    if n_rows < 2 or n_cols < 2:
        return 1
    field = np.full((n_rows, n_cols), np.nan)
    for i, c in enumerate(grid.cells):
        field[c.row, c.col] = values[i]
    mu = np.nanmean(field)
    z = field - mu
    var = np.nanvar(field)
    if var <= 0:
        return 1
    lim = max_lag or max(2, min(n_rows, n_cols) // 2)
    for lag in range(1, lim + 1):
        a = z[:, :-lag] * z[:, lag:]
        b = z[:-lag, :] * z[lag:, :]
        rho = (np.nanmean(a) + np.nanmean(b)) / (2 * var)
        if not np.isfinite(rho) or rho < threshold:
            return lag
    return lim


def block_permutation_test(values: np.ndarray, labels: np.ndarray, grid, *,
                           statistic=None, n_permutations: int = 999,
                           block_cells: int | None = None,
                           seed: int = 0,
                           mask: np.ndarray | None = None,
                           groups: np.ndarray | None = None) -> PermutationResult:
    """Permute labels in contiguous square blocks and compare the statistic."""
    from .univariate import roc_auc

    if statistic is None:
        def statistic(v, l):
            return roc_auc(v[l == 1], v[l == 0])

    labels = labels.astype(int)
    if block_cells is None and groups is None:
        block_cells = max(autocorrelation_range_cells(values, grid), 1)
    elif block_cells is None:
        block_cells = 1

    if groups is None:
        rows = np.array([c.row for c in grid.cells])
        cols = np.array([c.col for c in grid.cells])
        block_id = ((rows // block_cells) * (grid.n_cols // block_cells + 1)
                    + (cols // block_cells))
    else:
        block_id = np.asarray(groups)

    keep = np.ones(len(values), bool) if mask is None else np.asarray(mask, bool)
    # An explicit grouping lets a multi-die run permute within (die, block)
    # rather than across dies, which would destroy the die structure the
    # held-out-die validation depends on.
    ids = block_id if groups is None else np.asarray(groups)
    groups = [g[keep[g]] for g in
              (np.where(ids == b)[0] for b in np.unique(ids))]
    groups = [g for g in groups if len(g)]

    observed = statistic(values[keep], labels[keep])
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    for k in range(n_permutations):
        perm_order = rng.permutation(len(groups))
        shuffled = np.empty_like(labels)
        # Move whole blocks of labels onto other blocks, truncating or
        # recycling as needed when blocks differ in size at the die edge.
        pool = np.concatenate([labels[groups[j]] for j in perm_order])
        pos = 0
        for g in groups:
            take = pool[pos:pos + len(g)]
            if len(take) < len(g):
                take = np.concatenate([take, pool[:len(g) - len(take)]])
            shuffled[g] = take
            pos += len(g)
        null[k] = statistic(values[keep], shuffled[keep])

    finite = null[np.isfinite(null)]
    # Two-sided, centred on the null mean so a protective (AUC < 0.5)
    # association is not silently discarded.
    centre = float(np.mean(finite)) if len(finite) else 0.5
    p = (1 + np.sum(np.abs(finite - centre) >= abs(observed - centre))) / (len(finite) + 1)
    return PermutationResult(
        observed=float(observed), null_mean=centre,
        null_sd=float(np.std(finite)) if len(finite) else float("nan"),
        p_value=float(p), n_permutations=len(finite),
        block_cells=int(block_cells), block_um=float(block_cells * grid.scale_um),
    )


def spatial_block_ids(grid, block_cells: int = 1) -> np.ndarray:
    """Contiguous square block index per grid cell."""
    rows = np.array([c.row for c in grid.cells])
    cols = np.array([c.col for c in grid.cells])
    return ((rows // block_cells) * (grid.n_cols // block_cells + 1)
            + (cols // block_cells))

