"""Spatially separated cross-validation (spec section 17).

Random splitting of grid cells is invalid here. Cells adjacent in space share
the layout that produced them and, near a failure site, share the failure
itself; a random split puts both sides of the same physical feature in train
and test and reports a score that measures memorisation.

Three levels are provided, weakest to strongest:

* ``block_folds`` -- contiguous square blocks, assigned to folds.
* ``buffered_block_folds`` -- the same, with a buffer zone around each test
  block excluded from training. Blocking alone still leaks across the block
  boundary, where a training cell can sit one cell away from a test cell.
* ``grouped_folds`` -- whole dies or wafers held out. This is what spec section 17
  asks for whenever more than one die is available, and no amount of spatial
  blocking within a single die substitutes for it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    train: np.ndarray
    test: np.ndarray
    excluded: np.ndarray        # buffer cells, in neither set
    label: str = ""

    def __iter__(self):
        return iter((self.train, self.test))


def _block_id(grid, block_um: float) -> np.ndarray:
    per = max(int(round(block_um / grid.stride_um)), 1)
    rows = np.array([c.row for c in grid.cells])
    cols = np.array([c.col for c in grid.cells])
    n_block_cols = grid.n_cols // per + 1
    return (rows // per) * n_block_cols + (cols // per)


def block_folds(grid, *, block_um: float, n_folds: int = 5,
                seed: int = 0) -> list[Fold]:
    """Contiguous blocks dealt into *n_folds* groups."""
    bid = _block_id(grid, block_um)
    blocks = np.unique(bid)
    rng = np.random.default_rng(seed)
    assignment = rng.permutation(len(blocks)) % n_folds
    block_fold = dict(zip(blocks, assignment))
    fold_of = np.array([block_fold[b] for b in bid])

    out = []
    for k in range(n_folds):
        test = np.where(fold_of == k)[0]
        train = np.where(fold_of != k)[0]
        if len(test) == 0 or len(train) == 0:
            continue
        out.append(Fold(train=train, test=test,
                        excluded=np.array([], dtype=int), label=f"block-{k}"))
    return out


def buffered_block_folds(grid, *, block_um: float, n_folds: int = 5,
                         buffer_um: float | None = None,
                         seed: int = 0) -> list[Fold]:
    """Block folds with a buffer ring around each test set removed from training.

    ``buffer_um`` defaults to the block size. Without it, a training cell can
    sit one stride from a test cell and carry essentially the same layout.
    """
    buffer_um = block_um if buffer_um is None else buffer_um
    x = np.array([c.x_center for c in grid.cells])
    y = np.array([c.y_center for c in grid.cells])

    out = []
    for fold in block_folds(grid, block_um=block_um, n_folds=n_folds, seed=seed):
        test = fold.test
        keep = np.ones(len(grid), dtype=bool)
        keep[test] = False
        # Drop any candidate training cell within buffer_um of a test cell.
        tx, ty = x[test], y[test]
        block = 2048
        near = np.zeros(len(grid), dtype=bool)
        for s in range(0, len(grid), block):
            e = min(s + block, len(grid))
            d = np.hypot(x[s:e, None] - tx[None, :], y[s:e, None] - ty[None, :])
            near[s:e] = d.min(axis=1) < buffer_um
        train = np.where(keep & ~near)[0]
        excluded = np.where(keep & near)[0]
        if len(train) == 0:
            continue
        out.append(Fold(train=train, test=test, excluded=excluded,
                        label=fold.label))
    return out


def grouped_folds(groups: np.ndarray) -> list[Fold]:
    """Leave-one-group-out, where a group is a die, wafer or lot."""
    groups = np.asarray(groups)
    out = []
    for g in np.unique(groups):
        test = np.where(groups == g)[0]
        train = np.where(groups != g)[0]
        if len(train) == 0:
            continue
        out.append(Fold(train=train, test=test,
                        excluded=np.array([], dtype=int), label=f"held-out:{g}"))
    return out


def leakage_report(folds: list[Fold], grid, *, min_separation_um: float) -> dict:
    """Closest train/test approach per fold, against the required separation.

    A scheme that passes this is not thereby validated -- held-out dies remain
    stronger than any within-die split -- but one that fails it is reporting a
    score inflated by shared geometry.
    """
    x = np.array([c.x_center for c in grid.cells])
    y = np.array([c.y_center for c in grid.cells])
    rows = []
    for f in folds:
        if len(f.train) == 0 or len(f.test) == 0:
            continue
        tx, ty = x[f.test], y[f.test]
        best = np.inf
        block = 2048
        for s in range(0, len(f.train), block):
            idx = f.train[s:s + block]
            d = np.hypot(x[idx, None] - tx[None, :], y[idx, None] - ty[None, :])
            best = min(best, float(d.min()))
        rows.append({"fold": f.label, "n_train": len(f.train),
                     "n_test": len(f.test), "n_excluded": len(f.excluded),
                     "min_train_test_separation_um": best,
                     "passes": bool(best >= min_separation_um)})
    return {"required_separation_um": min_separation_um,
            "folds": rows,
            "all_pass": bool(rows) and all(r["passes"] for r in rows)}
