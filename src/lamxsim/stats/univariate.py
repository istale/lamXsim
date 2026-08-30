"""Univariate association analysis (spec section 12).

Every statistic here is STATISTICAL_ASSOCIATION evidence. Two things are
deliberate:

* Tests are two-sided. Zahedmanesh & Vanstreels (2019) show a stiff top metal
  group can *lower* the crack driving force in the layer beneath it through
  elastic stress shielding, so the same feature can associate with failure in
  opposite directions on different layers. A one-sided test would encode a
  monotone "denser is worse" prior the literature does not support.
* Effect sizes are signed and reported per layer, never pooled as magnitude.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats


@dataclass
class Association:
    feature: str
    layer: str
    scale_um: float
    n_case: int
    n_control: int
    case_mean: float
    control_mean: float
    median_difference: float
    effect_size: float          # Cliff's delta, signed, [-1, 1]
    cohens_d: float
    p_value: float
    roc_auc: float
    pr_auc: float
    prevalence: float
    enrichment_top_1pct: float
    enrichment_top_5pct: float
    enrichment_top_10pct: float
    enrichment_top_20pct: float
    effective_n: float
    auc_ci_low: float = float("nan")
    auc_ci_high: float = float("nan")
    fdr_q_value: float = float("nan")
    hypothesis_tier: str = "exploratory"

    def as_row(self) -> dict:
        return asdict(self)


def cliffs_delta(case: np.ndarray, control: np.ndarray) -> float:
    """Signed, distribution-free effect size. Equals 2*AUC - 1."""
    if len(case) == 0 or len(control) == 0:
        return float("nan")
    return 2.0 * roc_auc(case, control) - 1.0


def roc_auc(case: np.ndarray, control: np.ndarray) -> float:
    """AUC via the rank identity; ties contribute 0.5 as they should."""
    n1, n0 = len(case), len(control)
    if n1 == 0 or n0 == 0:
        return float("nan")
    allv = np.concatenate([case, control])
    r = stats.rankdata(allv)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def pr_auc(values: np.ndarray, labels: np.ndarray) -> float:
    """Average precision. Read against `prevalence`, never on its own."""
    order = np.argsort(-values, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    total = y.sum()
    if total == 0:
        return float("nan")
    return float((precision * y).sum() / total)


def enrichment(values: np.ndarray, labels: np.ndarray, top_frac: float) -> float:
    """P(failure | feature in top q) / P(failure | rest) (spec section 12)."""
    n = len(values)
    k = max(int(round(n * top_frac)), 1)
    if k >= n:
        return float("nan")
    order = np.argsort(-values, kind="mergesort")
    top, rest = labels[order[:k]], labels[order[k:]]
    p_top, p_rest = top.mean(), rest.mean()
    if p_rest == 0:
        return float("inf") if p_top > 0 else float("nan")
    return float(p_top / p_rest)


def effective_n(values: np.ndarray, grid) -> float:
    """Spatially-corrected sample size using Moran's I on a rook lattice.

    Grid cells are not independent samples. Reporting the raw cell count
    alongside an AUC invites reading a 6400-cell grid as 6400 observations
    when the layout may only vary over a few dozen independent patches.
    """
    n = len(values)
    if n < 4:
        return float(n)
    rows = np.array([c.row for c in grid.cells])
    cols = np.array([c.col for c in grid.cells])
    idx = {(r, c): i for i, (r, c) in enumerate(zip(rows, cols))}
    z = values - values.mean()
    denom = float((z ** 2).sum())
    if denom <= 0:
        return float(n)
    num = 0.0
    w = 0
    for (r, c), i in idx.items():
        for dr, dc in ((0, 1), (1, 0)):
            j = idx.get((r + dr, c + dc))
            if j is not None:
                num += z[i] * z[j]
                w += 1
    if w == 0:
        return float(n)
    moran = (n / w) * (num / denom)
    # Cressie's rule of thumb: n_eff = n * (1 - I) / (1 + I), clipped.
    moran = float(np.clip(moran, -0.999, 0.999))
    return float(np.clip(n * (1 - moran) / (1 + moran), 2.0, n))


def analyse(values: np.ndarray, labels: np.ndarray, *, feature: str, layer: str,
            scale_um: float, grid=None, tier: str = "exploratory") -> Association:
    """Full univariate association for one feature x layer x scale."""
    labels = labels.astype(int)
    case = values[labels == 1]
    control = values[labels == 0]

    if len(case) == 0 or len(control) == 0:
        nan = float("nan")
        return Association(feature, layer, scale_um, len(case), len(control),
                           nan, nan, nan, nan, nan, nan, nan, nan,
                           float(labels.mean()), nan, nan, nan, nan,
                           float(len(values)), hypothesis_tier=tier)

    auc = roc_auc(case, control)
    try:
        p = float(stats.mannwhitneyu(case, control, alternative="two-sided").pvalue)
    except ValueError:
        p = 1.0

    # Cohen's d needs at least two observations on each side. At coarse scales
    # a window can contain nearly every failure, leaving one control cell; the
    # rank-based effect size still applies there, so d is reported as NaN
    # rather than as a number computed from a degenerate variance.
    if len(case) > 1 and len(control) > 1:
        pooled_sd = np.sqrt(((len(case) - 1) * case.var(ddof=1) +
                             (len(control) - 1) * control.var(ddof=1)) /
                            (len(case) + len(control) - 2))
        d = float((case.mean() - control.mean()) / pooled_sd) if pooled_sd > 0 else float("nan")
    else:
        d = float("nan")

    return Association(
        feature=feature, layer=layer, scale_um=scale_um,
        n_case=len(case), n_control=len(control),
        case_mean=float(case.mean()), control_mean=float(control.mean()),
        median_difference=float(np.median(case) - np.median(control)),
        effect_size=float(2 * auc - 1), cohens_d=d, p_value=p,
        roc_auc=float(auc), pr_auc=pr_auc(values, labels),
        prevalence=float(labels.mean()),
        enrichment_top_1pct=enrichment(values, labels, 0.01),
        enrichment_top_5pct=enrichment(values, labels, 0.05),
        enrichment_top_10pct=enrichment(values, labels, 0.10),
        enrichment_top_20pct=enrichment(values, labels, 0.20),
        effective_n=effective_n(values, grid) if grid is not None else float(len(values)),
        hypothesis_tier=tier,
    )


def block_bootstrap_auc_ci(values: np.ndarray, labels: np.ndarray, grid, *,
                           n_boot: int = 999, block_cells: int | None = None,
                           alpha: float = 0.05, seed: int = 0
                           ) -> tuple[float, float]:
    """Percentile CI for an AUC, resampling contiguous blocks of cells.

    Resampling individual cells would treat neighbours as independent draws
    and return an interval far too narrow -- the same error that makes a
    56-cell grid look like 56 observations.
    """
    from .permutation import autocorrelation_range_cells

    labels = labels.astype(int)
    if block_cells is None:
        block_cells = max(autocorrelation_range_cells(values, grid), 1)

    rows = np.array([c.row for c in grid.cells])
    cols = np.array([c.col for c in grid.cells])
    bid = (rows // block_cells) * (grid.n_cols // block_cells + 1) + (cols // block_cells)
    groups = [np.where(bid == b)[0] for b in np.unique(bid)]
    if len(groups) < 3:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[j] for j in pick])
        v, l = values[idx], labels[idx]
        if l.sum() == 0 or l.sum() == len(l):
            continue
        out.append(roc_auc(v[l == 1], v[l == 0]))
    if len(out) < 20:
        return (float("nan"), float("nan"))
    return (float(np.percentile(out, 100 * alpha / 2)),
            float(np.percentile(out, 100 * (1 - alpha / 2))))
