"""Association, correction, resampling, power and validation.

Consolidated from ``stats/fdr.py``, ``stats/univariate.py``, ``stats/permutation.py``, ``stats/power.py``, ``stats/cv.py``, ``stats/baseline.py``, ``stats/ablation.py``.
"""
from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import re
from .foundation import EvidenceClass


# ----------------------------------------------------------------------
# stats/fdr.py
# ----------------------------------------------------------------------
"""Benjamini-Hochberg FDR, applied within hypothesis tiers (spec section 12).

Correcting all ~8000 feature x layer x scale combinations together leaves
nothing significant at realistic failure counts. The tiers come from
references/feature_evidence_map.csv: literature-backed features are
primary hypotheses and are corrected; derived descriptors are exploratory
and report effect size only, without a significance claim.
"""
def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    q[ok] = out
    return q


def apply_tiered(associations, primary_tiers=("tier1", "tier1_confounder")):
    """Correct the primary-tier rows, on both p-values, keeping them apart.

    ``spatial_q_value`` comes from the within-die block permutation and is what
    a primary claim rests on. ``fdr_q_value`` comes from Mann-Whitney, which
    assumes grid cells are independent observations -- on a die with no
    package-position effect that test called 11 of 12 position associations
    significant where the permutation called none. It is retained as a
    descriptive diagnostic and as the contrast that shows what the spatial
    correction is doing, never as the basis of a finding.

    Exploratory rows get neither: correcting them alongside the primary ones
    is what makes the primary ones unreachable.
    """
    prim = [a for a in associations if a.hypothesis_tier in primary_tiers]
    if not prim:
        return associations

    for source, target in (("p_value", "fdr_q_value"),
                           ("spatial_p_value", "spatial_q_value")):
        q = benjamini_hochberg(np.array([getattr(a, source) for a in prim]))
        for a, qq in zip(prim, q):
            setattr(a, target, float(qq))
    return associations

# ----------------------------------------------------------------------
# stats/univariate.py
# ----------------------------------------------------------------------
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
    #: Mann-Whitney, which assumes independent observations. Kept as a
    #: descriptive diagnostic; it is not what a primary claim rests on.
    fdr_q_value: float = float("nan")
    #: Block permutation within a die, which preserves the spatial structure.
    #: This is the p-value a primary result is corrected from.
    spatial_p_value: float = float("nan")
    spatial_q_value: float = float("nan")
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


def effective_n(values: np.ndarray, grid, mask: np.ndarray | None = None) -> float:
    """Spatially-corrected sample size using Moran's I on a rook lattice.

    Grid cells are not independent samples. Reporting the raw cell count
    alongside an AUC invites reading a 6400-cell grid as 6400 observations
    when the layout may only vary over a few dozen independent patches.
    """
    keep = np.ones(len(values), bool) if mask is None else np.asarray(mask, bool)
    n = int(keep.sum())
    if n < 4:
        return float(n)
    idx = {(c.row, c.col): i for i, c in enumerate(grid.cells) if keep[i]}
    z = np.where(keep, values - values[keep].mean(), 0.0)
    denom = float((z[keep] ** 2).sum())
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
                           alpha: float = 0.05, seed: int = 0,
                           mask: np.ndarray | None = None,
                           groups: np.ndarray | None = None
                           ) -> tuple[float, float]:
    """Percentile CI for an AUC, resampling contiguous blocks of cells.

    Resampling individual cells would treat neighbours as independent draws
    and return an interval far too narrow -- the same error that makes a
    56-cell grid look like 56 observations.
    """
    pass

    labels = labels.astype(int)
    if block_cells is None and groups is None:
        block_cells = max(autocorrelation_range_cells(values, grid), 1)
    elif block_cells is None:
        block_cells = 1

    if groups is None:
        rows = np.array([c.row for c in grid.cells])
        cols = np.array([c.col for c in grid.cells])
        bid = ((rows // block_cells) * (grid.n_cols // block_cells + 1)
               + (cols // block_cells))
    else:
        bid = np.asarray(groups)
    keep = np.ones(len(values), bool) if mask is None else np.asarray(mask, bool)
    groups = [g[keep[g]] for g in
              (np.where(bid == b)[0] for b in np.unique(bid))]
    groups = [g for g in groups if len(g)]
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

# ----------------------------------------------------------------------
# stats/permutation.py
# ----------------------------------------------------------------------
"""Spatial null model (spec section 15).

Grid cells adjacent in space are correlated, so shuffling individual cell
labels destroys that structure and produces a null distribution far narrower
than reality -- which makes ordinary features look significant. Labels are
therefore permuted in contiguous blocks whose size is chosen from the
measured spatial autocorrelation rather than picked by hand.
"""
@dataclass
class PermutationResult:
    observed: float
    null_mean: float
    null_sd: float
    p_value: float
    n_permutations: int
    block_cells: int
    block_um: float
    #: Blocks with no same-sized partner to exchange with. They keep their
    #: labels, so the test is conservative there rather than approximate.
    n_blocks_not_exchangeable: int = 0
    n_blocks: int = 0

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
                           groups: np.ndarray | None = None,
                           strata: np.ndarray | None = None) -> PermutationResult:
    """Permute labels in contiguous square blocks and compare the statistic.

    ``strata`` confines the permutation: blocks are only ever exchanged with
    other blocks in the same stratum. With one stratum per die that keeps each
    die's failure count fixed, which matters because dies differ in process
    lot, inspection sensitivity and base rate -- letting a permutation move
    failures between them builds a null in which those differences do not
    exist, and narrows it against a real effect.

    Grouping observations by ``(die, block)`` is not sufficient on its own: it
    only decides which cells form a block, while the exchange still pairs
    every block with every other.
    """
    pass

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

    # Partition the blocks by stratum; a permutation only ever exchanges
    # blocks that share one.
    if strata is None:
        block_strata = [list(range(len(groups)))]
    else:
        strata = np.asarray(strata)
        # A block lies in one stratum by construction when the grouping
        # already encodes it; take the first member's stratum.
        by_stratum: dict = {}
        for i, g in enumerate(groups):
            by_stratum.setdefault(strata[g[0]], []).append(i)
        block_strata = list(by_stratum.values())

    # Blocks are exchanged only with blocks of the same size, within a
    # stratum. Concatenating every block's labels and slicing them out by
    # target size instead splits one source block across two targets and
    # merges parts of two into a third, which is no longer a block
    # permutation: the within-block structure it exists to preserve is broken
    # exactly at the die edge and the ROI boundary, where blocks are ragged,
    # and the null comes out narrower than the truth.
    exchange_sets = []
    ragged = 0
    for members in block_strata:
        by_size: dict[int, list[int]] = {}
        for i in members:
            by_size.setdefault(len(groups[i]), []).append(i)
        for size, same in by_size.items():
            if len(same) > 1:
                exchange_sets.append(same)
            else:
                ragged += len(same)

    observed = statistic(values[keep], labels[keep])
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    for k in range(n_permutations):
        shuffled = labels.copy()          # blocks with no partner stay put
        for members in exchange_sets:
            order = rng.permutation(len(members))
            for target, source in zip(members, [members[j] for j in order]):
                shuffled[groups[target]] = labels[groups[source]]
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
        n_blocks_not_exchangeable=int(ragged), n_blocks=len(groups),
    )


def spatial_block_ids(grid, block_cells: int = 1) -> np.ndarray:
    """Contiguous square block index per grid cell."""
    rows = np.array([c.row for c in grid.cells])
    cols = np.array([c.col for c in grid.cells])
    return ((rows // block_cells) * (grid.n_cols // block_cells + 1)
            + (cols // block_cells))

# ----------------------------------------------------------------------
# stats/power.py
# ----------------------------------------------------------------------
"""Phase 0 power analysis.

Run before any feature engineering. It answers the question that decides
whether the platform can produce a scientific result at all: given the number
of feature x layer x scale hypotheses being tested and the effect sizes that
are realistically on offer, how many measured failure sites are needed?

Two corrections matter and are reported separately:

* Multiple comparisons. Testing every combination gives a hypothesis count in
  the thousands, and the per-test alpha collapses accordingly.
* Spatial autocorrelation. Grid cells are not independent samples, so the
  usable sample size is the number of independent *patches*, not cells. The
  design effect converts one into the other.
"""
@dataclass(frozen=True)
class HypothesisBudget:
    """How many tests the analysis will actually run (spec sections 12-14).

    The layer-pair set is the largest single lever. Testing every pair of 12
    layers is 66 combinations; the pairs the literature motivates -- adjacent
    layers for the local mechanical relationship, and the top layer against
    each underlying one for the chip-package relationship -- is 21. Choosing
    the pair set before extraction is legitimate; choosing it after seeing
    results is not.
    """
    n_features: int
    n_layers: int
    n_scales: int
    n_gradient_derived: int = 3      # dQ_dx, dQ_dy, |grad Q| per scalar feature
    n_cross_layer_features: int = 8  # 4 relationships, signed and magnitude
    pair_selection: str = "adjacent_and_top"

    @property
    def n_pairs(self) -> int:
        n = self.n_layers
        if self.pair_selection == "all":
            return n * (n - 1) // 2
        if self.pair_selection == "adjacent":
            return n - 1
        if self.pair_selection == "top_vs_all":
            return n - 1
        if self.pair_selection == "adjacent_and_top":
            return (n - 1) + (n - 2)          # adjacent, plus top-vs-non-adjacent
        raise ValueError(f"unknown pair selection {self.pair_selection!r}")

    @property
    def per_layer_hypotheses(self) -> int:
        return self.n_features * (1 + self.n_gradient_derived) * self.n_layers * self.n_scales

    @property
    def cross_layer_hypotheses(self) -> int:
        return self.n_pairs * self.n_cross_layer_features * self.n_scales

    @property
    def total(self) -> int:
        return self.per_layer_hypotheses + self.cross_layer_hypotheses


def auc_se(auc: float, n_case: int, n_control: int) -> float:
    """Hanley-McNeil standard error of an AUC."""
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc)
           + (n_case - 1) * (q1 - auc ** 2)
           + (n_control - 1) * (q2 - auc ** 2)) / (n_case * n_control)
    return float(np.sqrt(max(var, 1e-12)))


def auc_power(auc: float, n_case: int, n_control: int, alpha: float) -> float:
    """Two-sided power to detect *auc* against the 0.5 null."""
    se_null = auc_se(0.5, n_case, n_control)
    se_alt = auc_se(auc, n_case, n_control)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    z = (abs(auc - 0.5) - z_crit * se_null) / se_alt
    return float(stats.norm.cdf(z))


def required_cases(auc: float, alpha: float, power: float = 0.80,
                   control_ratio: float = 4.0, design_effect: float = 1.0,
                   max_n: int = 200_000) -> float:
    """Smallest case count reaching *power*, inflated by the design effect.

    ``design_effect`` is the spatial-autocorrelation penalty: with correlated
    cells, N nominal observations carry the information of N/design_effect
    independent ones.
    """
    lo, hi = 2, max_n
    if auc_power(auc, hi, int(hi * control_ratio), alpha) < power:
        return float("inf")
    while lo < hi:
        mid = (lo + hi) // 2
        if auc_power(auc, mid, max(int(mid * control_ratio), 1), alpha) >= power:
            hi = mid
        else:
            lo = mid + 1
    return float(lo * design_effect)


def design_effect_from_moran(moran_i: float, cells_per_patch: float) -> float:
    """Kish-style design effect for spatially clustered observations."""
    rho = float(np.clip(moran_i, 0.0, 0.999))
    return float(1 + (cells_per_patch - 1) * rho)


def sample_size_table(budget: HypothesisBudget, *,
                      aucs=(0.60, 0.65, 0.70, 0.75, 0.80, 0.85),
                      alpha: float = 0.05, power: float = 0.80,
                      control_ratio: float = 4.0,
                      design_effect: float = 1.0,
                      tier1_hypotheses: int | None = None) -> pd.DataFrame:
    """Required failure-site counts under three correction regimes."""
    m_all = budget.total
    m_t1 = tier1_hypotheses or 20

    rows = []
    for auc in aucs:
        for label, m in (("uncorrected", 1),
                         ("tiered_FDR (literature-backed only)", m_t1),
                         ("full_grid_FDR (every combination)", m_all)):
            a = alpha if m == 1 else alpha / m
            n = required_cases(auc, a, power, control_ratio, design_effect)
            rows.append({
                "target_roc_auc": auc,
                "correction": label,
                "n_hypotheses": m,
                "alpha_per_test": a,
                "design_effect": design_effect,
                "required_failure_sites": n,
                "required_control_sites": n * control_ratio if np.isfinite(n) else np.inf,
            })
    return pd.DataFrame(rows)


def registration_scale_floor(position_sigma_um: float, factor: float = 3.0) -> dict:
    """Which analysis scales the measurement's positional accuracy supports."""
    floor = factor * position_sigma_um
    default_scales = [25, 50, 100, 250, 500, 1000]
    return {
        "position_sigma_um": position_sigma_um,
        "min_trustworthy_scale_um": floor,
        "trustworthy_scales_um": [s for s in default_scales if s >= floor],
        "rejected_scales_um": [s for s in default_scales if s < floor],
    }


def min_achievable_p(n_permutations: int) -> float:
    """The smallest p a permutation test can return.

    With ``n`` permutations the observed statistic can at best be the most
    extreme of ``n + 1`` values, so no p below ``1 / (n + 1)`` exists. A test
    that cannot produce a small enough p cannot be corrected into
    significance however strong the effect is.
    """
    return 1.0 / (n_permutations + 1)


def required_permutations(n_tests: int, alpha: float = 0.05,
                          rank: int = 1) -> int:
    """Permutations needed for a test at *rank* to be able to reach *alpha*.

    Benjamini-Hochberg compares the rank-i p-value against ``alpha * i / m``,
    so the strictest requirement falls on the most significant test. Below
    this count the spatial correction has a resolution floor above its own
    threshold, and the strongest possible result is indistinguishable from a
    marginal one.
    """
    threshold = alpha * rank / max(n_tests, 1)
    return int(np.ceil(1.0 / threshold)) - 1


def permutation_budget(n_tests: int, n_permutations: int,
                       alpha: float = 0.05) -> dict:
    """Whether the configured permutation count can resolve this family."""
    floor = min_achievable_p(n_permutations)
    needed = required_permutations(n_tests, alpha)
    # With every test tied at the floor, BH still clears alpha at the largest
    # rank; the binding case is one strong result among mostly-null ones.
    best_q = floor * n_tests
    return {
        "n_tests": n_tests, "n_permutations": n_permutations,
        "min_achievable_p": floor,
        "best_achievable_q_for_a_lone_result": min(best_q, 1.0),
        "permutations_needed_for_alpha": needed,
        "sufficient": bool(best_q <= alpha),
    }

# ----------------------------------------------------------------------
# stats/cv.py
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# stats/baseline.py
# ----------------------------------------------------------------------
"""Interpretable multivariate baseline (spec section 16).

Deliberately a regularised logistic regression rather than anything with more
capacity. The question at this stage is whether deterministic geometry carries
information, not how much accuracy can be extracted from it, and a model whose
coefficients can be read is worth more here than one that scores higher.

Every score is produced under spatially separated folds, and every geometry
model is reported as a *difference* against the position-only baseline. An
absolute AUC answers "can something predict this", which is not the question;
"does layout geometry add anything beyond where on the die you are" is.
"""
def make_model(C: float = 1.0, l1_ratio: float = 0.0) -> Pipeline:
    """Standardised, class-balanced logistic regression.

    Standardisation is not cosmetic here: features arrive in wildly different
    units (a dimensionless density, a per-um perimeter, a per-um^2 line-end
    count), and an unscaled penalty would regularise them by unit rather than
    by importance.

    ``l1_ratio`` selects the penalty: 0 is ridge, 1 is lasso, in between is
    elastic net. Sparse solutions need the saga solver, which is slower, so
    ridge stays the default.
    """
    if l1_ratio == 0.0:
        lr = LogisticRegression(C=C, solver="lbfgs", max_iter=5000,
                                class_weight="balanced")
    else:
        lr = LogisticRegression(C=C, solver="saga", l1_ratio=l1_ratio,
                                max_iter=8000, class_weight="balanced")
    return Pipeline([("scale", StandardScaler()), ("lr", lr)])


@dataclass
class ModelScore:
    name: str
    evidence_class: str
    n_features: int
    feature_names: list[str]
    roc_auc: float
    pr_auc: float
    prevalence: float
    brier: float
    calibration_slope: float
    calibration_intercept: float
    enrichment_top_1pct: float
    enrichment_top_5pct: float
    enrichment_top_10pct: float
    n_test_total: int
    n_case_total: int
    folds: list[dict] = field(default_factory=list)
    coefficients: dict[str, float] = field(default_factory=dict)
    oof_pred: np.ndarray | None = None
    oof_true: np.ndarray | None = None

    def as_row(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()
             if k not in ("folds", "oof_pred", "oof_true", "coefficients",
                          "feature_names")}
        return d


def _calibration(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Slope and intercept of the logit-linear calibration line.

    A slope near 1 means the spread of predicted risk matches reality; a slope
    well below 1 is the signature of an overfitted model whose confident
    predictions are not earned.
    """
    eps = 1e-6
    logit = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    if len(np.unique(y)) < 2 or np.std(logit) == 0:
        return float("nan"), float("nan")
    lr = LogisticRegression(solver="lbfgs", max_iter=2000, C=1e6)
    lr.fit(logit.reshape(-1, 1), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def _enrichment(p: np.ndarray, y: np.ndarray, frac: float) -> float:
    n = len(p)
    k = max(int(round(n * frac)), 1)
    if k >= n:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    top, rest = y[order[:k]], y[order[k:]]
    if rest.mean() == 0:
        return float("inf") if top.mean() > 0 else float("nan")
    return float(top.mean() / rest.mean())


def evaluate(X: np.ndarray, y: np.ndarray, folds, *, name: str,
             feature_names: list[str], evidence_class: EvidenceClass,
             C: float = 1.0) -> ModelScore:
    """Out-of-fold evaluation under the supplied spatial folds."""
    y = np.asarray(y).astype(int)
    oof_p, oof_y, rows = [], [], []

    for f in folds:
        tr, te = f.train, f.test
        if len(np.unique(y[tr])) < 2 or len(te) == 0:
            continue
        model = make_model(C=C)
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        oof_p.append(p)
        oof_y.append(y[te])
        rows.append({
            "fold": f.label, "n_train": len(tr), "n_test": len(te),
            "n_excluded": len(f.excluded), "n_case_test": int(y[te].sum()),
            "roc_auc": (roc_auc(p[y[te] == 1], p[y[te] == 0])
                        if 0 < y[te].sum() < len(te) else float("nan")),
        })

    if not oof_p:
        raise ValueError(f"{name}: no fold had both classes in training")

    p = np.concatenate(oof_p)
    t = np.concatenate(oof_y)
    auc = roc_auc(p[t == 1], p[t == 0]) if 0 < t.sum() < len(t) else float("nan")
    slope, intercept = _calibration(t, p)

    final = make_model(C=C).fit(X, y)
    coef = dict(zip(feature_names, final.named_steps["lr"].coef_[0]))

    return ModelScore(
        name=name, evidence_class=evidence_class.value, n_features=X.shape[1],
        feature_names=list(feature_names), roc_auc=float(auc),
        pr_auc=float(pr_auc(p, t)), prevalence=float(t.mean()),
        brier=float(np.mean((p - t) ** 2)),
        calibration_slope=slope, calibration_intercept=intercept,
        enrichment_top_1pct=_enrichment(p, t, 0.01),
        enrichment_top_5pct=_enrichment(p, t, 0.05),
        enrichment_top_10pct=_enrichment(p, t, 0.10),
        n_test_total=len(t), n_case_total=int(t.sum()),
        folds=rows, coefficients=coef, oof_pred=p, oof_true=t)


def delta_auc(model: ModelScore, baseline: ModelScore, *, n_boot: int = 999,
              block_size: int = 64, seed: int = 0) -> dict:
    """Paired improvement over the baseline, with a block bootstrap interval.

    Paired on the same out-of-fold predictions, and resampled in contiguous
    blocks rather than per observation, because neighbouring predictions are
    correlated and a per-observation bootstrap would return an interval far
    too narrow.
    """
    if model.oof_pred is None or baseline.oof_pred is None:
        raise ValueError("both scores must carry out-of-fold predictions")
    n = min(len(model.oof_pred), len(baseline.oof_pred))
    pm, pb, y = model.oof_pred[:n], baseline.oof_pred[:n], model.oof_true[:n]

    observed = model.roc_auc - baseline.roc_auc
    rng = np.random.default_rng(seed)
    starts = np.arange(0, n, block_size)
    deltas = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(starts), size=len(starts))
        idx = np.concatenate([np.arange(starts[j], min(starts[j] + block_size, n))
                              for j in pick])
        yy = y[idx]
        if not 0 < yy.sum() < len(yy):
            continue
        deltas.append(roc_auc(pm[idx][yy == 1], pm[idx][yy == 0])
                      - roc_auc(pb[idx][yy == 1], pb[idx][yy == 0]))
    if len(deltas) < 20:
        return {"delta_auc": observed, "ci_low": float("nan"),
                "ci_high": float("nan"), "n_boot": len(deltas)}
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "model": model.name, "baseline": baseline.name,
        "model_auc": model.roc_auc, "baseline_auc": baseline.roc_auc,
        "delta_auc": float(observed), "ci_low": float(lo), "ci_high": float(hi),
        "n_boot": len(deltas),
        "adds_information": bool(lo > 0),
    }

# ----------------------------------------------------------------------
# stats/ablation.py
# ----------------------------------------------------------------------
"""Feature-family ablation (spec section 18).

The question is whether sophisticated geometry carries predictive information
beyond metal density -- not whether the full model scores well. Families are
therefore nested, each adding one kind of geometry to the one before, and
every model is reported against the position-only baseline rather than
against chance.
"""
#: Nested families, in the order spec section 18 lists them. Patterns match the
#: feature-name column produced by the pipeline.
FAMILIES: dict[str, tuple[str, ...]] = {
    "A_metal_density": (r"^metal_density$",),
    "B_plus_via": (r"^metal_density$", r"^via_density", r"^via_count_density"),
    "C_plus_perimeter": (r"^metal_density$", r"^via_density", r"^via_count_density",
                         r"^perimeter_density$"),
    "D_plus_termination_corners": (
        r"^metal_density$", r"^via_density", r"^via_count_density",
        r"^perimeter_density$", r"^line_end_density$", r"corner_density$"),
    "E_plus_orientation": (
        r"^metal_density$", r"^via_density", r"^via_count_density",
        r"^perimeter_density$", r"^line_end_density$", r"corner_density$",
        r"^horizontal_fraction$", r"^vertical_fraction$",
        r"^orientation_anisotropy$"),
    "F_plus_gradients": (
        r"^metal_density$", r"^via_density", r"^via_count_density",
        r"^perimeter_density$", r"^line_end_density$", r"corner_density$",
        r"^horizontal_fraction$", r"^vertical_fraction$",
        r"^orientation_anisotropy$", r"_grad_mag$", r"_dx$", r"_dy$"),
    "G_plus_cross_layer": (
        r"^metal_density$", r"^via_density", r"^via_count_density",
        r"^perimeter_density$", r"^line_end_density$", r"corner_density$",
        r"^horizontal_fraction$", r"^vertical_fraction$",
        r"^orientation_anisotropy$", r"_grad_mag$", r"_dx$", r"_dy$",
        r"_difference_", r"_mismatch_", r"^cross_layer_", r"^stacked_",
        r"^density_variance_across_layers$", r"^top_to_underlying"),
}

#: Name of the mandatory reference model, referred to by tests and reports.
BASELINE_NAME = "P_position_and_conditions"

#: The mandatory reference model. Geometry is only interesting relative to it,
#: and that includes the package and process conditions declared as
#: covariates: leaving them out lets a geometry feature absorb their effect.
POSITION_FAMILY = (r"^distance_to_", r"^normalized_distance_", r"^bump_",
                   r"^under_bump_indicator$", r"^local_bump_pitch$",
                   r"^condition_")


@dataclass
class AblationResult:
    scores: list[ModelScore]
    deltas: list[dict]
    table: pd.DataFrame


def select_columns(columns, patterns) -> list[str]:
    """Column names matching any pattern, in the order they appear."""
    out = []
    for c in columns:
        base = c.split("|")[0]
        if any(re.search(p, base) for p in patterns):
            out.append(c)
    return out


def run(frame: pd.DataFrame, y: np.ndarray, folds, *,
        families: dict[str, tuple[str, ...]] | None = None,
        C: float = 1.0, seed: int = 0) -> AblationResult:
    """Fit the position baseline and each nested geometry family.

    Columns carrying non-finite values -- gradients with their die-edge ring
    dropped, for one -- are excluded rather than imputed. Filling them would
    reintroduce the boundary artifact the gradient module removes.
    """
    families = FAMILIES if families is None else families
    numeric = [c for c in frame.columns
               if frame[c].dtype.kind in "fiu"
               and c not in ("cell_id", "row", "col", "x_um", "y_um",
                             "scale_um", "failure_present",
                             "distance_to_nearest_failure")]
    usable = [c for c in numeric if np.isfinite(frame[c].to_numpy(float)).all()
              and frame[c].std() > 0]

    scores: list[ModelScore] = []

    pos_cols = select_columns(usable, POSITION_FAMILY)
    if not pos_cols:
        raise ValueError(
            "no PACKAGE_POSITION columns present; without the position-only "
            "baseline a geometry AUC cannot be interpreted")
    baseline = evaluate(frame[pos_cols].to_numpy(float), y, folds,
                        name=BASELINE_NAME, feature_names=pos_cols,
                        evidence_class=EvidenceClass.PACKAGE_POSITION, C=C)
    scores.append(baseline)

    deltas = []
    for name, patterns in families.items():
        cols = select_columns(usable, patterns)
        if not cols:
            continue
        s = evaluate(frame[cols].to_numpy(float), y, folds, name=name,
                     feature_names=cols,
                     evidence_class=EvidenceClass.GDS_GEOMETRY, C=C)
        scores.append(s)
        deltas.append(delta_auc(s, baseline, seed=seed))

        combined = pos_cols + [c for c in cols if c not in pos_cols]
        s2 = evaluate(frame[combined].to_numpy(float), y, folds,
                      name=f"{name}+position", feature_names=combined,
                      evidence_class=EvidenceClass.GDS_GEOMETRY, C=C)
        scores.append(s2)
        deltas.append(delta_auc(s2, baseline, seed=seed))

    table = pd.DataFrame([s.as_row() for s in scores])
    return AblationResult(scores=scores, deltas=deltas, table=table)
