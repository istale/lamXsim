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
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


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
