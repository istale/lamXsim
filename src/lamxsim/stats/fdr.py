"""Benjamini-Hochberg FDR, applied within hypothesis tiers (spec section 12).

Correcting all ~8000 feature x layer x scale combinations together leaves
nothing significant at realistic failure counts. The tiers come from
references/feature_evidence_map.csv: literature-backed features are
primary hypotheses and are corrected; derived descriptors are exploratory
and report effect size only, without a significance claim.
"""
from __future__ import annotations

import numpy as np


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
