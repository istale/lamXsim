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
    """Set fdr_q_value on primary-tier rows; leave exploratory rows as NaN."""
    prim = [a for a in associations if a.hypothesis_tier in primary_tiers]
    if prim:
        q = benjamini_hochberg(np.array([a.p_value for a in prim]))
        for a, qq in zip(prim, q):
            a.fdr_q_value = float(qq)
    return associations
