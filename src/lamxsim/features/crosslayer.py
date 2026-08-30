"""Cross-layer architecture features (spec section 7).

Vanstreels et al. (2020) correlate BEOL architecture with observed fracture
counts, and Zahedmanesh & Vanstreels (2019) show a stiff top metal group can
*lower* the crack driving force in the layer beneath it. Two consequences are
built in here.

**Layer identity is preserved.** Features are named ``density_difference_M8_M7``,
never ``generic_density_difference``. A pooled cross-layer index would average
a shielding pair against a loading pair and report neither.

**Differences are signed, in a fixed order.** ``density_difference_A_B`` is
``density(A) - density(B)``, upper layer first. Taking an absolute value would
erase precisely the distinction the shielding result rests on.

**The magnitude is emitted alongside the signed value, not instead of it.**
They answer different questions and neither substitutes for the other: a
signed difference cannot detect an effect driven by how *much* two layers
disagree, because both directions of disagreement sit at opposite ends of the
scale and the association collapses to chance. Measured on a die whose driver
is orientation mismatch, the signed feature scores AUC 0.50 while its own
absolute value scores 0.78 -- from identical inputs.

The pair set is the main lever on the hypothesis budget: all pairs of 12
layers is 66 combinations, while the pairs the literature actually motivates
-- adjacent layers, and the top layer against each underlying one -- is 21.
Choose the pair set before extraction, not after seeing results.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from ..evidence import EvidenceClass

EVIDENCE_CLASS = EvidenceClass.GDS_GEOMETRY

#: Features computed for each selected layer pair. Each appears twice: signed
#: (``*_difference_A_B``) and as a magnitude (``*_mismatch_A_B``).
PAIR_FEATURES = ("density_difference", "perimeter_density_difference",
                 "orientation_difference", "line_end_density_difference",
                 "density_mismatch", "perimeter_density_mismatch",
                 "orientation_mismatch", "line_end_density_mismatch")

#: Features computed once across the whole stack.
STACK_FEATURES = ("density_variance_across_layers", "stacked_dense_layer_count",
                  "stacked_sparse_layer_count", "cross_layer_transition_index")


@dataclass(frozen=True)
class LayerStack:
    """Ordered layer names, topmost first."""
    names: tuple[str, ...]

    @property
    def top(self) -> str:
        return self.names[0]

    def pairs(self, selection: str = "adjacent_and_top") -> list[tuple[str, str]]:
        """Layer pairs to compute, upper layer first.

        ``adjacent`` is the mechanically local relationship; ``top_vs_all``
        is the chip-package one. Together they cover what the literature
        motivates at a fraction of the hypothesis count of ``all``.
        """
        n = self.names
        if selection == "all":
            return list(combinations(n, 2))
        adjacent = [(n[i], n[i + 1]) for i in range(len(n) - 1)]
        if selection == "adjacent":
            return adjacent
        top = [(n[0], m) for m in n[1:]]
        if selection == "top_vs_all":
            return top
        if selection == "adjacent_and_top":
            seen, out = set(), []
            for p in adjacent + top:
                if p not in seen:
                    seen.add(p)
                    out.append(p)
            return out
        raise ValueError(f"unknown pair selection {selection!r}")

    def hypothesis_count(self, selection: str, n_scales: int,
                         n_pair_features: int = len(PAIR_FEATURES)) -> int:
        return len(self.pairs(selection)) * n_pair_features * n_scales


def pair_features(per_layer: dict[str, dict[str, np.ndarray]],
                  upper: str, lower: str) -> dict[str, np.ndarray]:
    """Signed differences between two layers on the same grid."""
    a, b = per_layer[upper], per_layer[lower]
    out = {}
    mapping = {
        "density_difference": "metal_density",
        "perimeter_density_difference": "perimeter_density",
        "line_end_density_difference": "line_end_density",
        "orientation_difference": "orientation_anisotropy",
    }
    for feature, source in mapping.items():
        if source in a and source in b:
            diff = a[source] - b[source]
            out[f"{feature}_{upper}_{lower}"] = diff
            magnitude = feature.replace("_difference", "_mismatch")
            out[f"{magnitude}_{upper}_{lower}"] = np.abs(diff)
    return out


def stack_features(per_layer: dict[str, dict[str, np.ndarray]],
                   stack: LayerStack, *, dense_threshold: float = 0.5,
                   sparse_threshold: float = 0.2) -> dict[str, np.ndarray]:
    """Whole-stack summaries at each location.

    The dense/sparse counts are the layout analogue of the stiff-group idea:
    how many layers are heavily metallised above a given point, rather than
    how much metal any one of them carries.
    """
    names = [n for n in stack.names if n in per_layer]
    if not names:
        return {}
    dens = np.vstack([per_layer[n]["metal_density"] for n in names])

    out = {
        "density_variance_across_layers": dens.var(axis=0),
        "stacked_dense_layer_count": (dens >= dense_threshold).sum(axis=0).astype(float),
        "stacked_sparse_layer_count": (dens <= sparse_threshold).sum(axis=0).astype(float),
    }
    # Transition index: how much the stack changes from layer to layer at this
    # point. A uniformly dense stack and a uniformly sparse one both score 0;
    # a dense-over-sparse interface scores high.
    if len(names) > 1:
        out["cross_layer_transition_index"] = np.abs(np.diff(dens, axis=0)).mean(axis=0)
    return out


def top_vs_underlying(per_layer: dict[str, dict[str, np.ndarray]],
                      stack: LayerStack) -> dict[str, np.ndarray]:
    """Top layer against the mean of everything beneath it (spec section 8)."""
    under = [n for n in stack.names[1:] if n in per_layer]
    if stack.top not in per_layer or not under:
        return {}
    top = per_layer[stack.top]
    out = {}
    for feature, source in (("top_to_underlying_density_mismatch", "metal_density"),
                            ("top_to_underlying_orientation_mismatch",
                             "orientation_anisotropy")):
        if source not in top:
            continue
        mean_under = np.mean([per_layer[n][source] for n in under
                              if source in per_layer[n]], axis=0)
        out[feature] = top[source] - mean_under
    return out


def extract(per_layer: dict[str, dict[str, np.ndarray]], stack: LayerStack, *,
            selection: str = "adjacent_and_top") -> dict[str, np.ndarray]:
    """All cross-layer features for one grid."""
    out: dict[str, np.ndarray] = {}
    for upper, lower in stack.pairs(selection):
        if upper in per_layer and lower in per_layer:
            out.update(pair_features(per_layer, upper, lower))
    out.update(stack_features(per_layer, stack))
    out.update(top_vs_underlying(per_layer, stack))
    return out
