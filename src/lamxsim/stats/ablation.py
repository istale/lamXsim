"""Feature-family ablation (spec section 18).

The question is whether sophisticated geometry carries predictive information
beyond metal density -- not whether the full model scores well. Families are
therefore nested, each adding one kind of geometry to the one before, and
every model is reported against the position-only baseline rather than
against chance.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..evidence import EvidenceClass
from .baseline import ModelScore, delta_auc, evaluate

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

#: The mandatory reference model. Geometry is only interesting relative to it.
POSITION_FAMILY = (r"^distance_to_", r"^normalized_distance_")


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
                        name="P_position_only", feature_names=pos_cols,
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
