"""Evidence classes (spec section 30).

The engine must never silently convert one evidence class into another.
Every feature, label and statistic carries the class it belongs to, and
:func:`assert_no_mixing` guards the boundaries that matter.
"""
from __future__ import annotations

from enum import Enum


class EvidenceClass(str, Enum):
    GDS_GEOMETRY = "GDS_GEOMETRY"
    PACKAGE_POSITION = "PACKAGE_POSITION"
    MEASURED_FAILURE = "MEASURED_FAILURE"
    STATISTICAL_ASSOCIATION = "STATISTICAL_ASSOCIATION"
    ML_PREDICTION = "ML_PREDICTION"
    VISION_EMBEDDING = "VISION_EMBEDDING"
    FEM_RESULT = "FEM_RESULT"
    FRACTURE_MECHANICS_RESULT = "FRACTURE_MECHANICS_RESULT"


#: Feature families that may enter a *geometry* model.
GEOMETRY_MODEL_CLASSES = frozenset({EvidenceClass.GDS_GEOMETRY})

#: Feature families that may enter the *position-only baseline* model.
#: Keeping these apart is what makes "does geometry add anything beyond
#: die position?" an answerable question (see stats.baseline).
POSITION_MODEL_CLASSES = frozenset({EvidenceClass.PACKAGE_POSITION})


class EvidenceMixingError(RuntimeError):
    pass


def assert_no_mixing(classes, allowed, context: str) -> None:
    """Raise if *classes* contains anything outside *allowed*."""
    bad = {EvidenceClass(c) for c in classes} - set(allowed)
    if bad:
        raise EvidenceMixingError(
            f"{context}: evidence classes {sorted(c.value for c in bad)} are not "
            f"permitted here (allowed: {sorted(c.value for c in allowed)})"
        )
