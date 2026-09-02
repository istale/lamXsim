"""Units, evidence classes and the feature registry.

Consolidated from ``units.py``, ``evidence.py``, ``registry.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
import csv


# ----------------------------------------------------------------------
# units.py
# ----------------------------------------------------------------------
"""Physical-unit helpers.

Everything the engine reasons about is in micrometres. KLayout works in
database units (dbu); conversion happens only at the layout boundary so that
no downstream module ever sees a dbu.
"""
class Units:
    __slots__ = ("dbu",)

    def __init__(self, dbu: float):
        if dbu <= 0:
            raise ValueError(f"dbu must be positive, got {dbu}")
        self.dbu = float(dbu)

    def um_to_dbu(self, um: float) -> int:
        return int(round(um / self.dbu))

    def dbu_to_um(self, d: float) -> float:
        return d * self.dbu

    def area_dbu2_to_um2(self, a: float) -> float:
        return a * self.dbu * self.dbu

    def length_dbu_to_um(self, l: float) -> float:
        return l * self.dbu

    def __repr__(self) -> str:
        return f"Units(dbu={self.dbu})"

# ----------------------------------------------------------------------
# evidence.py
# ----------------------------------------------------------------------
"""Evidence classes (spec section 30).

The engine must never silently convert one evidence class into another.
Every feature, label and statistic carries the class it belongs to, and
:func:`assert_no_mixing` guards the boundaries that matter.
"""
class EvidenceClass(str, Enum):
    GDS_GEOMETRY = "GDS_GEOMETRY"
    PACKAGE_POSITION = "PACKAGE_POSITION"
    #: Package and process conditions -- EMC thickness, underfill CTE, thermal
    #: cycle, dielectric stack. Not in the GDS and not a position, but they
    #: change the crack driving force, so they belong in the baseline a
    #: geometry model has to beat rather than in the geometry model itself.
    SAMPLE_CONDITION = "SAMPLE_CONDITION"
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
POSITION_MODEL_CLASSES = frozenset({EvidenceClass.PACKAGE_POSITION,
                                    EvidenceClass.SAMPLE_CONDITION})


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

# ----------------------------------------------------------------------
# registry.py
# ----------------------------------------------------------------------
"""The feature registry, and the check that makes it binding.

The engineering guide asks that every feature document its physical
hypothesis, its supporting paper, the exact GDS observable and unit, its
expected confounders, a discrimination test, a falsification condition, where
it is implemented, whether it is primary or exploratory, and what further
evidence would promote it from an association to an engineering rule.

A checklist nothing enforces is a wish. Every feature the pipeline reports is
matched against `references/feature_evidence_map.csv`, and one with no entry
is named in the run metadata -- so a feature can still be added quickly, but
it cannot quietly become a result with no stated hypothesis behind it.
"""
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "references" / \
    "feature_evidence_map.csv"

#: The traceability the guide asks for, as columns.
TRACE_COLUMNS = ("physical_hypothesis", "supporting_refs", "evidence_type",
                 "gds_observable", "unit", "expected_confounders",
                 "discrimination_test", "falsification", "implemented_in",
                 "hypothesis_tier", "promotes_to_rule_by")


@dataclass(frozen=True)
class RegistryEntry:
    family: str
    row: dict

    @property
    def missing_trace(self) -> list[str]:
        return [c for c in TRACE_COLUMNS if not (self.row.get(c) or "").strip()]


@lru_cache(maxsize=4)
def load(path: str | None = None) -> dict[str, RegistryEntry]:
    src = Path(path) if path else DEFAULT_PATH
    if not src.exists():
        return {}
    with open(src, newline="") as fh:
        return {r["feature_family"]: RegistryEntry(r["feature_family"], r)
                for r in csv.DictReader(fh)}


def lookup(name: str, registry: dict[str, RegistryEntry] | None = None
           ) -> RegistryEntry | None:
    """Find the family a reported feature belongs to.

    A family often emits several differently-named features -- routing in the
    bump frame yields an angle, an alignment and a diagonality -- so the names
    are listed explicitly in the ``emits`` column rather than guessed from the
    family name. Prefix matching then covers the derived forms a family
    generates mechanically: a gradient adds ``_dx``, a layer pair adds
    ``_M8_M7``.
    """
    reg = load() if registry is None else registry

    best, best_len = None, -1
    for entry in reg.values():
        candidates = [entry.family]
        emitted = (entry.row.get("emits") or "").strip()
        if emitted:
            candidates.extend(e for e in emitted.split(";") if e)
        for candidate in candidates:
            if name.startswith(candidate) and len(candidate) > best_len:
                best, best_len = entry, len(candidate)
    return best


def audit(feature_names) -> dict:
    """Which reported features have no registry entry, and which are thin."""
    reg = load()
    unregistered, thin = [], {}
    for name in sorted(set(feature_names)):
        entry = lookup(name, reg)
        if entry is None:
            unregistered.append(name)
        elif entry.missing_trace:
            thin.setdefault(entry.family, entry.missing_trace)
    return {
        "n_features": len(set(feature_names)),
        "unregistered": unregistered,
        "families_missing_traceability": thin,
        "complete": not unregistered and not thin,
    }


def matrix(path: str | None = None) -> "list[dict]":
    """The traceability matrix, one row per feature family."""
    return [e.row for e in load(path).values()]
