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
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "references" / \
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
