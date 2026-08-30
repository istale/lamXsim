"""Result presentation, separated by what each row is allowed to claim.

`best_features.csv` sorted by effect size is not an engineering
recommendation list. Ranking every row together puts an exploratory
descriptor measured at a scale the registration cannot support above a
literature-backed feature measured at one it can, and the file gives the
reader no way to tell.

Rows are therefore partitioned before they are ranked, along the three axes
that decide what a row may be used for:

* **evidence class** -- a PACKAGE_POSITION row is a confounder being
  controlled, not a finding.
* **hypothesis tier** -- exploratory families carry no FDR-corrected
  significance claim, because correcting them alongside the primary ones is
  what makes the primary ones unreachable.
* **scale trustworthiness** -- a scale below roughly three times the
  positional uncertainty measures registration noise.

Only rows that clear all three appear in the primary table.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PRIMARY_TIERS = ("tier1",)
CONFOUNDER_TIERS = ("tier1_confounder",)

#: Fewest cells a class may hold and still support an effect estimate.
#: At a coarse scale nearly every window can contain a failure, leaving a
#: handful of controls; the AUC then saturates at 1.0 with an interval of zero
#: width, and ranking by effect size alone puts that degenerate row above a
#: real, well-powered one.
MIN_CLASS_SIZE = 10


def _reasons(row) -> list[str]:
    out = []
    if row.get("evidence_class") != "GDS_GEOMETRY":
        out.append(f"evidence class {row.get('evidence_class')}")
    if row.get("hypothesis_tier") not in PRIMARY_TIERS:
        out.append(f"tier {row.get('hypothesis_tier')}")
    if row.get("scale_trustworthy") is False:
        out.append("scale below the registration floor")
    if not np.isfinite(row.get("fdr_q_value", np.nan)):
        out.append("no FDR correction applied")
    return out


def partition(associations: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split an association table into primary, confounder and exploratory."""
    if associations.empty:
        empty = associations.copy()
        return {k: empty.copy() for k in
                ("primary", "confounders", "exploratory",
                 "unsupported_scale", "underpowered")}

    df = associations.copy()
    df["abs_effect"] = df["effect_size"].abs()
    trustworthy = df.get("scale_trustworthy")
    ok_scale = (pd.Series(True, index=df.index) if trustworthy is None
                else trustworthy.fillna(True).astype(bool))

    is_geometry = df["evidence_class"] == "GDS_GEOMETRY"
    is_primary_tier = df["hypothesis_tier"].isin(PRIMARY_TIERS)
    powered = ((df.get("n_case", 0) >= MIN_CLASS_SIZE)
               & (df.get("n_control", 0) >= MIN_CLASS_SIZE))

    primary = df[is_geometry & is_primary_tier & ok_scale & powered]
    confounders = df[df["hypothesis_tier"].isin(CONFOUNDER_TIERS)]
    unsupported = df[~ok_scale]
    underpowered = df[is_geometry & is_primary_tier & ok_scale & ~powered]
    exploratory = df[is_geometry & ~is_primary_tier & ok_scale & powered]

    # Ranked by the effect the interval actually guarantees, so a wide
    # interval cannot outrank a tight one. An interval that straddles 0.5
    # guarantees nothing -- the data does not exclude "no effect" -- and
    # scores zero. Taking the nearer endpoint unconditionally would instead
    # reward an interval that is wide on both sides: on a real three-die run
    # that put a feature at AUC 0.512 with q = 0.94 above the driver at
    # AUC 0.698 with q = 0.0008.
    if "auc_ci_low" in df.columns and "auc_ci_high" in df.columns:
        for t in (primary, confounders, exploratory, unsupported, underpowered):
            lo, hi = t["auc_ci_low"], t["auc_ci_high"]
            guaranteed = np.where(lo > 0.5, lo - 0.5,
                                  np.where(hi < 0.5, 0.5 - hi, 0.0))
            t["conservative_effect"] = np.where(
                np.isfinite(lo) & np.isfinite(hi), guaranteed, 0.0)
        order = ["conservative_effect", "abs_effect"]
    else:
        order = ["abs_effect"]

    return {
        "primary": primary.sort_values(order, ascending=False),
        "confounders": confounders.sort_values(order, ascending=False),
        "exploratory": exploratory.sort_values(order, ascending=False),
        "unsupported_scale": unsupported.sort_values(order, ascending=False),
        "underpowered": underpowered.sort_values(order, ascending=False),
    }


#: Columns a reader needs to judge a row, in the order they should be read.
SUMMARY_COLUMNS = ("feature", "layer", "scale_um", "roc_auc", "auc_ci_low",
                   "auc_ci_high", "effect_size", "fdr_q_value", "n_case",
                   "n_control", "effective_n", "enrichment_top_10pct")


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in SUMMARY_COLUMNS if c in df.columns]
    return df[cols]


def write(associations: pd.DataFrame, outdir: str | Path, *,
          metadata: dict | None = None) -> dict[str, str]:
    """Write the partitioned tables plus a machine-readable summary."""
    out = Path(outdir) / "reports"
    out.mkdir(parents=True, exist_ok=True)
    parts = partition(associations)
    paths = {}
    for name, table in parts.items():
        p = out / f"{name}.csv"
        summary_table(table).to_csv(p, index=False)
        paths[name] = str(p)

    counts = {k: int(len(v)) for k, v in parts.items()}
    lines = [
        "# Results by what each row may claim",
        "",
        f"- primary            {counts['primary']:5d}  literature-backed geometry, "
        "FDR-corrected, at a scale the registration supports",
        f"- confounders        {counts['confounders']:5d}  package position; "
        "controlled for, never a finding",
        f"- exploratory        {counts['exploratory']:5d}  no direct delamination "
        "evidence; effect size only, no significance claim",
        f"- unsupported_scale  {counts['unsupported_scale']:5d}  below the "
        "registration floor; excluded from every conclusion",
        f"- underpowered       {counts['underpowered']:5d}  fewer than "
        f"{MIN_CLASS_SIZE} cells in one class, so the effect estimate is "
        "degenerate however large it looks",
        "",
    ]
    if metadata:
        for note in metadata.get("uncontrolled_confounding", []):
            lines.append(f"UNCONTROLLED: {note}")
        if metadata.get("uncontrolled_confounding"):
            lines.append("")
    lines.append(
        "This is a statistical association within one study population. It is "
        "not a causal claim, not a failure probability, and not a design rule.")
    p = out / "README.md"
    p.write_text("\n".join(lines) + "\n")
    paths["summary"] = str(p)
    return paths


def format_primary(associations: pd.DataFrame, limit: int = 10) -> str:
    parts = partition(associations)
    p = summary_table(parts["primary"]).head(limit)
    if p.empty:
        return ("no primary result: every row was a confounder, an exploratory "
                "family, or measured at a scale the registration cannot support")
    return p.to_string(index=False, float_format=lambda v: f"{v:.4f}")
