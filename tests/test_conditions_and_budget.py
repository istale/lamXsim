"""Package conditions actually controlling, and the budget that makes a
correction reachable.

Each of these was declared and then not used: conditions written into a
manifest that nothing consumed, a cross-stratum summary quoting the naive
q-value the primary table had stopped using, and a permutation count too small
for the family it corrects.
"""
import numpy as np
import pandas as pd
import pytest

from collective.geometry import build_grid
from collective.labels import FailureSet
from collective.layout import BBox
from collective.statistics import block_permutation_test
from collective.statistics import (min_achievable_p, permutation_budget,
                                   required_permutations)
from collective.study import SampleConditions


def _failures(**cols):
    n = len(next(iter(cols.values()))) if cols else 4
    base = pd.DataFrame({
        "sample_id": [f"S{i}" for i in range(n)],
        "lot_id": "L1", "wafer_id": "W1",
        "die_x": [i // 2 for i in range(n)], "die_y": 0,
        "x_um": np.linspace(10, 100, n), "y_um": np.linspace(10, 100, n),
        "failure_type": "delamination", "confidence": 1.0,
        "position_sigma_um": np.nan})
    for k, v in cols.items():
        base[k] = v
    return FailureSet(table=base)


# ---- conditions are consumed, not merely recorded --------------------

def test_a_condition_cannot_hold_two_roles():
    sc = SampleConditions(fixed={"emc_thickness_um": 400},
                          stratified=("emc_thickness_um",))
    with pytest.raises(ValueError, match="exactly one role"):
        sc.validate()


def test_a_condition_declared_fixed_that_varies_is_refused():
    """The declaration is contradicted by the data in hand."""
    sc = SampleConditions(fixed={"emc_thickness_um": 400})
    fs = _failures(emc_thickness_um=[400, 400, 550, 550])
    with pytest.raises(ValueError, match="declared fixed"):
        sc.check_against(fs)


def test_a_stratified_condition_with_no_column_is_refused():
    """A condition that cannot be read cannot be controlled for."""
    sc = SampleConditions(stratified=("thermal_cycle_condition",))
    with pytest.raises(ValueError, match="no such column"):
        sc.check_against(_failures())


def test_a_fixed_condition_with_no_column_is_only_unverifiable():
    sc = SampleConditions(fixed={"emc_thickness_um": 400})
    notes = sc.check_against(_failures())
    assert notes and "nothing confirms it" in notes[0]


def test_a_covariate_varying_within_a_die_is_reported():
    sc = SampleConditions(covariate=("underfill_cte_ppm_k",))
    fs = _failures(underfill_cte_ppm_k=[30.0, 45.0, 30.0, 30.0])
    notes = sc.check_against(fs)
    assert any("varies within" in n for n in notes)


def test_declared_covariates_reach_the_baseline_model():
    """Recording a condition and leaving it out of the model lets a geometry
    feature absorb its effect, which is what the declaration prevents."""
    from collective.statistics import POSITION_FAMILY, select_columns

    columns = ["metal_density|M8", "condition_emc_thickness_um|-",
               "distance_to_die_edge|-"]
    chosen = select_columns(columns, POSITION_FAMILY)
    assert "condition_emc_thickness_um|-" in chosen
    assert "metal_density|M8" not in chosen


def test_sample_conditions_are_their_own_evidence_class():
    """Not a position, not geometry, and not something GDS contains."""
    from collective.foundation import EvidenceClass, POSITION_MODEL_CLASSES

    assert EvidenceClass.SAMPLE_CONDITION in POSITION_MODEL_CLASSES
    assert EvidenceClass.SAMPLE_CONDITION is not EvidenceClass.GDS_GEOMETRY


# ---- the permutation budget -----------------------------------------

def test_a_permutation_test_has_a_resolution_floor():
    assert min_achievable_p(999) == pytest.approx(0.001)
    assert min_achievable_p(9999) == pytest.approx(0.0001)


def test_the_default_permutation_count_cannot_resolve_a_real_family():
    """240 corrected tests is what an ordinary two-layer run produces."""
    small = permutation_budget(20, 999)
    real = permutation_budget(240, 999)
    assert small["sufficient"]
    assert not real["sufficient"]
    assert real["best_achievable_q_for_a_lone_result"] == pytest.approx(0.24)
    assert real["permutations_needed_for_alpha"] > 4000


def test_enough_permutations_restores_the_resolution():
    assert permutation_budget(240, 9999)["sufficient"]
    assert required_permutations(240, alpha=0.05) == 4799


# ---- block exchange --------------------------------------------------

def test_only_same_sized_blocks_are_exchanged():
    """Slicing a concatenated pool splits one block across two targets.

    That merges parts of different blocks, which is no longer a block
    permutation -- and it breaks exactly at the die edge and the ROI
    boundary, where blocks are ragged.
    """
    grid = build_grid(BBox(0, 0, 700, 700), 100.0)      # 7x7, blocks of 3
    rng = np.random.default_rng(0)
    values = rng.normal(size=len(grid))
    labels = (rng.random(len(grid)) < 0.3).astype(int)

    result = block_permutation_test(values, labels, grid, n_permutations=99,
                                    block_cells=3, seed=0)
    assert result.n_blocks == 9
    assert result.n_blocks_not_exchangeable == 1     # the 1x1 corner block


def test_the_case_count_is_preserved_exactly():
    """The old truncate-and-recycle could change how many cases existed."""
    grid = build_grid(BBox(0, 0, 700, 700), 100.0)
    rng = np.random.default_rng(0)
    values = rng.normal(size=len(grid))
    labels = (rng.random(len(grid)) < 0.3).astype(int)

    seen = []
    block_permutation_test(
        values, labels, grid,
        statistic=lambda v, l: (seen.append(int(l.sum())), 0.5)[1],
        n_permutations=30, block_cells=3, seed=1)
    assert len(set(seen)) == 1
    assert seen[0] == int(labels.sum())


# ---- supported findings vs the hypothesis set ------------------------

def _row(feature, **kw):
    base = dict(feature=feature, layer="M8", scale_um=100.0,
                evidence_class="GDS_GEOMETRY", hypothesis_tier="tier1",
                scale_trustworthy=True, roc_auc=0.78, auc_ci_low=0.71,
                auc_ci_high=0.84, effect_size=0.56, fdr_q_value=0.001,
                spatial_q_value=0.01, n_case=100, n_control=100,
                effective_n=50.0, enrichment_top_10pct=2.0)
    base.update(kw)
    return base


def test_the_hypothesis_set_is_not_the_findings():
    """primary contains rows at q = 1 by construction, and is read as results."""
    from collective import exposure as report

    df = pd.DataFrame([
        _row("perimeter_density"),
        _row("metal_density", spatial_q_value=1.0, roc_auc=0.51,
             auc_ci_low=0.45, auc_ci_high=0.57, effect_size=0.02),
        _row("via_density", spatial_q_value=0.03, auc_ci_low=0.49,
             auc_ci_high=0.88, effect_size=0.40),
    ])
    parts = report.partition(df)
    assert set(parts["primary"].feature) == {"perimeter_density",
                                             "metal_density", "via_density"}
    assert list(parts["supported"].feature) == ["perimeter_density"]


def test_the_console_shows_findings_and_says_so_when_there_are_none():
    from collective import exposure as report

    df = pd.DataFrame([_row("metal_density", spatial_q_value=1.0,
                            auc_ci_low=0.45, auc_ci_high=0.57)])
    text = report.format_primary(df)
    assert "no supported finding" in text
    assert "1 hypotheses were tested" in text


def test_the_summary_distinguishes_the_two(tmp_path):
    from collective import exposure as report

    report.write_reports(pd.DataFrame([_row("perimeter_density")]), tmp_path)
    text = (tmp_path / "reports" / "README.md").read_text()
    assert "**the findings**" in text
    assert "not a list of findings" in text


def test_a_p_pinned_at_the_floor_is_flagged_as_a_bound():
    """Ties at 1/(n+1) give a low q without resolving anything.

    A family too large for the permutation count still produces small
    q-values, because many tests reach the floor together and BH ranks them
    against each other. The q is then an upper bound the permutation count
    imposed, not a value the data produced, and a reader has no way to tell
    from the number alone.
    """
    from collective import workflow as pipeline
    from collective.geometry import GeometryExtractor
    from collective.labels import failures_from_driver
    from collective.layout import LayerSpec, LayoutReader
    from collective.layout import validation_die
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "die.gds")
        validation_die(path, die_um=1200.0, block_um=50.0, seed=7)
        reader = LayoutReader(path)
        grid = build_grid(reader.bbox(), 100.0)
        m8 = LayerSpec("M8", 8, 0)
        feats = GeometryExtractor(reader, line_rules={"M8": (0.5, 4.0)}
                                  ).extract(m8, grid)
        fs = failures_from_driver(feats["perimeter_density"], grid,
                                  n_failures=60, strength=2.5, seed=1,
                                  position_sigma_um=3.0)
        res = pipeline.run(path, fs, layer=m8, scales_um=(100,),
                           n_permutations=99, line_rules={"M8": (0.5, 4.0)},
                           seed=1)

    a = res.associations
    assert "spatial_p_at_floor" in a.columns
    pinned = a[a.spatial_p_at_floor]
    assert len(pinned) > 0
    # Every pinned row shares the same p, so their q values are ties rather
    # than an ordering.
    assert pinned["spatial_p_value"].nunique() == 1
    assert res.metadata["permutation_budget"]["n_at_resolution_floor"] == len(pinned)
