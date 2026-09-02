"""Multi-die observation structure and failure-mode separation.

Two things the failure schema has always demanded and the analysis never used:
lot/wafer/die identity, required at import for the held-out validation of spec
section 17, and failure_type, required but never consulted before pooling.
Requiring data and then ignoring it is worse than not requiring it, because a
reader reasonably assumes it was used.
"""
import numpy as np
import pandas as pd
import pytest

from collective import workflow as pipeline
from collective.geometry import GeometryExtractor
from collective.geometry import build_grid
from collective.labels import FailureSet, map_to_grid_per_die
from collective.labels import failures_from_driver
from collective.layout import BBox, LayerSpec, LayoutReader
from collective.layout import validation_die
from collective.statistics import grouped_folds

M8 = LayerSpec("M8", 8, 0)


def _multi_die(n_dies, per_die=30, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for d in range(n_dies):
        pts = rng.uniform(0, 1000, (per_die, 2))
        frames.append(pd.DataFrame({
            "sample_id": [f"D{d}S{i}" for i in range(per_die)],
            "lot_id": "L1", "wafer_id": "W1", "die_x": d, "die_y": 0,
            "x_um": pts[:, 0], "y_um": pts[:, 1],
            "failure_type": "delamination", "confidence": 1.0,
            "position_sigma_um": np.nan}))
    return FailureSet(table=pd.concat(frames, ignore_index=True))


# ---- the pooling defect --------------------------------------------

def test_pooling_dies_onto_one_grid_inflates_prevalence():
    """Why the observation unit had to become (cell, die).

    Collapsing several dies to "did anything ever fail here" is a different
    question, not a rescaled one: prevalence climbs towards 1 with the number
    of dies, and a cell that failed on one die of ten becomes
    indistinguishable from one that failed on all ten.
    """
    grid = build_grid(BBox(0, 0, 1000, 1000), 100.0)
    prevalences = []
    for n in (1, 3, 10):
        per_die = map_to_grid_per_die(_multi_die(n), grid)
        pooled = np.max([v["failure_present"] for v in per_die.values()], axis=0)
        prevalences.append(pooled.mean())
    assert prevalences[0] < prevalences[1] < prevalences[2]
    assert prevalences[2] > 0.9

    # The per-observation prevalence stays where it belongs.
    per_die = map_to_grid_per_die(_multi_die(10), grid)
    rates = [v["failure_present"].mean() for v in per_die.values()]
    assert np.mean(rates) == pytest.approx(prevalences[0], abs=0.10)


def test_per_die_mapping_keeps_each_die_separate():
    grid = build_grid(BBox(0, 0, 1000, 1000), 100.0)
    per_die = map_to_grid_per_die(_multi_die(4), grid)
    assert len(per_die) == 4
    assert all(v["failure_present"].sum() > 0 for v in per_die.values())


@pytest.fixture(scope="module")
def die_path(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("md") / "die.gds")
    validation_die(path, die_um=1000.0, block_um=50.0, seed=7)
    return path


def test_pipeline_observations_are_cell_by_die(die_path):
    reader = LayoutReader(die_path)
    grid = build_grid(reader.bbox(), 100.0)
    feats = GeometryExtractor(reader, line_end_w_max_um=4.0).extract(M8, grid)

    frames = []
    for d in range(3):
        fs = failures_from_driver(feats["perimeter_density"], grid,
                                  n_failures=25, strength=2.5, seed=10 + d,
                                  position_sigma_um=5.0)
        t = fs.table.copy()
        t["lot_id"], t["wafer_id"], t["die_x"], t["die_y"] = "L1", "W1", d, 0
        t["sample_id"] = [f"D{d}_{s}" for s in t.sample_id]
        frames.append(t)
    multi = FailureSet(table=pd.concat(frames, ignore_index=True))

    res = pipeline.run(die_path, multi, layer=M8, scales_um=(100,),
                       n_permutations=0, line_end_w_max_um=4.0, seed=1)
    cov = res.metadata["coverage_by_scale"][100.0]
    assert cov["n_dies"] == 3
    assert cov["n_observations"] == cov["n_cells"] * 3
    assert res.metadata["n_dies"] == 3
    assert set(res.features["die_key"].unique()) == {"L1|W1|0|0", "L1|W1|1|0",
                                                     "L1|W1|2|0"}
    assert (res.associations["n_dies"] == 3).all()


def test_single_die_result_is_labelled_a_local_diagnostic(die_path):
    reader = LayoutReader(die_path)
    grid = build_grid(reader.bbox(), 100.0)
    feats = GeometryExtractor(reader, line_end_w_max_um=4.0).extract(M8, grid)
    fs = failures_from_driver(feats["perimeter_density"], grid, n_failures=40,
                              strength=2.5, seed=1, position_sigma_um=5.0)
    res = pipeline.run(die_path, fs, layer=M8, scales_um=(100,),
                       n_permutations=0, line_end_w_max_um=4.0, seed=1)
    assert any("single die" in n and "local diagnostic" in n
               for n in res.metadata["uncontrolled_confounding"])


def test_held_out_die_folds_never_share_a_die():
    keys = np.repeat(["L1|W1|0|0", "L1|W1|1|0", "L1|W1|2|0"], 50)
    folds = grouped_folds(keys)
    assert len(folds) == 3
    for f in folds:
        assert not set(keys[f.train]) & set(keys[f.test])


# ---- failure modes -------------------------------------------------

def _modes_table(**cols):
    base = dict(sample_id=["S0", "S1"], lot_id="L1", wafer_id="W1",
                die_x=0, die_y=0, x_um=[10.0, 20.0], y_um=[10.0, 20.0],
                failure_type=["delamination", "delamination"],
                confidence=1.0, position_sigma_um=np.nan)
    base.update(cols)
    return FailureSet(table=pd.DataFrame(base))


def test_mixed_failure_types_are_refused_by_default():
    """Whether two modes share a mechanism is a physics judgement."""
    fs = _modes_table(failure_type=["delamination", "channel_crack"])
    with pytest.raises(ValueError, match="mixes populations"):
        fs.assert_single_mode()


def test_mixed_failed_layers_are_refused_by_default():
    """Li et al. (2023) found different interfaces carrying different ERR."""
    fs = _modes_table(failed_layer=["M8", "M4"])
    with pytest.raises(ValueError, match="mixes populations"):
        fs.assert_single_mode()


def test_pooling_can_be_asserted_and_is_then_recorded():
    fs = _modes_table(failure_type=["delamination", "channel_crack"])
    notes = fs.assert_single_mode(allow_pooling=True)
    assert notes and "asserted by the operator" in notes[0]


def test_a_single_mode_needs_no_assertion():
    assert _modes_table().assert_single_mode() == []


def test_pipeline_refuses_mixed_modes_and_records_the_override(die_path):
    reader = LayoutReader(die_path)
    grid = build_grid(reader.bbox(), 100.0)
    feats = GeometryExtractor(reader, line_end_w_max_um=4.0).extract(M8, grid)
    fs = failures_from_driver(feats["perimeter_density"], grid, n_failures=60,
                              strength=2.5, seed=1, position_sigma_um=5.0)
    t = fs.table.copy()
    t["failure_type"] = ["delamination"] * 30 + ["channel_crack"] * (len(t) - 30)
    mixed = FailureSet(table=t)

    with pytest.raises(ValueError, match="mixes populations"):
        pipeline.run(die_path, mixed, layer=M8, scales_um=(100,),
                     n_permutations=0, line_end_w_max_um=4.0)

    res = pipeline.run(die_path, mixed, layer=M8, scales_um=(100,),
                       n_permutations=0, line_end_w_max_um=4.0,
                       allow_pooling_modes=True)
    assert any("asserted by the operator" in n
               for n in res.metadata["uncontrolled_confounding"])
    assert "failure_type" in res.metadata["failure_modes"]


def test_absent_failed_layer_is_recorded_as_an_unseparated_population(tmp_path):
    from collective.labels import load_failures
    p = tmp_path / "f.csv"
    pd.DataFrame({"sample_id": ["S0"], "lot_id": "L1", "wafer_id": "W1",
                  "die_x": 0, "die_y": 0, "x_um": [10.0], "y_um": [10.0],
                  "failure_type": ["delamination"]}).to_csv(p, index=False)
    fs = load_failures(p)
    joined = " ".join(fs.notes)
    assert "failed_layer absent" in joined
    assert "failed_interface absent" in joined


# ---- one layout for every die --------------------------------------

def _revision_set(revisions):
    table = pd.DataFrame({
        "sample_id": ["A", "B"], "lot_id": "L1", "wafer_id": "W1",
        "die_x": [0, 1], "die_y": 0, "x_um": [10.0, 20.0],
        "y_um": [10.0, 20.0], "failure_type": "delamination",
        "confidence": 1.0, "position_sigma_um": np.nan})
    if revisions is not None:
        table["layout_revision"] = revisions
    return FailureSet(table=table)


def test_failures_spanning_layout_revisions_are_refused():
    """Features come from one GDS; a failure from another revision would be
    scored against geometry that was never on its silicon."""
    with pytest.raises(ValueError, match="spans layout revisions"):
        _revision_set(["revA", "revB"]).assert_single_layout_revision(None)


def test_a_revision_that_disagrees_with_the_manifest_is_refused():
    with pytest.raises(ValueError, match="manifest declares"):
        _revision_set(["revA", "revA"]).assert_single_layout_revision("revB")


def test_a_matching_revision_needs_no_note():
    assert _revision_set(["revB", "revB"]).assert_single_layout_revision("revB") == []


def test_an_absent_revision_column_records_the_assumption():
    notes = _revision_set(None).assert_single_layout_revision(None)
    assert notes and "unverified" in notes[0]


def test_the_run_records_the_layout_it_analysed(die_path):
    reader = LayoutReader(die_path)
    grid = build_grid(reader.bbox(), 100.0)
    feats = GeometryExtractor(reader, line_end_w_max_um=4.0).extract(M8, grid)
    fs = failures_from_driver(feats["perimeter_density"], grid, n_failures=40,
                              strength=2.5, seed=1, position_sigma_um=5.0)
    res = pipeline.run(die_path, fs, layer=M8, scales_um=(100,),
                       n_permutations=0, line_end_w_max_um=4.0, seed=1)
    digest = res.metadata["gds_sha256"]
    assert len(digest) == 64 and int(digest, 16) >= 0
    assert any("no layout_revision" in n
               for n in res.metadata["uncontrolled_confounding"])


def test_manifest_records_an_undeclared_layout_revision(tmp_path):
    from collective.study import StudyManifest

    p = tmp_path / "m.yaml"
    p.write_text(
        "layout:\n  metal_layers:\n    - {name: M8, layer: 8, datatype: 0}\n")
    assert any("no layout_revision" in g for g in StudyManifest.load(p).gaps)
