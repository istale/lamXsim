"""Multivariate baseline, spatial CV and ablation (spec sections 16-18)."""
import numpy as np
import pytest

from lamxsim import pipeline
from lamxsim.features.geometry import GeometryExtractor
from lamxsim.features.grid import build_grid
from lamxsim.labels.simulate import failures_from_driver, uniform_failures
from lamxsim.layout.reader import BBox, LayerSpec, LayoutReader
from lamxsim.layout.synth import validation_die
from lamxsim.stats import ablation
from lamxsim.stats.cv import (Fold, block_folds, buffered_block_folds,
                               grouped_folds, leakage_report)

M8 = LayerSpec("M8", 8, 0)


@pytest.fixture(scope="module")
def die(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("p6") / "die.gds")
    validation_die(path, die_um=3000.0, block_um=50.0, seed=7)
    reader = LayoutReader(path)
    grid = build_grid(reader.bbox(), 100.0)
    feats = GeometryExtractor(reader, line_end_w_max_um=6.0).extract(M8, grid)
    folds = buffered_block_folds(grid, block_um=300.0, n_folds=5, buffer_um=300.0)
    return path, grid, feats, folds


# ---- spatial separation -------------------------------------------

def test_random_and_unbuffered_block_splits_both_leak():
    """Blocking alone does not separate train from test at the boundary."""
    grid = build_grid(BBox(0, 0, 2000, 2000), 50.0)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(grid))
    random_fold = [Fold(train=perm[len(grid) // 5:], test=perm[:len(grid) // 5],
                        excluded=np.array([], dtype=int), label="random")]

    for folds in (random_fold, block_folds(grid, block_um=250.0)):
        report = leakage_report(folds, grid, min_separation_um=200.0)
        assert not report["all_pass"]
        assert min(f["min_train_test_separation_um"]
                   for f in report["folds"]) <= grid.stride_um


def test_buffer_enforces_separation_at_a_cost():
    grid = build_grid(BBox(0, 0, 2000, 2000), 50.0)
    folds = buffered_block_folds(grid, block_um=250.0, n_folds=5, buffer_um=250.0)
    report = leakage_report(folds, grid, min_separation_um=250.0)
    assert report["all_pass"]
    # The cost is real and must not be hidden: buffered cells leave training.
    assert all(f["n_excluded"] > 0 for f in report["folds"])


def test_grouped_folds_hold_out_whole_dies():
    groups = np.repeat(["D1", "D2", "D3"], 40)
    folds = grouped_folds(groups)
    assert len(folds) == 3
    for f in folds:
        assert len(set(groups[f.train]) & set(groups[f.test])) == 0


# ---- ablation -----------------------------------------------------

def test_ablation_requires_a_position_baseline(die):
    """A geometry AUC with nothing to compare against is not interpretable."""
    path, grid, feats, folds = die
    fs = failures_from_driver(feats["perimeter_density"], grid, n_failures=300,
                              strength=2.5, seed=1, position_sigma_um=5.0)
    res = pipeline.run(path, fs, layer=M8, scales_um=(100,), n_permutations=0,
                       line_end_w_max_um=6.0, seed=1, include_position=False)
    y = res.features["failure_present"].to_numpy(int)
    with pytest.raises(ValueError, match="position-only baseline"):
        ablation.run(res.features, y, folds)


def test_ablation_localises_where_the_information_enters(die):
    """Spec section 18: does sophisticated geometry beat metal density?

    Failures are driven by perimeter density. Metal density is not the driver
    but does correlate with it (r ~ 0.44 on this die), so the test is not that
    metal density is uninformative -- it is that the increment from adding
    perimeter dwarfs everything metal density carries on its own. A correlated
    non-driver showing a small effect is the expected behaviour, and the reason
    an ablation is read by the size of each step rather than by which steps
    reach significance.
    """
    path, grid, feats, folds = die
    fs = failures_from_driver(feats["perimeter_density"], grid, n_failures=300,
                              strength=2.5, seed=1, position_sigma_um=5.0)
    res = pipeline.run(path, fs, layer=M8, scales_um=(100,), n_permutations=0,
                       line_end_w_max_um=6.0, seed=1)
    y = res.features["failure_present"].to_numpy(int)
    out = ablation.run(res.features, y, folds, seed=3)
    by_name = {d["model"]: d for d in out.deltas}

    metal = by_name["A_metal_density"]["delta_auc"]
    perimeter = by_name["C_plus_perimeter"]["delta_auc"]

    assert by_name["C_plus_perimeter"]["adds_information"]
    assert by_name["C_plus_perimeter"]["ci_low"] > 0
    assert perimeter > 0.15
    assert perimeter > metal * 3, (
        f"perimeter step (+{perimeter:.3f}) must dominate what metal density "
        f"carries alone (+{metal:.3f}); if it does not, the die is no longer "
        "decoupling the two features"
    )
    later = max(by_name[m]["delta_auc"] for m in
                ("D_plus_termination_corners", "E_plus_orientation",
                 "F_plus_gradients", "G_plus_cross_layer"))
    assert later <= perimeter + 0.02, (
        "families added after perimeter should not improve on it here"
    )

    baseline = next(s for s in out.scores if s.name == "P_position_only")
    assert abs(baseline.roc_auc - 0.5) < 0.10, (
        "the validation die has no position effect; a position baseline well "
        "away from chance means the folds or the labels are wrong"
    )


def test_ablation_reports_nothing_under_the_null(die):
    """Every family must fail to add information when failures are uniform."""
    path, grid, feats, folds = die
    fs = uniform_failures(grid, n_failures=300, seed=42, position_sigma_um=5.0)
    res = pipeline.run(path, fs, layer=M8, scales_um=(100,), n_permutations=0,
                       line_end_w_max_um=6.0, seed=1)
    y = res.features["failure_present"].to_numpy(int)
    out = ablation.run(res.features, y, folds, seed=3)
    adding = [d["model"] for d in out.deltas if d["adds_information"]]
    assert not adding, f"models claimed information under the null: {adding}"


def test_delta_auc_interval_is_paired_and_spatially_blocked(die):
    """The interval must come from the same folds, resampled in blocks."""
    path, grid, feats, folds = die
    fs = failures_from_driver(feats["perimeter_density"], grid, n_failures=300,
                              strength=2.5, seed=1, position_sigma_um=5.0)
    res = pipeline.run(path, fs, layer=M8, scales_um=(100,), n_permutations=0,
                       line_end_w_max_um=6.0, seed=1)
    y = res.features["failure_present"].to_numpy(int)
    out = ablation.run(res.features, y, folds, seed=3)
    d = next(x for x in out.deltas if x["model"] == "C_plus_perimeter")
    assert d["ci_low"] < d["delta_auc"] < d["ci_high"]
    assert d["n_boot"] > 100
    # A per-observation bootstrap would be far narrower than this.
    assert d["ci_high"] - d["ci_low"] > 0.05
