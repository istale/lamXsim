"""Association, correction, resampling and validation.

Folded from ``tests/test_statistical_pipeline.py``, ``tests/test_spatial_significance.py``, ``tests/test_phase6_baseline.py``, ``tests/test_stratification_and_registry.py``, ``tests/test_multi_die_and_modes.py``.
"""
from collective import foundation as registry
from collective import statistics as ablation
from collective import workflow as pipeline
from collective.exposure import partition
from collective.geometry import GeometryExtractor
from collective.geometry import build_grid
from collective.labels import FailureSet
from collective.labels import failures_from_driver
from collective.labels import map_to_grid_per_die
from collective.labels import stratify
from collective.labels import uniform_failures
from collective.layout import BBox
from collective.layout import LayerSpec
from collective.layout import LayoutReader
from collective.layout import validation_die
from collective.statistics import Fold
from collective.statistics import block_folds
from collective.statistics import buffered_block_folds
from collective.statistics import grouped_folds
from collective.statistics import leakage_report
import numpy as np
import pandas as pd
import pytest


# ----------------------------------------------------------------------
# test_statistical_pipeline.py
# ----------------------------------------------------------------------
"""Validation of the statistical pipeline itself, on ground truth we control.

These are the tests that decide whether any association the platform reports
can be believed. They run on a die where metal density and perimeter density
vary independently, so a pipeline that has collapsed into a density detector
fails them.
"""
M8 = LayerSpec("M8", 8, 0)
SCALES = (50, 100, 250)


@pytest.fixture(scope="module")
def die_statistical(tmp_path_factory):
    path = tmp_path_factory.mktemp("die") / "valdie.gds"
    validation_die(str(path), die_um=2000.0, block_um=50.0, seed=7)
    reader = LayoutReader(str(path))
    grid = build_grid(reader.bbox(), 100.0)
    feats = GeometryExtractor(reader).extract(M8, grid)
    return str(path), grid, feats


def test_die_decouples_density_from_perimeter(die_statistical):
    """The validation die is only meaningful if the two features are separable."""
    _, _, f = die_statistical
    r = np.corrcoef(f["metal_density"], f["perimeter_density"])[0, 1]
    assert abs(r) < 0.6, f"features too collinear to test discrimination (r={r:.2f})"


def test_recovers_the_true_driver(die_statistical):
    """Failures driven by perimeter must rank perimeter above metal density."""
    path, grid, f = die_statistical
    fs = failures_from_driver(f["perimeter_density"], grid, n_failures=150,
                              strength=2.5, seed=1, position_sigma_um=5.0)
    res = pipeline.run(path, fs, layer=M8, scales_um=SCALES,
                       n_permutations=0, seed=3)
    a = res.associations
    perim = a[a.feature == "perimeter_density"].roc_auc.max()
    metal = a[a.feature == "metal_density"].roc_auc.max()
    assert perim > metal + 0.10, (
        f"perimeter AUC {perim:.3f} did not clearly beat metal {metal:.3f}; "
        "the pipeline may be responding to density rather than the true driver"
    )


def test_negative_control_reports_nothing(die_statistical):
    """Spatially uniform failures must produce no significant association."""
    path, grid, _ = die_statistical
    fs = uniform_failures(grid, n_failures=150, seed=42, position_sigma_um=5.0)
    res = pipeline.run(path, fs, layer=M8, scales_um=SCALES,
                       n_permutations=0, seed=5)
    q = res.associations.fdr_q_value.dropna()
    assert (q < 0.05).sum() == 0, (
        f"{(q < 0.05).sum()} spurious findings under the null: the pipeline "
        "manufactures associations and no result from it can be trusted"
    )


def test_block_permutation_rejects_spurious_position_association(die_statistical):
    """Position features are confounded by autocorrelation, not by an effect.

    The validation die has no package-position effect built in, yet a test that
    treats grid cells as independent flags position features as significant.
    The spatial null model is what removes them.
    """
    path, grid, f = die_statistical
    fs = failures_from_driver(f["perimeter_density"], grid, n_failures=150,
                              strength=2.5, seed=1, position_sigma_um=5.0)
    res = pipeline.run(path, fs, layer=M8, scales_um=(50, 100),
                       n_permutations=299, seed=3)

    a = res.associations.set_index(["feature", "scale_um"])
    p = res.permutations.set_index(["feature", "scale_um"])
    pos = a[a.evidence_class == "PACKAGE_POSITION"].index
    geo = a[a.evidence_class == "GDS_GEOMETRY"].index

    naive_pos = (a.loc[pos, "p_value"] < 0.05).sum()
    perm_pos = (p.loc[pos, "p_value"] < 0.05).sum()

    assert naive_pos > 0, "test is vacuous unless the naive test is fooled"
    assert perm_pos == 0, (
        f"block permutation still calls {perm_pos} position associations "
        "significant on a die with no position effect"
    )
    # The driver itself must survive: a null model that also erases real
    # signal would be useless, not conservative. Only perimeter_density drives
    # the simulated failures here, so only it is required to survive.
    driver = p.loc[[i for i in geo if i[0] == "perimeter_density"], "p_value"]
    assert (driver < 0.05).all(), (
        "block permutation discarded the real perimeter-driven signal"
    )


def test_effective_n_is_far_below_cell_count(die_statistical):
    """Grid cells must not be reported as if they were independent samples."""
    path, grid, f = die_statistical
    fs = failures_from_driver(f["perimeter_density"], grid, n_failures=150,
                              strength=2.5, seed=1, position_sigma_um=5.0)
    res = pipeline.run(path, fs, layer=M8, scales_um=(50,), n_permutations=0, seed=3)
    row = res.associations[res.associations.feature == "perimeter_density"].iloc[0]
    assert row.effective_n < row.n_cells * 0.5


def test_registration_gates_small_scales(die_statistical):
    """Scales below ~3x the positional uncertainty are marked untrustworthy."""
    path, grid, f = die_statistical
    fs = failures_from_driver(f["perimeter_density"], grid, n_failures=80,
                              strength=2.0, seed=2, position_sigma_um=40.0)
    res = pipeline.run(path, fs, layer=M8, scales_um=(25, 50, 100, 250),
                       n_permutations=0, seed=3)
    a = res.associations
    assert set(a[a.scale_um <= 100].scale_trustworthy) == {False}
    assert set(a[a.scale_um == 250].scale_trustworthy) == {True}

# ----------------------------------------------------------------------
# test_spatial_significance.py
# ----------------------------------------------------------------------
"""What a primary claim is allowed to rest on.

The pipeline computed a within-die block permutation and then corrected the
Mann-Whitney p-value instead, leaving the spatial result in a side table. On a
die with no package-position effect the naive test called 11 of 12 position
associations significant where the permutation called none -- so the report
was quoting the test the repository's own README shows to be wrong.
"""
RULES = {"M8": (0.5, 4.0)}


@pytest.fixture(scope="module")
def die_spatial(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("sig") / "die.gds")
    validation_die(path, die_um=1500.0, block_um=50.0, seed=7)
    reader = LayoutReader(path)
    grid = build_grid(reader.bbox(), 100.0)
    feats = GeometryExtractor(reader, line_rules=RULES).extract(M8, grid)
    fs = failures_from_driver(feats["perimeter_density"], grid, n_failures=80,
                              strength=2.5, seed=1, position_sigma_um=3.0)
    return path, fs


def test_without_a_permutation_there_is_no_primary_evidence(die_spatial):
    """Significance must not be reachable by skipping the spatial null."""
    path, fs = die_spatial
    res = pipeline.run(path, fs, layer=M8, scales_um=(100,), n_permutations=0,
                       line_rules=RULES, seed=1)
    assert res.associations["spatial_q_value"].isna().all()
    parts = partition(res.associations)
    assert len(parts["primary"]) == 0
    assert len(parts["not_spatially_corrected"]) > 0
    assert any("no spatial permutation was run" in n
               for n in res.metadata["uncontrolled_confounding"])


def test_the_spatial_q_value_is_what_primary_rests_on(die_spatial):
    path, fs = die_spatial
    res = pipeline.run(path, fs, layer=M8, scales_um=(100,),
                       n_permutations=299, line_rules=RULES, seed=1)
    a = res.associations
    assert a["spatial_q_value"].notna().any()
    parts = partition(a)
    assert len(parts["primary"]) > 0
    assert parts["primary"]["spatial_q_value"].notna().all()


def test_the_naive_q_value_survives_as_a_diagnostic(die_spatial):
    """Kept for the contrast it provides, not as the basis of a finding."""
    path, fs = die_spatial
    res = pipeline.run(path, fs, layer=M8, scales_um=(100,),
                       n_permutations=299, line_rules=RULES, seed=1)
    a = res.associations[res.associations.spatial_q_value.notna()]
    assert a["fdr_q_value"].notna().any()
    # The spatial correction is the weaker claim, which is the point of it.
    driver = a[a.feature == "perimeter_density"].iloc[0]
    assert driver["spatial_q_value"] >= driver["fdr_q_value"]


def test_a_feature_without_a_registry_entry_cannot_be_primary():
    """Auditing a gap and printing the row anyway is not enforcement."""
    def row(feature, **kw):
        base = dict(feature=feature, layer="M8", scale_um=100.0,
                    evidence_class="GDS_GEOMETRY", hypothesis_tier="tier1",
                    scale_trustworthy=True, roc_auc=0.78, auc_ci_low=0.71,
                    auc_ci_high=0.84, effect_size=0.56, fdr_q_value=0.001,
                    spatial_q_value=0.01, n_case=100, n_control=100,
                    effective_n=50.0, enrichment_top_10pct=2.0)
        base.update(kw)
        return base

    parts = partition(pd.DataFrame([
        row("perimeter_density"),
        row("invented_feature"),
        row("metal_density", spatial_q_value=np.nan)]))
    assert list(parts["primary"].feature) == ["perimeter_density"]
    assert list(parts["not_traceable"].feature) == ["invented_feature"]
    assert list(parts["not_spatially_corrected"].feature) == ["metal_density"]


def test_the_summary_names_both_q_values(tmp_path):
    from collective import exposure as report

    df = pd.DataFrame([dict(
        feature="perimeter_density", layer="M8", scale_um=100.0,
        evidence_class="GDS_GEOMETRY", hypothesis_tier="tier1",
        scale_trustworthy=True, roc_auc=0.78, auc_ci_low=0.71,
        auc_ci_high=0.84, effect_size=0.56, fdr_q_value=0.001,
        spatial_q_value=0.01, n_case=100, n_control=100, effective_n=50.0,
        enrichment_top_10pct=2.0)])
    report.write_reports(df, tmp_path)
    text = (tmp_path / "reports" / "README.md").read_text()
    assert "spatial_q_value" in text and "Mann-Whitney" in text
    assert "not_spatially_corrected" in text
    assert "not_traceable" in text


# ---- package and process conditions --------------------------------

def test_undeclared_package_conditions_are_recorded_as_a_gap():
    from collective.study import StudyManifest

    m = StudyManifest.load("config/study_manifest.yaml")
    assert m.sample_conditions.undeclared()
    assert any("package/process condition" in g for g in m.gaps)


def test_each_condition_is_fixed_stratified_covariate_or_unknown():
    from collective.study import PACKAGE_PROCESS_CONDITIONS, SampleConditions

    sc = SampleConditions(fixed={"emc_thickness_um": 400},
                          stratified=("thermal_cycle_condition",),
                          covariate=("underfill_cte_ppm_k",))
    assert sc.status("emc_thickness_um") == "fixed"
    assert sc.status("thermal_cycle_condition") == "stratified"
    assert sc.status("underfill_cte_ppm_k") == "covariate"
    assert sc.status("reflow_profile") == "unknown"
    assert set(sc.report()["by_condition"]) == set(PACKAGE_PROCESS_CONDITIONS)


# ---- reproducible install ------------------------------------------

def test_every_imported_third_party_package_is_declared():
    """A clean install failed at test collection because sklearn was missing."""
    import tomllib

    with open("pyproject.toml", "rb") as fh:
        declared = tomllib.load(fh)["project"]["dependencies"]
    names = {d.split(">")[0].split("<")[0].split("=")[0].strip().lower()
             for d in declared}
    for required in ("numpy", "scipy", "pandas", "klayout", "scikit-learn",
                     "pyyaml", "pyarrow"):
        assert required in names, f"{required} is imported but not declared"


def test_numpy_two_removed_the_ndarray_ptp_method():
    """Every synthetic die reaches this line, so the failure cascaded."""
    import pathlib

    source = (pathlib.Path(__file__).parent / "layout.py").read_text()
    code = [line.split("#", 1)[0] for line in source.splitlines()]
    assert not [line for line in code if ".ptp()" in line]
    assert any("np.ptp(" in line for line in code)

# ----------------------------------------------------------------------
# test_phase6_baseline.py
# ----------------------------------------------------------------------
"""Multivariate baseline, spatial CV and ablation (spec sections 16-18).
"""
@pytest.fixture(scope="module")
def die_phase6(tmp_path_factory):
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

def test_ablation_requires_a_position_baseline(die_phase6):
    """A geometry AUC with nothing to compare against is not interpretable."""
    path, grid, feats, folds = die_phase6
    fs = failures_from_driver(feats["perimeter_density"], grid, n_failures=300,
                              strength=2.5, seed=1, position_sigma_um=5.0)
    res = pipeline.run(path, fs, layer=M8, scales_um=(100,), n_permutations=0,
                       line_end_w_max_um=6.0, seed=1, include_position=False)
    y = res.features["failure_present"].to_numpy(int)
    with pytest.raises(ValueError, match="position-only baseline"):
        ablation.run(res.features, y, folds)


def test_ablation_localises_where_the_information_enters(die_phase6):
    """Spec section 18: does sophisticated geometry beat metal density?

    Failures are driven by perimeter density. Metal density is not the driver
    but does correlate with it (r ~ 0.44 on this die), so the test is not that
    metal density is uninformative -- it is that the increment from adding
    perimeter dwarfs everything metal density carries on its own. A correlated
    non-driver showing a small effect is the expected behaviour, and the reason
    an ablation is read by the size of each step rather than by which steps
    reach significance.
    """
    path, grid, feats, folds = die_phase6
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

    baseline = next(s for s in out.scores if s.name == ablation.BASELINE_NAME)
    assert abs(baseline.roc_auc - 0.5) < 0.10, (
        "the validation die has no position effect; a position baseline well "
        "away from chance means the folds or the labels are wrong"
    )


def test_ablation_reports_nothing_under_the_null(die_phase6):
    """Every family must fail to add information when failures are uniform."""
    path, grid, feats, folds = die_phase6
    fs = uniform_failures(grid, n_failures=300, seed=42, position_sigma_um=5.0)
    res = pipeline.run(path, fs, layer=M8, scales_um=(100,), n_permutations=0,
                       line_end_w_max_um=6.0, seed=1)
    y = res.features["failure_present"].to_numpy(int)
    out = ablation.run(res.features, y, folds, seed=3)
    adding = [d["model"] for d in out.deltas if d["adds_information"]]
    assert not adding, f"models claimed information under the null: {adding}"


def test_delta_auc_interval_is_paired_and_spatially_blocked(die_phase6):
    """The interval must come from the same folds, resampled in blocks."""
    path, grid, feats, folds = die_phase6
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

# ----------------------------------------------------------------------
# test_stratification_and_registry.py
# ----------------------------------------------------------------------
"""Failure-mode stratification, and the registry that makes traceability bind.
"""
@pytest.fixture(scope="module")
def opposing(tmp_path_factory):
    """Two interfaces whose driver points the opposite way on each.

    Zahedmanesh and Vanstreels (2019) show a stiff top group lowering the
    crack driving force beneath it, so the same geometry helping on one
    interface and hurting on another is the expected shape, not a contrived
    one.
    """
    path = str(tmp_path_factory.mktemp("st") / "die.gds")
    validation_die(path, die_um=1500.0, block_um=50.0, seed=7)
    reader = LayoutReader(path)
    grid = build_grid(reader.bbox(), 100.0)
    feats = GeometryExtractor(reader, line_rules=RULES).extract(M8, grid)

    frames = []
    for label, driver in (("M8/ULK", feats["perimeter_density"]),
                          ("M8/CAP", -feats["perimeter_density"])):
        fs = failures_from_driver(driver, grid, n_failures=90, strength=2.5,
                                  seed=abs(hash(label)) % 1000,
                                  position_sigma_um=3.0)
        t = fs.table.copy()
        t["lot_id"], t["wafer_id"], t["die_x"], t["die_y"] = "L1", "W1", 0, 0
        t["sample_id"] = [f"{label[-3:]}_{s}" for s in t.sample_id]
        t["failed_interface"], t["failed_layer"] = label, "M8"
        frames.append(t)
    return path, FailureSet(table=pd.concat(frames, ignore_index=True))


def test_stratify_splits_on_the_declared_mode_column(opposing):
    _, mixed = opposing
    groups = stratify(mixed, ("failed_interface",))
    assert set(groups) == {"M8/ULK", "M8/CAP"}
    assert all(len(g) > 0 for g in groups.values())


def test_stratify_without_the_column_returns_one_population():
    table = pd.DataFrame({"sample_id": ["a"], "x_um": [1.0], "y_um": [1.0],
                          "failure_type": "delamination", "confidence": 1.0,
                          "position_sigma_um": np.nan})
    assert list(stratify(FailureSet(table=table))) == ["<all>"]


def test_pooling_opposite_effects_cancels_them_to_nothing(opposing):
    """The reason stratification is a population question, not bookkeeping."""
    path, mixed = opposing
    pooled = pipeline.run(path, mixed, layer=M8, scales_um=(100,),
                          n_permutations=0, line_rules=RULES, seed=1,
                          allow_pooling_modes=True)
    row = pooled.associations.set_index("feature").loc["perimeter_density"]
    assert abs(row["roc_auc"] - 0.5) < 0.15
    assert row["fdr_q_value"] > 0.05

    strat = pipeline.run_stratified(
        path, mixed, stratify_by=("failed_interface",), layer=M8,
        scales_um=(100,), n_permutations=0, line_rules=RULES, seed=1)
    effects = {name: res.associations.set_index("feature").loc[
        "perimeter_density", "effect_size"]
        for name, res in strat.per_stratum.items()}
    assert len(effects) == 2
    assert min(effects.values()) < -0.2 and max(effects.values()) > 0.2
    for name, res in strat.per_stratum.items():
        q = res.associations.set_index("feature").loc[
            "perimeter_density", "fdr_q_value"]
        assert q < 0.01, f"{name} should be significant on its own"


def test_consistency_reports_sign_disagreement_first(opposing):
    path, mixed = opposing
    strat = pipeline.run_stratified(
        path, mixed, stratify_by=("failed_interface",), layer=M8,
        scales_um=(100,), n_permutations=0, line_rules=RULES, seed=1)
    c = strat.consistency
    assert not c.empty
    assert not c.iloc[0]["signs_agree"], "disagreement must lead the table"
    row = c[c.feature == "perimeter_density"].iloc[0]
    assert not row["signs_agree"]
    assert row["effect_spread"] > 0.5
    assert "M8/ULK" in row["strata"] and "M8/CAP" in row["strata"]


def test_small_strata_are_skipped_and_named(opposing):
    path, mixed = opposing
    strat = pipeline.run_stratified(
        path, mixed, stratify_by=("failed_interface",), min_failures=1000,
        layer=M8, scales_um=(100,), n_permutations=0, line_rules=RULES, seed=1)
    assert len(strat) == 0


# ---- the registry --------------------------------------------------

@pytest.mark.parametrize("name,family", [
    ("metal_density", "metal_density"),
    ("metal_density_grad_mag", "metal_density"),
    ("routing_diagonality", "routing_bump_frame"),
    ("routing_radial_alignment", "routing_bump_frame"),
    ("wide_metal_perimeter_density", "wide_metal"),
    ("slotted_metal_fraction", "slot"),
    ("concave_corner_density", "corner_density"),
    ("perimeter_density_mismatch_M8_M7", "cross_layer_architecture"),
    ("under_pad_indicator", "bump_neighborhood"),
])
def test_every_reported_feature_maps_to_a_family(name, family):
    entry = registry.lookup(name)
    assert entry is not None and entry.family == family


def test_an_unregistered_feature_is_named():
    """A checklist nothing enforces is a wish."""
    audit = registry.audit(["metal_density", "invented_feature"])
    assert audit["unregistered"] == ["invented_feature"]
    assert not audit["complete"]


def test_tier1_families_carry_the_full_traceability():
    """Hypothesis, observable, unit, confounders, test, falsification, promotion."""
    reg = registry.load()
    incomplete = {name: e.missing_trace for name, e in reg.items()
                  if e.row.get("hypothesis_tier", "").startswith("tier1")
                  and e.missing_trace}
    assert not incomplete, f"tier-1 families missing traceability: {incomplete}"


def test_the_run_records_its_registry_audit(opposing):
    path, mixed = opposing
    res = pipeline.run(path, mixed, layer=M8, scales_um=(100,),
                       n_permutations=0, line_rules=RULES, seed=1,
                       allow_pooling_modes=True)
    audit = res.metadata["feature_registry"]
    assert audit["n_features"] > 0
    assert audit["unregistered"] == []


def _failure_file(path, interfaces):
    import pandas as pd

    n = len(interfaces)
    pd.DataFrame({
        "sample_id": ["s1"] * n,
        "x_um": [10.0 * (i + 1) for i in range(n)],
        "y_um": [10.0 * (i + 1) for i in range(n)],
        "lot_id": ["L1"] * n, "wafer_id": ["W1"] * n,
        "die_x": [0] * n, "die_y": [0] * n,
        "failure_type": ["delamination"] * n,
        "failed_interface": interfaces,
    }).to_csv(path, index=False)
    return str(path)


def test_a_missing_stratum_value_is_refused_not_bucketed(tmp_path):
    """The repair that suggests itself is worse than the crash it replaces.

    Joining a missing value in raised a TypeError -- a float NaN survives
    astype(str) on a nullable string column -- and the obvious fix, a "nan" or
    "<missing>" stratum, presents "we do not know which interface this was" as
    a mechanism alongside M8/ULK, with its own effect size, direction and
    q-value. Only rows whose mechanism is known can be analysed per mechanism.

    Coverage before this: all values present, or the column absent entirely.
    Not the real case, which is a column that exists and is partly filled.
    """
    from collective.labels import load_failures, stratify

    partly = load_failures(_failure_file(tmp_path / "partial.csv",
                                         ["M8/ULK", "M8/ULK", None]))
    with pytest.raises(ValueError, match="have no value in the stratifying"):
        stratify(partly, by=("failed_interface",))

    # An empty string is a missing value written a different way.
    blanked = load_failures(_failure_file(tmp_path / "blank.csv",
                                          ["M8/ULK", "", "M7/ULK"]))
    with pytest.raises(ValueError, match="have no value in the stratifying"):
        stratify(blanked, by=("failed_interface",))

    complete = load_failures(_failure_file(tmp_path / "complete.csv",
                                           ["M8/ULK", "M8/ULK", "M7/ULK"]))
    strata = stratify(complete, by=("failed_interface",))
    assert {k: len(v.table) for k, v in strata.items()} == {"M8/ULK": 2,
                                                            "M7/ULK": 1}

# ----------------------------------------------------------------------
# test_multi_die_and_modes.py
# ----------------------------------------------------------------------
"""Multi-die observation structure and failure-mode separation.

Two things the failure schema has always demanded and the analysis never used:
lot/wafer/die identity, required at import for the held-out validation of spec
section 17, and failure_type, required but never consulted before pooling.
Requiring data and then ignoring it is worse than not requiring it, because a
reader reasonably assumes it was used.
"""
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
