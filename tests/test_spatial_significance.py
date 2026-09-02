"""What a primary claim is allowed to rest on.

The pipeline computed a within-die block permutation and then corrected the
Mann-Whitney p-value instead, leaving the spatial result in a side table. On a
die with no package-position effect the naive test called 11 of 12 position
associations significant where the permutation called none -- so the report
was quoting the test the repository's own README shows to be wrong.
"""
import numpy as np
import pandas as pd
import pytest

from collective import workflow as pipeline
from collective.geometry import GeometryExtractor
from collective.geometry import build_grid
from collective.labels import failures_from_driver
from collective.layout import LayerSpec, LayoutReader
from collective.layout import validation_die
from collective.exposure import partition

M8 = LayerSpec("M8", 8, 0)
RULES = {"M8": (0.5, 4.0)}


@pytest.fixture(scope="module")
def die(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("sig") / "die.gds")
    validation_die(path, die_um=1500.0, block_um=50.0, seed=7)
    reader = LayoutReader(path)
    grid = build_grid(reader.bbox(), 100.0)
    feats = GeometryExtractor(reader, line_rules=RULES).extract(M8, grid)
    fs = failures_from_driver(feats["perimeter_density"], grid, n_failures=80,
                              strength=2.5, seed=1, position_sigma_um=3.0)
    return path, fs


def test_without_a_permutation_there_is_no_primary_evidence(die):
    """Significance must not be reachable by skipping the spatial null."""
    path, fs = die
    res = pipeline.run(path, fs, layer=M8, scales_um=(100,), n_permutations=0,
                       line_rules=RULES, seed=1)
    assert res.associations["spatial_q_value"].isna().all()
    parts = partition(res.associations)
    assert len(parts["primary"]) == 0
    assert len(parts["not_spatially_corrected"]) > 0
    assert any("no spatial permutation was run" in n
               for n in res.metadata["uncontrolled_confounding"])


def test_the_spatial_q_value_is_what_primary_rests_on(die):
    path, fs = die
    res = pipeline.run(path, fs, layer=M8, scales_um=(100,),
                       n_permutations=299, line_rules=RULES, seed=1)
    a = res.associations
    assert a["spatial_q_value"].notna().any()
    parts = partition(a)
    assert len(parts["primary"]) > 0
    assert parts["primary"]["spatial_q_value"].notna().all()


def test_the_naive_q_value_survives_as_a_diagnostic(die):
    """Kept for the contrast it provides, not as the basis of a finding."""
    path, fs = die
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

    code = [line.split("#", 1)[0]
            for line in pathlib.Path("collective/layout.py").read_text()
            .splitlines()]
    assert not [line for line in code if ".ptp()" in line]
    assert any("np.ptp(" in line for line in code)
