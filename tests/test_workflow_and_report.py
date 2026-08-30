"""Study manifest, the real-data workflow, and result partitioning."""
import numpy as np
import pandas as pd
import pytest

from lamxsim import report
from lamxsim.layout.reader import LayerSpec, LayoutReader
from lamxsim.layout.synth import packaged_die
from lamxsim.study import StudyManifest

MANIFEST = "config/study_manifest.yaml"


@pytest.fixture(scope="module")
def die(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("wf") / "die.gds")
    packaged_die(path, die_um=1500.0, block_um=100.0, seed=31)
    return path


# ---- manifest ------------------------------------------------------

def test_manifest_requires_an_ordered_metal_stack(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("layout: {}\n")
    with pytest.raises(ValueError, match="metal_layers is required"):
        StudyManifest.load(p)


def test_manifest_records_what_was_left_unspecified(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text(
        "layout:\n  metal_layers:\n    - {name: M8, layer: 8, datatype: 0}\n")
    m = StudyManifest.load(p)
    joined = " ".join(m.gaps)
    assert "no line_rules for M8" in joined
    assert "no die_outline_um" in joined
    assert "no via_layers" in joined
    assert "no registration fiducials" in joined


def test_manifest_rejects_a_layer_the_layout_does_not_have(die):
    m = StudyManifest.load(MANIFEST)
    reader = LayoutReader(die)
    m.validate_against(reader)          # the packaged die has all of them
    m.metal_layers.append(LayerSpec("M99", 99, 0))
    with pytest.raises(ValueError, match="is not in the layout"):
        m.validate_against(reader)


def test_line_end_width_comes_from_the_pdk_not_the_geometry():
    """Otherwise the shortest edge in the design defines a physical line end."""
    m = StudyManifest.load(MANIFEST)
    assert m.line_end_w_max_um() == 2.0


def test_full_die_footprint_needs_its_justification(die, tmp_path):
    reader = LayoutReader(die)
    m = StudyManifest.load(MANIFEST)
    m.footprint_spec = {"full_die": None}
    assert m.footprint(reader, reader.bbox()) is None
    m.footprint_spec = {"full_die": "whole-die C-SAM, all indications called"}
    fp = m.footprint(reader, reader.bbox())
    assert fp.assumed_full_coverage and fp.justification


# ---- report partitioning -------------------------------------------

def _row(**kw):
    base = dict(feature="metal_density", layer="M8", scale_um=100.0,
                evidence_class="GDS_GEOMETRY", hypothesis_tier="tier1",
                scale_trustworthy=True, roc_auc=0.75, auc_ci_low=0.70,
                auc_ci_high=0.80, effect_size=0.5, fdr_q_value=0.001,
                n_case=100, n_control=100, effective_n=50.0,
                enrichment_top_10pct=2.0)
    base.update(kw)
    return base


def test_partition_separates_the_four_things_a_row_can_be():
    df = pd.DataFrame([
        _row(feature="perimeter_density"),
        _row(feature="distance_to_die_edge", evidence_class="PACKAGE_POSITION",
             hypothesis_tier="tier1_confounder"),
        _row(feature="largest_polygon", hypothesis_tier="exploratory",
             fdr_q_value=np.nan),
        _row(feature="metal_density", scale_um=25.0, scale_trustworthy=False),
    ])
    parts = report.partition(df)
    assert list(parts["primary"].feature) == ["perimeter_density"]
    assert list(parts["confounders"].feature) == ["distance_to_die_edge"]
    assert list(parts["exploratory"].feature) == ["largest_polygon"]
    assert list(parts["unsupported_scale"].feature) == ["metal_density"]


def test_a_saturated_auc_does_not_lead_the_primary_table():
    """At a coarse scale nearly every window holds a failure.

    The AUC then saturates at 1.0 against a handful of controls, and ranking
    by effect size alone would put that above a real, well-powered result.
    """
    df = pd.DataFrame([
        _row(feature="saturated", roc_auc=1.0, effect_size=1.0,
             auc_ci_low=1.0, auc_ci_high=1.0, n_case=35, n_control=1,
             fdr_q_value=0.21),
        _row(feature="real", roc_auc=0.79, effect_size=0.58,
             auc_ci_low=0.69, auc_ci_high=0.88, n_case=108, n_control=36),
    ])
    parts = report.partition(df)
    assert list(parts["primary"].feature) == ["real"]
    assert list(parts["underpowered"].feature) == ["saturated"]


def test_a_wide_interval_does_not_outrank_a_tight_one():
    df = pd.DataFrame([
        _row(feature="wide", roc_auc=0.85, effect_size=0.70,
             auc_ci_low=0.52, auc_ci_high=0.99),
        _row(feature="tight", roc_auc=0.78, effect_size=0.56,
             auc_ci_low=0.71, auc_ci_high=0.84),
    ])
    assert list(report.partition(df)["primary"].feature) == ["tight", "wide"]


def test_summary_states_what_the_result_is_not(tmp_path):
    df = pd.DataFrame([_row()])
    paths = report.write(df, tmp_path, metadata={
        "uncontrolled_confounding": ["no bump layer supplied"]})
    text = (tmp_path / "reports" / "README.md").read_text()
    assert "not a causal claim" in text
    assert "not a design rule" in text
    assert "no bump layer supplied" in text
    assert set(paths) >= {"primary", "confounders", "exploratory",
                          "unsupported_scale", "underpowered", "summary"}


def test_empty_associations_partition_without_raising():
    parts = report.partition(pd.DataFrame())
    assert all(len(v) == 0 for v in parts.values())
    assert "no primary result" in report.format_primary(pd.DataFrame())
