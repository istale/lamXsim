"""Failure-mode stratification, and the registry that makes traceability bind."""
import numpy as np
import pandas as pd
import pytest

from collective import workflow as pipeline, foundation as registry
from collective.geometry import GeometryExtractor
from collective.geometry import build_grid
from collective.labels import FailureSet, stratify
from collective.labels import failures_from_driver
from collective.layout import LayerSpec, LayoutReader
from collective.layout import validation_die

M8 = LayerSpec("M8", 8, 0)
RULES = {"M8": (0.5, 4.0)}


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
