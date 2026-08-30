"""The GDS-only deliverable: a literature exposure atlas.

Everything here runs on a layout and a layer map, with no failure data of any
kind. The tests that matter are the ones that stop it becoming a risk score,
because that is what an exposure atlas turns into if nobody is watching.
"""
import numpy as np
import pandas as pd
import pytest

from lamxsim import atlas, exposure
from lamxsim.layout.synth import packaged_die
from lamxsim.study import StudyManifest

MANIFEST = """
layout:
  metal_layers:
    - {name: M8, layer: 8, datatype: 0}
    - {name: M7, layer: 7, datatype: 0}
  via_layers:
    M8: {name: V7, layer: 17, datatype: 0}
  package_layers:
    bump: {name: BUMP, layer: 60, datatype: 0}
    pi_opening: {name: PI, layer: 61, datatype: 0}
    crackstop: {name: CS, layer: 62, datatype: 0}
  line_rules:
    M8: {min_width_um: 0.2, line_max_width_um: 2.0}
    M7: {min_width_um: 0.1, line_max_width_um: 1.0}
  die_outline_um: [0, 0, 1500, 1500]
analysis:
  scales_um: [100, 250]
"""


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    d = tmp_path_factory.mktemp("atlas")
    gds = str(d / "chip.gds")
    packaged_die(gds, die_um=1500.0, block_um=100.0, seed=5)
    manifest_path = d / "layers.yaml"
    manifest_path.write_text(MANIFEST)
    manifest = StudyManifest.load(manifest_path)
    return gds, manifest, d


@pytest.fixture(scope="module")
def built(study):
    gds, manifest, _ = study
    return atlas.build(gds, manifest)


# ---- it runs on a layout and a layer map, and nothing else ----------

def test_an_atlas_needs_no_failure_data(built):
    assert not built.features.empty
    assert not built.candidates.empty
    assert "failure" not in " ".join(built.features.columns).lower()


def test_every_channel_reports_or_says_why_not(built):
    seen = {r.channel.channel_id for cs in built.channels.values()
            for _, r in cs}
    assert seen == {c.channel_id for c in exposure.CHANNELS}
    for scale_channels in built.channels.values():
        for _, result in scale_channels:
            assert result.available or result.reason


# ---- the things that stop it becoming a risk score ------------------

def test_channels_are_never_combined(built):
    """Spec section 1 forbids an arbitrary weighted probability, and a
    weighted sum of these channels is that under another name: the weights
    could only come from data this study does not have."""
    for column in built.candidates.columns:
        assert "score" not in column
        assert "risk" not in column
        assert "probability" not in column
    # A location on several channels is several records, each with its own
    # citation, not one row with a count.
    grouped = built.candidates.groupby(["x_um", "y_um", "scale_um"]).channel
    assert grouped.nunique().max() > 1
    assert "n_channels" not in built.candidates.columns


def test_the_overlay_has_no_combined_hotspot_layer(built, study, tmp_path):
    import klayout.db as db

    gds, manifest, _ = study
    paths = atlas.write(built, tmp_path, manifest)
    layout = db.Layout()
    layout.read(paths["candidate_regions"])
    layers = {str(layout.get_info(i)) for i in layout.layer_indexes()}
    assert len(layers) == built.candidates.channel.nunique()
    mapping = built.metadata["overlay_layers"]
    assert set(mapping) == set(built.candidates.channel.unique())


def test_ranking_is_a_percentile_within_this_die(built):
    """Nothing is calibrated, so a percentile is the strongest available claim."""
    assert built.candidates.percentile_in_die.between(0, 100).all()
    assert built.candidates.percentile_in_die.min() >= 95.0


def test_a_two_sided_channel_flags_both_tails():
    """Zahedmanesh and Vanstreels show a stiff top group lowering the driving
    force, so flagging only the high end would invent a direction."""
    values = np.linspace(0.0, 1.0, 101)
    one = exposure._percentile_rank(values, two_sided=False)
    two = exposure._percentile_rank(values, two_sided=True)
    assert one.argmax() == len(values) - 1
    assert two[0] == pytest.approx(100.0)
    assert two[-1] == pytest.approx(100.0)
    assert two[50] == pytest.approx(0.0, abs=1.0)


def test_perimeter_channel_is_not_a_density_map():
    """Yoo's result is that perimeter beats density, so the channel reads the
    residual; the raw value would rank the densest regions."""
    rng = np.random.default_rng(0)
    density = rng.uniform(0.2, 0.8, 400)
    perimeter = 3.0 * density + rng.normal(0, 0.05, 400)   # mostly density
    features = {"perimeter_density": perimeter, "metal_density": density}
    channel = next(c for c in exposure.CHANNELS
                   if c.channel_id == "perimeter_at_matched_density")
    result = exposure.evaluate(channel, features, 400)

    raw_rank = exposure._percentile_rank(perimeter, False)
    assert abs(np.corrcoef(raw_rank, density)[0, 1]) > 0.9
    assert abs(np.corrcoef(result.percentile, density)[0, 1]) < 0.3


# ---- scope: a shared feature is one candidate, not one per layer ----

def test_die_scoped_channels_are_reported_once(built):
    die_scoped = {c.channel_id for c in exposure.CHANNELS if c.scope == "die"}
    for channel in die_scoped:
        rows = built.candidates[built.candidates.channel == channel]
        if rows.empty:
            continue
        assert set(rows.layer) == {"-"}, (
            f"{channel} reads shared features; scoring it per layer would "
            "report one candidate as several and read as corroboration")


def test_layer_scoped_channels_carry_their_layer(built):
    rows = built.candidates[built.candidates.channel == "termination"]
    assert set(rows.layer) == {"M8", "M7"}


def test_a_channel_without_its_layer_is_not_scored(built):
    """M7 has no via layer declared, so via architecture cannot be scored."""
    rows = built.candidates[built.candidates.channel == "via_architecture"]
    assert set(rows.layer) == {"M8"}


# ---- provenance travels with every record --------------------------

def test_every_candidate_carries_its_citation_and_its_gap(built):
    for column in ("references", "mechanism", "unsupported_physics",
                   "inputs_used", "two_sided"):
        assert column in built.candidates.columns
        assert built.candidates[column].notna().all()
    assert built.candidates.references.str.len().min() > 0


def test_the_outputs_state_what_the_atlas_is_not(study, built, tmp_path):
    gds, manifest, _ = study
    paths = atlas.write(built, tmp_path, manifest)
    text = (tmp_path / "assumptions_and_limits.md").read_text()
    for claim in ("Not a statistical association", "Not a probability",
                  "Not a design rule", "Not a combined risk score"):
        assert claim in text
    assert set(paths) >= {"feature_maps", "literature_candidates",
                          "literature_traceability", "unsupported_physics",
                          "candidate_regions", "assumptions_and_limits"}


def test_traceability_links_every_channel_input_to_the_registry(study, built,
                                                               tmp_path):
    gds, manifest, _ = study
    atlas.write(built, tmp_path, manifest)
    trace = pd.read_csv(tmp_path / "literature_traceability.csv")
    assert set(trace.channel) == {c.channel_id for c in exposure.CHANNELS}
    assert trace.registry_complete.any()
    unregistered = trace[trace.registry_family == ""]
    assert unregistered.empty, f"channel inputs with no registry entry: " \
                               f"{list(unregistered.feature)}"


def test_unsupported_physics_is_listed_per_channel(study, built, tmp_path):
    gds, manifest, _ = study
    atlas.write(built, tmp_path, manifest)
    gaps = pd.read_csv(tmp_path / "unsupported_physics.csv")
    assert set(gaps.channel) == {c.channel_id for c in exposure.CHANNELS}
    assert (~gaps.recoverable_from_gds).all()
    assert "EMC thickness" in set(gaps.quantity)
