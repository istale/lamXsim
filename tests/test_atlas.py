"""The GDS-only deliverable: a literature exposure atlas.

Everything here runs on a layout and a layer map, with no failure data of any
kind. The tests that matter are the ones that stop it becoming a risk score,
because that is what an exposure atlas turns into if nobody is watching.
"""
from pathlib import Path

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
                          "unimplemented_gds_observables",
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
    gaps = pd.read_csv(tmp_path / "unsupported_non_gds_physics.csv")
    assert set(gaps.channel) == {c.channel_id for c in exposure.CHANNELS}
    assert (~gaps.recoverable_from_gds).all()
    assert "EMC thickness" in set(gaps.quantity)


def test_every_conditioned_candidate_lies_inside_its_own_condition(tmp_path):
    """The condition was written into the report and not into the computation.

    ``routing_in_bump_frame`` shipped 28 candidates on the regression die, all
    of them carrying ``conditioned_on=distance_to_nearest_corner`` in the CSV
    and every single one of them *outside* the corner region that condition
    names -- the ranking ran die-wide and the gate was never applied. Ranking
    inside the region gives 8, in different places.

    Feature-level unit tests could not see this: ``condition_mask`` and
    ``evaluate`` were both correct, and the caller simply did not put them
    together.
    """
    from lamxsim import atlas as atlas_mod
    from lamxsim.features.grid import build_multiscale
    from lamxsim.layout.reader import LayoutReader
    from lamxsim.study import StudyManifest

    golden = Path(__file__).parent / "golden"
    manifest = StudyManifest.load(golden / "golden_manifest.yaml")
    gds = str(golden / "golden_die.gds")
    result = atlas_mod.build(gds, manifest)

    reader = LayoutReader(gds, top_cell=manifest.top_cell)
    die_bbox = manifest.die_bbox(reader)
    conditioned = [c for c in exposure.CHANNELS if c.conditional_on]
    assert conditioned, "nothing to check; the test has lost its subject"

    checked = 0
    for scale, grid in sorted(build_multiscale(reader.bbox(),
                                               manifest.scales_um).items()):
        flat, _ = atlas_mod._extract_scale(reader, manifest, grid, die_bbox)
        index = {(round(c.x_center, 6), round(c.y_center, 6)): i
                 for i, c in enumerate(grid.cells)}
        for channel in conditioned:
            owners = (["-"] if channel.scope == "die"
                      else [s.name for s in manifest.metal_layers])
            for owner in owners:
                mask, _ = exposure.condition_mask(
                    channel, atlas_mod._channel_inputs(flat, owner), len(grid))
                rows = result.candidates[
                    (result.candidates.channel == channel.channel_id)
                    & (result.candidates.layer == owner)
                    & (result.candidates.scale_um == scale)]
                for _, row in rows.iterrows():
                    i = index[(round(row.x_um, 6), round(row.y_um, 6))]
                    assert mask[i], (
                        f"{channel.channel_id} candidate at "
                        f"({row.x_um}, {row.y_um}) is outside "
                        f"{channel.conditional_on}, which the same row "
                        "declares it was conditioned on")
                    assert row.condition_cells == int(mask.sum())
                    checked += 1
    assert checked, "no conditioned candidate was produced, so nothing was checked"


def test_die_relative_channels_need_a_declared_die_outline(tmp_path):
    """Whether this GDS is a whole die or a piece of one is not in the layout.

    With no outline the die bbox is the geometry bbox either way, so the test
    that used to guard this -- comparing the two -- could only ever be false,
    and the flag it set disabled nothing. Corner distance, offset from the die
    centre and bump radial direction are all measured from that frame.
    """
    import yaml

    from lamxsim import atlas as atlas_mod
    from lamxsim.study import StudyManifest

    golden = Path(__file__).parent / "golden"
    raw = yaml.safe_load((golden / "golden_manifest.yaml").read_text())
    assert raw["layout"].pop("die_outline_um", None) is not None
    path = tmp_path / "no_outline.yaml"
    path.write_text(yaml.safe_dump(raw))

    result = atlas_mod.build(str(golden / "golden_die.gds"),
                             StudyManifest.load(path))
    assert result.metadata["die_frame_declared"] is False
    assert result.metadata["die_relative_channels_disabled"] is True

    needs_frame = {c.channel_id for c in exposure.CHANNELS if c.needs_die_frame}
    assert needs_frame
    assert not set(result.candidates.channel) & needs_frame
    reasons = {r.reason for cs in result.channels.values() for _, r in cs
               if r.channel.channel_id in needs_frame}
    assert any("no die_outline_um" in r for r in reasons)


def test_a_channel_whose_condition_is_unavailable_is_refused_not_widened():
    """Widening is the same error as never conditioning, from the other side.

    The channel would be scored across the whole die and reported under a
    citation that is about one region of it. Refusing says less and says it
    accurately.
    """
    channel = next(c for c in exposure.CHANNELS if c.conditional_on)
    features = {name: np.linspace(0.0, 1.0, 25) for name in channel.inputs}
    assert channel.conditional_on not in features

    mask, reason = exposure.condition_mask(channel, features, 25)
    assert mask is None
    assert channel.conditional_on in reason

    results = {r.channel.channel_id: r
               for r in exposure.evaluate_all(features, 25)}
    assert results[channel.channel_id].available is False


def _slotting_die(path, *, pitch=50.0, n=8, solid_cells=((0, 0), (7, 7), (0, 7))):
    """An n x n array of separate plates, all slotted except a named few.

    The golden die cannot exercise these channels: its wide-metal fraction
    takes too few distinct values, so every cell ties and the channel
    correctly reports nothing. Nor can a die that is half solid -- half the
    cells then share the top value, which is the top 50 % and not the top 5 %.
    A channel that is never exercised is a channel whose direction nobody has
    checked, so the fixture is built to leave a genuine small extreme: three
    solid plates out of 64 is 4.7 %.

    The plates are separated so the boolean engine does not merge them, which
    matters because the unslotted measure is defined per polygon.
    """
    import klayout.db as db

    from lamxsim.layout import synth

    size = pitch * 0.8
    sl = synth.SynthLayout()
    # A frame on an unanalysed layer, so the geometry bounding box is the die
    # and the analysis grid lands one window per plate. Without it the bbox is
    # the plate extent, the grid is centred inside it, and a test asserting
    # which cell was flagged is really asserting where the grid happened to
    # fall.
    die = pitch * n
    for x0, y0, x1, y1 in ((0, 0, die, 0.5), (0, die - 0.5, die, die),
                           (0, 0, 0.5, die), (die - 0.5, 0, die, die)):
        sl.add_box(63, x0, y0, x1, y1)
    for j in range(n):
        for i in range(n):
            x0, y0 = i * pitch + (pitch - size) / 2, j * pitch + (pitch - size) / 2
            for layer in (8, 7):
                sl.add_box(layer, x0, y0, x0 + size, y0 + size)
            if (i, j) in solid_cells:
                continue
            for b in range(3):
                for a in range(3):
                    sl.add_box(99, x0 + 5 + a * 12, y0 + 5 + b * 12,
                               x0 + 11 + a * 12, y0 + 11 + b * 12)
    sl.write(str(path))

    layout = db.Layout()
    layout.read(str(path))
    top = layout.top_cells()[0]
    cut_index = layout.find_layer(99, 0)
    cutter = db.Region()
    cutter.insert(top.begin_shapes_rec(cut_index))
    cutter.merge()
    for number in (8, 7):
        li = layout.find_layer(number, 0)
        metal = db.Region()
        metal.insert(top.begin_shapes_rec(li))
        metal.merge()
        top.shapes(li).clear()
        top.shapes(li).insert(metal - cutter)
    top.shapes(cut_index).clear()
    layout.write(str(path))
    return str(path)


def _slotting_manifest(path, *, pitch=50.0, n=8):
    import yaml

    die = pitch * n
    yaml.safe_dump({
        "layout": {
            "top_cell": "TOP",
            "metal_layers": [{"name": "M8", "layer": 8, "datatype": 0},
                             {"name": "M7", "layer": 7, "datatype": 0}],
            "die_outline_um": [0, 0, die, die],
            "wide_width_um": 3.0,
            "line_rules": {"M8": {"min_width_um": 0.2, "line_max_width_um": 2.0},
                           "M7": {"min_width_um": 0.2, "line_max_width_um": 2.0}},
        },
        "analysis": {"scales_um": [pitch]},
    }, open(path, "w"))
    return str(path)


def test_slotting_lowers_the_score_it_is_supposed_to_lower(tmp_path):
    """Rabie's lever is slotting, so the recommended state must score lower.

    Ranking ``wide_metal_fraction`` would flag a correctly slotted plate
    exactly as hard as an unbroken one, which inverts the lever. The channel
    reads unslotted wide metal instead, and this asserts the direction on a
    die that is solid on one half and slotted on the other.
    """
    from lamxsim.study import StudyManifest

    solid = {(0, 0), (7, 7), (0, 7)}
    gds = _slotting_die(tmp_path / "slotting.gds", solid_cells=tuple(solid))
    manifest = StudyManifest.load(_slotting_manifest(tmp_path / "m.yaml"))
    result = atlas.build(gds, manifest)

    rows = result.candidates[result.candidates.channel == "wide_metal_slotting"]
    assert not rows.empty, "the channel found nothing on a die built to trip it"
    flagged = {(int(r.x_um // 50), int(r.y_um // 50)) for _, r in rows.iterrows()}
    assert flagged == solid, flagged

    values = result.features["unslotted_wide_metal_fraction|M8"]
    xs = result.features.x_um.to_numpy() // 50
    ys = result.features.y_um.to_numpy() // 50
    is_solid = np.array([(int(a), int(b)) in solid for a, b in zip(xs, ys)])
    assert values[is_solid].min() > 0.9
    assert values[~is_solid].max() == 0.0


def test_the_corner_tile_lever_is_top_layer_and_corner_only(tmp_path):
    """Scored on every layer it asserts something the citation does not.

    Rabie's lever is corner metal tiling on the top group. Reported on M7 it
    would be a claim about M7; reported die-wide it would be the
    wide_metal_slotting channel under a second citation from the same paper.
    """
    from lamxsim.study import StudyManifest

    # One solid corner plate, not three. The corner condition admits the 16
    # cells nearest a corner, so three of them share the top value and sit at
    # the 81st percentile -- correct tie compression, and useless for testing
    # where the candidates land.
    gds = _slotting_die(tmp_path / "corner.gds", solid_cells=((0, 0),))
    manifest = StudyManifest.load(_slotting_manifest(tmp_path / "m.yaml"))
    result = atlas.build(gds, manifest)

    scored = {(owner, r.channel.channel_id)
              for cs in result.channels.values() for owner, r in cs}
    assert ("M8", "corner_metal_tiles") in scored
    assert ("M7", "corner_metal_tiles") not in scored
    assert ("M7", "wide_metal_slotting") in scored

    rows = result.candidates[result.candidates.channel == "corner_metal_tiles"]
    assert not rows.empty, "the corner lever found nothing on a corner-solid die"
    assert set(rows.layer) == {"M8"}
    # Corner-conditioned: nothing may sit in the middle of the die.
    middle = ((rows.x_um - 200.0).abs() < 60.0) & \
             ((rows.y_um - 200.0).abs() < 60.0)
    assert not middle.any()
