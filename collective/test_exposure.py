"""Literature channels, the atlas they build, and the reports.

Folded from ``tests/test_atlas.py``, ``tests/test_golden_run.py``, ``tests/test_workflow_and_report.py``, ``tests/test_conditions_and_budget.py``.
"""
from collective import exposure
from collective import exposure as atlas
from collective import exposure as report
from collective.geometry import build_grid
from collective.labels import FailureSet
from collective.layout import BBox
from collective.layout import LayerSpec
from collective.layout import LayoutReader
from collective.layout import packaged_die
from collective.statistics import block_permutation_test
from collective.statistics import min_achievable_p
from collective.statistics import permutation_budget
from collective.statistics import required_permutations
from collective.study import SampleConditions
from collective.study import StudyManifest
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest


# ----------------------------------------------------------------------
# test_atlas.py
# ----------------------------------------------------------------------
"""The GDS-only deliverable: a literature exposure atlas.

Everything here runs on a layout and a layer map, with no failure data of any
kind. The tests that matter are the ones that stop it becoming a risk score,
because that is what an exposure atlas turns into if nobody is watching.
"""
MANIFEST_atlas = """
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
    manifest_path.write_text(MANIFEST_atlas)
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
    from collective import exposure as atlas_mod
    from collective.geometry import build_multiscale
    from collective.layout import LayoutReader
    from collective.study import StudyManifest

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

    from collective import exposure as atlas_mod
    from collective.study import StudyManifest

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

    from collective import layout as synth

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
    from collective.study import StudyManifest

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
    from collective.study import StudyManifest

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


def test_the_coverage_ledger_does_not_contradict_the_channels():
    """A ledger that lists implemented work is worse than no ledger.

    ``unimplemented_gds_observables.csv`` is what a user reads to know what the
    atlas does not cover. It listed corner metal tiles and wide-metal slotting
    after both became channels, so the coverage statement shipped to a user
    contradicted the channels shipped beside it. Nothing caught it because
    nothing compared the two.

    The check is structural rather than a word match: each row carries a
    status and, when partial, the channels that already cover part of it. A
    word match called a crackstop row a contradiction of corner_metal_tiles
    because both contain "corner", which is the kind of check that gets
    loosened until it passes.
    """
    from collective import exposure as atlas_mod

    ledger = atlas_mod._unimplemented_observables()
    assert set(ledger.status) <= {"absent", "partial", "not_recoverable"}

    known = {c.channel_id for c in exposure.CHANNELS}
    for _, row in ledger.iterrows():
        named = [c for c in row.covered_by.split(";") if c]
        assert set(named) <= known, f"{row.observable!r} names {named}"
        if row.status == "absent":
            assert not named, (
                f"{row.observable!r} is marked absent but names {named}")

    # Every channel that exists must be findable: if a channel's subject is in
    # the ledger at all, the row has to name it rather than read as untouched.
    for channel_id in known:
        rows = ledger[ledger.covered_by.str.contains(channel_id, regex=False)]
        for _, row in rows.iterrows():
            assert row.status == "partial", (
                f"{row.observable!r} names {channel_id} but is {row.status}")

    # The two a layout genuinely cannot supply must stay marked as such, and
    # must stay listed: each has a proxy nearby that is easy to mistake for it.
    not_recoverable = set(ledger[ledger.status == "not_recoverable"].observable)
    assert any("sidewall" in o for o in not_recoverable)
    assert any("critical bump" in o for o in not_recoverable)
    assert (~ledger.recoverable_from_gds).sum() == len(not_recoverable)


def test_every_channel_input_is_traceable_to_the_registry():
    """A channel whose observable has no registry row is an unsourced claim."""
    from collective import foundation as registry

    for channel in exposure.CHANNELS:
        for feature in channel.inputs:
            entry = registry.lookup(feature)
            assert entry is not None, f"{channel.channel_id}: {feature}"
            assert not entry.missing_trace, (
                f"{channel.channel_id}: {feature} -> {entry.missing_trace}")


def test_a_die_of_uniform_density_says_so_instead_of_warning(recwarn):
    """A guard on exact zero let floating-point dust through.

    A die of uniform metal density gave a spread of exactly zero and a
    standard deviation of 5.6e-17, so `std == 0` was false, the fit went ahead
    on a constant column, numpy printed "Polyfit may be poorly conditioned",
    and the channel returned a residual computed from a degenerate fit. The
    user saw a warning they could not act on, attached to a number that was
    arithmetic noise.
    """
    perimeter = np.linspace(0.0, 1.0, 50)
    constant = np.full(50, 0.4)
    constant[0] += 1e-17          # the dust that defeated the old guard

    values, note = exposure._residualise(perimeter, constant)
    assert not recwarn.list, [str(w.message) for w in recwarn.list]
    assert "does not vary" in note
    # The raw perimeter is used, so the channel is a perimeter map here and
    # the note says that rather than the values hiding it.
    assert np.allclose(values, perimeter)

    varying = np.linspace(0.2, 0.8, 50)
    residual, clean_note = exposure._residualise(perimeter, varying)
    assert clean_note == ""
    assert not np.allclose(residual, perimeter)
    assert abs(float(np.mean(residual))) < 1e-9


def test_the_uniform_density_note_reaches_the_channel_reason():
    channel = next(c for c in exposure.CHANNELS
                   if c.channel_id == "perimeter_at_matched_density")
    features = {"perimeter_density": np.linspace(0.0, 1.0, 40),
                "metal_density": np.full(40, 0.4)}
    result = exposure.evaluate(channel, features, 40)
    assert result.available
    assert "does not vary" in result.reason

# ----------------------------------------------------------------------
# test_golden_run.py
# ----------------------------------------------------------------------
"""A frozen study whose full conclusion is committed.

Every defect found in review this session was invisible to the unit tests and
visible only on an end-to-end run: a value declared in one module and not
consumed in another, a p-value corrected from the wrong test, a die frame that
never reached the pipeline. Unit tests stayed green throughout.

This is the regression that catches that class. The inputs are frozen, the
candidate list is committed, and any change that alters a conclusion shows up
as a diff rather than as a silently different report. Regenerating the
expectation is allowed; doing it without saying why in the commit is what this
exists to prevent.
"""
GOLDEN = Path(__file__).parent / "golden"
EXPECTED = GOLDEN / "expected"


def _rebuild():
    manifest = StudyManifest.load(GOLDEN / "golden_manifest.yaml")
    return manifest, atlas.build(str(GOLDEN / "golden_die.gds"), manifest)


def _read(path: Path) -> pd.DataFrame:
    """Read without turning an empty optional field into NaN.

    An all-empty text column round-trips to float NaN otherwise, and the
    comparison then fails on how pandas encodes absence rather than on the
    conclusion.
    """
    return pd.read_csv(path, keep_default_na=False)


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        if out[column].dtype == object:
            out[column] = out[column].fillna("").astype(str)
    return out.sort_values(list(out.columns)).reset_index(drop=True)


def _candidates() -> pd.DataFrame:
    _, result = _rebuild()
    return _normalise(result.candidates)


def test_the_frozen_study_reaches_the_same_conclusion():
    """The whole record, not a projection of it.

    An earlier version froze seven columns, so a citation dropped from a
    channel changed nothing here. The mechanism text, the references, the
    two-sidedness, the triggering input and the declared conditioning are all
    part of the conclusion.
    """
    expected = _read(GOLDEN / "golden_candidates.csv")
    actual = _candidates()

    assert list(actual.columns) == list(expected.columns), (
        "the candidate schema changed; regenerate the golden expectation and "
        "say in the commit what was added or removed")
    assert len(actual) == len(expected), (
        f"the candidate count moved from {len(expected)} to {len(actual)}. "
        "If that is intended, regenerate tests/golden/ and say in the commit "
        "which change moved it and why.")

    pd.testing.assert_frame_equal(actual, _normalise(expected),
                                  check_dtype=False, atol=1e-6)


def test_the_narrative_outputs_are_frozen_too(tmp_path):
    """A citation lost from a channel is a changed conclusion."""
    manifest, result = _rebuild()
    atlas.write(result, tmp_path, manifest)
    for name in ("literature_traceability.csv",
                 "unsupported_non_gds_physics.csv",
                 "unimplemented_gds_observables.csv"):
        pd.testing.assert_frame_equal(
            _normalise(_read(tmp_path / name)),
            _normalise(_read(EXPECTED / name)), check_dtype=False)
    assert ((tmp_path / "assumptions_and_limits.md").read_text()
            == (EXPECTED / "assumptions_and_limits.md").read_text())


def test_the_overlay_layer_mapping_is_stable(tmp_path):
    """A channel silently changing layer number breaks a reader's overlay."""
    manifest, result = _rebuild()
    atlas.write(result, tmp_path, manifest)
    expected = json.loads((GOLDEN / "golden_metadata.json").read_text())
    assert result.metadata["overlay_layers"] == expected["overlay_layers"]
    assert result.metadata["scales_um"] == expected["scales_um"]
    assert result.metadata["die_bbox_um"] == expected["die_bbox_um"]


def test_the_channel_breakdown_is_stable():
    """A shift between channels is a changed conclusion even at equal totals."""
    expected = _read(GOLDEN / "golden_candidates.csv")
    actual = _candidates()
    assert (actual.groupby("channel").size().to_dict()
            == expected.groupby("channel").size().to_dict())


def test_die_scoped_channels_stay_die_scoped():
    """Scoring a shared feature per layer would silently multiply candidates."""
    actual = _candidates()
    for channel in ("cross_layer_mismatch", "pi_opening_proximity"):
        rows = actual[actual.channel == channel]
        if not rows.empty:
            assert set(rows.layer) == {"-"}


def test_the_golden_inputs_are_present():
    for name in ("golden_die.gds", "golden_manifest.yaml",
                 "golden_candidates.csv", "golden_metadata.json"):
        assert (GOLDEN / name).exists(), name
    assert (EXPECTED / "assumptions_and_limits.md").exists()

# ----------------------------------------------------------------------
# test_workflow_and_report.py
# ----------------------------------------------------------------------
"""Study manifest, the real-data workflow, and result partitioning.
"""
MANIFEST_workflow = "config/study_manifest.yaml"


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
    m = StudyManifest.load(MANIFEST_workflow)
    reader = LayoutReader(die)
    m.validate_against(reader)          # the packaged die has all of them
    m.metal_layers.append(LayerSpec("M99", 99, 0))
    with pytest.raises(ValueError, match="is not in the layout"):
        m.validate_against(reader)


def test_line_end_width_comes_from_the_pdk_not_the_geometry():
    """Otherwise the shortest edge in the design defines a physical line end."""
    m = StudyManifest.load(MANIFEST_workflow)
    assert m.line_end_w_max_um() == 2.0


def test_full_die_footprint_needs_its_justification(die, tmp_path):
    reader = LayoutReader(die)
    m = StudyManifest.load(MANIFEST_workflow)
    m.footprint_spec = {"full_die": None}
    assert m.footprint(reader, reader.bbox()) is None
    m.footprint_spec = {"full_die": "whole-die C-SAM, all indications called"}
    fp = m.footprint(reader, reader.bbox())
    assert fp.assumed_full_coverage and fp.justification


# ---- report partitioning -------------------------------------------

def _row_workflow(**kw):
    # spatial_q_value is present because a primary row must be corrected from
    # the block permutation; without it the row belongs in
    # not_spatially_corrected, which its own test covers.
    base = dict(feature="metal_density", layer="M8", scale_um=100.0,
                evidence_class="GDS_GEOMETRY", hypothesis_tier="tier1",
                scale_trustworthy=True, roc_auc=0.75, auc_ci_low=0.70,
                auc_ci_high=0.80, effect_size=0.5, fdr_q_value=0.001,
                spatial_q_value=0.01, n_case=100, n_control=100,
                effective_n=50.0, enrichment_top_10pct=2.0)
    base.update(kw)
    return base


def test_partition_separates_the_four_things_a_row_can_be():
    df = pd.DataFrame([
        _row_workflow(feature="perimeter_density"),
        _row_workflow(feature="distance_to_die_edge", evidence_class="PACKAGE_POSITION",
             hypothesis_tier="tier1_confounder"),
        _row_workflow(feature="largest_polygon", hypothesis_tier="exploratory",
             fdr_q_value=np.nan, spatial_q_value=np.nan),
        _row_workflow(feature="metal_density", scale_um=25.0, scale_trustworthy=False),
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
    Feature names here are real registered ones because an unregistered
    feature cannot be primary at all -- which its own test covers.
    """
    saturated, well_powered = "metal_density", "perimeter_density"
    df = pd.DataFrame([
        _row_workflow(feature=saturated, roc_auc=1.0, effect_size=1.0,
             auc_ci_low=1.0, auc_ci_high=1.0, n_case=35, n_control=1,
             fdr_q_value=0.21),
        _row_workflow(feature=well_powered, roc_auc=0.79, effect_size=0.58,
             auc_ci_low=0.69, auc_ci_high=0.88, n_case=108, n_control=36),
    ])
    parts = report.partition(df)
    assert list(parts["primary"].feature) == [well_powered]
    assert list(parts["underpowered"].feature) == [saturated]


def test_a_wide_interval_does_not_outrank_a_tight_one():
    wide, tight = "metal_density", "perimeter_density"
    df = pd.DataFrame([
        _row_workflow(feature=wide, roc_auc=0.85, effect_size=0.70,
             auc_ci_low=0.52, auc_ci_high=0.99),
        _row_workflow(feature=tight, roc_auc=0.78, effect_size=0.56,
             auc_ci_low=0.71, auc_ci_high=0.84),
    ])
    assert list(report.partition(df)["primary"].feature) == [tight, wide]


def test_an_interval_straddling_chance_guarantees_nothing():
    """It must not outrank a result whose interval excludes 0.5.

    Ranking on the nearer endpoint alone rewards an interval that is wide on
    both sides. On a three-die run that put a feature at AUC 0.512 with
    q = 0.94 above the driver at AUC 0.698 with q = 0.0008, because 0.512's
    lower end sat marginally further from 0.5 than the driver's did.
    """
    straddles, driver = "metal_density", "perimeter_density"
    df = pd.DataFrame([
        _row_workflow(feature=straddles, roc_auc=0.512, effect_size=0.024,
             auc_ci_low=0.3625, auc_ci_high=0.6522, fdr_q_value=0.94),
        _row_workflow(feature=driver, roc_auc=0.698, effect_size=0.396,
             auc_ci_low=0.6306, auc_ci_high=0.7774, fdr_q_value=0.0008),
    ])
    primary = report.partition(df)["primary"]
    assert list(primary.feature) == [driver, straddles]
    assert primary.set_index("feature").loc[
        straddles, "conservative_effect"] == 0.0


def test_a_protective_association_is_ranked_on_its_own_merits():
    """An interval entirely below chance excludes no-effect just as firmly.

    Zahedmanesh and Vanstreels (2019) show a stiff top group lowering the
    crack driving force beneath it, so a feature associating with fewer
    failures is a result, not a non-result.
    """
    protective, weak = "metal_density", "perimeter_density"
    df = pd.DataFrame([
        _row_workflow(feature=protective, roc_auc=0.28, effect_size=-0.44,
             auc_ci_low=0.20, auc_ci_high=0.38),
        _row_workflow(feature=weak, roc_auc=0.55, effect_size=0.10,
             auc_ci_low=0.51, auc_ci_high=0.60),
    ])
    assert list(report.partition(df)["primary"].feature) == [protective, weak]


def test_summary_states_what_the_result_is_not(tmp_path):
    df = pd.DataFrame([_row_workflow()])
    paths = report.write_reports(df, tmp_path, metadata={
        "uncontrolled_confounding": ["no bump layer supplied"]})
    text = (tmp_path / "reports" / "README.md").read_text()
    assert "not a causal claim" in text
    assert "not a design rule" in text
    assert "no bump layer supplied" in text
    assert set(paths) >= {"primary", "confounders", "exploratory",
                          "unsupported_scale", "underpowered",
                          "not_spatially_corrected", "not_traceable",
                          "summary"}


def test_empty_associations_partition_without_raising():
    parts = report.partition(pd.DataFrame())
    assert all(len(v) == 0 for v in parts.values())
    assert "no primary hypothesis survived" in report.format_primary(pd.DataFrame())

# ----------------------------------------------------------------------
# test_conditions_and_budget.py
# ----------------------------------------------------------------------
"""Package conditions actually controlling, and the budget that makes a
correction reachable.

Each of these was declared and then not used: conditions written into a
manifest that nothing consumed, a cross-stratum summary quoting the naive
q-value the primary table had stopped using, and a permutation count too small
for the family it corrects.
"""
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

def _row_conditions(feature, **kw):
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
        _row_conditions("perimeter_density"),
        _row_conditions("metal_density", spatial_q_value=1.0, roc_auc=0.51,
             auc_ci_low=0.45, auc_ci_high=0.57, effect_size=0.02),
        _row_conditions("via_density", spatial_q_value=0.03, auc_ci_low=0.49,
             auc_ci_high=0.88, effect_size=0.40),
    ])
    parts = report.partition(df)
    assert set(parts["primary"].feature) == {"perimeter_density",
                                             "metal_density", "via_density"}
    assert list(parts["supported"].feature) == ["perimeter_density"]


def test_the_console_shows_findings_and_says_so_when_there_are_none():
    from collective import exposure as report

    df = pd.DataFrame([_row_conditions("metal_density", spatial_q_value=1.0,
                            auc_ci_low=0.45, auc_ci_high=0.57)])
    text = report.format_primary(df)
    assert "no supported finding" in text
    assert "1 hypotheses were tested" in text


def test_the_summary_distinguishes_the_two(tmp_path):
    from collective import exposure as report

    report.write_reports(pd.DataFrame([_row_conditions("perimeter_density")]), tmp_path)
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
