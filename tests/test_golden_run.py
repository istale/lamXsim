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
import json
from pathlib import Path

import pandas as pd
import pytest

from lamxsim import atlas
from lamxsim.study import StudyManifest

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
