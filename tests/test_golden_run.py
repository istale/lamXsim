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
from pathlib import Path

import pandas as pd
import pytest

from lamxsim import atlas
from lamxsim.study import StudyManifest

GOLDEN = Path(__file__).parent / "golden"
COLUMNS = ["channel", "layer", "scale_um", "x_um", "y_um",
           "percentile_in_die", "inputs_used"]


def _rebuild() -> pd.DataFrame:
    manifest = StudyManifest.load(GOLDEN / "golden_manifest.yaml")
    result = atlas.build(str(GOLDEN / "golden_die.gds"), manifest)
    return result.candidates[COLUMNS].sort_values(COLUMNS).reset_index(drop=True)


def test_the_frozen_study_reaches_the_same_conclusion():
    expected = pd.read_csv(GOLDEN / "golden_candidates.csv")
    actual = _rebuild()

    assert len(actual) == len(expected), (
        f"the candidate count moved from {len(expected)} to {len(actual)}. "
        "If that is intended, regenerate tests/golden/golden_candidates.csv "
        "and say in the commit which change moved it and why.")

    pd.testing.assert_frame_equal(
        actual, expected.sort_values(COLUMNS).reset_index(drop=True),
        check_dtype=False, atol=1e-6)


def test_the_channel_breakdown_is_stable():
    """A shift between channels is a changed conclusion even at equal totals."""
    expected = pd.read_csv(GOLDEN / "golden_candidates.csv")
    actual = _rebuild()
    assert (actual.groupby("channel").size().to_dict()
            == expected.groupby("channel").size().to_dict())


def test_die_scoped_channels_stay_die_scoped():
    """Scoring a shared feature per layer would silently multiply candidates."""
    actual = _rebuild()
    for channel in ("cross_layer_mismatch", "pi_opening_proximity"):
        rows = actual[actual.channel == channel]
        if not rows.empty:
            assert set(rows.layer) == {"-"}


def test_the_golden_inputs_are_present():
    assert (GOLDEN / "golden_die.gds").exists()
    assert (GOLDEN / "golden_manifest.yaml").exists()
    assert (GOLDEN / "golden_candidates.csv").exists()
