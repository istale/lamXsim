"""Measuring what extraction costs, rather than guessing it.

The runtime question cannot be answered from this repository's synthetic
dies, and the numbers measured on them should not be shipped as a budget: a
production layout differs in polygon density, hierarchy depth and the fraction
of non-Manhattan geometry, and the projection is linear in a constant nobody
has measured for it. So the command measures on the user's own clip, and
these tests check that the arithmetic and the verdict are right rather than
that any particular number comes out.
"""
import numpy as np
import pytest

from lamxsim import budget
from lamxsim.study import StudyManifest

from pathlib import Path

GOLDEN = Path(__file__).parent / "golden"


def _manifest():
    return StudyManifest.load(str(GOLDEN / "golden_manifest.yaml"))


def test_polygons_are_counted_merged_on_the_analysed_layers():
    """Merged, because that is what the extractors see.

    A layer drawn as ten thousand abutting rectangles is one polygon to them,
    so counting drawn shapes would overstate the cost by whatever the merge
    removes -- and the projection is linear in this count.
    """
    manifest = _manifest()
    total, per_layer = budget.count_polygons(str(GOLDEN / "golden_die.gds"),
                                             manifest)
    assert total == sum(per_layer.values())
    assert {"M8", "M7", "V7"} <= set(per_layer)
    # The package layers the manifest names are counted too: they are
    # extracted, so they cost.
    assert {"BUMP", "PI", "CS"} <= set(per_layer)
    assert all(v > 0 for v in per_layer.values())


def test_the_projection_is_linear_in_what_was_measured():
    measurement = budget.Measurement(
        polygons=100_000, cells=1_000, scales=2, seconds=10.0,
        peak_rss_bytes=3_000_000_000, baseline_rss_bytes=1_000_000_000)

    assert measurement.seconds_per_polygon_scale == pytest.approx(
        10.0 / (100_000 * 2))
    assert measurement.bytes_per_polygon == pytest.approx(2e9 / 100_000)

    projected = measurement.project(1_000_000_000, 2)
    assert projected["seconds"] == pytest.approx(10.0 * 10_000)
    # Baseline plus the marginal cost, not the whole peak scaled: the process
    # does not pay its own startup ten thousand times.
    assert projected["peak_rss_gb"] == pytest.approx(
        (1e9 + 2e4 * 1e9) / 1e9)


def test_a_measurement_on_the_golden_die_reports_both_constants():
    measurement = budget.measure(str(GOLDEN / "golden_die.gds"), _manifest())
    assert measurement.polygons > 0
    assert measurement.scales == len(_manifest().scales_um)
    assert measurement.seconds > 0
    assert measurement.seconds_per_polygon_scale > 0
    # Memory is allowed to come out at zero on a run small enough that the
    # interpreter's own high-water mark never moves; it must not come out
    # negative, which a signed subtraction of two peaks can.
    assert measurement.bytes_per_polygon >= 0


def test_peak_memory_is_read_in_the_units_the_platform_uses():
    """ru_maxrss is bytes on macOS and kilobytes on Linux.

    That is a factor of 1024 in the headline number of the whole command, and
    it cannot be sniffed from the value: a small process on Linux and a large
    one on macOS report the same figure.
    """
    import sys

    value = budget._peak_rss_bytes()
    assert value > 0
    # A Python process with numpy and KLayout loaded is somewhere between
    # 20 MB and 20 GB on any platform. The wrong unit puts it outside that by
    # three orders of magnitude in one direction or the other.
    assert 2e7 < value < 2e10, (
        f"{value} bytes on {sys.platform} is not a plausible peak RSS; the "
        "platform unit is probably wrong")
