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


def test_time_growth_is_fitted_rather_than_assumed_linear():
    """Assuming 1.0 is wrong in the direction that matters.

    The windowed extractors clip the layer once per grid row and again per
    window, so the work is rows times polygons and both grow with die area.
    Measured across a sixty-fourfold range the cost rose 4.8x, then 5.3x,
    then 5.9x per fourfold rise in polygons. A linear projection from a small
    clip understates a full chip by more than an order of magnitude.
    """
    def m(polygons, seconds):
        return budget.Measurement(polygons=polygons, cells=1, scales=1,
                                  seconds=seconds, peak_rss_bytes=2_000_000_000,
                                  baseline_rss_bytes=1_000_000_000)

    # A clean N^1.5 series must come back as 1.5.
    series = [m(1_000, 1.0), m(4_000, 8.0), m(16_000, 64.0)]
    exponent, how = budget.fit_exponent(series)
    assert exponent == pytest.approx(1.5, abs=0.01)
    assert "3 clips" in how and "16x" in how

    # One clip cannot be fitted, and the caller is told the number is a lower
    # bound rather than being handed a silent 1.0.
    lone, note = budget.fit_exponent([m(1_000, 1.0)])
    assert lone == 1.0
    assert "lower bound" in note


def test_projection_uses_the_exponent_and_keeps_memory_linear():
    measurement = budget.Measurement(
        polygons=1_000_000, cells=1_000, scales=1, seconds=100.0,
        peak_rss_bytes=3_000_000_000, baseline_rss_bytes=1_000_000_000)

    linear = measurement.project(100_000_000, 1, exponent=1.0)
    superlinear = measurement.project(100_000_000, 1, exponent=1.3)

    assert linear["seconds"] == pytest.approx(100.0 * 100)
    assert superlinear["seconds"] == pytest.approx(100.0 * 100 ** 1.3)
    assert superlinear["seconds"] > 3 * linear["seconds"], (
        "the exponent has to change the answer by enough to change a decision")

    # Memory does not take the exponent: it is the merged layers held at once.
    assert linear["peak_rss_gb"] == pytest.approx(
        superlinear["peak_rss_gb"])
    assert linear["peak_rss_gb"] == pytest.approx((1e9 + 2e3 * 1e8) / 1e9)
