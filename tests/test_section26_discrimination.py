"""Spec section 26: the extractor must not degenerate into a metal-density detector.

Each test builds two patterns with matched metal density and one deliberately
different geometric property, then requires the extractor to separate them on
that property while agreeing on density.
"""
import numpy as np
import pytest

from collective.geometry import GeometryExtractor
from collective import layout as synth
from collective.layout import LayerSpec, LayoutReader

M8 = LayerSpec("M8", 8, 0)
M7 = LayerSpec("M7", 7, 0)


def _measure(tmp_path, builder, layer=M8, **kw):
    sl = builder(**kw)
    path = tmp_path / "pair.gds"
    sl.write(str(path))
    ex = GeometryExtractor(LayoutReader(str(path)))
    rois = synth.pair_rois(kw.get("tile", 100.0))
    return {k: ex.extract_roi(layer, *box) for k, box in rois.items()}


def test_same_density_different_perimeter(tmp_path):
    f = _measure(tmp_path, synth.pair_density_vs_perimeter)
    assert f["A"]["metal_density"] == pytest.approx(f["B"]["metal_density"], abs=1e-6)
    ratio = f["B"]["perimeter_density"] / f["A"]["perimeter_density"]
    assert ratio > 5.0, f"perimeter must separate the pair, got ratio {ratio:.2f}"


def test_same_density_different_orientation(tmp_path):
    """Orientation differs; density and perimeter must not."""
    f = _measure(tmp_path, synth.pair_density_vs_orientation)
    assert f["A"]["metal_density"] == pytest.approx(f["B"]["metal_density"], abs=1e-6)
    # This pair is the control for the perimeter test: rotating the lines
    # changes neither scalar, so a perimeter difference here would mean the
    # extractor is sensitive to orientation through the wrong channel.
    assert f["A"]["perimeter_density"] == pytest.approx(
        f["B"]["perimeter_density"], rel=0.02)


def test_perimeter_alone_cannot_see_line_termination(tmp_path):
    """Perimeter is a poor proxy for terminations, which is why section 4D exists.

    Chopping a line into segments removes long-edge length at the same rate it
    adds end-cap length, so perimeter density barely moves even when the
    termination count rises by an order of magnitude. Recording the size of
    that blind spot is what justifies line_end_density as a separate
    first-class feature rather than something derived from perimeter.
    """
    f = _measure(tmp_path, synth.pair_density_vs_lineend)
    assert f["A"]["metal_density"] == pytest.approx(
        f["B"]["metal_density"], rel=0.05)
    ratio = f["B"]["perimeter_density"] / f["A"]["perimeter_density"]
    assert ratio < 1.10, (
        "perimeter unexpectedly separates the termination pair; if this starts "
        "passing, re-derive whether line_end_density is still independent"
    )


def test_extractor_distinguishes_line_termination(tmp_path):
    """Spec section 26: separate 'same density, different termination'.

    Perimeter cannot do this (see the test above), so line_end_density is what
    makes the pair separable at all.
    """
    f = _measure(tmp_path, synth.pair_density_vs_lineend)
    assert f["A"]["metal_density"] == pytest.approx(
        f["B"]["metal_density"], rel=0.05)
    ratio = f["B"]["line_end_density"] / max(f["A"]["line_end_density"], 1e-12)
    assert ratio > 5.0, (
        f"line_end_density separated the pair by only {ratio:.1f}x while the "
        "segmented half has an order of magnitude more terminations"
    )


def test_same_density_different_corner_content(tmp_path):
    f = _measure(tmp_path, synth.pair_density_vs_corner)
    assert f["A"]["metal_density"] == pytest.approx(
        f["B"]["metal_density"], rel=0.25)
    assert f["B"]["perimeter_density"] > f["A"]["perimeter_density"]


def test_cross_layer_alignment_same_per_layer_density(tmp_path):
    """Top-layer measurements alone cannot see an underlying-layer rotation."""
    f8 = _measure(tmp_path, synth.pair_crosslayer_alignment, layer=M8)
    f7 = _measure(tmp_path, synth.pair_crosslayer_alignment, layer=M7)
    assert f8["A"]["metal_density"] == pytest.approx(f8["B"]["metal_density"], abs=1e-6)
    assert f7["A"]["metal_density"] == pytest.approx(f7["B"]["metal_density"], abs=1e-6)
    # The two halves are genuinely different layouts, so a cross-layer feature
    # is required to tell them apart. Recording that here keeps the section 7
    # gap explicit until cross-layer features land in Phase 3.
    per_layer_identical = (
        f8["A"]["perimeter_density"] == pytest.approx(f8["B"]["perimeter_density"], rel=0.02)
        and f7["A"]["perimeter_density"] == pytest.approx(f7["B"]["perimeter_density"], rel=0.02)
    )
    assert per_layer_identical, (
        "per-layer features should be blind to this difference; if they are not, "
        "the cross-layer test pair is not isolating alignment"
    )
