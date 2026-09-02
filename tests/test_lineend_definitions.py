"""Scoring of the candidate line-end definitions (spec section 4D).

Line ends are the one tier-1 feature with no self-evident definition on merged
geometry, so the definition is chosen by measurement rather than by argument.
Each benchmark pattern carries the termination count implied by its
construction; a definition is judged by how exactly it recovers those counts.
"""
import numpy as np
import pytest

from collective import geometry as lineends
from collective import layout as synth
from collective.layout import LayerSpec, LayoutReader

DBU = 0.001
M = LayerSpec("M", 8, 0)
W_MAX = int(1.5 / DBU)          # benchmark lines are 1.0 um wide


def _region(tmp_path, name):
    sl = synth.SynthLayout()
    truth = synth.LINE_END_BENCH[name](sl, 8)
    path = tmp_path / f"{name.replace(' ', '_')}.gds"
    sl.write(str(path))
    return LayoutReader(str(path)).region(M), truth


@pytest.mark.parametrize("name", list(synth.LINE_END_BENCH))
def test_recommended_definition_is_exact_on_every_pattern(tmp_path, name):
    region, truth = _region(tmp_path, name)
    got = len(lineends.detect(region, W_MAX))
    assert got == truth, f"{name}: got {got}, construction implies {truth}"


def test_bare_cap_rule_floods_on_dummy_fill(tmp_path):
    """Why the elongation guard is not optional.

    Every side of an isolated fill square is a short edge with a convex corner
    at each end, so the direct reading of "terminated tip" turns a 36-square
    fill array into 144 line ends.
    """
    region, truth = _region(tmp_path, "dummy fill array")
    assert truth == 0
    assert len(lineends.detect_cap(region, W_MAX)) == 144
    assert len(lineends.detect(region, W_MAX)) == 0


def test_antiparallel_flank_condition_adds_nothing(tmp_path):
    """D3 costs an extra SVRF condition and never changes the answer.

    On Manhattan rings the antiparallel-flank test is implied by requiring a
    convex corner at both ends of the cap.
    """
    rng = np.random.default_rng(0)
    for trial in range(25):
        sl = synth.SynthLayout()
        for _ in range(rng.integers(5, 30)):
            x0, y0 = rng.uniform(0, 60, 2)
            length, width = rng.uniform(1, 25), rng.uniform(0.4, 2.5)
            if rng.random() < 0.5:
                sl.add_box(8, x0, y0, x0 + length, y0 + width)
            else:
                sl.add_box(8, x0, y0, x0 + width, y0 + length)
        path = tmp_path / f"r{trial}.gds"
        sl.write(str(path))
        r = LayoutReader(str(path)).region(M)
        w = int(3.0 / DBU)
        assert (len(lineends.detect_aspect(r, w, 1.5))
                == len(lineends.detect_flanked(r, w, 1.5)))


@pytest.mark.parametrize("aspect", [1.2, 1.5, 2.0])
def test_aspect_plateau_is_stable(tmp_path, aspect):
    """The recommended default sits mid-plateau, not on an edge."""
    total = 0
    for name in synth.LINE_END_BENCH:
        region, truth = _region(tmp_path, name)
        total += abs(len(lineends.detect_aspect(region, W_MAX, aspect)) - truth)
    assert total == 0


@pytest.mark.parametrize("aspect,expect_wrong", [(1.0, True), (10.0, True)])
def test_aspect_fails_outside_the_plateau(tmp_path, aspect, expect_wrong):
    """Documents where the parameter stops working, in both directions."""
    total = 0
    for name in synth.LINE_END_BENCH:
        region, truth = _region(tmp_path, name)
        total += abs(len(lineends.detect_aspect(region, W_MAX, aspect)) - truth)
    assert (total > 0) is expect_wrong


def test_w_max_separates_routing_from_a_wide_strap(tmp_path):
    """w_max is a step, not a dial: flat until it reaches the strap width."""
    sl = synth.SynthLayout()
    for i in range(6):
        sl.add_box(8, 0, i * 3.0, 40.0, i * 3.0 + 1.0)   # 1um signal lines
    sl.add_box(8, 0, 40.0, 120.0, 52.0)                   # 12um power strap
    path = tmp_path / "strap.gds"
    sl.write(str(path))
    r = LayoutReader(str(path)).region(M)

    for w in (1.5, 3.0, 8.0):
        assert len(lineends.detect(r, int(w / DBU))) == 12   # signal tips only
    assert len(lineends.detect(r, int(15.0 / DBU))) == 14    # strap ends included
