"""Validation of the Calibre edge-band approximation, measured against exact
edge lengths from the geometry engine.

The band trick turns a length density into an area density so that Calibre's
native moving-window DENSITY can compute it. These tests fix the two things
that decide whether the resulting numbers are usable, so a change to either
assumption fails here rather than silently in a full-chip run.
"""
import klayout.db as db
import pytest

from collective.calibre import EPS_WIDTH_FRACTION, CalibreLayer
from collective import geometry as corners
from collective import layout as synth
from collective.layout import LayerSpec, LayoutReader

DBU = 0.001
M = LayerSpec("M", 8, 0)


def _u(x):
    return int(round(x / DBU))


def _lines(tmp_path, pitch, density=0.5, tile=20.0):
    sl = synth.SynthLayout()
    synth.lines(sl, 8, 0, 0, tile, tile, pitch=pitch, density=density)
    path = tmp_path / f"p{pitch}.gds"
    sl.write(str(path))
    return LayoutReader(str(path)).region(M)


def _inside_band_perimeter(region, eps):
    band = region - region.sized(-_u(eps))  # METAL NOT (SIZE METAL BY -eps)
    return band.area() * DBU * DBU / eps


def _straddle_band_perimeter(region, eps):
    band = region.sized(_u(eps)) - region.sized(-_u(eps))
    return band.area() * DBU * DBU / (2 * eps)


def _exact(region):
    return region.edges().length() * DBU


@pytest.mark.parametrize("pitch", [1.0, 0.5, 0.2, 0.1])
def test_corner_corrected_band_is_exact(tmp_path, pitch):
    """P = band_area/eps + eps*(n_convex - n_concave) recovers the true length."""
    r = _lines(tmp_path, pitch)
    eps_dbu = max(int((pitch * 0.5 * EPS_WIDTH_FRACTION) / DBU), 1)
    got = corners.corrected_band_perimeter(r, eps_dbu, DBU)
    assert got == pytest.approx(_exact(r), rel=1e-9)


@pytest.mark.parametrize("builder,kw", [
    (synth.staircase_lines, dict(pitch=1.0, density=0.4, step=2.0)),
    (synth.segmented_lines, dict(pitch=1.0, density=0.4, seg_len=3.0, gap=1.0)),
])
def test_corner_correction_holds_for_concave_and_tip_rich_geometry(tmp_path, builder, kw):
    """The uncorrected band is worst exactly where terminations are dense."""
    sl = synth.SynthLayout()
    builder(sl, 8, 0, 0, 50, 50, **kw)
    path = tmp_path / "g.gds"
    sl.write(str(path))
    r = LayoutReader(str(path)).region(M)

    eps_dbu = max(int(0.4 * EPS_WIDTH_FRACTION / DBU), 1)
    eps = eps_dbu * DBU
    exact = _exact(r)
    raw = (r - r.sized(-eps_dbu)).area() * DBU * DBU / eps
    corrected = corners.corrected_band_perimeter(r, eps_dbu, DBU)

    assert corrected == pytest.approx(exact, rel=1e-9)
    assert abs(raw - exact) / exact > 0.002, (
        "uncorrected band happened to be exact here; the correction test is vacuous"
    )


def test_uncorrected_band_error_scales_with_corner_density(tmp_path):
    """Documents why the correction is worth wiring in rather than ignoring."""
    sl = synth.SynthLayout()
    synth.segmented_lines(sl, 8, 0, 0, 50, 50, pitch=1.0, density=0.4,
                          seg_len=3.0, gap=1.0)
    path = tmp_path / "tips.gds"
    sl.write(str(path))
    r = LayoutReader(str(path)).region(M)
    eps_dbu = int(0.1 / DBU)
    raw = (r - r.sized(-eps_dbu)).area() * DBU * DBU / (eps_dbu * DBU)
    assert (_exact(r) - raw) / _exact(r) > 0.05


def test_band_collapses_once_eps_reaches_half_the_width(tmp_path):
    """The failure is a cliff, not a drift, and it is silent -- hence the guard.

    A negative size of eps >= width/2 erases the line entirely, so the whole
    conductor becomes 'band' and the reported perimeter drops by tens of
    percent with nothing in the output to indicate it.
    """
    r = _lines(tmp_path, 0.2)          # width 0.1um
    exact = _exact(r)
    safe = _inside_band_perimeter(r, 0.1 * EPS_WIDTH_FRACTION)
    over = _inside_band_perimeter(r, 0.1)   # eps == width
    assert abs(safe - exact) / exact < 0.01
    assert (exact - over) / exact > 0.20


def test_inside_band_beats_straddle_band_at_a_window_edge(tmp_path):
    """A window border on a metal edge halves the straddle band, not the inside one.

    The bar exactly fills the window in y, so its top and bottom edges lie on
    the window border. True boundary inside the window is 15 + 15 + 10 = 40um.
    """
    r = db.Region(db.Box(0, 0, _u(30), _u(10)))
    r.merge()
    win = db.Region(db.Box(0, 0, _u(15), _u(10)))
    eps = 0.02

    inside = (r - r.sized(-_u(eps)))
    straddle = (r.sized(_u(eps)) - r.sized(-_u(eps)))

    p_inside = (inside & win).area() * DBU * DBU / eps
    p_straddle = (straddle & win).area() * DBU * DBU / (2 * eps)

    assert p_inside == pytest.approx(40.0, rel=0.01)
    assert p_straddle < 25.0, "straddle band should undercount here; if not, re-derive"


def test_generated_eps_keeps_a_factor_two_margin():
    layer = CalibreLayer("M8", 8, 0, min_width_um=0.2)
    assert layer.eps_um < layer.min_width_um / 2
    assert (layer.min_width_um / 2) / layer.eps_um == pytest.approx(2.0)


def test_corner_correction_is_exact_on_a_shape_with_a_hole(tmp_path):
    """A ring exercises the hole branch of the corner classification.

    Every other benchmark shape here is simply connected, so a hole ring whose
    corner types were inverted would pass all of them. On this ring the wrong
    classification gives 8 convex and 0 concave instead of 4 and 4, and the
    correction term is off by 8*eps.
    """
    sl = synth.SynthLayout()
    synth.bench_ring(sl, 8, outer=60.0, wall=10.0)
    path = tmp_path / "ring.gds"
    sl.write(str(path))
    r = LayoutReader(str(path)).region(M)

    n_convex, n_concave = corners.counts(r)
    assert (n_convex, n_concave) == (4, 4)

    eps_dbu = int(0.1 / DBU)
    assert corners.corrected_band_perimeter(r, eps_dbu, DBU) == pytest.approx(
        _exact(r), rel=1e-9)
