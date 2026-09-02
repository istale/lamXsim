"""Registration, the cost model and the bump-frame routing lever.

Folded from ``tests/test_registration.py``, ``tests/test_budget.py``, ``tests/test_bump_relative_routing.py``.
"""
from collective import layout as synth
from collective import workflow as budget
from collective import workflow as pipeline
from collective.geometry import OrientationExtractor
from collective.geometry import build_grid
from collective.geometry import bump_relative_extract as rel_extract
from collective.labels import FailureSet
from collective.labels import PackageLayers
from collective.labels import failures_from_driver
from collective.labels import package_context_extract as ctx_extract
from collective.layout import LayerSpec
from collective.layout import LayoutReader
from collective.study import StudyManifest
from collective.workflow import RegistrationError
from collective.workflow import Transform2D
from collective.workflow import fit
from collective.workflow import fit_transform
from collective.workflow import flag_outliers
from collective.workflow import register
from collective.workflow import robust_fit
from collective.workflow import scale_gate
from collective.workflow import select_model
from pathlib import Path
import numpy as np
import pandas as pd
import pytest


# ----------------------------------------------------------------------
# test_registration.py
# ----------------------------------------------------------------------
"""Registration of measured failures into the layout frame.

Registration decides which analysis scales carry information, so the tests
here are mostly about refusing to certify a scale the data cannot support.
"""
def _make(n, noise=0.0, rotation_deg=0.0, translation=(0.0, 0.0),
          scale=1.0, reflect=False, seed=0):
    rng = np.random.default_rng(seed)
    src = rng.uniform(200, 4800, (n, 2))
    th = np.radians(rotation_deg)
    r = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    if reflect:
        r = r @ np.diag([-1.0, 1.0])
    dst = src @ (scale * r).T + np.asarray(translation)
    if noise:
        dst = dst + rng.normal(0, noise, (n, 2))
    return src, dst


# ---- transform ----------------------------------------------------

@pytest.mark.parametrize("model", ["translation", "rigid", "similarity", "affine"])
def test_noiseless_transform_is_recovered(model):
    src, dst = _make(8, rotation_deg=0 if model == "translation" else 0.4,
                     translation=(120.0, -45.0))
    t = fit_transform(src, dst, model)
    assert np.abs(t.apply(src) - dst).max() < 1e-6


def test_reflection_is_reported_not_absorbed():
    """Backside imaging mirrors the frame; a good residual must not hide it."""
    src, dst = _make(8, rotation_deg=0.2, reflect=True)
    t = fit_transform(src, dst, "similarity", allow_reflection=True)
    assert t.is_reflection
    assert np.abs(t.apply(src) - dst).max() < 1e-6

    blocked = fit_transform(src, dst, "similarity", allow_reflection=False)
    assert not blocked.is_reflection
    assert np.abs(blocked.apply(src) - dst).max() > 100.0


def test_inverse_round_trips():
    src, dst = _make(6, rotation_deg=0.7, translation=(50.0, 80.0), scale=1.0004)
    t = fit_transform(src, dst, "similarity")
    assert np.abs(t.inverse().apply(t.apply(src)) - src).max() < 1e-6


# ---- honest error -------------------------------------------------

def test_exactly_determined_fit_reports_a_meaningless_zero_residual():
    """Three fiducials against a six-parameter model cannot reveal any error."""
    src, dst = _make(3, noise=8.0, seed=3)
    f = fit(src, dst, "affine")
    assert f.residual_dof == 0
    assert f.rms_um < 1e-6            # zero by construction, not by accuracy
    assert not f.is_determined
    assert any("zero by construction" in w for w in f.warnings)


def test_in_fit_residual_understates_error_when_dof_is_thin():
    """The reason position_sigma comes from leave-one-out, not from the fit.

    Averaged over draws rather than asserted on one: the shrinkage is a
    property of the estimator, and a single realisation is noisy. With four
    fiducials against a six-parameter model only two residual degrees of
    freedom remain, so the in-fit RMS is expected near
    ``sigma * sqrt(2/8) = 0.5 sigma``.
    """
    noise = 8.0
    in_fit, loo = [], []
    for seed in range(200):
        src, dst = _make(4, noise=noise, seed=seed)
        f = fit(src, dst, "affine")
        assert f.residual_dof == 2
        in_fit.append(f.rms_um)
        if np.isfinite(f.loo_rms_um):
            loo.append(f.loo_rms_um)

    assert np.mean(in_fit) < noise * 0.75, (
        "in-fit RMS should shrink well below the true noise at this dof")
    assert np.median(loo) > np.mean(in_fit) * 2.0, (
        "leave-one-out should expose the error the in-fit residual hides")

    src, dst = _make(4, noise=noise, seed=3)
    f = fit(src, dst, "affine")
    assert f.position_sigma_um == pytest.approx(f.loo_rms_um)


def test_error_estimate_converges_with_enough_fiducials():
    noise = 8.0
    src, dst = _make(30, noise=noise, seed=5)
    f = fit(src, dst, "similarity")
    assert f.rms_um == pytest.approx(noise, rel=0.5)
    assert f.loo_rms_um == pytest.approx(noise, rel=0.6)


def test_model_selection_prefers_prediction_over_fit():
    """A richer model always fits better and does not always predict better."""
    src, dst = _make(10, noise=12.0, rotation_deg=0.15,
                     translation=(300.0, -150.0), seed=11)
    best, rows = select_model(src, dst)
    table = {r["model"]: r for r in rows if "in_fit_rms_um" in r}
    assert table["affine"]["in_fit_rms_um"] <= table["rigid"]["in_fit_rms_um"]
    assert table["affine"]["leave_one_out_rms_um"] > table[best]["leave_one_out_rms_um"]
    assert best != "affine"


def test_outlier_fiducial_is_found_and_removing_it_helps():
    src, dst = _make(10, noise=12.0, rotation_deg=0.15, seed=11)
    dst[5] += np.array([400.0, -300.0])
    before = fit(src, dst, "rigid")
    assert flag_outliers(before)[5]

    after, keep, _ = robust_fit(src, dst)
    assert not keep[5]
    assert after.loo_rms_um < before.loo_rms_um / 3


# ---- gating -------------------------------------------------------

def _failure_set(n=5):
    return FailureSet(table=pd.DataFrame({
        "sample_id": [f"S{i}" for i in range(n)],
        "x_um": np.linspace(100, 900, n), "y_um": np.linspace(200, 800, n),
        "failure_type": "delamination", "confidence": 1.0,
        "position_sigma_um": np.nan}))


def test_registration_refuses_to_certify_an_under_determined_fit():
    src, dst = _make(3, noise=8.0, seed=3)
    with pytest.raises(RegistrationError, match="zero by construction|degrees of freedom"):
        register(_failure_set(), fit(src, dst, "affine"))


def test_registered_failures_carry_the_registration_uncertainty():
    src, dst = _make(12, noise=10.0, rotation_deg=0.2,
                     translation=(300.0, -150.0), seed=7)
    f = fit(src, dst, "rigid")
    out = register(_failure_set(), f)
    assert np.allclose(out.table["position_sigma_um"], f.position_sigma_um)
    assert out.min_trustworthy_scale_um() == pytest.approx(3 * f.position_sigma_um)
    assert "measured_x_um" in out.table
    assert (out.table["coord_frame"] == "layout").all()


def test_reported_and_registration_uncertainty_combine_in_quadrature():
    src, dst = _make(12, noise=10.0, seed=7)
    f = fit(src, dst, "rigid")
    fs = _failure_set()
    fs.table["position_sigma_um"] = 30.0
    out = register(fs, f)
    expected = np.hypot(30.0, f.position_sigma_um)
    assert np.allclose(out.table["position_sigma_um"], expected)
    assert out.position_sigma_um > 30.0


def test_scale_gate_rejects_scales_below_the_floor():
    src, dst = _make(12, noise=20.0, seed=7)
    f = fit(src, dst, "rigid")
    gate = scale_gate(f, [25, 50, 100, 250, 500, 1000])
    assert gate["min_trustworthy_scale_um"] == pytest.approx(3 * f.position_sigma_um)
    assert 25 in gate["rejected"]
    assert 250 in gate["trustworthy"]
    assert set(gate["trustworthy"]) & set(gate["rejected"]) == set()

# ----------------------------------------------------------------------
# test_budget.py
# ----------------------------------------------------------------------
"""Measuring what extraction costs, rather than guessing it.

The runtime question cannot be answered from this repository's synthetic
dies, and the numbers measured on them should not be shipped as a budget: a
production layout differs in polygon density, hierarchy depth and the fraction
of non-Manhattan geometry, and the projection is linear in a constant nobody
has measured for it. So the command measures on the user's own clip, and
these tests check that the arithmetic and the verdict are right rather than
that any particular number comes out.
"""
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

# ----------------------------------------------------------------------
# test_bump_relative_routing.py
# ----------------------------------------------------------------------
"""Routing orientation resolved against the package loading direction.

Rabie et al. (2018) recommend running the final metal diagonally under corner
bumps. That is directional, and no scalar distance to a bump can express it:
two cells the same distance from the same bump, one routed radially and one
diagonally, are identical in every other feature the engine has.
"""
M8 = LayerSpec("M8", 8, 0)
PKG = PackageLayers(bump=LayerSpec("B", 60, 0),
                    pi_opening=LayerSpec("PI", 61, 0),
                    crackstop=LayerSpec("CS", 62, 0))


# ---- axial orientation ---------------------------------------------

@pytest.mark.parametrize("vertical,expected_deg", [(False, 0.0), (True, 90.0)])
def test_routing_direction_is_recovered(tmp_path, vertical, expected_deg):
    sl = synth.SynthLayout()
    synth.lines(sl, 8, 0, 0, 60, 60, pitch=4.0, density=0.5, vertical=vertical)
    path = tmp_path / "d.gds"
    sl.write(str(path))
    f = OrientationExtractor(LayoutReader(str(path))).extract_roi(M8, 0, 0, 60, 60)
    assert np.degrees(f["routing_direction_rad"]) == pytest.approx(expected_deg, abs=0.5)
    assert f["orientation_coherence"] > 0.9


def test_two_orthogonal_populations_have_no_dominant_direction(tmp_path):
    """Averaging raw angles would put the mean perpendicular to both."""
    sl = synth.SynthLayout()
    synth.lines(sl, 8, 0, 0, 60, 30, pitch=4.0, density=0.5)
    synth.lines(sl, 8, 0, 30, 60, 60, pitch=4.0, density=0.5, vertical=True)
    path = tmp_path / "mix.gds"
    sl.write(str(path))
    f = OrientationExtractor(LayoutReader(str(path))).extract_roi(M8, 0, 0, 60, 60)
    assert f["orientation_coherence"] < 0.1


# ---- resolving against the bump frame -------------------------------

def test_alignment_and_diagonality_separate_the_three_cases():
    routing = np.radians([0.0, 45.0, 90.0, 135.0])
    radial = np.zeros(4)
    coherence = np.full(4, 0.9)
    f = rel_extract(routing, coherence, radial)

    assert f["routing_radial_alignment"][0] == pytest.approx(1.0)    # radial
    assert f["routing_radial_alignment"][2] == pytest.approx(-1.0)   # tangential
    assert f["routing_diagonality"][1] == pytest.approx(1.0)         # 45 degrees
    # 135 and 45 are the same axis relative to the radial direction.
    assert f["routing_diagonality"][3] == pytest.approx(1.0)
    assert f["routing_diagonality"][0] == pytest.approx(0.0)


def test_diagonality_is_its_own_feature_not_the_midpoint_of_alignment():
    """Radial and tangential both sit at zero diagonality, at opposite ends
    of alignment. A single axis could not give the diagonal case its own
    extreme, which is where the recommendation lives."""
    f = rel_extract(np.radians([0.0, 45.0, 90.0]), np.full(3, 0.9), np.zeros(3))
    assert f["routing_diagonality"][1] > f["routing_diagonality"][0]
    assert f["routing_diagonality"][1] > f["routing_diagonality"][2]
    assert f["routing_radial_alignment"][0] != pytest.approx(
        f["routing_radial_alignment"][2])


def test_a_window_without_a_dominant_direction_gets_no_angle():
    """Isotropic and deliberately diagonal sit at the same alignment."""
    f = rel_extract(np.array([0.0]), np.array([0.05]), np.array([0.0]))
    assert np.isnan(f["routing_vs_radial_angle_rad"][0])
    assert np.isnan(f["routing_diagonality"][0])


# ---- end to end ------------------------------------------------------

@pytest.fixture(scope="module")
def routing_die(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("rr") / "rr.gds")
    synth.radial_routing_die(path, die_um=3000.0, block_um=150.0, seed=41)
    reader = LayoutReader(path)
    grid = build_grid(reader.bbox(), 150.0)
    ori = OrientationExtractor(reader).extract(M8, grid)
    ctx = ctx_extract(grid, reader.bbox(), reader, PKG)
    rel = rel_extract(ori["routing_direction_rad"], ori["orientation_coherence"],
                      ctx["bump_radial_direction_rad"])
    return path, grid, rel


def test_the_die_varies_only_in_routing_direction(routing_die):
    from collective.geometry import GeometryExtractor

    path, grid, rel = routing_die
    geo = GeometryExtractor(LayoutReader(path),
                            line_rules={"M8": (0.5, 4.0)}).extract(M8, grid)
    assert geo["metal_density"].std() < 1e-6
    ok = np.isfinite(rel["routing_diagonality"])
    assert rel["routing_diagonality"][ok].std() > 0.2


def test_pipeline_finds_a_direction_driver_no_scalar_feature_can_see(routing_die):
    path, grid, rel = routing_die
    driver = np.nan_to_num(rel["routing_diagonality"], nan=0.0)
    fs = failures_from_driver(driver, grid, n_failures=200, strength=2.5,
                              seed=4, position_sigma_um=2.0)
    fs.table["position_sigma_um"] = 2.0
    res = pipeline.run(path, fs, layers=[M8], package_layers=PKG,
                       scales_um=(150,), n_permutations=0,
                       line_rules={"M8": (0.5, 4.0)}, seed=1)
    a = res.associations.set_index("feature")

    assert a.loc["routing_diagonality", "roc_auc"] > 0.70
    assert a.loc["routing_diagonality", "fdr_q_value"] < 0.01
    for scalar in ("metal_density", "perimeter_density", "corner_density",
                   "orientation_anisotropy"):
        assert abs(a.loc[scalar, "roc_auc"] - 0.5) < 0.10, (
            f"{scalar} should be blind to a pure direction driver"
        )


def test_bump_relative_routing_is_geometry_not_position(routing_die):
    """It needs a bump map to compute and is still a layout property.

    Classifying it as package position would put the designer's lever into
    the baseline that the lever is supposed to beat.
    """
    from collective.foundation import EvidenceClass
    from collective import geometry as bump_relative

    assert bump_relative.EVIDENCE_CLASS is EvidenceClass.GDS_GEOMETRY

    path, grid, rel = routing_die
    driver = np.nan_to_num(rel["routing_diagonality"], nan=0.0)
    fs = failures_from_driver(driver, grid, n_failures=150, strength=2.5,
                              seed=4, position_sigma_um=2.0)
    fs.table["position_sigma_um"] = 2.0
    res = pipeline.run(path, fs, layers=[M8], package_layers=PKG,
                       scales_um=(150,), n_permutations=0,
                       line_rules={"M8": (0.5, 4.0)}, seed=1)
    row = res.associations.set_index("feature").loc["routing_diagonality"]
    assert row["evidence_class"] == "GDS_GEOMETRY"
    assert row["layer"] == "M8"
