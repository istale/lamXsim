"""Registration of measured failures into the layout frame.

Registration decides which analysis scales carry information, so the tests
here are mostly about refusing to certify a scale the data cannot support.
"""
import numpy as np
import pandas as pd
import pytest

from lamxsim.labels.failure import FailureSet
from lamxsim.registration.apply import RegistrationError, register, scale_gate
from lamxsim.registration.fit import fit, flag_outliers, robust_fit, select_model
from lamxsim.registration.transform import Transform2D, fit_transform


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
