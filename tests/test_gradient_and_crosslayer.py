"""Gradient (spec section 5) and cross-layer (spec section 7) features.

Both sections exist because a layout cannot be reduced to a set of independent
per-layer scalar maps. These tests build dies where that reduction provably
loses the driver, so a regression that quietly drops either family fails here.
"""
import numpy as np
import pytest
from scipy import stats

from collective import workflow as pipeline
from collective.geometry import LayerStack, crosslayer_extract as xl_extract
from collective.geometry import GeometryExtractor
from collective.geometry import gradients, interior_mask
from collective.geometry import build_grid
from collective.geometry import OrientationExtractor
from collective import labels as position
from collective.labels import failures_from_driver
from collective import layout as synth
from collective.layout import BBox, LayerSpec, LayoutReader

M8 = LayerSpec("M8", 8, 0)
M7 = LayerSpec("M7", 7, 0)


# ---- gradients ----------------------------------------------------

def test_gradient_is_in_physical_units():
    """dQ_dx is per micrometre, so scales stay comparable."""
    grid = build_grid(BBox(0, 0, 2000, 2000), 100.0)
    x = np.array([c.x_center for c in grid.cells])
    g = gradients(0.001 * x, grid, "Q", drop_boundary=False)
    assert np.nanmean(g["Q_dx"]) == pytest.approx(0.001, rel=1e-9)
    assert np.nanmean(np.abs(g["Q_dy"])) == pytest.approx(0.0, abs=1e-12)


def test_one_sided_boundary_gradients_fake_a_die_edge_effect():
    """Why the boundary ring is dropped rather than filled.

    On a field with no structure at all, one-sided differences at the die edge
    are systematically larger than centred ones, and the cells carrying them
    form a ring -- the shape of distance_to_die_edge. The artifact is
    significant on pure noise.
    """
    grid = build_grid(BBox(0, 0, 2000, 2000), 100.0)
    noise = np.random.default_rng(0).normal(size=len(grid))
    d_edge = position.position_extract(grid, grid.bbox)["distance_to_die_edge"]

    kept = gradients(noise, grid, "N", drop_boundary=False)["N_grad_mag"]
    mask = interior_mask(grid)
    assert kept[~mask].mean() > kept[mask].mean() * 1.3

    _, p_biased = stats.spearmanr(kept, d_edge)
    assert p_biased < 0.05, "artifact not reproduced; the test would be vacuous"

    dropped = gradients(noise, grid, "N", drop_boundary=True)["N_grad_mag"]
    ok = ~np.isnan(dropped)
    _, p_clean = stats.spearmanr(dropped[ok], d_edge[ok])
    assert p_clean > 0.05


def test_pipeline_recovers_a_gradient_driver_the_value_cannot_see(tmp_path):
    """Spec section 5: the gradient must not be assumed secondary to the value.

    Density varies sinusoidally, so its value and its gradient are orthogonal.
    Failures are driven by the gradient alone.
    """
    path = str(tmp_path / "grad.gds")
    synth.gradient_driver_die(path)
    reader = LayoutReader(path)
    grid = build_grid(reader.bbox(), 100.0)
    feats = GeometryExtractor(reader, line_end_w_max_um=4.0).extract(M8, grid)
    grad = gradients(feats["metal_density"], grid, "metal_density",
                     drop_boundary=False)

    assert abs(np.corrcoef(feats["metal_density"],
                           grad["metal_density_grad_mag"])[0, 1]) < 0.05

    fs = failures_from_driver(grad["metal_density_grad_mag"], grid,
                              n_failures=200, strength=2.5, seed=3,
                              position_sigma_um=5.0)
    res = pipeline.run(path, fs, layer=M8, scales_um=(50, 100),
                       n_permutations=0, line_end_w_max_um=4.0, seed=1)
    a = res.associations
    grad_auc = a[a.feature == "metal_density_grad_mag"].roc_auc.max()
    value = a[a.feature == "metal_density"]
    assert grad_auc > 0.70
    assert (value.roc_auc - 0.5).abs().max() < 0.10, (
        "the absolute value should be uninformative on this die"
    )


# ---- orientation --------------------------------------------------

def test_orientation_is_length_weighted(tmp_path):
    """One long line must outweigh one short stub in the same direction."""
    sl = synth.SynthLayout()
    sl.add_box(8, 0, 0, 100, 1)      # long horizontal line
    sl.add_box(8, 0, 10, 1, 13)      # short vertical stub
    path = tmp_path / "ori.gds"
    sl.write(str(path))
    f = OrientationExtractor(LayoutReader(str(path))).extract_roi(
        M8, 0, 0, 100, 20)
    assert f["horizontal_fraction"] > 0.85
    assert f["orientation_anisotropy"] > 0.7


def test_orientation_separates_the_section26_pair(tmp_path):
    """Same density and same perimeter, orthogonal orientation."""
    sl = synth.pair_density_vs_orientation()
    path = tmp_path / "pair.gds"
    sl.write(str(path))
    ex = OrientationExtractor(LayoutReader(str(path)))
    rois = synth.pair_rois()
    a = ex.extract_roi(M8, *rois["A"])
    b = ex.extract_roi(M8, *rois["B"])
    assert a["orientation_anisotropy"] > 0.9
    assert b["orientation_anisotropy"] < -0.9


# ---- cross-layer --------------------------------------------------

def test_pair_features_preserve_layer_identity(tmp_path):
    path = str(tmp_path / "xl.gds")
    synth.crosslayer_driver_die(path)
    reader = LayoutReader(path)
    grid = build_grid(reader.bbox(), 100.0)
    geo = GeometryExtractor(reader, line_end_w_max_um=4.0)
    ori = OrientationExtractor(reader)
    per = {s.name: {**geo.extract(s, grid), **ori.extract(s, grid)}
           for s in (M8, M7)}
    xl = xl_extract(per, LayerStack(("M8", "M7")))
    assert "orientation_difference_M8_M7" in xl
    assert "orientation_mismatch_M8_M7" in xl
    assert not any(k == "orientation_difference" for k in xl), (
        "a layer-agnostic name would pool a shielding pair with a loading pair"
    )


def test_signed_difference_cannot_see_a_magnitude_driver(tmp_path):
    """Why both the signed and the absolute form are emitted.

    Both directions of disagreement sit at opposite ends of a signed feature,
    so an effect driven by how much two layers disagree collapses to chance
    on it while its own absolute value recovers the driver.
    """
    path = str(tmp_path / "xl.gds")
    synth.crosslayer_driver_die(path)
    reader = LayoutReader(path)
    grid = build_grid(reader.bbox(), 100.0)
    geo = GeometryExtractor(reader, line_end_w_max_um=4.0)
    ori = OrientationExtractor(reader)
    per = {s.name: {**geo.extract(s, grid), **ori.extract(s, grid)}
           for s in (M8, M7)}
    xl = xl_extract(per, LayerStack(("M8", "M7")))

    fs = failures_from_driver(xl["orientation_mismatch_M8_M7"], grid,
                              n_failures=180, strength=2.5, seed=7,
                              position_sigma_um=5.0)
    res = pipeline.run(path, fs, layers=[M8, M7], scales_um=(100,),
                       n_permutations=0, line_end_w_max_um=4.0, seed=2)
    a = res.associations.set_index("feature")

    mismatch = a.loc["orientation_mismatch_M8_M7", "roc_auc"]
    signed = a.loc["orientation_difference_M8_M7", "roc_auc"]
    best_per_layer = a[a.layer.isin(["M8", "M7"])].roc_auc
    best_per_layer = max(abs(best_per_layer - 0.5)) + 0.5

    assert mismatch > 0.75
    assert abs(signed - 0.5) < abs(mismatch - 0.5) * 0.6
    assert mismatch > best_per_layer + 0.15, (
        "cross-layer feature must beat every per-layer one on a die where the "
        "driver is defined only by the relationship between layers"
    )


def test_pair_selection_controls_the_hypothesis_budget():
    stack = LayerStack(tuple(f"M{i}" for i in range(12, 0, -1)))
    assert len(stack.pairs("all")) == 66
    assert len(stack.pairs("adjacent")) == 11
    assert len(stack.pairs("adjacent_and_top")) == 21
    assert (stack.hypothesis_count("adjacent_and_top", 6)
            < stack.hypothesis_count("all", 6) * 0.4)
