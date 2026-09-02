"""Window features: the grid, and everything measured on it.

Folded from ``tests/test_perimeter_clipping.py``, ``tests/test_section26_discrimination.py``, ``tests/test_lineend_definitions.py``, ``tests/test_structures.py``, ``tests/test_gradient_and_crosslayer.py``.
"""
from collective import geometry as lineends
from collective import labels as position
from collective import layout as synth
from collective import workflow as pipeline
from collective.geometry import GeometryExtractor
from collective.geometry import LayerStack
from collective.geometry import OrientationExtractor
from collective.geometry import StructureExtractor
from collective.geometry import build_grid
from collective.geometry import crosslayer_extract as xl_extract
from collective.geometry import gradients
from collective.geometry import interior_mask
from collective.labels import failures_from_driver
from collective.layout import BBox
from collective.layout import LayerSpec
from collective.layout import LayoutReader
from scipy import stats
import klayout.db as db
import numpy as np
import pytest


# ----------------------------------------------------------------------
# test_perimeter_clipping.py
# ----------------------------------------------------------------------
"""Window-local perimeter must count metal boundary only, not the window cut.
"""
M8 = LayerSpec("M8", 8, 0)


def test_window_cut_is_not_counted_as_metal_boundary(tmp_path):
    sl = synth.SynthLayout()
    sl.add_box(8, 0, 0, 30, 10)          # single 30x10 um bar
    path = tmp_path / "bar.gds"
    sl.write(str(path))
    ex = GeometryExtractor(LayoutReader(str(path)))

    # Window covers the left half. True metal boundary inside it is
    # 15 (bottom) + 15 (top) + 10 (left cap) = 40 um. Clipping the polygon and
    # taking its perimeter would give 50 um by counting the cut at x=15.
    f = ex.extract_roi(M8, 0, 0, 15, 10)
    assert f["perimeter_density"] * (15 * 10) == pytest.approx(40.0, abs=1e-6)
    assert f["metal_density"] == pytest.approx(1.0, abs=1e-9)


def test_region_survives_a_temporary_reader(tmp_path):
    """Features must not silently vanish when the reader is not kept alive.

    A Region constructed directly from a RecursiveShapeIterator stays lazily
    bound to its Layout and empties once that Layout is collected. Written as
    a one-liner -- LayoutReader(path).region(spec) -- that yields zero-valued
    features with no error anywhere.
    """
    import gc

    sl = synth.SynthLayout()
    sl.add_box(8, 0, 0, 40, 40)
    path = tmp_path / "plate.gds"
    sl.write(str(path))

    region = LayoutReader(str(path)).region(M8)   # reader discarded immediately
    gc.collect()

    assert region.count() == 1
    assert region.area() > 0

# ----------------------------------------------------------------------
# test_section26_discrimination.py
# ----------------------------------------------------------------------
"""Spec section 26: the extractor must not degenerate into a metal-density detector.

Each test builds two patterns with matched metal density and one deliberately
different geometric property, then requires the extractor to separate them on
that property while agreeing on density.
"""
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

# ----------------------------------------------------------------------
# test_lineend_definitions.py
# ----------------------------------------------------------------------
"""Scoring of the candidate line-end definitions (spec section 4D).

Line ends are the one tier-1 feature with no self-evident definition on merged
geometry, so the definition is chosen by measurement rather than by argument.
Each benchmark pattern carries the termination count implied by its
construction; a definition is judged by how exactly it recovers those counts.
"""
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

# ----------------------------------------------------------------------
# test_structures.py
# ----------------------------------------------------------------------
"""Wide metal, slotting and declared dummy fill.

Rabie et al. (2018) list wide-metal slotting among the layout levers, and fill
changes what every other feature means: it contributes to density and it sets
the shortest edge on a layer, which is what the line-end fallback would
otherwise take for a routing width.
"""
FILL = LayerSpec("M8_FILL", 8, 10)


def _slotted_plate(tmp_path, name, *, slot=True, x0=0.0):
    """A 100x100um plate, optionally cut with a 9x9 array of 4um slots."""
    sl = synth.SynthLayout()
    sl.add_box(8, x0, 0, x0 + 100, 100)
    if slot:
        for j in range(9):
            for i in range(9):
                sl.add_box(9, x0 + 5 + i * 10, 5 + j * 10,
                           x0 + 9 + i * 10, 9 + j * 10)
    path = tmp_path / name
    sl.write(str(path))

    layout = db.Layout()
    layout.read(str(path))
    top = layout.top_cells()[0]
    metal = db.Region()
    metal.insert(top.begin_shapes_rec(layout.find_layer(8, 0)))
    metal.merge()
    cutter = db.Region()
    idx = layout.find_layer(9, 0)
    if idx is not None:
        cutter.insert(top.begin_shapes_rec(idx))
        cutter.merge()

    out = db.Layout()
    out.dbu = layout.dbu
    cell = out.create_cell("TOP")
    cell.shapes(out.layer(8, 0)).insert(metal - cutter)
    final = tmp_path / f"cut_{name}"
    out.write(str(final))
    return LayoutReader(str(final))


def test_narrow_routing_is_not_wide_metal(tmp_path):
    sl = synth.SynthLayout()
    synth.lines(sl, 8, 0, 0, 100, 100, pitch=4.0, density=0.5)   # 2um lines
    path = tmp_path / "narrow.gds"
    sl.write(str(path))
    ex = StructureExtractor(LayoutReader(str(path)), wide_width_um=3.0)
    f = ex.extract_roi(M8, 0, 0, 100, 100)
    assert f["wide_metal_fraction"] == pytest.approx(0.0, abs=1e-6)
    assert f["slot_density"] == 0.0


def test_a_solid_plate_is_wide_metal_without_slots(tmp_path):
    reader = _slotted_plate(tmp_path, "solid.gds", slot=False)
    f = StructureExtractor(reader, wide_width_um=3.0).extract_roi(
        M8, 0, 0, 100, 100)
    assert f["wide_metal_fraction"] == pytest.approx(1.0, rel=1e-3)
    assert f["slot_density"] == 0.0


def test_slots_are_counted_and_raise_the_wide_metal_boundary(tmp_path):
    """The slot boundary is where an abrupt stiffness change sits."""
    solid = StructureExtractor(
        _slotted_plate(tmp_path, "s0.gds", slot=False), wide_width_um=3.0
    ).extract_roi(M8, 0, 0, 100, 100)
    slotted = StructureExtractor(
        _slotted_plate(tmp_path, "s1.gds", slot=True), wide_width_um=3.0
    ).extract_roi(M8, 0, 0, 100, 100)

    assert slotted["slot_density"] * (100 * 100) == pytest.approx(81, abs=1)
    assert slotted["wide_metal_perimeter_density"] > (
        solid["wide_metal_perimeter_density"] * 3)
    # Both are wide metal; only the slotting tells them apart.
    assert solid["wide_metal_fraction"] == pytest.approx(
        slotted["wide_metal_fraction"], rel=0.05)


def test_slot_counts_are_conserved_over_the_grid(tmp_path):
    reader = _slotted_plate(tmp_path, "grid.gds", slot=True)
    grid = build_grid(reader.bbox(), 50.0)
    out = StructureExtractor(reader, wide_width_um=3.0).extract(M8, grid)
    counted = out["slot_density"] * (50.0 ** 2)
    assert round(counted.sum()) == 81


def test_declared_fill_is_separated_from_functional_metal(tmp_path):
    """Fill is declared, not inferred from shape."""
    sl = synth.SynthLayout()
    synth.lines(sl, 8, 0, 0, 100, 50, pitch=4.0, density=0.5)      # routing
    for j in range(10):
        for i in range(10):
            sl.add_box(8, 5 + i * 10, 55 + j * 4, 6 + i * 10, 56 + j * 4,
                       datatype=10)                                  # fill
    path = tmp_path / "fill.gds"
    sl.write(str(path))
    reader = LayoutReader(str(path))

    without = StructureExtractor(reader, wide_width_um=3.0)
    withfill = StructureExtractor(reader, wide_width_um=3.0,
                                  fill_layers={"M8": FILL})
    roi = (0, 0, 100, 100)
    assert without.extract_roi(M8, *roi)["fill_density"] == 0.0
    f = withfill.extract_roi(M8, *roi)
    assert f["fill_density"] > 0
    assert 0.0 < f["fill_fraction"] < 1.0


def test_manifest_records_an_undeclared_fill_layer_as_a_gap(tmp_path):
    from collective.study import StudyManifest

    p = tmp_path / "m.yaml"
    p.write_text(
        "layout:\n  metal_layers:\n    - {name: M8, layer: 8, datatype: 0}\n")
    m = StudyManifest.load(p)
    assert any("no fill_layers" in g for g in m.gaps)
    assert m.wide_width_um == 3.0

# ----------------------------------------------------------------------
# test_gradient_and_crosslayer.py
# ----------------------------------------------------------------------
"""Gradient (spec section 5) and cross-layer (spec section 7) features.

Both sections exist because a layout cannot be reduced to a set of independent
per-layer scalar maps. These tests build dies where that reduction provably
loses the driver, so a regression that quietly drops either family fails here.
"""
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
