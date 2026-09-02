"""Position, package context, failure files and footprints.

Folded from ``tests/test_inspection_footprint.py``, ``tests/test_via_corner_package.py``, ``tests/test_reference_frames.py``, ``tests/test_input_validation.py``.
"""
from collective import layout as synth
from collective import workflow as pipeline
from collective.calibre import area_conversion
from collective.calibre import to_grid
from collective.geometry import GeometryExtractor
from collective.geometry import ViaExtractor
from collective.geometry import build_grid
from collective.labels import FailureSet
from collective.labels import InspectionFootprint
from collective.labels import PackageLayers
from collective.labels import absent_context_note
from collective.labels import audit_failures
from collective.labels import coverage
from collective.labels import eligibility
from collective.labels import failures_from_driver
from collective.labels import load_failures
from collective.labels import map_to_grid
from collective.labels import package_context_extract as ctx_extract
from collective.layout import BBox
from collective.layout import LayerSpec
from collective.layout import LayoutReader
from collective.layout import packaged_die
from collective.layout import validation_die
from collective.statistics import block_permutation_test
from dataclasses import replace
from scipy import stats
import numpy as np
import pandas as pd
import pytest


# ----------------------------------------------------------------------
# test_inspection_footprint.py
# ----------------------------------------------------------------------
"""Inspection footprint and control opportunity.

A cell is a valid control only if inspection had a real chance to find a
failure there. Where it did not, the cell is missing data, and counting it as
clean biases the denominator -- which no downstream correction can repair,
because block permutation, FDR and the position baseline all operate on
whatever population they are given.
"""
M8 = LayerSpec("M8", 8, 0)
DIE = 2000.0
CORNER = 500.0
CORNER_BOXES = [(0, 0, CORNER, CORNER), (DIE - CORNER, 0, DIE, CORNER),
                (0, DIE - CORNER, CORNER, DIE),
                (DIE - CORNER, DIE - CORNER, DIE, DIE)]


# ---- footprint construction ----------------------------------------

def test_full_die_coverage_must_be_justified():
    """Asserting whole-die inspection is a claim someone has to own."""
    bbox = BBox(0, 0, 1000, 1000)
    with pytest.raises(ValueError, match="must be justified"):
        InspectionFootprint.full_die(bbox, "")
    ok = InspectionFootprint.full_die(bbox, "whole-die C-SAM at 50um, 100% called")
    assert ok.assumed_full_coverage
    assert ok.report()["justification"]


def test_empty_footprint_is_refused():
    with pytest.raises(ValueError, match="no inspection rectangles"):
        InspectionFootprint.from_rectangles([])


def test_coverage_is_the_inspected_fraction_of_each_cell():
    grid = build_grid(BBox(0, 0, 1000, 1000), 100.0)
    fp = InspectionFootprint.from_rectangles([(0, 0, 550, 1000)])
    frac = coverage(fp, grid)
    assert frac.max() == pytest.approx(1.0)
    # The column straddling x = 550 is exactly half covered.
    half = [c.cell_id for c in grid.cells if c.x0 == 500.0]
    assert np.allclose(frac[half], 0.5)
    assert (frac[[c.cell_id for c in grid.cells if c.x0 >= 600.0]] == 0).all()


def test_partly_covered_cells_are_excluded_by_the_threshold():
    """There is no way to express half a control in a binary label."""
    grid = build_grid(BBox(0, 0, 1000, 1000), 100.0)
    fp = InspectionFootprint.from_rectangles([(0, 0, 550, 1000)])
    strict, _ = eligibility(fp, grid, min_coverage=0.9)
    loose, _ = eligibility(fp, grid, min_coverage=0.4)
    assert loose.sum() > strict.sum()


def test_a_failure_outside_the_footprint_is_a_contradiction(tmp_path):
    """Something found where nothing was looked at means an input is wrong."""
    from collective.labels import FailureSet
    import pandas as pd

    fp = InspectionFootprint.from_rectangles([(0, 0, 100, 100)])
    fs = FailureSet(table=pd.DataFrame({
        "sample_id": ["inside", "outside"], "x_um": [50.0, 500.0],
        "y_um": [50.0, 500.0], "failure_type": "delamination",
        "confidence": 1.0, "position_sigma_um": np.nan}))
    audit = audit_failures(fp, fs)
    assert not audit["consistent"]
    assert audit["n_outside_footprint"] == 1
    assert audit["outside_sample_ids"] == ["outside"]


# ---- the bias it exists to prevent ---------------------------------

@pytest.fixture(scope="module")
def targeted_inspection(tmp_path_factory):
    """Failures everywhere, but only the die corners were ever inspected."""
    path = str(tmp_path_factory.mktemp("fp") / "die.gds")
    validation_die(path, die_um=DIE, block_um=50.0, seed=7)
    reader = LayoutReader(path)
    grid = build_grid(reader.bbox(), 100.0)
    feats = GeometryExtractor(reader, line_end_w_max_um=4.0).extract(M8, grid)

    everywhere = failures_from_driver(feats["perimeter_density"], grid,
                                      n_failures=400, strength=2.5, seed=1,
                                      position_sigma_um=5.0)
    table = everywhere.table
    inside = np.zeros(len(table), dtype=bool)
    for x0, y0, x1, y1 in CORNER_BOXES:
        inside |= ((table.x_um >= x0) & (table.x_um < x1)
                   & (table.y_um >= y0) & (table.y_um < y1)).to_numpy()
    found = replace(everywhere, table=table[inside].reset_index(drop=True))
    fp = InspectionFootprint.from_rectangles(CORNER_BOXES,
                                             source="corner-targeted FIB")
    assert 0 < len(found) < len(everywhere)
    return path, found, fp


def test_targeted_inspection_fakes_a_position_effect(targeted_inspection):
    """Without a footprint the targeting itself becomes the strongest finding.

    The die carries no package-position effect and the failures were driven by
    perimeter density alone, yet counting never-inspected area as control makes
    distance-to-corner one of the strongest associations the pipeline can
    produce. FDR does not help: this is a wrong population, not a multiple
    comparison.
    """
    path, found, _ = targeted_inspection
    res = pipeline.run(path, found, layer=M8, scales_um=(100,),
                       n_permutations=0, line_end_w_max_um=4.0, seed=1)
    a = res.associations.set_index("feature")
    assert abs(a.loc["distance_to_nearest_corner", "roc_auc"] - 0.5) > 0.35
    assert a.loc["distance_to_nearest_corner", "fdr_q_value"] < 1e-10
    assert any("no inspection footprint supplied" in n
               for n in res.metadata["uncontrolled_confounding"])


def test_declaring_the_footprint_removes_the_artifact(targeted_inspection):
    """And the real driver comes out stronger, not weaker."""
    path, found, fp = targeted_inspection
    naive = pipeline.run(path, found, layer=M8, scales_um=(100,),
                         n_permutations=0, line_end_w_max_um=4.0, seed=1)
    gated = pipeline.run(path, found, layer=M8, scales_um=(100,),
                         n_permutations=0, line_end_w_max_um=4.0, seed=1,
                         footprint=fp)

    n = naive.associations.set_index("feature")
    g = gated.associations.set_index("feature")

    for feature in ("distance_to_nearest_corner", "distance_to_die_edge",
                    "normalized_distance_from_die_center"):
        assert abs(g.loc[feature, "roc_auc"] - 0.5) < 0.10
        assert g.loc[feature, "fdr_q_value"] > 0.5

    assert g.loc["perimeter_density", "roc_auc"] > n.loc["perimeter_density", "roc_auc"]


def test_coverage_and_audit_reach_the_run_metadata(targeted_inspection):
    path, found, fp = targeted_inspection
    res = pipeline.run(path, found, layer=M8, scales_um=(100,),
                       n_permutations=0, line_end_w_max_um=4.0, seed=1,
                       footprint=fp)
    meta = res.metadata
    assert meta["inspection_footprint"]["uniform"]
    assert meta["inspection_footprint"]["default"]["source"] == "corner-targeted FIB"
    assert meta["failure_footprint_audit"]["consistent"]
    cov = meta["coverage_by_scale"][100.0]
    assert cov["n_eligible"] < cov["n_cells"]
    assert cov["n_cases_eligible"] > 0


# ---- per-die footprints ---------------------------------------------

def test_a_footprint_per_die_changes_who_is_a_control(tmp_path):
    """A cell inspected on one die and not another is not the same observation.

    Collapsing a campaign to one footprint either discards the dies inspected
    more thoroughly, or credits the ones inspected less with controls nobody
    earned.
    """
    import pandas as pd
    from dataclasses import replace
    from collective.labels import FailureSet
    from collective.labels import FootprintSet
    from collective.geometry import GeometryExtractor
    from collective.layout import validation_die

    path = str(tmp_path / "die.gds")
    validation_die(path, die_um=1000.0, block_um=50.0, seed=7)
    reader = LayoutReader(path)
    grid = build_grid(reader.bbox(), 100.0)
    feats = GeometryExtractor(reader, line_end_w_max_um=4.0).extract(M8, grid)

    frames = []
    for d in range(2):
        fs = failures_from_driver(feats["perimeter_density"], grid,
                                  n_failures=30, strength=2.5, seed=20 + d,
                                  position_sigma_um=5.0)
        t = fs.table.copy()
        t["lot_id"], t["wafer_id"], t["die_x"], t["die_y"] = "L1", "W1", d, 0
        t["sample_id"] = [f"D{d}_{s}" for s in t.sample_id]
        # Die 0 was only inspected on its left half, so its right-hand
        # failures are outside its own footprint; drop them as the campaign
        # would never have found them.
        if d == 0:
            t = t[t.x_um < 500.0].reset_index(drop=True)
        frames.append(t)
    multi = FailureSet(table=pd.concat(frames, ignore_index=True))

    footprints = FootprintSet(
        default=InspectionFootprint.from_rectangles([(0, 0, 1000, 1000)],
                                                    source="whole die"),
        per_die={"L1|W1|0|0": InspectionFootprint.from_rectangles(
            [(0, 0, 500, 1000)], source="left half only")})

    res = pipeline.run(path, multi, layer=M8, scales_um=(100,),
                       n_permutations=0, line_end_w_max_um=4.0, seed=1,
                       footprints=footprints)
    cov = res.metadata["coverage_by_scale"][100.0]
    assert not cov["uniform_footprint"]
    # Two dies over the same grid, but only one and a half dies' worth of
    # cells are eligible.
    assert cov["n_eligible"] < cov["n_observations"]
    assert cov["n_eligible"] > cov["n_cells"]
    assert res.metadata["failure_footprint_audit"]["consistent"]


def test_a_die_with_no_footprint_is_refused(tmp_path):
    """No declared footprint means no control population for that die."""
    import pandas as pd
    from collective.labels import FailureSet
    from collective.labels import FootprintSet
    from collective.layout import validation_die

    path = str(tmp_path / "die.gds")
    validation_die(path, die_um=500.0, block_um=50.0, seed=1)
    table = pd.DataFrame({
        "sample_id": ["A", "B"], "lot_id": "L1", "wafer_id": "W1",
        "die_x": [0, 1], "die_y": 0, "x_um": [100.0, 120.0],
        "y_um": [100.0, 120.0], "failure_type": "delamination",
        "confidence": 1.0, "position_sigma_um": np.nan})

    # A per-die set covering only one of the two dies leaves the other with
    # no declared control population.
    footprints = FootprintSet(per_die={
        "L1|W1|0|0": InspectionFootprint.from_rectangles([(0, 0, 500, 500)])})
    with pytest.raises(ValueError, match="no inspected footprint for die"):
        pipeline.run(path, FailureSet(table=table), layer=M8,
                     scales_um=(100,), n_permutations=0, footprints=footprints)

    # Adding a default covers the remaining dies, and the run proceeds.
    footprints.default = InspectionFootprint.from_rectangles(
        [(0, 0, 500, 500)], source="whole die")
    res = pipeline.run(path, FailureSet(table=table), layer=M8,
                       scales_um=(100,), n_permutations=0,
                       footprints=footprints)
    assert res.metadata["failure_footprint_audit"]["consistent"]


def test_a_failure_just_outside_is_measurement_error_not_a_contradiction():
    """Otherwise the check fires on every real campaign and gets overridden.

    A near-edge failure whose registered position lands a few micrometres
    outside, measured with an uncertainty larger than that, is the same
    failure seen through its own error. Beyond the tolerance, measurement
    error is no longer an explanation.
    """
    import pandas as pd
    from collective.labels import FailureSet
    from collective.labels import TOLERANCE_SIGMAS

    fp = InspectionFootprint.from_rectangles([(0, 0, 1000, 1000)])
    sigma = 10.0
    just_outside = 1000.0 + sigma          # inside 3 sigma
    far_outside = 1000.0 + sigma * TOLERANCE_SIGMAS * 4

    table = pd.DataFrame({
        "sample_id": ["near", "far"], "x_um": [just_outside, far_outside],
        "y_um": [500.0, 500.0], "failure_type": "delamination",
        "confidence": 1.0, "position_sigma_um": sigma})
    audit = audit_failures(fp, FailureSet(table=table))

    assert audit["tolerance_um"] == pytest.approx(TOLERANCE_SIGMAS * sigma)
    assert audit["n_within_tolerance"] == 1
    assert audit["outside_sample_ids"] == ["far"]
    assert not audit["consistent"]


def test_without_a_reported_sigma_the_boundary_is_strict():
    """No stated uncertainty means no room to grant."""
    import pandas as pd
    from collective.labels import FailureSet

    fp = InspectionFootprint.from_rectangles([(0, 0, 1000, 1000)])
    table = pd.DataFrame({
        "sample_id": ["a"], "x_um": [1005.0], "y_um": [500.0],
        "failure_type": "delamination", "confidence": 1.0,
        "position_sigma_um": np.nan})
    audit = audit_failures(fp, FailureSet(table=table))
    assert audit["tolerance_um"] == 0.0
    assert not audit["consistent"]

# ----------------------------------------------------------------------
# test_via_corner_package.py
# ----------------------------------------------------------------------
"""Via, corner and package-context features.

These close the tier-1 gaps in references/feature_evidence_map.csv: via
density (Vanstreels 2020, Zahedmanesh 2019), corner density (Tan 2008) and
bump/PI context (Rabie 2018, Li 2023/2025). Each is validated the same way as
the earlier families -- by construction, on a layout where the quantity being
measured is known rather than inferred.
"""
V7 = LayerSpec("V7", 17, 0)


# ---- vias ----------------------------------------------------------

def test_via_area_and_count_density_are_independent(tmp_path):
    """Equal via area, four times the via count.

    Vanstreels counts fractured vias rather than via area, so a layer can hold
    area density constant while changing the number of interfaces. A single
    via feature could not express that.
    """
    sl = synth.SynthLayout()
    synth.via_array(sl, 17, 0, 0, 100, 100, pitch=10.0, size=4.0)
    synth.via_array(sl, 17, 200, 0, 300, 100, pitch=5.0, size=2.0)
    path = tmp_path / "vias.gds"
    sl.write(str(path))
    ex = ViaExtractor(LayoutReader(str(path)))
    rois = synth.pair_rois()
    a = ex.extract_roi(V7, *rois["A"])
    b = ex.extract_roi(V7, *rois["B"])

    assert a["via_density"] == pytest.approx(b["via_density"], rel=1e-9)
    assert b["via_count_density"] == pytest.approx(4 * a["via_count_density"], rel=1e-9)
    assert a["mean_via_area"] == pytest.approx(4 * b["mean_via_area"], rel=1e-9)


def test_via_counts_sum_to_the_vias_the_grid_covers(tmp_path):
    """Counting by centroid means a via on a window edge belongs to one cell.

    The comparison is against the vias inside the grid's coverage, not the
    whole layer: build_grid drops partial cells at the far edge so that every
    cell of a given scale has the same footprint, which necessarily leaves a
    strip of the layout unanalysed.
    """
    sl = synth.SynthLayout()
    synth.via_array(sl, 17, 0, 0, 400, 400, pitch=20.0, size=6.0)
    path = tmp_path / "grid_vias.gds"
    sl.write(str(path))
    reader = LayoutReader(str(path))
    ex = ViaExtractor(reader)
    pts, _ = ex.centroids(V7)

    grid = build_grid(reader.bbox(), 100.0)
    x_max = max(c.x1 for c in grid.cells)
    y_max = max(c.y1 for c in grid.cells)
    covered = int(((pts[:, 0] < x_max) & (pts[:, 1] < y_max)).sum())
    assert 0 < covered < len(pts), "the test needs a grid that drops a strip"

    counts = ex.extract(V7, grid)["via_count_density"] * (100.0 ** 2)
    assert round(counts.sum()) == covered


def test_absent_via_layer_yields_zeros_not_an_error(tmp_path):
    sl = synth.SynthLayout()
    synth.lines(sl, 8, 0, 0, 100, 100, pitch=4.0, density=0.5)
    path = tmp_path / "novia.gds"
    sl.write(str(path))
    reader = LayoutReader(str(path))
    grid = build_grid(reader.bbox(), 50.0)
    out = ViaExtractor(reader).extract(V7, grid)
    assert all((v == 0).all() for v in out.values())


# ---- corners -------------------------------------------------------

def test_corner_density_separates_straight_from_staircase(tmp_path):
    sl = synth.pair_density_vs_corner()
    path = tmp_path / "corners.gds"
    sl.write(str(path))
    ex = GeometryExtractor(LayoutReader(str(path)), line_end_w_max_um=3.0)
    rois = synth.pair_rois()
    a = ex.extract_roi(M8, *rois["A"])
    b = ex.extract_roi(M8, *rois["B"])

    assert b["corner_density"] > a["corner_density"] * 10
    # Straight lines are rectangles: four convex corners each, no re-entrant
    # ones. Concave corners are the stress-concentrating case, so conflating
    # them with convex corners in a single vertex count loses the distinction.
    assert a["concave_corner_density"] == 0.0
    assert b["concave_corner_density"] > 0.0


def test_corner_counts_are_conserved_over_the_grid_coverage(tmp_path):
    """Every corner inside the analysed area is counted exactly once."""
    sl = synth.SynthLayout()
    synth.staircase_lines(sl, 8, 0, 0, 200, 200, pitch=8.0, density=0.4, step=10.0)
    path = tmp_path / "stair.gds"
    sl.write(str(path))
    reader = LayoutReader(str(path))
    ex = GeometryExtractor(reader, line_end_w_max_um=4.0)
    convex, concave = ex.corners(M8)
    pts = np.vstack([convex, concave]) if len(concave) else convex

    grid = build_grid(reader.bbox(), 50.0)
    x_max = max(c.x1 for c in grid.cells)
    y_max = max(c.y1 for c in grid.cells)
    covered = int(((pts[:, 0] < x_max) & (pts[:, 1] < y_max)).sum())

    counted = ex.extract(M8, grid)["corner_density"] * (50.0 ** 2)
    assert round(counted.sum()) == covered


def test_holes_do_not_invert_corner_classification(tmp_path):
    """A ring's inner corners wind the opposite way to its outer ones."""
    sl = synth.SynthLayout()
    synth.bench_ring(sl, 8, outer=60.0, wall=10.0)
    path = tmp_path / "ring.gds"
    sl.write(str(path))
    ex = GeometryExtractor(LayoutReader(str(path)), line_end_w_max_um=12.0)
    convex, concave = ex.corners(M8)
    assert len(convex) == 4          # the outer frame
    assert len(concave) == 4         # the hole, re-entrant from the metal side


# ---- package context -----------------------------------------------

@pytest.fixture(scope="module")
def packaged_via(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("pkg") / "pkg.gds")
    _, bumps = synth.packaged_die(path, die_um=3000.0, block_um=100.0, seed=31)
    reader = LayoutReader(path)
    grid = build_grid(reader.bbox(), 100.0)
    layers = PackageLayers(bump=LayerSpec("BUMP", 60, 0),
                           pi_opening=LayerSpec("PI", 61, 0),
                           crackstop=LayerSpec("CS", 62, 0))
    return reader, grid, layers, bumps


def test_bump_distance_matches_the_known_bump_positions(packaged_via):
    reader, grid, layers, bumps = packaged_via
    ctx = ctx_extract(grid, reader.bbox(), reader, layers)
    x = np.array([c.x_center for c in grid.cells])
    y = np.array([c.y_center for c in grid.cells])
    expected = np.min(np.hypot(x[:, None] - bumps[None, :, 0],
                               y[:, None] - bumps[None, :, 1]), axis=1)
    assert np.allclose(ctx["distance_to_nearest_bump"], expected)


def test_local_bump_pitch_recovers_the_generated_pitch(packaged_via):
    reader, grid, layers, _ = packaged_via
    ctx = ctx_extract(grid, reader.bbox(), reader, layers)
    assert np.allclose(ctx["local_bump_pitch"], 400.0)


def test_radial_and_tangential_offsets_decompose_the_bump_vector(packaged_via):
    """The two components must reconstruct the distance to the nearest bump."""
    reader, grid, layers, _ = packaged_via
    ctx = ctx_extract(grid, reader.bbox(), reader, layers)
    recomposed = np.hypot(ctx["bump_radial_offset"], ctx["bump_tangential_offset"])
    assert np.allclose(recomposed, ctx["distance_to_nearest_bump"])


def test_under_bump_indicator_finds_cells_inside_a_bump(packaged_via):
    """A zero-area probe intersects nothing; the test is a real containment."""
    reader, grid, layers, _ = packaged_via
    ctx = ctx_extract(grid, reader.bbox(), reader, layers)
    flagged = ctx["under_bump_indicator"]
    assert 0 < flagged.sum() < len(flagged)
    # Every flagged cell must be closer to its bump than the bump half-width.
    assert (ctx["distance_to_nearest_bump"][flagged > 0]
            <= 150.0 / 2 * np.sqrt(2) + 1e-6).all()


def test_missing_package_layers_are_recorded_as_uncontrolled(packaged_via):
    reader, grid, layers, _ = packaged_via
    notes = absent_context_note(PackageLayers())
    assert any("UNCONTROLLED" in n for n in notes)
    assert {n.split()[1] for n in notes} == {"bump/C4", "PI-opening",
                                             "crackstop", "pad"}
    # The fixture supplies everything except pad, so exactly that is reported.
    assert [n.split()[1] for n in absent_context_note(layers)] == ["pad"]


def test_package_context_is_position_evidence_not_geometry():
    """It comes from GDS layers but belongs in the baseline it exists to control."""
    from collective.foundation import EvidenceClass
    from collective import labels as package_context
    assert package_context.EVIDENCE_CLASS is EvidenceClass.PACKAGE_POSITION


# ---- a feature deliberately not built ------------------------------

def test_effective_modulus_would_be_redundant_with_metal_density():
    """Why no effective-stiffness proxy exists, so it is not added later.

    A rule-of-mixtures effective modulus is a monotone function of the metal
    fraction, so it is rank-identical to metal_density for every rank-based
    statistic the pipeline uses -- Mann-Whitney, ROC-AUC, Cliff's delta,
    Spearman. It cannot add univariate information at any material contrast,
    and a linear homogenisation is in any case a linear combination of
    features the multivariate baseline already fits.
    """
    f = np.random.default_rng(0).uniform(0.2, 0.8, 2000)
    for e_metal, e_diel in [(110.0, 4.0), (2.0, 1.0), (1e6, 1.0)]:
        voigt = f * e_metal + (1 - f) * e_diel
        reuss = 1.0 / (f / e_metal + (1 - f) / e_diel)
        assert stats.spearmanr(voigt, f).statistic == pytest.approx(1.0)
        assert stats.spearmanr(reuss, f).statistic == pytest.approx(1.0)

    # Equal moduli make it a constant, which carries nothing at all.
    assert np.std(f * 1.0 + (1 - f) * 1.0) == pytest.approx(0.0)

# ----------------------------------------------------------------------
# test_reference_frames.py
# ----------------------------------------------------------------------
"""The three coordinate frames, and the boundaries derived from them.

Each of these was a case where the analysis boundary came from what happened
to be in the file rather than from what a human declared, and none of them
produced an error -- only a plausible number measured from the wrong origin.
"""
MANIFEST_PATH = "config/study_manifest.yaml"


@pytest.fixture(scope="module")
def die(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("rf") / "die.gds")
    validation_die(path, die_um=1000.0, block_um=50.0, seed=7)
    reader = LayoutReader(path)
    grid = build_grid(reader.bbox(), 100.0)
    feats = GeometryExtractor(reader, line_end_w_max_um=4.0).extract(M8, grid)
    fs = failures_from_driver(feats["perimeter_density"], grid, n_failures=60,
                              strength=2.5, seed=1, position_sigma_um=5.0)
    return path, reader, fs


# ---- A1: the declared die is the die ------------------------------

def test_declared_die_outline_sets_the_position_origin(die):
    """A region of interest must not put the die centre inside the region."""
    path, reader, fs = die
    geometry = reader.bbox()
    declared = BBox(-3000.0, -3000.0, 4000.0, 4000.0)

    naive = pipeline.run(path, fs, layer=M8, scales_um=(100,),
                         n_permutations=0, line_end_w_max_um=4.0, seed=1)
    framed = pipeline.run(path, fs, layer=M8, scales_um=(100,),
                          n_permutations=0, line_end_w_max_um=4.0, seed=1,
                          die_bbox=declared)

    assert naive.metadata["die_bbox_um"][:2] == [geometry.xmin, geometry.ymin]
    assert framed.metadata["die_bbox_um"] == [-3000.0, -3000.0, 4000.0, 4000.0]
    # Same layout, same failures, different declared die: the distance to the
    # die edge is a different quantity.
    a = naive.features["distance_to_die_edge|-"]
    b = framed.features["distance_to_die_edge|-"]
    assert not np.allclose(a, b)
    assert b.min() > a.max()


def test_the_grid_follows_the_geometry_not_the_declared_die(die):
    """Features exist only where geometry does; folds index the same rows."""
    path, reader, fs = die
    declared = BBox(-3000.0, -3000.0, 4000.0, 4000.0)
    res = pipeline.run(path, fs, layer=M8, scales_um=(100,), n_permutations=0,
                       line_end_w_max_um=4.0, seed=1, die_bbox=declared)
    expected = len(build_grid(reader.bbox(), 100.0))
    assert res.metadata["coverage_by_scale"][100.0]["n_cells"] == expected


def test_a_region_of_interest_says_what_it_cannot_conclude(die):
    path, reader, fs = die
    res = pipeline.run(path, fs, layer=M8, scales_um=(100,), n_permutations=0,
                       line_end_w_max_um=4.0, seed=1,
                       die_bbox=BBox(-3000.0, -3000.0, 4000.0, 4000.0))
    joined = " ".join(res.metadata["uncontrolled_confounding"])
    assert "region of interest" in joined
    assert "no claim about which part of the die is worst" in joined


def test_an_undeclared_die_outline_is_flagged_as_assumed(die):
    path, reader, fs = die
    res = pipeline.run(path, fs, layer=M8, scales_um=(100,), n_permutations=0,
                       line_end_w_max_um=4.0, seed=1)
    assert any("no die outline declared" in n
               for n in res.metadata["uncontrolled_confounding"])


def test_a_die_outline_that_excludes_the_geometry_is_refused(die):
    path, reader, fs = die
    with pytest.raises(ValueError, match="does not contain the loaded geometry"):
        pipeline.run(path, fs, layer=M8, scales_um=(100,), n_permutations=0,
                     line_end_w_max_um=4.0, die_bbox=BBox(0, 0, 100, 100))


# ---- A2: unknown accuracy is not accuracy --------------------------

def test_unknown_registration_accuracy_cannot_be_primary(die):
    """Not knowing how well a failure was placed is not knowing it was placed well."""
    from collective.exposure import partition

    path, reader, fs = die
    fs.table["position_sigma_um"] = np.nan
    res = pipeline.run(path, fs, layer=M8, scales_um=(25, 50, 100),
                       n_permutations=0, line_end_w_max_um=4.0, seed=1)
    assert set(res.associations["scale_status"]) == {"uncertified"}
    parts = partition(res.associations)
    assert len(parts["primary"]) == 0
    assert len(parts["unsupported_scale"]) == len(res.associations)


def test_a_measured_sigma_certifies_only_the_scales_above_its_floor(die):
    path, reader, fs = die
    fs.table["position_sigma_um"] = 20.0        # floor 60um
    res = pipeline.run(path, fs, layer=M8, scales_um=(25, 50, 100, 250),
                       n_permutations=0, line_end_w_max_um=4.0, seed=1)
    a = res.associations
    assert set(a[a.scale_um <= 50].scale_status) == {"below_registration_floor"}
    assert set(a[a.scale_um >= 100].scale_status) == {"supported"}


# ---- A3: permutations stay inside a die ----------------------------

def test_permutation_without_strata_moves_failures_between_dies():
    """The grouping alone does not confine the exchange -- the stratum does."""
    grid = build_grid(BBox(0, 0, 400, 400), 100.0)
    n = len(grid)
    cell = np.tile(np.arange(n), 3)
    die_index = np.repeat(np.arange(3), n)
    values = np.tile(np.linspace(0, 1, n), 3)
    labels = (die_index == 0).astype(int)      # every failure on die 0
    groups = die_index * 1000 + cell

    def run(**kw):
        seen = []
        block_permutation_test(
            values, labels, grid,
            statistic=lambda v, l: (seen.append(l.copy()), 0.5)[1],
            n_permutations=15, groups=groups, block_cells=1, seed=0, **kw)
        return sum(1 for l in seen[1:]
                   if [int(l[die_index == d].sum()) for d in range(3)] != [n, 0, 0])

    assert run() == 15
    assert run(strata=die_index) == 0


def test_stratified_permutation_still_permutes_within_a_die():
    grid = build_grid(BBox(0, 0, 400, 400), 100.0)
    n = len(grid)
    cell = np.tile(np.arange(n), 2)
    die_index = np.repeat(np.arange(2), n)
    values = np.tile(np.linspace(0, 1, n), 2)
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 2 * n)
    groups = die_index * 1000 + cell

    seen = []
    block_permutation_test(
        values, labels, grid,
        statistic=lambda v, l: (seen.append(l.copy()), 0.5)[1],
        n_permutations=15, groups=groups, block_cells=1, seed=0,
        strata=die_index)
    changed = sum(1 for l in seen[1:] if not np.array_equal(l, labels))
    assert changed > 10, "a stratified permutation must still shuffle"


# ---- A4: distances are measured to shapes --------------------------

@pytest.fixture(scope="module")
def packaged_reference(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("pc") / "pkg.gds")
    _, bumps = packaged_die(path, die_um=3000.0, block_um=150.0, seed=3)
    reader = LayoutReader(path)
    grid = build_grid(reader.bbox(), 100.0)
    layers = PackageLayers(bump=LayerSpec("B", 60, 0),
                           pi_opening=LayerSpec("PI", 61, 0),
                           crackstop=LayerSpec("CS", 62, 0))
    return reader, grid, layers


def test_crackstop_distance_is_not_a_copy_of_distance_to_centre(packaged_reference):
    """A ring's bounding-box centre is the die centre.

    Measuring to that centre made distance_to_crackstop numerically identical
    to distance-to-die-centre -- correlation 1.000000, maximum difference
    0.0000um -- so the position baseline carried the same column twice, one of
    them under a name suggesting crackstop proximity had been measured.
    """
    reader, grid, layers = packaged_reference
    ctx = ctx_extract(grid, reader.bbox(), reader, layers)
    x = np.array([c.x_center for c in grid.cells])
    y = np.array([c.y_center for c in grid.cells])
    to_centre = np.hypot(x - 1500.0, y - 1500.0)

    d = ctx["distance_to_crackstop"]
    assert np.abs(d - to_centre).max() > 100.0
    # A rail 20um inside a 3000um die: nothing is 1500um from it.
    assert d.max() < 1500.0


def test_pi_opening_distance_is_measured_to_the_opening_edge(packaged_reference):
    """Li et al. locate the stress concentration at the opening edge."""
    reader, grid, layers = packaged_reference
    ctx = ctx_extract(grid, reader.bbox(), reader, layers)
    d = ctx["distance_to_nearest_pi_opening"]
    # Openings are 105um across on a 400um pitch, so no cell is further than
    # about half a pitch from one; a centroid-based distance could not bound it.
    assert d.max() < 300.0
    assert ctx["distance_to_pi_opening_corner"].min() >= d.min()


def test_pad_features_appear_only_when_a_pad_layer_is_supplied(packaged_reference):
    reader, grid, layers = packaged_reference
    ctx = ctx_extract(grid, reader.bbox(), reader, layers)
    assert np.isnan(ctx["distance_to_pad_edge"]).all()
    assert (ctx["under_pad_indicator"] == 0).all()


# ---- B1: per-layer PDK rules ---------------------------------------

def test_line_rules_are_applied_per_layer(tmp_path):
    """Collapsing the stack to one cutoff misreads a wide line as a tip.

    M8 routes at 2um and M7 at 1um. Applying the widest rule in the stack to
    every layer lets an M7 power strap of 1.8um qualify as a terminated
    routing line.
    """
    from collective import layout as synth

    sl = synth.SynthLayout()
    for i in range(6):
        sl.add_box(8, 0, i * 6.0, 60.0, i * 6.0 + 2.0)      # M8, 2um
        sl.add_box(7, 0, i * 6.0, 60.0, i * 6.0 + 1.0)      # M7, 1um
    sl.add_box(7, 0, 40.0, 60.0, 41.8)                       # M7 strap, 1.8um
    path = tmp_path / "rules.gds"
    sl.write(str(path))
    reader = LayoutReader(str(path))
    m7 = LayerSpec("M7", 7, 0)
    roi = (0, 0, 60, 50)

    collapsed = GeometryExtractor(reader, line_end_w_max_um=2.0)
    per_layer = GeometryExtractor(
        reader, line_rules={"M8": (0.2, 2.0), "M7": (0.1, 1.0)})

    n_collapsed = collapsed.extract_roi(m7, *roi)["line_end_density"] * (60 * 50)
    n_per_layer = per_layer.extract_roi(m7, *roi)["line_end_density"] * (60 * 50)
    assert n_collapsed > n_per_layer, "the strap must stop counting as a tip"


def test_manifest_exposes_rules_per_layer_not_one_maximum():
    from collective.study import StudyManifest

    m = StudyManifest.load(MANIFEST_PATH)
    rules = m.line_rule_map()
    assert rules["M8"] == (0.20, 2.0)
    assert rules["M7"] == (0.10, 1.0)
    # The single-number form remains only as a fallback, and is the maximum.
    assert m.line_end_w_max_um() == 2.0


# ---- B2: a contradiction stops the run -----------------------------

def test_a_failure_outside_the_footprint_stops_the_run(die):
    """It disproves the population definition rather than qualifying it."""
    from collective.labels import InspectionFootprint

    path, reader, fs = die
    half = InspectionFootprint.from_rectangles([(0, 0, 500, 1000)],
                                               source="left half only")
    with pytest.raises(ValueError, match="disproves the population definition"):
        pipeline.run(path, fs, layer=M8, scales_um=(100,), n_permutations=0,
                     line_end_w_max_um=4.0, footprint=half)


def test_continuing_past_the_contradiction_must_be_asserted(die):
    from collective.labels import InspectionFootprint

    path, reader, fs = die
    half = InspectionFootprint.from_rectangles([(0, 0, 500, 1000)],
                                               source="left half only")
    res = pipeline.run(path, fs, layer=M8, scales_um=(100,), n_permutations=0,
                       line_end_w_max_um=4.0, footprint=half,
                       allow_failures_outside_footprint=True)
    assert any("asserted by the operator" in n and "dropped" in n
               for n in res.metadata["uncontrolled_confounding"])


def test_a_consistent_footprint_needs_no_override(die):
    from collective.labels import InspectionFootprint

    path, reader, fs = die
    whole = InspectionFootprint.from_rectangles([(0, 0, 1000, 1000)],
                                                source="whole die")
    res = pipeline.run(path, fs, layer=M8, scales_um=(100,), n_permutations=0,
                       line_end_w_max_um=4.0, footprint=whole)
    assert res.metadata["failure_footprint_audit"]["consistent"]

# ----------------------------------------------------------------------
# test_input_validation.py
# ----------------------------------------------------------------------
"""Boundary handling for data arriving from outside the pipeline.

Every case here is one where the previous behaviour produced a plausible
number rather than an error. That is the failure mode worth testing for: a
crash gets noticed, a silently biased label set does not.
"""
GRID_BBOX = BBox(0, 0, 1000, 1000)


def _failures(points, **cols):
    n = len(points)
    table = pd.DataFrame({
        "sample_id": [f"S{i}" for i in range(n)],
        "x_um": [p[0] for p in points], "y_um": [p[1] for p in points],
        "failure_type": "delamination", "confidence": 1.0,
        "position_sigma_um": np.nan})
    for k, v in cols.items():
        table[k] = v
    return FailureSet(table=table)


def _write(tmp_path, name, **cols):
    base = {"sample_id": ["S0"], "lot_id": "L1", "wafer_id": "W1",
            "die_x": 0, "die_y": 0, "x_um": [100.0], "y_um": [100.0],
            "failure_type": "delamination"}
    base.update(cols)
    path = tmp_path / name
    pd.DataFrame(base).to_csv(path, index=False)
    return path


# ---- failure-to-cell assignment ------------------------------------

def test_failure_in_a_cell_corner_is_assigned_to_that_cell():
    """A radius against the cell centre inscribes a circle and loses corners."""
    grid = build_grid(GRID_BBOX, 100.0)
    labels = map_to_grid(_failures([(1.0, 1.0)]), grid)
    assert labels["failure_present"][0] == 1
    assert labels["failure_count"].sum() == 1


def test_uniform_failures_are_all_assigned():
    """The circular test lost 1 - pi/4 of the die, on a regular lattice."""
    grid = build_grid(GRID_BBOX, 100.0)
    pts = np.random.default_rng(0).uniform(0, 1000, (5000, 2))
    labels = map_to_grid(_failures(list(map(tuple, pts))), grid)
    assert labels["failure_count"].sum() == len(pts)


def test_failure_on_a_shared_edge_is_counted_once():
    """Bounds are half-open inside the grid, so an edge belongs to one cell."""
    grid = build_grid(GRID_BBOX, 100.0)
    labels = map_to_grid(_failures([(100.0, 50.0)]), grid)
    assert labels["failure_count"].sum() == 1


def test_failure_exactly_on_the_die_boundary_is_not_lost():
    """The outer edge is closed, or a corner failure would belong nowhere."""
    grid = build_grid(GRID_BBOX, 100.0)
    for point in [(1000.0, 1000.0), (0.0, 0.0), (1000.0, 500.0)]:
        labels = map_to_grid(_failures([point]), grid)
        assert labels["failure_count"].sum() == 1, f"{point} was dropped"


def test_overlapping_grid_credits_a_failure_to_every_containing_cell():
    grid = build_grid(GRID_BBOX, 100.0, stride_um=50.0)
    labels = map_to_grid(_failures([(275.0, 275.0)]), grid)
    assert labels["failure_count"].sum() == 4


def test_explicit_radius_still_selects_the_circular_test():
    """The circular behaviour remains available, just not as the default."""
    grid = build_grid(GRID_BBOX, 100.0)
    corner = _failures([(1.0, 1.0)])
    assert map_to_grid(corner, grid, radius_um=50.0)["failure_count"].sum() == 0
    assert map_to_grid(corner, grid)["failure_count"].sum() == 1


def test_distance_to_nearest_failure_stays_euclidean():
    grid = build_grid(GRID_BBOX, 100.0)
    labels = map_to_grid(_failures([(0.0, 0.0)]), grid)
    assert labels["distance_to_nearest_failure"][0] == pytest.approx(
        np.hypot(50.0, 50.0))


# ---- failure CSV values --------------------------------------------

@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_coordinates_are_rejected(tmp_path, bad):
    """One NaN coordinate makes distance_to_nearest_failure NaN everywhere."""
    path = _write(tmp_path, "nan.csv", x_um=[bad])
    with pytest.raises(ValueError, match="non-finite"):
        load_failures(path)


def test_negative_position_sigma_is_rejected(tmp_path):
    """It would produce a negative scale floor and certify every scale."""
    path = _write(tmp_path, "sigma.csv", position_sigma_um=[-40.0])
    with pytest.raises(ValueError, match="negative"):
        load_failures(path)


@pytest.mark.parametrize("value", [-0.5, 1.5])
def test_confidence_outside_the_unit_interval_is_rejected(tmp_path, value):
    path = _write(tmp_path, "conf.csv", confidence=[value])
    with pytest.raises(ValueError, match="confidence"):
        load_failures(path)


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_sample_id_is_rejected(tmp_path, value):
    path = _write(tmp_path, "id.csv", sample_id=[value])
    with pytest.raises(ValueError, match="sample_id"):
        load_failures(path)


def test_a_valid_file_still_loads(tmp_path):
    path = _write(tmp_path, "ok.csv", confidence=[0.8], position_sigma_um=[20.0])
    fs = load_failures(path)
    assert len(fs) == 1
    assert fs.position_sigma_um == 20.0
    assert fs.min_trustworthy_scale_um() == pytest.approx(60.0)


def test_missing_optional_columns_are_defaulted_not_rejected(tmp_path):
    path = _write(tmp_path, "sparse.csv")
    fs = load_failures(path)
    assert (fs.table["confidence"] == 1.0).all()
    assert np.isnan(fs.position_sigma_um)
    assert any("position_sigma_um absent" in n for n in fs.notes)


# ---- Calibre ingest ------------------------------------------------

def test_tolerance_is_a_distance_not_a_bucket():
    """Rounding both sides to a shared bucket rejects pairs inside tolerance."""
    grid = build_grid(GRID_BBOX, 100.0)          # centres at 50, 150, ...
    centre = grid.cells[0]
    for offset in (0.0, 24.0, 26.0, 49.0):       # 26 straddles a bucket edge
        df = pd.DataFrame({"x_um": [centre.x_center + offset],
                           "y_um": [centre.y_center], "value": [0.42]})
        assert to_grid(df, grid, area_conversion("m"))[0] == pytest.approx(0.42)


def test_a_record_beyond_the_tolerance_is_ignored_not_snapped():
    """A far record must not be dragged onto its nearest cell.

    Total failure to match is a frame error and raises; individual records
    outside the tolerance are dropped, which is what lets a grid cover a
    sub-region of what the deck reported.
    """
    grid = build_grid(GRID_BBOX, 100.0)
    near = grid.cells[0]
    df = pd.DataFrame({"x_um": [near.x_center, near.x_center + 40.0],
                       "y_um": [near.y_center, near.y_center],
                       "value": [0.42, 9.99]})
    out = to_grid(df, grid, area_conversion("m"), tol_um=10.0)
    assert out[0] == pytest.approx(0.42)
    assert out.sum() == pytest.approx(0.42), "the far record leaked into a cell"


def test_two_records_claiming_one_cell_is_an_error():
    """Letting the later record win leaves the other cell reading a real zero."""
    grid = build_grid(BBox(0, 0, 200, 200), 100.0)
    df = pd.DataFrame({"x_um": [40.0, 60.0], "y_um": [50.0, 50.0],
                       "value": [1.0, 2.0]})
    with pytest.raises(ValueError, match="claimed by more than one"):
        to_grid(df, grid, area_conversion("m"))


def test_nothing_matching_at_all_is_an_error():
    grid = build_grid(GRID_BBOX, 100.0)
    df = pd.DataFrame({"x_um": [50000.0], "y_um": [50000.0], "value": [0.42]})
    with pytest.raises(ValueError, match="no Calibre window matched"):
        to_grid(df, grid, area_conversion("m"))


def test_perimeter_conversion_divides_by_eps():
    from collective.calibre import perimeter_conversion
    grid = build_grid(GRID_BBOX, 100.0)
    centre = grid.cells[0]
    df = pd.DataFrame({"x_um": [centre.x_center], "y_um": [centre.y_center],
                       "value": [0.004]})
    out = to_grid(df, grid, perimeter_conversion(0.02))
    assert out[0] == pytest.approx(0.004 / 0.02)


# ---- empty results -------------------------------------------------

def test_writing_an_empty_association_frame_does_not_raise(tmp_path):
    result = pipeline.RunResult(
        associations=pd.DataFrame(), permutations=pd.DataFrame(),
        features=pd.DataFrame({"cell_id": [0]}), metadata={})
    paths = pipeline.write_results(result, tmp_path)
    assert set(paths) >= {"associations", "features", "metadata",
                          "primary", "underpowered", "summary"}


def test_pipeline_refuses_when_no_failure_lands_on_the_die(tmp_path):
    """The usual cause is coordinates that were never registered."""
    from collective.layout import validation_die
    from collective.layout import LayerSpec

    path = str(tmp_path / "die.gds")
    validation_die(path, die_um=500.0, block_um=50.0, seed=1)
    far_away = _failures([(1e6, 1e6), (1.1e6, 1e6)])
    with pytest.raises(ValueError,
                       match="outside the inspected footprint|bounding box|could be scored"):
        pipeline.run(path, far_away, layer=LayerSpec("M8", 8, 0),
                     scales_um=(100,), n_permutations=0)
