"""Via, corner and package-context features.

These close the tier-1 gaps in references/feature_evidence_map.csv: via
density (Vanstreels 2020, Zahedmanesh 2019), corner density (Tan 2008) and
bump/PI context (Rabie 2018, Li 2023/2025). Each is validated the same way as
the earlier families -- by construction, on a layout where the quantity being
measured is known rather than inferred.
"""
import numpy as np
import pytest
from scipy import stats

from lamxsim.features.geometry import GeometryExtractor
from lamxsim.features.grid import build_grid
from lamxsim.features.vias import ViaExtractor
from lamxsim.labels.package_context import (PackageLayers, absent_context_note,
                                            extract as ctx_extract)
from lamxsim.layout import synth
from lamxsim.layout.reader import LayerSpec, LayoutReader

M8 = LayerSpec("M8", 8, 0)
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
def packaged(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("pkg") / "pkg.gds")
    _, bumps = synth.packaged_die(path, die_um=3000.0, block_um=100.0, seed=31)
    reader = LayoutReader(path)
    grid = build_grid(reader.bbox(), 100.0)
    layers = PackageLayers(bump=LayerSpec("BUMP", 60, 0),
                           pi_opening=LayerSpec("PI", 61, 0),
                           crackstop=LayerSpec("CS", 62, 0))
    return reader, grid, layers, bumps


def test_bump_distance_matches_the_known_bump_positions(packaged):
    reader, grid, layers, bumps = packaged
    ctx = ctx_extract(grid, reader.bbox(), reader, layers)
    x = np.array([c.x_center for c in grid.cells])
    y = np.array([c.y_center for c in grid.cells])
    expected = np.min(np.hypot(x[:, None] - bumps[None, :, 0],
                               y[:, None] - bumps[None, :, 1]), axis=1)
    assert np.allclose(ctx["distance_to_nearest_bump"], expected)


def test_local_bump_pitch_recovers_the_generated_pitch(packaged):
    reader, grid, layers, _ = packaged
    ctx = ctx_extract(grid, reader.bbox(), reader, layers)
    assert np.allclose(ctx["local_bump_pitch"], 400.0)


def test_radial_and_tangential_offsets_decompose_the_bump_vector(packaged):
    """The two components must reconstruct the distance to the nearest bump."""
    reader, grid, layers, _ = packaged
    ctx = ctx_extract(grid, reader.bbox(), reader, layers)
    recomposed = np.hypot(ctx["bump_radial_offset"], ctx["bump_tangential_offset"])
    assert np.allclose(recomposed, ctx["distance_to_nearest_bump"])


def test_under_bump_indicator_finds_cells_inside_a_bump(packaged):
    """A zero-area probe intersects nothing; the test is a real containment."""
    reader, grid, layers, _ = packaged
    ctx = ctx_extract(grid, reader.bbox(), reader, layers)
    flagged = ctx["under_bump_indicator"]
    assert 0 < flagged.sum() < len(flagged)
    # Every flagged cell must be closer to its bump than the bump half-width.
    assert (ctx["distance_to_nearest_bump"][flagged > 0]
            <= 150.0 / 2 * np.sqrt(2) + 1e-6).all()


def test_missing_package_layers_are_recorded_as_uncontrolled(packaged):
    reader, grid, layers, _ = packaged
    notes = absent_context_note(PackageLayers())
    assert any("UNCONTROLLED" in n for n in notes)
    assert len(notes) == 3
    assert absent_context_note(layers) == []


def test_package_context_is_position_evidence_not_geometry():
    """It comes from GDS layers but belongs in the baseline it exists to control."""
    from lamxsim.evidence import EvidenceClass
    from lamxsim.labels import package_context
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
