"""Routing orientation resolved against the package loading direction.

Rabie et al. (2018) recommend running the final metal diagonally under corner
bumps. That is directional, and no scalar distance to a bump can express it:
two cells the same distance from the same bump, one routed radially and one
diagonally, are identical in every other feature the engine has.
"""
import numpy as np
import pytest

from lamxsim import pipeline
from lamxsim.features.bump_relative import extract as rel_extract
from lamxsim.features.grid import build_grid
from lamxsim.features.orientation import OrientationExtractor
from lamxsim.labels.package_context import PackageLayers, extract as ctx_extract
from lamxsim.labels.simulate import failures_from_driver
from lamxsim.layout import synth
from lamxsim.layout.reader import LayerSpec, LayoutReader

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
    from lamxsim.features.geometry import GeometryExtractor

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
    from lamxsim.evidence import EvidenceClass
    from lamxsim.features import bump_relative

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
