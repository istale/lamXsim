"""Boundary handling for data arriving from outside the pipeline.

Every case here is one where the previous behaviour produced a plausible
number rather than an error. That is the failure mode worth testing for: a
crash gets noticed, a silently biased label set does not.
"""
import numpy as np
import pandas as pd
import pytest

from lamxsim import pipeline
from lamxsim.calibre.ingest import area_conversion, to_grid
from lamxsim.features.grid import build_grid
from lamxsim.labels.failure import FailureSet, load_failures, map_to_grid
from lamxsim.layout.reader import BBox

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
    from lamxsim.calibre.ingest import perimeter_conversion
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
    from lamxsim.layout.synth import validation_die
    from lamxsim.layout.reader import LayerSpec

    path = str(tmp_path / "die.gds")
    validation_die(path, die_um=500.0, block_um=50.0, seed=1)
    far_away = _failures([(1e6, 1e6), (1.1e6, 1e6)])
    with pytest.raises(ValueError,
                       match="outside the inspected footprint|bounding box|could be scored"):
        pipeline.run(path, far_away, layer=LayerSpec("M8", 8, 0),
                     scales_um=(100,), n_permutations=0)
