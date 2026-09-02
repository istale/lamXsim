"""Inspection footprint and control opportunity.

A cell is a valid control only if inspection had a real chance to find a
failure there. Where it did not, the cell is missing data, and counting it as
clean biases the denominator -- which no downstream correction can repair,
because block permutation, FDR and the position baseline all operate on
whatever population they are given.
"""
from dataclasses import replace

import numpy as np
import pytest

from collective import workflow as pipeline
from collective.geometry import GeometryExtractor
from collective.geometry import build_grid
from collective.labels import (InspectionFootprint, audit_failures,
                               coverage, eligibility)
from collective.labels import failures_from_driver
from collective.layout import BBox, LayerSpec, LayoutReader
from collective.layout import validation_die

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
