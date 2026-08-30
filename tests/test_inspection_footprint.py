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

from lamxsim import pipeline
from lamxsim.features.geometry import GeometryExtractor
from lamxsim.features.grid import build_grid
from lamxsim.labels.inspection import (InspectionFootprint, audit_failures,
                                       coverage, eligibility)
from lamxsim.labels.simulate import failures_from_driver
from lamxsim.layout.reader import BBox, LayerSpec, LayoutReader
from lamxsim.layout.synth import validation_die

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
    from lamxsim.labels.failure import FailureSet
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
    assert meta["inspection_footprint"]["source"] == "corner-targeted FIB"
    assert meta["failure_footprint_audit"]["consistent"]
    cov = meta["coverage_by_scale"][100.0]
    assert cov["n_eligible"] < cov["n_cells"]
    assert cov["n_cases_eligible"] > 0
