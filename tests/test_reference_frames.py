"""The three coordinate frames, and the boundaries derived from them.

Each of these was a case where the analysis boundary came from what happened
to be in the file rather than from what a human declared, and none of them
produced an error -- only a plausible number measured from the wrong origin.
"""
import numpy as np
import pytest

from lamxsim import pipeline
from lamxsim.features.geometry import GeometryExtractor
from lamxsim.features.grid import build_grid
from lamxsim.labels.package_context import PackageLayers, extract as ctx_extract
from lamxsim.labels.simulate import failures_from_driver
from lamxsim.layout.reader import BBox, LayerSpec, LayoutReader
from lamxsim.layout.synth import packaged_die, validation_die
from lamxsim.stats.permutation import block_permutation_test

M8 = LayerSpec("M8", 8, 0)
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
    from lamxsim.report import partition

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
def packaged(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("pc") / "pkg.gds")
    _, bumps = packaged_die(path, die_um=3000.0, block_um=150.0, seed=3)
    reader = LayoutReader(path)
    grid = build_grid(reader.bbox(), 100.0)
    layers = PackageLayers(bump=LayerSpec("B", 60, 0),
                           pi_opening=LayerSpec("PI", 61, 0),
                           crackstop=LayerSpec("CS", 62, 0))
    return reader, grid, layers


def test_crackstop_distance_is_not_a_copy_of_distance_to_centre(packaged):
    """A ring's bounding-box centre is the die centre.

    Measuring to that centre made distance_to_crackstop numerically identical
    to distance-to-die-centre -- correlation 1.000000, maximum difference
    0.0000um -- so the position baseline carried the same column twice, one of
    them under a name suggesting crackstop proximity had been measured.
    """
    reader, grid, layers = packaged
    ctx = ctx_extract(grid, reader.bbox(), reader, layers)
    x = np.array([c.x_center for c in grid.cells])
    y = np.array([c.y_center for c in grid.cells])
    to_centre = np.hypot(x - 1500.0, y - 1500.0)

    d = ctx["distance_to_crackstop"]
    assert np.abs(d - to_centre).max() > 100.0
    # A rail 20um inside a 3000um die: nothing is 1500um from it.
    assert d.max() < 1500.0


def test_pi_opening_distance_is_measured_to_the_opening_edge(packaged):
    """Li et al. locate the stress concentration at the opening edge."""
    reader, grid, layers = packaged
    ctx = ctx_extract(grid, reader.bbox(), reader, layers)
    d = ctx["distance_to_nearest_pi_opening"]
    # Openings are 105um across on a 400um pitch, so no cell is further than
    # about half a pitch from one; a centroid-based distance could not bound it.
    assert d.max() < 300.0
    assert ctx["distance_to_pi_opening_corner"].min() >= d.min()


def test_pad_features_appear_only_when_a_pad_layer_is_supplied(packaged):
    reader, grid, layers = packaged
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
    from lamxsim.layout import synth

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
    from lamxsim.study import StudyManifest

    m = StudyManifest.load(MANIFEST_PATH)
    rules = m.line_rule_map()
    assert rules["M8"] == (0.20, 2.0)
    assert rules["M7"] == (0.10, 1.0)
    # The single-number form remains only as a fallback, and is the maximum.
    assert m.line_end_w_max_um() == 2.0


# ---- B2: a contradiction stops the run -----------------------------

def test_a_failure_outside_the_footprint_stops_the_run(die):
    """It disproves the population definition rather than qualifying it."""
    from lamxsim.labels.inspection import InspectionFootprint

    path, reader, fs = die
    half = InspectionFootprint.from_rectangles([(0, 0, 500, 1000)],
                                               source="left half only")
    with pytest.raises(ValueError, match="disproves the population definition"):
        pipeline.run(path, fs, layer=M8, scales_um=(100,), n_permutations=0,
                     line_end_w_max_um=4.0, footprint=half)


def test_continuing_past_the_contradiction_must_be_asserted(die):
    from lamxsim.labels.inspection import InspectionFootprint

    path, reader, fs = die
    half = InspectionFootprint.from_rectangles([(0, 0, 500, 1000)],
                                               source="left half only")
    res = pipeline.run(path, fs, layer=M8, scales_um=(100,), n_permutations=0,
                       line_end_w_max_um=4.0, footprint=half,
                       allow_failures_outside_footprint=True)
    assert any("asserted by the operator" in n and "dropped" in n
               for n in res.metadata["uncontrolled_confounding"])


def test_a_consistent_footprint_needs_no_override(die):
    from lamxsim.labels.inspection import InspectionFootprint

    path, reader, fs = die
    whole = InspectionFootprint.from_rectangles([(0, 0, 1000, 1000)],
                                                source="whole die")
    res = pipeline.run(path, fs, layer=M8, scales_um=(100,), n_permutations=0,
                       line_end_w_max_um=4.0, footprint=whole)
    assert res.metadata["failure_footprint_audit"]["consistent"]
