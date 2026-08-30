"""Per-object shape descriptors, before anything is averaged into a window.

Every other feature in this repository measures a window. That is the right
unit for routing and the wrong one for a bump: a pad's aspect ratio is a
property of the pad, and two pads of equal area and opposite elongation
produce the same window mean. So these tests work on shapes whose descriptors
are known by construction, and check the object table before the grid sees it.

Everything here is **drawn** geometry in plan view. None of it is the
post-reflow bump, the printed opening, the assembled overlay, or any sidewall
angle -- a GDS holds no Z information, so no vertical angle is derivable from
it by any means.
"""
import math

import numpy as np
import pytest

from lamxsim.features import objects as obj
from lamxsim.layout.reader import BBox, LayerSpec, LayoutReader
from lamxsim.layout import synth

DIE = BBox(0.0, 0.0, 200.0, 200.0)


def _octagon(cx, cy, r, *, rotation=0.0):
    return [(cx + r * math.cos(rotation + k * math.pi / 4),
             cy + r * math.sin(rotation + k * math.pi / 4)) for k in range(8)]


def _write(tmp_path, name, shapes_by_layer):
    """One GDS holding the given polygons, per layer number."""
    import klayout.db as db

    layout = db.Layout()
    layout.dbu = 0.001
    top = layout.create_cell("TOP")
    for number, shapes in shapes_by_layer.items():
        li = layout.layer(number, 0)
        for pts in shapes:
            top.shapes(li).insert(db.Polygon([
                db.Point(int(x / layout.dbu), int(y / layout.dbu))
                for x, y in pts]))
    path = tmp_path / name
    layout.write(str(path))
    return LayoutReader(str(path))


def _square(cx, cy, half):
    return [(cx - half, cy - half), (cx + half, cy - half),
            (cx + half, cy + half), (cx - half, cy + half)]


def _rect(cx, cy, hx, hy):
    return [(cx - hx, cy - hy), (cx + hx, cy - hy),
            (cx + hx, cy + hy), (cx - hx, cy + hy)]


def test_bump_descriptors_are_per_object(tmp_path):
    """Two bumps of equal area and opposite elongation must not agree.

    This is the whole reason the object table exists: the window mean of the
    two is identical, so a grid-first pipeline cannot tell them apart no
    matter what it computes afterwards.
    """
    reader = _write(tmp_path, "bumps.gds", {80: [
        _rect(50, 50, 20, 5),      # 40 x 10, elongated along x
        _rect(150, 150, 5, 20),    # 10 x 40, the same area, along y
        _square(50, 150, math.sqrt(200) / 2),   # equal area, square
    ]})
    bumps = obj.objects_for(reader, LayerSpec("BUMP", 80, 0), kind="bump",
                            polarity="positive", die_bbox=DIE)
    assert len(bumps) == 3
    # The square's half-width is irrational, so it lands on the database unit
    # grid rather than exactly where it was asked for -- 1e-4 relative.
    areas = sorted(round(b.area_um2, 3) for b in bumps)
    assert areas == pytest.approx([200.0, 400.0, 400.0], rel=2e-4)

    wide = next(b for b in bumps if b.x_um < 100 and b.y_um < 100)
    tall = next(b for b in bumps if b.x_um > 100)
    # Long side over short side of the minimum-area rectangle, so a 40x10 bar
    # is 4.0 rather than the 4.12 a caliper ratio would give (its maximum
    # Feret diameter is the diagonal).
    assert wide.aspect_ratio == pytest.approx(4.0, rel=1e-6)
    assert tall.aspect_ratio == pytest.approx(4.0, rel=1e-6)
    assert wide.feret_max_um == pytest.approx(math.hypot(40, 10), abs=1e-3)
    assert wide.feret_min_um == pytest.approx(10.0, abs=1e-3)
    # Same aspect, orthogonal axes: the descriptor that separates them is the
    # orientation, which is why it is kept.
    assert abs(wide.principal_axis_rad - tall.principal_axis_rad) == \
        pytest.approx(math.pi / 2, abs=1e-6)

    # Equivalent diameter is the circle of equal area, and nothing more.
    assert wide.equivalent_diameter_um == pytest.approx(
        2 * math.sqrt(400.0 / math.pi), rel=1e-9)


def test_a_square_has_no_orientation_and_says_so(tmp_path):
    """Reporting an axis from rounding noise is worse than reporting none."""
    reader = _write(tmp_path, "square.gds", {80: [_square(100, 100, 10)]})
    square = obj.objects_for(reader, LayerSpec("BUMP", 80, 0), kind="bump",
                             polarity="positive", die_bbox=DIE)[0]
    assert not np.isfinite(square.principal_axis_rad)
    assert "no long axis" in square.orientation_undefined_reason

    octagon = obj.describe(
        _write(tmp_path, "oct.gds", {80: [_octagon(100, 100, 10)]})
        .region(LayerSpec("BUMP", 80, 0)).each().__next__(),
        object_id="bump:0", kind="bump", source_layer="BUMP(80/0)",
        polarity="positive", dbu=0.001, die_bbox=DIE)
    assert not np.isfinite(octagon.principal_axis_rad)


def test_placement_angle_is_measured_about_the_die_centre(tmp_path):
    reader = _write(tmp_path, "place.gds", {80: [
        _square(150, 100, 5), _square(100, 150, 5), _square(50, 100, 5)]})
    bumps = {round(b.x_um): b for b in obj.objects_for(
        reader, LayerSpec("BUMP", 80, 0), kind="bump", polarity="positive",
        die_bbox=DIE)}
    assert bumps[150].placement_angle_rad == pytest.approx(0.0, abs=1e-9)
    assert bumps[100].placement_angle_rad == pytest.approx(math.pi / 2, abs=1e-9)
    assert abs(bumps[50].placement_angle_rad) == pytest.approx(math.pi, abs=1e-9)
    assert bumps[150].radial_distance_um == pytest.approx(50.0, rel=1e-9)

    # Without a die frame there is no angle to measure, and none is invented.
    loose = obj.objects_for(reader, LayerSpec("BUMP", 80, 0), kind="bump",
                            polarity="positive", die_bbox=None)
    assert all(not np.isfinite(b.placement_angle_rad) for b in loose)


def test_pad_departure_measures_the_declared_target(tmp_path):
    """An octagon is at 135 degrees; a square is 45 degrees away from it.

    The target comes from the manifest because nothing in a layout says which
    pad shape a process recommends -- the departure is from a declared
    reference, not from a risk.
    """
    reader = _write(tmp_path, "pads.gds", {
        81: [_octagon(60, 60, 10)],
        82: [_square(140, 140, 10)]})
    octagonal = obj.objects_for(reader, LayerSpec("PAD", 81, 0), kind="pad",
                                polarity="positive", die_bbox=DIE)[0]
    square = obj.objects_for(reader, LayerSpec("PAD", 82, 0), kind="pad",
                             polarity="positive", die_bbox=DIE)[0]

    assert obj.corner_angle_departure(octagonal, 135.0) == pytest.approx(0.0, abs=0.5)
    assert obj.target_corner_fraction(octagonal, 135.0) == pytest.approx(1.0)
    assert obj.corner_angle_departure(square, 135.0) == pytest.approx(45.0, abs=0.5)
    assert obj.target_corner_fraction(square, 135.0) == 0.0


def test_pi_shape_is_plan_view_only(tmp_path):
    """Elongation is measurable; a taper angle is not, at any price."""
    reader = _write(tmp_path, "pi.gds", {
        61: [_rect(60, 60, 20, 5), _octagon(140, 140, math.sqrt(400 / 2.828))]})
    openings = obj.objects_for(reader, LayerSpec("PI", 61, 0),
                               kind="pi_opening", polarity="opening",
                               die_bbox=DIE)
    elongated = next(o for o in openings if o.x_um < 100)
    round_ish = next(o for o in openings if o.x_um > 100)
    assert elongated.aspect_ratio > 3.0
    assert round_ish.aspect_ratio == pytest.approx(1.0, abs=0.1)
    assert round_ish.n_convex_corners == 8 and round_ish.n_concave_corners == 0
    assert round_ish.circularity > elongated.circularity
    for o in openings:
        assert o.polarity == "opening"
        assert "no Z information" in o.definitions["angles"]


def test_matching_records_its_rule_and_its_doubt(tmp_path):
    """An unmatched pad produces no row, and an ambiguous one says so.

    A pad with no bump over it is not a pad with a perfectly concentric bump,
    so a missing match must not become a zero offset.
    """
    reader = _write(tmp_path, "match.gds", {
        81: [_square(50, 50, 10), _square(150, 50, 10), _square(50, 150, 10)],
        80: [_square(52, 51, 6), _square(150, 50, 6)]})   # third pad has none
    pads = obj.objects_for(reader, LayerSpec("PAD", 81, 0), kind="pad",
                           polarity="positive", die_bbox=DIE)
    bumps = obj.objects_for(reader, LayerSpec("BUMP", 80, 0), kind="bump",
                            polarity="positive", die_bbox=DIE)

    matches = obj.match(pads, bumps, rule="containment", die_bbox=DIE,
                        regions=(reader.region(LayerSpec("PAD", 81, 0)),
                                 reader.region(LayerSpec("BUMP", 80, 0))))
    assert len(matches) == 2, "the unmatched pad must not produce a row"
    offset = {m.primary_id: m for m in matches}
    concentric = next(m for m in matches if m.centroid_offset_um == 0.0)
    assert concentric.overlap_fraction == pytest.approx(0.36, rel=0.05)
    shifted = next(m for m in matches if m.centroid_offset_um > 0)
    # Quantised to the database unit: a GDS cannot express a finer offset,
    # and leaving the arithmetic tail in gets it ranked.
    assert shifted.centroid_offset_um == pytest.approx(math.hypot(2, 1), abs=1e-3)
    assert all(m.rule == "containment" for m in matches)

    with pytest.raises(ValueError, match="unknown object matching rule"):
        obj.match(pads, bumps, rule="whatever_is_closest", die_bbox=DIE)


def test_crackstop_structure_separates_width_count_and_gaps(tmp_path):
    """A distance cannot express any of these, which is why they exist.

    A ring's bounding box is the die, so a centre-based measure of it is
    numerically the distance to the die centre -- a different feature arriving
    under a name that suggests the seal ring was measured.
    """
    import klayout.db as db

    def ring(outer, wall):
        o, w = outer, wall
        return (_square(100, 100, o), _square(100, 100, o - w))

    layout = db.Layout()
    layout.dbu = 0.001
    top = layout.create_cell("TOP")
    li = layout.layer(62, 0)

    def insert(outer, wall, layer_index):
        out_pts, in_pts = ring(outer, wall)
        region = db.Region(db.Polygon([db.Point(int(x / layout.dbu),
                                                int(y / layout.dbu))
                                       for x, y in out_pts]))
        region -= db.Region(db.Polygon([db.Point(int(x / layout.dbu),
                                                 int(y / layout.dbu))
                                        for x, y in in_pts]))
        top.shapes(layer_index).insert(region)

    insert(90, 4, li)                       # single 4um rail
    single = tmp_path / "single.gds"
    layout.write(str(single))

    li2 = layout.layer(63, 0)
    insert(80, 8, li2)                      # a second, wider rail
    insert(90, 8, li2)
    double = tmp_path / "double.gds"
    layout.write(str(double))

    thin = obj.crackstop_structure(LayoutReader(str(single)),
                                   LayerSpec("CS", 62, 0), DIE)
    wide = obj.crackstop_structure(LayoutReader(str(double)),
                                   LayerSpec("CS", 63, 0), DIE)
    assert thin.n_rails == 1
    assert thin.rail_width_min_um == pytest.approx(4.0, abs=0.2)
    assert wide.n_rails == 2
    assert wide.rail_width_min_um == pytest.approx(8.0, abs=0.3)
    assert thin.n_gaps == 0 and thin.continuity_ratio == pytest.approx(1.0)


def test_a_cut_ring_is_reported_as_segmented(tmp_path):
    import klayout.db as db

    layout = db.Layout()
    layout.dbu = 0.001
    top = layout.create_cell("TOP")
    li = layout.layer(62, 0)
    u = 1000

    def box(x0, y0, x1, y1):
        top.shapes(li).insert(db.Box(x0 * u, y0 * u, x1 * u, y1 * u))

    box(10, 10, 190, 14)          # bottom rail
    box(10, 186, 190, 190)        # top rail, separate: the ring is cut
    structure = obj.crackstop_structure(LayoutReader(str(_write_layout(
        layout, tmp_path / "cut.gds"))), LayerSpec("CS", 62, 0), DIE)
    assert structure.n_components == 2
    assert structure.n_gaps == 1
    assert structure.continuity_ratio < 1.0


def _write_layout(layout, path):
    layout.write(str(path))
    return path


def test_rasterising_leaves_empty_cells_undefined(tmp_path):
    """Zero is a value; "there is no pad here" is not that value."""
    from lamxsim.features.grid import build_grid

    reader = _write(tmp_path, "sparse.gds", {81: [_square(25, 25, 5)]})
    pads = obj.objects_for(reader, LayerSpec("PAD", 81, 0), kind="pad",
                           polarity="positive", die_bbox=DIE)
    grid = build_grid(BBox(0.0, 0.0, 100.0, 100.0), 50.0)
    out = obj.rasterise(pads, grid, {
        "aspect_ratio": np.array([p.aspect_ratio for p in pads])},
        prefix="pad")
    values = out["pad_aspect_ratio"]
    assert np.isfinite(values).sum() == 1
    assert np.nanmax(values) == pytest.approx(1.0)   # a square pad
    assert out["pad_count"].sum() == 1


def _shape_manifest(path, *, die_um=1200.0, targets=True):
    import yaml

    layout = {
        "top_cell": "TOP",
        "metal_layers": [{"name": "M8", "layer": 8, "datatype": 0},
                         {"name": "M7", "layer": 7, "datatype": 0}],
        "die_outline_um": [0, 0, die_um, die_um],
        "wide_width_um": 3.0,
        "line_rules": {"M8": {"min_width_um": 0.2, "line_max_width_um": 2.0},
                       "M7": {"min_width_um": 0.2, "line_max_width_um": 2.0}},
        "package_layers": {
            "bump": {"name": "BUMP", "layer": 60, "datatype": 0},
            "pad": {"name": "PAD", "layer": 64, "datatype": 0},
            "pi_opening": {"name": "PI", "layer": 61, "datatype": 0,
                           "polarity": "opening"},
            "crackstop": {"name": "CS", "layer": 62, "datatype": 0}},
        "object_matching": "containment",
    }
    if targets:
        layout["shape_targets"] = {"pad_corner_angle_deg": 135.0,
                                   "pi_plan_view_corner_angle_deg": 90.0}
    yaml.safe_dump({"layout": layout, "analysis": {"scales_um": [150]}},
                   open(path, "w"))
    return str(path)


def test_the_shape_channels_flag_the_site_whose_shape_differs(tmp_path):
    """End to end: one odd pad and opening out of sixteen sites.

    Half-and-half would put half the cells at the top value, which is the top
    50 % and not the top 5 %, so the fixture leaves one genuine extreme.
    """
    from lamxsim import atlas
    from lamxsim.study import StudyManifest

    gds = synth.shape_variation_die(str(tmp_path / "shapes.gds"),
                                    odd_sites=((0, 0),))
    manifest = StudyManifest.load(_shape_manifest(tmp_path / "m.yaml"))
    result = atlas.build(gds, manifest)

    pad_rows = result.candidates[
        result.candidates.channel == "pad_geometry_departure"]
    assert not pad_rows.empty, "the pad lever found nothing on a varied die"
    assert (pad_rows.x_um < 300).all() and (pad_rows.y_um < 300).all()

    pi_rows = result.candidates[result.candidates.channel == "pi_opening_shape"]
    assert not pi_rows.empty
    assert (pi_rows.x_um < 300).all() and (pi_rows.y_um < 300).all()

    # The crackstop is one fact about the whole ring, so every cell carries the
    # same value and the within-die ranking reports nothing. That is the
    # channel working, and it says so rather than staying silent.
    reasons = [r.reason for cs in result.channels.values() for _, r in cs
               if r.channel.channel_id == "crackstop_structure"]
    assert reasons and all("too few distinct values" in r for r in reasons)


def test_without_a_declared_target_the_pad_channel_says_what_is_missing(tmp_path):
    """Nothing in a layout says which pad shape a process recommends."""
    from lamxsim import atlas
    from lamxsim.study import StudyManifest

    gds = synth.shape_variation_die(str(tmp_path / "shapes.gds"))
    manifest = StudyManifest.load(
        _shape_manifest(tmp_path / "m.yaml", targets=False))
    assert any("pad_corner_angle_deg" in g for g in manifest.gaps)

    result = atlas.build(gds, manifest)
    reasons = [r.reason for cs in result.channels.values() for _, r in cs
               if r.channel.channel_id == "pad_geometry_departure"]
    assert reasons and all("pad_corner_angle_departure_deg" in r
                           for r in reasons)


def test_a_sidewall_angle_is_refused_rather_than_reinterpreted(tmp_path):
    """A GDS holds no Z information, so no vertical angle is derivable."""
    import yaml

    from lamxsim.study import StudyManifest

    path = tmp_path / "sidewall.yaml"
    raw = yaml.safe_load(open(_shape_manifest(tmp_path / "base.yaml")))
    raw["layout"]["shape_targets"]["pi_sidewall_angle_deg"] = 80.0
    yaml.safe_dump(raw, open(path, "w"))

    with pytest.raises(ValueError, match="no Z information"):
        StudyManifest.load(path)


def test_an_undeclarable_matching_rule_is_refused(tmp_path):
    import yaml

    from lamxsim.study import StudyManifest

    path = tmp_path / "rule.yaml"
    raw = yaml.safe_load(open(_shape_manifest(tmp_path / "base.yaml")))
    raw["layout"]["object_matching"] = "whatever_looks_right"
    yaml.safe_dump(raw, open(path, "w"))
    with pytest.raises(ValueError, match="object_matching"):
        StudyManifest.load(path)

    raw["layout"]["object_matching"] = "nearest"
    raw["layout"]["package_layers"]["pi_opening"]["polarity"] = "hole"
    yaml.safe_dump(raw, open(path, "w"))
    with pytest.raises(ValueError, match="polarity"):
        StudyManifest.load(path)
