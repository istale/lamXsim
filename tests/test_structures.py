"""Wide metal, slotting and declared dummy fill.

Rabie et al. (2018) list wide-metal slotting among the layout levers, and fill
changes what every other feature means: it contributes to density and it sets
the shortest edge on a layer, which is what the line-end fallback would
otherwise take for a routing width.
"""
import klayout.db as db
import numpy as np
import pytest

from lamxsim.features.grid import build_grid
from lamxsim.features.structures import StructureExtractor
from lamxsim.layout import synth
from lamxsim.layout.reader import LayerSpec, LayoutReader

M8 = LayerSpec("M8", 8, 0)
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
    from lamxsim.study import StudyManifest

    p = tmp_path / "m.yaml"
    p.write_text(
        "layout:\n  metal_layers:\n    - {name: M8, layer: 8, datatype: 0}\n")
    m = StudyManifest.load(p)
    assert any("no fill_layers" in g for g in m.gaps)
    assert m.wide_width_um == 3.0
