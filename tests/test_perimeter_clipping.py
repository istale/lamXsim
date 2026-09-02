"""Window-local perimeter must count metal boundary only, not the window cut."""
import pytest

from collective.geometry import GeometryExtractor
from collective import layout as synth
from collective.layout import LayerSpec, LayoutReader

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
