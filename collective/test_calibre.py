"""The rule deck, its emulator, and the two paths against each other.

Folded from ``tests/test_calibre_band.py``, ``tests/test_calibre_crosscheck.py``.
"""
from __future__ import annotations
from collective import calibre as emulate
from collective import calibre as ingest
from collective import geometry as corners
from collective import layout as synth
from collective.calibre import CalibreLayer
from collective.calibre import EPS_WIDTH_FRACTION
from collective.geometry import GeometryExtractor
from collective.geometry import ViaExtractor
from collective.geometry import build_grid
from collective.layout import LayerSpec
from collective.layout import LayoutReader
from collective.study import StudyManifest
from pathlib import Path
import klayout.db as db
import numpy as np
import pytest
import tempfile


# ----------------------------------------------------------------------
# test_calibre_band.py
# ----------------------------------------------------------------------
"""Validation of the Calibre edge-band approximation, measured against exact
edge lengths from the geometry engine.

The band trick turns a length density into an area density so that Calibre's
native moving-window DENSITY can compute it. These tests fix the two things
that decide whether the resulting numbers are usable, so a change to either
assumption fails here rather than silently in a full-chip run.
"""
DBU = 0.001
M = LayerSpec("M", 8, 0)


def _u(x):
    return int(round(x / DBU))


def _lines(tmp_path, pitch, density=0.5, tile=20.0):
    sl = synth.SynthLayout()
    synth.lines(sl, 8, 0, 0, tile, tile, pitch=pitch, density=density)
    path = tmp_path / f"p{pitch}.gds"
    sl.write(str(path))
    return LayoutReader(str(path)).region(M)


def _inside_band_perimeter(region, eps):
    band = region - region.sized(-_u(eps))  # METAL NOT (SIZE METAL BY -eps)
    return band.area() * DBU * DBU / eps


def _straddle_band_perimeter(region, eps):
    band = region.sized(_u(eps)) - region.sized(-_u(eps))
    return band.area() * DBU * DBU / (2 * eps)


def _exact(region):
    return region.edges().length() * DBU


@pytest.mark.parametrize("pitch", [1.0, 0.5, 0.2, 0.1])
def test_corner_corrected_band_is_exact(tmp_path, pitch):
    """P = band_area/eps + eps*(n_convex - n_concave) recovers the true length."""
    r = _lines(tmp_path, pitch)
    eps_dbu = max(int((pitch * 0.5 * EPS_WIDTH_FRACTION) / DBU), 1)
    got = corners.corrected_band_perimeter(r, eps_dbu, DBU)
    assert got == pytest.approx(_exact(r), rel=1e-9)


@pytest.mark.parametrize("builder,kw", [
    (synth.staircase_lines, dict(pitch=1.0, density=0.4, step=2.0)),
    (synth.segmented_lines, dict(pitch=1.0, density=0.4, seg_len=3.0, gap=1.0)),
])
def test_corner_correction_holds_for_concave_and_tip_rich_geometry(tmp_path, builder, kw):
    """The uncorrected band is worst exactly where terminations are dense."""
    sl = synth.SynthLayout()
    builder(sl, 8, 0, 0, 50, 50, **kw)
    path = tmp_path / "g.gds"
    sl.write(str(path))
    r = LayoutReader(str(path)).region(M)

    eps_dbu = max(int(0.4 * EPS_WIDTH_FRACTION / DBU), 1)
    eps = eps_dbu * DBU
    exact = _exact(r)
    raw = (r - r.sized(-eps_dbu)).area() * DBU * DBU / eps
    corrected = corners.corrected_band_perimeter(r, eps_dbu, DBU)

    assert corrected == pytest.approx(exact, rel=1e-9)
    assert abs(raw - exact) / exact > 0.002, (
        "uncorrected band happened to be exact here; the correction test is vacuous"
    )


def test_uncorrected_band_error_scales_with_corner_density(tmp_path):
    """Documents why the correction is worth wiring in rather than ignoring."""
    sl = synth.SynthLayout()
    synth.segmented_lines(sl, 8, 0, 0, 50, 50, pitch=1.0, density=0.4,
                          seg_len=3.0, gap=1.0)
    path = tmp_path / "tips.gds"
    sl.write(str(path))
    r = LayoutReader(str(path)).region(M)
    eps_dbu = int(0.1 / DBU)
    raw = (r - r.sized(-eps_dbu)).area() * DBU * DBU / (eps_dbu * DBU)
    assert (_exact(r) - raw) / _exact(r) > 0.05


def test_band_collapses_once_eps_reaches_half_the_width(tmp_path):
    """The failure is a cliff, not a drift, and it is silent -- hence the guard.

    A negative size of eps >= width/2 erases the line entirely, so the whole
    conductor becomes 'band' and the reported perimeter drops by tens of
    percent with nothing in the output to indicate it.
    """
    r = _lines(tmp_path, 0.2)          # width 0.1um
    exact = _exact(r)
    safe = _inside_band_perimeter(r, 0.1 * EPS_WIDTH_FRACTION)
    over = _inside_band_perimeter(r, 0.1)   # eps == width
    assert abs(safe - exact) / exact < 0.01
    assert (exact - over) / exact > 0.20


def test_inside_band_beats_straddle_band_at_a_window_edge(tmp_path):
    """A window border on a metal edge halves the straddle band, not the inside one.

    The bar exactly fills the window in y, so its top and bottom edges lie on
    the window border. True boundary inside the window is 15 + 15 + 10 = 40um.
    """
    r = db.Region(db.Box(0, 0, _u(30), _u(10)))
    r.merge()
    win = db.Region(db.Box(0, 0, _u(15), _u(10)))
    eps = 0.02

    inside = (r - r.sized(-_u(eps)))
    straddle = (r.sized(_u(eps)) - r.sized(-_u(eps)))

    p_inside = (inside & win).area() * DBU * DBU / eps
    p_straddle = (straddle & win).area() * DBU * DBU / (2 * eps)

    assert p_inside == pytest.approx(40.0, rel=0.01)
    assert p_straddle < 25.0, "straddle band should undercount here; if not, re-derive"


def test_generated_eps_keeps_a_factor_two_margin():
    layer = CalibreLayer("M8", 8, 0, min_width_um=0.2)
    assert layer.eps_um < layer.min_width_um / 2
    assert (layer.min_width_um / 2) / layer.eps_um == pytest.approx(2.0)


def test_corner_correction_is_exact_on_a_shape_with_a_hole(tmp_path):
    """A ring exercises the hole branch of the corner classification.

    Every other benchmark shape here is simply connected, so a hole ring whose
    corner types were inverted would pass all of them. On this ring the wrong
    classification gives 8 convex and 0 concave instead of 4 and 4, and the
    correction term is off by 8*eps.
    """
    sl = synth.SynthLayout()
    synth.bench_ring(sl, 8, outer=60.0, wall=10.0)
    path = tmp_path / "ring.gds"
    sl.write(str(path))
    r = LayoutReader(str(path)).region(M)

    n_convex, n_concave = corners.counts(r)
    assert (n_convex, n_concave) == (4, 4)

    eps_dbu = int(0.1 / DBU)
    assert corners.corrected_band_perimeter(r, eps_dbu, DBU) == pytest.approx(
        _exact(r), rel=1e-9)

# ----------------------------------------------------------------------
# test_calibre_crosscheck.py
# ----------------------------------------------------------------------
"""Same die, both extraction paths, one comparison.

Until this existed the Calibre path had never been run end to end. The band
correction was measured exact on isolated patterns and the ingest code was
reviewed, but nothing put a layout in at one end and compared what came out
of both paths -- so a wrong eps, a window offset, or a marker read as a
density would have produced a plausible map with no test to notice.

The comparison runs against ``calibre.emulate``, which states in KLayout
region algebra what each generated SVRF rule means. That is enough to test
the ingest path, the conversions and the grid alignment. It is not enough to
test Calibre: if the deck's ``SIZE ... BY -eps`` does something other than
``Region.sized``, both sides are wrong together and agree.
"""
GOLDEN = Path(__file__).parent / "golden"
SCALE = 100.0


def _layers(manifest):
    rules = manifest.line_rule_map()
    out = [CalibreLayer(s.name, s.layer, s.datatype,
                        min_width_um=rules.get(s.name, (0.1, 0.0))[0] or 0.1)
           for s in manifest.metal_layers]
    out += [CalibreLayer(v.name, v.layer, v.datatype, is_via=True)
            for v in manifest.via_layers.values()]
    return out


@pytest.fixture(scope="module")
def both_paths(tmp_path_factory):
    """Run the emulated deck and the KLayout extractor on the golden die."""
    manifest = StudyManifest.load(str(GOLDEN / "golden_manifest.yaml"))
    gds = str(GOLDEN / "golden_die.gds")
    layers = _layers(manifest)
    out = tmp_path_factory.mktemp("calibre_out")
    run = emulate.run(gds, layers, scales_um=(SCALE,), step_ratio=1.0,
                      outdir=out)

    reader = LayoutReader(gds)
    bbox = reader.bbox()
    grid = build_grid(bbox, SCALE, stride_um=SCALE)
    geo = GeometryExtractor(reader, line_rules=manifest.line_rule_map())
    via = ViaExtractor(reader)

    calibre, klayout = {}, {}
    for spec in manifest.metal_layers:
        klayout[spec.name] = geo.extract(spec, grid)
        calibre[spec.name] = ingest.load_scale(
            bbox, SCALE, spec.name, run.density_files(spec.name, SCALE),
            eps_um=run.eps_um[spec.name], step_ratio=1.0,
            marker_files={k: str(p) for (l, k), p in run.markers.items()
                          if l == spec.name}).values
    for spec in manifest.via_layers.values():
        klayout[spec.name] = via.extract(spec, grid)
        calibre[spec.name] = ingest.load_scale(
            bbox, SCALE, spec.name, run.density_files(spec.name, SCALE),
            eps_um=1.0, step_ratio=1.0,
            marker_files={k: str(p) for (l, k), p in run.markers.items()
                          if l == spec.name}).values
    return reader, grid, manifest, klayout, calibre


@pytest.mark.parametrize("feature", ["metal_density", "convex_corner_density",
                                     "concave_corner_density", "corner_density"])
def test_the_two_paths_agree_exactly_on_area_and_counts(both_paths, feature):
    """No approximation is involved in these, so anything but equality is a bug."""
    _, _, manifest, klayout, calibre = both_paths
    for spec in manifest.metal_layers:
        a, b = klayout[spec.name][feature], calibre[spec.name][feature]
        assert np.abs(a - b).max() < 1e-12, f"{spec.name} {feature}"


def test_via_area_and_count_agree_exactly(both_paths):
    _, _, manifest, klayout, calibre = both_paths
    for spec in manifest.via_layers.values():
        for feature in ("via_density", "via_count_density"):
            a, b = klayout[spec.name][feature], calibre[spec.name][feature]
            assert np.abs(a - b).max() < 1e-12, f"{spec.name} {feature}"


def test_both_paths_recover_the_true_perimeter_of_the_layer(both_paths):
    """The tiled windows must sum to the perimeter the layer actually has.

    This is the check that caught the real defect. Clipping edges with a
    closed window box counted every edge lying on a shared tile border twice,
    so the tiles summed to 4.7 % (M8) and 7.0 % (M7) more perimeter than
    existed -- an inflation that depends on where the grid falls, not on the
    layout. Both paths now land within 0.01 % of the truth.
    """
    reader, grid, manifest, klayout, calibre = both_paths
    area = np.array([c.area_um2 for c in grid.cells])
    for spec in manifest.metal_layers:
        truth = reader.units.length_dbu_to_um(reader.edges(spec).length())
        for name, source in (("klayout", klayout), ("calibre", calibre)):
            total = float((source[spec.name]["perimeter_density"] * area).sum())
            assert abs(total - truth) / truth < 1e-4, (
                f"{name} {spec.name}: {total:.1f} vs {truth:.1f}")


def test_perimeter_agrees_per_window_to_within_the_boundary_error(both_paths):
    """Per window the band is not exact, and the size of that is the claim.

    A corner just outside a window still owns band area inside it, so the
    corner correction lands in the neighbouring window. The whole-layer total
    is unaffected -- that is the test above -- but a single window carries a
    low-percent error, and pinning it here means a future change that turns it
    into a large one cannot pass quietly.
    """
    _, _, manifest, klayout, calibre = both_paths
    for spec in manifest.metal_layers:
        a = klayout[spec.name]["perimeter_density"]
        b = calibre[spec.name]["perimeter_density"]
        nz = a > 0
        rel = np.abs(a - b)[nz] / a[nz]
        assert np.median(rel) < 0.005, f"{spec.name} median {np.median(rel):.4%}"
        assert rel.max() < 0.05, f"{spec.name} max {rel.max():.4%}"


def test_the_corner_correction_is_what_closes_the_gap(both_paths):
    """Without it the band is low, and the deck would still look reasonable.

    Loaded without the corner markers, so the band goes through uncorrected.
    A low total is the expected behaviour, not a defect -- the point is that
    it is low by an amount nothing else would flag, which is why the
    uncorrected value never carries the name ``perimeter_density`` when the
    corners are available.
    """
    reader, _, manifest, _, _ = both_paths
    gds = str(GOLDEN / "golden_die.gds")
    metal_only = [l for l in _layers(manifest) if not l.is_via]
    with tempfile.TemporaryDirectory() as tmp:
        run = emulate.run(gds, metal_only, scales_um=(SCALE,),
                          step_ratio=1.0, outdir=tmp)
        bbox = reader.bbox()
        for spec in manifest.metal_layers:
            truth = reader.units.length_dbu_to_um(reader.edges(spec).length())
            uncorrected = ingest.load_scale(
                bbox, SCALE, spec.name, run.density_files(spec.name, SCALE),
                eps_um=run.eps_um[spec.name], step_ratio=1.0)
            area = np.array([c.area_um2 for c in build_grid(
                bbox, SCALE, stride_um=SCALE).cells])
            total = float((uncorrected.values["perimeter_density"] * area).sum())
            assert total < truth, spec.name
            assert "perimeter_density_band_only" not in uncorrected.values


def test_an_empty_scan_is_zeros_and_a_broken_file_is_an_error(tmp_path):
    """The two used to be indistinguishable, so a format change read as empty."""
    empty = tmp_path / "empty.rdb"
    empty.write_text("// DENSITY_METAL_M1_100um\n")
    assert ingest.read_density_rdb(empty).empty

    broken = tmp_path / "broken.rdb"
    broken.write_text("// check\n0.0 0.0 not-a-number 10.0 0.5\n")
    with pytest.raises(ValueError, match="unparseable"):
        ingest.read_density_rdb(broken)


def test_the_deck_does_not_claim_to_produce_a_line_end_count(both_paths):
    """The opening measures minimum width, and is named for that.

    It was briefly carried here as a line-end proxy. On the benchmark
    patterns it returns the whole array where there are 16 line ends (opening
    by w/2 erases a line of width w), 36 pieces where there are none, and one
    connected piece where there are nine -- so under the line-end name it
    would be a number that tracks fill density.
    """
    _, _, manifest, klayout, calibre = both_paths
    for spec in manifest.metal_layers:
        assert "line_end_density" in klayout[spec.name]
        assert "line_end_density" not in calibre[spec.name]
        assert "line_end_protrusion_density" not in calibre[spec.name]


def test_the_opening_measures_minimum_width_not_line_ends(tmp_path):
    """Pinning the numbers that make the renaming a fact rather than a view."""
    from collective import geometry as lineends
    from collective import layout as synth
    from collective.layout import LayerSpec, LayoutReader

    dbu = 0.001
    cases = {}
    for name in ("continuous lines", "dummy fill array", "comb"):
        sl = synth.SynthLayout()
        truth = synth.LINE_END_BENCH[name](sl, 8)
        path = tmp_path / f"{name.replace(' ', '_')}.gds"
        sl.write(str(path))
        region = LayoutReader(str(path)).region(LayerSpec("M", 8, 0))
        opened = lineends.detect_protrusion(region, int(1.0 / dbu))
        cases[name] = (truth, len(lineends.detect(region, int(1.5 / dbu))),
                       opened.count())

    assert cases["continuous lines"][:2] == (16, 16)
    assert cases["dummy fill array"][:2] == (0, 0)
    assert cases["comb"][:2] == (9, 9)
    # The opening disagrees with all three, in three different directions.
    assert cases["dummy fill array"][2] == 36     # 0 line ends, 36 pieces
    assert cases["comb"][2] == 1                  # 9 line ends, 1 piece


def test_the_atlas_is_the_same_whichever_path_extracted_it(tmp_path):
    """The claim that matters: swapping the extractor must not move a finding.

    Feature-level agreement is necessary but not sufficient. The channels rank
    within the die, so a sub-percent shift in one map can still move a cell
    across the 95th-percentile line -- which is the level a reader acts on.
    """
    from collective import exposure as atlas_mod

    manifest = StudyManifest.load(str(GOLDEN / "golden_manifest.yaml"))
    gds = str(GOLDEN / "golden_die.gds")
    out = tmp_path / "deck_out"
    # step_ratio 1.0 because the atlas grids are non-overlapping; a deck that
    # stepped differently is rejected rather than silently re-binned.
    emulate.run(gds, _layers(manifest), scales_um=tuple(manifest.scales_um),
                step_ratio=1.0, outdir=out)

    from_python = atlas_mod.build(gds, manifest)
    from_deck = atlas_mod.build(gds, manifest, calibre_dir=str(out))

    key = ["channel", "layer", "scale_um", "x_um", "y_um"]
    assert (set(map(tuple, from_python.candidates[key].values))
            == set(map(tuple, from_deck.candidates[key].values)))
    assert from_deck.metadata["feature_source"] == "calibre"
    assert from_deck.metadata["calibre"]["emulated"] is True


def test_a_deck_stepped_differently_from_the_grid_is_refused(tmp_path):
    """Silently re-binning would attribute values to the wrong cells."""
    from collective import exposure as atlas_mod

    manifest = StudyManifest.load(str(GOLDEN / "golden_manifest.yaml"))
    gds = str(GOLDEN / "golden_die.gds")
    out = tmp_path / "half_step"
    emulate.run(gds, _layers(manifest), scales_um=tuple(manifest.scales_um),
                step_ratio=0.5, outdir=out)
    with pytest.raises(ValueError, match="do not line up"):
        atlas_mod.build(gds, manifest, calibre_dir=str(out))


def test_deck_output_without_its_extraction_manifest_is_refused(tmp_path):
    """eps cannot be guessed: the wrong one is a factor of 20 to 40."""
    (tmp_path / "metal_density_M8_100um.rdb").write_text("// x\n")
    with pytest.raises(FileNotFoundError, match="extraction_manifest"):
        ingest.discover(tmp_path)


def test_a_deck_missing_a_manifest_scale_is_refused(tmp_path):
    """Half the atlas from each path, with nothing saying which half.

    Caught by the contract gate, which sees the deck declaring one scale where
    the manifest asks for two; the completeness gate would catch the same run
    on the missing files if the deck had claimed both.
    """
    from collective import exposure as atlas_mod

    manifest = StudyManifest.load(str(GOLDEN / "golden_manifest.yaml"))
    gds = str(GOLDEN / "golden_die.gds")
    out = tmp_path / "one_scale"
    emulate.run(gds, _layers(manifest), scales_um=(float(manifest.scales_um[0]),),
                step_ratio=1.0, outdir=out)
    with pytest.raises(ValueError, match="scales") as excinfo:
        atlas_mod.build(gds, manifest, calibre_dir=str(out))
    assert "250" in str(excinfo.value)


def test_a_partial_deck_directory_is_refused(tmp_path):
    """It used to succeed and report itself as a Calibre extraction.

    A directory holding one corner file produced a run whose header said
    ``extraction: calibre`` and whose density, perimeter and via maps had all
    come from KLayout, with nothing in the output saying so.
    """
    from collective import exposure as atlas_mod

    manifest = StudyManifest.load(str(GOLDEN / "golden_manifest.yaml"))
    gds = str(GOLDEN / "golden_die.gds")
    full = tmp_path / "full"
    emulate.run(gds, _layers(manifest), scales_um=tuple(manifest.scales_um),
                step_ratio=1.0, outdir=full)

    partial = tmp_path / "partial"
    partial.mkdir()
    for name in ("extraction_manifest.json", "convex_corner_M8.rdb"):
        (partial / name).write_bytes((full / name).read_bytes())
    with pytest.raises(ValueError, match="incomplete"):
        atlas_mod.build(gds, manifest, calibre_dir=str(partial))


def test_the_eps_guard_is_a_gate_and_not_a_note(tmp_path):
    """It was generated into every deck and consumed by nobody.

    The CLI told a human the checks must come back empty. A precondition
    checked by a human is not part of an evidence chain: the result is now a
    required file, and a non-empty one stops the run.
    """
    from collective import exposure as atlas_mod

    manifest = StudyManifest.load(str(GOLDEN / "golden_manifest.yaml"))
    gds = str(GOLDEN / "golden_die.gds")
    out = tmp_path / "deck"
    emulate.run(gds, _layers(manifest), scales_um=tuple(manifest.scales_um),
                step_ratio=1.0, outdir=out)

    guard = out / "eps_violation_M8.rdb"
    assert guard.exists() and not len(ingest.read_marker_rdb(guard))
    atlas_mod.build(gds, manifest, calibre_dir=str(out))       # passes clean

    guard.write_text("// EPS_VIOLATION_M8\n10.0 10.0 10.02 10.02\n")
    with pytest.raises(ValueError, match="narrower than the declared"):
        atlas_mod.build(gds, manifest, calibre_dir=str(out))

    guard.unlink()
    with pytest.raises(FileNotFoundError, match="eps_violation_M8"):
        atlas_mod.build(gds, manifest, calibre_dir=str(out))


def test_the_guard_uses_the_projected_metric(tmp_path):
    """A guard that fires on every layout is a guard that gets switched off.

    Measured corner to corner, every re-entrant corner is a pair of edges a
    vanishing distance apart: 177 violations on this die, where an opening
    confirms nothing is actually narrow.
    """
    import klayout.db as db

    from collective.layout import LayerSpec, LayoutReader

    reader = LayoutReader(str(GOLDEN / "golden_die.gds"))
    region = reader.region(LayerSpec("M8", 8, 0))
    w = reader.units.um_to_dbu(0.2)
    assert region.width_check(w, False, db.Region.Euclidian).count() == 177
    assert region.width_check(w, False, db.Region.Projection).count() == 0
    half = reader.units.um_to_dbu(0.1)
    assert (region - region.sized(-half).sized(half)).count() == 0


def test_every_deck_output_path_is_a_real_path(tmp_path):
    """The emulator writes files directly, so it cannot see a bad deck path.

    EPS_VIOLATION_* was emitted with a literal ``{outdir}`` -- an f-string
    brace escaped one level too far -- while every other DFM RDB line expanded
    correctly. On a real Calibre run the guard would have landed in a
    directory named ``{outdir}`` and the ingest gate would have refused every
    run for a missing file. The whole suite passed, because nothing read the
    generated text.
    """
    from collective import calibre as svrf

    manifest = StudyManifest.load(str(GOLDEN / "golden_manifest.yaml"))
    results = str(tmp_path / "results")
    deck = svrf.generate(_layers(manifest), scales_um=tuple(manifest.scales_um),
                         outdir=results)

    rdb_lines = [l for l in deck.splitlines() if l.startswith("DFM RDB ")]
    assert rdb_lines, "the deck writes nothing out"
    for line in rdb_lines:
        assert "{" not in line and "}" not in line, line
        assert results in line, line
    # and the guard is among them, since that is the one that was wrong
    assert any("eps_violation_M8.rdb" in l for l in rdb_lines)


def test_deck_output_from_another_layout_is_refused(tmp_path):
    """A complete set from another revision passes every other check.

    Same layer names, same scales, same coordinates -- so the completeness
    gate is satisfied, and the density maps would then be mixed with
    orientation, gradient and package-context maps computed from the layout
    actually loaded. The result is internally consistent and describes two
    different chips.
    """
    import klayout.db as db

    from collective import exposure as atlas_mod

    manifest = StudyManifest.load(str(GOLDEN / "golden_manifest.yaml"))
    gds = str(GOLDEN / "golden_die.gds")
    out = tmp_path / "deck"
    emulate.run(gds, _layers(manifest), scales_um=tuple(manifest.scales_um),
                step_ratio=1.0, outdir=out, manifest=manifest)
    atlas_mod.build(gds, manifest, calibre_dir=str(out))       # matching: fine

    # A revision of the same design: same top cell, same bounding box, same
    # layers, one shape different. Every check other than the binding passes.
    layout = db.Layout()
    layout.read(gds)
    top = layout.top_cell()
    top.shapes(layout.layer(8, 0)).insert(db.Box(700000, 700000, 700400, 700400))
    other = tmp_path / "revision.gds"
    layout.write(str(other))

    with pytest.raises(ValueError, match="not produced from this layout"):
        atlas_mod.build(str(other), manifest, calibre_dir=str(out))


def test_deck_output_with_no_binding_at_all_is_refused(tmp_path):
    """Absence of a binding is not a pass."""
    import json

    from collective import exposure as atlas_mod

    manifest = StudyManifest.load(str(GOLDEN / "golden_manifest.yaml"))
    gds = str(GOLDEN / "golden_die.gds")
    out = tmp_path / "deck"
    emulate.run(gds, _layers(manifest), scales_um=tuple(manifest.scales_um),
                step_ratio=1.0, outdir=out, manifest=manifest)

    side = out / "extraction_manifest.json"
    data = json.loads(side.read_text())
    data["binding"] = {}
    side.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="no layout binding"):
        atlas_mod.build(gds, manifest, calibre_dir=str(out))


def test_a_deck_built_against_different_layer_rules_is_refused(tmp_path):
    """Recording traceability is not the same as gating on it.

    The sidecar carried the manifest digest and nothing compared it. A deck
    generated at min_width 0.2um (eps 0.05um) was accepted by a run whose
    manifest declared 0.4um, and the metadata then reported both numbers.

    The gate is on what the deck depends on -- layer identity, minimum width,
    scales -- not on the digest: an unrelated manifest edit changes the digest
    while changing nothing the deck measured, and a gate that fires on those
    gets worked around.
    """
    import yaml

    from collective import exposure as atlas_mod

    manifest_path = GOLDEN / "golden_manifest.yaml"
    manifest = StudyManifest.load(str(manifest_path))
    gds = str(GOLDEN / "golden_die.gds")
    out = tmp_path / "deck"
    emulate.run(gds, _layers(manifest), scales_um=tuple(manifest.scales_um),
                step_ratio=1.0, outdir=out, manifest=manifest)

    raw = yaml.safe_load(manifest_path.read_text())
    raw["layout"]["line_rules"]["M8"]["min_width_um"] = 0.4
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="different layer rules") as excinfo:
        atlas_mod.build(gds, StudyManifest.load(changed), calibre_dir=str(out))
    assert "0.2um" in str(excinfo.value) and "0.4um" in str(excinfo.value)

    # An edit the deck does not depend on must not invalidate it.
    raw = yaml.safe_load(manifest_path.read_text())
    raw.setdefault("sample_conditions", {}).setdefault("fixed", {})
    raw["sample_conditions"]["fixed"]["emc_thickness_um"] = 400.0
    unrelated = tmp_path / "unrelated.yaml"
    unrelated.write_text(yaml.safe_dump(raw))
    atlas_mod.build(gds, StudyManifest.load(unrelated), calibre_dir=str(out))
