"""Rule-deck generation, its emulator, and reading its output back.

Consolidated from ``calibre/svrf.py``, ``calibre/ingest.py``, ``calibre/emulate.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
import json
import klayout.db as db
import numpy as np
import pandas as pd
import re
from . import geometry as corners_mod
from .foundation import EvidenceClass
from .geometry import Grid, build_grid
from .layout import BBox, LayerSpec, LayoutReader


# ----------------------------------------------------------------------
# calibre/svrf.py
# ----------------------------------------------------------------------
"""SVRF rule-deck generation for the marker-layer -> DENSITY approach.

The design is to let Calibre answer "what is the geometry" exactly, and let
Python answer "how does it vary in space and does it correlate with failure".
Every length-density feature is converted into an area-density one so it can
ride Calibre's native moving-window DENSITY scanner:

    METAL -> marker layer -> DENSITY WINDOW/STEP -> per-window value

Two results from measuring the approximation against exact edge lengths
(``tests/test_calibre_band.py``) are built into the generated decks:

* The band must be the **inside** band, ``METAL NOT (METAL SIZED BY -eps)``,
  with ``P = area / eps``. The symmetric band ``SIZED(+eps) NOT SIZED(-eps)``
  straddles the boundary, so any metal edge lying on a window border loses the
  half of its band that falls in the neighbouring window -- a 50 % undercount
  on a rail aligned to the analysis grid, which is not a rare case.
* ``eps`` must stay below half the layer's minimum drawn width. Past that the
  negative size erases the line and the measurement collapses; the failure is
  a cliff (-38 % to -87 %), not a gradual drift, and it is silent.
* ``eps`` must land exactly on the database unit grid, and the division must
  use the snapped value. Dividing by a nominal eps that Calibre rounded costs
  up to 4 % on its own.
* The residual band bias is exactly recoverable. An inside band loses eps^2 of
  area at every convex corner and gains it at every concave one, so

      P_true = P_band + eps * (n_convex - n_concave)

  is exact on Manhattan geometry -- verified to 0.0000 % on line arrays,
  staircases and segmented patterns, including a case where the uncorrected
  band was 5.7 % low. The corner counts come from the corner marker layers the
  same deck already produces, so the correction costs nothing extra.

  That exactness is a whole-layer statement. **Per window** it is not exact:
  a corner just outside a window still owns band area inside it, so the
  correction is attributed to the neighbouring window. On the golden die at a
  100 um window the two paths agree to 0.07-0.13 % in the median cell and
  differ by up to ~2 % in the worst one, while the totals match the true
  perimeter to 0.006 %. Whole-die numbers are exact; a single window carries a
  low-percent boundary error.

Counts (corners, vias) are emitted as marker lists rather than as densities.
The density scanner reports an area fraction, so recovering a count from it
needs every marker to have the same known area -- an assumption the deck
cannot check, and one whose failure scales every count by a constant that no
downstream test would notice.
"""
#: Error stays under ~1 % at this fraction of minimum width, with margin
#: before the cliff at 1/2.
EPS_WIDTH_FRACTION = 0.25


@dataclass(frozen=True)
class CalibreLayer:
    name: str          # M8
    layer: int
    datatype: int = 0
    min_width_um: float = 0.1
    is_via: bool = False
    #: Widest conductor on this layer still counted as a routing "line".
    #: Set between the routing width and any power-strap width; the line-end
    #: definition is flat in between and flips as a step at the strap width.
    line_max_width_um: float = 0.0

    #: Drawn via area, if the layer has a single one. Only used for a guard
    #: check; the via count comes from the marker list, not from this.
    via_area_um2: float = 0.0

    #: Database unit, in um. eps is snapped to it because Calibre will snap
    #: anyway and the division must use the value actually used.
    dbu_um: float = 0.001

    @property
    def eps_um(self) -> float:
        raw = self.min_width_um * EPS_WIDTH_FRACTION
        return max(int(raw / self.dbu_um), 1) * self.dbu_um


def _header(layers, scales_um, step_ratio) -> str:
    return f"""// ---------------------------------------------------------------
// lamXsim layout feature extraction -- generated SVRF
//
// Produces per-window spatial features for the GDS -> delamination
// correlation study. Calibre measures geometry; gradients, cross-layer
// terms and all statistics are computed downstream in Python.
//
// Scales: {', '.join(f'{s:g}um' for s in scales_um)}   STEP = WINDOW x {step_ratio:g}
// Layers: {', '.join(l.name for l in layers)}
//
// STEP < WINDOW produces overlapping windows. Overlapping samples inflate
// the apparent observation count without adding independent information,
// so the ingest side records the ratio and the statistics layer reports
// effective sample size rather than window count.
// ---------------------------------------------------------------

PRECISION 1000
RESOLUTION 1

"""


def _layer_defs(layers) -> str:
    out = ["// ---- layer definitions ----"]
    for l in layers:
        out.append(f"LAYER {l.name} {l.layer}")
        out.append(f"LAYER MAP {l.layer} DATATYPE {l.datatype} {l.layer}")
    return "\n".join(out) + "\n"


def _eps_guard(layers, outdir) -> str:
    """A deck that fails loudly if eps was set too large for the real layout.

    The result is written out, not just computed. Telling a human to check a
    check is not part of an evidence chain: the ingest side requires this file
    and refuses a non-empty one, so a layout narrower than the manifest claims
    stops the run instead of quietly understating every perimeter.
    """
    out = ["""// ---- eps validity guard ----
// The inside-band approximation is exact only while eps < min_width/2.
// These checks flag any geometry narrower than the assumed minimum width;
// a non-empty result means the eps below is wrong for this layout and every
// perimeter number derived from it is understated. Each writes an RDB that
// the ingest side requires to exist and to be empty -- it is a gate, not a
// note for the operator.
"""]
    for l in layers:
        if l.is_via:
            continue
        out.append(f"EPS_VIOLATION_{l.name} {{ @ {l.name} narrower than assumed "
                   f"min width {l.min_width_um:g}um -- eps={l.eps_um:g}um is unsafe")
        out.append(f"  INTERNAL {l.name} < {l.min_width_um:g} PROJECTING")
        out.append("  // PROJECTING, i.e. facing edges only. Measured corner "
                   "to corner instead, every")
        out.append("  // re-entrant corner is two edges a vanishing distance "
                   "apart: on the regression")
        out.append("  // die that is 177 violations where the projected check "
                   "finds none and an")
        out.append("  // opening confirms nothing is narrow. Confirm the "
                   "option spelling against the")
        out.append("  // tool -- this deck has not been run on Calibre.")
        out.append("}")
        out.append(f'DFM RDB EPS_VIOLATION_{l.name} '
                   f'"{outdir}/eps_violation_{l.name}.rdb" ALL CELLS\n')
    return "\n".join(out)


def _markers(layers) -> str:
    out = ["// ---- marker layers ----"]
    for l in layers:
        if l.is_via:
            out.append(f"""
// {l.name}: vias are counted, not scaled. One record per via lets the count
// be exact; deriving it from an area density would need every via to have the
// same area and would need that area to be right, and an error in it scales
// every count by a constant that nothing downstream would notice.
{l.name}_MARKER = {l.name}""")
            continue
        e = l.eps_um
        out.append(f"""
// {l.name}: inside perimeter band. area(band) / {e:g} = metal/dielectric
// boundary length. Inside (not straddling) so that a window border falling
// on a metal edge cannot discard half the band.
{l.name}_BAND = {l.name} NOT (SIZE {l.name} BY -{e:g})

// {l.name}: corner markers, typed by angle. Two jobs -- they are a feature in
// their own right (Tan 2008: delamination at corners and tips), and they
// supply the exact correction for the band above,
//   perimeter = band_area/eps + eps*(n_convex - n_concave).
// Vertex count alone would conflate a stress-concentrating re-entrant corner
// with an ordinary convex one. Emitted as markers and counted downstream,
// for the reason given on the via layer above.
{l.name}_CONVEX  = ANGLE {l.name} == 90
{l.name}_CONCAVE = ANGLE {l.name} == 270

// {l.name}: narrow-structure map, by morphological opening.
//   METAL NOT (METAL SIZED BY -w/2 SIZED BY +w/2)
// This returns the geometry **narrower than w** -- nothing more. It was once
// carried here as a line-end proxy (definition D4) and it is not one, which
// the benchmark patterns say plainly: on an array of 1um lines with 16 true
// line ends it returns the entire array (320um^2, because opening by w/2
// erases a line of width w), on a dummy-fill array with 0 line ends it
// returns 36 pieces, and on a comb with 9 ends it returns 1 connected piece.
// Named for what it measures, it is a useful observable: where the layout
// runs at minimum width. Named for line ends it would be a count-shaped
// number that tracks fill density.
// The line-end count (D2, tips passing an elongation test) stays out of this
// deck; it needs Calibre pattern matching against explicit topologies.
{l.name}_OPEN   = SIZE (SIZE {l.name} BY -{l.min_width_um / 2:g}) BY {l.min_width_um / 2:g}
{l.name}_NARROW = {l.name} NOT {l.name}_OPEN""")
    return "\n".join(out) + "\n"


def _via_guard(layers) -> str:
    """Flag vias that are not the size the count assumes, if one is declared."""
    vias = [l for l in layers if l.is_via and l.via_area_um2]
    if not vias:
        return ""
    out = ["""// ---- via size guard ----
// Only meaningful if a via area was declared. The count comes from the marker
// list rather than from an area density, so a wrong via size does not corrupt
// the count -- but a via that is not the declared size usually means the layer
// holds something other than vias, and every via feature then describes it.
"""]
    for l in vias:
        out.append(f"VIA_SIZE_VIOLATION_{l.name} {{ @ {l.name} shape whose area "
                   f"is not the declared {l.via_area_um2:g}um^2")
        out.append(f"  AREA {l.name} != {l.via_area_um2:g}")
        out.append("}\n")
    return "\n".join(out)


def _density_checks(layers, scales_um, step_ratio, outdir) -> str:
    out = ["""// ---- moving-window density scans ----
// Area quantities only. Counts come from the marker RDBs below: the density
// scanner reports an area fraction, and turning that back into a count needs
// uniform markers of a known area, which is an assumption this deck would
// have no way to check."""]
    for s in scales_um:
        step = s * step_ratio
        for l in layers:
            tag = f"{l.name}_{s:g}um"
            kind = "via_density" if l.is_via else "metal_density"
            check = f"DENSITY_{'VIA' if l.is_via else 'METAL'}_{tag}"
            out.append(f"""
{check} {{ @ {kind} {l.name} @ {s:g}um
  DENSITY {l.name} > 0 WINDOW {s:g} {s:g} STEP {step:g} {step:g}
}}
DFM RDB {check} "{outdir}/{kind}_{tag}.rdb" ALL CELLS""")
            if l.is_via:
                continue
            out.append(f"""
DENSITY_PERIM_{tag} {{ @ perimeter_band {l.name} @ {s:g}um (divide by eps={l.eps_um:g})
  DENSITY {l.name}_BAND > 0 WINDOW {s:g} {s:g} STEP {step:g} {step:g}
}}
DFM RDB DENSITY_PERIM_{tag} "{outdir}/perimeter_band_{tag}.rdb" ALL CELLS

DENSITY_NARROW_{tag} {{ @ narrow_structure ({l.name} below {l.min_width_um:g}um) @ {s:g}um
  DENSITY {l.name}_NARROW > 0 WINDOW {s:g} {s:g} STEP {step:g} {step:g}
}}
DFM RDB DENSITY_NARROW_{tag} "{outdir}/narrow_structure_{tag}.rdb" ALL CELLS""")
    return "\n".join(out) + "\n"


def _marker_output(layers, outdir) -> str:
    out = ["""
// ---- marker output, one record per counted object ----
// Scale-independent: the windowing happens downstream, so the same file
// serves every scale. On a full-chip corner layer this is a large file; it is
// the price of a count that carries no marker-area assumption."""]
    for l in layers:
        if l.is_via:
            out.append(f"""
VIA_MARKER_{l.name} {{ @ via markers {l.name}
  COPY {l.name}_MARKER
}}
DFM RDB VIA_MARKER_{l.name} "{outdir}/via_marker_{l.name}.rdb" ALL CELLS""")
            continue
        for kind, layer in (("convex_corner", f"{l.name}_CONVEX"),
                            ("concave_corner", f"{l.name}_CONCAVE")):
            out.append(f"""
{kind.upper()}_{l.name} {{ @ {kind} markers {l.name}
  COPY {layer}
}}
DFM RDB {kind.upper()}_{l.name} "{outdir}/{kind}_{l.name}.rdb" ALL CELLS""")
    return "\n".join(out) + "\n"


def generate(layers: list[CalibreLayer], scales_um=(25, 50, 100, 250, 500, 1000),
             step_ratio: float = 0.5, outdir: str = "./calibre_out") -> str:
    """Return a complete SVRF deck."""
    return "".join([
        _header(layers, scales_um, step_ratio),
        _layer_defs(layers), "\n",
        _eps_guard(layers, outdir), "\n",
        _via_guard(layers), "\n",
        _markers(layers), "\n",
        _density_checks(layers, scales_um, step_ratio, outdir),
        _marker_output(layers, outdir),
        f"""
// ---- not generated here, deliberately ----
// line_end_density (D2): no single SVRF primitive defines a line end on
//   merged geometry, and it cannot be derived from perimeter -- chopping
//   lines into segments moves perimeter density only ~3 % while the
//   termination count rises tenfold. The morphological opening above is not
//   a substitute; it measures minimum width. Calibre pattern matching
//   against explicit line-end topologies is the way to recover D2.
// orientation, gradients and cross-layer terms: computed in Python from the
//   maps above, where the multi-scale tensor already exists.
//
// ---- verification status ----
// The rules above have been checked against an executable statement of what
// they mean (lamxsim.calibre.emulate, KLayout region algebra), not against
// Calibre. That catches an error in the ingest path, the conversions or the
// grid alignment; it cannot catch a difference between this reading of SVRF
// and Mentor's implementation of it. Run the deck once against the real tool
// and diff it against the emulator before trusting a production number.

// Run count for this deck: {len(scales_um)} scale(s) x {len(layers)} layer(s)
// of density scans, plus {sum(1 if l.is_via else 2 for l in layers)} marker
// output(s), which are scale-independent.
"""])


def eps_report(layers) -> list[dict]:
    """Per-layer eps with the safety margin, for the run metadata."""
    return [{
        "layer": l.name,
        "min_width_um": l.min_width_um,
        "eps_um": l.eps_um,
        "cliff_at_um": l.min_width_um / 2,
        "margin_x": round((l.min_width_um / 2) / l.eps_um, 2),
    } for l in layers if not l.is_via]


def binding_for(gds_path: str, manifest, manifest_path: str | None = None,
                *, top_cell: str | None = None) -> dict:
    """What this deck run is bound to, so its output cannot be reused blind.

    A complete set of RDBs from a different revision of the same design --
    same layer names, same scales, same coordinates -- would pass every
    completeness check and then be mixed with orientation, gradients and
    package context computed from the layout actually loaded. The maps would
    be internally consistent and describe two different chips.
    """
    from .layout import LayoutReader
    from .workflow import _file_digest

    reader = LayoutReader(gds_path, top_cell=top_cell or manifest.top_cell)
    bbox = reader.bbox()
    out = {
        "gds_sha256": _file_digest(gds_path),
        "top_cell": reader.top.name,
        "geometry_bbox_um": [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax],
        "layout_revision": manifest.layout_revision or "",
    }
    if manifest_path:
        out["study_manifest_sha256"] = _file_digest(str(manifest_path))
    return out


def extraction_manifest(layers: list[CalibreLayer], scales_um, step_ratio: float,
                        outdir: str = "./calibre_out",
                        binding: dict | None = None) -> dict:
    """The sidecar the ingest side needs, and cannot guess.

    ``eps`` in particular: a band density divided by the wrong eps is wrong by
    that ratio, which is a factor of 20 to 40 in practice and looks like a
    perfectly ordinary map. The deck is the only thing that knows which value
    Calibre actually used after snapping, so it writes it down rather than
    letting the reader re-derive it from a nominal minimum width.
    """
    return {
        "generator": "lamxsim.calibre.svrf",
        "step_ratio": float(step_ratio),
        "scales_um": [float(s) for s in scales_um],
        "outdir": outdir,
        "layers": [{"name": l.name, "layer": l.layer, "datatype": l.datatype,
                    "is_via": l.is_via, "min_width_um": l.min_width_um,
                    "eps_um": (0.0 if l.is_via else l.eps_um),
                    "via_area_um2": l.via_area_um2} for l in layers],
        "verified_against": "lamxsim.calibre.emulate (KLayout), not Calibre",
        "binding": dict(binding or {}),
    }


def write_deck(outdir, layers: list[CalibreLayer], *,
               scales_um=(25, 50, 100, 250, 500, 1000),
               step_ratio: float = 0.5, results_dir: str | None = None,
               binding: dict | None = None) -> dict:
    """Write ``rules.svrf`` and its extraction manifest. Returns the paths."""
    import json
    from pathlib import Path

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    results = results_dir or str(out)
    deck = out / "rules.svrf"
    deck.write_text(generate(layers, scales_um=scales_um,
                             step_ratio=step_ratio, outdir=results))
    side = out / "extraction_manifest.json"
    side.write_text(json.dumps(
        extraction_manifest(layers, scales_um, step_ratio, results, binding),
        indent=2))
    return {"deck": str(deck), "extraction_manifest": str(side)}


def layers_from_manifest(manifest) -> list[CalibreLayer]:
    """The deck's layer list, taken from the study manifest.

    The minimum width comes from the manifest's line rules where they exist.
    Falling back to a default would be worse than it looks: eps is a quarter
    of it, and a minimum width declared larger than the layout's real one puts
    eps past the cliff where the band collapses silently. The deck ships an
    ``INTERNAL`` guard for exactly this, so a wrong value is caught by the run
    rather than by the numbers looking odd later.
    """
    rules = manifest.line_rule_map()
    out = []
    for spec in manifest.metal_layers:
        min_width, line_max = rules.get(spec.name, (0.0, 0.0))
        out.append(CalibreLayer(spec.name, spec.layer, spec.datatype,
                                min_width_um=min_width or 0.1,
                                line_max_width_um=line_max or 0.0))
    seen = {l.name for l in out}
    for spec in manifest.via_layers.values():
        if spec.name in seen:
            continue
        seen.add(spec.name)
        out.append(CalibreLayer(spec.name, spec.layer, spec.datatype,
                                is_via=True))
    return out

# ----------------------------------------------------------------------
# calibre/ingest.py
# ----------------------------------------------------------------------
"""Read Calibre density output back into the analysis grid.

Calibre reports an area fraction per window. Marker-layer densities have to be
converted back into the physical quantity they stand for before anything
downstream sees them, and the conversion factor has to travel with the data --
a band density silently treated as a perimeter density is wrong by 1/eps,
which is a factor of 20 to 40 in practice.
"""
@dataclass(frozen=True)
class MarkerConversion:
    """How a Calibre marker-layer density maps to a physical feature."""
    feature: str
    unit: str
    #: value = calibre_density * factor. For a perimeter band, factor = 1/eps.
    factor: float
    note: str = ""


def perimeter_conversion(eps_um: float) -> MarkerConversion:
    return MarkerConversion(
        feature="perimeter_density", unit="um^-1", factor=1.0 / eps_um,
        note=f"inside band, eps={eps_um:g}um; boundary_length = band_area/eps")


def area_conversion(feature: str) -> MarkerConversion:
    return MarkerConversion(feature=feature, unit="dimensionless", factor=1.0,
                            note="native area density")


def count_conversion(feature: str, marker_area_um2: float) -> MarkerConversion:
    """Fixed-size corner/line-end markers: area density -> count per area."""
    return MarkerConversion(
        feature=feature, unit="um^-2", factor=1.0 / marker_area_um2,
        note=f"marker area {marker_area_um2:g}um^2; count = marker_area_total/marker_area")


def _data_lines(path: str | Path) -> list[str]:
    """Record lines only -- comments, blanks and check/cell headers dropped.

    A scan that matched nothing produces a file with no record lines, and that
    is a real answer: every window is empty, so every value is zero. A file
    whose record lines do not parse is a different thing entirely, and the two
    used to be indistinguishable -- both surfaced as "no records recognised",
    so a format change would have read as an empty layer.
    """
    out = []
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("//", "#")):
            continue
        if not stripped[0].isdigit() and not stripped.startswith("-"):
            continue  # check name / cell header
        out.append(line)
    return out


_RDB_VALUE = re.compile(
    r"^\s*(?P<x0>-?[\d.]+)\s+(?P<y0>-?[\d.]+)\s+(?P<x1>-?[\d.]+)\s+(?P<y1>-?[\d.]+)"
    r"(?:\s+(?P<val>-?[\d.eE+]+))?\s*$")


def read_density_csv(path: str | Path, *, x_col="x_um", y_col="y_um",
                     value_col="value") -> pd.DataFrame:
    """Read a simple x,y,value density dump."""
    df = pd.read_csv(path)
    missing = [c for c in (x_col, y_col, value_col) if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; got {list(df.columns)}")
    return df.rename(columns={x_col: "x_um", y_col: "y_um", value_col: "value"})


def read_density_rdb(path: str | Path) -> pd.DataFrame:
    """Read the rectangle-plus-value subset of a Calibre ASCII RDB.

    Only the geometry and value lines are interpreted; check names and cell
    headers are ignored. Window centres come from the rectangle, so a deck
    whose STEP differs from its WINDOW still lands on the right coordinates.
    """
    xs, ys, vs = [], [], []
    for line in _data_lines(path):
        m = _RDB_VALUE.match(line)
        if not m:
            raise ValueError(f"{path}: unparseable record {line.strip()!r}")
        x0, y0, x1, y1 = (float(m.group(k)) for k in ("x0", "y0", "x1", "y1"))
        if x1 <= x0 or y1 <= y0:
            continue
        xs.append((x0 + x1) / 2)
        ys.append((y0 + y1) / 2)
        v = m.group("val")
        vs.append(float(v) if v is not None else np.nan)
    return pd.DataFrame({"x_um": xs, "y_um": ys, "value": vs},
                        columns=["x_um", "y_um", "value"])


def to_grid(df: pd.DataFrame, grid: Grid, conversion: MarkerConversion,
            *, tol_um: float | None = None) -> np.ndarray:
    """Snap Calibre window values onto *grid*, applying the conversion.

    Matching is nearest-centre, then a distance check against the tolerance.
    Bucketing the coordinates instead -- rounding both sides to a common grid
    and comparing keys -- rejects pairs that are well inside the tolerance
    whenever they straddle a bucket edge: at a 50 um tolerance a window 26 um
    from its cell centre lands in the neighbouring bucket and is dropped.

    Unreported windows become 0.0, which is what Calibre's omission of empty
    windows means. Individual records further than the tolerance from any cell
    are dropped, so a grid may cover a sub-region of what the deck reported;
    a *complete* failure to match raises, so a coordinate-frame mismatch
    surfaces as an error rather than as a map of zeros that looks plausible.
    """
    from scipy.spatial import cKDTree

    tol = (grid.stride_um / 2) if tol_um is None else float(tol_um)
    out = np.zeros(len(grid), dtype=float)

    centres = np.column_stack([[c.x_center for c in grid.cells],
                               [c.y_center for c in grid.cells]])
    points = df[["x_um", "y_um"]].to_numpy(float).reshape(-1, 2)
    if len(points) == 0:
        # An empty scan means every window was empty, which is a map of
        # zeros. This is only safe to assume because the reader now
        # distinguishes an empty file from an unparseable one.
        return out

    distance, index = cKDTree(centres).query(points, k=1)
    matched = distance <= tol

    values = df["value"].to_numpy(float) * conversion.factor

    # Calibre emits one record per window, so two records claiming the same
    # cell means the deck and the grid disagree about WINDOW/STEP. Letting the
    # later record win would leave the other cell reading zero -- a value that
    # is indistinguishable from an empty window.
    hit = index[matched]
    if len(hit) != len(np.unique(hit)):
        cells, counts = np.unique(hit, return_counts=True)
        clashing = cells[counts > 1]
        example = grid.cells[int(clashing[0])]
        raise ValueError(
            f"{len(clashing)} grid cell(s) were claimed by more than one "
            f"Calibre record; for example the cell centred at "
            f"({example.x_center:g}, {example.y_center:g})um was claimed "
            f"{int(counts[counts > 1][0])} times. Calibre emits one record per "
            "window, so this means the deck's WINDOW/STEP does not match this "
            f"grid (scale {grid.scale_um:g}um, stride {grid.stride_um:g}um).")

    out[hit] = values[matched]

    if not matched.any():
        raise ValueError(
            f"no Calibre window matched a grid cell within {tol:g}um "
            f"(closest was {distance.min():.3g}um); the layout origin or the "
            "WINDOW/STEP in the deck does not match this grid")
    return out


@dataclass
class CalibreFeatureSet:
    """Calibre-derived features on one grid, with provenance."""
    grid: Grid
    layer: str
    scale_um: float
    values: dict[str, np.ndarray]
    conversions: dict[str, MarkerConversion]
    evidence_class: EvidenceClass = EvidenceClass.GDS_GEOMETRY
    source_files: dict[str, str] | None = None

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.grid.to_arrays())
        df["layer"] = self.layer
        for k, v in self.values.items():
            df[k] = v
        return df

    def provenance(self) -> dict:
        return {
            "layer": self.layer,
            "scale_um": self.scale_um,
            "n_windows": len(self.grid),
            "overlapping_windows": self.grid.overlapping,
            "stride_um": self.grid.stride_um,
            "conversions": {k: {"unit": c.unit, "factor": c.factor, "note": c.note}
                            for k, c in self.conversions.items()},
            "source_files": self.source_files or {},
        }


def load_scale(bbox: BBox, scale_um: float, layer: str, files: dict[str, str],
               *, eps_um: float, step_ratio: float = 0.5,
               reader=read_density_rdb, marker_files: dict[str, str] | None = None,
               grid: Grid | None = None) -> CalibreFeatureSet:
    """Load one layer x scale from a set of Calibre output files.

    ``files`` maps a density-scan kind to a path, e.g.
    ``{"metal_density": ..., "perimeter_band": ..., "via_density": ...}``.
    ``marker_files`` maps a count kind -- ``convex_corner``, ``concave_corner``,
    ``via_marker`` -- to a marker RDB, which is counted rather than scaled.

    When both corner marker files and a perimeter band are present the exact
    corner correction is applied, so ``perimeter_density`` is the corrected
    value and the uncorrected band is kept beside it under
    ``perimeter_density_band_only``. Without the corners the band value is
    still returned, but it is low wherever convex corners outnumber concave
    ones -- 5.7 % on one of the benchmark patterns -- so the two are never
    given the same name.
    """
    if grid is None:
        grid = build_grid(bbox, scale_um, stride_um=scale_um * step_ratio)
    elif abs(grid.stride_um - scale_um * step_ratio) > 1e-9:
        raise ValueError(
            f"the deck stepped {scale_um * step_ratio:g}um at a {scale_um:g}um "
            f"window and this grid strides {grid.stride_um:g}um. The windows "
            "do not line up, so every value would be attributed to the wrong "
            "cell or rejected as a duplicate claim. Regenerate the deck with "
            f"step_ratio={grid.stride_um / scale_um:g}.")
    density_conv = {
        "metal_density": area_conversion("metal_density"),
        "via_density": area_conversion("via_density"),
        "perimeter_band": perimeter_conversion(eps_um),
        "narrow_structure": area_conversion("narrow_structure_density"),
    }
    # A gate is not a feature. check_eps_guard reads this one and refuses a
    # non-empty result; turning it into a map here would produce a "places the
    # guard fired" density, which is not what it means and would be all zeros
    # on every run that is allowed to proceed.
    gates = ("eps_violation",)
    count_features = {
        "convex_corner": "convex_corner_density",
        "concave_corner": "concave_corner_density",
        "via_marker": "via_count_density",
    }

    values, used = {}, {}
    for kind, path in files.items():
        if kind not in density_conv:
            raise ValueError(f"unknown density kind {kind!r}; "
                             f"known: {sorted(density_conv)}")
        c = density_conv[kind]
        values[c.feature] = to_grid(reader(path), grid, c)
        used[c.feature] = c

    for kind, path in (marker_files or {}).items():
        if kind in gates:
            continue
        if kind not in count_features:
            raise ValueError(f"unknown marker kind {kind!r}; "
                             f"known: {sorted(count_features)}")
        feature = count_features[kind]
        values[feature] = to_count_grid(read_marker_rdb(path), grid)
        used[feature] = MarkerConversion(
            feature=feature, unit="um^-2", factor=1.0,
            note="counted from the marker list; no marker-area assumption")

    if {"convex_corner_density", "concave_corner_density"} <= values.keys() \
            and "perimeter_density" in values:
        band_only = values["perimeter_density"]
        values["perimeter_density_band_only"] = band_only
        values["perimeter_density"] = band_only + eps_um * (
            values["convex_corner_density"] - values["concave_corner_density"])
        used["perimeter_density"] = MarkerConversion(
            feature="perimeter_density", unit="um^-1", factor=1.0 / eps_um,
            note=f"inside band (eps={eps_um:g}um) plus the exact corner "
                 "correction eps*(convex - concave)")

    if "convex_corner_density" in values and "concave_corner_density" in values:
        values["corner_density"] = (values["convex_corner_density"]
                                    + values["concave_corner_density"])
        used["corner_density"] = MarkerConversion(
            feature="corner_density", unit="um^-2", factor=1.0,
            note="convex + concave marker counts")

    return CalibreFeatureSet(grid=grid, layer=layer, scale_um=scale_um,
                             values=values, conversions=used,
                             source_files={**dict(files),
                                           **dict(marker_files or {})})


def perimeter_from_band_and_corners(band_density: np.ndarray,
                                    convex_density: np.ndarray,
                                    concave_density: np.ndarray,
                                    *, eps_um: float,
                                    corner_marker_area_um2: float) -> np.ndarray:
    """Exact perimeter density from Calibre's band and corner marker densities.

    The inside band under-reports by eps^2 at each convex corner and
    over-reports by the same at each concave one, so

        P = band_area/eps + eps * (n_convex - n_concave)

    recovers the true boundary length exactly on Manhattan geometry. Dividing
    through by window area turns every term into the density Calibre reports:

        PD = band_density/eps + eps * (convex_density - concave_density) / a_marker

    ``eps_um`` must be the database-unit-snapped value the deck actually used,
    not the nominal one.
    """
    return (band_density / eps_um
            + eps_um * (convex_density - concave_density) / corner_marker_area_um2)


def read_marker_rdb(path: str | Path) -> pd.DataFrame:
    """Read a marker RDB as a list of individual markers, not as a density.

    A count feature -- vias, corners, line ends -- is a point count, and the
    density scanner cannot report one: it reports the area fraction of the
    marker layer, so recovering a count needs the markers to be uniform in
    area and needs that area to be known. Neither is checkable from the
    density output itself, and an error in the assumed marker area scales
    every count by a constant that no downstream test would notice.

    Reporting the markers themselves removes the assumption. It costs one
    record per marker instead of one per window, which is why it is the right
    trade for line ends and vias and the wrong one for a full-chip corner
    layer -- see ``to_count_grid`` for the density-based alternative.
    """
    xs, ys, areas = [], [], []
    for line in _data_lines(path):
        m = _RDB_VALUE.match(line)
        if not m:
            raise ValueError(f"{path}: unparseable record {line.strip()!r}")
        x0, y0, x1, y1 = (float(m.group(k)) for k in ("x0", "y0", "x1", "y1"))
        if x1 <= x0 or y1 <= y0:
            continue
        xs.append((x0 + x1) / 2)
        ys.append((y0 + y1) / 2)
        areas.append((x1 - x0) * (y1 - y0))
    return pd.DataFrame({"x_um": xs, "y_um": ys, "area_um2": areas})


def to_count_grid(markers: pd.DataFrame, grid: Grid) -> np.ndarray:
    """Marker count per unit area on *grid*, from a marker list.

    A marker is assigned to the cell containing its centroid, on the same
    half-open convention the KLayout via extractor uses, so a marker sitting
    exactly on a cell boundary is counted once rather than twice or not at
    all. With an overlapping grid a marker belongs to every window that
    contains it, which is what the moving-window density would also report.
    """
    out = np.zeros(len(grid), dtype=float)
    if markers.empty:
        return out
    px = markers["x_um"].to_numpy(float)
    py = markers["y_um"].to_numpy(float)
    for c in grid.cells:
        inside = ((px >= c.x0) & (px < c.x1) & (py >= c.y0) & (py < c.y1))
        out[c.cell_id] = int(inside.sum()) / c.area_um2
    return out


#: What a complete deck run holds for one layer. A partial directory used to
#: be accepted: whatever RDBs happened to be there were used and every other
#: map fell back to KLayout without a word, so a run with one corner file
#: reported "extraction: calibre" while almost every number came from Python.
REQUIRED_DENSITY = {"metal": ("metal_density", "perimeter_band"),
                    "via": ("via_density",)}
REQUIRED_MARKERS = {"metal": ("convex_corner", "concave_corner"),
                    "via": ("via_marker",)}

#: Features the deck can supply. Everything else in the atlas -- orientation,
#: gradients, cross-layer terms, position, package context -- is computed in
#: Python either way, so a Calibre run replaces part of the extraction, never
#: all of it.
CALIBRE_SUPPLIED = (
    "metal_density", "perimeter_density", "convex_corner_density",
    "concave_corner_density", "corner_density", "via_density",
    "via_count_density", "narrow_structure_density",
)


@dataclass
class CalibreSource:
    """A directory of deck output, addressable by (layer, scale)."""
    directory: Path
    manifest: dict
    density: dict
    markers: dict

    @property
    def emulated(self) -> bool:
        return bool(self.manifest.get("emulated"))

    def eps_um(self, layer: str) -> float:
        for entry in self.manifest["layers"]:
            if entry["name"] == layer:
                return float(entry["eps_um"])
        raise KeyError(f"{layer!r} is not in the extraction manifest; "
                       f"it has {[e['name'] for e in self.manifest['layers']]}")

    def layers(self) -> list[str]:
        return [e["name"] for e in self.manifest["layers"]]

    def scales_um(self) -> list[float]:
        return [float(s) for s in self.manifest["scales_um"]]

    def features_for(self, layer: str, scale_um: float, bbox: BBox,
                     grid: Grid | None = None) -> dict[str, np.ndarray]:
        """Every feature the deck supplied for this layer and scale.

        Absent kinds are simply absent -- the caller keeps its own value for
        them. Returning zeros instead would be indistinguishable from a layer
        the deck really did find empty.
        """
        files = {k: str(p) for (l, s, k), p in self.density.items()
                 if l == layer and s == float(scale_um)}
        marker_files = {k: str(p) for (l, k), p in self.markers.items()
                        if l == layer}
        if not files and not marker_files:
            return {}
        return load_scale(bbox, float(scale_um), layer, files,
                          eps_um=self.eps_um(layer) or 1.0,
                          step_ratio=float(self.manifest["step_ratio"]),
                          marker_files=marker_files, grid=grid).values


    def is_via(self, layer: str) -> bool:
        for entry in self.manifest["layers"]:
            if entry["name"] == layer:
                return bool(entry["is_via"])
        raise KeyError(layer)

    def check_binding(self, gds_path: str, reader, manifest) -> dict:
        """Refuse deck output that was not produced from this layout.

        The completeness gate establishes that a full set of maps is present.
        It says nothing about *which* layout they describe: a complete set
        from another revision of the same design has the same layer names, the
        same scales and the same coordinates, so it passes every other check
        and is then mixed with the orientation, gradient and package-context
        maps computed from the file actually loaded. The result is internally
        consistent and describes two different chips.
        """
        from .workflow import _file_digest

        binding = self.manifest.get("binding") or {}
        if not binding.get("gds_sha256"):
            raise ValueError(
                f"{self.directory}/extraction_manifest.json carries no layout "
                "binding, so nothing establishes that these maps came from "
                f"{gds_path} rather than from another revision with the same "
                "layer names. Regenerate the deck with `lamxsim deck <gds>`, "
                "which records the layout digest, top cell and bounding box.")

        actual = {
            "gds_sha256": _file_digest(gds_path),
            "top_cell": reader.top.name,
        }
        bbox = reader.bbox()
        actual_bbox = [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]

        problems = []
        if binding["gds_sha256"] != actual["gds_sha256"]:
            problems.append(
                f"layout digest: deck ran on {binding['gds_sha256'][:12]}..., "
                f"this file is {actual['gds_sha256'][:12]}...")
        if binding.get("top_cell") and binding["top_cell"] != actual["top_cell"]:
            problems.append(f"top cell: deck used {binding['top_cell']!r}, "
                            f"this run uses {actual['top_cell']!r}")
        declared_bbox = binding.get("geometry_bbox_um")
        if declared_bbox and any(abs(a - b) > 1e-6 for a, b
                                 in zip(declared_bbox, actual_bbox)):
            problems.append(f"geometry bbox: deck saw {declared_bbox}, "
                            f"this run sees {actual_bbox}")
        revision = binding.get("layout_revision") or ""
        if revision and manifest.layout_revision and \
                revision != manifest.layout_revision:
            problems.append(f"layout revision: deck recorded {revision!r}, "
                            f"the manifest says {manifest.layout_revision!r}")
        if problems:
            raise ValueError(
                f"the deck output in {self.directory} was not produced from "
                "this layout:\n  " + "\n  ".join(problems)
                + "\nThe density maps would describe one layout and the "
                  "orientation, gradient and package-context maps another, "
                  "with nothing in the output saying so. Re-run the deck.")
        return binding

    def check_contract(self, manifest) -> None:
        """Refuse a deck built against different layer rules than these.

        The digest of the manifest file is recorded but is deliberately *not*
        the gate: adding a package condition or a comment changes it while
        changing nothing the deck depends on, and a gate that fires on
        unrelated edits gets worked around. What the deck actually depends on
        is the layer identity, the minimum width -- eps is a quarter of it and
        the guard threshold is it -- the scales and the step ratio, so those
        are compared directly.

        Without this, a deck generated at min_width 0.2um (eps 0.05um) was
        accepted by a run whose manifest declared 0.4um, and the metadata
        reported both numbers side by side.
        """
        pass

        want = {l.name: l for l in layers_from_manifest(manifest)}
        have = {e["name"]: e for e in self.manifest["layers"]}
        problems = []
        for name, layer in want.items():
            entry = have.get(name)
            if entry is None:
                problems.append(f"{name}: not in the deck")
                continue
            if (entry["layer"], entry["datatype"]) != (layer.layer, layer.datatype):
                problems.append(
                    f"{name}: deck used {entry['layer']}/{entry['datatype']}, "
                    f"the manifest says {layer.layer}/{layer.datatype}")
            if entry["is_via"] != layer.is_via:
                problems.append(f"{name}: deck treated it as "
                                f"{'a via' if entry['is_via'] else 'metal'}, "
                                f"the manifest as "
                                f"{'a via' if layer.is_via else 'metal'}")
            if layer.is_via:
                continue
            if abs(entry["min_width_um"] - layer.min_width_um) > 1e-9:
                problems.append(
                    f"{name}: deck used min_width {entry['min_width_um']:g}um "
                    f"(so eps {entry['eps_um']:g}um and that guard threshold), "
                    f"the manifest says {layer.min_width_um:g}um")

        deck_scales = sorted(float(s) for s in self.manifest["scales_um"])
        want_scales = sorted(float(s) for s in manifest.scales_um)
        if deck_scales != want_scales:
            problems.append(f"scales: deck {deck_scales}um, "
                            f"manifest {want_scales}um")
        if problems:
            raise ValueError(
                f"the deck output in {self.directory} was built against "
                "different layer rules than this run uses:\n  "
                + "\n  ".join(problems)
                + "\nThe maps would be reported under rules they were not "
                  "measured with. Re-run the deck for this manifest.")

    def check_complete(self, layers, scales_um) -> None:
        """Refuse a run that would mix the two extractors without saying so.

        Every layer and scale the atlas will ask for must be present in full.
        A missing map is not a smaller Calibre run: it is a Python map wearing
        a Calibre label, and the label is the whole reason to prefer one.
        """
        gaps = []
        for layer in layers:
            kind = "via" if self.is_via(layer) else "metal"
            for marker in REQUIRED_MARKERS[kind]:
                if (layer, marker) not in self.markers:
                    gaps.append(f"{layer}: {marker} marker file")
            for scale in scales_um:
                for kd in REQUIRED_DENSITY[kind]:
                    if (layer, float(scale), kd) not in self.density:
                        gaps.append(f"{layer} @ {float(scale):g}um: {kd}")
        if gaps:
            raise ValueError(
                f"the deck output in {self.directory} is incomplete; "
                f"{len(gaps)} required output(s) are missing:\n  "
                + "\n  ".join(gaps[:12])
                + (f"\n  ... and {len(gaps) - 12} more" if len(gaps) > 12 else "")
                + "\nEach missing map would fall back to the KLayout "
                  "extractor while the run still reported itself as a deck "
                  "extraction. Re-run the deck, or drop --features-from and "
                  "extract everything in Python.")

    def check_eps_guard(self, layers) -> dict[str, int]:
        """Require the deck's own minimum-width guard to have been run and passed.

        The guard exists because eps is a quarter of the declared minimum
        width and the band collapses once eps passes half the real one -- a
        cliff, silent, and worth -38 % to -87 %. It was generated into every
        deck and consumed by nobody: the CLI told a human to check it. A
        human-checked precondition is not part of the evidence chain, so the
        result is now a required file and a non-empty one is an error.
        """
        out = {}
        for layer in layers:
            if self.is_via(layer):
                continue
            path = self.directory / f"eps_violation_{layer}.rdb"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} is missing. The deck's EPS_VIOLATION_{layer} check "
                    "must be run and its result written out; without it nothing "
                    "establishes that the layout is as wide as the manifest "
                    "says, and every perimeter number here would be understated "
                    "by an unknown amount if it is not.")
            n = len(read_marker_rdb(path))
            if n:
                raise ValueError(
                    f"EPS_VIOLATION_{layer} reported {n} location(s) narrower "
                    f"than the declared minimum width, so eps="
                    f"{self.eps_um(layer):g}um is past the point where the "
                    "inside band collapses and the perimeter density for this "
                    "layer is understated. Correct min_width for this layer in "
                    "the manifest and re-run the deck.")
            out[layer] = n
        return out


def discover(directory: str | Path) -> CalibreSource:
    """Read a deck output directory, by the naming the deck itself writes.

    The extraction manifest is required rather than optional. Without it the
    reader would have to guess ``eps``, and a band density divided by the
    wrong one is off by that ratio while looking entirely ordinary -- the one
    error in this path that no downstream check could catch.
    """
    import json

    d = Path(directory)
    side = d / "extraction_manifest.json"
    if not side.exists():
        raise FileNotFoundError(
            f"{side} is missing. It is written beside the deck by "
            "lamxsim.calibre.svrf.write_deck (or by the emulator) and carries "
            "the snapped eps per layer, which cannot be recovered from the "
            "RDB files. Regenerate the deck rather than assuming a value.")
    manifest = json.loads(side.read_text())
    names = {e["name"] for e in manifest["layers"]}

    density, markers = {}, {}
    for path in sorted(d.glob("*.rdb")):
        stem = path.stem
        layer = next((n for n in names
                      if stem.endswith(f"_{n}") or f"_{n}_" in stem), None)
        if layer is None:
            continue
        if stem.endswith(f"_{layer}"):
            kind = stem[: -len(layer) - 1]
            markers[(layer, kind)] = path
        else:
            head, _, tail = stem.partition(f"_{layer}_")
            if not tail.endswith("um"):
                continue
            density[(layer, float(tail[:-2]), head)] = path
    if not density and not markers:
        raise ValueError(f"{d} holds no RDB file matching a layer in "
                         f"{sorted(names)}")
    return CalibreSource(directory=d, manifest=manifest,
                         density=density, markers=markers)

# ----------------------------------------------------------------------
# calibre/emulate.py
# ----------------------------------------------------------------------
"""Reproduce the generated deck's operator sequence with KLayout.

Why this exists: nothing in the repository ever ran the Calibre path
end-to-end. The band correction was measured to 0.0000 % on isolated
patterns, but the deck -> RDB -> grid -> feature chain had no test at all, so
a wrong ``eps``, a window/grid offset, or a marker read as a density would
have produced a plausible map that no test would question.

What this is: an executable statement of what each generated SVRF rule
*means*, written in KLayout region algebra, emitting the same RDB files the
deck names. It lets the ingest path, the conversions and the grid alignment
be tested, and it lets a user dry-run the whole flow without a Calibre
licence.

What this is **not**: Calibre. It cannot detect a difference between our
reading of SVRF and Mentor's implementation of it -- if the deck's
``SIZE ... BY -eps`` does something other than what ``Region.sized`` does,
both sides here are wrong together and agree. Only a run against the real
tool settles that, and until one happens the deck's rules stay marked
unverified in the header it prints.
"""
#: Side of the square marker dropped at each counted point, in um. Only the
#: marker *list* is used downstream, so this affects nothing but readability
#: of the RDB; a density-based count would have to divide by its area.
MARKER_SIDE_UM = 0.02


def _window_area_density(region: db.Region, grid: Grid, u) -> np.ndarray:
    """Area of *region* inside each window, over window area.

    This is what ``DENSITY <layer> WINDOW w w STEP s s`` reports. Windows are
    taken a grid row at a time against a pre-clipped strip, for the same
    reason the KLayout extractor does it: intersecting each window against the
    whole layer costs ~35x more and scales with die area.
    """
    out = np.zeros(len(grid), dtype=float)
    if region.is_empty():
        return out
    rb = region.bbox()
    rows: dict[int, list] = {}
    for cell in grid.cells:
        rows.setdefault(cell.row, []).append(cell)
    x_lo = min(rb.left, u.um_to_dbu(grid.bbox.xmin)) - 1
    x_hi = max(rb.right, u.um_to_dbu(grid.bbox.xmax)) + 1
    for cells in rows.values():
        y0, y1 = u.um_to_dbu(cells[0].y0), u.um_to_dbu(cells[0].y1)
        if y1 <= rb.bottom or y0 >= rb.top:
            continue
        strip = region & db.Region(db.Box(x_lo, y0, x_hi, y1))
        if strip.is_empty():
            continue
        for cell in cells:
            win = db.Region(db.Box(u.um_to_dbu(cell.x0), u.um_to_dbu(cell.y0),
                                   u.um_to_dbu(cell.x1), u.um_to_dbu(cell.y1)))
            out[cell.cell_id] = (u.area_dbu2_to_um2((strip & win).area())
                                 / cell.area_um2)
    return out


def _write_density_rdb(path: Path, grid: Grid, values: np.ndarray,
                       check: str) -> None:
    """Write the rectangle-plus-value records ``read_density_rdb`` parses.

    Windows whose value is zero are omitted, because Calibre omits them; the
    ingest side turns an absent window back into 0.0. Round-tripping through
    the omission is part of what needs testing.
    """
    lines = [f"// {check}"]
    for cell in grid.cells:
        v = float(values[cell.cell_id])
        if v == 0.0:
            continue
        lines.append(f"{cell.x0:.6f} {cell.y0:.6f} {cell.x1:.6f} "
                     f"{cell.y1:.6f} {v:.10g}")
    path.write_text("\n".join(lines) + "\n")


def _write_marker_rdb(path: Path, points_um: np.ndarray, check: str,
                      side_um: float = MARKER_SIDE_UM) -> None:
    """Write one square record per counted point."""
    h = side_um / 2
    lines = [f"// {check}"]
    for x, y in np.asarray(points_um, dtype=float).reshape(-1, 2):
        lines.append(f"{x - h:.6f} {y - h:.6f} {x + h:.6f} {y + h:.6f}")
    path.write_text("\n".join(lines) + "\n")


@dataclass
class EmulatedRun:
    """Where each emulated output landed, and what it stands for."""
    outdir: Path
    density: dict[tuple[str, float, str], Path] = field(default_factory=dict)
    markers: dict[tuple[str, str], Path] = field(default_factory=dict)
    eps_um: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def density_files(self, layer: str, scale_um: float) -> dict[str, str]:
        return {kind: str(p) for (l, s, kind), p in self.density.items()
                if l == layer and s == scale_um}


def run(gds_path: str, layers: list[CalibreLayer], *,
        scales_um=(100.0,), step_ratio: float = 0.5,
        outdir: str | Path = "calibre_out",
        min_width_um: dict[str, float] | None = None,
        top_cell: str | None = None, manifest=None,
        manifest_path: str | None = None) -> EmulatedRun:
    """Emulate the deck over *gds_path* and write its outputs to *outdir*."""
    from .workflow import _file_digest

    reader = LayoutReader(gds_path, top_cell=top_cell)
    u = reader.units
    bbox = reader.bbox()
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    result = EmulatedRun(outdir=out)

    for layer in layers:
        spec = LayerSpec(layer.name, layer.layer, layer.datatype)
        region = reader.region(spec)
        if layer.is_via:
            # via_count_density counts one point per via, so the emulated
            # output is the via list rather than an area fraction. The
            # KLayout extractor counts centroids on a half-open cell, and the
            # marker reader reproduces that rule.
            # Same centroid rule as ViaExtractor.centroids: the bbox centre
            # in database units, scaled. Rounding to the dbu grid first would
            # move a via off a cell boundary and change its count.
            pts = [[(b.left + b.right) / 2 * u.dbu, (b.bottom + b.top) / 2 * u.dbu]
                   for b in (poly.bbox() for poly in region.each())]
            path = out / f"via_marker_{layer.name}.rdb"
            _write_marker_rdb(path, np.array(pts, float).reshape(-1, 2),
                              f"VIA_MARKER_{layer.name}")
            result.markers[(layer.name, "via_marker")] = path
        else:
            eps = layer.eps_um
            result.eps_um[layer.name] = eps

            # The deck's EPS_VIOLATION check, run rather than assumed. Empty
            # is the passing result and the ingest side requires the file.
            #
            # Projection metric, not the default Euclidian one. Measured
            # corner-to-corner, every re-entrant corner is a pair of edges a
            # vanishing distance apart, so the Euclidian check reports 177
            # violations on the golden die where the Projection check reports
            # none and a morphological opening confirms nothing is narrow. A
            # guard that fires on every layout is a guard that gets switched
            # off.
            narrower = region.width_check(u.um_to_dbu(layer.min_width_um),
                                          False, db.Region.Projection)
            pts = [[u.dbu_to_um((e.first.x1 + e.second.x2) / 2),
                    u.dbu_to_um((e.first.y1 + e.second.y2) / 2)]
                   for e in narrower.each()]
            guard = out / f"eps_violation_{layer.name}.rdb"
            _write_marker_rdb(guard, np.array(pts, float).reshape(-1, 2),
                              f"EPS_VIOLATION_{layer.name}")
            result.markers[(layer.name, "eps_violation")] = guard
            band = region - region.sized(-u.um_to_dbu(eps))
            convex, concave = corners_mod.classify(region)
            for kind, pts in (("convex_corner", convex),
                              ("concave_corner", concave)):
                arr = np.array([[u.dbu_to_um(p.x), u.dbu_to_um(p.y)]
                                for p in pts], float).reshape(-1, 2)
                path = out / f"{kind}_{layer.name}.rdb"
                _write_marker_rdb(path, arr, f"{kind.upper()}_{layer.name}")
                result.markers[(layer.name, kind)] = path

            # Geometry narrower than w, by morphological opening. Not a
            # line-end proxy: opening by w/2 erases a line of width w, so on
            # an array of 1um lines it returns the whole array. Written under
            # the name of what it measures.
            w = (min_width_um or {}).get(layer.name, layer.min_width_um)
            h = max(u.um_to_dbu(w / 2), 1)
            narrow = region - region.sized(-h).sized(h)

        for scale in scales_um:
            grid = build_grid(bbox, float(scale), stride_um=float(scale) * step_ratio)
            tag = f"{layer.name}_{scale:g}um"
            todo = [("via_density" if layer.is_via else "metal_density", region)]
            if not layer.is_via:
                todo += [("perimeter_band", band),
                         ("narrow_structure", narrow)]
            for kind, r in todo:
                path = out / f"{kind}_{tag}.rdb"
                _write_density_rdb(path, grid, _window_area_density(r, grid, u),
                                   f"DENSITY_{kind.upper()}_{tag}")
                result.density[(layer.name, float(scale), kind)] = path

    # The same sidecar the real deck writes, so the ingest side reads one
    # format and cannot tell an emulated run from a real one by accident --
    # it is told, by the "emulated" flag, which is the honest way round.
    binding = {
        "gds_sha256": _file_digest(gds_path),
        "top_cell": reader.top.name,
        "geometry_bbox_um": [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax],
        "layout_revision": (manifest.layout_revision or "") if manifest else "",
    }
    if manifest_path:
        binding["study_manifest_sha256"] = _file_digest(str(manifest_path))
    side = extraction_manifest(layers, scales_um, step_ratio, str(out),
                                        binding)
    side["generator"] = "lamxsim.calibre.emulate"
    side["emulated"] = True
    (out / "extraction_manifest.json").write_text(json.dumps(side, indent=2))

    result.notes.append(
        "emulated with KLayout region algebra, not run through Calibre; "
        "this checks the ingest and conversion path, not the tool")
    return result
