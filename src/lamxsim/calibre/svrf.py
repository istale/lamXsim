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
from __future__ import annotations

from dataclasses import dataclass

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
                   f'"{{outdir}}/eps_violation_{l.name}.rdb" ALL CELLS\n')
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


def extraction_manifest(layers: list[CalibreLayer], scales_um, step_ratio: float,
                        outdir: str = "./calibre_out") -> dict:
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
    }


def write_deck(outdir, layers: list[CalibreLayer], *,
               scales_um=(25, 50, 100, 250, 500, 1000),
               step_ratio: float = 0.5, results_dir: str | None = None) -> dict:
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
        extraction_manifest(layers, scales_um, step_ratio, results), indent=2))
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
