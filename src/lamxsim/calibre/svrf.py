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


def _eps_guard(layers) -> str:
    """A deck that fails loudly if eps was set too large for the real layout."""
    out = ["""// ---- eps validity guard ----
// The inside-band approximation is exact only while eps < min_width/2.
// These checks flag any geometry narrower than the assumed minimum width;
// a non-empty result means the eps below is wrong for this layout and every
// perimeter number derived from it is understated.
"""]
    for l in layers:
        if l.is_via:
            continue
        out.append(f"EPS_VIOLATION_{l.name} {{ @ {l.name} narrower than assumed "
                   f"min width {l.min_width_um:g}um -- eps={l.eps_um:g}um is unsafe")
        out.append(f"  INTERNAL {l.name} < {l.min_width_um:g}")
        out.append("}\n")
    return "\n".join(out)


def _markers(layers) -> str:
    out = ["// ---- marker layers ----"]
    for l in layers:
        if l.is_via:
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
// with an ordinary convex one.
{l.name}_CONVEX  = ANGLE {l.name} == 90
{l.name}_CONCAVE = ANGLE {l.name} == 270

// {l.name}: line-end caps. A short edge with a convex corner at each end,
// guarded by an elongation ratio -- without the guard every side of a dummy
// fill square counts, which on a 36-square array yields 144 phantom line ends
// against a true zero. Scored exactly (0 errors over eight benchmark
// patterns) at an elongation between 1.2 and 2.0.
{l.name}_CAP = CONVEX EDGE {l.name} WITH LENGTH <= {(l.line_max_width_um or l.min_width_um * 4):g} ANGLE1 == 90 ANGLE2 == 90
{l.name}_LINE_END = {l.name}_CAP  // apply the elongation guard here; see
                                  // features/lineends.py detect_aspect()""")
    return "\n".join(out) + "\n"


def _density_checks(layers, scales_um, step_ratio, outdir) -> str:
    out = ["// ---- moving-window density scans ----"]
    for s in scales_um:
        step = s * step_ratio
        for l in layers:
            tag = f"{l.name}_{s:g}um"
            out.append(f"""
DENSITY_METAL_{tag} {{ @ metal_density {l.name} @ {s:g}um
  DENSITY {l.name} > 0 WINDOW {s:g} {s:g} STEP {step:g} {step:g}
}}
DFM RDB DENSITY_METAL_{tag} "{outdir}/metal_density_{tag}.rdb" ALL CELLS""")
            if l.is_via:
                continue
            out.append(f"""
DENSITY_PERIM_{tag} {{ @ perimeter_density {l.name} @ {s:g}um (divide by eps={l.eps_um:g})
  DENSITY {l.name}_BAND > 0 WINDOW {s:g} {s:g} STEP {step:g} {step:g}
}}
DFM RDB DENSITY_PERIM_{tag} "{outdir}/perimeter_band_{tag}.rdb" ALL CELLS

DENSITY_CONVEX_{tag} {{ @ convex_corner_density {l.name} @ {s:g}um
  DENSITY {l.name}_CONVEX > 0 WINDOW {s:g} {s:g} STEP {step:g} {step:g}
}}
DFM RDB DENSITY_CONVEX_{tag} "{outdir}/convex_corner_{tag}.rdb" ALL CELLS

DENSITY_CONCAVE_{tag} {{ @ concave_corner_density {l.name} @ {s:g}um
  DENSITY {l.name}_CONCAVE > 0 WINDOW {s:g} {s:g} STEP {step:g} {step:g}
}}
DFM RDB DENSITY_CONCAVE_{tag} "{outdir}/concave_corner_{tag}.rdb" ALL CELLS""")
    return "\n".join(out) + "\n"


def generate(layers: list[CalibreLayer], scales_um=(25, 50, 100, 250, 500, 1000),
             step_ratio: float = 0.5, outdir: str = "./calibre_out") -> str:
    """Return a complete SVRF deck."""
    return "".join([
        _header(layers, scales_um, step_ratio),
        _layer_defs(layers), "\n",
        _eps_guard(layers), "\n",
        _markers(layers), "\n",
        _density_checks(layers, scales_um, step_ratio, outdir),
        f"""
// ---- not generated here, deliberately ----
// line_end_density: no single SVRF primitive defines a line end on merged
//   geometry, and it cannot be derived from perimeter -- chopping lines into
//   segments moves perimeter density only ~3 % while the termination count
//   rises tenfold. Calibre pattern matching against explicit line-end
//   topologies is more robust than stacking LENGTH/ANGLE/INTERNAL heuristics.
// gradients and cross-layer terms: computed in Python from the maps above,
//   where the multi-scale tensor already exists.

// Run count for this deck: {len(scales_um)} scales x {len(layers)} layers
// x up to 4 marker types.
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
