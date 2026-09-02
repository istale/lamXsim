"""Literature channels, the atlas they build, and the reports.

Consolidated from ``exposure.py``, ``atlas.py``, ``report.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
import json
import numpy as np
import pandas as pd
from . import foundation as registry
from . import geometry as grad_mod
from . import labels as package_context
from . import labels as position
from .foundation import EvidenceClass
from .geometry import GeometryExtractor, LayerStack, OrientationExtractor, StructureExtractor, ViaExtractor, build_multiscale, bump_relative_extract, crosslayer_extract
from .layout import LayoutReader


# ----------------------------------------------------------------------
# exposure.py
# ----------------------------------------------------------------------
"""Literature exposure channels: where a layout departs from a documented lever.

This is what the engine can say with no failure data at all. Each channel is
one mechanism from one paper, scored as a percentile **within this die**,
because nothing here is calibrated: with no measured failures there is no
scale on which an absolute threshold would mean anything. "In the top 5% of
this die on Cu/low-k boundary length at matched density" is a fact about the
layout. "5% risk" is not, and is not derivable from what is here.

Channels are never combined. Spec section 1 forbids an arbitrary weighted
delamination probability, and a weighted sum of these channels is exactly that
under another name -- the weights could only come from data this study does
not have. A location extreme on three channels produces three records with
three citations, not a score of three.

Two-sidedness is per channel, from the physics. Zahedmanesh & Vanstreels
(2019) show a stiff top group *lowering* the crack driving force beneath it,
so flagging only the high end of via architecture would be inventing a
direction the literature explicitly denies.
"""
@dataclass(frozen=True)
class Channel:
    """One literature mechanism, expressed as a GDS observable."""
    channel_id: str
    mechanism: str
    references: tuple[str, ...]
    observable: str
    #: Feature columns this channel reads, in order of preference.
    inputs: tuple[str, ...]
    #: True where the literature does not fix a direction, so both tails are
    #: reported rather than one being assumed harmful.
    two_sided: bool
    #: Quantities needed to turn this exposure into a risk, none of which is
    #: in a GDS. Reported per channel so the gap travels with the finding.
    unsupported_physics: tuple[str, ...]
    #: True where the departure is the *low* end of the observable, because
    #: the literature recommends having more of it. Ranking such a channel
    #: two-sided would score the recommended design as exposed.
    invert: bool = False
    #: Inputs whose departure is at the low end, where the channel's own
    #: ``invert`` is not the right answer for all of them. One flag shared by
    #: every input is wrong wherever a channel reads two quantities pointing
    #: opposite ways -- a narrowest width, where low is the departure, beside
    #: an asymmetry, where high is. Naming the inputs makes the direction a
    #: property of the observable rather than of the channel.
    invert_inputs: tuple[str, ...] = ()
    #: Optional gate: a cell is only a candidate where this feature is in the
    #: top ``conditional_percentile`` of the die. The 20 nm study locates the
    #: global loading at the outermost bumps first and only then compares
    #: layers beneath them, so a channel that skips the conditioning is
    #: answering a broader question than the paper asked.
    conditional_on: str = ""
    conditional_percentile: float = 75.0
    #: True when the conditioning feature's *low* end is the relevant region,
    #: as for distance to the nearest die corner.
    conditional_invert: bool = False
    #: "layer" for a channel whose inputs belong to one metal layer, "die"
    #: for one whose inputs are shared across the stack. A die-scoped channel
    #: evaluated per layer reports the same candidate once per layer, which
    #: reads as corroboration and is duplication.
    scope: str = "layer"

    #: True when the channel is measured from the die frame -- distance to a
    #: corner, offset from the die centre, the radial direction of a bump.
    #: None of those means anything unless the die outline is known, and the
    #: geometry bounding box is the die outline only when the GDS happens to
    #: be a whole die. Nothing in a layout says which it is, so the manifest
    #: has to, and a channel marked here is refused when it does not.
    needs_die_frame: bool = False
    #: True for a channel whose lever is about the topmost metal group. Scored
    #: on that layer alone; scoring it on every layer would report a claim the
    #: citation does not make about the layers beneath.
    top_layer_only: bool = False
    requires: tuple[str, ...] = ()
    note: str = ""


def _validate_channels(channels) -> None:
    """Every named direction has to name a real input of that channel.

    A typo here would silently fall back to the channel-wide flag, which is
    the behaviour the field exists to replace.
    """
    for channel in channels:
        unknown = set(channel.invert_inputs) - set(channel.inputs)
        if unknown:
            raise ValueError(
                f"{channel.channel_id}: invert_inputs names {sorted(unknown)}, "
                f"which {'is' if len(unknown) == 1 else 'are'} not among its "
                f"inputs {list(channel.inputs)}")


CHANNELS: tuple[Channel, ...] = (
    Channel(
        channel_id="perimeter_at_matched_density",
        mechanism="Cu/low-k boundary length is the interface available for "
                  "interfacial damage, and was more decisive than pattern "
                  "density in a dummy-pattern CMP experiment",
        references=("yoo2004perimeter",),
        observable="perimeter density, with the part explained by metal "
                   "density removed",
        inputs=("perimeter_density", "metal_density"),
        two_sided=False, scope="layer",
        unsupported_physics=("Cu/low-k interfacial adhesion", "CMP down-force "
                             "and slurry chemistry", "ULK modulus and porosity"),
        note="The residual, not the raw value: Yoo's result is that perimeter "
             "beats density, so a channel reading perimeter directly would "
             "rank the densest regions and reproduce a density map.",
    ),
    Channel(
        channel_id="termination",
        mechanism="delamination observed at terminated tips and corners "
                  "rather than along parallel comb lines",
        references=("tan2008delamination",),
        observable="line-end density and re-entrant corner density",
        inputs=("line_end_density", "concave_corner_density"),
        two_sided=False, scope="layer",
        unsupported_physics=("interface fracture toughness", "residual stress "
                             "at the cap interface", "local stress "
                             "concentration factor"),
    ),
    Channel(
        channel_id="via_architecture",
        mechanism="metal and via density set the layer's effective stiffness; "
                  "a stiff top group can shield the layer beneath it, so the "
                  "sign is not universally 'denser is worse'",
        references=("vanstreels2020beol", "zahedmanesh2019metallization"),
        observable="via area and count density",
        inputs=("via_density", "via_count_density"),
        two_sided=True, scope="layer",
        unsupported_physics=("layer elastic moduli", "energy release rate at "
                             "the pre-crack", "package-level loading"),
        note="Two-sided because the shielding result denies a fixed direction.",
    ),
    Channel(
        channel_id="layout_transition",
        mechanism="an abrupt change in metallisation concentrates load even "
                  "where either local value is ordinary",
        references=("rabie2018cpi", "vanstreels2020beol"),
        observable="spatial gradient magnitude of density and perimeter",
        inputs=("metal_density_grad_mag", "perimeter_density_grad_mag"),
        two_sided=False, scope="layer",
        unsupported_physics=("stiffness contrast across the transition",
                             "thermal expansion mismatch"),
    ),
    Channel(
        channel_id="cross_layer_mismatch",
        mechanism="BEOL architecture, not any single layer, correlates with "
                  "observed fracture; the topmost group's cross-sectional "
                  "metal area stood out",
        references=("vanstreels2020beol", "zahedmanesh2019metallization"),
        observable="top-to-underlying density and orientation mismatch",
        inputs=("top_to_underlying_density_mismatch",
                "top_to_underlying_orientation_mismatch"),
        two_sided=True, scope="die",
        unsupported_physics=("per-layer moduli and thicknesses",
                             "interface toughness of each pair"),
        requires=("two or more metal layers",),
    ),
    Channel(
        channel_id="wide_metal_slotting",
        mechanism="a continuous span of wide metal carries the stiffness "
                  "mismatch across its whole extent; slotting breaks the span, "
                  "and Rabie lists wide-metal slotting among the layout levers",
        references=("rabie2018cpi",),
        observable="wide-metal area fraction that is not slotted, from a "
                   "morphological opening at the declared wide-metal width",
        inputs=("unslotted_wide_metal_fraction",),
        two_sided=False, scope="layer",
        unsupported_physics=("stiffness contrast at the wide-metal boundary",
                             "package-level warpage", "EMC and underfill CTE"),
        requires=("a declared wide_width_um",),
        note="Unslotted wide metal, not wide metal. Ranking wide-metal "
             "fraction alone would flag a correctly slotted plate exactly as "
             "hard as an unbroken one, which inverts the lever: slotting is "
             "the recommended state, so its presence must lower the score. "
             "Can co-fire with corner_metal_tiles on top-layer geometry near "
             "a die corner; that is one piece of geometry seen through two of "
             "Rabie's levers, not two independent observations.",
    ),
    Channel(
        channel_id="corner_metal_tiles",
        mechanism="corner metal tiling is the first of Rabie's die-corner "
                  "levers: unbroken top metal at the die corner couples the "
                  "package corner load straight into the stack",
        references=("rabie2018cpi",),
        observable="unslotted wide-metal fraction on the topmost metal layer, "
                   "inside the die-corner region",
        inputs=("unslotted_wide_metal_fraction",),
        two_sided=False, scope="layer", top_layer_only=True,
        needs_die_frame=True,
        conditional_on="distance_to_nearest_corner", conditional_invert=True,
        unsupported_physics=("EMC thickness", "underfill CTE and modulus",
                             "package warpage", "corner bump stiffness"),
        requires=("a declared die outline",),
        note="Top layer only and corner only, because that is the lever as "
             "stated. Scored on every layer it would assert something about "
             "the layers beneath that the reference does not; scored die-wide "
             "it would be the wide_metal_slotting channel under a second "
             "citation.",
    ),
    Channel(
        channel_id="pad_geometry_departure",
        mechanism="pad geometry is one of Rabie's five layout levers; a pad "
                  "that departs from the recommended shape, or that sits off "
                  "the bump it carries, changes how the package load enters "
                  "the stack at that site",
        references=("rabie2018cpi",),
        observable="departure of the drawn pad's plan-view corner angles from "
                   "the declared target, and the pad-to-bump centroid offset",
        inputs=("pad_corner_angle_departure_deg",
                "pad_bump_centroid_offset_um"),
        two_sided=False, scope="die",
        unsupported_physics=("pad stiffness", "bond and joint quality",
                             "assembly overlay", "package warpage"),
        requires=("a pad layer", "shape_targets.pad_corner_angle_deg"),
        note="Departure from a declared target, not risk. The target is the "
             "manifest's, because nothing in a GDS says which pad shape a "
             "process recommends. Drawn geometry only: assembly overlay and "
             "the manufactured pad are not in a layout, so a concentric drawn "
             "pair says nothing about the assembled one. Where every pad is "
             "identical the ranking reports no candidate rather than picking "
             "among equals.",
    ),
    Channel(
        channel_id="pi_opening_shape",
        mechanism="Li et al. vary the PI opening directly and locate the "
                  "critical BEOL stress at its edge, so the opening's size "
                  "and elongation are levers in their own right, separately "
                  "from how close a cell is to one",
        references=("li2023beol_failure_locations", "li2025beol_design_factors"),
        observable="drawn opening area, equivalent diameter, aspect ratio "
                   "and plan-view corner-angle departure",
        inputs=("pi_aspect_ratio", "pi_corner_angle_departure_deg",
                "pi_equivalent_diameter_um"),
        two_sided=True, scope="die",
        unsupported_physics=("PI modulus and CTE", "opening sidewall and "
                             "taper angle, which no layout contains",
                             "EMC thickness", "energy release rate"),
        requires=("a PI opening layer",),
        note="Two-sided: the studies vary the opening and report the response "
             "without fixing a direction that holds for every stack, so "
             "flagging only large openings would invent one. Plan view only. "
             "A sidewall or taper angle is not derivable from a GDS by any "
             "means -- there is no Z information in a layout -- and the "
             "manifest refuses to accept one under that name.",
    ),
    Channel(
        channel_id="crackstop_structure",
        mechanism="the crackstop lever Rabie reports is about the ring "
                  "itself -- how wide it is, whether there are two, and "
                  "whether it is continuous -- not about how far a cell is "
                  "from it",
        references=("rabie2018cpi",),
        observable="the seal ring's local drawn width and the length of any "
                   "break in it, per analysis window, measured where the ring "
                   "actually runs",
        inputs=("crackstop_local_width_um", "crackstop_local_gap_um"),
        two_sided=False, invert=False, invert_inputs=("crackstop_local_width_um",),
        scope="die",
        unsupported_physics=("crack arrest effectiveness", "interface "
                             "toughness at the ring", "dicing damage"),
        requires=("a crackstop layer",),
        note="A local width map, because two coarser versions of this "
             "channel could not report anything at all. A whole-ring number "
             "broadcast to every cell has no variation to rank; a per-quadrant "
             "corner width puts a quarter of the die on one value, and a "
             "quarter of the cells tied sit at the 88th percentile, below the "
             "95th the atlas selects at, however narrow that corner is. The "
             "width where the ring actually runs ranks ring against ring and "
             "points at the pinch. Inverted: narrow is the departure. Rail "
             "count, continuity, gap count and the per-corner figures are "
             "still extracted -- they compare die rather than locate within "
             "one, and they are in package_objects.csv for that. Distance to "
             "the crackstop is a different feature and stays separate.\n"
             "The two inputs point opposite ways -- a narrow rail is the "
             "departure at the low end, a long break at the high end -- which "
             "is what invert_inputs is for. A break is invisible to the width "
             "map: where the ring is absent there is nothing to measure, the "
             "cell is NaN, and NaN is not an extreme.",
    ),
    Channel(
        channel_id="routing_in_bump_frame",
        mechanism="the package loads the layout through the bumps, and "
                  "diagonal final metal under the corner bumps is one of the "
                  "documented levers, so routing that is radial or tangential "
                  "there is the departure",
        references=("rabie2018cpi",),
        observable="routing direction resolved against the bump radial "
                   "direction",
        inputs=("routing_diagonality",),
        two_sided=False, invert=True, scope="layer", needs_die_frame=True,
        conditional_on="distance_to_nearest_corner", conditional_invert=True,
        unsupported_physics=("EMC thickness", "underfill CTE and modulus",
                             "bump stiffness", "package warpage",
                             "thermal cycle profile"),
        requires=("a bump layer",),
        note="Inverted, not two-sided. Diagonality is already folded: radial "
             "and tangential both sit at 0 and diagonal at 1, so a two-sided "
             "rank would score Rabie's recommendation as the departure. "
             "Conditioned on die-corner proximity, because the recommendation "
             "is about the corner bumps.",
    ),
    Channel(
        channel_id="pi_opening_proximity",
        mechanism="the BEOL stress concentration sits near the PI opening of "
                  "the bumps farthest from the die centre",
        references=("rabie2018cpi", "li2023beol_failure_locations",
                    "li2025beol_design_factors"),
        observable="distance to the nearest PI-opening edge and corner, "
                   "within the outermost-bump region",
        inputs=("distance_to_nearest_pi_opening",
                "distance_to_pi_opening_corner"),
        two_sided=False, invert=True, scope="die", needs_die_frame=True,
        conditional_on="nearest_bump_distance_from_die_center",
        unsupported_physics=("EMC thickness", "underfill CTE and modulus",
                             "bump stiffness", "package warpage",
                             "thermal cycle profile"),
        requires=("a PI-opening layer",),
        note="Scored die-wide: the opening is a package feature, the same for "
             "every metal layer, so scoring it per layer would report one "
             "candidate as several. Conditioned on the nearest bump being one "
             "of the outermost, which is where the 20 nm study places the "
             "global loading before it compares anything beneath.",
    ),
)


@dataclass
class ChannelResult:
    channel: Channel
    #: Percentile within the die, 0-100, per cell. NaN where the inputs were
    #: not available.
    percentile: np.ndarray
    #: Which input produced the percentile, per channel.
    inputs_used: tuple[str, ...]
    available: bool
    reason: str = ""
    evidence_class: EvidenceClass = EvidenceClass.GDS_GEOMETRY
    values: dict = field(default_factory=dict)
    #: Which input carried each cell, so a candidate names what triggered it.
    triggering_input: "np.ndarray | None" = None
    #: Cells the channel's conditioning excluded, if it has any.
    excluded_by_condition: "np.ndarray | None" = None


def _percentile_rank(values: np.ndarray, two_sided: bool = False, *,
                     invert: bool = False) -> np.ndarray:
    """Rank within the die, 0-100. Ties share the average rank.

    A percentile is the strongest claim available without calibration: it
    orders this die against itself and says nothing about any other die, any
    other technology, or any probability.

    Ties must share a rank. Ordinal ranking on a uniform map hands the last
    cell in array order a percentile of 100 and makes it a candidate, which is
    an artefact of row-major traversal rather than anything about the layout --
    and uniform density, a regular via array and constant routing orientation
    are all ordinary. A map with no variation produces no candidates at all.

    ``invert`` is for a channel whose departure is at the *low* end, where the
    literature recommends having more of the quantity rather than less.
    """
    from scipy.stats import rankdata

    out = np.full(len(values), np.nan)
    ok = np.isfinite(values)
    if ok.sum() < 2:
        return out
    v = values[ok]
    if np.ptp(v) == 0:
        # No variation: every cell is the median of itself. Ranking would
        # invent an order that only exists in the array.
        out[ok] = 0.0
        return out
    if invert:
        v = -v
    pct = 100.0 * (rankdata(v, method="average") - 1) / max(len(v) - 1, 1)
    if two_sided:
        # Distance from the middle, so both extremes rank high and the
        # literature's refusal to fix a direction is preserved.
        pct = 2.0 * np.abs(pct - 50.0)
    out[ok] = pct
    return out


#: A regressor whose spread is below this fraction of its own magnitude is
#: constant as far as any layout is concerned. Below it the fit is a constant
#: column, which numpy warns about and then answers anyway.
RESIDUAL_SPREAD_FLOOR = 1e-9


def _residualise(target: np.ndarray, explained_by: np.ndarray
                 ) -> "tuple[np.ndarray, str]":
    """Target with the part a linear fit on the other variable removed.

    Returns the note as well as the values, because when the fit cannot be
    made the channel is no longer measuring what its name says and the report
    has to carry that.

    The guard is on *relative* spread. Testing ``std == 0`` let floating-point
    dust through: a die of uniform metal density gave a spread of exactly zero
    and a standard deviation of 5.6e-17, so the fit went ahead on a constant
    column, numpy printed "Polyfit may be poorly conditioned", and the channel
    returned a residual computed from a degenerate fit. A user saw a warning
    they could not act on, attached to a number that was arithmetic noise.
    """
    ok = np.isfinite(target) & np.isfinite(explained_by)
    out = np.full(len(target), np.nan)
    if ok.sum() < 3:
        out[ok] = target[ok]
        return out, ("fewer than three cells carry both values, so nothing "
                     "can be regressed away and the raw perimeter is used")

    x = explained_by[ok]
    scale = max(float(np.max(np.abs(x))), 1.0)
    if float(np.ptp(x)) <= RESIDUAL_SPREAD_FLOOR * scale:
        out[ok] = target[ok]
        return out, ("metal density does not vary across this die, so there "
                     "is nothing for it to explain and the raw perimeter is "
                     "used; the channel degenerates towards a perimeter map "
                     "and Yoo's result -- that perimeter beats density -- "
                     "cannot be separated here")

    slope, intercept = np.polyfit(x, target[ok], 1)
    out[ok] = target[ok] - (slope * x + intercept)
    return out, ""


def condition_mask(channel: Channel, features: dict[str, np.ndarray],
                   n_cells: int) -> "tuple[np.ndarray | None, str]":
    """Cells where the channel's literature conditioning holds.

    Returns ``None`` for the mask when the channel declares a condition whose
    feature is not available: the channel cannot be scored as its citation
    describes, and the caller must refuse it rather than widen it.
    """
    if not channel.conditional_on:
        return np.ones(n_cells, dtype=bool), ""
    if channel.conditional_on not in features:
        # Not a fallback. Widening to the whole die answers a broader question
        # than the citation asked, under the citation's name -- the same
        # failure as scoring the condition and not applying it, arrived at
        # from the other side. The channel is refused instead.
        return None, (
            f"the conditioning feature {channel.conditional_on} is not "
            "available, and this channel's citation is about the region that "
            "feature defines. Scoring it die-wide would report a broader "
            "claim than the reference supports")
    gate = _percentile_rank(features[channel.conditional_on],
                            two_sided=False, invert=channel.conditional_invert)
    return (np.isfinite(gate) & (gate >= channel.conditional_percentile)), ""


def evaluate(channel: Channel, features: dict[str, np.ndarray],
             n_cells: int, mask: "np.ndarray | None" = None) -> ChannelResult:
    """Score one channel over a grid, or say why it could not be scored.

    ``mask`` restricts *which cells are ranked against each other*, not which
    are reported afterwards. Li et al. (2023) place the loading at the
    outermost bumps and compare within that region; ranking die-wide and
    filtering afterwards asks whether a cell is extreme across the whole die
    *and* happens to sit there, which is a different and usually empty
    question.
    """
    missing = [c for c in channel.inputs if c not in features]
    if len(missing) == len(channel.inputs):
        return ChannelResult(
            channel=channel, percentile=np.full(n_cells, np.nan),
            inputs_used=(), available=False,
            reason=f"none of {list(channel.inputs)} is available"
                   + (f"; this channel needs {list(channel.requires)}"
                      if channel.requires else ""))

    used = tuple(c for c in channel.inputs if c in features)
    partial = tuple(c for c in channel.inputs if c not in features)
    partial_note = (
        f"scored without {list(partial)}, which {'is' if len(partial) == 1 else 'are'} "
        "not available; the channel is narrower than its observable says"
        if partial else "")
    if mask is not None:
        features = {k: np.where(mask, v, np.nan) for k, v in features.items()}
    if channel.channel_id == "perimeter_at_matched_density":
        if "metal_density" not in features:
            series = features["perimeter_density"]
            reason = ("metal density unavailable, so the raw perimeter is "
                      "used and the channel degenerates towards a density map")
        else:
            series, reason = _residualise(features["perimeter_density"],
                                          features["metal_density"])
        combined = _percentile_rank(series, channel.two_sided,
                                    invert=channel.invert)
        reason = "; ".join(x for x in (reason, partial_note) if x)
        return ChannelResult(channel, combined, used, True, reason,
                             values={"perimeter_residual": series},
                             triggering_input=np.array(
                                 ["perimeter_residual"] * len(series)))

    # Several inputs: rank each, then take the strongest. A mean would let a
    # quiet input dilute a genuine extreme, and a sum would be the weighted
    # score this module exists to avoid.
    # Each input is ranked, the strongest is taken, and the result is ranked
    # again. Thresholding the maximum directly selects the union of each
    # input's top tail -- on two uncorrelated inputs that is close to twice the
    # intended fraction, while the report says "top 5%". Re-ranking makes the
    # stated fraction true.
    ranks = {c: _percentile_rank(
        features[c], channel.two_sided,
        invert=(c in channel.invert_inputs) if channel.invert_inputs
        else channel.invert) for c in used}
    stacked = np.vstack([ranks[c] for c in used])
    all_nan = np.all(~np.isfinite(stacked), axis=0)
    strongest = np.full(stacked.shape[1], np.nan)
    trigger = np.full(stacked.shape[1], "", dtype=object)
    if (~all_nan).any():
        strongest[~all_nan] = np.nanmax(stacked[:, ~all_nan], axis=0)
        winner = np.nanargmax(np.nan_to_num(stacked[:, ~all_nan], nan=-1.0),
                              axis=0)
        trigger[~all_nan] = np.array(used)[winner]
    combined = (_percentile_rank(strongest, two_sided=False)
                if len(used) > 1 else strongest)
    return ChannelResult(channel, combined, used, True, partial_note,
                         values=ranks, triggering_input=trigger)


def evaluate_all(features: dict[str, np.ndarray], n_cells: int
                 ) -> list[ChannelResult]:
    """Every channel, each ranked inside the region its citation covers."""
    out = []
    for channel in CHANNELS:
        mask, note = condition_mask(channel, features, n_cells)
        if mask is None:
            out.append(ChannelResult(
                channel=channel, percentile=np.full(n_cells, np.nan),
                inputs_used=(), available=False, reason=note))
            continue
        result = evaluate(channel, features, n_cells,
                          mask=None if not channel.conditional_on else mask)
        if note:
            result.reason = ((result.reason + "; ") if result.reason else "") + note
        result.excluded_by_condition = ~mask
        out.append(result)
    return out


_validate_channels(CHANNELS)


def tie_compression_note(result: ChannelResult, threshold: float) -> str:
    """Why an available channel can report nothing.

    Ties share a rank, so a large group at the extreme sits below the
    threshold: twenty cells sharing the most extreme of six distinct values
    are not the top 5% of the die, they are the top 18%. Reporting them anyway
    would be picking arbitrarily among identical values. Reporting nothing
    without saying why looks like the channel found nothing.
    """
    pct = result.percentile
    ok = np.isfinite(pct)
    if not result.available or ok.sum() == 0 or (pct[ok] >= threshold).any():
        return ""
    top = pct[ok].max()
    n_tied = int((pct[ok] >= top - 1e-9).sum())
    return (f"no candidate: the most extreme group holds {n_tied} of "
            f"{int(ok.sum())} ranked cells ({100 * n_tied / ok.sum():.0f}% of "
            f"them), so it sits at percentile {top:.0f} and below the "
            f"{threshold:.0f} threshold. The input takes too few distinct "
            "values at this scale to isolate a smaller group, and choosing "
            "among identical values would be arbitrary.")

# ----------------------------------------------------------------------
# atlas.py
# ----------------------------------------------------------------------
"""The GDS-only deliverable: a literature exposure atlas.

Given a layout and a study manifest, and no failure data at all, this reports
where the layout departs from levers the literature documents -- with the
citation, the exact GDS observable, and the physics that would be needed to
turn the departure into a risk.

What it is:

* a deterministic geometry fact (evidence level 1) for every feature map;
* a mechanistic engineering hypothesis (level 3) for every candidate region,
  which is a reason to look there first;
* an input to a measurement campaign, and to the Phase 0 sample-size question.

What it is not: a statistical association (level 2 needs measured failures),
a failure probability, or a design rule. Nothing here is calibrated, so
candidates are ranked by percentile *within this die* and by nothing else.
"""
#: A cell is a candidate on a channel when it sits at or above this percentile
#: of the die. Chosen for review bandwidth, not calibrated against anything.
CANDIDATE_PERCENTILE = 95.0


@dataclass
class Atlas:
    features: pd.DataFrame
    channels: dict[float, list]
    candidates: pd.DataFrame
    metadata: dict = field(default_factory=dict)


def _apply_calibre(vals, source, layer, grid, bbox, provenance):
    """Replace the maps the deck supplied, and record which those were.

    Only the features in ``CALIBRE_SUPPLIED`` are taken. Everything else the
    load produces -- the uncorrected band in particular -- stays out of the
    atlas: it exists to be compared against the corrected value, not to be
    scored as a feature under a name a reader would take for perimeter.
    """
    from .calibre import CALIBRE_SUPPLIED

    supplied = source.features_for(layer, grid.scale_um, bbox, grid=grid)
    taken = []
    for name, values in supplied.items():
        if name not in CALIBRE_SUPPLIED or name not in vals:
            continue
        vals[name] = values
        taken.append(name)
    provenance.setdefault(layer, {})[float(grid.scale_um)] = sorted(taken)
    return vals


def _extract_scale(reader, manifest, grid, bbox, calibre=None, provenance=None):
    """Every feature map available at one scale, with no failure data."""
    geo = GeometryExtractor(reader, line_rules=manifest.line_rule_map())
    ori = OrientationExtractor(reader)
    via = ViaExtractor(reader)
    struct = StructureExtractor(reader, wide_width_um=manifest.wide_width_um,
                                fill_layers=manifest.fill_layers)

    per_layer, flat = {}, {}
    for spec in manifest.metal_layers:
        vals = dict(geo.extract(spec, grid))
        vals.update(ori.extract(spec, grid))
        vals.update(struct.extract(spec, grid))
        via_spec = manifest.via_layers.get(spec.name)
        if via_spec is not None:
            vals.update(via.extract(via_spec, grid))
        if calibre is not None:
            _apply_calibre(vals, calibre, spec.name, grid, bbox, provenance)
            if via_spec is not None and via_spec.name in calibre.layers():
                _apply_calibre(vals, calibre, via_spec.name, grid, bbox,
                               provenance)
        per_layer[spec.name] = vals
        vals = dict(vals)
        vals.update(grad_mod.gradient_set(
            per_layer[spec.name], grid,
            only=("metal_density", "perimeter_density", "corner_density",
                  "line_end_density", "via_density")))
        for name, v in vals.items():
            flat[f"{name}|{spec.name}"] = v

    if len(manifest.metal_layers) > 1:
        stack = LayerStack(tuple(s.name for s in manifest.metal_layers))
        for name, v in crosslayer_extract(per_layer, stack).items():
            flat[f"{name}|CROSS"] = v

    for name, v in position.position_extract(grid, bbox).items():
        flat[f"{name}|-"] = v

    if manifest.package_layers.any_present:
        ctx = package_context.package_context_extract(grid, bbox, reader, manifest.package_layers)
        # Object-level shape, kept separate from the window scan above. A
        # pad's aspect ratio belongs to the pad; averaging it into a window
        # first would destroy exactly the quantity the shape levers are about.
        ctx.update(package_context.extract_shapes(
            grid, bbox, reader, manifest.package_layers,
            manifest.shape_semantics))
        for name, v in ctx.items():
            flat[f"{name}|-"] = v
        radial = ctx.get("bump_radial_direction_rad")
        if radial is not None and np.isfinite(radial).any():
            for spec in manifest.metal_layers:
                base = per_layer[spec.name]
                for name, v in bump_relative_extract(
                        base["routing_direction_rad"],
                        base["orientation_coherence"], radial).items():
                    flat[f"{name}|{spec.name}"] = v
    return flat, per_layer


def _channel_inputs(flat: dict, layer: str) -> dict[str, np.ndarray]:
    """Feature maps for one layer, under their bare names."""
    out = {}
    for key, values in flat.items():
        name, _, owner = key.partition("|")
        if owner in (layer, "-", "CROSS"):
            out.setdefault(name, values)
    return out


def _score(channel, inputs, n_cells, die_frame_declared, no_frame_reason):
    """One channel, with its literature conditioning actually applied.

    The conditioning used to be attached to the record and not to the
    computation: the candidate rows carried ``conditioned_on`` while the
    ranking ran die-wide. On the regression die that meant all 28
    ``routing_in_bump_frame`` candidates sat *outside* the corner region the
    citation is about -- the condition named in the report excluded every row
    the report contained. Ranking inside the region gives 8, in different
    places.

    A channel measured from the die frame is refused outright when no die
    outline was declared, rather than scored against a frame that may not be
    the die.
    """
    if channel.needs_die_frame and not die_frame_declared:
        return ChannelResult(
            channel=channel, percentile=np.full(n_cells, np.nan),
            inputs_used=(), available=False, reason=no_frame_reason)

    mask, note = condition_mask(channel, inputs, n_cells)
    if mask is None:
        return ChannelResult(
            channel=channel, percentile=np.full(n_cells, np.nan),
            inputs_used=(), available=False, reason=note)
    result = evaluate(channel, inputs, n_cells,
                               mask=None if not channel.conditional_on else mask)
    if note:
        result.reason = ((result.reason + "; ") if result.reason else "") + note
    result.excluded_by_condition = ~mask
    return result


def build(gds_path: str, manifest, *, candidate_percentile: float =
          CANDIDATE_PERCENTILE, calibre_dir: str | None = None) -> Atlas:
    from .workflow import _covers
    from .workflow import _fmt

    reader = LayoutReader(gds_path, top_cell=manifest.top_cell)
    manifest.validate_against(reader)
    geometry_bbox = reader.bbox()
    die_bbox = manifest.die_bbox(reader)

    # The same contract the correlation pipeline enforces. Without it a die
    # outline that does not contain the layout is accepted and every
    # die-relative quantity -- centre, normalised position, bump radial
    # direction, corner context -- is measured from a frame that does not
    # exist.
    if not _covers(die_bbox, geometry_bbox):
        raise ValueError(
            f"the declared die outline {_fmt(die_bbox)} does not contain the "
            f"loaded geometry {_fmt(geometry_bbox)}. One of them is wrong, and "
            "every die-relative channel depends on which.")
    # Whether the loaded geometry is a whole die or a region cut out of one is
    # not decidable from the layout: with no declared outline the die bbox is
    # the geometry bbox either way, so a test comparing the two can only ever
    # be false. An earlier version had exactly that test and a flag that was
    # therefore always false, disabling nothing. The frame has to be declared,
    # and the channels that need one are refused when it is not.
    die_frame_declared = bool(manifest.die_outline_um)
    no_die_frame_reason = (
        "" if die_frame_declared else
        "no die_outline_um in the manifest, so the die frame is the bounding "
        "box of whatever geometry this file holds. Corner distance, offset "
        "from the die centre and bump radial direction are measured from that "
        "frame, and if this GDS is a region cut out of a larger die they are "
        "all measured from a frame that does not exist. Nothing in a layout "
        "distinguishes the two cases, so this channel is not scored")

    calibre, calibre_provenance, eps_guard = None, {}, {}
    if calibre_dir:
        from . import calibre as calibre_ingest

        calibre = calibre_ingest.discover(calibre_dir)
        wanted = [s.name for s in manifest.metal_layers]
        wanted += [v.name for v in manifest.via_layers.values()]
        absent = [l for l in wanted if l not in calibre.layers()]
        if absent:
            raise ValueError(
                f"the deck output in {calibre_dir} has no layer {absent}; the "
                "manifest analyses them, so they would fall back to the "
                "KLayout extractor while the run reported itself as a deck "
                "extraction.")
        calibre.check_binding(gds_path, reader, manifest)
        calibre.check_contract(manifest)
        calibre.check_complete(wanted, manifest.scales_um)
        eps_guard = calibre.check_eps_guard(wanted)

    grids = build_multiscale(geometry_bbox, manifest.scales_um)
    frames, channels, candidate_rows = [], {}, []

    for scale, grid in sorted(grids.items()):
        flat, _ = _extract_scale(reader, manifest, grid, die_bbox,
                                 calibre, calibre_provenance)
        frame = pd.DataFrame(grid.to_arrays())
        frame = pd.concat([frame, pd.DataFrame(flat, index=frame.index)], axis=1)
        frames.append(frame)

        scale_channels = []
        # Die-scoped channels once, layer-scoped channels per layer. Reporting
        # a shared package or cross-layer feature under every metal layer
        # would present one candidate as several and read as corroboration.
        # Die-scoped channels once, layer-scoped channels per layer. Reporting
        # a shared package or cross-layer feature under every metal layer
        # would present one candidate as several and read as corroboration.
        # A top-layer lever is scored on the topmost layer only, for the same
        # reason: on M7 it would assert something about M7 that the citation
        # says about the top group.
        top_layer = manifest.metal_layers[0].name
        scored = [("-", c, _channel_inputs(flat, "-"))
                  for c in CHANNELS if c.scope == "die"]
        scored += [(spec.name, c, _channel_inputs(flat, spec.name))
                   for spec in manifest.metal_layers
                   for c in CHANNELS
                   if c.scope == "layer"
                   and (not c.top_layer_only or spec.name == top_layer)]

        for owner, channel, inputs in scored:
            for result in [_score(channel, inputs, len(grid),
                                  die_frame_declared, no_die_frame_reason)]:
                scale_channels.append((owner, result))
                if result.available:
                    note = tie_compression_note(result,
                                                         candidate_percentile)
                    if note:
                        result.reason = ((result.reason + "; ") if result.reason
                                         else "") + note
                # Every reason carries where it applies. Without it one run
                # said via_architecture was unavailable while listing M8 via
                # candidates -- the layer with no via layer was M7 -- and two
                # layers reporting the same tie compression collapsed into one
                # line that named neither.
                if result.reason:
                    result.reason = f"{result.reason} [{owner} @ {scale:g}um]"
                if not result.available:
                    continue
                pct = result.percentile
                flagged = np.where(np.isfinite(pct)
                                   & (pct >= candidate_percentile))[0]
                for i in flagged:
                    cell = grid.cells[i]
                    candidate_rows.append({
                        "channel": result.channel.channel_id,
                        "layer": owner,
                        "scale_um": scale,
                        "x_um": cell.x_center, "y_um": cell.y_center,
                        "percentile_in_die": round(float(pct[i]), 2),
                        "inputs_used": ";".join(result.inputs_used),
                        "triggered_by": (str(result.triggering_input[i])
                                         if result.triggering_input is not None
                                         else ""),
                        "conditioned_on": result.channel.conditional_on,
                        "condition_cells": (
                            int((~result.excluded_by_condition).sum())
                            if result.channel.conditional_on
                            and result.excluded_by_condition is not None
                            else ""),
                        "references": ";".join(result.channel.references),
                        "mechanism": result.channel.mechanism,
                        "two_sided": result.channel.two_sided,
                        "unsupported_physics":
                            ";".join(result.channel.unsupported_physics),
                    })
        channels[scale] = scale_channels

    candidates = pd.DataFrame(candidate_rows)
    if not candidates.empty:
        candidates = candidates.sort_values(
            ["channel", "scale_um", "percentile_in_die"],
            ascending=[True, True, False])

    metadata = {
        "gds_path": str(gds_path),
        "top_cell": reader.top.name,
        "geometry_bbox_um": [geometry_bbox.xmin, geometry_bbox.ymin,
                             geometry_bbox.xmax, geometry_bbox.ymax],
        "die_bbox_um": [die_bbox.xmin, die_bbox.ymin,
                        die_bbox.xmax, die_bbox.ymax],
        "die_outline_declared": bool(manifest.die_outline_um),
        "die_frame_declared": die_frame_declared,
        "die_relative_channels_disabled": not die_frame_declared,
        "scales_um": sorted(grids),
        "candidate_percentile": candidate_percentile,
        "feature_source": ("calibre" if calibre else "klayout"),
        "calibre": ({
            "directory": str(calibre_dir),
            "emulated": calibre.emulated,
            "generator": calibre.manifest.get("generator", ""),
            "binding": calibre.manifest.get("binding", {}),
            "eps_um": {l: calibre.eps_um(l) for l in calibre.layers()},
            "eps_guard_violations": eps_guard,
            "features_taken": {l: v for l, v in calibre_provenance.items()},
            "note": ("features not listed here were computed in Python from "
                     "the GDS: orientation, gradients, cross-layer terms, "
                     "position and package context are not in the deck."),
        } if calibre else None),
        "manifest": manifest.report(),
        "evidence_level": (
            "1 for the feature maps (deterministic geometry, checkable against "
            "KLayout or Calibre); 3 for the candidates (a mechanistic "
            "engineering hypothesis with a citation). Not level 2: no measured "
            "failure was involved, so nothing here is a statistical "
            "association, a probability, or a design rule."),
    }
    return Atlas(features=pd.concat(frames, ignore_index=True),
                 channels=channels, candidates=candidates, metadata=metadata)


def _bbox_from(values) -> "BBox":
    from .layout import BBox

    return BBox(*[float(v) for v in values])


def _overlay_gds(atlas: Atlas, path: Path, manifest, *,
                 base_layer: int = 200) -> dict:
    """One marker layer per channel, never a combined hotspot layer.

    A single merged layer is what a downstream reader opens and treats as the
    answer. Keeping the channels apart in the file keeps the citation attached
    to the mark, and makes it impossible to read a location flagged on three
    mechanisms as three times worse than one flagged on one.
    """
    from .layout import SynthLayout

    if atlas.candidates.empty:
        return {}
    sl = SynthLayout(top_name="EXPOSURE")
    mapping = {}
    for i, channel in enumerate(sorted(atlas.candidates.channel.unique())):
        layer_no = base_layer + i
        mapping[channel] = layer_no
        rows = atlas.candidates[atlas.candidates.channel == channel]
        for _, r in rows.iterrows():
            half = r.scale_um / 2
            sl.add_box(layer_no, r.x_um - half, r.y_um - half,
                       r.x_um + half, r.y_um + half)
    sl.write(str(path))
    return mapping


def _object_table(gds_path: str, manifest, die_bbox) -> pd.DataFrame:
    """Every package object, one row each, before any window averaging.

    Written out because the grid destroys it: once a pad's aspect ratio is a
    window mean, the pad it came from, the definition it was computed with and
    any doubt about which bump it was matched to are all gone. An engineer
    reading a flagged region needs to be able to ask which pad, and this is
    the file that answers.
    """
    if not manifest.package_layers.any_present:
        return pd.DataFrame()
    reader = LayoutReader(gds_path, top_cell=manifest.top_cell)
    table, matches, extra = package_context.object_table(
        reader, manifest.package_layers, manifest.shape_semantics, die_bbox)

    matched = {}
    for (primary, secondary), rows in matches.items():
        for m in rows:
            matched[m.primary_id] = {
                "matched_to": m.secondary_id,
                "match_rule": m.rule,
                "match_ambiguity": m.ambiguity,
                "centroid_offset_um": m.centroid_offset_um,
                "radial_offset_um": m.radial_offset_um,
                "tangential_offset_um": m.tangential_offset_um,
                "overlap_fraction": m.overlap_fraction,
            }

    rows = []
    for kind, objects in table.items():
        for o in objects:
            row = o.as_row()
            row.update(matched.get(o.object_id, {
                "matched_to": "", "match_rule": "",
                "match_ambiguity": "no match under the declared rule",
                "centroid_offset_um": float("nan"),
                "radial_offset_um": float("nan"),
                "tangential_offset_um": float("nan"),
                "overlap_fraction": float("nan")}))
            if kind == "crackstop":
                row.update(_crackstop_columns(extra))
            row["geometry"] = "drawn, plan view; not manufactured"
            rows.append(row)
    return pd.DataFrame(rows)


def _crackstop_columns(extra: dict) -> dict:
    """The structure facts, on the crackstop rows that claim to carry them.

    The channel note and the limits document both said rail count, continuity
    and the per-corner figures were in this file. They were computed, they
    were in the feature maps, and the crackstop row here held only the generic
    shape columns -- a traceability claim that the final output did not
    consume.
    """
    structure = extra.get("structure")
    topology = extra.get("corner_topology") or {}
    out = {}
    if structure is not None:
        out.update({
            "rail_count": structure.n_rails,
            "rail_width_min_um": structure.rail_width_min_um,
            "rail_width_median_um": structure.rail_width_median_um,
            "rail_width_p10_um": structure.rail_width_p10_um,
            "rail_spacing_um": structure.rail_spacing_um,
            "n_components": structure.n_components,
            "continuity_ratio": structure.continuity_ratio,
            "n_gaps": structure.n_gaps,
            "structure_undefined_reason": structure.undefined_reason,
        })
    if topology:
        out["corner_window_um"] = topology.get("window_um", float("nan"))
        out["corner_narrowest_um"] = topology.get("corner_narrowest_um",
                                                  float("nan"))
        out["corner_asymmetry_um"] = topology.get("corner_asymmetry",
                                                  float("nan"))
        out["corner_undefined_reason"] = topology.get("undefined_reason", "")
        for name, values in (topology.get("per_corner") or {}).items():
            out[f"corner_{name}_narrowest_um"] = values["narrowest_um"]
            out[f"corner_{name}_pieces"] = values["n_pieces"]
    return out


def _traceability(atlas: Atlas) -> pd.DataFrame:
    """Every channel, and every feature it read, against the registry."""
    rows = []
    for channel in CHANNELS:
        for feature in channel.inputs:
            entry = registry.lookup(feature)
            rows.append({
                "channel": channel.channel_id,
                "mechanism": channel.mechanism,
                "references": ";".join(channel.references),
                "gds_observable": channel.observable,
                "feature": feature,
                "registry_family": entry.family if entry else "",
                "registry_complete": bool(entry and not entry.missing_trace),
                "physical_hypothesis": (entry.row.get("physical_hypothesis", "")
                                        if entry else ""),
                "discrimination_test": (entry.row.get("discrimination_test", "")
                                        if entry else ""),
                "falsification": (entry.row.get("falsification", "")
                                  if entry else ""),
                "two_sided": channel.two_sided,
                "requires": ";".join(channel.requires),
            })
    return pd.DataFrame(rows)


#: Observables the literature varies that a GDS does contain, and that this
#: repository has not implemented. Keeping them apart from the genuinely
#: unavailable physics matters: one list is work, the other is a limit.
#: Each row is (area, observable, reference, status, covered_by, why).
#:
#: ``status`` is the part that keeps this file honest, and it is why the row
#: shape changed. The list once said "corner metal tiles" and "wide-metal
#: slotting" were unimplemented after both had become channels, so the
#: coverage statement shipped to a user contradicted the channels shipped
#: beside it. Three statuses, and a test that a channel's own subject can only
#: appear here as ``partial`` or ``not_recoverable``:
#:
#: * ``absent`` -- nothing of this is implemented.
#: * ``partial`` -- the channels in ``covered_by`` cover part of it, and the
#:   row says which part is still missing. Naming them is what makes the
#:   check exact: a word match on the observable text called a crackstop row
#:   a contradiction of corner_metal_tiles because both contain "corner".
#: * ``not_recoverable`` -- no layout can supply it at all. Listed anyway
#:   because each has a GDS-derived proxy nearby that is easy to mistake for
#:   it.
UNIMPLEMENTED_GDS_OBSERVABLES = (
    ("top metal", "explicit corner-tile morphology: tile size, count, pitch "
     "and the topology of the tile array",
     "rabie2018cpi", "partial", ("corner_metal_tiles",),
     "the corner_metal_tiles channel scores a proxy -- unslotted wide metal "
     "inside the corner region -- which says a corner is unbroken but not how "
     "it is tiled. Recovering the array itself needs the tile layer named in "
     "the manifest, which nothing currently asks for"),
    ("bump geometry", "a scored channel for drawn bump geometry",
     "li2025beol_design_factors", "partial", (),
     "the descriptors are extracted per bump and reported as feature maps, "
     "but no channel scores them: the study varies bump geometry and reports "
     "the response without fixing a direction that holds across stacks, and "
     "inventing one is what a channel would do"),
    ("bump geometry", "mechanically critical bump identity",
     "li2023beol_failure_locations", "not_recoverable", (),
     "the outermost ring is flagged as a geometric fact, which is as far as a "
     "layout goes. Which bump carries the largest driving force depends on "
     "package loading and on the stiffness of everything above it"),
    ("crackstop", "rail-to-rail spacing beyond the outermost pair, and the "
     "connectivity graph inside a corner window",
     "rabie2018cpi", "partial", ("crackstop_structure",),
     "the local width and the length of any break are mapped per window and "
     "are what the channel ranks; rail count, continuity, gap count and the "
     "per-corner figures are extracted for comparing die. A stack of three or "
     "more rails reports only the outer spacing, and a corner is summarised "
     "by width and piece count rather than by how its rails are bridged"),
    ("shape measurement", "an object-level quantisation declared per layer "
     "rather than taken from the database unit",
     "-", "absent", (),
     "descriptors are rounded to the database unit and corner angles to 0.1 "
     "degrees, which is what a layout can express. A process whose drawn grid "
     "is coarser than its database unit would want its own figure, and "
     "nothing currently asks for one"),
    ("PI opening", "sidewall and taper angle",
     "rabie2018cpi", "not_recoverable", (),
     "a GDS holds no Z information, so no vertical angle exists in it. The "
     "manifest refuses a sidewall angle offered under the plan-view key "
     "rather than silently treating one as the other"),
)


def _unimplemented_observables() -> pd.DataFrame:
    return pd.DataFrame([
        {"area": area, "observable": observable, "reference": ref,
         "status": status, "covered_by": ";".join(covered),
         "recoverable_from_gds": status != "not_recoverable",
         "why_it_matters": why}
        for area, observable, ref, status, covered, why
        in UNIMPLEMENTED_GDS_OBSERVABLES])


def _unsupported_physics(atlas: Atlas) -> pd.DataFrame:
    """Per channel, what no GDS can supply and what its absence costs."""
    rows = []
    for channel in CHANNELS:
        for quantity in channel.unsupported_physics:
            rows.append({
                "channel": channel.channel_id,
                "quantity": quantity,
                "why_it_matters": channel.mechanism,
                "recoverable_from_gds": False,
                "consequence": (
                    "the channel reports where the layout is unusual, not "
                    "where it is at risk; without this quantity the departure "
                    "cannot be converted into a driving force"),
            })
    return pd.DataFrame(rows)


def _limits_document(atlas: Atlas, manifest, overlay: dict) -> str:
    n = len(atlas.candidates)
    channels = (atlas.candidates.channel.value_counts().to_dict()
                if n else {})
    unavailable = []
    for scale_channels in atlas.channels.values():
        for layer, result in scale_channels:
            if not result.available:
                unavailable.append(
                    f"- `{result.channel.channel_id}` on {layer}: {result.reason}")
    lines = [
        "# What this atlas is, and what it is not",
        "",
        "## What it is",
        "",
        "Every feature map is a **deterministic geometry fact** -- checkable "
        "against KLayout or Calibre, independent of any failure data.",
        "",
        f"Every one of the {n} candidate records is a **mechanistic "
        "engineering hypothesis**: a location where this layout departs from a "
        "lever the literature documents, with the citation attached. It is a "
        "reason to look there first.",
        "",
        "## What it is not",
        "",
        "- Not a statistical association. No measured failure was involved.",
        "- Not a probability. Nothing here is calibrated, so candidates are "
        "ranked by percentile **within this die** and by nothing else. The "
        "same layout on another package or process would rank the same and "
        "mean something different.",
        "- Not a design rule. That needs held-out hardware.",
        "- Not a combined risk score. Channels are reported separately "
        "because combining them requires weights, and the weights could only "
        "come from data this study does not have. A location flagged on three "
        "channels is three records with three citations, not a score of three.",
        "",
        "## Channels reported",
        "",
    ]
    for channel in CHANNELS:
        count = channels.get(channel.channel_id, 0)
        lines.append(f"**{channel.channel_id}** -- {count} candidate(s), "
                     f"{'two-sided' if channel.two_sided else 'one-sided'}")
        lines.append(f"  - mechanism: {channel.mechanism}")
        lines.append(f"  - references: {', '.join(channel.references)}")
        lines.append(f"  - observable: {channel.observable}")
        if channel.note:
            lines.append(f"  - note: {channel.note}")
        lines.append("")

    if unavailable:
        lines += ["## Channels that could not be scored", ""]
        lines += sorted(set(unavailable))
        lines.append("")

    gaps = manifest.gaps
    if gaps:
        lines += ["## Declared gaps in the manifest", ""]
        lines += [f"- {g}" for g in gaps]
        lines.append("")

    if overlay:
        lines += ["## GDS overlay layers", ""]
        lines += [f"- layer {no}: `{ch}`" for ch, no in sorted(overlay.items(),
                                                               key=lambda kv: kv[1])]
        lines.append("")
        lines.append("There is deliberately no combined hotspot layer.")
        lines.append("")

    lines += ["## The die frame", ""]
    if atlas.metadata.get("die_frame_declared"):
        lines += [
            "A die outline is declared in the manifest, so distance to a "
            "corner, offset from the die centre and bump radial direction are "
            "measured from a frame the manifest vouches for.",
            "",
        ]
    else:
        lines += [
            "**No die outline is declared.** The die frame is therefore the "
            "bounding box of whatever geometry this file holds. If this GDS is "
            "a region cut out of a larger die, every die-relative quantity is "
            "measured from a frame that does not exist -- and nothing in a "
            "layout distinguishes the two cases, so the channels that depend "
            "on that frame were not scored at all rather than scored against "
            "a guess. Declare `die_outline_um` to enable them.",
            "",
            "This also means the candidates here must not be read as \"the "
            "regions of the die most worth inspecting\": they are the extremes "
            "of the geometry supplied.",
            "",
        ]

    lines += ["## Object-level shape, and what drawn geometry is", ""]
    lines += [
        "Bump, pad, PI-opening and crackstop descriptors are computed per "
        "object and only then projected onto the grid; `package_objects.csv` "
        "holds one row per object with its id, source layer, the definition "
        "each descriptor was computed with, and any doubt about which object "
        "it was matched to. A window mean would have destroyed all of that: "
        "two pads of equal area and opposite elongation produce the same "
        "mean.",
        "",
        "All of it is **drawn** geometry in plan view. A GDS says what was "
        "drawn, not what was manufactured, so none of these is the "
        "post-reflow bump, the printed opening after lithography, or the "
        "assembled overlay -- a drawn pad concentric with its drawn bump says "
        "nothing about the assembled pair. No sidewall or taper angle is "
        "derivable at all: a layout holds no Z information, and the manifest "
        "refuses a sidewall angle offered under a plan-view key rather than "
        "silently treating one as the other.",
        "",
        "The outermost bump ring is flagged as a geometric fact. Which bump "
        "carries the largest driving force depends on package loading and on "
        "the stiffness of everything above it, none of which is in a layout.",
        "",
        "Descriptors are rounded to the database unit, and corner angles to "
        "0.1 degrees. Below that there is no geometry, only the arithmetic of "
        "computing a centroid from snapped vertices -- left in, it gets "
        "ranked, and on a die of identical pads it manufactures a candidate "
        "out of rounding.",
        "",
    ]

    conditioned = [c for c in CHANNELS if c.conditional_on]
    if conditioned:
        lines += ["## Literature conditioning", ""]
        for channel in conditioned:
            lines.append(
                f"- `{channel.channel_id}` is ranked **inside** the top "
                f"{100 - channel.conditional_percentile:g}% of "
                f"`{channel.conditional_on}`"
                + (" (low end)" if channel.conditional_invert else "")
                + ", because that is the region its citation is about. Cells "
                  "outside it are not ranked and cannot be candidates.")
        lines.append("")

    cal = atlas.metadata.get("calibre")
    lines += ["## Where the feature maps came from", ""]
    if cal:
        taken = sorted({f for by_scale in cal["features_taken"].values()
                        for fs in by_scale.values() for f in fs})
        lines += [
            f"Density and count maps were read from a rule-deck run in "
            f"`{cal['directory']}` (`{cal['generator']}`): "
            f"{', '.join(taken) if taken else 'nothing matched'}.",
            "",
            "Everything else -- orientation, gradients, cross-layer terms, "
            "position and package context -- was computed in Python from the "
            "GDS, because the deck does not produce it.",
            "",
            "Per-layer eps used for the perimeter band: "
            + ", ".join(f"{k} {v:g}um" for k, v in sorted(cal["eps_um"].items())
                        if v) + ".",
            "",
            "The deck's minimum-width guard ran and was empty on every metal "
            "layer ("
            + ", ".join(f"{k}: {v}" for k, v in
                        sorted(cal["eps_guard_violations"].items()))
            + "). A non-empty one would have stopped this run: eps is a "
              "quarter of the declared minimum width, and the inside band "
              "collapses silently once eps passes half the real one.",
            "",
        ]
        if cal["emulated"]:
            lines += [
                "**These maps were emulated, not produced by Calibre.** They "
                "come from a KLayout statement of what each rule means, which "
                "checks the ingest path and the grid alignment but cannot "
                "check the tool. Do not report these as a Calibre result.",
                "",
            ]
    else:
        lines += [
            "All feature maps were extracted with KLayout from the GDS "
            "directly. `lamxsim characterize --features-from DIR` reads the "
            "density and count maps from a Calibre deck run instead; on the "
            "regression die the two paths produce the same candidate set.",
            "",
        ]

    lines += [
        "## What would turn this into evidence",
        "",
        "Measured failure locations in the same coordinate frame, with "
        "lot/wafer/die identity, a registration fiducial set, an inspected "
        "footprint, and the failed layer or interface. "
        "`unsupported_non_gds_physics.csv` lists the package and material "
        "quantities that no GDS contains and that a study must hold fixed, "
        "stratify, or measure. `unimplemented_gds_observables.csv` carries "
        "everything else this atlas does not cover, with a status on every "
        "row: `absent` when nothing of it is implemented, `partial` when a "
        "channel covers part of it -- the row names that channel and says "
        "which part is still missing -- and `not_recoverable` when no layout "
        "can supply it at all. The last are listed anyway, because each has a "
        "GDS-derived proxy nearby that is easy to mistake for it.",
        "",
        "Run `lamxsim phase0` for how many failure sites the association "
        "analysis would need before it could say anything at all.",
    ]
    return "\n".join(lines) + "\n"


def write(atlas: Atlas, outdir: str | Path, manifest) -> dict[str, str]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}

    p = out / "feature_maps.parquet"
    atlas.features.to_parquet(p, index=False)
    paths["feature_maps"] = str(p)

    p = out / "literature_candidates.csv"
    atlas.candidates.to_csv(p, index=False)
    paths["literature_candidates"] = str(p)

    p = out / "literature_traceability.csv"
    _traceability(atlas).to_csv(p, index=False)
    paths["literature_traceability"] = str(p)

    p = out / "unsupported_non_gds_physics.csv"
    _unsupported_physics(atlas).to_csv(p, index=False)
    paths["unsupported_physics"] = str(p)

    # Separated deliberately: "the GDS has it and we have not implemented it"
    # is a backlog, and "no GDS contains it" is a limit. Filing the first
    # under the second makes the tool look more bounded than it is.
    p = out / "unimplemented_gds_observables.csv"
    _unimplemented_observables().to_csv(p, index=False)
    paths["unimplemented_gds_observables"] = str(p)

    overlay = {}
    if not atlas.candidates.empty:
        p = out / "candidate_regions.gds"
        overlay = _overlay_gds(atlas, p, manifest)
        paths["candidate_regions"] = str(p)
    atlas.metadata["overlay_layers"] = overlay

    objects = _object_table(atlas.metadata["gds_path"], manifest,
                            _bbox_from(atlas.metadata["die_bbox_um"]))
    if not objects.empty:
        p = out / "package_objects.csv"
        objects.to_csv(p, index=False)
        paths["package_objects"] = str(p)

    p = out / "assumptions_and_limits.md"
    p.write_text(_limits_document(atlas, manifest, overlay))
    paths["assumptions_and_limits"] = str(p)

    p = out / "atlas_metadata.json"
    p.write_text(json.dumps(atlas.metadata, indent=2, default=str))
    paths["metadata"] = str(p)
    return paths

# ----------------------------------------------------------------------
# report.py
# ----------------------------------------------------------------------
"""Result presentation, separated by what each row is allowed to claim.

`best_features.csv` sorted by effect size is not an engineering
recommendation list. Ranking every row together puts an exploratory
descriptor measured at a scale the registration cannot support above a
literature-backed feature measured at one it can, and the file gives the
reader no way to tell.

Rows are therefore partitioned before they are ranked, along the three axes
that decide what a row may be used for:

* **evidence class** -- a PACKAGE_POSITION row is a confounder being
  controlled, not a finding.
* **hypothesis tier** -- exploratory families carry no FDR-corrected
  significance claim, because correcting them alongside the primary ones is
  what makes the primary ones unreachable.
* **scale trustworthiness** -- a scale below roughly three times the
  positional uncertainty measures registration noise.
* **a spatially corrected q-value** -- a row whose significance rests on a
  test that assumed grid cells are independent has not been corrected for the
  thing most likely to have produced it. On a die with no package-position
  effect that test called 11 of 12 position associations significant where the
  block permutation called none.
* **a complete registry entry** -- a feature with no stated physical
  hypothesis, no named discrimination test and no falsification condition can
  be reported, but not as a primary finding. Auditing that and then printing
  the row anyway is not enforcement.

Only rows that clear all five appear in the primary table.

`primary` is the **pre-specified hypothesis set**: every combination the study
committed to testing, whatever it came out at. That is the right object to
correct over, and a row with q = 1 belongs in it. It is not a list of
findings, and read as one it is badly misleading, so `supported` is written
separately -- the subset whose spatially corrected q clears alpha and whose
interval excludes chance.
"""
PRIMARY_TIERS = ("tier1",)
CONFOUNDER_TIERS = ("tier1_confounder",)

#: Fewest cells a class may hold and still support an effect estimate.
#: At a coarse scale nearly every window can contain a failure, leaving a
#: handful of controls; the AUC then saturates at 1.0 with an interval of zero
#: width, and ranking by effect size alone puts that degenerate row above a
#: real, well-powered one.
MIN_CLASS_SIZE = 10

#: A supported finding must clear this on the spatially corrected q-value.
ALPHA = 0.05


def _is_traced(feature: str) -> bool:
    """Whether the feature has a complete registry entry behind it."""
    entry = registry.lookup(feature)
    return entry is not None and not entry.missing_trace


def _reasons(row) -> list[str]:
    out = []
    if row.get("evidence_class") != "GDS_GEOMETRY":
        out.append(f"evidence class {row.get('evidence_class')}")
    if row.get("hypothesis_tier") not in PRIMARY_TIERS:
        out.append(f"tier {row.get('hypothesis_tier')}")
    if row.get("scale_trustworthy") is False:
        out.append("scale below the registration floor")
    if not np.isfinite(row.get("fdr_q_value", np.nan)):
        out.append("no FDR correction applied")
    return out


def partition(associations: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split an association table into primary, confounder and exploratory."""
    if associations.empty:
        empty = associations.copy()
        return {k: empty.copy() for k in
                ("primary", "supported", "confounders", "exploratory",
                 "unsupported_scale", "underpowered",
                 "not_spatially_corrected", "not_traceable")}

    df = associations.copy()
    df["abs_effect"] = df["effect_size"].abs()
    trustworthy = df.get("scale_trustworthy")
    if trustworthy is None:
        ok_scale = pd.Series(True, index=df.index)
    else:
        # Strict: only an explicit True qualifies. An unknown registration
        # accuracy leaves the scale uncertified, and an uncertified scale
        # cannot carry a primary result -- treating unknown as good is how a
        # 25um conclusion survives a measurement that could not place a
        # failure to better than 100um.
        ok_scale = trustworthy.astype("object").eq(True)

    is_geometry = df["evidence_class"] == "GDS_GEOMETRY"
    is_primary_tier = df["hypothesis_tier"].isin(PRIMARY_TIERS)
    powered = ((df.get("n_case", 0) >= MIN_CLASS_SIZE)
               & (df.get("n_control", 0) >= MIN_CLASS_SIZE))
    # A primary claim is corrected from the within-die block permutation. Where
    # that was not run there is no primary evidence, only a naive diagnostic.
    if "spatial_q_value" in df.columns:
        spatially_corrected = df["spatial_q_value"].notna()
    else:
        spatially_corrected = pd.Series(False, index=df.index)

    traced = df["feature"].map(_is_traced) if "feature" in df.columns \
        else pd.Series(True, index=df.index)

    primary = df[is_geometry & is_primary_tier & ok_scale & powered
                 & spatially_corrected & traced]
    confounders = df[df["hypothesis_tier"].isin(CONFOUNDER_TIERS)]
    unsupported = df[~ok_scale]
    if "scale_status" in df.columns:
        unsupported = unsupported.assign(
            excluded_because=unsupported["scale_status"])
    underpowered = df[is_geometry & is_primary_tier & ok_scale & ~powered
                      & spatially_corrected]
    uncorrected = df[is_geometry & is_primary_tier & ok_scale & powered
                     & ~spatially_corrected & traced]
    untraced = df[is_geometry & is_primary_tier & ~traced]
    exploratory = df[is_geometry & ~is_primary_tier & ok_scale & powered]

    # Ranked by the effect the interval actually guarantees, so a wide
    # interval cannot outrank a tight one. An interval that straddles 0.5
    # guarantees nothing -- the data does not exclude "no effect" -- and
    # scores zero. Taking the nearer endpoint unconditionally would instead
    # reward an interval that is wide on both sides: on a real three-die run
    # that put a feature at AUC 0.512 with q = 0.94 above the driver at
    # AUC 0.698 with q = 0.0008.
    if "auc_ci_low" in df.columns and "auc_ci_high" in df.columns:
        for t in (primary, confounders, exploratory, unsupported,
                  underpowered, uncorrected, untraced):
            lo, hi = t["auc_ci_low"], t["auc_ci_high"]
            guaranteed = np.where(lo > 0.5, lo - 0.5,
                                  np.where(hi < 0.5, 0.5 - hi, 0.0))
            t["conservative_effect"] = np.where(
                np.isfinite(lo) & np.isfinite(hi), guaranteed, 0.0)
        order = ["conservative_effect", "abs_effect"]
    else:
        order = ["abs_effect"]

    # The subset that actually says something. Separating it is not a
    # statistical nicety: "primary_results.csv" is read as "the findings", and
    # the hypothesis set contains rows at q = 1 by construction.
    supported = primary
    if "spatial_q_value" in primary.columns:
        clears_alpha = primary["spatial_q_value"] <= ALPHA
        excludes_chance = primary.get("conservative_effect", 0) > 0 \
            if "conservative_effect" in primary.columns else True
        supported = primary[clears_alpha & excludes_chance]

    return {
        "primary": primary.sort_values(order, ascending=False),
        "supported": supported.sort_values(order, ascending=False),
        "confounders": confounders.sort_values(order, ascending=False),
        "exploratory": exploratory.sort_values(order, ascending=False),
        "unsupported_scale": unsupported.sort_values(order, ascending=False),
        "underpowered": underpowered.sort_values(order, ascending=False),
        "not_spatially_corrected": uncorrected.sort_values(order, ascending=False),
        "not_traceable": untraced.sort_values(order, ascending=False),
    }


#: Columns a reader needs to judge a row, in the order they should be read.
SUMMARY_COLUMNS = ("feature", "layer", "scale_um", "scale_status",
                   "roc_auc", "auc_ci_low", "auc_ci_high", "effect_size",
                   "spatial_q_value", "fdr_q_value", "n_case", "n_control",
                   "effective_n", "enrichment_top_10pct")


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in SUMMARY_COLUMNS if c in df.columns]
    return df[cols]


def write_reports(associations: pd.DataFrame, outdir: str | Path, *,
          metadata: dict | None = None) -> dict[str, str]:
    """Write the partitioned tables plus a machine-readable summary."""
    out = Path(outdir) / "reports"
    out.mkdir(parents=True, exist_ok=True)
    parts = partition(associations)
    paths = {}
    for name, table in parts.items():
        p = out / f"{name}.csv"
        summary_table(table).to_csv(p, index=False)
        paths[name] = str(p)

    counts = {k: int(len(v)) for k, v in parts.items()}
    lines = [
        "# Results by what each row may claim",
        "",
        f"- supported          {counts['supported']:5d}  **the findings**: "
        f"spatially corrected q <= {ALPHA}, with an interval excluding chance",
        f"- primary            {counts['primary']:5d}  the pre-specified "
        "hypothesis set that was corrected over -- it contains rows at q = 1 "
        "by construction and is not a list of findings",
        f"- confounders        {counts['confounders']:5d}  package position; "
        "controlled for, never a finding",
        f"- exploratory        {counts['exploratory']:5d}  no direct delamination "
        "evidence; effect size only, no significance claim",
        f"- unsupported_scale  {counts['unsupported_scale']:5d}  below the "
        "registration floor, or uncertified because the registration accuracy "
        "was never measured; excluded from every conclusion",
        f"- underpowered       {counts['underpowered']:5d}  fewer than "
        f"{MIN_CLASS_SIZE} cells in one class, so the effect estimate is "
        "degenerate however large it looks",
        f"- not_spatially_corrected {counts['not_spatially_corrected']:5d}  no "
        "block-permutation q-value, so significance rests on a test that "
        "assumed grid cells are independent",
        f"- not_traceable      {counts['not_traceable']:5d}  no complete "
        "registry entry: no stated physical hypothesis, discrimination test "
        "or falsification condition",
        "",
        "`spatial_q_value` is the within-die block permutation, corrected. "
        "`fdr_q_value` is Mann-Whitney, kept as a diagnostic: on a die with no "
        "position effect it called 11 of 12 position associations significant "
        "where the permutation called none.",
        "",
    ]
    if metadata:
        for note in metadata.get("uncontrolled_confounding", []):
            lines.append(f"UNCONTROLLED: {note}")
        if metadata.get("uncontrolled_confounding"):
            lines.append("")
    lines.append(
        "This is a statistical association within one study population. It is "
        "not a causal claim, not a failure probability, and not a design rule.")
    p = out / "README.md"
    p.write_text("\n".join(lines) + "\n")
    paths["summary"] = str(p)
    return paths


def format_primary(associations: pd.DataFrame, limit: int = 10) -> str:
    """The supported findings, not the hypothesis set.

    The console shows what the run actually supports; the full pre-specified
    set is in primary.csv for anyone checking the correction.
    """
    parts = partition(associations)
    table = summary_table(parts["supported"]).head(limit)
    if table.empty:
        n_primary = len(parts["primary"])
        if n_primary:
            return (f"no supported finding: {n_primary} hypotheses were tested "
                    f"and corrected, none reached a spatial q of {ALPHA} with "
                    "an interval excluding chance")
        return ("no primary hypothesis survived: every row was a confounder, an "
                "exploratory family, unsupported by the registration accuracy, "
                "underpowered, or not spatially corrected")
    return table.to_string(index=False, float_format=lambda v: f"{v:.4f}")
