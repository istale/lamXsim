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
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .evidence import EvidenceClass


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


def _residualise(target: np.ndarray, explained_by: np.ndarray) -> np.ndarray:
    """Target with the part a linear fit on the other variable removed."""
    ok = np.isfinite(target) & np.isfinite(explained_by)
    out = np.full(len(target), np.nan)
    if ok.sum() < 3 or np.std(explained_by[ok]) == 0:
        out[ok] = target[ok]
        return out
    slope, intercept = np.polyfit(explained_by[ok], target[ok], 1)
    out[ok] = target[ok] - (slope * explained_by[ok] + intercept)
    return out


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
    if mask is not None:
        features = {k: np.where(mask, v, np.nan) for k, v in features.items()}
    if channel.channel_id == "perimeter_at_matched_density":
        if "metal_density" not in features:
            series = features["perimeter_density"]
            reason = ("metal density unavailable, so the raw perimeter is "
                      "used and the channel degenerates towards a density map")
        else:
            series = _residualise(features["perimeter_density"],
                                  features["metal_density"])
            reason = ""
        combined = _percentile_rank(series, channel.two_sided,
                                    invert=channel.invert)
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
    ranks = {c: _percentile_rank(features[c], channel.two_sided,
                                 invert=channel.invert) for c in used}
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
    return ChannelResult(channel, combined, used, True,
                         "" if len(missing) == 0
                         else f"missing {missing}, scored on {list(used)}",
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
