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
    #: "layer" for a channel whose inputs belong to one metal layer, "die"
    #: for one whose inputs are shared across the stack. A die-scoped channel
    #: evaluated per layer reports the same candidate once per layer, which
    #: reads as corroboration and is duplication.
    scope: str = "layer"
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
        channel_id="routing_in_bump_frame",
        mechanism="the package loads the layout through the bumps, and "
                  "diagonal final metal under the corner bumps is one of the "
                  "documented levers, so routing that is radial or tangential "
                  "there is the departure",
        references=("rabie2018cpi",),
        observable="routing direction resolved against the bump radial "
                   "direction",
        inputs=("routing_diagonality",),
        two_sided=True, scope="layer",
        unsupported_physics=("EMC thickness", "underfill CTE and modulus",
                             "bump stiffness", "package warpage",
                             "thermal cycle profile"),
        requires=("a bump layer",),
        note="Two-sided: radial and tangential are both departures from the "
             "diagonal recommendation, at opposite ends of one axis.",
    ),
    Channel(
        channel_id="pi_opening_proximity",
        mechanism="the BEOL stress concentration sits near the PI opening of "
                  "the bumps farthest from the die centre",
        references=("rabie2018cpi",),
        observable="distance to the nearest PI-opening edge and corner",
        inputs=("distance_to_nearest_pi_opening",
                "distance_to_pi_opening_corner"),
        two_sided=False, scope="die",
        unsupported_physics=("PI opening angle and profile", "EMC thickness",
                             "underfill CTE", "bump stiffness"),
        requires=("a PI-opening layer",),
        note="Scored die-wide: the opening is a package feature, the same for "
             "every metal layer, so scoring it per layer would report one "
             "candidate as several.",
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


def _percentile_rank(values: np.ndarray, two_sided: bool) -> np.ndarray:
    """Rank within the die, 0-100. Both tails count when two-sided.

    A percentile is the strongest claim available without calibration: it
    orders this die against itself and says nothing about any other die, any
    other technology, or any probability.
    """
    out = np.full(len(values), np.nan)
    ok = np.isfinite(values)
    if ok.sum() < 2:
        return out
    v = values[ok]
    order = v.argsort().argsort().astype(float)
    pct = 100.0 * order / max(len(v) - 1, 1)
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


def evaluate(channel: Channel, features: dict[str, np.ndarray],
             n_cells: int) -> ChannelResult:
    """Score one channel over a grid, or say why it could not be scored."""
    missing = [c for c in channel.inputs if c not in features]
    if len(missing) == len(channel.inputs):
        return ChannelResult(
            channel=channel, percentile=np.full(n_cells, np.nan),
            inputs_used=(), available=False,
            reason=f"none of {list(channel.inputs)} is available"
                   + (f"; this channel needs {list(channel.requires)}"
                      if channel.requires else ""))

    used = tuple(c for c in channel.inputs if c in features)
    # For a one-sided channel whose departure is at the *low* end -- the
    # recommendation is to have more of it, not less -- the percentile is
    # inverted at the channel level rather than left to the reader.
    if channel.channel_id == "pi_opening_proximity":
        series = features[used[0]]
        return ChannelResult(
            channel, _percentile_rank(-series, False), used, True,
            "" if not missing else f"missing {missing}",
            values={used[0]: series})

    if channel.channel_id == "perimeter_at_matched_density":
        if "metal_density" not in features:
            series = features["perimeter_density"]
            reason = ("metal density unavailable, so the raw perimeter is "
                      "used and the channel degenerates towards a density map")
        else:
            series = _residualise(features["perimeter_density"],
                                  features["metal_density"])
            reason = ""
        combined = _percentile_rank(series, channel.two_sided)
        return ChannelResult(channel, combined, used, True, reason,
                             values={"perimeter_residual": series})

    # Several inputs: rank each, then take the strongest. A mean would let a
    # quiet input dilute a genuine extreme, and a sum would be the weighted
    # score this module exists to avoid.
    ranks = {c: _percentile_rank(features[c], channel.two_sided) for c in used}
    stacked = np.vstack([ranks[c] for c in used])
    all_nan = np.all(~np.isfinite(stacked), axis=0)
    combined = np.full(stacked.shape[1], np.nan)
    if (~all_nan).any():
        combined[~all_nan] = np.nanmax(stacked[:, ~all_nan], axis=0)
    return ChannelResult(channel, combined, used, True,
                         "" if len(missing) == 0
                         else f"missing {missing}, scored on {list(used)}",
                         values=ranks)


def evaluate_all(features: dict[str, np.ndarray], n_cells: int
                 ) -> list[ChannelResult]:
    return [evaluate(c, features, n_cells) for c in CHANNELS]
