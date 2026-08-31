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
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import exposure, registry
from .pipeline import _covers, _fmt, _is_roi
from .features import gradient as grad_mod
from .features.bump_relative import extract as bump_relative_extract
from .features.crosslayer import LayerStack, extract as crosslayer_extract
from .features.geometry import GeometryExtractor
from .features.grid import build_multiscale
from .features.orientation import OrientationExtractor
from .features.structures import StructureExtractor
from .features.vias import ViaExtractor
from .labels import package_context, position
from .layout.reader import LayoutReader

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
    from .calibre.ingest import CALIBRE_SUPPLIED

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

    for name, v in position.extract(grid, bbox).items():
        flat[f"{name}|-"] = v

    if manifest.package_layers.any_present:
        ctx = package_context.extract(grid, bbox, reader, manifest.package_layers)
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
        return exposure.ChannelResult(
            channel=channel, percentile=np.full(n_cells, np.nan),
            inputs_used=(), available=False, reason=no_frame_reason)

    mask, note = exposure.condition_mask(channel, inputs, n_cells)
    if mask is None:
        return exposure.ChannelResult(
            channel=channel, percentile=np.full(n_cells, np.nan),
            inputs_used=(), available=False, reason=note)
    result = exposure.evaluate(channel, inputs, n_cells,
                               mask=None if not channel.conditional_on else mask)
    if note:
        result.reason = ((result.reason + "; ") if result.reason else "") + note
    result.excluded_by_condition = ~mask
    return result


def build(gds_path: str, manifest, *, candidate_percentile: float =
          CANDIDATE_PERCENTILE, calibre_dir: str | None = None) -> Atlas:
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
        from .calibre import ingest as calibre_ingest

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
                  for c in exposure.CHANNELS if c.scope == "die"]
        scored += [(spec.name, c, _channel_inputs(flat, spec.name))
                   for spec in manifest.metal_layers
                   for c in exposure.CHANNELS
                   if c.scope == "layer"
                   and (not c.top_layer_only or spec.name == top_layer)]

        for owner, channel, inputs in scored:
            for result in [_score(channel, inputs, len(grid),
                                  die_frame_declared, no_die_frame_reason)]:
                scale_channels.append((owner, result))
                if result.available:
                    note = exposure.tie_compression_note(result,
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
    from .layout.reader import BBox

    return BBox(*[float(v) for v in values])


def _overlay_gds(atlas: Atlas, path: Path, manifest, *,
                 base_layer: int = 200) -> dict:
    """One marker layer per channel, never a combined hotspot layer.

    A single merged layer is what a downstream reader opens and treats as the
    answer. Keeping the channels apart in the file keeps the citation attached
    to the mark, and makes it impossible to read a location flagged on three
    mechanisms as three times worse than one flagged on one.
    """
    from .layout.synth import SynthLayout

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
    for channel in exposure.CHANNELS:
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
    for channel in exposure.CHANNELS:
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
    for channel in exposure.CHANNELS:
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

    conditioned = [c for c in exposure.CHANNELS if c.conditional_on]
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
