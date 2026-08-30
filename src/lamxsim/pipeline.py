"""Layout -> features -> case/control -> association.

Runs one or more layers through geometry, orientation, gradient and
cross-layer extraction at every configured scale, then scores every
feature x layer x scale combination against the failure labels.

The statistical machinery was validated before the feature catalogue was
widened, so adding a feature family here is a mechanical extension of
something already known to report nothing when there is nothing to report.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .evidence import EvidenceClass
from .features import gradient as grad_mod
from .features.crosslayer import LayerStack
from .features import crosslayer
from .features.geometry import GeometryExtractor
from .features.grid import build_multiscale
from .features.orientation import OrientationExtractor
from .features.vias import ViaExtractor
from .labels import inspection, package_context, position
from .labels.failure import FailureSet, map_to_grid, map_to_grid_per_die
from .layout.reader import BBox, LayerSpec, LayoutReader


def _fmt(b) -> str:
    return f"[{b.xmin:g}, {b.ymin:g}] to [{b.xmax:g}, {b.ymax:g}]um"


def _covers(outer, inner, tol: float = 1e-6) -> bool:
    return (outer.xmin <= inner.xmin + tol and outer.ymin <= inner.ymin + tol
            and outer.xmax >= inner.xmax - tol and outer.ymax >= inner.ymax - tol)


def _is_roi(die, geometry, tol: float = 1e-6) -> bool:
    return (die.width > geometry.width + tol or die.height > geometry.height + tol)
from . import report as report_mod
from .stats import fdr, permutation, univariate

#: Hypothesis tiers, sourced from references/feature_evidence_map.csv.
#: Matching is by prefix so that gradients and layer-qualified cross-layer
#: names inherit the tier of the family they derive from.
TIER_PREFIXES = (
    ("metal_density", "tier1"),
    ("perimeter_density", "tier1"),
    ("line_end_density", "tier1"),
    ("via_density", "tier1"),
    ("via_count_density", "tier1"),
    ("mean_via_area", "exploratory"),
    ("corner_density", "tier1"),
    ("convex_corner_density", "tier1"),
    ("concave_corner_density", "tier1"),
    ("horizontal_fraction", "tier1"),
    ("vertical_fraction", "tier1"),
    ("orientation_anisotropy", "tier1"),
    ("density_difference", "tier1"),
    ("perimeter_density_difference", "tier1"),
    ("orientation_difference", "tier1"),
    ("line_end_density_difference", "tier1"),
    ("density_mismatch", "tier1"),
    ("perimeter_density_mismatch", "tier1"),
    ("orientation_mismatch", "tier1"),
    ("line_end_density_mismatch", "tier1"),
    ("cross_layer_transition_index", "tier1"),
    ("top_to_underlying", "tier1"),
    ("stacked_dense_layer_count", "exploratory"),
    ("stacked_sparse_layer_count", "exploratory"),
    ("density_variance_across_layers", "exploratory"),
    ("distance_to_", "tier1_confounder"),
    ("normalized_distance_", "tier1_confounder"),
    ("bump_", "tier1_confounder"),
    ("under_bump_indicator", "tier1_confounder"),
    ("local_bump_pitch", "tier1_confounder"),
)

POSITION_FEATURES = set(position.POSITION_FEATURES)


def tier_of(name: str) -> str:
    best = "exploratory"
    best_len = -1
    for prefix, tier in TIER_PREFIXES:
        if name.startswith(prefix) and len(prefix) > best_len:
            best, best_len = tier, len(prefix)
    return best


@dataclass
class RunResult:
    associations: pd.DataFrame
    permutations: pd.DataFrame
    features: pd.DataFrame
    metadata: dict = field(default_factory=dict)


#: Scalars whose spatial gradient is itself a tier-1 feature (spec section 5).
#: Gradients are not taken of every scalar: each one triples the hypothesis
#: count, and the literature motivates transitions in density, perimeter and
#: architecture rather than in every derived descriptor.
GRADIENT_OF = ("metal_density", "perimeter_density", "line_end_density",
               "corner_density", "orientation_anisotropy", "via_density")


def _extract_layer(reader, geo_ex, ori_ex, via_ex, layer, via_layer, grid, *,
                   with_gradients=True):
    vals = dict(geo_ex.extract(layer, grid))
    vals.update(ori_ex.extract(layer, grid))
    if via_layer is not None:
        vals.update(via_ex.extract(via_layer, grid))
    base = dict(vals)
    if with_gradients:
        vals.update(grad_mod.gradient_set(base, grid, only=GRADIENT_OF))
    return vals, base


def run(gds_path: str, failures: FailureSet, *,
        layer: LayerSpec | None = None,
        layers: list[LayerSpec] | None = None,
        via_layers: dict[str, LayerSpec] | None = None,
        scales_um=(25, 50, 100, 250, 500), n_permutations: int = 499,
        include_position: bool = True, with_gradients: bool = True,
        pair_selection: str = "adjacent_and_top",
        package_layers: "package_context.PackageLayers | None" = None,
        footprint: "inspection.InspectionFootprint | None" = None,
        min_coverage: float = 0.5,
        allow_pooling_modes: bool = False,
        allow_failures_outside_footprint: bool = False,
        die_bbox: "BBox | None" = None,
        top_cell: str | None = None,
        line_end_w_max_um: float | None = None,
        line_rules: "dict[str, tuple[float, float]] | None" = None,
        seed: int = 0) -> RunResult:
    t0 = time.time()
    specs = layers if layers is not None else [layer]
    if not specs or specs[0] is None:
        raise ValueError("pass layer= or layers=")

    reader = LayoutReader(gds_path, top_cell=top_cell)
    geo_ex = GeometryExtractor(reader, line_end_w_max_um=line_end_w_max_um,
                               line_rules=line_rules)
    ori_ex = OrientationExtractor(reader)
    via_ex = ViaExtractor(reader)
    # Vias are keyed by the metal layer they sit under, so via features carry
    # that metal layer's identity into the association table rather than
    # appearing as an unattached layer of their own.
    via_layers = via_layers or {}
    # Three frames, kept apart because they answer different questions and
    # substituting one for another silently changes what the die is.
    #
    #   geometry_bbox -- what this file actually contains. Sets the grid,
    #                    because features only exist where geometry does.
    #   die_bbox      -- the physical die the operator declares. Sets the die
    #                    centre, normalised position, edge distance and the
    #                    radial direction bump context is resolved along.
    #   footprint     -- where inspection looked. Sets the eligible population.
    #
    # Using the geometry bbox as the die is only correct when the file is the
    # whole die. On a region of interest it puts the die centre inside the
    # region, and every position feature and bump radial direction is then
    # measured from the wrong origin.
    geometry_bbox = reader.bbox()
    if die_bbox is None:
        die_bbox = geometry_bbox
        frame_note = ("no die outline declared: the geometry bounding box is "
                      "being used as the die, which is only correct if this "
                      "file is the whole die")
    elif not _covers(die_bbox, geometry_bbox):
        raise ValueError(
            f"the declared die outline {_fmt(die_bbox)} does not contain the "
            f"loaded geometry {_fmt(geometry_bbox)}. One of them is wrong, and "
            "every position feature depends on which.")
    elif _is_roi(die_bbox, geometry_bbox):
        frame_note = (
            f"region of interest: the file covers {_fmt(geometry_bbox)} of a "
            f"declared die {_fmt(die_bbox)}. Position features are measured "
            "from the declared die, but controls exist only inside the loaded "
            "region, so no claim about which part of the die is worst can be "
            "made from this run.")
    else:
        frame_note = ""

    bbox = die_bbox
    grids = build_multiscale(geometry_bbox, scales_um)
    stack = LayerStack(tuple(s.name for s in specs))
    scale_floor = failures.min_trustworthy_scale_um()
    package_layers = package_layers or package_context.PackageLayers()
    context_notes = package_context.absent_context_note(package_layers)
    if frame_note:
        context_notes.append(frame_note)

    # An uninspected cell is not a control, it is missing data. Without a
    # footprint the analysis silently treats never-inspected area as clean,
    # and any feature correlated with where inspection was targeted picks up
    # an association from that alone.
    if footprint is None:
        footprint = inspection.InspectionFootprint.full_die(
            geometry_bbox, "no inspection footprint supplied",
            dbu=reader.units.dbu)
        context_notes.append(
            "no inspection footprint supplied: the whole die is being treated "
            "as inspected, so every cell without a recorded failure counts as "
            "a control. If inspection was partial or targeted, features "
            "correlated with where it was targeted will show spurious "
            "association.")
    context_notes.extend(failures.assert_single_mode(
        allow_pooling=allow_pooling_modes))
    n_dies = failures.n_dies()
    if n_dies == 1:
        context_notes.append(
            "a single die: spec section 17 asks for held-out dies, so nothing "
            "here can be shown to generalise. Treat the result as a local "
            "diagnostic of this piece of silicon.")

    audit = inspection.audit_failures(footprint, failures, dbu=reader.units.dbu)
    if not audit["consistent"]:
        message = (
            f"{audit['n_outside_footprint']} of {audit['n_failures']} failures "
            f"lie outside the inspected footprint (e.g. "
            f"{audit['outside_sample_ids']}). Something was found where nothing "
            "was looked at, which disproves the population definition rather "
            "than merely qualifying it: the coordinate frame, the "
            "registration, the footprint or the die frame is wrong, and each "
            "of those invalidates a different part of the analysis.")
        if not allow_failures_outside_footprint:
            raise ValueError(
                message + " Fix the input, or pass "
                "allow_failures_outside_footprint=True to continue with those "
                "failures dropped -- the override is recorded in the metadata.")
        context_notes.append(
            message + " Continuing was asserted by the operator; those "
            "failures are dropped from the analysis.")

    assoc_rows, perm_rows, feat_frames = [], [], []

    coverage_summary = {}
    for scale, grid in sorted(grids.items()):
        # The observation unit is (cell, die). With one die this is the cell
        # itself; with several, each die contributes its own labels over the
        # same layout, and features repeat rather than labels being collapsed.
        per_die = map_to_grid_per_die(failures, grid)
        die_names = sorted(per_die)
        n_cells = len(grid)
        cell_index = np.tile(np.arange(n_cells), len(die_names))
        die_index = np.repeat(np.arange(len(die_names)), n_cells)
        y = np.concatenate([per_die[d]["failure_present"].astype(int)
                            for d in die_names])
        nearest = np.concatenate([per_die[d]["distance_to_nearest_failure"]
                                  for d in die_names])

        eligible_cell, cover = inspection.eligibility(
            footprint, grid, min_coverage=min_coverage, dbu=reader.units.dbu)
        eligible = eligible_cell[cell_index]
        coverage_summary[scale] = {
            "n_cells": len(grid), "n_dies": len(die_names),
            "n_observations": int(len(y)),
            "n_eligible": int(eligible.sum()),
            "n_cases_eligible": int(y[eligible].sum()),
            "n_cases_excluded": int(y[~eligible].sum()),
            "prevalence": float(y[eligible].mean()) if eligible.any() else float("nan"),
            "mean_coverage": float(cover.mean()),
        }

        cell_arrays = grid.to_arrays()
        frame = pd.DataFrame({k: v[cell_index] for k, v in cell_arrays.items()})
        frame["die_key"] = np.array(die_names)[die_index]
        frame["inspected_fraction"] = cover[cell_index]
        frame["eligible"] = eligible
        frame["failure_present"] = y
        frame["distance_to_nearest_failure"] = nearest

        columns: list[tuple[str, str, np.ndarray, EvidenceClass]] = []
        per_layer_base = {}

        for spec in specs:
            vals, base = _extract_layer(reader, geo_ex, ori_ex, via_ex, spec,
                                        via_layers.get(spec.name), grid,
                                        with_gradients=with_gradients)
            per_layer_base[spec.name] = base
            for name, v in vals.items():
                columns.append((name, spec.name, v, EvidenceClass.GDS_GEOMETRY))

        if len(specs) > 1:
            for name, v in crosslayer.extract(per_layer_base, stack,
                                              selection=pair_selection).items():
                columns.append((name, "CROSS", v, EvidenceClass.GDS_GEOMETRY))

        if include_position:
            for name, v in position.extract(grid, bbox).items():
                columns.append((name, "-", v, EvidenceClass.PACKAGE_POSITION))
            if package_layers.any_present:
                ctx = package_context.extract(grid, bbox, reader, package_layers)
                for name, v in ctx.items():
                    columns.append((name, "-", v, EvidenceClass.PACKAGE_POSITION))

        def observation_groups(cell_values):
            """Permutation groups: the spatial block, within one die.

            The block size comes from the feature's own spatial
            autocorrelation -- fixing it at one cell would turn the block
            permutation back into the naive per-cell shuffle it exists to
            replace. The die index makes each die's blocks distinct, and
            ``die_index`` is passed separately as the stratum so the exchange
            stays inside a die; the grouping alone does not achieve that.
            """
            size = max(permutation.autocorrelation_range_cells(cell_values, grid), 1)
            block_of_cell = permutation.spatial_block_ids(grid, size)
            span = int(block_of_cell.max()) + 1
            return die_index * span + block_of_cell[cell_index], size

        for name, layer_name, cell_vals, ecls in columns:
            vals = cell_vals[cell_index]
            frame[f"{name}|{layer_name}"] = vals
            finite = np.isfinite(vals) & eligible
            if finite.sum() < 8 or y[finite].sum() == 0:
                continue
            a = univariate.analyse(vals[finite], y[finite], feature=name,
                                   layer=layer_name, scale_um=scale,
                                   tier=tier_of(name))
            # effective_n and the CI need the grid, so only compute them on the
            # complete field; a gradient with its boundary ring dropped is
            # scored without them rather than with a wrong neighbour graph.
            # Computed on whatever is finite rather than skipped when
            # anything is not. A gradient drops its die-edge ring by design,
            # and skipping its interval would leave it ranked by effect size
            # alone -- which lets a feature that could not be given an
            # interval outrank one that could.
            finite_cell = np.isfinite(cell_vals) & eligible_cell
            if finite_cell.sum() >= 8:
                # Spatial dependence is a property of one die's lattice;
                # separate dies contribute independently, so the effective
                # count scales with the number of them.
                per_die_eff = univariate.effective_n(cell_vals, grid,
                                                     mask=finite_cell)
                a.effective_n = per_die_eff * len(die_names)
                ci_groups, _ = observation_groups(
                    np.where(finite_cell, cell_vals, np.nanmean(cell_vals[finite_cell])))
                a.auc_ci_low, a.auc_ci_high = univariate.block_bootstrap_auc_ci(
                    vals, y, grid, n_boot=299, seed=seed, mask=finite,
                    groups=ci_groups)
            row = a.as_row()
            row["evidence_class"] = ecls.value
            row["n_cells"] = len(grid)
            row["n_dies"] = len(die_names)
            row["n_eligible"] = int(eligible.sum())
            row["n_finite"] = int(finite.sum())
            # Tri-state on purpose. "We do not know the registration
            # accuracy" is not "the registration is good enough"; at 5-10um
            # line-end, via and corner scales it is the more dangerous of the
            # two, because nothing in the numbers looks wrong.
            if not np.isfinite(scale_floor):
                row["scale_trustworthy"] = None
                row["scale_status"] = "uncertified"
            elif scale >= scale_floor:
                row["scale_trustworthy"] = True
                row["scale_status"] = "supported"
            else:
                row["scale_trustworthy"] = False
                row["scale_status"] = "below_registration_floor"
            assoc_rows.append((a, row))

            if n_permutations and finite.sum() >= 8:
                filled = np.where(finite_cell, cell_vals,
                                  np.nanmean(cell_vals[finite_cell]))
                obs_groups, block_size = observation_groups(filled)
                pr = permutation.block_permutation_test(
                    vals, y, grid, n_permutations=n_permutations, seed=seed,
                    mask=finite, groups=obs_groups, block_cells=block_size,
                    strata=die_index)
                p = pr.as_row()
                p.update(feature=name, layer=layer_name, scale_um=scale)
                perm_rows.append(p)

        feat_frames.append(frame)

    if not assoc_rows:
        raise ValueError(
            "no feature x layer x scale combination could be scored. Every "
            "candidate had too few finite values or no failures in the grid; "
            "check that the failure coordinates lie inside the die bounding "
            f"box {[bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]} and are in "
            "layout coordinates (registration/apply.register does this).")

    fdr.apply_tiered([a for a, _ in assoc_rows])
    rows = []
    for a, row in assoc_rows:
        row["fdr_q_value"] = a.fdr_q_value
        rows.append(row)

    meta = {
        "gds_path": str(gds_path),
        "layers": [str(s) for s in specs],
        "via_layers": {k: str(v) for k, v in via_layers.items()},
        "package_layers": {k: (str(v) if v else None) for k, v in
                           vars(package_layers).items()},
        "uncontrolled_confounding": context_notes,
        "inspection_footprint": footprint.report(reader.units.dbu),
        "min_coverage": min_coverage,
        "coverage_by_scale": coverage_summary,
        "failure_footprint_audit": audit,
        "pair_selection": pair_selection if len(specs) > 1 else None,
        "with_gradients": with_gradients,
        "line_rules": line_rules or {},
        "die_bbox_um": [die_bbox.xmin, die_bbox.ymin, die_bbox.xmax, die_bbox.ymax],
        "geometry_bbox_um": [geometry_bbox.xmin, geometry_bbox.ymin,
                             geometry_bbox.xmax, geometry_bbox.ymax],
        "die_outline_declared": frame_note == "" or "region of interest" in frame_note,
        "top_cell": reader.top.name,
        "scales_um": sorted(grids),
        "n_failures": len(failures),
        "n_dies": n_dies,
        "failure_modes": failures.modes(),
        "failures_simulated": failures.simulated,
        "failure_source": failures.source,
        "failure_notes": failures.notes,
        "position_sigma_um": failures.position_sigma_um,
        "min_trustworthy_scale_um": scale_floor,
        "n_hypotheses_tested": len(rows),
        "n_permutations": n_permutations,
        "seed": seed,
        "runtime_s": round(time.time() - t0, 2),
    }
    return RunResult(
        associations=pd.DataFrame(rows),
        permutations=pd.DataFrame(perm_rows),
        features=pd.concat(feat_frames, ignore_index=True),
        metadata=meta,
    )


def write_results(result: RunResult, outdir: str | Path) -> dict[str, str]:
    out = Path(outdir)
    for sub in ("features", "reports", "metadata"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    paths = {}
    p = out / "features" / "feature_association.csv"
    result.associations.to_csv(p, index=False)
    paths["associations"] = str(p)

    # Partitioned by what each row may claim, rather than ranked together.
    # A single ranking puts an exploratory descriptor at an unsupported scale
    # above a literature-backed feature at a supported one, with nothing in
    # the file to tell them apart.
    paths.update(report_mod.write(result.associations, out,
                                  metadata=result.metadata))

    p = out / "features" / "spatial_features.parquet"
    result.features.to_parquet(p, index=False)
    paths["features"] = str(p)

    if len(result.permutations):
        p = out / "reports" / "spatial_permutation.csv"
        result.permutations.to_csv(p, index=False)
        paths["permutations"] = str(p)

    p = out / "metadata" / "run_metadata.json"
    p.write_text(json.dumps(result.metadata, indent=2))
    paths["metadata"] = str(p)
    return paths
