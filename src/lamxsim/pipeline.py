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
from .labels import position
from .labels.failure import FailureSet, map_to_grid
from .layout.reader import LayerSpec, LayoutReader
from .stats import fdr, permutation, univariate

#: Hypothesis tiers, sourced from references/feature_evidence_map.csv.
#: Matching is by prefix so that gradients and layer-qualified cross-layer
#: names inherit the tier of the family they derive from.
TIER_PREFIXES = (
    ("metal_density", "tier1"),
    ("perimeter_density", "tier1"),
    ("line_end_density", "tier1"),
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


def _extract_layer(reader, geo_ex, ori_ex, layer, grid, *, with_gradients=True):
    vals = dict(geo_ex.extract(layer, grid))
    vals.update(ori_ex.extract(layer, grid))
    base = dict(vals)
    if with_gradients:
        vals.update(grad_mod.gradient_set(
            base, grid,
            only=("metal_density", "perimeter_density", "line_end_density",
                  "orientation_anisotropy")))
    return vals, base


def run(gds_path: str, failures: FailureSet, *,
        layer: LayerSpec | None = None,
        layers: list[LayerSpec] | None = None,
        scales_um=(25, 50, 100, 250, 500), n_permutations: int = 499,
        include_position: bool = True, with_gradients: bool = True,
        pair_selection: str = "adjacent_and_top",
        line_end_w_max_um: float | None = None, seed: int = 0) -> RunResult:
    t0 = time.time()
    specs = layers if layers is not None else [layer]
    if not specs or specs[0] is None:
        raise ValueError("pass layer= or layers=")

    reader = LayoutReader(gds_path)
    geo_ex = GeometryExtractor(reader, line_end_w_max_um=line_end_w_max_um)
    ori_ex = OrientationExtractor(reader)
    bbox = reader.bbox()
    grids = build_multiscale(bbox, scales_um)
    stack = LayerStack(tuple(s.name for s in specs))
    scale_floor = failures.min_trustworthy_scale_um()

    assoc_rows, perm_rows, feat_frames = [], [], []

    for scale, grid in sorted(grids.items()):
        labels = map_to_grid(failures, grid)
        y = labels["failure_present"].astype(int)

        frame = pd.DataFrame(grid.to_arrays())
        frame["failure_present"] = y
        frame["distance_to_nearest_failure"] = labels["distance_to_nearest_failure"]

        columns: list[tuple[str, str, np.ndarray, EvidenceClass]] = []
        per_layer_base = {}

        for spec in specs:
            vals, base = _extract_layer(reader, geo_ex, ori_ex, spec, grid,
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

        for name, layer_name, vals, ecls in columns:
            frame[f"{name}|{layer_name}"] = vals
            finite = np.isfinite(vals)
            if finite.sum() < 8 or y[finite].sum() == 0:
                continue
            a = univariate.analyse(vals[finite], y[finite], feature=name,
                                   layer=layer_name, scale_um=scale,
                                   tier=tier_of(name))
            # effective_n and the CI need the grid, so only compute them on the
            # complete field; a gradient with its boundary ring dropped is
            # scored without them rather than with a wrong neighbour graph.
            if finite.all():
                a.effective_n = univariate.effective_n(vals, grid)
                a.auc_ci_low, a.auc_ci_high = univariate.block_bootstrap_auc_ci(
                    vals, y, grid, n_boot=299, seed=seed)
            row = a.as_row()
            row["evidence_class"] = ecls.value
            row["n_cells"] = len(grid)
            row["n_finite"] = int(finite.sum())
            row["scale_trustworthy"] = (
                bool(scale >= scale_floor) if np.isfinite(scale_floor) else None)
            assoc_rows.append((a, row))

            if n_permutations and finite.all():
                pr = permutation.block_permutation_test(
                    vals, y, grid, n_permutations=n_permutations, seed=seed)
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
        "pair_selection": pair_selection if len(specs) > 1 else None,
        "with_gradients": with_gradients,
        "die_bbox_um": [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax],
        "scales_um": sorted(grids),
        "n_failures": len(failures),
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

    best = result.associations.copy()
    if "effect_size" in best.columns:
        best["abs_effect"] = best["effect_size"].abs()
        best = best.sort_values("abs_effect", ascending=False)
    p = out / "features" / "best_features.csv"
    best.to_csv(p, index=False)
    paths["best_features"] = str(p)

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
