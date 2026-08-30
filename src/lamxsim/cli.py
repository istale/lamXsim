"""Command line entry points.

    python -m lamxsim phase0      feasibility: how much failure data is needed
    python -m lamxsim thinslice   end-to-end run on the synthetic validation die
    python -m lamxsim run         end-to-end run on a real layout + failure CSV
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .features.geometry import GeometryExtractor
from .features.grid import build_grid
from .labels.failure import load_failures
from .labels.simulate import failures_from_driver
from .layout.reader import LayerSpec, LayoutReader
from .layout.synth import validation_die
from .registration.apply import load_fiducials, scale_gate
from .registration.fit import robust_fit
from .stats import ablation, power
from .stats.cv import buffered_block_folds, grouped_folds, leakage_report
from . import pipeline


def _load_config(path: str | None) -> dict:
    if path is None:
        return {}
    return yaml.safe_load(Path(path).read_text()) or {}


def cmd_phase0(args) -> int:
    cfg = _load_config(args.config)
    p = cfg.get("phase0", {})
    budget = power.HypothesisBudget(
        n_features=p.get("n_features", 25),
        n_layers=p.get("n_layers", 12),
        n_scales=p.get("n_scales", 6),
    )
    de = power.design_effect_from_moran(
        p.get("expected_moran_i", 0.6), p.get("cells_per_patch", 9))
    table = power.sample_size_table(
        budget, design_effect=de,
        tier1_hypotheses=p.get("tier1_hypotheses", 20),
        control_ratio=p.get("control_ratio", 4.0),
        power=p.get("target_power", 0.80))
    floor = power.registration_scale_floor(p.get("position_sigma_um", 50.0))

    out = Path(args.outdir)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "reports" / "phase0_sample_size.csv", index=False)
    summary = {
        "hypothesis_budget": {
            "per_layer": budget.per_layer_hypotheses,
            "cross_layer": budget.cross_layer_hypotheses,
            "total": budget.total,
        },
        "design_effect": de,
        "registration": floor,
    }
    (out / "reports" / "phase0_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"hypothesis budget: {budget.total} tests "
          f"({budget.per_layer_hypotheses} per-layer + {budget.cross_layer_hypotheses} cross-layer)")
    print(f"design effect (spatial autocorrelation): {de:.2f}")
    print(f"\nrequired measured failure sites (power={p.get('target_power', 0.80):.0%}):")
    pivot = table.pivot(index="target_roc_auc", columns="correction",
                        values="required_failure_sites")
    print(pivot.round(0).to_string())
    print(f"\nregistration sigma {floor['position_sigma_um']}um -> "
          f"trustworthy scales {floor['trustworthy_scales_um']}um, "
          f"rejected {floor['rejected_scales_um']}um")
    print(f"\nwritten to {out / 'reports'}")
    return 0


def cmd_thinslice(args) -> int:
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    gds = out / "gds"
    gds.mkdir(exist_ok=True)
    path = str(gds / "validation_die.gds")
    validation_die(path, die_um=args.die_um, block_um=50.0, seed=7)

    layer = LayerSpec("M8", 8, 0)
    reader = LayoutReader(path)
    grid = build_grid(reader.bbox(), args.driver_scale_um)
    feats = GeometryExtractor(reader).extract(layer, grid)

    driver = args.driver
    fs = failures_from_driver(feats[driver], grid, n_failures=args.n_failures,
                              strength=args.strength, seed=1,
                              position_sigma_um=args.position_sigma_um)
    print(f"validation die {args.die_um:.0f}um, driver={driver} @{args.driver_scale_um:.0f}um, "
          f"{len(fs)} SIMULATED failures, sigma={args.position_sigma_um}um")

    res = pipeline.run(path, fs, layer=layer,
                       scales_um=tuple(args.scales_um),
                       n_permutations=args.n_permutations, seed=3)
    paths = pipeline.write_results(res, out)

    a = res.associations
    cols = ["feature", "scale_um", "roc_auc", "auc_ci_low", "auc_ci_high",
            "effect_size", "fdr_q_value", "effective_n", "n_cells", "scale_trustworthy"]
    print("\n=== association (GDS_GEOMETRY) ===")
    print(a[a.evidence_class == "GDS_GEOMETRY"][cols].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    if len(res.permutations):
        j = a.set_index(["feature", "scale_um"]).join(
            res.permutations.set_index(["feature", "scale_um"])[["p_value", "block_um"]],
            rsuffix="_perm")
        print("\n=== naive p vs spatial block-permutation p ===")
        print(j[["evidence_class", "p_value", "p_value_perm", "block_um"]].to_string(
            float_format=lambda v: f"{v:.4f}"))
    print("\nwritten:")
    for k, v in paths.items():
        print(f"  {k:15s} {v}")
    return 0


def cmd_run(args) -> int:
    cfg = _load_config(args.config)
    lyr = cfg.get("layer", {"name": "M8", "layer": 8, "datatype": 0})
    layer = LayerSpec(lyr["name"], lyr["layer"], lyr.get("datatype", 0))
    fs = load_failures(args.failures)
    if fs.notes:
        print("failure import notes:")
        for n in fs.notes:
            print(f"  - {n}")
    res = pipeline.run(args.gds, fs, layer=layer,
                       scales_um=tuple(cfg.get("scales_um", [25, 50, 100, 250, 500, 1000])),
                       n_permutations=cfg.get("n_permutations", 999))
    paths = pipeline.write_results(res, args.outdir)
    print(json.dumps(paths, indent=2))
    return 0


def cmd_register(args) -> int:
    """Fit a layout-to-measurement registration and report the scale floor."""
    cfg = _load_config(args.config)
    scales = cfg.get("scales_um", [25, 50, 100, 250, 500, 1000])
    src, dst, names = load_fiducials(args.fiducials)
    fit_result, keep, rows = robust_fit(src, dst,
                                        allow_reflection=not args.no_reflection)

    print(f"fiducials: {len(src)} supplied, {int(keep.sum())} kept")
    dropped = [names[i] for i in np.where(~keep)[0]]
    if dropped:
        print(f"  dropped as outliers: {dropped}")
    print("\nmodel comparison (chosen by prediction error, not in-fit residual):")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    gate = scale_gate(fit_result, scales)
    r = gate["registration"]
    print(f"\nselected model     : {r['model']}  ({r['residual_dof']} residual dof)")
    print(f"in-fit RMS         : {r['in_fit_rms_um']} um")
    print(f"leave-one-out RMS  : {r['leave_one_out_rms_um']} um   <- the honest number")
    print(f"transform          : {r['transform']}")
    for w in r["warnings"]:
        print(f"  WARNING: {w}")
    print(f"\nscale floor        : {gate['min_trustworthy_scale_um']:.1f} um")
    print(f"verdict            : {gate['verdict']}")

    out = Path(args.outdir) / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "registration.json").write_text(json.dumps(gate, indent=2, default=str))
    print(f"\nwritten to {out / 'registration.json'}")
    return 0


def cmd_phase6(args) -> int:
    """Multivariate baseline, spatial CV and feature ablation (sections 16-18)."""
    from .features.geometry import GeometryExtractor
    from .features.grid import build_grid
    from .labels.simulate import failures_from_driver, uniform_failures

    out = Path(args.outdir)
    (out / "gds").mkdir(parents=True, exist_ok=True)
    path = str(out / "gds" / "phase6_die.gds")
    validation_die(path, die_um=args.die_um, block_um=50.0, seed=7)

    layer = LayerSpec("M8", 8, 0)
    reader = LayoutReader(path)
    grid = build_grid(reader.bbox(), args.driver_scale_um)
    feats = GeometryExtractor(reader, line_end_w_max_um=args.line_end_w_max_um
                              ).extract(layer, grid)

    if args.null:
        fs = uniform_failures(grid, n_failures=args.n_failures, seed=42,
                              position_sigma_um=5.0)
        print(f"NEGATIVE CONTROL: {len(fs)} spatially uniform failures")
    else:
        fs = failures_from_driver(feats[args.driver], grid,
                                  n_failures=args.n_failures, strength=2.5,
                                  seed=1, position_sigma_um=5.0)
        print(f"driver={args.driver} @{args.driver_scale_um:.0f}um, "
              f"{len(fs)} SIMULATED failures")

    res = pipeline.run(path, fs, layer=layer, scales_um=(args.driver_scale_um,),
                       n_permutations=0, line_end_w_max_um=args.line_end_w_max_um,
                       seed=1)
    y = res.features["failure_present"].to_numpy(int)

    folds = buffered_block_folds(grid, block_um=args.block_um, n_folds=args.n_folds,
                                 buffer_um=args.buffer_um or args.block_um)
    leak = leakage_report(folds, grid, min_separation_um=args.buffer_um or args.block_um)
    print(f"\nspatial CV: {len(folds)} folds, buffered block "
          f"{args.block_um:.0f}um; separation satisfied: {leak['all_pass']}")
    print(f"  mean train {np.mean([f['n_train'] for f in leak['folds']]):.0f} cells, "
          f"{np.mean([f['n_excluded'] for f in leak['folds']]):.0f} withheld as buffer")

    result = ablation.run(res.features, y, folds, seed=3)
    print("\n=== models (out-of-fold, spatially separated) ===")
    print(result.table[["name", "n_features", "roc_auc", "pr_auc", "prevalence",
                        "calibration_slope", "enrichment_top_10pct"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    deltas = pd.DataFrame(result.deltas)
    core = deltas[~deltas.model.str.contains(r"\+position")]
    print("\n=== improvement over the position-only baseline (block bootstrap 95% CI) ===")
    print(core[["model", "model_auc", "baseline_auc", "delta_auc", "ci_low",
                "ci_high", "adds_information"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    (out / "model").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    result.table.to_csv(out / "model" / "ablation_models.csv", index=False)
    deltas.to_csv(out / "model" / "ablation_deltas.csv", index=False)
    (out / "model" / "metrics.json").write_text(json.dumps({
        "null_run": bool(args.null),
        "driver": None if args.null else args.driver,
        "n_failures": len(fs),
        "cv": {"scheme": "buffered_block", "block_um": args.block_um,
               "buffer_um": args.buffer_um or args.block_um,
               "n_folds": len(folds), "leakage": leak},
        "models": result.table.to_dict("records"),
        "deltas": result.deltas,
    }, indent=2, default=str))
    print(f"\nwritten to {out / 'model'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lamxsim")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("phase0", help="feasibility / required sample size")
    p0.add_argument("--config", default="config/thin_slice.yaml")
    p0.add_argument("--outdir", default="results")
    p0.set_defaults(func=cmd_phase0)

    ts = sub.add_parser("thinslice", help="end-to-end run on the validation die")
    ts.add_argument("--outdir", default="results")
    ts.add_argument("--die-um", type=float, default=2000.0)
    ts.add_argument("--driver", default="perimeter_density",
                    choices=["perimeter_density", "metal_density"])
    ts.add_argument("--driver-scale-um", type=float, default=100.0)
    ts.add_argument("--strength", type=float, default=2.5,
                    help="0 gives the negative control (no driver)")
    ts.add_argument("--n-failures", type=int, default=150)
    ts.add_argument("--position-sigma-um", type=float, default=5.0)
    ts.add_argument("--scales-um", type=float, nargs="+",
                    default=[25, 50, 100, 250])
    ts.add_argument("--n-permutations", type=int, default=299)
    ts.set_defaults(func=cmd_thinslice)

    p6 = sub.add_parser("phase6", help="baseline, spatial CV and ablation")
    p6.add_argument("--outdir", default="results")
    p6.add_argument("--die-um", type=float, default=3000.0)
    p6.add_argument("--driver", default="perimeter_density")
    p6.add_argument("--driver-scale-um", type=float, default=100.0)
    p6.add_argument("--n-failures", type=int, default=300)
    p6.add_argument("--line-end-w-max-um", type=float, default=6.0)
    p6.add_argument("--block-um", type=float, default=300.0)
    p6.add_argument("--buffer-um", type=float, default=None)
    p6.add_argument("--n-folds", type=int, default=5)
    p6.add_argument("--null", action="store_true",
                    help="negative control: spatially uniform failures")
    p6.set_defaults(func=cmd_phase6)

    rg = sub.add_parser("register", help="fit registration, report the scale floor")
    rg.add_argument("fiducials", help="CSV with layout_x_um, layout_y_um, "
                                      "measured_x_um, measured_y_um")
    rg.add_argument("--config", default="config/thin_slice.yaml")
    rg.add_argument("--outdir", default="results")
    rg.add_argument("--no-reflection", action="store_true",
                    help="refuse a mirrored fit (frontside imaging)")
    rg.set_defaults(func=cmd_register)

    rn = sub.add_parser("run", help="run on a real layout and failure CSV")
    rn.add_argument("gds")
    rn.add_argument("failures")
    rn.add_argument("--config", default="config/thin_slice.yaml")
    rn.add_argument("--outdir", default="results")
    rn.set_defaults(func=cmd_run)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
