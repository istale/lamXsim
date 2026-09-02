"""Registration, the correlation pipeline, the cost model and the command line.

Consolidated from ``registration/transform.py``, ``registration/fit.py``, ``registration/apply.py``, ``pipeline.py``, ``budget.py``, ``cli.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import re
import resource
import sys
import time
import yaml
from . import exposure as report_mod
from . import foundation as registry_mod
from . import geometry as bump_relative
from . import geometry as crosslayer
from . import geometry as grad_mod
from . import labels as inspection
from . import labels as package_context
from . import labels as position
from . import statistics as ablation
from . import statistics as fdr
from . import statistics as permutation
from . import statistics as power
from . import statistics as univariate
from .foundation import EvidenceClass
from .geometry import GeometryExtractor, LayerStack, OrientationExtractor, StructureExtractor, ViaExtractor, build_grid, build_multiscale
from .labels import FailureSet, failures_from_driver, load_failures, map_to_grid, map_to_grid_per_die
from .layout import BBox, LayerSpec, LayoutReader, validation_die
from .statistics import buffered_block_folds, grouped_folds, leakage_report
from .study import StudyManifest


# ----------------------------------------------------------------------
# registration/transform.py
# ----------------------------------------------------------------------
"""2D coordinate transforms between the layout frame and a measurement frame.

Failure locations arrive in whatever frame the inspection tool produced --
wafer stage coordinates, an image frame, a backside acoustic scan. Bringing
them into layout coordinates is a fitted transform, not a relabelling, and the
quality of that fit decides which analysis scales mean anything.
"""
#: Free parameters per model. Each correspondence supplies two equations, so a
#: fit has 2*n - dof residual degrees of freedom; at zero the residual is zero
#: by construction and says nothing about accuracy.
MODEL_DOF = {"translation": 2, "rigid": 3, "similarity": 4, "affine": 6}


@dataclass(frozen=True)
class Transform2D:
    """Affine map ``dst = A @ src + t``, stored as a 2x3 matrix."""
    matrix: np.ndarray

    @classmethod
    def identity(cls) -> "Transform2D":
        return cls(np.hstack([np.eye(2), np.zeros((2, 1))]))

    @property
    def linear(self) -> np.ndarray:
        return self.matrix[:, :2]

    @property
    def translation(self) -> np.ndarray:
        return self.matrix[:, 2]

    def apply(self, points: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(points, dtype=float))
        return p @ self.linear.T + self.translation

    def inverse(self) -> "Transform2D":
        inv = np.linalg.inv(self.linear)
        return Transform2D(np.hstack([inv, (-inv @ self.translation)[:, None]]))

    # -- interpretable decomposition ---------------------------------
    @property
    def is_reflection(self) -> bool:
        """True if the map flips handedness.

        Backside acoustic imaging mirrors the coordinate frame. A fit that
        silently absorbs the flip into a negative scale still lands the
        fiducials, so the reflection has to be reported rather than inferred
        from a good residual.
        """
        return bool(np.linalg.det(self.linear) < 0)

    @property
    def rotation_deg(self) -> float:
        a = self.linear
        if self.is_reflection:
            a = a @ np.diag([-1.0, 1.0])
        return float(np.degrees(np.arctan2(a[1, 0], a[0, 0])))

    @property
    def scale(self) -> tuple[float, float]:
        a = self.linear
        return (float(np.hypot(a[0, 0], a[1, 0])), float(np.hypot(a[0, 1], a[1, 1])))

    @property
    def shear(self) -> float:
        """Non-orthogonality of the fitted axes, in degrees."""
        a = self.linear
        c0, c1 = a[:, 0], a[:, 1]
        n0, n1 = np.linalg.norm(c0), np.linalg.norm(c1)
        if n0 == 0 or n1 == 0:
            return float("nan")
        cos = float(np.clip(np.dot(c0, c1) / (n0 * n1), -1, 1))
        return float(90.0 - np.degrees(np.arccos(cos)))

    def describe(self) -> dict:
        sx, sy = self.scale
        return {
            "translation_um": [round(float(v), 4) for v in self.translation],
            "rotation_deg": round(self.rotation_deg, 6),
            "scale_x": round(sx, 8), "scale_y": round(sy, 8),
            "shear_deg": round(self.shear, 6),
            "reflection": self.is_reflection,
        }


def _umeyama(src: np.ndarray, dst: np.ndarray, *, with_scale: bool,
             allow_reflection: bool) -> Transform2D:
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    s, d = src - mu_s, dst - mu_d
    cov = d.T @ s / len(src)
    u, sv, vt = np.linalg.svd(cov)
    correction = np.eye(2)
    if not allow_reflection and np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[1, 1] = -1
    r = u @ correction @ vt
    scale = 1.0
    if with_scale:
        var_s = (s ** 2).sum() / len(src)
        scale = float((sv * np.diag(correction)).sum() / var_s) if var_s > 0 else 1.0
    a = scale * r
    return Transform2D(np.hstack([a, (mu_d - a @ mu_s)[:, None]]))


def fit_transform(src: np.ndarray, dst: np.ndarray, model: str = "similarity",
                  *, allow_reflection: bool = True) -> Transform2D:
    """Least-squares fit of *model* mapping ``src`` onto ``dst``."""
    src = np.atleast_2d(np.asarray(src, dtype=float))
    dst = np.atleast_2d(np.asarray(dst, dtype=float))
    if src.shape != dst.shape or src.shape[1] != 2:
        raise ValueError("src and dst must both be (n, 2) and the same length")
    if model not in MODEL_DOF:
        raise ValueError(f"unknown model {model!r}; known: {sorted(MODEL_DOF)}")

    n = len(src)
    minimum = {"translation": 1, "rigid": 2, "similarity": 2, "affine": 3}[model]
    if n < minimum:
        raise ValueError(f"{model} needs at least {minimum} correspondences, got {n}")

    if model == "translation":
        t = dst.mean(axis=0) - src.mean(axis=0)
        return Transform2D(np.hstack([np.eye(2), t[:, None]]))
    if model == "rigid":
        return _umeyama(src, dst, with_scale=False, allow_reflection=allow_reflection)
    if model == "similarity":
        return _umeyama(src, dst, with_scale=True, allow_reflection=allow_reflection)

    design = np.hstack([src, np.ones((n, 1))])
    sol, *_ = np.linalg.lstsq(design, dst, rcond=None)
    return Transform2D(sol.T)

# ----------------------------------------------------------------------
# registration/fit.py
# ----------------------------------------------------------------------
"""Registration quality, and the scale floor it implies.

The residual of a fit is not the accuracy of the mapping. Each correspondence
supplies two equations while the model consumes ``dof`` of them, so a fit with
few fiducials and a rich model reports a small residual because it has enough
freedom to pass close to every point -- not because it will place an
unseen failure correctly. Leave-one-out prediction error is the honest number
and is what the scale floor is derived from.
"""
#: A window smaller than this multiple of the positional uncertainty no longer
#: reliably contains the failure it is credited with.
SCALE_FACTOR = 3.0


@dataclass
class RegistrationFit:
    model: str
    transform: Transform2D
    n_points: int
    residuals_um: np.ndarray          # per-point, in-fit
    rms_um: float
    max_um: float
    loo_residuals_um: np.ndarray      # per-point, leave-one-out prediction
    loo_rms_um: float
    residual_dof: int
    allow_reflection: bool
    warnings: list[str] = field(default_factory=list)

    @property
    def is_determined(self) -> bool:
        """False when the fit has no freedom left to reveal error."""
        return self.residual_dof > 0

    @property
    def position_sigma_um(self) -> float:
        """The uncertainty to attach to every mapped failure location.

        Taken from the leave-one-out error, never from the in-fit residual.
        """
        if np.isfinite(self.loo_rms_um):
            return float(self.loo_rms_um)
        return float("inf")

    def min_trustworthy_scale_um(self, factor: float = SCALE_FACTOR) -> float:
        return factor * self.position_sigma_um

    def usable_scales(self, scales_um, factor: float = SCALE_FACTOR) -> dict:
        floor = self.min_trustworthy_scale_um(factor)
        return {
            "min_trustworthy_scale_um": floor,
            "trustworthy": [s for s in scales_um if s >= floor],
            "rejected": [s for s in scales_um if s < floor],
        }

    def report(self) -> dict:
        return {
            "model": self.model,
            "n_fiducials": self.n_points,
            "residual_dof": self.residual_dof,
            "in_fit_rms_um": round(self.rms_um, 4),
            "in_fit_max_um": round(self.max_um, 4),
            "leave_one_out_rms_um": (round(self.loo_rms_um, 4)
                                     if np.isfinite(self.loo_rms_um) else None),
            "position_sigma_um": (round(self.position_sigma_um, 4)
                                  if np.isfinite(self.position_sigma_um) else None),
            "transform": self.transform.describe(),
            "warnings": self.warnings,
        }


def _min_points(model: str) -> int:
    return {"translation": 1, "rigid": 2, "similarity": 2, "affine": 3}[model]


def fit(src: np.ndarray, dst: np.ndarray, model: str = "similarity", *,
        allow_reflection: bool = True) -> RegistrationFit:
    """Fit and honestly characterise a layout-to-measurement registration.

    ``src`` are fiducial positions in layout coordinates (um), ``dst`` the same
    fiducials as reported by the measurement, in its own frame.
    """
    src = np.atleast_2d(np.asarray(src, dtype=float))
    dst = np.atleast_2d(np.asarray(dst, dtype=float))
    n = len(src)

    t = fit_transform(src, dst, model, allow_reflection=allow_reflection)
    resid = np.linalg.norm(t.apply(src) - dst, axis=1)
    dof = MODEL_DOF[model]
    residual_dof = 2 * n - dof

    warnings: list[str] = []
    if residual_dof <= 0:
        warnings.append(
            f"{n} fiducials against a {dof}-parameter {model} model leaves "
            f"{residual_dof} residual degrees of freedom: the residual is zero "
            "by construction and carries no information about accuracy")
    elif residual_dof < 4:
        warnings.append(
            f"only {residual_dof} residual degrees of freedom; the in-fit RMS "
            "understates the true placement error, use the leave-one-out value")

    # Leave-one-out: refit without each point, then predict it.
    loo = np.full(n, np.nan)
    if n - 1 >= _min_points(model):
        for i in range(n):
            keep = np.arange(n) != i
            try:
                ti = fit_transform(src[keep], dst[keep], model,
                                   allow_reflection=allow_reflection)
            except (ValueError, np.linalg.LinAlgError):
                continue
            loo[i] = float(np.linalg.norm(ti.apply(src[i])[0] - dst[i]))
    else:
        warnings.append(
            f"cannot cross-validate: dropping one point leaves fewer than the "
            f"{_min_points(model)} correspondences {model} needs")

    loo_rms = float(np.sqrt(np.nanmean(loo ** 2))) if np.isfinite(loo).any() else float("nan")

    if t.is_reflection:
        warnings.append(
            "the fitted transform reflects the frame; confirm this is expected "
            "(backside imaging does this) rather than a swapped axis convention")
    sx, sy = t.scale
    if max(abs(sx - 1), abs(sy - 1)) > 0.01:
        warnings.append(
            f"fitted scale differs from unity by more than 1% (sx={sx:.5f}, "
            f"sy={sy:.5f}); check the units of the measurement frame")

    return RegistrationFit(
        model=model, transform=t, n_points=n, residuals_um=resid,
        rms_um=float(np.sqrt((resid ** 2).mean())), max_um=float(resid.max()),
        loo_residuals_um=loo, loo_rms_um=loo_rms, residual_dof=residual_dof,
        allow_reflection=allow_reflection, warnings=warnings)


def select_model(src: np.ndarray, dst: np.ndarray, *,
                 models=("translation", "rigid", "similarity", "affine"),
                 allow_reflection: bool = True) -> tuple[str, list[dict]]:
    """Compare models by prediction error, not by in-fit residual.

    A richer model always fits the fiducials better; it does not always place
    an unseen point better. Choosing on in-fit residual therefore always picks
    ``affine``, and lets real registration error be absorbed into shear and
    scale where it stops being visible.
    """
    rows, best, best_loo = [], None, np.inf
    for m in models:
        try:
            f = fit(src, dst, m, allow_reflection=allow_reflection)
        except ValueError as exc:
            rows.append({"model": m, "error": str(exc)})
            continue
        rows.append({
            "model": m, "residual_dof": f.residual_dof,
            "in_fit_rms_um": f.rms_um, "leave_one_out_rms_um": f.loo_rms_um,
            "n_warnings": len(f.warnings),
        })
        if np.isfinite(f.loo_rms_um) and f.loo_rms_um < best_loo:
            best, best_loo = m, f.loo_rms_um
    return best, rows


def flag_outliers(f: RegistrationFit, k: float = 3.0) -> np.ndarray:
    """Fiducials whose leave-one-out error is an outlier among the rest.

    A single mis-identified fiducial inflates the whole registration; finding
    it is more useful than accepting a degraded sigma across the die.
    """
    loo = f.loo_residuals_um
    ok = np.isfinite(loo)
    if ok.sum() < 4:
        return np.zeros(len(loo), dtype=bool)
    med = np.median(loo[ok])
    mad = np.median(np.abs(loo[ok] - med))
    if mad <= 0:
        return np.zeros(len(loo), dtype=bool)
    out = np.zeros(len(loo), dtype=bool)
    out[ok] = np.abs(loo[ok] - med) / (1.4826 * mad) > k
    return out


def robust_fit(src: np.ndarray, dst: np.ndarray, *,
               allow_reflection: bool = True, k: float = 3.0,
               max_rounds: int = 2) -> tuple[RegistrationFit, np.ndarray, list[dict]]:
    """Select a model, drop outlying fiducials, then re-select on what remains.

    Re-selecting matters: a single mis-identified fiducial dominates the
    prediction error of every model, which flattens the comparison and tends to
    pick the simplest one for the wrong reason. Returns the final fit, a mask
    of the fiducials kept, and the model comparison table from the last round.
    """
    src = np.atleast_2d(np.asarray(src, dtype=float))
    dst = np.atleast_2d(np.asarray(dst, dtype=float))
    keep = np.ones(len(src), dtype=bool)

    model, rows = select_model(src, dst, allow_reflection=allow_reflection)
    for _ in range(max_rounds):
        f = fit(src[keep], dst[keep], model, allow_reflection=allow_reflection)
        flagged = flag_outliers(f, k)
        if not flagged.any():
            break
        idx = np.where(keep)[0][flagged]
        keep[idx] = False
        if keep.sum() < 4:
            keep[idx] = True
            break
        model, rows = select_model(src[keep], dst[keep],
                                   allow_reflection=allow_reflection)

    return fit(src[keep], dst[keep], model, allow_reflection=allow_reflection), keep, rows

# ----------------------------------------------------------------------
# registration/apply.py
# ----------------------------------------------------------------------
"""Bring measured failure locations into the layout coordinate frame.

Registration is the step spec section 10 assumes has already happened when it
says to map failures "into the same physical coordinate system". It is a
fitted transform with its own uncertainty, and that uncertainty propagates:
it becomes the ``position_sigma_um`` that decides which analysis scales are
admissible, so it must travel with the data rather than being asserted in a
config file.
"""
class RegistrationError(RuntimeError):
    pass


def load_fiducials(path, *, layout_x="layout_x_um", layout_y="layout_y_um",
                   measured_x="measured_x_um", measured_y="measured_y_um"):
    """Read a fiducial correspondence table.

    Returns (src, dst, names). Both frames must be present in the same file so
    the pairing is explicit rather than positional.
    """
    df = pd.read_csv(path)
    need = [layout_x, layout_y, measured_x, measured_y]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing fiducial columns {missing}")
    src = df[[layout_x, layout_y]].to_numpy(float)
    dst = df[[measured_x, measured_y]].to_numpy(float)
    names = (df["fiducial_id"].astype(str).tolist()
             if "fiducial_id" in df else [str(i) for i in range(len(df))])
    return src, dst, names


def register(failures: FailureSet, fit: RegistrationFit, *,
             require_determined: bool = True,
             sigma_mode: str = "combine") -> FailureSet:
    """Map a FailureSet into layout coordinates and record the uncertainty.

    The transform runs measurement -> layout, i.e. the inverse of the fit,
    which was estimated layout -> measurement from the fiducials.

    ``sigma_mode``:

    * ``combine`` (default) adds the registration uncertainty in quadrature to
      whatever per-point uncertainty the measurement already reported. Both are
      real and independent; keeping only the larger would understate the total.
    * ``registration`` replaces the reported value.

    A fit with no residual degrees of freedom is refused rather than used: its
    residual is zero by construction, so accepting it would stamp every failure
    with a positional uncertainty of zero and certify every analysis scale.
    """
    if require_determined and not fit.is_determined:
        raise RegistrationError(
            f"registration has {fit.residual_dof} residual degrees of freedom "
            f"({fit.n_points} fiducials against a {fit.model} model). Its "
            "residual is zero by construction and cannot certify any analysis "
            "scale. Add fiducials or choose a simpler model.")
    if not np.isfinite(fit.position_sigma_um):
        raise RegistrationError(
            "registration could not be cross-validated, so no positional "
            "uncertainty is available and no analysis scale can be trusted")

    inverse = fit.transform.inverse()
    table = failures.table.copy()
    mapped = inverse.apply(table[["x_um", "y_um"]].to_numpy(float))
    table["measured_x_um"] = table["x_um"]
    table["measured_y_um"] = table["y_um"]
    table["x_um"] = mapped[:, 0]
    table["y_um"] = mapped[:, 1]

    reg_sigma = fit.position_sigma_um
    if sigma_mode == "registration":
        table["position_sigma_um"] = reg_sigma
    elif sigma_mode == "combine":
        own = table.get("position_sigma_um")
        own = np.zeros(len(table)) if own is None else own.fillna(0.0).to_numpy(float)
        table["position_sigma_um"] = np.hypot(own, reg_sigma)
    else:
        raise ValueError(f"unknown sigma_mode {sigma_mode!r}")

    table["coord_frame"] = "layout"
    notes = list(failures.notes) + [
        f"registered with a {fit.model} transform from {fit.n_points} fiducials; "
        f"leave-one-out RMS {reg_sigma:.2f}um",
    ] + [f"registration warning: {w}" for w in fit.warnings]

    return replace(failures, table=table, notes=notes)


def scale_gate(fit: RegistrationFit, scales_um, factor: float = SCALE_FACTOR) -> dict:
    """Which configured scales the registration accuracy can support."""
    report = fit.usable_scales(scales_um, factor)
    report["registration"] = fit.report()
    if not report["trustworthy"]:
        report["verdict"] = (
            "no configured scale survives; either improve the registration or "
            "add coarser scales before running any association analysis")
    else:
        report["verdict"] = (
            f"analysis limited to {report['trustworthy']}um; "
            f"{report['rejected']}um rejected")
    return report

# ----------------------------------------------------------------------
# pipeline.py
# ----------------------------------------------------------------------
"""Layout -> features -> case/control -> association.

Runs one or more layers through geometry, orientation, gradient and
cross-layer extraction at every configured scale, then scores every
feature x layer x scale combination against the failure labels.

The statistical machinery was validated before the feature catalogue was
widened, so adding a feature family here is a mechanical extension of
something already known to report nothing when there is nothing to report.
"""
def _die_level_covariate(failures, name: str, die_names) -> "np.ndarray | None":
    """One numeric value per die for a declared condition, or None.

    A condition varying within a die cannot be a die-level covariate; the
    first value is taken and the discrepancy is reported by
    SampleConditions.check_against rather than silently averaged.
    """
    table = failures.table
    if name not in table:
        return None
    keys = failures.die_keys()
    per_die = {}
    for key, group in table.groupby(keys):
        value = pd.to_numeric(group[name], errors="coerce").dropna()
        per_die[str(key)] = float(value.iloc[0]) if len(value) else np.nan
    values = np.array([per_die.get(d, np.nan) for d in die_names], dtype=float)
    if not np.isfinite(values).any() or np.nanstd(values) == 0:
        # A covariate that does not vary across the dies analysed explains
        # nothing and would only add a constant column to the baseline.
        return None
    return values


def _file_digest(path: str) -> str:
    """SHA-256 of the layout, so a result names the file that produced it."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fmt(b) -> str:
    return f"[{b.xmin:g}, {b.ymin:g}] to [{b.xmax:g}, {b.ymax:g}]um"


def _covers(outer, inner, tol: float = 1e-6) -> bool:
    return (outer.xmin <= inner.xmin + tol and outer.ymin <= inner.ymin + tol
            and outer.xmax >= inner.xmax - tol and outer.ymax >= inner.ymax - tol)


def _is_roi(die, geometry, tol: float = 1e-6) -> bool:
    return (die.width > geometry.width + tol or die.height > geometry.height + tol)

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
    ("orientation_coherence", "tier1"),
    ("routing_direction_rad", "exploratory"),
    ("routing_vs_radial_angle", "tier1"),
    ("routing_radial_alignment", "tier1"),
    ("routing_diagonality", "tier1"),
    ("density_difference", "tier1"),
    ("perimeter_density_difference", "tier1"),
    ("orientation_difference", "tier1"),
    ("line_end_density_difference", "tier1"),
    ("density_mismatch", "tier1"),
    ("perimeter_density_mismatch", "tier1"),
    ("orientation_mismatch", "tier1"),
    ("line_end_density_mismatch", "tier1"),
    ("wide_metal_fraction", "tier1"),
    ("wide_metal_perimeter_density", "tier1"),
    ("slot_density", "tier1"),
    ("slotted_metal_fraction", "tier1"),
    ("fill_density", "exploratory"),
    ("fill_fraction", "exploratory"),
    ("cross_layer_transition_index", "tier1"),
    ("top_to_underlying", "tier1"),
    ("stacked_dense_layer_count", "exploratory"),
    ("stacked_sparse_layer_count", "exploratory"),
    ("density_variance_across_layers", "exploratory"),
    ("distance_to_", "tier1_confounder"),
    ("normalized_distance_", "tier1_confounder"),
    ("condition_", "tier1_confounder"),
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


def _extract_layer(reader, geo_ex, ori_ex, via_ex, struct_ex, layer, via_layer,
                   grid, *, with_gradients=True):
    vals = dict(geo_ex.extract(layer, grid))
    vals.update(ori_ex.extract(layer, grid))
    vals.update(struct_ex.extract(layer, grid))
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
        footprints: "inspection.FootprintSet | None" = None,
        layout_revision: str | None = None,
        die_covariates: "tuple[str, ...]" = (),
        min_coverage: float = 0.5,
        allow_pooling_modes: bool = False,
        allow_failures_outside_footprint: bool = False,
        die_bbox: "BBox | None" = None,
        top_cell: str | None = None,
        line_end_w_max_um: float | None = None,
        line_rules: "dict[str, tuple[float, float]] | None" = None,
        fill_layers: "dict[str, LayerSpec] | None" = None,
        wide_width_um: float = 3.0,
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
    struct_ex = StructureExtractor(reader, wide_width_um=wide_width_um,
                                   fill_layers=fill_layers)
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
    if footprints is None:
        footprints = inspection.FootprintSet(default=footprint)
    elif footprint is not None:
        raise ValueError("pass footprint= or footprints=, not both")

    if footprints.default is None and footprints.is_uniform:
        footprints = inspection.FootprintSet(
            default=inspection.InspectionFootprint.full_die(
                geometry_bbox, "no inspection footprint supplied",
                dbu=reader.units.dbu))
    if (footprints.default is not None
            and footprints.default.assumed_full_coverage
            and footprints.default.justification ==
            "no inspection footprint supplied"):
        context_notes.append(
            "no inspection footprint supplied: the whole die is being treated "
            "as inspected, so every cell without a recorded failure counts as "
            "a control. If inspection was partial or targeted, features "
            "correlated with where it was targeted will show spurious "
            "association.")
    context_notes.extend(failures.assert_single_mode(
        allow_pooling=allow_pooling_modes))
    context_notes.extend(failures.assert_single_layout_revision(layout_revision))
    n_dies = failures.n_dies()
    if n_dies == 1:
        context_notes.append(
            "a single die: spec section 17 asks for held-out dies, so nothing "
            "here can be shown to generalise. Treat the result as a local "
            "diagnostic of this piece of silicon.")

    audit = inspection.audit_failures_per_die(footprints, failures,
                                              dbu=reader.units.dbu)
    if audit["dies_without_a_footprint"]:
        raise ValueError(
            f"no inspected footprint for die(s) "
            f"{audit['dies_without_a_footprint']}. A die with no declared "
            "footprint has no control population: every cell on it would be "
            "counted as inspected and clean. Declare one, or supply a default.")
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

        # Eligibility is per (cell, die), because the footprint is: a cell
        # inspected on one die and not on another is a control on the first
        # and missing data on the second.
        per_die_cover = {}
        for name in die_names:
            fp = footprints.for_die(name)
            ok, frac = inspection.eligibility(
                fp, grid, min_coverage=min_coverage, dbu=reader.units.dbu)
            per_die_cover[name] = (ok, frac)
        eligible = np.concatenate([per_die_cover[n][0] for n in die_names])
        cover = np.concatenate([per_die_cover[n][1] for n in die_names])
        # The cell-level mask used for lattice statistics takes a cell as
        # usable when any die inspected it.
        eligible_cell = np.any(
            np.vstack([per_die_cover[n][0] for n in die_names]), axis=0)
        coverage_summary[scale] = {
            "n_cells": len(grid), "n_dies": len(die_names),
            "n_observations": int(len(y)),
            "n_eligible": int(eligible.sum()),
            "n_cases_eligible": int(y[eligible].sum()),
            "n_cases_excluded": int(y[~eligible].sum()),
            "prevalence": float(y[eligible].mean()) if eligible.any() else float("nan"),
            "mean_coverage": float(cover.mean()),
            "uniform_footprint": footprints.is_uniform,
        }

        cell_arrays = grid.to_arrays()
        frame = pd.DataFrame({k: v[cell_index] for k, v in cell_arrays.items()})
        frame["die_key"] = np.array(die_names)[die_index]
        frame["inspected_fraction"] = cover
        frame["eligible"] = eligible
        frame["failure_present"] = y
        frame["distance_to_nearest_failure"] = nearest

        columns: list[tuple[str, str, np.ndarray, EvidenceClass]] = []
        per_layer_base = {}

        for spec in specs:
            vals, base = _extract_layer(reader, geo_ex, ori_ex, via_ex,
                                        struct_ex, spec,
                                        via_layers.get(spec.name), grid,
                                        with_gradients=with_gradients)
            per_layer_base[spec.name] = base
            for name, v in vals.items():
                columns.append((name, spec.name, v, EvidenceClass.GDS_GEOMETRY))

        if len(specs) > 1:
            for name, v in crosslayer.crosslayer_extract(per_layer_base, stack,
                                              selection=pair_selection).items():
                columns.append((name, "CROSS", v, EvidenceClass.GDS_GEOMETRY))

        if include_position:
            for name, v in position.position_extract(grid, bbox).items():
                columns.append((name, "-", v, EvidenceClass.PACKAGE_POSITION))
            # Package and process conditions declared as covariates enter the
            # baseline the geometry model has to beat. Recording them in the
            # manifest and then leaving them out of the model would let a
            # geometry feature absorb their effect, which is the thing the
            # declaration exists to prevent.
            for cov in die_covariates:
                values = _die_level_covariate(failures, cov, die_names)
                if values is None:
                    continue
                columns.append((f"condition_{cov}", "-", values[die_index],
                                EvidenceClass.SAMPLE_CONDITION))

            if package_layers.any_present:
                ctx = package_context.package_context_extract(grid, bbox, reader, package_layers)
                for name, v in ctx.items():
                    columns.append((name, "-", v, EvidenceClass.PACKAGE_POSITION))

                # Routing resolved against the package loading direction is a
                # layout property -- the thing a designer changes -- even
                # though it takes a bump map to compute. It is scored as
                # geometry, against a baseline that already holds the bump
                # distances it would otherwise be confounded with.
                radial = ctx.get("bump_radial_direction_rad")
                if radial is not None and np.isfinite(radial).any():
                    for spec in specs:
                        base = per_layer_base[spec.name]
                        rel = bump_relative.bump_relative_extract(
                            base["routing_direction_rad"],
                            base["orientation_coherence"], radial)
                        for name, v in rel.items():
                            columns.append((name, spec.name, v,
                                            EvidenceClass.GDS_GEOMETRY))

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

        # Columns are collected and joined once. Assigning them one at a
        # time re-allocates the frame on every insert, which on a real run is
        # both slow and drowns the output in fragmentation warnings.
        feature_columns: dict[str, np.ndarray] = {}
        for name, layer_name, cell_vals, ecls in columns:
            vals = cell_vals[cell_index]
            feature_columns[f"{name}|{layer_name}"] = vals
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
                # The spatial p-value goes onto the association itself, not
                # only into a side table. A primary claim is corrected from
                # it; leaving it in a separate file let the naive test decide
                # what counted as significant.
                a.spatial_p_value = pr.p_value
                p = pr.as_row()
                p.update(feature=name, layer=layer_name, scale_um=scale)
                perm_rows.append(p)

        frame = pd.concat(
            [frame, pd.DataFrame(feature_columns, index=frame.index)], axis=1)
        feat_frames.append(frame)

    if not assoc_rows:
        raise ValueError(
            "no feature x layer x scale combination could be scored. Every "
            "candidate had too few finite values or no failures in the grid; "
            "check that the failure coordinates lie inside the die bounding "
            f"box {[bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]} and are in "
            "layout coordinates (registration/apply.register does this).")

    fdr.apply_tiered([a for a, _ in assoc_rows])

    n_corrected = sum(1 for a, _ in assoc_rows
                      if a.hypothesis_tier.startswith("tier1"))
    budget = power.permutation_budget(n_corrected, n_permutations)
    floor_p = budget["min_achievable_p"]

    rows = []
    for a, row in assoc_rows:
        row["fdr_q_value"] = a.fdr_q_value
        row["spatial_p_value"] = a.spatial_p_value
        row["spatial_q_value"] = a.spatial_q_value
        # A p pinned at 1/(n+1) is a bound the permutation count imposed, not
        # a value the data produced. Marked so a reader does not read it as a
        # resolved result.
        row["spatial_p_at_floor"] = bool(
            np.isfinite(a.spatial_p_value)
            and a.spatial_p_value <= floor_p + 1e-12)
        rows.append(row)
    at_floor = sum(1 for a, _ in assoc_rows
                   if np.isfinite(a.spatial_p_value)
                   and a.spatial_p_value <= floor_p + 1e-12)
    budget["n_at_resolution_floor"] = at_floor
    if n_permutations and not budget["sufficient"]:
        context_notes.append(
            f"{n_permutations} permutations cannot resolve a family of "
            f"{n_corrected} corrected tests: the smallest p a permutation can "
            f"return is {floor_p:.5f}, so a single result among nulls could "
            f"reach no better than q = "
            f"{budget['best_achievable_q_for_a_lone_result']:.3f}. "
            f"{at_floor} test(s) are sitting at that floor, and their q values "
            "come from being tied there rather than from being resolved -- "
            "they are an upper bound on significance, not a measurement of it. "
            f"Use at least {budget['permutations_needed_for_alpha']} "
            "permutations, or reduce the hypothesis family.")

    if not n_permutations:
        context_notes.append(
            "no spatial permutation was run (n_permutations=0), so no result "
            "can be primary evidence. The Mann-Whitney q-values in the table "
            "assume grid cells are independent observations, which on spatial "
            "data produces false positives in quantity.")

    meta = {
        "gds_path": str(gds_path),
        "layers": [str(s) for s in specs],
        "via_layers": {k: str(v) for k, v in via_layers.items()},
        "package_layers": {k: (str(v) if v else None) for k, v in
                           vars(package_layers).items()},
        "uncontrolled_confounding": context_notes,
        "inspection_footprint": footprints.report(reader.units.dbu),
        "layout_revision": layout_revision,
        "min_coverage": min_coverage,
        "coverage_by_scale": coverage_summary,
        "failure_footprint_audit": audit,
        "pair_selection": pair_selection if len(specs) > 1 else None,
        "with_gradients": with_gradients,
        "line_rules": line_rules or {},
        "fill_layers": {k: str(v) for k, v in (fill_layers or {}).items()},
        "wide_width_um": wide_width_um,
        "die_bbox_um": [die_bbox.xmin, die_bbox.ymin, die_bbox.xmax, die_bbox.ymax],
        "geometry_bbox_um": [geometry_bbox.xmin, geometry_bbox.ymin,
                             geometry_bbox.xmax, geometry_bbox.ymax],
        "die_outline_declared": frame_note == "" or "region of interest" in frame_note,
        "top_cell": reader.top.name,
        "scales_um": sorted(grids),
        "n_failures": len(failures),
        "n_dies": n_dies,
        "failure_modes": failures.modes(),
        "failure_layout_revisions": failures.layout_revisions(),
        "gds_sha256": _file_digest(gds_path),
        "feature_registry": registry_mod.audit(
            [r["feature"] for r in rows]),
        "failures_simulated": failures.simulated,
        "failure_source": failures.source,
        "failure_notes": failures.notes,
        "position_sigma_um": failures.position_sigma_um,
        "min_trustworthy_scale_um": scale_floor,
        "n_hypotheses_tested": len(rows),
        "n_permutations": n_permutations,
        "permutation_budget": budget,
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
    paths.update(report_mod.write_reports(result.associations, out,
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


@dataclass
class StratifiedResult:
    """One run per failure population, plus how far they agree."""
    per_stratum: dict[str, RunResult]
    consistency: pd.DataFrame
    strata_by: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.per_stratum)


def run_stratified(gds_path: str, failures: FailureSet, *,
                   stratify_by=("failed_interface",),
                   min_failures: int = 20, **kwargs) -> StratifiedResult:
    """Analyse each failure population separately and compare them.

    Pooling modes asks whether a feature associates with failure in general.
    Splitting them asks whether it associates with each mechanism, and whether
    it does so in the same direction -- which is what separates a mechanism
    from a proxy for one. A feature that reverses sign between interfaces is
    not a weaker version of one that does not; it is a different finding.

    Strata with fewer than ``min_failures`` are skipped rather than analysed
    into noise, and named in the result so the omission is visible.
    """
    from .labels import stratify as _stratify

    groups = _stratify(failures, stratify_by)
    results, skipped = {}, {}
    for name, subset in groups.items():
        if len(subset) < min_failures:
            skipped[name] = len(subset)
            continue
        results[name] = run(gds_path, subset, **kwargs)

    for r in results.values():
        r.metadata["strata_skipped_for_size"] = skipped
        r.metadata["strata_analysed"] = sorted(results)

    return StratifiedResult(per_stratum=results,
                            consistency=_consistency(results),
                            strata_by=tuple(stratify_by))


def _consistency(results: dict[str, RunResult]) -> pd.DataFrame:
    """Per feature, how the strata compare.

    ``signs_agree`` is the question Gate 4 asks of dies, applied to
    mechanisms: an effect that points one way on one interface and the other
    way on another is not a single effect measured twice.
    """
    if len(results) < 2:
        return pd.DataFrame()

    frames = []
    for name, res in results.items():
        if res.associations.empty:
            continue
        keep = res.associations[["feature", "layer", "scale_um", "effect_size",
                                 "roc_auc", "spatial_q_value", "fdr_q_value",
                                 "n_case"]].copy()
        keep["stratum"] = name
        frames.append(keep)
    if len(frames) < 2:
        return pd.DataFrame()

    joined = pd.concat(frames, ignore_index=True)
    rows = []
    for (feature, layer, scale), group in joined.groupby(
            ["feature", "layer", "scale_um"]):
        if len(group) < 2:
            continue
        effects = group["effect_size"].to_numpy(float)
        finite = effects[np.isfinite(effects)]
        if len(finite) < 2:
            continue
        signs = np.sign(finite)
        rows.append({
            "feature": feature, "layer": layer, "scale_um": scale,
            "n_strata": len(finite),
            "effect_min": float(finite.min()), "effect_max": float(finite.max()),
            "effect_spread": float(finite.max() - finite.min()),
            "signs_agree": bool(np.all(signs == signs[0])),
            # The spatial q, for the same reason the primary table uses it:
            # a cross-stratum agreement resting on a test that assumed
            # independent cells is weaker than the single-stratum results it
            # is summarising.
            "min_spatial_q": float(group["spatial_q_value"].min()),
            "min_naive_q": float(group["fdr_q_value"].min()),
            "strata": "; ".join(f"{s}={e:+.3f}" for s, e in
                                zip(group["stratum"], group["effect_size"])),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Disagreement first: a feature pointing opposite ways on two mechanisms
    # is the thing a reader most needs to see.
    return out.sort_values(["signs_agree", "effect_spread"],
                           ascending=[True, False])

# ----------------------------------------------------------------------
# budget.py
# ----------------------------------------------------------------------
"""Measure the cost of extraction on a clip, and project it to a full chip.

The runtime question cannot be answered from this repository's synthetic
dies. Measured here, the atlas costs about 79 us and 2 kB of peak memory per
polygon per scale -- flat across a sevenfold range of polygon count -- but
those constants come from Manhattan geometry with no hierarchy on one machine.
A production layout differs in polygon density, hierarchy depth, the fraction
of non-Manhattan geometry and the machine it runs on, and the projection is
linear in a constant nobody has measured for it.

So this measures the constant on the user's own layout and projects with it,
and reports what the projection is sensitive to. It answers "can this run at
all", which for a full chip is a memory question rather than a time one: at
2 kB per polygon a hundred million polygons is 200 GB, and there is no tiling
in the Python path to reduce it. Time is the easy constraint.
"""
def _peak_rss_bytes() -> int:
    """Peak resident set size, in bytes.

    ru_maxrss is bytes on macOS and kilobytes on Linux, which is a difference
    of 1024 in the headline number of this whole command. It is decided by
    the platform rather than sniffed from the value, because a small process
    on Linux and a large one on macOS produce the same figure.
    """
    import sys

    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw if sys.platform == "darwin" else raw * 1024


@dataclass
class Measurement:
    """What one extraction actually cost, and what it implies."""
    polygons: int
    cells: int
    scales: int
    seconds: float
    peak_rss_bytes: int
    baseline_rss_bytes: int

    @property
    def seconds_per_polygon_scale(self) -> float:
        denom = max(self.polygons * self.scales, 1)
        return self.seconds / denom

    @property
    def bytes_per_polygon(self) -> float:
        used = max(self.peak_rss_bytes - self.baseline_rss_bytes, 0)
        return used / max(self.polygons, 1)

    def project(self, polygons: int, scales: int,
                exponent: float = 1.0) -> dict:
        """Extrapolate to a layout of *polygons* at *scales* scales.

        ``exponent`` is how the *time* grows with polygon count. It is not 1.
        The windowed extractors clip the layer once per grid row and then once
        per window against that row, so the work is rows times polygons; on a
        layout both grow with die area, and measured across a sixty-fourfold
        range the cost rose 4.8x, then 5.3x, then 5.9x for each fourfold rise
        in polygons -- a local exponent climbing from 1.14 to 1.28. Projecting
        linearly from a small clip understates a full chip by more than an
        order of magnitude, which is the difference between an overnight job
        and a fortnight.

        Memory stays linear: it is the merged layers held at once.
        """
        ratio = polygons / max(self.polygons, 1)
        seconds = self.seconds * (ratio ** exponent) * (scales / max(self.scales, 1))
        peak = self.baseline_rss_bytes + self.bytes_per_polygon * polygons
        return {"polygons": polygons, "scales": scales, "exponent": exponent,
                "seconds": seconds, "hours": seconds / 3600.0,
                "peak_rss_gb": peak / 1e9}


def fit_exponent(measurements: "list[Measurement]") -> "tuple[float, str]":
    """How time grows with polygon count, from two or more clips.

    A log-log slope through the measurements. With one clip there is nothing
    to fit and the caller has to say so rather than assume 1: the assumption
    is wrong in the direction that matters, and it is wrong by a factor that
    grows with how far the projection reaches.
    """
    import numpy as np

    usable = [m for m in measurements if m.polygons > 0 and m.seconds > 0]
    if len(usable) < 2:
        return 1.0, ("only one clip was measured, so the growth of time with "
                     "polygon count could not be fitted and 1.0 was assumed. "
                     "It is not 1.0 -- the windowed extractors cost rows times "
                     "polygons, and both grow with die area -- so the time "
                     "below is a lower bound, and one that gets weaker the "
                     "further the projection reaches. Measure a second, larger "
                     "clip to fit it")
    x = np.log([m.polygons for m in usable])
    y = np.log([m.seconds / max(m.scales, 1) for m in usable])
    slope = float(np.polyfit(x, y, 1)[0])
    spread = max(m.polygons for m in usable) / min(m.polygons for m in usable)
    return slope, (f"fitted over {len(usable)} clips spanning {spread:.0f}x in "
                   "polygon count")


def count_polygons(gds_path: str, manifest) -> tuple[int, dict]:
    """Merged polygons on every layer the manifest analyses.

    Merged, because that is what the extractors see: a layer drawn as ten
    thousand abutting rectangles is one polygon to them, and counting the
    drawn shapes instead would overstate the cost by whatever the merge
    removes.
    """
    reader = LayoutReader(gds_path, top_cell=manifest.top_cell)
    per_layer = {}
    for spec in manifest.metal_layers:
        per_layer[spec.name] = reader.region(spec).count()
    for name, spec in manifest.via_layers.items():
        per_layer[spec.name] = reader.region(spec).count()
    for kind, spec in vars(manifest.package_layers).items():
        if spec is not None:
            per_layer[spec.name] = reader.region(spec).count()
    return sum(per_layer.values()), per_layer


def measure(gds_path: str, manifest) -> Measurement:
    """Run the atlas once and record what it cost."""
    from . import exposure as atlas_mod
    from .geometry import build_multiscale

    baseline = _peak_rss_bytes()
    polygons, _ = count_polygons(gds_path, manifest)
    reader = LayoutReader(gds_path, top_cell=manifest.top_cell)
    grids = build_multiscale(reader.bbox(), manifest.scales_um)
    cells = sum(len(g) for g in grids.values())

    start = time.time()
    atlas_mod.build(gds_path, manifest)
    elapsed = time.time() - start

    return Measurement(polygons=polygons, cells=cells, scales=len(grids),
                       seconds=elapsed, peak_rss_bytes=_peak_rss_bytes(),
                       baseline_rss_bytes=baseline)

# ----------------------------------------------------------------------
# cli.py
# ----------------------------------------------------------------------
"""Command line entry points.

python -m lamxsim phase0      feasibility: how much failure data is needed
python -m lamxsim thinslice   end-to-end run on the synthetic validation die
python -m lamxsim run         end-to-end run on a real layout + failure CSV
"""
def _load_config(path: str | None) -> dict:
    if path is None:
        return {}
    return yaml.safe_load(Path(path).read_text()) or {}


def _tier1_family_size(manifest_path) -> int:
    """Tier-1 rows the FDR will actually correct, from the manifest.

    Registered tier-1 families, times the layers and scales the study
    declares. Twenty was the count of families, not of tests, and the
    difference decides both the required sample size and whether the
    permutation count can resolve the correction at all.
    """
    from . import foundation as registry

    families = sum(1 for e in registry.load().values()
                   if e.row.get("hypothesis_tier", "").startswith("tier1"))
    try:
        m = StudyManifest.load(manifest_path)
    except Exception:
        return max(families, 1)
    return max(families * max(len(m.metal_layers), 1) * max(len(m.scales_um), 1), 1)


def cmd_phase0(args) -> int:
    cfg = _load_config(args.config)
    p = cfg.get("phase0", {})
    # Derive the hypothesis family from the manifest where one is available,
    # rather than from a default that has no relation to the run. The FDR
    # family is feature x layer x scale, which is hundreds of rows, not the
    # twenty the old default assumed.
    manifest_tier1 = _tier1_family_size(args.config)
    budget = power.HypothesisBudget(
        n_features=p.get("n_features", 25),
        n_layers=p.get("n_layers", 12),
        n_scales=p.get("n_scales", 6),
    )
    de = power.design_effect_from_moran(
        p.get("expected_moran_i", 0.6), p.get("cells_per_patch", 9))
    table = power.sample_size_table(
        budget, design_effect=de,
        tier1_hypotheses=p.get("tier1_hypotheses", manifest_tier1),
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

    res = run(path, fs, layer=layer,
                       scales_um=tuple(args.scales_um),
                       n_permutations=args.n_permutations, seed=3)
    paths = write_results(res, out)

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


def cmd_characterize(args) -> int:
    """GDS + manifest -> literature exposure atlas. No failure data involved."""
    from . import exposure as atlas_mod

    manifest = StudyManifest.load(args.manifest)
    manifest.sample_conditions.validate()
    result = atlas_mod.build(args.gds, manifest,
                             candidate_percentile=args.candidate_percentile,
                             calibre_dir=args.features_from)
    meta = result.metadata

    print(f"layout   : {args.gds}")
    print(f"  top cell : {meta['top_cell']}")
    print(f"  geometry : {meta['geometry_bbox_um']} um")
    print(f"  die      : {meta['die_bbox_um']} um"
          f"   ({'declared' if meta['die_outline_declared'] else 'ASSUMED from geometry'})")
    print(f"  scales   : {meta['scales_um']} um")
    print(f"  features : {sum(1 for c in result.features.columns if '|' in c)} maps")
    cal = meta.get("calibre")
    if cal:
        taken = sorted({f for byscale in cal["features_taken"].values()
                        for fs in byscale.values() for f in fs})
        print(f"  extraction: {cal['generator']}"
              f"{'  (EMULATED, not run through Calibre)' if cal['emulated'] else ''}")
        print(f"    from deck  : {', '.join(taken) if taken else 'nothing matched'}")
        print(f"    eps guard  : {', '.join(f'{k} {v} violation(s)' for k, v in sorted(cal['eps_guard_violations'].items())) or 'no metal layer'}")
        print("    from Python: everything else -- orientation, gradients, "
              "cross-layer terms, position and package context. The deck is "
              "required to be complete for the layers and scales the manifest "
              "asks for, so nothing else fell back silently.")

    if manifest.gaps:
        print("\ndeclared gaps in the manifest:")
        for g in manifest.gaps:
            print(f"  - {g}")

    notes = sorted({f"{r.channel.channel_id}: {r.reason}"
                    for cs in result.channels.values()
                    for _, r in cs if r.reason})
    if notes:
        print("\nchannels that could not be scored, or scored nothing:")
        for u in notes:
            print(f"  - {u}")

    print(f"\n=== literature exposure, at or above the "
          f"{args.candidate_percentile:g}th percentile per channel ===")
    print("  (ties share a rank, so a channel whose input takes few distinct "
          "values can report nothing and say so rather than choose among "
          "identical cells)")
    if result.candidates.empty:
        print("  no candidate regions")
    else:
        summary = (result.candidates.groupby(["channel", "layer", "scale_um"])
                   .size().rename("n_candidates").reset_index())
        print(summary.to_string(index=False))
        print("\nchannels are reported separately and never summed: a location "
              "flagged on three is three records with three citations, not a "
              "score of three.")

    paths = atlas_mod.write(result, args.outdir, manifest)
    print("\nwritten:")
    for k, v in paths.items():
        print(f"  {k:26s} {v}")
    print("\nRead assumptions_and_limits.md first. Nothing here is a "
          "probability: with no measured failure there is no scale on which "
          "one could be defined.")
    return 0


def cmd_budget(args) -> int:
    """Measure extraction cost on one or more clips and project it."""
    pass

    manifest = StudyManifest.load(args.manifest)
    measurements = []
    for path in args.gds:
        polygons, per_layer = count_polygons(path, manifest)
        print(f"clip     : {path}")
        for name, count in sorted(per_layer.items(), key=lambda kv: -kv[1]):
            print(f"  {name:8s} {count:12,d} merged polygon(s)")
        print(f"  {'total':8s} {polygons:12,d}")
        if polygons == 0:
            print("\nNo geometry on any layer the manifest analyses. Check "
                  "the layer numbers before measuring anything.")
            return 1
        m = measure(path, manifest)
        measurements.append(m)
        print(f"  measured {m.seconds:.1f}s over {m.scales} scale(s), "
              f"{m.cells:,d} window(s), peak RSS "
              f"{m.peak_rss_bytes / 1e9:.2f} GB")
        print(f"           {m.seconds_per_polygon_scale * 1e6:.1f} us and "
              f"{m.bytes_per_polygon / 1e3:.2f} kB per polygon per scale\n")

    exponent, how = fit_exponent(measurements)
    print(f"time grows as polygons^{exponent:.2f}  ({how})")

    if not args.full_chip_polygons:
        print("\nPass --full-chip-polygons to project. Count them the same "
              "way this does -- merged, on the layers the manifest analyses.")
        return 0

    biggest = max(measurements, key=lambda m: m.polygons)
    p = biggest.project(args.full_chip_polygons, len(manifest.scales_um),
                        exponent)
    reach = args.full_chip_polygons / biggest.polygons
    print(f"\nprojected for {p['polygons']:,d} polygon(s) at "
          f"{p['scales']} scale(s), reaching {reach:.0f}x beyond the largest "
          "clip measured:")
    print(f"  time      {p['hours']:.1f} hour(s)"
          f"   ({p['hours'] / 24:.1f} day(s))")
    print(f"  peak RSS  {p['peak_rss_gb']:.0f} GB")

    span = (max(m.polygons for m in measurements)
            / min(m.polygons for m in measurements))
    if reach > 10 * span:
        print(f"\n  The exponent was fitted over {span:.0f}x and is being "
              f"used over {reach:.0f}x. It is not a constant: measured on "
              "synthetic dies the local exponent climbed from 1.14 to 1.28 as "
              "the die grew, so a fit from small clips understates a full "
              "chip, and the time above is optimistic. Add a larger clip "
              "before trusting it against a tight budget.")

    fits_ram = p["peak_rss_gb"] <= args.available_ram_gb / 2
    fits_time = p["hours"] <= args.max_hours
    if fits_ram and fits_time:
        print(f"\nThis fits: {p['peak_rss_gb']:.0f} GB against "
              f"{args.available_ram_gb:g} GB, {p['hours']:.1f}h against a "
              f"{args.max_hours:g}h budget. Give the machine to the job.")
        return 0

    print("")
    if not fits_ram:
        print(f"Memory does not fit: {p['peak_rss_gb']:.0f} GB projected "
              f"against {args.available_ram_gb:g} GB. The Python path holds "
              "every analysed layer merged at once and has no tiling.")
    if not fits_time:
        print(f"Time does not fit: {p['hours']:.1f} hour(s) against a "
              f"{args.max_hours:g} hour budget. Time grows faster than the "
              "polygon count because the windowed extractors clip the layer "
              "once per grid row and again per window, so the work is rows "
              "times polygons and both grow with die area. Fewer scales and "
              "fewer layers reduce it proportionally; a bigger machine does "
              "not.")
    print("\nGenerate the Calibre deck instead -- the moving window is native "
          "there, and Python then reads one value per window rather than "
          "scanning the layer:")
    print(f"  lamxsim deck {args.gds[0]} --manifest {args.manifest} "
          "--outdir deck")
    print("Run it, diff it against --emulate, then "
          "`characterize --features-from deck`.")
    return 0


def cmd_deck(args) -> int:
    """Generate the Calibre rule deck for a study manifest."""
    from . import calibre as svrf

    manifest = StudyManifest.load(args.manifest)
    layers = svrf.layers_from_manifest(manifest)
    # The deck is bound to the layout it was generated for. Without that, a
    # complete set of RDBs from another revision of the same design passes
    # every check and is then mixed with maps computed from this one.
    binding = svrf.binding_for(args.gds, manifest, args.manifest)
    paths = svrf.write_deck(args.outdir, layers, scales_um=manifest.scales_um,
                            step_ratio=args.step_ratio, binding=binding)

    print(f"layout : {args.gds}")
    print(f"  top cell : {binding['top_cell']}")
    print(f"  sha256   : {binding['gds_sha256'][:16]}...")
    print(f"  bbox     : {binding['geometry_bbox_um']} um")
    print(f"layers : {', '.join(str(l.name) for l in layers)}")
    print(f"scales : {list(manifest.scales_um)} um   STEP = WINDOW x "
          f"{args.step_ratio:g}")
    for row in svrf.eps_report(layers):
        print(f"  {row['layer']:6s} min width {row['min_width_um']:g}um -> "
              f"eps {row['eps_um']:g}um  ({row['margin_x']:g}x margin before "
              f"the band collapses at {row['cliff_at_um']:g}um)")
    print("\nwritten:")
    for k, v in paths.items():
        print(f"  {k:22s} {v}")

    if args.emulate:
        from . import calibre as emulate

        run = emulate.run(args.gds, layers, scales_um=manifest.scales_um,
                          step_ratio=args.step_ratio, outdir=args.outdir,
                          manifest=manifest, manifest_path=args.manifest)
        print(f"\nemulated {len(run.density)} density scan(s) and "
              f"{len(run.markers)} marker file(s) into {args.outdir}")
        print("These are KLayout results shaped like deck output, for testing "
              "the ingest path. They are NOT Calibre results; anything derived "
              "from them says so.")

    print(f"\nRun the deck, then: lamxsim characterize {args.gds} --manifest "
          f"{args.manifest} --features-from {args.outdir}")
    print("The eps guard checks must come back empty. A non-empty one means "
          "the layout is narrower than the manifest says and every perimeter "
          "number is understated.")
    return 0


def cmd_run(args) -> int:
    """The real-data workflow: manifest -> registration -> gated analysis.

    Every step that can invalidate a result runs here rather than being left
    to the operator to remember: the manifest is checked against the layout,
    registration is fitted and its error propagated into the scale gate, and
    scales the registration cannot support are dropped before any statistic
    is computed rather than being reported with a caveat.
    """
    pass
    from .statistics import buffered_block_folds
    from .statistics import grouped_folds
    from .statistics import leakage_report
    from . import statistics as ablation

    manifest = StudyManifest.load(args.manifest)
    manifest.sample_conditions.validate()
    reader = LayoutReader(args.gds, top_cell=manifest.top_cell)
    manifest.validate_against(reader)
    bbox = manifest.die_bbox(reader)

    geometry_bbox = reader.bbox()
    print(f"layout   : {args.gds}")
    print(f"  top cell : {reader.top.name}")
    print(f"  geometry : [{geometry_bbox.xmin:g}, {geometry_bbox.ymin:g}] to "
          f"[{geometry_bbox.xmax:g}, {geometry_bbox.ymax:g}] um   (what this file holds)")
    print(f"  die      : [{bbox.xmin:g}, {bbox.ymin:g}] to [{bbox.xmax:g}, {bbox.ymax:g}] um"
          f"   ({'declared' if manifest.die_outline_um else 'ASSUMED from geometry'})")
    print(f"  metal  : {[str(m) for m in manifest.metal_layers]}")
    print(f"  vias   : {{{', '.join(f'{k}: {v}' for k, v in manifest.via_layers.items())}}}")
    if manifest.gaps:
        print("\ndeclared gaps in the manifest:")
        for g in manifest.gaps:
            print(f"  - {g}")

    failures = load_failures(args.failures)
    for n in failures.notes:
        print(f"  failure import: {n}")
    for n in manifest.sample_conditions.check_against(failures):
        print(f"  condition: {n}")
    print(f"  {len(failures)} failures across {failures.n_dies()} die(s); "
          f"modes {failures.modes()}")

    # -- registration ------------------------------------------------
    scales = list(manifest.scales_um)
    registration_report = None
    if manifest.fiducials:
        src, dst, names = load_fiducials(manifest.fiducials)
        fit_result, keep, _ = robust_fit(
            src, dst, allow_reflection=manifest.allow_reflection)
        failures = register(failures, fit_result)
        gate = scale_gate(fit_result, scales)
        registration_report = gate
        print(f"\nregistration: {fit_result.model} from {int(keep.sum())}/{len(src)} "
              f"fiducials, leave-one-out RMS {fit_result.position_sigma_um:.2f}um")
        for w in fit_result.warnings:
            print(f"  WARNING: {w}")
        if manifest.enforce_scale_gate:
            dropped = gate["rejected"]
            scales = gate["trustworthy"]
            print(f"  scale gate: analysing {scales}um; {dropped}um dropped")
            if not scales:
                print("  no configured scale survives the registration accuracy; "
                      "nothing can be analysed")
                return 1
    else:
        floor = failures.min_trustworthy_scale_um()
        if manifest.enforce_scale_gate and np.isfinite(floor):
            keep_scales = [s for s in scales if s >= floor]
            print(f"\nscale gate from the failure file's own sigma "
                  f"({failures.position_sigma_um:g}um): analysing {keep_scales}um")
            scales = keep_scales
            if not scales:
                return 1

    # -- analysis ----------------------------------------------------
    footprints = manifest.footprint_set(reader, bbox)
    run_kwargs = dict(
        layers=manifest.metal_layers,
        via_layers=manifest.via_layers, package_layers=manifest.package_layers,
        footprints=footprints, min_coverage=manifest.min_coverage,
        die_bbox=bbox, top_cell=manifest.top_cell,
        scales_um=tuple(scales), n_permutations=manifest.n_permutations,
        with_gradients=manifest.with_gradients,
        pair_selection=manifest.pair_selection,
        line_end_w_max_um=manifest.line_end_w_max_um(),
        line_rules=manifest.line_rule_map(),
        fill_layers=manifest.fill_layers, wide_width_um=manifest.wide_width_um,
        layout_revision=manifest.layout_revision,
        die_covariates=tuple(manifest.sample_conditions.covariate),
        seed=args.seed,
        allow_pooling_modes=args.allow_pooling_modes,
        allow_failures_outside_footprint=args.allow_failures_outside_footprint)

    # A condition declared stratified is stratified. Requiring the operator to
    # repeat it on the command line would let a manifest say a condition is
    # controlled while the run pools across it.
    stratify_by = list(args.stratify_by or ())
    for cond in manifest.sample_conditions.stratified:
        if cond not in stratify_by:
            stratify_by.append(cond)
    if stratify_by:
        args.stratify_by = stratify_by

    if args.stratify_by:
        strat = run_stratified(
            args.gds, failures, stratify_by=tuple(args.stratify_by),
            min_failures=args.min_stratum_failures, **run_kwargs)
        out = Path(args.outdir)
        out.mkdir(parents=True, exist_ok=True)

        print(f"\nstratified by {list(args.stratify_by)}: "
              f"{len(strat)} population(s) analysed")
        if not strat.per_stratum:
            print("  every stratum fell below "
                  f"--min-stratum-failures={args.min_stratum_failures}; "
                  "nothing was analysed. Lower the threshold or pool "
                  "deliberately with --allow-pooling-modes.")
            (out / "stratification_metadata.json").write_text(json.dumps({
                "stratify_by": list(args.stratify_by),
                "min_stratum_failures": args.min_stratum_failures,
                "strata_analysed": [], "manifest": manifest.report(),
            }, indent=2, default=str))
            return 1

        if not strat.consistency.empty:
            disagree = strat.consistency[~strat.consistency.signs_agree]
            print(f"  features whose effect reverses between strata: "
                  f"{len(disagree)}")
            if len(disagree):
                print("  (pooling these would cancel two real effects into none)")
                print(disagree[["feature", "layer", "scale_um", "strata"]]
                      .head(8).to_string(index=False))
            strat.consistency.to_csv(out / "stratum_consistency.csv", index=False)

        # No root primary table. Each mechanism is its own study, and a root
        # result would be read as the overall one while being whichever
        # stratum happened to come first.
        written = {}
        for name, sub in strat.per_stratum.items():
            sub.metadata["manifest"] = manifest.report()
            if registration_report is not None:
                sub.metadata["registration"] = registration_report
            safe = re.sub(r"\W+", "_", name).strip("_") or "unnamed"
            target = out / f"stratum_{safe}"
            if target in written.values():          # sanitisation collision
                target = out / f"stratum_{safe}_{len(written)}"
            written[name] = target
            write_results(sub, target)
            print(f"\n=== {name}: primary results ===")
            print(report_mod.format_primary(sub.associations, limit=6))

        (out / "stratification_metadata.json").write_text(json.dumps({
            "stratify_by": list(args.stratify_by),
            "min_stratum_failures": args.min_stratum_failures,
            "strata_analysed": sorted(strat.per_stratum),
            "strata_skipped_for_size": next(
                iter(strat.per_stratum.values())
            ).metadata.get("strata_skipped_for_size", {}),
            "output_directories": {k: str(v) for k, v in written.items()},
            "manifest": manifest.report(),
        }, indent=2, default=str))
        print(f"\nwritten: {out / 'stratification_metadata.json'}, "
              f"{out / 'stratum_consistency.csv'}, and one directory per stratum")
        return 0

    res = run(args.gds, failures, **run_kwargs)
    res.metadata["manifest"] = manifest.report()
    if registration_report is not None:
        res.metadata["registration"] = registration_report

    print(f"\nanalysed {len(res.associations)} feature x layer x scale combinations")
    b = res.metadata.get("permutation_budget", {})
    if b:
        print(f"  permutation budget: {b['n_permutations']} permutations over "
              f"{b['n_tests']} corrected tests -> best reachable q "
              f"{b['best_achievable_q_for_a_lone_result']:.3f}"
              f"{'' if b['sufficient'] else '  INSUFFICIENT'}")
    for note in res.metadata["uncontrolled_confounding"]:
        print(f"  UNCONTROLLED: {note}")

    # -- ablation against the position baseline ----------------------
    if args.ablation:
        from .geometry import build_grid
        # The finest scale that survived the gate, because it carries the most
        # cells and therefore the most usable folds once a buffer is removed.
        # Picking the scale after seeing which one associates best would be
        # choosing the hypothesis from the result.
        scale = args.ablation_scale_um or min(scales)
        if scale not in scales:
            print(f"\nablation skipped: scale {scale:g}um is not among the "
                  f"analysed scales {scales}")
            args.ablation = False
        # The grid must be the one the features were extracted on -- the
        # loaded geometry, not the declared die. Building it from the die
        # outline on a region of interest indexes folds into rows that do not
        # exist.
        grid = build_grid(geometry_bbox, scale)
        frame = res.features[res.features.scale_um == scale].reset_index(drop=True)
        y = frame["failure_present"].to_numpy(int)

        # Spec section 17 prefers held-out dies over any within-die split, and
        # the failure file is required to carry lot/wafer/die identity for
        # exactly this. Within-die blocking is the fallback when there is only
        # one die, and then the result cannot speak to generalisation.
        n_dies = frame["die_key"].nunique()
        if n_dies > 1:
            folds = grouped_folds(frame["die_key"].to_numpy())
            leak = {"scheme": "held-out die", "n_folds": len(folds),
                    "held_out": [f.label for f in folds]}
            print(f"\nspatial CV: leave-one-die-out over {n_dies} dies")
        else:
            folds = buffered_block_folds(grid, block_um=args.block_um,
                                         n_folds=args.n_folds,
                                         buffer_um=args.block_um)
            leak = leakage_report(folds, grid, min_separation_um=args.block_um)
            print(f"\nspatial CV: buffered blocks within a single die "
                  f"({args.block_um:g}um). This cannot show generalisation; "
                  "held-out dies would.")
        try:
            result = ablation.run(frame, y, folds, seed=args.seed)
        except ValueError as exc:
            print(f"\nablation skipped: {exc}")
            print(f"  ({len(grid)} cells at {scale:g}um, {y.sum()} cases, "
                  f"{len(folds)} folds at block {args.block_um:g}um -- either "
                  "use a finer scale or a smaller block)")
        else:
            deltas = pd.DataFrame(result.deltas)
            core = deltas[~deltas.model.str.contains(r"\+position")]
            scheme = ("held-out die" if n_dies > 1
                      else f"buffered blocks, {args.block_um:g}um")
            print(f"\n=== does geometry add anything beyond position? "
                  f"({scale:g}um, {len(folds)} folds, {scheme}) ===")
            print(core[["model", "model_auc", "baseline_auc", "delta_auc",
                        "ci_low", "ci_high", "adds_information"]]
                  .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
            out = Path(args.outdir) / "model"
            out.mkdir(parents=True, exist_ok=True)
            result.table.to_csv(out / "ablation_models.csv", index=False)
            deltas.to_csv(out / "ablation_deltas.csv", index=False)
            res.metadata["ablation_cv"] = leak

    print("\n=== primary results ===")
    print("(literature-backed geometry, FDR-corrected, at a scale the "
          "registration supports, with both classes populated)")
    print(report_mod.format_primary(res.associations, limit=10))

    paths = write_results(res, args.outdir)
    print("\nwritten:")
    for k, v in paths.items():
        print(f"  {k:18s} {v}")
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
    from .geometry import GeometryExtractor
    from .geometry import build_grid
    from .labels import failures_from_driver
    from .labels import uniform_failures

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

    res = run(path, fs, layer=layer, scales_um=(args.driver_scale_um,),
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

    ch = sub.add_parser(
        "characterize",
        help="GDS + manifest -> literature exposure atlas, no failure data")
    ch.add_argument("gds")
    ch.add_argument("--manifest", default="config/study_manifest.yaml")
    ch.add_argument("--outdir", default="results/atlas")
    ch.add_argument("--features-from", metavar="DIR",
                    help="read the density/marker features from a Calibre deck "
                         "output directory instead of extracting them with "
                         "KLayout. The directory must hold the "
                         "extraction_manifest.json written beside the deck; "
                         "everything the deck does not produce is still "
                         "computed in Python and the split is reported.")
    ch.add_argument("--candidate-percentile", type=float, default=95.0,
                    help="a cell is a candidate on a channel at or above this "
                         "percentile of the die (default 95, i.e. the top 5%%)")
    ch.set_defaults(func=cmd_characterize)

    bg = sub.add_parser(
        "budget",
        help="measure extraction cost on a clip and project it to a full chip")
    bg.add_argument("gds", nargs="+",
                    help="one or more clips that resemble the real layout "
                         "rather than a quiet corner of it. Two clips of "
                         "different size let the command fit how time grows "
                         "with polygon count; with one it has to assume 1.0, "
                         "which understates a full chip badly")
    bg.add_argument("--manifest", default="config/study_manifest.yaml")
    bg.add_argument("--full-chip-polygons", type=int, default=0,
                    help="merged polygon count over the analysed layers of the "
                         "full layout, to project to")
    bg.add_argument("--max-hours", type=float, default=24.0,
                    help="wall-clock budget for one full run (default 24)")
    bg.add_argument("--available-ram-gb", type=float, default=64.0,
                    help="RAM the run may use (default 64). The projection is "
                         "compared against half of it, because the peak is a "
                         "peak")
    bg.set_defaults(func=cmd_budget)

    dk = sub.add_parser("deck", help="generate the Calibre rule deck")
    dk.add_argument("gds", help="the layout this deck is for. Its digest, top "
                                "cell and bounding box are recorded, and "
                                "characterize refuses deck output whose "
                                "binding does not match the layout it loads.")
    dk.add_argument("--manifest", default="config/study_manifest.yaml")
    dk.add_argument("--outdir", default="calibre_deck")
    dk.add_argument("--step-ratio", type=float, default=1.0,
                    help="STEP as a fraction of WINDOW (default 1.0, "
                         "non-overlapping). The atlas grids do not overlap, so "
                         "a deck stepped differently is refused rather than "
                         "re-binned onto cells it was not measured on.")
    dk.add_argument("--emulate", action="store_true",
                    help="also produce emulated output for this layout, so the "
                         "ingest path can be exercised without a Calibre "
                         "licence. The result is labelled as emulated.")
    dk.set_defaults(func=cmd_deck)

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

    rn = sub.add_parser("run", help="real-data workflow driven by a study manifest")
    rn.add_argument("gds")
    rn.add_argument("failures")
    rn.add_argument("--manifest", default="config/study_manifest.yaml")
    rn.add_argument("--outdir", default="results")
    rn.add_argument("--seed", type=int, default=0)
    rn.add_argument("--ablation", action="store_true",
                    help="also fit the position baseline and feature ablation")
    rn.add_argument("--block-um", type=float, default=300.0,
                    help="spatial CV block and buffer size")
    rn.add_argument("--n-folds", type=int, default=5)
    rn.add_argument("--stratify-by", nargs="+", default=None,
                    metavar="COLUMN",
                    help="analyse each failure population separately, e.g. "
                         "failed_interface. Pooling mechanisms that differ can "
                         "cancel two real effects into none")
    rn.add_argument("--min-stratum-failures", type=int, default=20,
                    help="strata smaller than this are skipped and named")
    rn.add_argument("--allow-failures-outside-footprint", action="store_true",
                    help="continue when a failure lies outside the inspected "
                         "footprint; those failures are dropped and the "
                         "override is recorded")
    rn.add_argument("--allow-pooling-modes", action="store_true",
                    help="assert that the mixed failure types/layers in the "
                         "file share a defensible mechanism")
    rn.add_argument("--ablation-scale-um", type=float, default=None,
                    help="scale for the multivariate model; defaults to the "
                         "finest scale that survived the registration gate")
    rn.set_defaults(func=cmd_run)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
