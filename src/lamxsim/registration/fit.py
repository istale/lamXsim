"""Registration quality, and the scale floor it implies.

The residual of a fit is not the accuracy of the mapping. Each correspondence
supplies two equations while the model consumes ``dof`` of them, so a fit with
few fiducials and a rich model reports a small residual because it has enough
freedom to pass close to every point -- not because it will place an
unseen failure correctly. Leave-one-out prediction error is the honest number
and is what the scale floor is derived from.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .transform import MODEL_DOF, Transform2D, fit_transform

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
