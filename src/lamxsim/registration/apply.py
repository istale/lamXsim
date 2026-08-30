"""Bring measured failure locations into the layout coordinate frame.

Registration is the step spec section 10 assumes has already happened when it
says to map failures "into the same physical coordinate system". It is a
fitted transform with its own uncertainty, and that uncertainty propagates:
it becomes the ``position_sigma_um`` that decides which analysis scales are
admissible, so it must travel with the data rather than being asserted in a
config file.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from ..labels.failure import FailureSet
from .fit import RegistrationFit, SCALE_FACTOR


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
