"""2D coordinate transforms between the layout frame and a measurement frame.

Failure locations arrive in whatever frame the inspection tool produced --
wafer stage coordinates, an image frame, a backside acoustic scan. Bringing
them into layout coordinates is a fitted transform, not a relabelling, and the
quality of that fit decides which analysis scales mean anything.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
