"""Interpretable multivariate baseline (spec section 16).

Deliberately a regularised logistic regression rather than anything with more
capacity. The question at this stage is whether deterministic geometry carries
information, not how much accuracy can be extracted from it, and a model whose
coefficients can be read is worth more here than one that scores higher.

Every score is produced under spatially separated folds, and every geometry
model is reported as a *difference* against the position-only baseline. An
absolute AUC answers "can something predict this", which is not the question;
"does layout geometry add anything beyond where on the die you are" is.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..evidence import EvidenceClass
from .univariate import pr_auc, roc_auc


def make_model(C: float = 1.0, l1_ratio: float = 0.0) -> Pipeline:
    """Standardised, class-balanced logistic regression.

    Standardisation is not cosmetic here: features arrive in wildly different
    units (a dimensionless density, a per-um perimeter, a per-um^2 line-end
    count), and an unscaled penalty would regularise them by unit rather than
    by importance.

    ``l1_ratio`` selects the penalty: 0 is ridge, 1 is lasso, in between is
    elastic net. Sparse solutions need the saga solver, which is slower, so
    ridge stays the default.
    """
    if l1_ratio == 0.0:
        lr = LogisticRegression(C=C, solver="lbfgs", max_iter=5000,
                                class_weight="balanced")
    else:
        lr = LogisticRegression(C=C, solver="saga", l1_ratio=l1_ratio,
                                max_iter=8000, class_weight="balanced")
    return Pipeline([("scale", StandardScaler()), ("lr", lr)])


@dataclass
class ModelScore:
    name: str
    evidence_class: str
    n_features: int
    feature_names: list[str]
    roc_auc: float
    pr_auc: float
    prevalence: float
    brier: float
    calibration_slope: float
    calibration_intercept: float
    enrichment_top_1pct: float
    enrichment_top_5pct: float
    enrichment_top_10pct: float
    n_test_total: int
    n_case_total: int
    folds: list[dict] = field(default_factory=list)
    coefficients: dict[str, float] = field(default_factory=dict)
    oof_pred: np.ndarray | None = None
    oof_true: np.ndarray | None = None

    def as_row(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()
             if k not in ("folds", "oof_pred", "oof_true", "coefficients",
                          "feature_names")}
        return d


def _calibration(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Slope and intercept of the logit-linear calibration line.

    A slope near 1 means the spread of predicted risk matches reality; a slope
    well below 1 is the signature of an overfitted model whose confident
    predictions are not earned.
    """
    eps = 1e-6
    logit = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    if len(np.unique(y)) < 2 or np.std(logit) == 0:
        return float("nan"), float("nan")
    lr = LogisticRegression(solver="lbfgs", max_iter=2000, C=1e6)
    lr.fit(logit.reshape(-1, 1), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def _enrichment(p: np.ndarray, y: np.ndarray, frac: float) -> float:
    n = len(p)
    k = max(int(round(n * frac)), 1)
    if k >= n:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    top, rest = y[order[:k]], y[order[k:]]
    if rest.mean() == 0:
        return float("inf") if top.mean() > 0 else float("nan")
    return float(top.mean() / rest.mean())


def evaluate(X: np.ndarray, y: np.ndarray, folds, *, name: str,
             feature_names: list[str], evidence_class: EvidenceClass,
             C: float = 1.0) -> ModelScore:
    """Out-of-fold evaluation under the supplied spatial folds."""
    y = np.asarray(y).astype(int)
    oof_p, oof_y, rows = [], [], []

    for f in folds:
        tr, te = f.train, f.test
        if len(np.unique(y[tr])) < 2 or len(te) == 0:
            continue
        model = make_model(C=C)
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        oof_p.append(p)
        oof_y.append(y[te])
        rows.append({
            "fold": f.label, "n_train": len(tr), "n_test": len(te),
            "n_excluded": len(f.excluded), "n_case_test": int(y[te].sum()),
            "roc_auc": (roc_auc(p[y[te] == 1], p[y[te] == 0])
                        if 0 < y[te].sum() < len(te) else float("nan")),
        })

    if not oof_p:
        raise ValueError(f"{name}: no fold had both classes in training")

    p = np.concatenate(oof_p)
    t = np.concatenate(oof_y)
    auc = roc_auc(p[t == 1], p[t == 0]) if 0 < t.sum() < len(t) else float("nan")
    slope, intercept = _calibration(t, p)

    final = make_model(C=C).fit(X, y)
    coef = dict(zip(feature_names, final.named_steps["lr"].coef_[0]))

    return ModelScore(
        name=name, evidence_class=evidence_class.value, n_features=X.shape[1],
        feature_names=list(feature_names), roc_auc=float(auc),
        pr_auc=float(pr_auc(p, t)), prevalence=float(t.mean()),
        brier=float(np.mean((p - t) ** 2)),
        calibration_slope=slope, calibration_intercept=intercept,
        enrichment_top_1pct=_enrichment(p, t, 0.01),
        enrichment_top_5pct=_enrichment(p, t, 0.05),
        enrichment_top_10pct=_enrichment(p, t, 0.10),
        n_test_total=len(t), n_case_total=int(t.sum()),
        folds=rows, coefficients=coef, oof_pred=p, oof_true=t)


def delta_auc(model: ModelScore, baseline: ModelScore, *, n_boot: int = 999,
              block_size: int = 64, seed: int = 0) -> dict:
    """Paired improvement over the baseline, with a block bootstrap interval.

    Paired on the same out-of-fold predictions, and resampled in contiguous
    blocks rather than per observation, because neighbouring predictions are
    correlated and a per-observation bootstrap would return an interval far
    too narrow.
    """
    if model.oof_pred is None or baseline.oof_pred is None:
        raise ValueError("both scores must carry out-of-fold predictions")
    n = min(len(model.oof_pred), len(baseline.oof_pred))
    pm, pb, y = model.oof_pred[:n], baseline.oof_pred[:n], model.oof_true[:n]

    observed = model.roc_auc - baseline.roc_auc
    rng = np.random.default_rng(seed)
    starts = np.arange(0, n, block_size)
    deltas = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(starts), size=len(starts))
        idx = np.concatenate([np.arange(starts[j], min(starts[j] + block_size, n))
                              for j in pick])
        yy = y[idx]
        if not 0 < yy.sum() < len(yy):
            continue
        deltas.append(roc_auc(pm[idx][yy == 1], pm[idx][yy == 0])
                      - roc_auc(pb[idx][yy == 1], pb[idx][yy == 0]))
    if len(deltas) < 20:
        return {"delta_auc": observed, "ci_low": float("nan"),
                "ci_high": float("nan"), "n_boot": len(deltas)}
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "model": model.name, "baseline": baseline.name,
        "model_auc": model.roc_auc, "baseline_auc": baseline.roc_auc,
        "delta_auc": float(observed), "ci_low": float(lo), "ci_high": float(hi),
        "n_boot": len(deltas),
        "adds_information": bool(lo > 0),
    }
