"""Robust modeling: LR + gradient boosting, CV, imbalance handling, metrics.

Non-negotiable guarantees:
  * Stratified train/test split AND stratified k-fold CV (mean ± std).
  * Every learned transform + any resampling happens INSIDE a pipeline fit on
    the training fold only (imblearn ``Pipeline`` for SMOTE) — never touches the
    test set. The test set is never resampled.
  * Imbalance-aware metrics lead with PR-AUC; calibration + lift/gains included.

Two models, by design:
  1. Logistic Regression — interpretable; coefficients -> odds ratios with 95%
     CIs (statsmodels Logit when it converges, else bootstrap).
  2. Gradient boosting — XGBoost if its native lib loads, else sklearn
     ``HistGradientBoostingClassifier`` (functionally equivalent for our needs).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from . import RANDOM_STATE
from .features import FeatureSpec, build_preprocessor, feature_name_map, preprocessor_steps

IMBALANCE_CLASS_WEIGHT = "class_weight"
IMBALANCE_SMOTE = "smote"
IMBALANCE_NONE = "none"


def xgboost_available() -> bool:
    """True only if xgboost imports AND its native library actually loads."""
    try:
        import xgboost as xgb  # noqa: F401

        xgb.XGBClassifier()  # forces the native lib (libxgboost) to load
        return True
    except Exception:  # noqa: BLE001 - missing libomp etc.
        return False


def gbm_estimator(scale_pos_weight: float | None, balanced: bool):
    """Return (estimator, display_name). Prefers XGBoost, falls back to HistGBM."""
    if xgboost_available():
        import xgboost as xgb

        return (
            xgb.XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight if balanced else 1.0,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "XGBoost",
        )
    return (
        HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.05,
            max_iter=300,
            class_weight="balanced" if balanced else None,
            random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting",
    )


@dataclass
class ModelResult:
    name: str
    kind: str  # "lr" | "gbm"
    pipeline: Any
    scaled: bool
    threshold: float
    cv: dict[str, tuple[float, float]]  # metric -> (mean, std)
    test: dict[str, float]
    proba: np.ndarray
    y_test: np.ndarray
    pred: np.ndarray
    confusion: np.ndarray
    roc: tuple[np.ndarray, np.ndarray]
    pr: tuple[np.ndarray, np.ndarray]
    calibration: tuple[np.ndarray, np.ndarray]
    gains: pd.DataFrame
    odds_ratios: pd.DataFrame | None = None
    encoded_names: list[str] = field(default_factory=list)
    parent_map: dict[str, str] = field(default_factory=dict)
    label_map: dict[str, str] = field(default_factory=dict)


@dataclass
class ModelingOutput:
    spec: FeatureSpec
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    base_rate: float
    imbalance_method: str
    lr: ModelResult | None = None
    gbm: ModelResult | None = None
    gbm_name: str = "gradient boosting"


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def _gains_table(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Decile lift / cumulative gains table (top deciles first)."""
    df = pd.DataFrame({"y": y_true, "p": proba}).sort_values("p", ascending=False)
    df = df.reset_index(drop=True)
    n = len(df)
    base = df["y"].mean() if n else 0.0
    df["decile"] = pd.qcut(df["p"].rank(method="first", ascending=False),
                           q=min(n_bins, max(1, n)), labels=False) + 1
    rows = []
    cum_pos = 0
    total_pos = df["y"].sum()
    for d in sorted(df["decile"].unique()):
        seg = df[df["decile"] == d]
        cum_pos += seg["y"].sum()
        seg_rate = seg["y"].mean()
        rows.append({
            "decile": int(d),
            "n": len(seg),
            "churn_rate": round(float(seg_rate), 4),
            "lift": round(float(seg_rate / base), 3) if base else np.nan,
            "cum_pct_churners_captured": round(
                float(cum_pos / total_pos), 4) if total_pos else np.nan,
        })
    return pd.DataFrame(rows)


def _metrics_at_threshold(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = {
            "roc_auc": float(roc_auc_score(y_true, proba)) if len(set(y_true)) > 1 else float("nan"),
            "pr_auc": float(average_precision_score(y_true, proba)) if len(set(y_true)) > 1 else float("nan"),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "brier": float(brier_score_loss(y_true, proba)),
            "accuracy": float((pred == y_true).mean()),
        }
    return out


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Threshold that maximizes F1 on the given data."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    if len(thr) == 0:
        return 0.5
    # f1 has len(thr)+1; align to thresholds
    best = int(np.nanargmax(f1[:-1])) if len(f1) > 1 else 0
    return float(thr[min(best, len(thr) - 1)])


def cost_weighted_threshold(
    y_true: np.ndarray, proba: np.ndarray, cost_fn: float = 5.0, cost_fp: float = 1.0
) -> float:
    """Threshold minimizing expected cost (default: a missed churner costs 5x a false alarm)."""
    thresholds = np.unique(np.round(proba, 3))
    best_t, best_cost = 0.5, np.inf
    for t in thresholds:
        pred = (proba >= t).astype(int)
        fn = int(((pred == 0) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        cost = cost_fn * fn + cost_fp * fp
        if cost < best_cost:
            best_cost, best_t = cost, float(t)
    return best_t


# ----------------------------------------------------------------------
# Odds ratios (LR interpretability)
# ----------------------------------------------------------------------
def _odds_ratios_statsmodels(Z: pd.DataFrame, y: np.ndarray) -> pd.DataFrame | None:
    import statsmodels.api as sm

    # drop perfectly collinear / constant columns to help convergence
    Zc = Z.loc[:, Z.nunique() > 1]
    corr = Zc.corr().abs()
    drop = set()
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if b not in drop and corr.loc[a, b] > 0.995:
                drop.add(b)
    Zc = Zc.drop(columns=list(drop))
    Xc = sm.add_constant(Zc, has_constant="add")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.Logit(y, Xc).fit(disp=0, maxiter=100, method="lbfgs")
        params = res.params.drop("const", errors="ignore")
        ci = res.conf_int().drop("const", errors="ignore")
        pvals = res.pvalues.drop("const", errors="ignore")
        out = pd.DataFrame({
            "feature": params.index,
            "coef": params.values,
            "odds_ratio": np.exp(params.values),
            "ci_low": np.exp(ci[0].values),
            "ci_high": np.exp(ci[1].values),
            "p_value": pvals.values,
            "method": "statsmodels",
        })
        return out
    except Exception:  # noqa: BLE001
        return None


def _odds_ratios_bootstrap(
    spec: FeatureSpec, X: pd.DataFrame, y: np.ndarray, n_boot: int = 150
) -> pd.DataFrame:
    """Bootstrap 95% CIs for odds ratios from the sklearn LR (robust fallback)."""
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(X)
    coefs = []
    base_names: list[str] | None = None
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        pipe = build_preprocessor(spec, scale=True)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        from sklearn.pipeline import Pipeline as SkPipe

        full = SkPipe([("prep", pipe), ("clf", clf)])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                full.fit(X.iloc[idx], y[idx])
        except Exception:  # noqa: BLE001
            continue
        names = list(full.named_steps["prep"].get_feature_names_out())
        if base_names is None:
            base_names = names
        if names == base_names:
            coefs.append(full.named_steps["clf"].coef_[0])
    if not coefs or base_names is None:
        return pd.DataFrame()
    arr = np.vstack(coefs)
    lo = np.percentile(arr, 2.5, axis=0)
    hi = np.percentile(arr, 97.5, axis=0)
    mean = arr.mean(axis=0)
    # bootstrap two-sided p: fraction of sign disagreement
    p = 2 * np.minimum((arr > 0).mean(axis=0), (arr < 0).mean(axis=0))
    return pd.DataFrame({
        "feature": base_names,
        "coef": mean,
        "odds_ratio": np.exp(mean),
        "ci_low": np.exp(lo),
        "ci_high": np.exp(hi),
        "p_value": np.clip(p, 0, 1),
        "method": "bootstrap",
    })


def compute_odds_ratios(
    spec: FeatureSpec, fitted_pipe, X_train: pd.DataFrame, y_train: np.ndarray
) -> pd.DataFrame:
    """Odds-ratio table with direction. Tries statsmodels, falls back to bootstrap."""
    recency = fitted_pipe.named_steps["recency"]
    prep = fitted_pipe.named_steps["prep"]
    Z = prep.transform(recency.transform(X_train))
    Z = pd.DataFrame(Z).reset_index(drop=True)
    sm_table = _odds_ratios_statsmodels(Z, np.asarray(y_train))
    # statsmodels can "converge" yet return NaN standard errors (singular Hessian
    # under quasi-separation). In that case the CIs/p-values are useless -> bootstrap.
    sm_ok = (
        sm_table is not None and not sm_table.empty
        and sm_table["p_value"].notna().any()
        and sm_table["ci_low"].notna().any()
    )
    table = sm_table if sm_ok else _odds_ratios_bootstrap(spec, X_train, np.asarray(y_train))
    if table.empty:
        return table
    table["direction"] = np.where(table["odds_ratio"] > 1, "↑ churn", "↓ churn")
    return table.sort_values("odds_ratio", key=lambda s: (s - 1).abs(), ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------
def _build_full_pipeline(spec: FeatureSpec, kind: str, imbalance: str, base_rate: float):
    scale = kind == "lr"
    steps = preprocessor_steps(spec, scale=scale)  # flat: recency + ColumnTransformer
    balanced = imbalance == IMBALANCE_CLASS_WEIGHT
    if kind == "lr":
        est = LogisticRegression(
            max_iter=1000,
            class_weight="balanced" if balanced else None,
            random_state=RANDOM_STATE,
        )
    else:
        pos = max(base_rate, 1e-6)
        spw = (1 - pos) / pos
        est, _ = gbm_estimator(scale_pos_weight=spw, balanced=balanced)

    if imbalance == IMBALANCE_SMOTE:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipe

        pipe = ImbPipe([*steps, ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
                        ("clf", est)])
    else:
        from sklearn.pipeline import Pipeline as SkPipe

        pipe = SkPipe([*steps, ("clf", est)])
    pipe.set_output(transform="pandas")
    return pipe


_CV_SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "f1": "f1",
    "brier": "neg_brier_score",
}


def _run_cv(pipeline, X, y, n_splits: int) -> dict[str, tuple[float, float]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_validate(pipeline, X, y, cv=skf, scoring=_CV_SCORING, n_jobs=1)
    out: dict[str, tuple[float, float]] = {}
    for name in _CV_SCORING:
        vals = scores[f"test_{name}"]
        if name == "brier":
            vals = -vals
        out[name] = (float(np.mean(vals)), float(np.std(vals)))
    return out


def fit_model(
    spec: FeatureSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    kind: str,
    imbalance: str,
    base_rate: float,
    threshold: float = 0.5,
    n_splits: int = 5,
    compute_or: bool = True,
) -> ModelResult:
    y_tr = np.asarray(y_train).astype(int)
    y_te = np.asarray(y_test).astype(int)

    pipeline = _build_full_pipeline(spec, kind, imbalance, base_rate)
    cv = _run_cv(pipeline, X_train, y_tr, n_splits)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(X_train, y_tr)
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)

    test = _metrics_at_threshold(y_te, proba, threshold)
    cm = confusion_matrix(y_te, pred, labels=[0, 1])
    fpr, tpr, _ = roc_curve(y_te, proba) if len(set(y_te)) > 1 else (np.array([0, 1]), np.array([0, 1]), None)
    prec, rec, _ = precision_recall_curve(y_te, proba)
    n_cal = min(10, max(2, len(np.unique(proba)) // 5))
    try:
        prob_true, prob_pred = calibration_curve(y_te, proba, n_bins=n_cal, strategy="quantile")
    except Exception:  # noqa: BLE001
        prob_true, prob_pred = np.array([]), np.array([])
    gains = _gains_table(y_te, proba)

    enc, parent, label = feature_name_map(pipeline.named_steps["prep"], spec)

    odds = None
    if kind == "lr" and compute_or:
        odds = compute_odds_ratios(spec, pipeline, X_train, y_tr)

    name = "Logistic Regression" if kind == "lr" else gbm_estimator(None, False)[1]
    return ModelResult(
        name=name, kind=kind, pipeline=pipeline, scaled=(kind == "lr"),
        threshold=threshold, cv=cv, test=test, proba=proba, y_test=y_te, pred=pred,
        confusion=cm, roc=(fpr, tpr), pr=(prec, rec),
        calibration=(prob_true, prob_pred), gains=gains, odds_ratios=odds,
        encoded_names=enc, parent_map=parent, label_map=label,
    )


def train_all(
    spec: FeatureSpec,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    imbalance: str = IMBALANCE_CLASS_WEIGHT,
    test_size: float = 0.25,
    threshold: float = 0.5,
    n_splits: int = 5,
    fit_lr: bool = True,
    fit_gbm: bool = True,
) -> ModelingOutput:
    """Stratified split, then fit LR and GBM with shared train/test folds."""
    y = pd.Series(y).astype(int).reset_index(drop=True)
    X = X.reset_index(drop=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE,
    )
    base = float(y_train.mean())
    out = ModelingOutput(
        spec=spec, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        base_rate=base, imbalance_method=imbalance,
    )
    if fit_lr:
        out.lr = fit_model(spec, X_train, y_train, X_test, y_test,
                           kind="lr", imbalance=imbalance, base_rate=base,
                           threshold=threshold, n_splits=n_splits)
    if fit_gbm:
        out.gbm = fit_model(spec, X_train, y_train, X_test, y_test,
                            kind="gbm", imbalance=imbalance, base_rate=base,
                            threshold=threshold, n_splits=n_splits)
        out.gbm_name = out.gbm.name
    return out
