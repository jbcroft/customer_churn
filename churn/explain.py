"""Key drivers — the headline output.

Three lenses, reconciled into ONE ranked driver table with a direction
(↑ churn / ↓ churn):

  1. Logistic-Regression odds ratios   — OR > 1 increases churn odds (+ p-value).
  2. Permutation importance on the GBM  — model-agnostic magnitude of predictive
     contribution (less biased than impurity importance).
  3. SHAP (TreeExplainer on the GBM)    — global mean |SHAP| for ranking, plus the
     sign of the SHAP-vs-value relationship for direction.

Encoded features are reconciled back to their original fields via the §5.5
name map, so the executive view aggregates one-hot columns to a parent feature
while the granular view stays available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from . import RANDOM_STATE
from .modeling import ModelResult

SHAP_MAX_SAMPLES = 2000  # subsample for SHAP on large datasets (with a notice)


@dataclass
class ShapResult:
    values: np.ndarray            # (n_samples, n_features) SHAP values
    base_value: float
    X_encoded: pd.DataFrame       # the encoded design matrix SHAP was computed on
    feature_names: list[str]
    mean_abs: pd.Series           # global importance per encoded feature
    sample_note: str = ""


def _transform_for_explain(model: ModelResult, X: pd.DataFrame) -> pd.DataFrame:
    """Run X through the fitted recency + ColumnTransformer (no classifier)."""
    recency = model.pipeline.named_steps["recency"]
    prep = model.pipeline.named_steps["prep"]
    Z = prep.transform(recency.transform(X))
    return pd.DataFrame(Z, columns=model.encoded_names).reset_index(drop=True)


def compute_shap(
    gbm: ModelResult, X: pd.DataFrame, max_samples: int = SHAP_MAX_SAMPLES
) -> ShapResult | None:
    """SHAP values for the tree/boosting model. Returns None if SHAP unavailable."""
    try:
        import shap
    except Exception:  # noqa: BLE001
        return None

    Z = _transform_for_explain(gbm, X)
    note = ""
    if len(Z) > max_samples:
        Z = Z.sample(max_samples, random_state=RANDOM_STATE).reset_index(drop=True)
        note = f"SHAP computed on a random subsample of {max_samples:,} rows for speed."

    clf = gbm.pipeline.named_steps["clf"]
    try:
        explainer = shap.TreeExplainer(clf)
        raw = explainer.shap_values(Z)
        base = explainer.expected_value
    except Exception:  # noqa: BLE001 - fall back to model-agnostic explainer
        try:
            explainer = shap.Explainer(clf.predict_proba, Z)
            sv = explainer(Z)
            raw = sv.values
            base = sv.base_values
        except Exception:  # noqa: BLE001
            return None

    vals = np.asarray(raw)
    # binary classifiers may return (n, f, 2) or a list per class -> take positive
    if vals.ndim == 3:
        vals = vals[:, :, -1]
    if isinstance(base, (list, np.ndarray)):
        base = float(np.asarray(base).ravel()[-1])

    mean_abs = pd.Series(np.abs(vals).mean(axis=0), index=gbm.encoded_names)
    return ShapResult(
        values=vals, base_value=float(base), X_encoded=Z,
        feature_names=gbm.encoded_names, mean_abs=mean_abs, sample_note=note,
    )


def shap_direction(shap_res: ShapResult) -> dict[str, str]:
    """Per encoded feature: does a HIGHER value push churn up or down?"""
    out: dict[str, str] = {}
    for i, name in enumerate(shap_res.feature_names):
        x = shap_res.X_encoded.iloc[:, i].values
        s = shap_res.values[:, i]
        if np.std(x) < 1e-12 or np.std(s) < 1e-12:
            out[name] = "—"
            continue
        corr = np.corrcoef(x, s)[0, 1]
        out[name] = "↑ churn" if corr > 0 else "↓ churn"
    return out


def compute_permutation_importance(
    gbm: ModelResult, X_test: pd.DataFrame, y_test: np.ndarray, n_repeats: int = 5
) -> pd.Series:
    """Permutation importance on the GBM (model-agnostic), keyed by encoded name."""
    # permute on the encoded matrix so importances align with encoded names
    Z = _transform_for_explain(gbm, X_test)
    clf = gbm.pipeline.named_steps["clf"]
    try:
        r = permutation_importance(
            clf, Z, y_test, n_repeats=n_repeats,
            random_state=RANDOM_STATE, scoring="average_precision", n_jobs=-1,
        )
        return pd.Series(r.importances_mean, index=gbm.encoded_names)
    except Exception:  # noqa: BLE001
        return pd.Series(0.0, index=gbm.encoded_names)


def _rank01(s: pd.Series) -> pd.Series:
    """Min-max normalize a magnitude series to [0, 1]."""
    s = s.astype(float)
    rng = s.max() - s.min()
    if rng <= 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.min()) / rng


def build_driver_table(
    lr: ModelResult | None,
    gbm: ModelResult | None,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    *,
    shap_res: ShapResult | None = None,
    perm: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (granular_table, parent_table).

    granular_table: one row per encoded feature with OR, SHAP, permutation,
    significance, direction, and a one-line interpretation.
    parent_table: aggregated to the original field for the executive summary.
    """
    ref = gbm or lr
    if ref is None:
        return pd.DataFrame(), pd.DataFrame()
    encoded = ref.encoded_names
    label_map = ref.label_map
    parent_map = ref.parent_map

    rows = []
    # odds ratios indexed by encoded feature
    or_map: dict[str, dict[str, Any]] = {}
    if lr is not None and lr.odds_ratios is not None and not lr.odds_ratios.empty:
        for _, r in lr.odds_ratios.iterrows():
            or_map[r["feature"]] = r.to_dict()

    shap_dir = shap_direction(shap_res) if shap_res is not None else {}
    shap_imp = shap_res.mean_abs if shap_res is not None else pd.Series(dtype=float)
    perm = perm if perm is not None else pd.Series(dtype=float)

    for feat in encoded:
        orr = or_map.get(feat, {})
        odds = orr.get("odds_ratio", np.nan)
        p = orr.get("p_value", np.nan)
        # direction precedence: SHAP sign, else OR
        direction = shap_dir.get(feat, "—")
        if direction == "—" and not np.isnan(odds):
            direction = "↑ churn" if odds > 1 else "↓ churn"
        rows.append({
            "feature": feat,
            "label": label_map.get(feat, feat),
            "parent": parent_map.get(feat, feat),
            "direction": direction,
            "odds_ratio": float(odds) if odds == odds else np.nan,
            "or_p_value": float(p) if p == p else np.nan,
            "significant": bool(p < 0.05) if p == p else False,
            "shap_importance": float(shap_imp.get(feat, np.nan)),
            "perm_importance": float(perm.get(feat, np.nan)),
        })

    gran = pd.DataFrame(rows)
    if gran.empty:
        return gran, gran

    # unified strength: blend normalized SHAP + permutation + |log OR|
    comp = pd.DataFrame(index=gran["feature"])
    comp["shap"] = _rank01(gran.set_index("feature")["shap_importance"].fillna(0))
    comp["perm"] = _rank01(gran.set_index("feature")["perm_importance"].fillna(0))
    logor = np.abs(np.log(gran.set_index("feature")["odds_ratio"].replace(0, np.nan)))
    comp["or"] = _rank01(logor.fillna(0))
    gran["strength"] = (comp[["shap", "perm", "or"]].mean(axis=1)).values
    gran = gran.sort_values("strength", ascending=False).reset_index(drop=True)
    gran["strength_rank"] = np.arange(1, len(gran) + 1)
    gran["interpretation"] = gran.apply(_one_liner, axis=1)

    parent = _aggregate_to_parent(gran)
    return gran, parent


def _one_liner(row: pd.Series) -> str:
    direction = row["direction"]
    verb = "raises" if "↑" in direction else ("lowers" if "↓" in direction else "is associated with")
    bits = [f"{row['label']} {verb} churn risk"]
    if row.get("odds_ratio") == row.get("odds_ratio") and not np.isnan(row["odds_ratio"]):
        bits.append(f"(odds ratio {row['odds_ratio']:.2f}"
                    + (", significant)" if row["significant"] else ")"))
    return " ".join(bits)


def _aggregate_to_parent(gran: pd.DataFrame) -> pd.DataFrame:
    """Collapse encoded features to their parent field for the exec view."""
    agg = (
        gran.groupby("parent")
        .agg(
            strength=("strength", "max"),
            shap_importance=("shap_importance", "sum"),
            perm_importance=("perm_importance", "max"),
            any_significant=("significant", "any"),
            n_encoded=("feature", "count"),
        )
        .reset_index()
        .sort_values("strength", ascending=False)
        .reset_index(drop=True)
    )
    # parent direction = direction of its strongest encoded child
    dir_by_parent = (
        gran.sort_values("strength", ascending=False)
        .groupby("parent")["direction"].first()
    )
    agg["direction"] = agg["parent"].map(dir_by_parent)
    agg["strength_rank"] = np.arange(1, len(agg) + 1)
    return agg


def reconcile_with_stats(
    parent_table: pd.DataFrame, stats_table: pd.DataFrame
) -> pd.DataFrame:
    """Cross-check: do model drivers and univariate stats agree on direction?"""
    from .features import RECENCY_SUFFIX

    if parent_table.empty or stats_table.empty:
        return parent_table
    # stats may key recency features as "<col>__days_since"; map back to parent
    stat_dir: dict[str, str] = {}
    for _, r in stats_table.iterrows():
        key = str(r["feature"])
        if key.endswith(RECENCY_SUFFIX):
            key = key[: -len(RECENCY_SUFFIX)]
        stat_dir[key] = r["direction"]

    out = parent_table.copy()
    out["univariate_direction"] = out["parent"].map(stat_dir).fillna("n/a")

    def _agree(r):
        ud = str(r["univariate_direction"])
        if "↑" not in ud and "↓" not in ud:
            return "n/a"  # univariate test is non-directional (categorical) or absent
        return ("↑" in str(r["direction"]) and "↑" in ud) or ("↓" in str(r["direction"]) and "↓" in ud)

    out["agrees"] = out.apply(_agree, axis=1)
    return out
