"""Headless orchestration of the full analysis.

This is the single code path that ``app.py`` (interactive) and the smoke test
(headless) both call, so "runs end to end" is tested exactly as shipped. It
wires cleaning -> features -> stats -> modeling -> drivers -> figures using a
config, returning a populated :class:`~churn.state.AppState`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from . import cleaning, explain, features, modeling, profiling, stats_drivers, viz
from .state import AppState, Stage


@dataclass
class AnalysisConfig:
    target_col: str
    positive_class: Any
    cleaning: cleaning.CleaningConfig = field(default_factory=cleaning.CleaningConfig)
    imbalance: str = modeling.IMBALANCE_CLASS_WEIGHT
    test_size: float = 0.25
    threshold: float = 0.5
    n_splits: int = 5
    compute_shap: bool = True
    max_segment_drivers: int = 3


def run_full_analysis(
    raw_df: pd.DataFrame,
    roles: dict[str, str],
    config: AnalysisConfig,
    state: AppState | None = None,
    progress: Any = None,
) -> AppState:
    """Execute clean -> transform -> model -> drivers -> visualize.

    ``progress`` is an optional callback ``(fraction: float, message: str)``.
    """
    state = state or AppState()

    def _tick(frac: float, msg: str) -> None:
        if progress is not None:
            progress(frac, msg)

    # --- target ---
    y_full = profiling.encode_target(raw_df[config.target_col], config.positive_class)
    state.target_col = config.target_col
    state.positive_class = config.positive_class

    # --- clean ---
    _tick(0.04, "Cleaning data (dedupe, drops, type coercion)…")
    clean_df, clean_report, feature_cols = cleaning.clean_dataframe(
        raw_df, roles, config.cleaning, target_col=config.target_col, log=state.log,
    )
    y = y_full.loc[clean_df.index].reset_index(drop=True)
    clean_df = clean_df.reset_index(drop=True)
    state.clean_df = clean_df
    state.cleaning_config = {"report": clean_report}
    state.base_rate = profiling.base_rate(y)

    # --- transform spec + VIF ---
    _tick(0.12, "Building feature pipeline + checking collinearity (VIF)…")
    spec = features.build_feature_spec(roles, feature_cols, config.cleaning)
    state.feature_spec = spec
    ndm = features.numeric_design_matrix(clean_df, spec)
    state.vif_table = features.compute_vif(ndm)
    state.log.add("transform", "feature spec",
                  f"{len(spec.numeric_cols)} numeric, {len(spec.categorical_cols)} categorical, "
                  f"{len(spec.datetime_cols)} datetime->recency")

    # --- univariate stats ---
    _tick(0.18, "Univariate statistics (tests + effect sizes + FDR)…")
    stats_table = stats_drivers.run_univariate(clean_df, y, roles, feature_cols)

    # --- model (the long pole) — sub-steps stream through the progress band ---
    _model_msgs = {"n": 0}
    _MODEL_TOTAL = 5  # LR cv, LR fit, odds-ratios, GBM cv, GBM fit

    def _model_progress(msg: str) -> None:
        _model_msgs["n"] += 1
        frac = 0.22 + 0.56 * (_model_msgs["n"] / _MODEL_TOTAL)
        _tick(min(frac, 0.78), msg)

    X = clean_df[spec.all_input_cols]
    model_out = modeling.train_all(
        spec, X, y, imbalance=config.imbalance, test_size=config.test_size,
        threshold=config.threshold, n_splits=config.n_splits, progress=_model_progress,
    )
    state.model_result = model_out
    state.threshold = config.threshold
    state.log.add("model", "trained LR + GBM",
                  f"imbalance={config.imbalance}, {config.n_splits}-fold CV, test={config.test_size:.0%}")

    # --- drivers (SHAP + permutation + OR) ---
    shap_res = None
    if config.compute_shap:
        _tick(0.80, "Computing SHAP values (per-feature attribution)…")
        shap_res = explain.compute_shap(model_out.gbm, model_out.X_test)
    _tick(0.86, "Computing permutation importance…")
    perm = explain.compute_permutation_importance(model_out.gbm, model_out.X_test, model_out.y_test)
    _tick(0.90, "Assembling the unified driver table…")
    gran, parent = explain.build_driver_table(
        model_out.lr, model_out.gbm, model_out.X_test, model_out.y_test,
        shap_res=shap_res, perm=perm,
    )
    parent = explain.reconcile_with_stats(parent, stats_table)
    state.driver_table = {"granular": gran, "parent": parent, "stats": stats_table, "shap": shap_res}

    # --- figures ---
    _tick(0.94, "Rendering figures…")
    state.figures = _build_figures(state, clean_df, y, spec, model_out, gran, parent, shap_res, ndm)
    _tick(1.0, "Done.")
    return state


def _build_figures(state, clean_df, y, spec, model_out, gran, parent, shap_res, ndm) -> dict[str, Any]:
    figs: dict[str, Any] = {}
    figs["target_balance"] = viz.target_balance(y, state.base_rate)
    figs["correlation"] = viz.correlation_heatmap(ndm)
    figs["driver_importance"] = viz.driver_importance_bar(parent)

    # top categorical + numeric drivers for segment/dist plots
    top_parents = parent["parent"].tolist()
    cat_done = num_done = 0
    for p in top_parents:
        if p in spec.categorical_cols and cat_done < 3 and p in clean_df.columns:
            figs[f"segment::{p}"] = viz.churn_rate_by_segment(clean_df, p, y)
            cat_done += 1
        elif p in spec.numeric_cols and num_done < 3 and p in clean_df.columns:
            figs[f"distribution::{p}"] = viz.numeric_distribution(clean_df, p, y)
            num_done += 1

    if shap_res is not None:
        figs["shap_beeswarm"] = viz.shap_beeswarm(shap_res, model_out.gbm.label_map)
        for feat in gran["feature"].head(3):
            figs[f"shap_dependence::{feat}"] = viz.shap_dependence(shap_res, feat, model_out.gbm.label_map)

    figs["roc"] = viz.roc_curve_fig(model_out.lr, model_out.gbm)
    figs["pr"] = viz.pr_curve_fig(model_out.lr, model_out.gbm, base_rate=state.base_rate)
    figs["confusion"] = viz.confusion_heatmap(model_out.gbm)
    figs["calibration"] = viz.calibration_fig(model_out.lr, model_out.gbm)
    figs["lift"] = viz.lift_gains_fig(model_out.gbm)
    figs["cumulative_gains"] = viz.cumulative_gains_fig(model_out.gbm)
    figs["threshold"] = viz.threshold_tradeoff_fig(model_out.gbm)
    return figs


def segment_rates_for_payload(clean_df, y, spec, parent, max_drivers: int = 3) -> dict:
    """Notable segment churn rates for the report payload (aggregates only)."""
    out: dict[str, list[dict]] = {}
    import numpy as np

    done = 0
    for p in parent["parent"].tolist():
        if p in spec.categorical_cols and p in clean_df.columns and done < max_drivers:
            d = pd.DataFrame({"level": clean_df[p].astype(str).values, "y": np.asarray(y)})
            g = d.groupby("level").agg(rate=("y", "mean"), n=("y", "size")).reset_index()
            g = g.sort_values("rate", ascending=False)
            out[p] = [
                {"level": r["level"], "churn_rate": round(float(r["rate"]), 4), "n": int(r["n"])}
                for _, r in g.iterrows()
            ]
            done += 1
    return out
