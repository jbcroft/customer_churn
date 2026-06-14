"""All Plotly figures — pure functions (data in, figure out, no Streamlit).

Every figure carries titles/labels readable by a non-technical viewer. Figures
are returned (not shown) so ``app.py`` can render them and ``report.py`` can
optionally embed static copies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from .explain import ShapResult
from .modeling import ModelResult

CHURN_COLOR = "#d62728"
RETAIN_COLOR = "#2ca02c"
ACCENT = "#1f77b4"


def _layout(fig: go.Figure, title: str, x: str = "", y: str = "") -> go.Figure:
    fig.update_layout(
        title=title, xaxis_title=x, yaxis_title=y,
        template="plotly_white", margin=dict(l=60, r=30, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ----------------------------------------------------------------------
# Target / EDA
# ----------------------------------------------------------------------
def target_balance(y: pd.Series, base_rate: float) -> go.Figure:
    counts = pd.Series(y).map({0: "Retained", 1: "Churned"}).value_counts()
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker_color=[RETAIN_COLOR if k == "Retained" else CHURN_COLOR for k in counts.index],
        text=counts.values, textposition="auto",
    ))
    return _layout(fig, f"Target balance — base churn rate {base_rate:.1%}", "Class", "Customers")


def churn_rate_by_segment(df: pd.DataFrame, col: str, y: pd.Series, top_n: int = 12) -> go.Figure:
    """Churn rate per level of a categorical driver (with counts)."""
    d = pd.DataFrame({"level": df[col].astype(str).values, "y": np.asarray(y)})
    g = d.groupby("level").agg(rate=("y", "mean"), n=("y", "size")).reset_index()
    g = g.sort_values("rate", ascending=False).head(top_n)
    fig = go.Figure(go.Bar(
        x=g["level"], y=g["rate"], marker_color=ACCENT,
        text=[f"{r:.0%}<br>n={n}" for r, n in zip(g["rate"], g["n"])], textposition="auto",
    ))
    base = float(np.mean(y))
    fig.add_hline(y=base, line_dash="dash", line_color=CHURN_COLOR,
                  annotation_text=f"base {base:.0%}", annotation_position="top left")
    return _layout(fig, f"Churn rate by {col}", col, "Churn rate")


def numeric_distribution(df: pd.DataFrame, col: str, y: pd.Series) -> go.Figure:
    """Distribution overlay (churned vs retained) for a numeric driver."""
    d = pd.DataFrame({"val": pd.to_numeric(df[col], errors="coerce"), "y": np.asarray(y)}).dropna()
    fig = go.Figure()
    for label, code, color in [("Retained", 0, RETAIN_COLOR), ("Churned", 1, CHURN_COLOR)]:
        fig.add_trace(go.Violin(
            y=d[d["y"] == code]["val"], name=label, line_color=color,
            box_visible=True, meanline_visible=True, opacity=0.7,
        ))
    return _layout(fig, f"{col}: churned vs retained", "", col)


def correlation_heatmap(numeric_df: pd.DataFrame) -> go.Figure:
    corr = numeric_df.corr(numeric_only=True)
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        aspect="auto",
    )
    return _layout(fig, "Correlation heatmap (numeric features)")


# ----------------------------------------------------------------------
# Drivers / SHAP
# ----------------------------------------------------------------------
def driver_importance_bar(driver_table: pd.DataFrame, top_n: int = 15, value: str = "strength") -> go.Figure:
    d = driver_table.head(top_n).iloc[::-1]
    name_col = "label" if "label" in d.columns else "parent"
    colors = [CHURN_COLOR if "↑" in str(dr) else RETAIN_COLOR for dr in d.get("direction", [""] * len(d))]
    fig = go.Figure(go.Bar(
        x=d[value], y=d[name_col], orientation="h", marker_color=colors,
        text=[f"{v:.3f}" for v in d[value]], textposition="auto",
    ))
    return _layout(fig, "Key churn drivers (red ↑churn, green ↓churn)", value.title(), "")


def shap_beeswarm(shap_res: ShapResult, label_map: dict[str, str], top_n: int = 12) -> go.Figure:
    """Plotly approximation of the SHAP summary beeswarm plot."""
    order = shap_res.mean_abs.sort_values(ascending=False).head(top_n).index.tolist()
    fig = go.Figure()
    for i, feat in enumerate(order[::-1]):
        j = shap_res.feature_names.index(feat)
        s = shap_res.values[:, j]
        x = shap_res.X_encoded.iloc[:, j].values.astype(float)
        # color by normalized feature value
        rng = np.nanmax(x) - np.nanmin(x)
        cval = (x - np.nanmin(x)) / rng if rng > 0 else np.zeros_like(x)
        jitter = (np.random.default_rng(i).random(len(s)) - 0.5) * 0.35
        fig.add_trace(go.Scatter(
            x=s, y=np.full(len(s), i) + jitter, mode="markers",
            marker=dict(color=cval, colorscale="RdBu_r", size=4, opacity=0.6,
                        colorbar=dict(title="feature<br>value") if i == len(order) - 1 else None,
                        showscale=(i == len(order) - 1)),
            name=label_map.get(feat, feat), showlegend=False,
            hovertext=label_map.get(feat, feat),
        ))
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(order))),
                     ticktext=[label_map.get(f, f) for f in order[::-1]])
    fig.add_vline(x=0, line_color="gray")
    return _layout(fig, "SHAP summary (impact on churn prediction)", "SHAP value → churn", "")


def shap_dependence(shap_res: ShapResult, feature: str, label_map: dict[str, str]) -> go.Figure:
    """SHAP dependence plot: how the driver's value moves churn."""
    j = shap_res.feature_names.index(feature)
    x = shap_res.X_encoded.iloc[:, j].values.astype(float)
    s = shap_res.values[:, j]
    fig = go.Figure(go.Scatter(
        x=x, y=s, mode="markers", marker=dict(color=s, colorscale="RdBu_r", size=5, opacity=0.6),
    ))
    fig.add_hline(y=0, line_color="gray", line_dash="dash")
    name = label_map.get(feature, feature)
    return _layout(fig, f"How {name} moves churn (SHAP dependence)", name, "SHAP value → churn")


# ----------------------------------------------------------------------
# Model performance
# ----------------------------------------------------------------------
def roc_curve_fig(*models: ModelResult) -> go.Figure:
    fig = go.Figure()
    for m in models:
        fpr, tpr = m.roc
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 name=f"{m.name} (AUC={m.test['roc_auc']:.2f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color="gray"), name="no-skill"))
    return _layout(fig, "ROC curve", "False positive rate", "True positive rate")


def pr_curve_fig(*models: ModelResult, base_rate: float | None = None) -> go.Figure:
    fig = go.Figure()
    for m in models:
        prec, rec = m.pr
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                                 name=f"{m.name} (PR-AUC={m.test['pr_auc']:.2f})"))
    if base_rate is not None:
        fig.add_hline(y=base_rate, line_dash="dash", line_color="gray",
                      annotation_text=f"no-skill = base rate {base_rate:.0%}")
    return _layout(fig, "Precision–Recall curve (lead metric for imbalanced churn)",
                   "Recall", "Precision")


def confusion_heatmap(model: ModelResult) -> go.Figure:
    cm = model.confusion
    labels = ["Retained (0)", "Churned (1)"]
    fig = px.imshow(cm, x=[f"pred {l}" for l in labels], y=[f"true {l}" for l in labels],
                    text_auto=True, color_continuous_scale="Blues", aspect="auto")
    return _layout(fig, f"Confusion matrix @ threshold {model.threshold:.2f} — {model.name}")


def calibration_fig(*models: ModelResult) -> go.Figure:
    fig = go.Figure()
    for m in models:
        prob_true, prob_pred = m.calibration
        if len(prob_true):
            fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode="lines+markers", name=m.name))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color="gray"), name="perfectly calibrated"))
    return _layout(fig, "Calibration curve", "Mean predicted probability", "Observed churn rate")


def lift_gains_fig(model: ModelResult) -> go.Figure:
    g = model.gains
    fig = go.Figure()
    fig.add_trace(go.Bar(x=g["decile"], y=g["lift"], marker_color=ACCENT, name="lift"))
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                  annotation_text="random = 1.0×")
    return _layout(fig, f"Lift by risk decile — {model.name} (decile 1 = highest risk)",
                   "Risk decile", "Lift vs base rate")


def cumulative_gains_fig(model: ModelResult) -> go.Figure:
    g = model.gains
    x = np.concatenate([[0], g["decile"].values / g["decile"].max()])
    y = np.concatenate([[0], g["cum_pct_churners_captured"].values])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=model.name, line_color=ACCENT))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color="gray"), name="random"))
    return _layout(fig, f"Cumulative gains — {model.name}",
                   "Fraction of customers targeted (highest-risk first)",
                   "Fraction of churners captured")


def threshold_tradeoff_fig(model: ModelResult) -> go.Figure:
    """Precision/recall/F1 as the decision threshold moves."""
    from sklearn.metrics import precision_recall_curve

    prec, rec, thr = precision_recall_curve(model.y_test, model.proba)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thr, y=prec[:-1], mode="lines", name="precision", line_color="#9467bd"))
    fig.add_trace(go.Scatter(x=thr, y=rec[:-1], mode="lines", name="recall", line_color="#ff7f0e"))
    fig.add_trace(go.Scatter(x=thr, y=f1[:-1], mode="lines", name="F1", line_color=ACCENT))
    fig.add_vline(x=model.threshold, line_dash="dash", line_color=CHURN_COLOR,
                  annotation_text=f"threshold {model.threshold:.2f}")
    return _layout(fig, f"Precision / recall trade-off — {model.name}", "Threshold", "Score")
