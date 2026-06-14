"""Univariate, model-free statistical screening of association with the target.

For each feature vs the binary target:
  * numeric    -> point-biserial correlation + Mann-Whitney U (non-parametric);
                  effect size = rank-biserial correlation.
  * categorical -> chi-square test of independence; effect size = Cramer's V.
  * everything  -> mutual information (monotonic-agnostic ranker).

P-values are corrected for multiple comparisons with Benjamini-Hochberg FDR.

This gives usable insight *before* any ML, and serves as a cross-check that the
model-based drivers (explain.py) point the same direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif

from .io import ROLE_CATEGORICAL, ROLE_DATETIME, ROLE_NUMERIC


@dataclass
class StatResult:
    feature: str
    role: str
    test: str
    statistic: float
    p_value: float
    effect_size: float
    effect_name: str
    direction: str  # "↑ churn" | "↓ churn" | "—"
    n: int


def _direction_numeric(x: pd.Series, y: pd.Series) -> str:
    """Sign of association: do higher values of x go with churn=1?"""
    churned = x[y == 1].mean()
    retained = x[y == 0].mean()
    if np.isnan(churned) or np.isnan(retained) or churned == retained:
        return "—"
    return "↑ churn" if churned > retained else "↓ churn"


def _cramers_v(confusion: np.ndarray) -> float:
    chi2 = stats.chi2_contingency(confusion, correction=False)[0]
    n = confusion.sum()
    if n == 0:
        return 0.0
    phi2 = chi2 / n
    r, k = confusion.shape
    denom = min(k - 1, r - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0


def _rank_biserial(u: float, n1: int, n2: int) -> float:
    """Rank-biserial effect size from a Mann-Whitney U statistic."""
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(1.0 - (2.0 * u) / (n1 * n2))


def analyze_numeric(x: pd.Series, y: pd.Series, name: str) -> StatResult:
    mask = x.notna() & y.notna()
    xv, yv = x[mask].astype(float), y[mask].astype(int)
    n = int(mask.sum())
    # point-biserial == Pearson r between binary y and numeric x
    if xv.nunique() <= 1 or yv.nunique() <= 1:
        return StatResult(name, ROLE_NUMERIC, "point-biserial", 0.0, 1.0, 0.0,
                          "rank-biserial", "—", n)
    r_pb, _ = stats.pointbiserialr(yv, xv)
    g1, g0 = xv[yv == 1], xv[yv == 0]
    try:
        u, p = stats.mannwhitneyu(g1, g0, alternative="two-sided")
        rb = _rank_biserial(u, len(g1), len(g0))
    except ValueError:
        u, p, rb = 0.0, 1.0, 0.0
    return StatResult(
        feature=name, role=ROLE_NUMERIC, test="Mann-Whitney U",
        statistic=float(u), p_value=float(p),
        effect_size=float(abs(rb)), effect_name="rank-biserial",
        direction=_direction_numeric(xv, yv), n=n,
    )


def analyze_categorical(x: pd.Series, y: pd.Series, name: str) -> StatResult:
    mask = x.notna() & y.notna()
    xv, yv = x[mask].astype(str), y[mask].astype(int)
    n = int(mask.sum())
    ct = pd.crosstab(xv, yv)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return StatResult(name, ROLE_CATEGORICAL, "chi-square", 0.0, 1.0, 0.0,
                          "Cramér's V", "—", n)
    chi2, p, _, _ = stats.chi2_contingency(ct, correction=False)
    v = _cramers_v(ct.values)
    # direction: which level(s) over-index on churn vs base rate
    base = yv.mean()
    rate_by_level = xv.groupby(xv).apply(lambda idx: yv[idx.index].mean())
    direction = "—"
    if not rate_by_level.empty:
        spread = rate_by_level.max() - rate_by_level.min()
        direction = "varies by level" if spread > 0.02 else "—"
    return StatResult(
        feature=name, role=ROLE_CATEGORICAL, test="chi-square",
        statistic=float(chi2), p_value=float(p),
        effect_size=float(v), effect_name="Cramér's V",
        direction=direction, n=n,
    )


def mutual_information(
    df: pd.DataFrame, y: pd.Series, numeric: list[str], categorical: list[str]
) -> dict[str, float]:
    """Mutual information of each feature with the target (uniform ranker)."""
    use = [c for c in numeric + categorical if c in df.columns]
    if not use:
        return {}
    X = df[use].copy()
    discrete_mask = []
    for c in use:
        if c in categorical:
            X[c] = X[c].astype("category").cat.codes
            discrete_mask.append(True)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
            discrete_mask.append(False)
    mask = X.notna().all(axis=1) & y.notna()
    if mask.sum() < 3:
        return {c: 0.0 for c in use}
    mi = mutual_info_classif(
        X[mask], y[mask].astype(int),
        discrete_features=discrete_mask, random_state=42,
    )
    return dict(zip(use, [float(m) for m in mi]))


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR-adjusted p-values (q-values)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(1, n + 1))
    # enforce monotonicity from the largest p downward
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out.tolist()


def run_univariate(
    df: pd.DataFrame,
    y: pd.Series,
    roles: dict[str, str],
    feature_cols: list[str],
) -> pd.DataFrame:
    """Full univariate screen -> tidy table with raw + FDR-adjusted p-values.

    Datetime features are screened via their "days since most recent" recency.
    """
    from .features import RecencyTransformer, RECENCY_SUFFIX

    y = pd.Series(y).astype(int).reset_index(drop=True)
    work = df.reset_index(drop=True).copy()

    numeric = [c for c in feature_cols if roles.get(c) == ROLE_NUMERIC]
    categorical = [c for c in feature_cols if roles.get(c) == ROLE_CATEGORICAL]
    datetime = [c for c in feature_cols if roles.get(c) == ROLE_DATETIME]

    # convert datetimes to recency numerics for screening
    if datetime:
        rec = RecencyTransformer(datetime).fit(work[feature_cols])
        rec_df = rec.transform(work[feature_cols])
        for c in datetime:
            new = f"{c}{RECENCY_SUFFIX}"
            if new in rec_df.columns:
                work[new] = rec_df[new]
                numeric.append(new)

    results: list[StatResult] = []
    for col in numeric:
        results.append(analyze_numeric(work[col], y, col))
    for col in categorical:
        results.append(analyze_categorical(work[col], y, col))

    mi = mutual_information(work, y, numeric, categorical)

    table = pd.DataFrame([r.__dict__ for r in results])
    if table.empty:
        return table
    table["mutual_info"] = table["feature"].map(mi).fillna(0.0)
    table["p_fdr"] = benjamini_hochberg(table["p_value"].tolist())
    table["significant_fdr"] = table["p_fdr"] < 0.05
    table = table.sort_values(
        ["significant_fdr", "effect_size", "mutual_info"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    # friendly column order
    cols = [
        "feature", "role", "test", "statistic", "p_value", "p_fdr",
        "significant_fdr", "effect_size", "effect_name", "mutual_info",
        "direction", "n",
    ]
    return table[[c for c in cols if c in table.columns]]
