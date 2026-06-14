"""Feature transformations as a leakage-safe sklearn Pipeline.

Everything that *learns* from the data — recency reference dates, imputation
fills, one-hot vocabularies, scaling statistics, rare-category buckets — lives
inside a ``Pipeline`` that is fit on the training fold ONLY. The identical
fitted transform is then applied to the test fold and any future data. This is
the leakage firewall.

Two views are produced from the same column spec:
  * ``scale=True``  — numeric standardized; used by Logistic Regression so its
    coefficients (and odds ratios) are comparable across features.
  * ``scale=False`` — numeric left raw; used by the tree/boosting model.

A feature-name map traces every encoded column back to its original field
(e.g. ``cat__plan_Premium`` -> parent ``plan``, label "Plan = Premium").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .cleaning import CleaningConfig
from .io import ROLE_CATEGORICAL, ROLE_DATETIME, ROLE_NUMERIC

RECENCY_SUFFIX = "__days_since"


# ----------------------------------------------------------------------
# Custom transformer: datetime -> "days since reference"
# ----------------------------------------------------------------------
class RecencyTransformer(BaseEstimator, TransformerMixin):
    """Convert datetime columns to numeric "days since reference".

    The reference per column defaults to that column's max date **as seen in
    fit** (i.e. the training fold) so no test-set information leaks in. The raw
    datetime column is dropped after deriving recency.
    """

    def __init__(self, datetime_cols: list[str] | None = None,
                 reference_dates: dict[str, Any] | None = None):
        # sklearn clone contract: store params verbatim, normalize lazily.
        self.datetime_cols = datetime_cols
        self.reference_dates = reference_dates

    @property
    def _dt_cols(self) -> list[str]:
        return list(self.datetime_cols or [])

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = list(X.columns)
        self.reference_: dict[str, Any] = {}
        overrides = self.reference_dates or {}
        for c in self._dt_cols:
            if c in X.columns:
                col = pd.to_datetime(X[c], errors="coerce")
                ref = overrides.get(c, None)
                self.reference_[c] = pd.Timestamp(ref) if ref is not None else col.max()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in self._dt_cols:
            if c in X.columns:
                col = pd.to_datetime(X[c], errors="coerce")
                ref = self.reference_.get(c)
                if ref is None or pd.isna(ref):
                    X[f"{c}{RECENCY_SUFFIX}"] = np.nan
                else:
                    X[f"{c}{RECENCY_SUFFIX}"] = (ref - col).dt.days.astype("float64")
                X = X.drop(columns=[c])
        return X

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        kept = [c for c in self.feature_names_in_ if c not in self._dt_cols]
        derived = [f"{c}{RECENCY_SUFFIX}" for c in self._dt_cols
                   if c in self.feature_names_in_]
        return np.array(kept + derived, dtype=object)


# ----------------------------------------------------------------------
# Feature spec + preprocessor construction
# ----------------------------------------------------------------------
@dataclass
class FeatureSpec:
    numeric_cols: list[str]
    categorical_cols: list[str]
    datetime_cols: list[str]
    config: CleaningConfig = field(default_factory=CleaningConfig)

    @property
    def numeric_after_recency(self) -> list[str]:
        return self.numeric_cols + [f"{c}{RECENCY_SUFFIX}" for c in self.datetime_cols]

    @property
    def all_input_cols(self) -> list[str]:
        return self.numeric_cols + self.categorical_cols + self.datetime_cols


def build_feature_spec(
    roles: dict[str, str], feature_cols: list[str], config: CleaningConfig
) -> FeatureSpec:
    numeric = [c for c in feature_cols if roles.get(c) == ROLE_NUMERIC]
    categorical = [c for c in feature_cols if roles.get(c) == ROLE_CATEGORICAL]
    datetime = [c for c in feature_cols if roles.get(c) == ROLE_DATETIME]
    return FeatureSpec(numeric, categorical, datetime, config)


def preprocessor_steps(spec: FeatureSpec, *, scale: bool) -> list[tuple[str, Any]]:
    """The two leakage-safe preprocessing steps, *flat* (not nested in a Pipeline).

    Returned as ``[('recency', ...), ('prep', ColumnTransformer)]`` so they can be
    composed into either a sklearn ``Pipeline`` or an imblearn ``Pipeline`` (the
    latter forbids nested Pipelines as intermediate steps).
    """
    cfg = spec.config

    num_steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(
            strategy=cfg.impute_numeric if cfg.impute_numeric in ("median", "mean") else "constant",
            fill_value=0.0 if cfg.impute_numeric == "constant" else None,
        )),
    ]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    if cfg.impute_categorical == "constant":
        cat_impute = SimpleImputer(strategy="constant", fill_value="Missing")
    else:
        cat_impute = SimpleImputer(strategy="most_frequent")
    cat_pipe = Pipeline([
        ("impute", cat_impute),
        ("ohe", OneHotEncoder(
            handle_unknown="ignore",
            max_categories=cfg.max_categories,
            min_frequency=0.01,
            sparse_output=False,
        )),
    ])

    transformers = []
    if spec.numeric_after_recency:
        transformers.append(("num", num_pipe, spec.numeric_after_recency))
    if spec.categorical_cols:
        transformers.append(("cat", cat_pipe, spec.categorical_cols))

    ct = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)
    return [("recency", RecencyTransformer(spec.datetime_cols)), ("prep", ct)]


def build_preprocessor(spec: FeatureSpec, *, scale: bool) -> Pipeline:
    """Leakage-safe preprocessing as a standalone sklearn Pipeline.

    ``scale`` standardizes numeric features (for the LR path). Tree models pass
    ``scale=False``.
    """
    pipe = Pipeline(preprocessor_steps(spec, scale=scale))
    pipe.set_output(transform="pandas")
    return pipe


# ----------------------------------------------------------------------
# Feature-name reconciliation (encoded -> parent field + human label)
# ----------------------------------------------------------------------
def _strip_prefix(name: str) -> str:
    # ColumnTransformer prefixes with "num__" / "cat__"
    return re.sub(r"^(num|cat)__", "", name)


def parent_feature(encoded_name: str, categorical_cols: list[str] | None = None) -> str:
    """Map an encoded column back to its original field name.

    One-hot columns ``<parent>_<level>`` collapse to ``<parent>``; recency
    columns ``<col>__days_since`` collapse to ``<col>``.
    """
    base = _strip_prefix(encoded_name)
    if base.endswith(RECENCY_SUFFIX):
        return base[: -len(RECENCY_SUFFIX)]
    for parent in sorted(categorical_cols or [], key=len, reverse=True):
        if base == parent or base.startswith(parent + "_"):
            return parent
    return base


def humanize(encoded_name: str, categorical_cols: list[str]) -> str:
    """Human-readable label for an encoded feature, for tables and the report."""
    base = _strip_prefix(encoded_name)
    if base.endswith(RECENCY_SUFFIX):
        field = base[: -len(RECENCY_SUFFIX)].replace("_", " ")
        return f"Days since {field}"
    # one-hot column looks like "<parent>_<level>" for a categorical parent
    for parent in sorted(categorical_cols, key=len, reverse=True):
        if base.startswith(parent + "_"):
            level = base[len(parent) + 1:]
            return f"{parent.replace('_', ' ').title()} = {level}"
    return base.replace("_", " ")


def feature_name_map(
    fitted_prep: Any, spec: FeatureSpec
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Return (encoded_names, parent_map, label_map) for a fitted ColumnTransformer."""
    encoded = list(fitted_prep.get_feature_names_out())
    parent = {e: parent_feature(e, spec.categorical_cols) for e in encoded}
    label = {e: humanize(e, spec.categorical_cols) for e in encoded}
    return encoded, parent, label


# ----------------------------------------------------------------------
# Multicollinearity (VIF) + correlation
# ----------------------------------------------------------------------
def compute_vif(X_numeric: pd.DataFrame) -> pd.DataFrame:
    """Variance Inflation Factor per numeric feature. Warn on VIF > 10."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X = X_numeric.replace([np.inf, -np.inf], np.nan).dropna()
    cols = [c for c in X.columns if X[c].nunique() > 1]
    X = X[cols]
    if X.shape[1] < 2 or X.shape[0] < 3:
        return pd.DataFrame({"feature": cols, "VIF": [np.nan] * len(cols)})
    Xv = X.values.astype(float)
    rows = []
    for i, col in enumerate(cols):
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                vif = variance_inflation_factor(Xv, i)  # inf on perfect collinearity, by design
        except Exception:  # noqa: BLE001
            vif = np.nan
        rows.append({"feature": col, "VIF": round(float(vif), 2)})
    out = pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)
    out["high_collinearity"] = out["VIF"] > 10
    return out


def numeric_design_matrix(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Numeric matrix (incl. recency) used for VIF + correlation heatmap."""
    rec = RecencyTransformer(spec.datetime_cols).fit(df[spec.all_input_cols])
    transformed = rec.transform(df[spec.all_input_cols])
    cols = [c for c in spec.numeric_after_recency if c in transformed.columns]
    return transformed[cols].apply(pd.to_numeric, errors="coerce")
