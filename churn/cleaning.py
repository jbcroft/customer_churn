"""Data cleansing — user-controllable, with sane defaults, fully logged.

Design boundary (the leakage firewall starts in :mod:`churn.features`):

  * **cleaning.py** does *structural* decisions that do not learn parameters
    from the data distribution — drop duplicate rows, drop constant/leaky/
    high-missingness columns, drop rows, coerce types. These are safe to apply
    to the full dataset before the train/test split.
  * **features.py** does every *learned* transform — imputation, encoding,
    scaling, rare-category bucketing — inside a sklearn Pipeline fit on the
    training fold only. That is where median/mode imputation actually happens.

So the per-column imputation *strategy* is chosen here (and recorded in the
:class:`CleaningConfig`) but executed downstream. Outlier handling defaults to
report-only.

Every action appends to a :class:`~churn.state.TransformationLog`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .io import ROLE_DATETIME, ROLE_ID, ROLE_NUMERIC
from .state import TransformationLog

DEFAULT_MISSING_COL_DROP_THRESHOLD = 0.6
DEFAULT_MAX_CATEGORIES = 20


@dataclass
class CleaningConfig:
    """User-controllable cleaning choices with sane defaults."""

    drop_duplicates: bool = True
    drop_constant: bool = True
    # drop any column whose missing fraction exceeds this
    missing_col_drop_threshold: float = DEFAULT_MISSING_COL_DROP_THRESHOLD
    # explicit user-chosen column drops (ids, flagged leakage, etc.)
    drop_columns: list[str] = field(default_factory=list)
    # drop rows that are missing in any of these columns
    rowdrop_columns: list[str] = field(default_factory=list)
    # imputation strategy — recorded here, executed inside the pipeline
    impute_numeric: str = "median"  # median | mean | constant
    impute_categorical: str = "most_frequent"  # most_frequent | constant("Missing")
    # rare-category bucketing budget for the OHE (executed in the pipeline)
    max_categories: int = DEFAULT_MAX_CATEGORIES
    # outliers: report-only by default
    winsorize: bool = False
    winsor_limits: tuple[float, float] = (0.01, 0.99)


@dataclass
class OutlierReport:
    column: str
    n_outliers_iqr: int
    n_outliers_z: int
    lower: float
    upper: float


@dataclass
class CleaningReport:
    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    dropped_columns: dict[str, str]  # column -> reason
    duplicates_removed: int
    rows_dropped_missing: int
    outliers: list[OutlierReport] = field(default_factory=list)


def detect_outliers(df: pd.DataFrame, numeric_cols: list[str], z_thresh: float = 3.0) -> list[OutlierReport]:
    """IQR + z-score outlier counts per numeric column (report only)."""
    reports: list[OutlierReport] = []
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_iqr = int(((s < lo) | (s > hi)).sum())
        std = s.std(ddof=0)
        if std and std > 0:
            n_z = int((((s - s.mean()).abs() / std) > z_thresh).sum())
        else:
            n_z = 0
        reports.append(OutlierReport(col, n_iqr, n_z, float(lo), float(hi)))
    return reports


def clean_dataframe(
    df: pd.DataFrame,
    roles: dict[str, str],
    config: CleaningConfig,
    target_col: str | None = None,
    log: TransformationLog | None = None,
) -> tuple[pd.DataFrame, CleaningReport, list[str]]:
    """Apply structural cleaning. Returns (clean_df, report, kept_feature_cols).

    Never mutates ``df`` in place.
    """
    log = log if log is not None else TransformationLog()
    out = df.copy()
    rows_before, cols_before = out.shape
    dropped: dict[str, str] = {}

    # 1. explicit user drops (ids, flagged leakage)
    for col in config.drop_columns:
        if col in out.columns and col != target_col:
            out = out.drop(columns=[col])
            dropped[col] = "user-excluded"
            log.add("clean", "drop column", f"{col} (user-excluded)")

    # 2. constant / quasi-constant columns
    if config.drop_constant:
        for col in list(out.columns):
            if col == target_col:
                continue
            nun = out[col].nunique(dropna=True)
            if nun <= 1:
                out = out.drop(columns=[col])
                dropped[col] = "constant"
                log.add("clean", "drop column", f"{col} (constant)")

    # 3. high-missingness columns
    for col in list(out.columns):
        if col == target_col:
            continue
        miss = out[col].isna().mean()
        if miss > config.missing_col_drop_threshold:
            out = out.drop(columns=[col])
            dropped[col] = f"{miss:.0%} missing"
            log.add("clean", "drop column", f"{col} ({miss:.0%} missing)")

    # 4. duplicate rows
    dups_removed = 0
    if config.drop_duplicates:
        before = len(out)
        out = out.drop_duplicates().reset_index(drop=True)
        dups_removed = before - len(out)
        if dups_removed:
            log.add("clean", "drop duplicate rows", f"{dups_removed} exact duplicates")

    # 5. drop rows missing in selected columns
    rows_dropped_missing = 0
    rowdrop = [c for c in config.rowdrop_columns if c in out.columns]
    if rowdrop:
        before = len(out)
        out = out.dropna(subset=rowdrop).reset_index(drop=True)
        rows_dropped_missing = before - len(out)
        if rows_dropped_missing:
            log.add(
                "clean",
                "drop rows with missing",
                f"{rows_dropped_missing} rows missing in {rowdrop}",
            )

    # 6. outlier reporting (+ optional winsorize)
    numeric_cols = [
        c for c in out.columns
        if c != target_col and roles.get(c) == ROLE_NUMERIC
    ]
    outliers = detect_outliers(out, numeric_cols)
    if config.winsorize and numeric_cols:
        lo_q, hi_q = config.winsor_limits
        for col in numeric_cols:
            s = pd.to_numeric(out[col], errors="coerce")
            lo, hi = s.quantile(lo_q), s.quantile(hi_q)
            out[col] = s.clip(lo, hi)
        log.add(
            "clean",
            "winsorize numeric",
            f"capped {numeric_cols} to [{lo_q:.0%}, {hi_q:.0%}] quantiles",
        )

    if config.impute_numeric or config.impute_categorical:
        log.add(
            "clean",
            "imputation scheduled",
            f"numeric={config.impute_numeric}, categorical={config.impute_categorical} "
            "(applied inside the train-only pipeline)",
        )

    report = CleaningReport(
        rows_before=rows_before,
        rows_after=len(out),
        cols_before=cols_before,
        cols_after=out.shape[1],
        dropped_columns=dropped,
        duplicates_removed=dups_removed,
        rows_dropped_missing=rows_dropped_missing,
        outliers=outliers,
    )

    kept_features = [
        c for c in out.columns
        if c != target_col and roles.get(c) != ROLE_ID
    ]
    return out, report, kept_features


def suggested_drops(profile_leakage: list[Any], roles: dict[str, str]) -> list[str]:
    """Columns we *suggest* (not force) dropping: ids + leakage flags."""
    cols = {w.column for w in profile_leakage if w.kind in ("id_like", "suspicious_name")}
    cols |= {c for c, r in roles.items() if r == ROLE_ID}
    return sorted(cols)
