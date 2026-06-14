"""Data profiling + leakage heuristics + target validation.

Leakage heuristics are surfaced as *warnings*, never auto-applied — the analyst
stays in control. Target validation anchors the whole app on the base churn
rate (positive prevalence).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .io import ROLE_DATETIME, ROLE_ID

# names that often encode the *outcome* and would leak if used as features
SUSPICIOUS_LEAK_TOKENS = (
    "churn_date",
    "cancel",
    "cancellation",
    "end_date",
    "termination",
    "terminated",
    "exit_date",
    "reason",
    "deactiv",
    "closed_date",
    "left_date",
)

HIGH_CARDINALITY_THRESHOLD = 50


@dataclass
class ColumnProfile:
    column: str
    role: str
    dtype: str
    pct_missing: float
    n_unique: int
    is_constant: bool
    is_high_cardinality: bool
    sample_values: list[Any]


@dataclass
class LeakageWarning:
    column: str
    kind: str  # "id_like" | "suspicious_name" | "perfect_separation"
    message: str


@dataclass
class ProfileResult:
    columns: list[ColumnProfile]
    leakage: list[LeakageWarning] = field(default_factory=list)
    n_rows: int = 0
    n_duplicate_rows: int = 0

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "column": [c.column for c in self.columns],
                "role": [c.role for c in self.columns],
                "dtype": [c.dtype for c in self.columns],
                "pct_missing": [c.pct_missing for c in self.columns],
                "n_unique": [c.n_unique for c in self.columns],
                "constant": [c.is_constant for c in self.columns],
                "high_cardinality": [c.is_high_cardinality for c in self.columns],
            }
        )


def profile_dataframe(
    df: pd.DataFrame,
    roles: dict[str, str],
    target_col: str | None = None,
    high_cardinality_threshold: int = HIGH_CARDINALITY_THRESHOLD,
) -> ProfileResult:
    """Per-column profile + leakage heuristics."""
    n = len(df)
    cols: list[ColumnProfile] = []
    for col in df.columns:
        s = df[col]
        nunique = int(s.nunique(dropna=True))
        cols.append(
            ColumnProfile(
                column=col,
                role=roles.get(col, "?"),
                dtype=str(s.dtype),
                pct_missing=round(100 * s.isna().mean(), 2),
                n_unique=nunique,
                is_constant=nunique <= 1,
                is_high_cardinality=(
                    roles.get(col) not in (ROLE_DATETIME,)
                    and nunique > high_cardinality_threshold
                ),
                sample_values=list(s.dropna().unique()[:5]),
            )
        )

    leak = detect_leakage(df, roles, target_col)
    return ProfileResult(
        columns=cols,
        leakage=leak,
        n_rows=n,
        n_duplicate_rows=int(df.duplicated().sum()),
    )


def detect_leakage(
    df: pd.DataFrame, roles: dict[str, str], target_col: str | None
) -> list[LeakageWarning]:
    """Heuristic leakage flags. Warnings only — never auto-drop."""
    warnings: list[LeakageWarning] = []
    n = len(df)
    for col in df.columns:
        if col == target_col:
            continue
        s = df[col]
        name = col.lower()

        # 1. id-like (near-unique per row)
        if roles.get(col) == ROLE_ID or (
            n > 0 and s.nunique(dropna=True) >= 0.99 * n
        ):
            warnings.append(
                LeakageWarning(
                    col,
                    "id_like",
                    f"'{col}' is ~unique per row (likely an identifier). Suggest excluding.",
                )
            )

        # 2. suspicious post-outcome name
        if any(tok in name for tok in SUSPICIOUS_LEAK_TOKENS):
            warnings.append(
                LeakageWarning(
                    col,
                    "suspicious_name",
                    f"'{col}' name suggests post-outcome information (possible leakage).",
                )
            )

    # 3. near-perfect univariate separation of the target
    if target_col is not None and target_col in df.columns:
        y = df[target_col]
        if y.nunique(dropna=True) == 2:
            warnings.extend(_separation_warnings(df, roles, target_col))

    return warnings


def _separation_warnings(
    df: pd.DataFrame, roles: dict[str, str], target_col: str
) -> list[LeakageWarning]:
    out: list[LeakageWarning] = []
    y = df[target_col]
    classes = list(y.dropna().unique())
    if len(classes) != 2:
        return out
    pos = classes[0]
    for col in df.columns:
        if col == target_col:
            continue
        role = roles.get(col)
        s = df[col]
        try:
            if role in ("categorical", "id"):
                # any level that almost perfectly predicts one class
                ct = pd.crosstab(s, y)
                if ct.shape[1] == 2 and len(s.dropna()) > 0:
                    purity = ct.max(axis=1) / ct.sum(axis=1).replace(0, np.nan)
                    coverage = ct.sum(axis=1) / ct.values.sum()
                    flagged = (purity > 0.99) & (coverage > 0.05)
                    if flagged.any():
                        out.append(
                            LeakageWarning(
                                col,
                                "perfect_separation",
                                f"'{col}' has level(s) that almost perfectly split the target.",
                            )
                        )
            elif role == "numeric":
                grp = df.dropna(subset=[col]).groupby(y.name)[col]
                if grp.ngroups == 2:
                    lo = grp.max().min()
                    hi = grp.min().max()
                    # disjoint ranges between classes => perfect separation
                    if hi >= lo and grp.min().max() > grp.max().min():
                        out.append(
                            LeakageWarning(
                                col,
                                "perfect_separation",
                                f"'{col}' ranges are disjoint across classes (perfect separation).",
                            )
                        )
        except Exception:  # noqa: BLE001 - profiling must never crash
            continue
    return out


# ----------------------------------------------------------------------
# Target validation
# ----------------------------------------------------------------------
@dataclass
class TargetValidation:
    ok: bool
    is_binary: bool
    n_classes: int
    classes: list[Any]
    suggested_positive: Any | None
    message: str
    needs_threshold: bool = False  # continuous target


def validate_target(df: pd.DataFrame, target_col: str) -> TargetValidation:
    """Check the chosen target is (or can be coerced to) binary."""
    if target_col not in df.columns:
        return TargetValidation(False, False, 0, [], None, "Column not found.")
    s = df[target_col].dropna()
    classes = list(pd.unique(s))
    n_classes = len(classes)

    if n_classes < 2:
        return TargetValidation(
            False, False, n_classes, classes, None,
            "Target has fewer than 2 classes after dropping missing — unusable.",
        )

    if n_classes == 2:
        pos = _guess_positive_class(classes)
        return TargetValidation(
            True, True, 2, classes, pos,
            f"Binary target. Positive ('churned') class suggested: {pos!r}.",
        )

    # numeric continuous -> needs a threshold
    if pd.api.types.is_numeric_dtype(s) and n_classes > 10:
        return TargetValidation(
            False, False, n_classes, [], None,
            "Target looks continuous — define a threshold to binarize it.",
            needs_threshold=True,
        )

    # multiclass categorical -> user must pick the positive class
    return TargetValidation(
        False, False, n_classes, classes, _guess_positive_class(classes),
        f"Target has {n_classes} classes — choose which one means 'churned'.",
    )


def _guess_positive_class(classes: list[Any]) -> Any:
    """Best guess for the positive ('churned') label."""
    truthy = {"yes", "true", "1", "churn", "churned", "y", "t", "left", "cancelled", "canceled"}
    for c in classes:
        if str(c).strip().lower() in truthy:
            return c
    # numeric: 1 is positive
    for c in classes:
        try:
            if float(c) == 1:
                return c
        except (TypeError, ValueError):
            pass
    return classes[-1]


def encode_target(
    series: pd.Series, positive_class: Any
) -> pd.Series:
    """Map the chosen positive class to 1, everything else to 0."""
    return (series == positive_class).astype(int)


def binarize_continuous(series: pd.Series, threshold: float, above_is_positive: bool = True) -> pd.Series:
    """Binarize a continuous target at ``threshold``."""
    s = pd.to_numeric(series, errors="coerce")
    out = (s >= threshold) if above_is_positive else (s <= threshold)
    return out.astype(int)


def base_rate(y: pd.Series) -> float:
    """Positive prevalence — the number that anchors the whole analysis."""
    return float(pd.Series(y).mean())
