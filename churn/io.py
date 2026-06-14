"""Upload + read + type inference for a single dataset.

Pure functions (no Streamlit) so they can be unit-tested. ``app.py`` wires the
``st.file_uploader`` to :func:`read_table` and renders :func:`schema_summary`.

Column *roles* (not just dtypes) are the unit the rest of the pipeline speaks:

    numeric | categorical | datetime | id  (id = ignore as a feature)
"""

from __future__ import annotations

import io as _io
from dataclasses import dataclass
from typing import IO, Any

import numpy as np
import pandas as pd

ROLE_NUMERIC = "numeric"
ROLE_CATEGORICAL = "categorical"
ROLE_DATETIME = "datetime"
ROLE_ID = "id"
ROLES = (ROLE_NUMERIC, ROLE_CATEGORICAL, ROLE_DATETIME, ROLE_ID)

MAX_MB_DEFAULT = 200


class DataReadError(ValueError):
    """Raised for unreadable / empty / malformed uploads (caught by the UI)."""


@dataclass
class TableInfo:
    df: pd.DataFrame
    n_rows: int
    n_cols: int
    memory_mb: float
    roles: dict[str, str]
    dtypes: dict[str, str]
    sheet: str | None = None


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------
def _sniff_csv(raw: bytes) -> tuple[str, str]:
    """Return (encoding, delimiter) sniffed from raw CSV bytes."""
    encoding = "utf-8"
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            raw.decode(enc)
            encoding = enc
            break
        except UnicodeDecodeError:
            continue
    # delimiter sniff on the first few KB
    import csv

    sample = raw[:8192].decode(encoding, errors="replace")
    delimiter = ","
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        # fall back to the most frequent candidate in the header line
        header = sample.splitlines()[0] if sample.splitlines() else ""
        delimiter = max(",;\t|", key=lambda d: header.count(d))
    return encoding, delimiter


def list_excel_sheets(source: str | bytes | IO[bytes]) -> list[str]:
    buf = _as_buffer(source)
    xl = pd.ExcelFile(buf)
    return list(xl.sheet_names)


def _as_buffer(source: str | bytes | IO[bytes]) -> Any:
    if isinstance(source, bytes):
        return _io.BytesIO(source)
    return source


def read_table(
    source: str | bytes | IO[bytes],
    *,
    filename: str,
    sheet: str | None = None,
    max_mb: int = MAX_MB_DEFAULT,
) -> TableInfo:
    """Robustly read a CSV/XLSX into a :class:`TableInfo`.

    Raises :class:`DataReadError` (never an opaque stack trace) on failure.
    """
    name = (filename or "").lower()
    raw: bytes | None = None
    if isinstance(source, bytes):
        raw = source
    elif hasattr(source, "read"):
        raw = source.read()
        try:
            source.seek(0)
        except Exception:  # pragma: no cover - non-seekable
            pass

    if raw is not None and len(raw) == 0:
        raise DataReadError("The uploaded file is empty.")
    if raw is not None and len(raw) > max_mb * 1024 * 1024:
        raise DataReadError(
            f"File is {len(raw) / 1e6:.0f} MB, over the {max_mb} MB limit. "
            "Subsample it or raise the limit."
        )

    try:
        if name.endswith((".xlsx", ".xls")):
            buf = _as_buffer(raw if raw is not None else source)
            if sheet is None:
                sheet = list_excel_sheets(raw if raw is not None else source)[0]
            df = pd.read_excel(buf, sheet_name=sheet)
        elif name.endswith(".csv") or name.endswith(".txt"):
            if raw is None:
                with open(source, "rb") as fh:  # type: ignore[arg-type]
                    raw = fh.read()
            encoding, delimiter = _sniff_csv(raw)
            df = pd.read_csv(_io.BytesIO(raw), encoding=encoding, sep=delimiter)
        else:
            raise DataReadError(
                f"Unsupported file type '{filename}'. Use .csv or .xlsx."
            )
    except DataReadError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface a clean message
        raise DataReadError(f"Could not parse the file: {exc}") from exc

    if df.shape[1] == 0:
        raise DataReadError("No columns were detected — check the delimiter.")
    if df.shape[0] == 0:
        raise DataReadError("The file has headers but no data rows.")

    # tidy column names (strip whitespace, dedupe)
    df.columns = _dedupe_columns([str(c).strip() for c in df.columns])

    roles = infer_roles(df)
    df = coerce_datetimes(df, roles)

    return TableInfo(
        df=df,
        n_rows=len(df),
        n_cols=df.shape[1],
        memory_mb=float(df.memory_usage(deep=True).sum()) / 1e6,
        roles=roles,
        dtypes={c: str(t) for c, t in df.dtypes.items()},
        sheet=sheet,
    )


def _dedupe_columns(cols: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            out.append(f"{c}.{seen[c]}")
        else:
            seen[c] = 0
            out.append(c)
    return out


# ----------------------------------------------------------------------
# Type / role inference
# ----------------------------------------------------------------------
def _looks_like_datetime(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if not (s.dtype == object or pd.api.types.is_string_dtype(s)):
        return False
    sample = s.dropna().astype(str).head(50)
    if sample.empty:
        return False
    # avoid treating pure ints/floats as dates
    if sample.str.fullmatch(r"-?\d+(\.\d+)?").mean() > 0.8:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() > 0.8


def infer_roles(df: pd.DataFrame) -> dict[str, str]:
    """Infer a *role* per column: numeric / categorical / datetime / id."""
    n = len(df)
    roles: dict[str, str] = {}
    for col in df.columns:
        s = df[col]
        nunique = s.nunique(dropna=True)
        name = col.lower()

        if _looks_like_datetime(s):
            roles[col] = ROLE_DATETIME
            continue

        is_numeric = pd.api.types.is_numeric_dtype(s)
        # id heuristic: near-unique, or name says id
        near_unique = n > 0 and nunique >= 0.95 * s.notna().sum() and nunique > 20
        name_is_id = name in {"id"} or name.endswith("_id") or name.endswith("id") and nunique > 20
        if (near_unique and not is_numeric) or name_is_id:
            roles[col] = ROLE_ID
            continue

        if is_numeric:
            # low-cardinality integers that look like codes -> categorical
            if nunique <= 2:
                roles[col] = ROLE_CATEGORICAL
            else:
                roles[col] = ROLE_NUMERIC
        else:
            roles[col] = ROLE_CATEGORICAL
    return roles


def coerce_datetimes(df: pd.DataFrame, roles: dict[str, str]) -> pd.DataFrame:
    """Coerce columns inferred as datetime to actual datetime dtype."""
    df = df.copy()
    for col, role in roles.items():
        if role == ROLE_DATETIME and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
    return df


def apply_role_overrides(
    df: pd.DataFrame, roles: dict[str, str], overrides: dict[str, str]
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Apply user role overrides, coercing dtypes to match the new roles."""
    roles = {**roles, **{k: v for k, v in overrides.items() if v in ROLES}}
    df = df.copy()
    for col, role in overrides.items():
        if col not in df.columns:
            continue
        if role == ROLE_DATETIME:
            df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
        elif role == ROLE_NUMERIC:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif role in (ROLE_CATEGORICAL, ROLE_ID):
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype(str)
    return df, roles


def schema_summary(info: TableInfo) -> pd.DataFrame:
    """A compact per-column schema table for display."""
    df = info.df
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append(
            {
                "column": col,
                "role": info.roles.get(col, "?"),
                "dtype": str(s.dtype),
                "n_unique": int(s.nunique(dropna=True)),
                "pct_missing": round(100 * s.isna().mean(), 1),
                "sample": ", ".join(map(str, s.dropna().unique()[:3])),
            }
        )
    return pd.DataFrame(rows)


def feature_columns(roles: dict[str, str], target_col: str | None) -> list[str]:
    """Columns usable as features: everything that isn't an id or the target."""
    return [
        c
        for c, r in roles.items()
        if r != ROLE_ID and c != target_col
    ]
