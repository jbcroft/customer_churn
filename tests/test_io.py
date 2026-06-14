from __future__ import annotations

import pandas as pd
import pytest

from churn import io


def test_reads_csv_with_role_inference(ingested):
    info = ingested
    assert info.n_rows > 0 and info.n_cols > 0
    assert info.roles["customer_id"] == io.ROLE_ID
    assert info.roles["signup_date"] == io.ROLE_DATETIME
    assert info.roles["tenure_months"] == io.ROLE_NUMERIC
    assert info.roles["contract"] == io.ROLE_CATEGORICAL


def test_reads_xlsx_roundtrip(synthetic_df, tmp_path):
    """Excel support must work end to end (regression for the openpyxl dep)."""
    p = tmp_path / "d.xlsx"
    synthetic_df.head(200).to_excel(p, index=False, sheet_name="customers")
    assert io.list_excel_sheets(str(p)) == ["customers"]
    info = io.read_table(str(p), filename="d.xlsx", sheet="customers")
    assert info.n_rows == 200
    assert "churn" in info.df.columns


def test_empty_file_raises_clean_error(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("")
    with pytest.raises(io.DataReadError):
        io.read_table(str(p), filename="empty.csv")


def test_unsupported_extension_raises(tmp_path):
    p = tmp_path / "data.parquet"
    p.write_bytes(b"not really parquet")
    with pytest.raises(io.DataReadError):
        io.read_table(str(p), filename="data.parquet")
