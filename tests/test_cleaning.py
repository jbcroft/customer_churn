from __future__ import annotations

import numpy as np
import pandas as pd

from churn import cleaning
from churn.state import TransformationLog


def test_drops_duplicates_constant_and_user_columns(ingested):
    df, roles = ingested.df, ingested.roles
    cfg = cleaning.CleaningConfig(drop_columns=["customer_id"])
    out, rep, feats = cleaning.clean_dataframe(df, roles, cfg, target_col="churn")
    # synthetic injects 2 exact dup rows + a constant column "data_source"
    assert rep.duplicates_removed >= 1
    assert "data_source" in rep.dropped_columns           # constant
    assert "customer_id" in rep.dropped_columns           # user-excluded
    assert "customer_id" not in feats and "churn" not in feats


def test_high_missingness_column_dropped():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "mostly_missing": [np.nan, np.nan, np.nan, 1.0],
        "churn": [0, 1, 0, 1],
    })
    roles = {"a": "numeric", "mostly_missing": "numeric", "churn": "categorical"}
    cfg = cleaning.CleaningConfig(missing_col_drop_threshold=0.5)
    out, rep, feats = cleaning.clean_dataframe(df, roles, cfg, target_col="churn")
    assert "mostly_missing" in rep.dropped_columns


def test_target_never_dropped_even_if_constant():
    df = pd.DataFrame({"x": [1, 2, 3], "churn": [1, 1, 1]})
    roles = {"x": "numeric", "churn": "categorical"}
    out, rep, feats = cleaning.clean_dataframe(df, roles, cleaning.CleaningConfig(), target_col="churn")
    assert "churn" in out.columns


def test_winsorize_caps_outliers(ingested):
    df, roles = ingested.df, ingested.roles
    cfg = cleaning.CleaningConfig(winsorize=True, drop_columns=["customer_id"])
    out, rep, feats = cleaning.clean_dataframe(df, roles, cfg, target_col="churn")
    # the injected 900-1500 price outliers must be capped well below 900
    assert out["monthly_charges"].max() < 900


def test_log_is_populated(ingested):
    df, roles = ingested.df, ingested.roles
    log = TransformationLog()
    cfg = cleaning.CleaningConfig(drop_columns=["customer_id"])
    cleaning.clean_dataframe(df, roles, cfg, target_col="churn", log=log)
    assert len(log) >= 2  # at least a drop + the imputation-scheduled note


def test_does_not_mutate_input(ingested):
    df, roles = ingested.df, ingested.roles
    before = df.copy()
    cleaning.clean_dataframe(df, roles, cleaning.CleaningConfig(drop_columns=["customer_id"]), "churn")
    pd.testing.assert_frame_equal(df, before)
