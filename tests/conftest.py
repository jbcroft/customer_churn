"""Shared fixtures: a synthetic churn dataset with a known signal."""

from __future__ import annotations

import pandas as pd
import pytest

from churn import cleaning, features, io, profiling
from sample_data.make_synthetic import make_synthetic


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    return make_synthetic(n=2000, seed=42, messy=True)


@pytest.fixture(scope="session")
def clean_synthetic() -> pd.DataFrame:
    """A tidy (no-mess) version for tests that don't exercise cleaning."""
    return make_synthetic(n=2000, seed=7, messy=False)


@pytest.fixture()
def ingested(synthetic_df, tmp_path):
    """Round-trip through the real reader to get roles like the app does."""
    p = tmp_path / "d.csv"
    synthetic_df.to_csv(p, index=False)
    info = io.read_table(str(p), filename="d.csv")
    return info


@pytest.fixture()
def prepared(ingested):
    """Cleaned df + roles + feature spec + encoded target, ready for modeling."""
    df, roles = ingested.df, ingested.roles
    y = profiling.encode_target(df["churn"], "Yes")
    cfg = cleaning.CleaningConfig(drop_columns=["customer_id", "account_note", "signup_date"])
    clean_df, report, feats = cleaning.clean_dataframe(df, roles, cfg, target_col="churn")
    y = y.loc[clean_df.index].reset_index(drop=True)
    clean_df = clean_df.reset_index(drop=True)
    spec = features.build_feature_spec(roles, feats, cfg)
    return {"df": clean_df, "y": y, "roles": roles, "spec": spec, "features": feats}
