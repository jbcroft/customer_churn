from __future__ import annotations

import numpy as np
import pandas as pd

from churn import features
from churn.cleaning import CleaningConfig


def test_recency_reference_is_train_only_no_leakage():
    """RecencyTransformer must reference the TRAIN max date, not the full data."""
    train = pd.DataFrame({"d": pd.to_datetime(["2020-01-01", "2020-01-11"])})
    future = pd.DataFrame({"d": pd.to_datetime(["2020-02-01"])})  # later than any train date
    rt = features.RecencyTransformer(["d"]).fit(train)
    assert rt.reference_["d"] == pd.Timestamp("2020-01-11")          # train max, not future
    out = rt.transform(future)
    # days since a train-anchored reference -> negative for a later date (no peeking)
    assert out["d__days_since"].iloc[0] < 0


def test_onehot_and_recency_shapes(prepared):
    spec = prepared["spec"]
    pipe = features.build_preprocessor(spec, scale=True)
    X = prepared["df"][spec.all_input_cols]
    Z = pipe.fit_transform(X, prepared["y"])
    # every datetime column becomes exactly one recency numeric
    for c in spec.datetime_cols:
        assert f"num__{c}{features.RECENCY_SUFFIX}" in Z.columns
    # categorical columns expand to >= their cardinality-1 one-hot columns
    assert any(col.startswith("cat__contract_") for col in Z.columns)


def test_scaling_only_in_lr_view(prepared):
    spec = prepared["spec"]
    X = prepared["df"][spec.all_input_cols]
    scaled = features.build_preprocessor(spec, scale=True).fit_transform(X, prepared["y"])
    raw = features.build_preprocessor(spec, scale=False).fit_transform(X, prepared["y"])
    num_col = f"num__{spec.numeric_cols[0]}"
    # scaled numeric ~ mean 0 / unit std; unscaled keeps original magnitude
    assert abs(scaled[num_col].mean()) < 1e-6
    assert raw[num_col].std() > scaled[num_col].std() or raw[num_col].abs().max() > 3


def test_scaler_learns_train_stats_not_test():
    """The leakage firewall: scaler mean must equal the TRAIN mean only."""
    from sklearn.model_selection import train_test_split

    df = pd.DataFrame({"x": np.arange(100.0), "c": ["a"] * 100})
    spec = features.FeatureSpec(["x"], ["c"], [], CleaningConfig())
    y = (df["x"] > 50).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(df, y, test_size=0.5, shuffle=False)
    pipe = features.build_preprocessor(spec, scale=True).fit(Xtr, ytr)
    learned_mean = pipe.named_steps["prep"].named_transformers_["num"].named_steps["scale"].mean_[0]
    assert np.isclose(learned_mean, Xtr["x"].mean())
    assert not np.isclose(learned_mean, df["x"].mean())  # would be 49.5 if it leaked


def test_feature_name_map_traces_to_parent(prepared):
    spec = prepared["spec"]
    pipe = features.build_preprocessor(spec, scale=True)
    pipe.fit(prepared["df"][spec.all_input_cols], prepared["y"])
    enc, parent, label = features.feature_name_map(pipe.named_steps["prep"], spec)
    contract_cols = [e for e in enc if "contract_" in e]
    assert contract_cols
    assert all(parent[c] == "contract" for c in contract_cols)
    assert any("Contract =" in label[c] for c in contract_cols)


def test_vif_flags_perfect_collinearity():
    df = pd.DataFrame({"a": np.arange(50.0), "b": np.arange(50.0) * 2})  # b = 2a
    vif = features.compute_vif(df)
    assert vif["high_collinearity"].any()
