from __future__ import annotations

import numpy as np
import pandas as pd

from churn import stats_drivers


def test_correct_test_chosen_per_dtype(prepared):
    tab = stats_drivers.run_univariate(prepared["df"], prepared["y"], prepared["roles"], prepared["features"])
    by_feature = tab.set_index("feature")["test"].to_dict()
    assert by_feature["tenure_months"] == "Mann-Whitney U"   # numeric
    assert by_feature["contract"] == "chi-square"            # categorical


def test_known_signal_surfaces_with_correct_direction(prepared):
    tab = stats_drivers.run_univariate(prepared["df"], prepared["y"], prepared["roles"], prepared["features"])
    d = tab.set_index("feature")
    # ground truth: more tickets -> churn up; longer tenure -> churn down
    assert d.loc["support_tickets", "direction"] == "↑ churn"
    assert d.loc["tenure_months", "direction"] == "↓ churn"
    assert d.loc["support_tickets", "significant_fdr"]
    assert d.loc["tenure_months", "significant_fdr"]


def test_null_feature_not_significant(prepared):
    """region has no engineered effect -> should not be FDR-significant."""
    tab = stats_drivers.run_univariate(prepared["df"], prepared["y"], prepared["roles"], prepared["features"])
    d = tab.set_index("feature")
    assert not bool(d.loc["region", "significant_fdr"])


def test_benjamini_hochberg_monotone_and_bounded():
    p = [0.001, 0.01, 0.2, 0.04, 0.5]
    q = stats_drivers.benjamini_hochberg(p)
    assert all(0 <= v <= 1 for v in q)
    # q-values are >= raw p-values (FDR adjustment never shrinks p below raw)
    assert all(qi >= pi - 1e-9 for qi, pi in zip(q, p))


def test_cramers_v_bounded():
    ct = np.array([[10, 20], [30, 5]])
    v = stats_drivers._cramers_v(ct)
    assert 0 <= v <= 1
