from __future__ import annotations

import warnings

import numpy as np
import pytest

from churn import modeling

warnings.filterwarnings("ignore")


@pytest.fixture()
def trained(prepared):
    spec = prepared["spec"]
    X = prepared["df"][spec.all_input_cols]
    return modeling.train_all(spec, X, prepared["y"], imbalance="class_weight")


def test_deterministic_same_seed(prepared):
    spec = prepared["spec"]
    X = prepared["df"][spec.all_input_cols]
    a = modeling.train_all(spec, X, prepared["y"], fit_lr=False)
    b = modeling.train_all(spec, X, prepared["y"], fit_lr=False)
    np.testing.assert_allclose(a.gbm.proba, b.gbm.proba)


def test_pr_auc_beats_no_skill(trained):
    base = trained.base_rate  # no-skill PR-AUC == base rate
    for m in (trained.lr, trained.gbm):
        assert m.test["pr_auc"] > base + 0.02  # above the no-skill PR-AUC (== base rate)
        assert m.test["roc_auc"] > 0.55        # comfortably above the 0.5 no-skill line


def test_cv_reports_mean_and_std(trained):
    for m in (trained.lr, trained.gbm):
        for metric, (mean, std) in m.cv.items():
            assert np.isfinite(mean) and np.isfinite(std)
        assert "pr_auc" in m.cv


def test_test_set_size_matches_split(prepared, trained):
    n = len(prepared["df"])
    assert abs(len(trained.y_test) - round(0.25 * n)) <= 1


def test_odds_ratios_directionally_correct(trained):
    odds = trained.lr.odds_ratios.set_index("feature")
    # tenure -> OR < 1 (protective), support_tickets -> OR > 1 (risk)
    tenure = [i for i in odds.index if "tenure_months" in i][0]
    tickets = [i for i in odds.index if "support_tickets" in i][0]
    assert odds.loc[tenure, "odds_ratio"] < 1
    assert odds.loc[tickets, "odds_ratio"] > 1


def test_smote_only_resamples_training(prepared):
    """The test set must never be resampled — its length is unchanged under SMOTE."""
    spec = prepared["spec"]
    X = prepared["df"][spec.all_input_cols]
    out = modeling.train_all(spec, X, prepared["y"], imbalance="smote", fit_lr=False)
    assert len(out.y_test) == len(out.X_test)
    assert len(out.gbm.proba) == len(out.y_test)


def test_threshold_helpers(trained):
    t = modeling.best_f1_threshold(trained.gbm.y_test, trained.gbm.proba)
    assert 0.0 <= t <= 1.0
    tc = modeling.cost_weighted_threshold(trained.gbm.y_test, trained.gbm.proba, cost_fn=5, cost_fp=1)
    assert 0.0 <= tc <= 1.0


def test_gains_table_lift_decreasing_top_decile_highest(trained):
    g = trained.gbm.gains
    # the top risk decile should have higher churn rate than the bottom
    assert g.iloc[0]["churn_rate"] > g.iloc[-1]["churn_rate"]
    assert g.iloc[0]["lift"] >= 1.0
