"""End-to-end headless smoke test of the full pipeline + report payload + exports.

The Anthropic call is mocked — only the manual run hits the real API.
"""

from __future__ import annotations

import warnings

import pandas as pd

from churn import cleaning, profiling, report
from churn.pipeline import AnalysisConfig, run_full_analysis, segment_rates_for_payload
from churn.state import Stage

warnings.filterwarnings("ignore")


def test_full_pipeline_end_to_end(ingested):
    df, roles = ingested.df, ingested.roles
    prof = profiling.profile_dataframe(df, roles, target_col="churn")
    drops = cleaning.suggested_drops(prof.leakage, roles) + ["account_note", "signup_date"]
    cfg = AnalysisConfig(target_col="churn", positive_class="Yes",
                         cleaning=cleaning.CleaningConfig(drop_columns=drops))
    state = run_full_analysis(df, roles, cfg)

    assert state.furthest_ready_stage() == Stage.VISUALIZE
    assert 0.0 < state.base_rate < 1.0
    assert len(state.figures) >= 12
    assert state.model_result.lr is not None and state.model_result.gbm is not None
    parent = state.driver_table["parent"]
    # the strongest driver should be one of the engineered signals
    assert parent.iloc[0]["parent"] in {"tenure_months", "support_tickets", "last_login_date"}
    assert len(state.log) >= 3


def test_report_payload_and_exports_no_raw_rows(ingested, monkeypatch):
    df, roles = ingested.df, ingested.roles
    cfg = AnalysisConfig(target_col="churn", positive_class="Yes",
                         cleaning=cleaning.CleaningConfig(drop_columns=["customer_id", "account_note"]))
    state = run_full_analysis(df, roles, cfg)
    mo = state.model_result
    payload = report.build_findings_payload(
        dataset_name="d", n_rows=len(state.clean_df),
        n_features=len(state.feature_spec.all_input_cols), base_rate=state.base_rate,
        cleaning_log=state.log.as_lines(), driver_parent_table=state.driver_table["parent"],
        stats_table=state.driver_table["stats"],
        model_metrics={"lr": report.metrics_for_payload(mo.lr), "gbm": report.metrics_for_payload(mo.gbm)},
        segment_rates=segment_rates_for_payload(
            state.clean_df, profiling.encode_target(state.clean_df["churn"], "Yes"),
            state.feature_spec, state.driver_table["parent"]),
    )
    # privacy: payload must be aggregates only — no per-customer ids, no raw row count of values
    import json
    blob = json.dumps(payload, default=str)
    assert "CUST-" not in blob          # no customer identifiers leaked
    assert payload["base_churn_rate"] > 0
    assert len(payload["top_drivers"]) >= 3

    # exports that don't need an API key
    assert report.export_markdown("# hi").startswith(b"#")
    assert b"<html" in report.export_html("# hi", state.figures).lower()

    # mock the Anthropic call
    def fake_generate(p, c=None):
        return "# Executive summary\n\nChurn is " + f"{p['base_churn_rate']:.0%}."
    monkeypatch.setattr(report, "generate_report", fake_generate)
    md = report.generate_report(payload)
    assert "Executive summary" in md
