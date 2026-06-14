"""Streamlit entrypoint — thin orchestration + UI only.

All real logic lives in the importable ``churn`` package. This file wires the
linear, gated pipeline (UPLOAD -> PROFILE -> TARGET -> CLEAN -> TRANSFORM ->
MODEL -> DRIVERS -> VISUALIZE -> REPORT) into ``st.session_state`` and renders
each stage. A "Run full analysis" button executes clean->…->visualize in one go.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from churn import cleaning, io, profiling, report
from churn.pipeline import AnalysisConfig, run_full_analysis, segment_rates_for_payload
from churn.modeling import (
    IMBALANCE_CLASS_WEIGHT, IMBALANCE_NONE, IMBALANCE_SMOTE, best_f1_threshold,
)
from churn.state import AppState, Stage

load_dotenv()
st.set_page_config(page_title="Churn Analysis", page_icon="📉", layout="wide")


# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
def S() -> AppState:
    if "app" not in st.session_state:
        st.session_state.app = AppState()
    return st.session_state.app


@st.cache_data(show_spinner=False)
def _read(file_bytes: bytes, filename: str, sheet: str | None):
    return io.read_table(file_bytes, filename=filename, sheet=sheet)


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
STAGES = list(Stage)


def sidebar(state: AppState) -> Stage:
    st.sidebar.title("📉 Churn Analysis")
    st.sidebar.caption("Upload → drivers → board-ready report")

    if state.dataset_name:
        st.sidebar.markdown(f"**Dataset:** `{state.dataset_name}`")
    if state.target_col:
        st.sidebar.markdown(f"**Target:** `{state.target_col}` = `{state.positive_class}`")
    if state.base_rate is not None:
        st.sidebar.metric("Base churn rate", f"{state.base_rate:.1%}")

    ready = state.furthest_ready_stage()
    st.sidebar.progress(min(1.0, (ready + 1) / len(STAGES)),
                        text=f"Progress: {ready.label}")

    nav = st.sidebar.radio(
        "Stage", options=STAGES, format_func=lambda s: f"{s+1}. {s.label}",
        index=st.session_state.get("nav_idx", 0), key="nav_radio",
    )
    st.session_state.nav_idx = int(nav)

    # Run full analysis
    st.sidebar.divider()
    can_run = state.raw_df is not None and state.target_col is not None
    if st.sidebar.button("⚡ Run full analysis", type="primary", disabled=not can_run,
                         width="stretch"):
        _run_everything(state)

    # API key status
    cfg = report.ReportConfig()
    icon = "✅" if cfg.api_key else "⚠️"
    st.sidebar.caption(f"{icon} ANTHROPIC_API_KEY {'set' if cfg.api_key else 'missing'} · model `{cfg.model}`")

    # Transformation log (audit trail)
    with st.sidebar.expander(f"📝 Transformation log ({len(state.log)})", expanded=False):
        if len(state.log):
            for line in state.log.as_lines():
                st.write("•", line)
        else:
            st.caption("No steps yet.")
    return nav


def _run_everything(state: AppState):
    import time

    cfg = _current_analysis_config(state)
    start = time.time()
    with st.sidebar:
        status = st.status("Running full analysis…", expanded=True)
    bar = status.progress(0.0)
    done: list[str] = []

    def cb(frac: float, msg: str) -> None:
        # mark the previous step done, show the current one in-progress
        if done:
            status.write(f"✓ {done[-1]}")
        done.append(msg)
        elapsed = time.time() - start
        bar.progress(min(max(frac, 0.0), 1.0), text=f"{msg}  ·  {elapsed:0.0f}s")

    try:
        run_full_analysis(state.raw_df, state.roles_or_overrides(), cfg, state=state, progress=cb)
        if done:
            status.write(f"✓ {done[-1]}")
        status.update(label=f"✅ Analysis complete in {time.time() - start:0.0f}s — explore the stages.",
                      state="complete", expanded=False)
    except Exception as exc:  # noqa: BLE001
        status.update(label=f"❌ Analysis failed: {exc}", state="error", expanded=True)


def _current_analysis_config(state: AppState) -> AnalysisConfig:
    c = st.session_state.get("clean_cfg", cleaning.CleaningConfig())
    return AnalysisConfig(
        target_col=state.target_col,
        positive_class=state.positive_class,
        cleaning=c,
        imbalance=st.session_state.get("imbalance", IMBALANCE_CLASS_WEIGHT),
        test_size=st.session_state.get("test_size", 0.25),
        threshold=st.session_state.get("threshold", 0.5),
        n_splits=st.session_state.get("n_splits", 5),
        compute_shap=st.session_state.get("compute_shap", True),
    )


# AppState helper bound at runtime (roles possibly overridden)
def _roles(state: AppState) -> dict[str, str]:
    base = getattr(state, "_roles", {}) or {}
    return {**base, **state.type_overrides}


AppState.roles_or_overrides = lambda self: {**getattr(self, "_roles", {}), **self.type_overrides}  # type: ignore


# ----------------------------------------------------------------------
# Stage 1 — Upload
# ----------------------------------------------------------------------
def stage_upload(state: AppState):
    st.header("1 · Upload")
    st.write("Upload a single **CSV** or **Excel** file. The app sniffs encoding/delimiter "
             "and infers a *role* for each column you can override below.")
    up = st.file_uploader("Dataset", type=["csv", "xlsx", "xls"], accept_multiple_files=False)
    if up is None:
        if st.button("Use the bundled synthetic demo dataset"):
            _load_demo(state)
            st.rerun()
        return

    sheet = None
    raw = up.getvalue()
    if up.name.lower().endswith((".xlsx", ".xls")):
        sheets = io.list_excel_sheets(raw)
        sheet = st.selectbox("Worksheet", sheets)
    try:
        info = _read(raw, up.name, sheet)
    except io.DataReadError as e:
        st.error(str(e))
        return

    _ingest(state, info)
    _render_schema_editor(state, info)


def _load_demo(state: AppState):
    import pathlib

    p = pathlib.Path(__file__).parent / "sample_data" / "churn_synthetic.csv"
    info = io.read_table(str(p), filename="churn_synthetic.csv")
    _ingest(state, info)


def _ingest(state: AppState, info: io.TableInfo):
    if state.dataset_name != info.df.attrs.get("name") or state.raw_df is None:
        state.invalidate_from(Stage.PROFILE)
    state.raw_df = info.df
    state._roles = info.roles  # type: ignore[attr-defined]
    state.dataset_name = state.dataset_name or "uploaded"
    st.success(f"Loaded **{info.n_rows:,} rows × {info.n_cols} columns** "
               f"({info.memory_mb:.1f} MB).")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{info.n_rows:,}")
    c2.metric("Columns", info.n_cols)
    c3.metric("Memory", f"{info.memory_mb:.1f} MB")
    st.dataframe(info.df.head(20), width="stretch")


def _render_schema_editor(state: AppState, info: io.TableInfo):
    st.subheader("Column roles")
    st.caption("Override the inferred role if needed. `id` columns are excluded from modeling.")
    schema = io.schema_summary(info)
    schema["override"] = schema["column"].map(lambda c: state.type_overrides.get(c, info.roles.get(c)))
    edited = st.data_editor(
        schema, width="stretch", hide_index=True, key="schema_editor",
        column_config={
            "override": st.column_config.SelectboxColumn("role", options=list(io.ROLES), required=True),
        },
        disabled=["column", "role", "dtype", "n_unique", "pct_missing", "sample"],
    )
    new_overrides = {
        r["column"]: r["override"] for _, r in edited.iterrows()
        if r["override"] != info.roles.get(r["column"])
    }
    if new_overrides != state.type_overrides:
        state.type_overrides = new_overrides
        df2, roles2 = io.apply_role_overrides(info.df, info.roles, new_overrides)
        state.raw_df = df2
        state._roles = roles2  # type: ignore[attr-defined]
        state.invalidate_from(Stage.PROFILE)
    st.session_state.setdefault("dataset_name_input", "uploaded")
    name = st.text_input("Dataset name (for the report)", value=state.dataset_name or "uploaded")
    state.dataset_name = name


# ----------------------------------------------------------------------
# Stage 2 — Profile
# ----------------------------------------------------------------------
def stage_profile(state: AppState):
    st.header("2 · Profile")
    if state.raw_df is None:
        st.info("Upload a dataset first.")
        return
    roles = _roles(state)
    target = state.target_col
    prof = profiling.profile_dataframe(state.raw_df, roles, target_col=target)
    state.profile = prof

    st.subheader("Per-column profile")
    st.dataframe(prof.as_frame(), width="stretch", hide_index=True)
    cols = st.columns(2)
    cols[0].metric("Rows", f"{prof.n_rows:,}")
    cols[1].metric("Exact duplicate rows", prof.n_duplicate_rows)

    st.subheader("⚠️ Leakage & data-quality warnings")
    st.caption("Surfaced as warnings — nothing is auto-dropped. You decide in **Clean**.")
    if not prof.leakage:
        st.success("No obvious leakage heuristics triggered.")
    for w in prof.leakage:
        st.warning(f"**{w.column}** — {w.message}")
    hc = [c.column for c in prof.columns if c.is_high_cardinality]
    if hc:
        st.info(f"High-cardinality categoricals (will be top-N bucketed): {', '.join(hc)}")


# ----------------------------------------------------------------------
# Stage 3 — Target
# ----------------------------------------------------------------------
def stage_target(state: AppState):
    st.header("3 · Select target")
    if state.raw_df is None:
        st.info("Upload a dataset first.")
        return
    df = state.raw_df
    cols = list(df.columns)
    default = cols.index(state.target_col) if state.target_col in cols else (
        cols.index("churn") if "churn" in cols else 0)
    target = st.selectbox("Which column indicates churn?", cols, index=default)

    val = profiling.validate_target(df, target)
    if val.needs_threshold:
        st.warning(val.message)
        thr = st.number_input("Threshold (>= is 'churned')", value=float(pd.to_numeric(df[target], errors="coerce").median()))
        y = profiling.binarize_continuous(df[target], thr)
        pos = f">= {thr}"
    elif val.is_binary:
        st.success(val.message)
        pos = st.selectbox("Positive ('churned') class", val.classes,
                           index=val.classes.index(val.suggested_positive) if val.suggested_positive in val.classes else 0)
        y = profiling.encode_target(df[target], pos)
    else:
        st.warning(val.message)
        if not val.classes:
            return
        pos = st.selectbox("Which class means 'churned'?", val.classes)
        y = profiling.encode_target(df[target], pos)

    rate = profiling.base_rate(y)
    if state.target_col != target or state.positive_class != pos:
        state.invalidate_from(Stage.CLEAN)
    state.target_col, state.positive_class, state.base_rate = target, pos, rate

    c1, c2 = st.columns([1, 2])
    c1.metric("Base churn rate", f"{rate:.1%}")
    from churn import viz
    c2.plotly_chart(viz.target_balance(y, rate), width="stretch")
    if rate < 0.05 or rate > 0.95:
        st.warning("Severe class imbalance — lean on PR-AUC/lift and class weights or SMOTE.")


# ----------------------------------------------------------------------
# Stage 4 — Clean
# ----------------------------------------------------------------------
def stage_clean(state: AppState):
    st.header("4 · Clean")
    if state.raw_df is None or state.target_col is None:
        st.info("Upload a dataset and select the target first.")
        return
    roles = _roles(state)
    prof = state.profile or profiling.profile_dataframe(state.raw_df, roles, state.target_col)
    suggested = cleaning.suggested_drops(prof.leakage, roles)

    st.caption("Defaults are sane; adjust as needed. Imputation runs inside the train-only pipeline.")
    c1, c2 = st.columns(2)
    with c1:
        drop_cols = st.multiselect("Columns to exclude (ids / leakage)",
                                   [c for c in state.raw_df.columns if c != state.target_col],
                                   default=suggested)
        drop_dupes = st.checkbox("Drop exact duplicate rows", value=True)
        drop_const = st.checkbox("Drop constant columns", value=True)
        miss_thresh = st.slider("Drop columns missing more than", 0.1, 1.0, 0.6, 0.05)
    with c2:
        impute_num = st.selectbox("Numeric imputation", ["median", "mean", "constant"])
        impute_cat = st.selectbox("Categorical imputation", ["most_frequent", "constant"])
        max_cats = st.slider("Max categories per feature (top-N bucketing)", 5, 50, 20)
        winsor = st.checkbox("Winsorize numeric outliers (1%/99%)", value=False)

    cfg = cleaning.CleaningConfig(
        drop_duplicates=drop_dupes, drop_constant=drop_const,
        missing_col_drop_threshold=miss_thresh, drop_columns=drop_cols,
        impute_numeric=impute_num, impute_categorical=impute_cat,
        max_categories=max_cats, winsorize=winsor,
    )
    st.session_state.clean_cfg = cfg

    preview = st.button("Preview cleaning")
    if preview or state.clean_df is not None:
        clean_df, rep, feats = cleaning.clean_dataframe(state.raw_df, roles, cfg, state.target_col)
        a, b, c, d = st.columns(4)
        a.metric("Rows", f"{rep.rows_after:,}", f"{rep.rows_after - rep.rows_before:,}")
        b.metric("Columns", rep.cols_after, rep.cols_after - rep.cols_before)
        c.metric("Duplicates removed", rep.duplicates_removed)
        d.metric("Feature columns", len(feats))
        if rep.dropped_columns:
            st.write("**Dropped:**", ", ".join(f"`{k}` ({v})" for k, v in rep.dropped_columns.items()))
        if rep.outliers:
            with st.expander("Outlier report (IQR / z-score)"):
                st.dataframe(pd.DataFrame([o.__dict__ for o in rep.outliers]), hide_index=True,
                             width="stretch")


# ----------------------------------------------------------------------
# Stage 5 — Transform
# ----------------------------------------------------------------------
def stage_transform(state: AppState):
    st.header("5 · Transform")
    st.caption("Leakage firewall: imputation, encoding, scaling, recency and any resampling "
               "are fit on the **training fold only**, inside a sklearn/imblearn Pipeline.")
    if state.feature_spec is None:
        st.info("Run **Clean**, then **⚡ Run full analysis** (or train in the Model stage) to "
                "build the feature pipeline. VIF/correlation appear here afterwards.")
        return
    spec = state.feature_spec
    st.write(f"**Numeric:** {spec.numeric_cols} · **Categorical:** {spec.categorical_cols} · "
             f"**Datetime → recency:** {spec.datetime_cols}")
    if state.vif_table is not None:
        st.subheader("Multicollinearity (VIF)")
        st.caption("VIF > 10 flags redundant features that distort LR coefficients.")
        st.dataframe(state.vif_table, hide_index=True, width="stretch")
        if state.vif_table.get("high_collinearity", pd.Series(dtype=bool)).any():
            bad = state.vif_table[state.vif_table["high_collinearity"]]["feature"].tolist()
            st.warning(f"High collinearity: {bad}. Consider excluding one of each redundant pair.")
    if "correlation" in state.figures:
        st.plotly_chart(state.figures["correlation"], width="stretch")


# ----------------------------------------------------------------------
# Stage 6 — Model
# ----------------------------------------------------------------------
def stage_model(state: AppState):
    st.header("6 · Model")
    if state.clean_df is None and state.raw_df is not None and state.target_col is not None:
        st.info("Tip: set options below, then **⚡ Run full analysis** in the sidebar.")
    c1, c2, c3 = st.columns(3)
    imb = c1.selectbox("Imbalance handling", [IMBALANCE_CLASS_WEIGHT, IMBALANCE_SMOTE, IMBALANCE_NONE],
                       format_func=lambda s: {"class_weight": "Class weights (balanced)",
                                              "smote": "SMOTE (in-fold)", "none": "None"}[s])
    test_size = c2.slider("Test size", 0.1, 0.4, 0.25, 0.05)
    n_splits = c3.slider("CV folds", 3, 10, 5)
    st.session_state.update(imbalance=imb, test_size=test_size, n_splits=n_splits)

    mo = state.model_result
    if mo is None:
        st.info("No trained model yet. Use **⚡ Run full analysis**.")
        return

    st.subheader("Cross-validated metrics (mean ± std)")
    rows = []
    for m in (mo.lr, mo.gbm):
        row = {"model": m.name}
        for k, (mean, sd) in m.cv.items():
            row[k] = f"{mean:.3f} ± {sd:.3f}"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.subheader("Held-out test metrics")
    test_rows = [{"model": m.name, **{k: round(v, 3) for k, v in m.test.items() if v == v}}
                 for m in (mo.lr, mo.gbm)]
    st.dataframe(pd.DataFrame(test_rows), hide_index=True, width="stretch")

    st.subheader("Decision threshold")
    f1opt = best_f1_threshold(mo.gbm.y_test, mo.gbm.proba)
    thr = st.slider("Classify as churn when probability ≥", 0.05, 0.95,
                    float(st.session_state.get("threshold", 0.5)), 0.01)
    st.caption(f"F1-optimal threshold on the GBM ≈ **{f1opt:.2f}**.")
    st.session_state.threshold = thr
    cols = st.columns(2)
    cols[0].plotly_chart(state.figures.get("threshold"), width="stretch") if "threshold" in state.figures else None
    cols[1].plotly_chart(state.figures.get("confusion"), width="stretch") if "confusion" in state.figures else None


# ----------------------------------------------------------------------
# Stage 7 — Drivers
# ----------------------------------------------------------------------
def stage_drivers(state: AppState):
    st.header("7 · Key drivers")
    if not state.driver_table:
        st.info("Run **⚡ Run full analysis** to compute drivers (odds ratios + permutation + SHAP).")
        return
    parent = state.driver_table["parent"]
    gran = state.driver_table["granular"]

    st.subheader("Executive view — ranked drivers (aggregated to original fields)")
    show = parent.copy()
    cols = ["strength_rank", "parent", "direction", "strength", "any_significant"]
    if "univariate_direction" in show:
        cols += ["univariate_direction", "agrees"]
    st.dataframe(show[[c for c in cols if c in show]], hide_index=True, width="stretch")
    if "driver_importance" in state.figures:
        st.plotly_chart(state.figures["driver_importance"], width="stretch")

    if "shap_beeswarm" in state.figures:
        st.subheader("SHAP summary")
        st.plotly_chart(state.figures["shap_beeswarm"], width="stretch")

    deps = {k: v for k, v in state.figures.items() if k.startswith("shap_dependence::")}
    if deps:
        st.subheader("How top drivers move churn (SHAP dependence)")
        for fig in deps.values():
            st.plotly_chart(fig, width="stretch")

    with st.expander("Granular driver table (per encoded feature)"):
        st.dataframe(gran, hide_index=True, width="stretch")


# ----------------------------------------------------------------------
# Stage 8 — Visualize
# ----------------------------------------------------------------------
def stage_visualize(state: AppState):
    st.header("8 · Visualize")
    if not state.figures:
        st.info("Run **⚡ Run full analysis** first.")
        return
    figs = state.figures

    st.subheader("Who is leaving")
    a, b = st.columns(2)
    a.plotly_chart(figs["target_balance"], width="stretch")
    seg = {k: v for k, v in figs.items() if k.startswith("segment::")}
    dist = {k: v for k, v in figs.items() if k.startswith("distribution::")}
    for col, fig in zip([b] + st.columns(2) * 3, list(seg.values()) + list(dist.values())):
        col.plotly_chart(fig, width="stretch")

    st.subheader("Model performance")
    a, b = st.columns(2)
    a.plotly_chart(figs["pr"], width="stretch")
    b.plotly_chart(figs["roc"], width="stretch")
    a.plotly_chart(figs["calibration"], width="stretch")
    b.plotly_chart(figs["confusion"], width="stretch")
    a.plotly_chart(figs["lift"], width="stretch")
    b.plotly_chart(figs["cumulative_gains"], width="stretch")


# ----------------------------------------------------------------------
# Stage 9 — Report
# ----------------------------------------------------------------------
def stage_report(state: AppState):
    st.header("9 · Report")
    if not state.driver_table or state.model_result is None:
        st.info("Run **⚡ Run full analysis** first.")
        return

    st.caption("🔒 Privacy: only aggregated findings (schema + statistics) are sent to Anthropic — "
               "never raw customer rows.")
    cfg = report.ReportConfig()
    if not cfg.api_key:
        st.warning("ANTHROPIC_API_KEY is not set. Add it to `.env` to generate the written memo. "
                   "You can still export the figures and tables.")

    mo = state.model_result
    if st.button("✍️ Generate report with Claude", type="primary", disabled=not cfg.api_key):
        payload = report.build_findings_payload(
            dataset_name=state.dataset_name or "dataset",
            n_rows=len(state.clean_df), n_features=len(state.feature_spec.all_input_cols),
            base_rate=state.base_rate,
            cleaning_log=state.log.as_lines(),
            driver_parent_table=state.driver_table["parent"],
            stats_table=state.driver_table["stats"],
            model_metrics={"lr": report.metrics_for_payload(mo.lr),
                           "gbm": report.metrics_for_payload(mo.gbm)},
            segment_rates=segment_rates_for_payload(state.clean_df,
                profiling.encode_target(state.clean_df[state.target_col], state.positive_class),
                state.feature_spec, state.driver_table["parent"]),
        )
        with st.spinner("Writing the memo…"):
            try:
                state.report_markdown = report.generate_report(payload, cfg)
            except report.ReportError as e:
                st.error(str(e))

    if state.report_markdown:
        st.markdown(state.report_markdown)

        # Driver visuals embedded with the report (SHAP + drivers + segments).
        report_figs = report.report_figure_subset(state.figures)
        if report_figs:
            st.divider()
            st.subheader("📊 Key driver visuals")
            st.caption("The SHAP charts explain *why* customers churn; the segment charts show *who*.")
            for title, fig in report_figs.items():
                st.markdown(f"**{title}**")
                st.plotly_chart(fig, width="stretch", key=f"rep_{title}")

        st.divider()
        st.subheader("Export")
        c1, c2, c3 = st.columns(3)
        c1.download_button("⬇️ Markdown", report.export_markdown(state.report_markdown),
                           "churn_report.md", "text/markdown")
        c2.download_button("⬇️ HTML (+figures)",
                           report.export_html(state.report_markdown, report_figs),
                           "churn_report.html", "text/html")
        pdf = report.export_pdf(state.report_markdown, report_figs)
        if pdf:
            c3.download_button("⬇️ PDF (+figures)", pdf, "churn_report.pdf", "application/pdf")
        else:
            c3.caption("PDF needs weasyprint system libs — use HTML/Markdown.")
        docx = report.export_docx(state.report_markdown)
        if docx:
            st.download_button("⬇️ DOCX", docx, "churn_report.docx",
                               "application/vnd.openxmlformats-officedocument.wordprocessingml.document")


# ----------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------
DISPATCH = {
    Stage.UPLOAD: stage_upload, Stage.PROFILE: stage_profile, Stage.TARGET: stage_target,
    Stage.CLEAN: stage_clean, Stage.TRANSFORM: stage_transform, Stage.MODEL: stage_model,
    Stage.DRIVERS: stage_drivers, Stage.VISUALIZE: stage_visualize, Stage.REPORT: stage_report,
}


def main():
    state = S()
    nav = sidebar(state)
    DISPATCH[nav](state)


main()
