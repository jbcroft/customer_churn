# Customer Churn Analysis

An analyst-facing web app that turns a raw customer dataset into **directional,
plain-language churn drivers** and a **board-ready, AI-written report**. Built
for a private-equity / operating-partner lens: it connects *statistical drivers*
→ *retention economics (NRR/GRR/LTV)* → *intervention hypotheses*.

A non-technical user uploads a file, picks the churn column, and the app handles
profiling, cleaning, leakage-safe feature engineering, robust modeling,
statistical driver analysis, visual explanation, and report generation.

> **What this is, technically:** a binary-classification + driver-attribution
> problem on top of a data-quality pipeline. The value isn't the model (three
> lines fits a classifier) — it's (a) *not lying to you* with leaky or
> imbalance-distorted results, and (b) translating coefficients into directional
> drivers a business owner can act on. A "driver" here means **associated +
> predictive, not causal** — the report says so explicitly.

---

## Quick start

### Option A — Docker (recommended, one command)

```bash
cp .env.example .env          # optional: add your ANTHROPIC_API_KEY for the report
docker compose up --build
```

Open **http://localhost:8501**. Click **"Use the bundled synthetic demo dataset"**
or upload your own CSV/XLSX.

Inside the Linux container you get the *full* experience — XGBoost loads natively
and PDF export works. (On a bare macOS host the app transparently falls back to
sklearn's `HistGradientBoostingClassifier` and markdown/HTML export.)

### Option B — Local Python

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m sample_data.make_synthetic        # writes sample_data/churn_synthetic.csv
streamlit run app.py
```

The AI report stage needs an `ANTHROPIC_API_KEY` (in `.env`); **every other stage
works without one.**

---

## The pipeline

A linear, gated pipeline — each stage writes a versioned artifact to
`st.session_state`, and changing an upstream choice invalidates everything
downstream:

```
UPLOAD → PROFILE → SELECT TARGET → CLEAN → TRANSFORM → MODEL → DRIVERS → VISUALIZE → REPORT
```

Use the per-stage controls for fine-grained work, or hit **⚡ Run full analysis**
to execute clean → … → visualize in one go. A persistent **transformation log**
in the sidebar records everything done to your data (audit trail + report input).

| Stage | What happens |
|---|---|
| **Upload** | CSV/XLSX with encoding + delimiter sniffing; per-column *role* inference (numeric / categorical / datetime / id) you can override. |
| **Profile** | Missingness, cardinality, duplicates, and **leakage heuristics** (id-like columns, post-outcome names, near-perfect separation) surfaced as *warnings* — never auto-dropped. |
| **Target** | Pick the churn column; binary validation or threshold/positive-class selection; the **base churn rate** anchors everything. |
| **Clean** | Drop duplicates/constant/high-missingness/leaky columns, outlier reporting + optional winsorize, imputation strategy. All logged, with before/after counts. |
| **Transform** | sklearn `ColumnTransformer` pipeline: date→recency, one-hot, scaling (LR only), VIF + correlation heatmap. **Fit on the training fold only.** |
| **Model** | Stratified split **and** k-fold CV (mean ± std); class-weights or in-fold SMOTE; **Logistic Regression** (→ odds ratios w/ CIs) + **gradient boosting**. Imbalance-aware metrics led by **PR-AUC**, plus calibration & lift. Tunable threshold. |
| **Drivers** | One ranked, directional driver table unifying **odds ratios + permutation importance + SHAP**, reconciled to original fields and cross-checked against the univariate stats. |
| **Visualize** | Churn-by-segment, distribution overlays, correlation heatmap, SHAP summary/dependence, ROC/PR, confusion, calibration, lift & cumulative-gains. |
| **Report** | A compact **aggregated** findings payload → Claude → a PE-audience memo (exec summary, the churn problem, key drivers, interventions-as-hypotheses, caveats). Export to Markdown / HTML / PDF / DOCX. |

---

## Architecture

A package, not a monolith. `app.py` is a thin Streamlit orchestration layer; all
logic lives in importable, unit-testable modules.

```
app.py                  # Streamlit entrypoint — page flow + session_state wiring only
churn/
├── state.py            # typed session-state container + pipeline-stage enum (no Streamlit)
├── io.py               # robust read, type/role inference, schema summary
├── profiling.py        # missingness, cardinality, leakage heuristics, target validation
├── cleaning.py         # structural cleaning + transformation log
├── features.py         # leakage-safe ColumnTransformer: recency, one-hot, scaling, VIF, name map
├── stats_drivers.py    # univariate tests + effect sizes + mutual info + BH-FDR
├── modeling.py         # split/CV, imbalance, LR + GBM, metrics, calibration, odds ratios
├── explain.py          # SHAP + permutation + odds ratios → unified driver table
├── viz.py              # all Plotly figures (pure: data in, figure out)
├── report.py           # findings payload → Anthropic → md/html/pdf/docx
└── pipeline.py         # headless orchestration shared by app.py and the smoke test
tests/                  # cleaning, features (leakage), stats, modeling, app, end-to-end
sample_data/
└── make_synthetic.py   # realistic churn CSV with a known signal for the demo + tests
```

**Determinism:** a single global `RANDOM_STATE = 42` is threaded everywhere; CV
results are reproducible.

---

## No data leakage, ever

The hard guarantee of the app. Every transform that *learns* from data —
imputation fills, one-hot vocabularies, scaling statistics, recency reference
dates, and SMOTE resampling — happens **inside a pipeline fit on the training
fold only**, then applied unchanged to the test fold. The test set is never
resampled. This is enforced by regression tests
(`tests/test_features.py::test_scaler_learns_train_stats_not_test`,
`test_recency_reference_is_train_only_no_leakage`, and
`tests/test_modeling.py::test_smote_only_resamples_training`).

## Privacy

The report sends only a compact, **aggregated** findings payload (schema +
statistics: base rate, top drivers, metrics, segment rates) to the Anthropic
API — **never raw customer rows.** This is both a token-budget and a
data-sensitivity decision, important for PE data. The model is instructed to use
only numbers present in the payload and to avoid fabricated statistics or
invented benchmarks.

---

## Testing

```bash
pytest                      # full suite: unit + leakage + determinism + end-to-end + app boot
python -m sample_data.make_synthetic && streamlit run app.py   # manual end-to-end
```

The suite covers cleaning, feature leakage, the correct statistical test per
dtype, deterministic + above-no-skill modeling, a headless full-pipeline smoke
test, and a Streamlit `AppTest` that runs the whole flow in-UI. The Anthropic
call is mocked in tests; only a manual run hits the real API.

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | Auth for the report stage (only) | — (report disabled if unset) |
| `REPORT_MODEL` | Model that writes the memo | `claude-sonnet-4-6` |

Secrets live in `.env` (git-ignored); `.env.example` is committed. No secrets in code.

## Optional / stretch

XGBoost and weasyprint are optional: the app detects them and falls back
gracefully (sklearn GBM; markdown/HTML export). Survival analysis (Kaplan–Meier +
Cox), cohort retention curves, a what-if simulator, and CLV linkage are natural
next steps on top of this core.
