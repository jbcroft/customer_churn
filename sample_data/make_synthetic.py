"""Generate a realistic synthetic churn dataset with a *known* signal.

The signal is engineered so tests can assert that the right drivers surface:

  * SHORTER tenure              -> HIGHER churn
  * MORE support_tickets        -> HIGHER churn
  * month-to-month `contract`   -> HIGHER churn (vs one/two-year)
  * HIGHER monthly_charges       -> mild HIGHER churn
  * recent `last_login_date`     -> LOWER churn (stale login -> higher churn)
  * `region`                     -> negligible effect (a near-null feature)

It also injects realistic messiness: missing values, a couple of exact
duplicate rows, a constant column, a high-cardinality id column, and an
outlier or two — so the cleaning/profiling stages have something to do.

Run as a script to (re)write ``sample_data/churn_synthetic.csv``::

    python -m sample_data.make_synthetic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 42

CONTRACTS = ["Month-to-month", "One year", "Two year"]
PLANS = ["Basic", "Standard", "Premium"]
REGIONS = ["North", "South", "East", "West", "Central"]

# Coefficients on standardized / encoded inputs feeding a logistic link. These
# define the ground-truth drivers the tests assert on.
_COEF = {
    "intercept": -1.15,        # sets base rate near ~22%
    "tenure_months": -1.30,    # longer tenure -> less churn (strong)
    "support_tickets": 0.95,   # more tickets -> more churn (strong)
    "monthly_charges": 0.45,   # pricier -> a bit more churn (moderate)
    "days_since_login": 0.80,  # stale login -> more churn (strong)
    "contract_m2m": 0.85,      # month-to-month -> more churn (strong)
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_synthetic(n: int = 5000, seed: int = RANDOM_STATE, messy: bool = True) -> pd.DataFrame:
    """Return a synthetic churn DataFrame of ``n`` rows."""
    rng = np.random.default_rng(seed)

    customer_id = np.array([f"CUST-{i:06d}" for i in range(n)])
    tenure_months = rng.integers(0, 72, size=n)
    monthly_charges = np.round(rng.normal(70, 25, size=n).clip(15, 200), 2)
    support_tickets = rng.poisson(1.4, size=n)
    contract = rng.choice(CONTRACTS, size=n, p=[0.55, 0.25, 0.20])
    plan = rng.choice(PLANS, size=n, p=[0.4, 0.35, 0.25])
    region = rng.choice(REGIONS, size=n)  # near-null driver

    # days since last login: most recent for engaged users, long for at-risk
    days_since_login = rng.exponential(20, size=n).clip(0, 365).round().astype(int)

    # --- ground-truth churn via logistic link on standardized features ---
    def z(a: np.ndarray) -> np.ndarray:
        a = a.astype(float)
        return (a - a.mean()) / (a.std() + 1e-9)

    logit = (
        _COEF["intercept"]
        + _COEF["tenure_months"] * z(tenure_months)
        + _COEF["support_tickets"] * z(support_tickets)
        + _COEF["monthly_charges"] * z(monthly_charges)
        + _COEF["days_since_login"] * z(days_since_login)
        + _COEF["contract_m2m"] * (contract == "Month-to-month").astype(float)
    )
    p = _sigmoid(logit)
    churn = (rng.random(n) < p).astype(int)

    # last_login_date derived from days_since_login relative to a fixed "today"
    today = pd.Timestamp("2026-06-01")
    last_login_date = today - pd.to_timedelta(days_since_login, unit="D")

    # signup_date implied by tenure (gives the app a chance to derive tenure too)
    signup_date = today - pd.to_timedelta(tenure_months * 30, unit="D")

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "signup_date": signup_date,
            "last_login_date": last_login_date,
            "tenure_months": tenure_months,
            "contract": contract,
            "plan": plan,
            "region": region,
            "monthly_charges": monthly_charges,
            "support_tickets": support_tickets,
            "churn": np.where(churn == 1, "Yes", "No"),  # string target on purpose
        }
    )

    if messy:
        df = _add_mess(df, rng)

    return df


def _add_mess(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    df = df.copy()

    # missing values scattered through a few columns
    for col, frac in [("monthly_charges", 0.04), ("region", 0.03), ("plan", 0.02)]:
        idx = rng.choice(n, size=int(n * frac), replace=False)
        df.loc[idx, col] = np.nan

    # a constant column (should be flagged + droppable)
    df["data_source"] = "billing_system_v2"

    # a high-cardinality free-text-ish column
    df["account_note"] = [f"note-{i}" for i in rng.integers(0, n, size=n)]

    # a couple of exact duplicate rows
    dup = df.iloc[[10, 11]].copy()
    df = pd.concat([df, dup], ignore_index=True)

    # a few price outliers
    out_idx = rng.choice(len(df), size=5, replace=False)
    df.loc[out_idx, "monthly_charges"] = rng.uniform(900, 1500, size=5).round(2)

    # shuffle column order a touch (id not first) to exercise inference
    return df.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic churn CSV")
    ap.add_argument("--rows", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    ap.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).with_name("churn_synthetic.csv")),
    )
    ap.add_argument("--clean", action="store_true", help="emit a tidy version (no mess)")
    args = ap.parse_args()

    df = make_synthetic(n=args.rows, seed=args.seed, messy=not args.clean)
    df.to_csv(args.out, index=False)
    rate = (df["churn"] == "Yes").mean()
    print(f"Wrote {len(df):,} rows -> {args.out}")
    print(f"Base churn rate: {rate:.1%}")


if __name__ == "__main__":
    main()
