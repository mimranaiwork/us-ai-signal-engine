"""
Fundamentals from SEC XBRL company facts (spec sections 20-21), with the
point-in-time correctness the spec explicitly requires: every value carries
its actual `filed` timestamp, and `as_of_fundamentals()` only ever returns
values that were genuinely public knowledge by the requested date. A 10-K
filed 2024-02-01 covering fiscal year 2023 is invisible to any as_of_date
before 2024-02-01, even though the fiscal year it describes ended earlier —
that's the whole point (see tests/test_fundamentals.py's leakage test).

XBRL tag names aren't fully standardized across filers (some use `Revenues`,
others `RevenueFromContractWithCustomerExcludingAssessedTax`, etc.) — each
metric below lists candidate tags in preference order; the first one present
in a company's facts wins.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

CONCEPT_CANDIDATES: dict[str, list[str]] = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
    "net_income": ["NetIncomeLoss"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "stockholders_equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "shares_outstanding": ["CommonStockSharesOutstanding"],
    # Sharia-screen inputs (features/compliance/sharia_screen.py) — interest-bearing
    # debt is typically split across current/noncurrent tags with less-standardized
    # naming than the core metrics above; treated as best-effort, see that module.
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "short_term_debt": ["ShortTermBorrowings", "DebtCurrent", "LongTermDebtCurrent"],
    "short_term_investments": ["ShortTermInvestments", "MarketableSecuritiesCurrent"],
    "interest_income": ["InvestmentIncomeInterest", "InterestIncomeOther"],
}

RATIO_METRICS = ["revenue", "net_income", "gross_profit", "operating_income", "assets",
                  "stockholders_equity", "liabilities", "current_assets", "current_liabilities",
                  "operating_cash_flow", "capex"]

# "Flow" metrics (revenue, income, cash flow — reported over a PERIOD) vs
# "point-in-time" metrics (balance-sheet items — assets/liabilities/equity/
# shares, reported AS OF an instant, no meaningful start/end duration).
# XBRL 10-Q filings report flow concepts for BOTH the 3-month quarter AND
# the cumulative year-to-date period as separate records with the SAME
# `filed` date — without filtering by period length, "latest filed" can
# silently pick a quarterly figure for one metric and a YTD figure for
# another filed the same day, producing apples-to-oranges ratios (this was
# caught by a real AAPL operating margin coming out at 71%, which is not
# a real number for that company).
FLOW_METRICS = {"revenue", "net_income", "eps_diluted", "gross_profit",
                 "operating_income", "operating_cash_flow", "capex", "interest_income"}


def _extract_concept(facts: dict, candidates: list[str]) -> list[dict]:
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for name in candidates:
        concept = us_gaap.get(name)
        if concept and "USD" in concept.get("units", {}):
            return concept["units"]["USD"]
        if concept and "USD/shares" in concept.get("units", {}):
            return concept["units"]["USD/shares"]
        if concept and "shares" in concept.get("units", {}):
            return concept["units"]["shares"]
    return []


def _keep_matching_period(metric: str, record: dict) -> bool:
    """For flow metrics, keep only the single-period figure matching the
    filing type (~1 quarter for 10-Q, ~1 year for 10-K) — drops YTD/other
    cumulative-period duplicates. Point-in-time metrics (no `start`) always pass."""
    if metric not in FLOW_METRICS:
        return True
    start, end, form = record.get("start"), record.get("end"), record.get("form")
    if not start or not end:
        return False  # a flow metric with no period bounds isn't usable
    period_days = (pd.to_datetime(end) - pd.to_datetime(start)).days
    if form == "10-Q":
        return 80 <= period_days <= 100        # ~1 quarter
    if form == "10-K":
        return 350 <= period_days <= 380       # ~1 fiscal year
    return False


def build_fundamentals_long(facts: dict) -> pd.DataFrame:
    """Long-format table: one row per (metric, filing) — the shape
    as_of_fundamentals()/yoy_growth() operate on."""
    rows = []
    for metric, candidates in CONCEPT_CANDIDATES.items():
        for record in _extract_concept(facts, candidates):
            if record.get("form") not in ("10-K", "10-Q"):
                continue  # skip 8-Ks and other filing types for structured fundamentals
            if not _keep_matching_period(metric, record):
                continue
            rows.append({
                "metric": metric,
                "val": record["val"],
                "fy": record.get("fy"),
                "fp": record.get("fp"),
                "form": record["form"],
                "filed": pd.to_datetime(record["filed"]).date(),
                "end": pd.to_datetime(record["end"]).date(),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["metric", "filed"]).reset_index(drop=True)


def as_of_fundamentals(long_df: pd.DataFrame, as_of_date: date) -> dict[str, float]:
    """The latest known value of EACH metric as of as_of_date — independently
    per metric, since different filings covering different metrics can land
    on different dates. Never includes a record filed after as_of_date."""
    if long_df.empty:
        return {}
    available = long_df[long_df["filed"] <= as_of_date]
    if available.empty:
        return {}
    latest = available.sort_values("filed").groupby("metric").last()
    return latest["val"].to_dict()


def yoy_growth(long_df: pd.DataFrame, metric: str, as_of_date: date) -> float | None:
    """Year-over-year growth using the two most recent 10-K (annual) values
    known as of as_of_date. Quarterly YoY (matching same fiscal quarter across
    years) is a reasonable extension but out of scope for Phase 5."""
    available = long_df[
        (long_df["metric"] == metric) & (long_df["filed"] <= as_of_date) & (long_df["form"] == "10-K")
    ].sort_values("end")
    if len(available) < 2:
        return None
    latest, prior = available.iloc[-1], available.iloc[-2]
    if prior["val"] == 0:
        return None
    return float((latest["val"] - prior["val"]) / abs(prior["val"]))


def compute_ratios(values: dict[str, float]) -> dict[str, float | None]:
    def safe_div(a, b):
        if a is None or b is None or b == 0:
            return None
        return float(a / b)

    revenue = values.get("revenue")
    return {
        "gross_margin": safe_div(values.get("gross_profit"), revenue),
        "operating_margin": safe_div(values.get("operating_income"), revenue),
        "net_margin": safe_div(values.get("net_income"), revenue),
        "roe": safe_div(values.get("net_income"), values.get("stockholders_equity")),
        "roa": safe_div(values.get("net_income"), values.get("assets")),
        "debt_to_equity": safe_div(values.get("liabilities"), values.get("stockholders_equity")),
        "current_ratio": safe_div(values.get("current_assets"), values.get("current_liabilities")),
        "free_cash_flow": (
            values["operating_cash_flow"] - values["capex"]
            if values.get("operating_cash_flow") is not None and values.get("capex") is not None
            else None
        ),
    }


def fundamentals_snapshot(long_df: pd.DataFrame, as_of_date: date) -> dict:
    """The full point-in-time fundamentals feature set for one as_of_date:
    raw values + derived ratios + revenue/EPS/net-income YoY growth."""
    values = as_of_fundamentals(long_df, as_of_date)
    ratios = compute_ratios(values)
    growth = {
        f"{m}_growth_yoy": yoy_growth(long_df, m, as_of_date)
        for m in ("revenue", "net_income", "eps_diluted")
    }
    return {"as_of": as_of_date.isoformat(), **values, **ratios, **growth}
