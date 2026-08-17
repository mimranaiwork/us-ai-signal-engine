from datetime import date

import pandas as pd
import pytest

from features.earnings.earnings import (
    build_filing_events,
    days_to_next_and_since_last,
    earnings_risk_flag,
    earnings_snapshot,
    estimate_next_filing_date,
)
from features.fundamentals.fundamentals import (
    as_of_fundamentals,
    build_fundamentals_long,
    compute_ratios,
    fundamentals_snapshot,
    yoy_growth,
)


def _fake_company_facts() -> dict:
    """Mimics the real shape of SEC's companyfacts JSON closely enough for
    build_fundamentals_long — two fiscal years of revenue/net income, each
    with a realistic gap between fiscal year-end and actual filing date."""
    return {
        "entityName": "Test Corp",
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {"USD": [
                        {"val": 1_000_000, "fy": 2022, "fp": "FY", "form": "10-K",
                         "filed": "2023-02-15", "start": "2022-01-01", "end": "2022-12-31"},
                        {"val": 1_200_000, "fy": 2023, "fp": "FY", "form": "10-K",
                         "filed": "2024-02-14", "start": "2023-01-01", "end": "2023-12-31"},
                        # A restated/late-filed value should still respect its OWN filed date.
                        {"val": 1_500_000, "fy": 2024, "fp": "FY", "form": "10-K",
                         "filed": "2025-02-20", "start": "2024-01-01", "end": "2024-12-31"},
                        # A same-day-filed YTD (9-month cumulative) duplicate that must be
                        # DROPPED — only the ~1-year 10-K figure above should survive.
                        {"val": 900_000, "fy": 2023, "fp": "FY", "form": "10-K",
                         "filed": "2024-02-14", "start": "2023-04-01", "end": "2023-12-31"},
                    ]},
                },
                "NetIncomeLoss": {
                    "units": {"USD": [
                        {"val": 100_000, "fy": 2022, "fp": "FY", "form": "10-K",
                         "filed": "2023-02-15", "start": "2022-01-01", "end": "2022-12-31"},
                        {"val": 150_000, "fy": 2023, "fp": "FY", "form": "10-K",
                         "filed": "2024-02-14", "start": "2023-01-01", "end": "2023-12-31"},
                    ]},
                },
                "GrossProfit": {
                    "units": {"USD": [
                        {"val": 400_000, "fy": 2023, "fp": "FY", "form": "10-K",
                         "filed": "2024-02-14", "start": "2023-01-01", "end": "2023-12-31"},
                    ]},
                },
                "StockholdersEquity": {
                    "units": {"USD": [
                        {"val": 500_000, "fy": 2023, "fp": "FY", "form": "10-K",
                         "filed": "2024-02-14", "end": "2023-12-31"},
                    ]},
                },
                # An 8-K should be ignored entirely — not a structured fundamentals filing.
                "SomeMisc": {
                    "units": {"USD": [
                        {"val": 999, "fy": 2023, "fp": "FY", "form": "8-K",
                         "filed": "2023-06-01", "end": "2023-05-31"},
                    ]},
                },
            }
        },
    }


@pytest.fixture
def long_df():
    return build_fundamentals_long(_fake_company_facts())


def test_build_fundamentals_long_excludes_non_10k_10q_forms(long_df):
    assert "SomeMisc" not in set(long_df["metric"])  # was an 8-K, must be dropped
    assert set(long_df["form"].unique()) <= {"10-K", "10-Q"}


def test_build_fundamentals_long_drops_mismatched_period_duplicates(long_df):
    """The 900,000 YTD (9-month) revenue record filed the same day as the real
    1,200,000 annual figure must be dropped — only the ~365-day period survives."""
    revenue_rows = long_df[long_df["metric"] == "revenue"]
    assert 900_000 not in set(revenue_rows["val"])
    assert 1_200_000 in set(revenue_rows["val"])


def test_point_in_time_correctness_before_any_filing(long_df):
    """The whole point of this module: before ANY filing date, nothing is visible."""
    snapshot = as_of_fundamentals(long_df, date(2023, 1, 1))
    assert snapshot == {}


def test_point_in_time_correctness_between_filings(long_df):
    """As of a date between the FY2022 10-K filing and the FY2023 10-K filing,
    only the FY2022 value must be visible — NOT FY2023, even though FY2023
    the *fiscal year* had already ended by then."""
    as_of = date(2023, 6, 1)  # after 2023-02-15 filing, before 2024-02-14 filing
    snapshot = as_of_fundamentals(long_df, as_of)
    assert snapshot["revenue"] == 1_000_000  # FY2022 value only


def test_point_in_time_correctness_never_leaks_future_filing(long_df):
    """Automated leakage proof, mirroring tests/test_leakage.py's philosophy:
    for every filing's `filed` date, the day BEFORE it, that filing's value
    must not yet be visible; the day OF/AFTER it, it must be."""
    from datetime import timedelta

    for _, row in long_df.iterrows():
        day_before = row["filed"] - timedelta(days=1)
        before_snapshot = as_of_fundamentals(long_df, day_before)
        on_snapshot = as_of_fundamentals(long_df, row["filed"])

        # This specific filing's value must not appear before its filed date...
        later_filings_of_same_metric = long_df[
            (long_df["metric"] == row["metric"]) & (long_df["filed"] < row["filed"])
        ]
        if later_filings_of_same_metric.empty:
            assert row["metric"] not in before_snapshot
        # ...and must appear on/after its filed date (as the latest known value,
        # assuming no even-newer filing of the same metric supersedes it — true
        # here since row is iterated in filed order per metric).
        if row["filed"] == long_df[long_df["metric"] == row["metric"]]["filed"].max():
            assert on_snapshot[row["metric"]] == row["val"]


def test_yoy_growth_computed_correctly(long_df):
    growth = yoy_growth(long_df, "revenue", date(2024, 6, 1))
    # (1,200,000 - 1,000,000) / 1,000,000 = 0.20
    assert growth == pytest.approx(0.20)


def test_yoy_growth_none_before_two_filings_exist(long_df):
    assert yoy_growth(long_df, "revenue", date(2023, 6, 1)) is None


def test_compute_ratios():
    values = {
        "revenue": 1_200_000, "gross_profit": 400_000, "net_income": 150_000,
        "stockholders_equity": 500_000, "assets": 900_000, "liabilities": 400_000,
        "current_assets": 300_000, "current_liabilities": 150_000,
        "operating_cash_flow": 200_000, "capex": 50_000,
    }
    ratios = compute_ratios(values)
    assert ratios["gross_margin"] == pytest.approx(400_000 / 1_200_000)
    assert ratios["net_margin"] == pytest.approx(150_000 / 1_200_000)
    assert ratios["roe"] == pytest.approx(150_000 / 500_000)
    assert ratios["current_ratio"] == pytest.approx(300_000 / 150_000)
    assert ratios["free_cash_flow"] == pytest.approx(150_000)


def test_compute_ratios_handles_missing_values_gracefully():
    ratios = compute_ratios({"revenue": 1000})
    assert ratios["gross_margin"] is None
    assert ratios["roe"] is None


def test_fundamentals_snapshot_shape(long_df):
    snapshot = fundamentals_snapshot(long_df, date(2024, 6, 1))
    assert snapshot["as_of"] == "2024-06-01"
    assert "revenue" in snapshot
    assert "gross_margin" in snapshot
    assert "revenue_growth_yoy" in snapshot


def _fake_submissions() -> dict:
    return {
        "filings": {
            "recent": {
                "form": ["10-K", "10-Q", "10-Q", "8-K", "10-Q"],
                "filingDate": ["2024-02-14", "2024-05-01", "2024-08-01", "2024-06-15", "2024-11-01"],
            }
        }
    }


def test_build_filing_events_excludes_8k():
    events = build_filing_events(_fake_submissions())
    assert set(events["form"]) == {"10-K", "10-Q"}
    assert len(events) == 4


def test_days_to_next_and_since_last():
    events = build_filing_events(_fake_submissions())
    result = days_to_next_and_since_last(events, date(2024, 6, 1))
    assert result["days_since_earnings"] == (date(2024, 6, 1) - date(2024, 5, 1)).days
    assert result["days_to_earnings"] == (date(2024, 8, 1) - date(2024, 6, 1)).days
    assert result["next_earnings_date"] == date(2024, 8, 1)


def test_days_to_next_and_since_last_no_future_filing():
    events = build_filing_events(_fake_submissions())
    result = days_to_next_and_since_last(events, date(2024, 12, 1))  # after every known filing
    assert result["days_to_earnings"] is None
    assert result["next_earnings_date"] is None


def test_earnings_risk_flag():
    assert earnings_risk_flag(0) is True
    assert earnings_risk_flag(1) is True
    assert earnings_risk_flag(5) is True  # default threshold_days=5, inclusive
    assert earnings_risk_flag(6) is False
    assert earnings_risk_flag(None) is False


def test_estimate_next_filing_date_projects_forward():
    events = pd.DataFrame({
        "filed": [date(2024, 2, 14), date(2024, 5, 1), date(2024, 8, 1), date(2024, 11, 1)],
        "form": ["10-K", "10-Q", "10-Q", "10-Q"],
    })
    estimate = estimate_next_filing_date(events, date(2025, 1, 20))
    assert estimate is not None
    assert estimate > date(2025, 1, 20)


def test_estimate_next_filing_date_never_returns_a_past_date():
    """Regression test: an overdue single-cycle estimate must be projected
    forward past as_of_date, never returned as a stale past date (which
    would produce a negative days_to_earnings that reads as 'imminent'
    forever)."""
    events = pd.DataFrame({
        "filed": [date(2024, 2, 14), date(2024, 5, 1), date(2024, 8, 1), date(2024, 11, 1)],
        "form": ["10-K", "10-Q", "10-Q", "10-Q"],
    })
    as_of = date(2025, 2, 10)  # past the naive single-cycle estimate of 2025-02-01
    estimate = estimate_next_filing_date(events, as_of)
    assert estimate is not None
    assert estimate > as_of


def test_estimate_next_filing_date_gives_up_when_too_stale():
    events = pd.DataFrame({
        "filed": [date(2024, 2, 14), date(2024, 5, 1), date(2024, 8, 1), date(2024, 11, 1)],
        "form": ["10-K", "10-Q", "10-Q", "10-Q"],
    })
    assert estimate_next_filing_date(events, date(2027, 1, 1)) is None


def test_earnings_snapshot_uses_estimate_when_no_future_filing_known():
    events = pd.DataFrame({
        "filed": [date(2024, 2, 14), date(2024, 5, 1), date(2024, 8, 1), date(2024, 11, 1)],
        "form": ["10-K", "10-Q", "10-Q", "10-Q"],
    })
    snapshot = earnings_snapshot(events, date(2025, 1, 20))
    assert snapshot["days_to_earnings"] is not None
    assert snapshot["days_to_earnings"] >= 0
    assert snapshot["days_to_earnings_is_estimate"] is True
    assert snapshot["next_earnings_date"] is not None
    assert snapshot["next_earnings_date"] > date(2025, 1, 20)


def test_earnings_snapshot_prefers_real_future_filing_over_estimate():
    """When a later filing IS already known (the backtesting case), that
    real date wins over the cadence-based estimate."""
    events = build_filing_events(_fake_submissions())
    snapshot = earnings_snapshot(events, date(2024, 6, 1))
    assert snapshot["days_to_earnings"] == (date(2024, 8, 1) - date(2024, 6, 1)).days
    assert snapshot["next_earnings_date"] == date(2024, 8, 1)
    assert snapshot["days_to_earnings_is_estimate"] is False
