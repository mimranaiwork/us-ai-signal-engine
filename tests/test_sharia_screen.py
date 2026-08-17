"""
Tests the DIY Sharia-compliance screen (features/compliance/sharia_screen.py).
Mocks the SEC EDGAR client rather than hitting the network, same pattern as
tests/test_signal_engine_earnings.py — runs offline/deterministically.
"""

from datetime import date
from unittest.mock import MagicMock

from features.compliance.sharia_screen import (
    DEBT_TO_MARKET_CAP_MAX,
    business_activity_screen,
    financial_ratio_screen,
    screen_symbol,
)


# ── Business activity ───────────────────────────────────────────────────────

def test_business_activity_excludes_banks():
    result = business_activity_screen("BANKS")
    assert result["passed"] is False
    assert "BANKS" in result["reason"]


def test_business_activity_excludes_insurance():
    assert business_activity_screen("INSURANCE")["passed"] is False


def test_business_activity_excludes_tobacco():
    assert business_activity_screen("TOBACCO")["passed"] is False


def test_business_activity_passes_ordinary_industry():
    result = business_activity_screen("SOFTWARE")
    assert result["passed"] is True
    assert result["reason"] is None


# ── Financial ratio screen ──────────────────────────────────────────────────

def test_financial_ratio_screen_all_pass():
    values = {
        "long_term_debt": 10_000_000, "short_term_debt": 5_000_000,
        "cash": 20_000_000, "short_term_investments": 5_000_000,
        "interest_income": 100_000, "revenue": 1_000_000_000,
    }
    result = financial_ratio_screen(values, market_cap=1_000_000_000)
    assert result["passed"] is True
    assert result["fully_determined"] is True
    assert result["checks"]["debt_to_market_cap"]["value"] == 0.015


def test_financial_ratio_screen_fails_on_high_debt():
    values = {
        "long_term_debt": 500_000_000, "short_term_debt": 0,
        "cash": 1_000_000, "short_term_investments": 0,
        "interest_income": 0, "revenue": 1_000_000_000,
    }
    result = financial_ratio_screen(values, market_cap=1_000_000_000)
    assert result["checks"]["debt_to_market_cap"]["value"] == 0.5
    assert result["checks"]["debt_to_market_cap"]["passed"] is False
    assert result["passed"] is False


def test_financial_ratio_screen_boundary_is_exclusive():
    """Exactly at the threshold should fail (< not <=) — matches the
    common AAOIFI convention of a strict inequality."""
    values = {
        "long_term_debt": DEBT_TO_MARKET_CAP_MAX * 1_000_000_000, "short_term_debt": 0,
        "cash": 0, "short_term_investments": 0, "interest_income": 0, "revenue": 1,
    }
    result = financial_ratio_screen(values, market_cap=1_000_000_000)
    assert result["checks"]["debt_to_market_cap"]["passed"] is False


def test_financial_ratio_screen_missing_data_is_unknown_not_pass():
    """No debt figures extracted at all -> unknown, never silently assumed
    to be zero debt (a real filer with zero debt and a filer whose data
    just wasn't extracted look identical from this data alone)."""
    values = {"revenue": 1_000_000_000}
    result = financial_ratio_screen(values, market_cap=1_000_000_000)
    assert result["checks"]["debt_to_market_cap"]["known"] is False
    assert result["checks"]["debt_to_market_cap"]["passed"] is None
    assert result["fully_determined"] is False
    # Unknown ratios don't fail the overall screen by themselves.
    assert result["passed"] is True


def test_financial_ratio_screen_missing_market_cap():
    values = {"long_term_debt": 1000, "revenue": 1000}
    result = financial_ratio_screen(values, market_cap=None)
    assert result["checks"]["debt_to_market_cap"]["known"] is False
    assert result["fully_determined"] is False


def test_financial_ratio_screen_interest_income_ratio():
    values = {"interest_income": 60_000_000, "revenue": 1_000_000_000}
    result = financial_ratio_screen(values, market_cap=None)
    assert result["checks"]["interest_income_to_revenue"]["value"] == 0.06
    assert result["checks"]["interest_income_to_revenue"]["passed"] is False  # 6% > 5% max


# ── screen_symbol (business activity short-circuit + mocked SEC client) ────

def test_screen_symbol_short_circuits_on_business_activity():
    """A bank should never even query SEC EDGAR — financial ratios are moot
    once business activity already fails."""
    fake_client = MagicMock()
    result = screen_symbol("JPM", "BANKS", market_cap=600_000_000_000, sec_client=fake_client, as_of_date=date(2026, 1, 1))

    assert result["compliant"] is False
    assert result["fully_determined"] is True
    assert result["financial_ratios"] is None
    fake_client.company_facts.assert_not_called()


def test_screen_symbol_passes_with_clean_ratios():
    fake_client = MagicMock()
    fake_client.company_facts.return_value = {
        "facts": {
            "us-gaap": {
                "LongTermDebtNoncurrent": {"units": {"USD": [
                    {"val": 10_000_000, "form": "10-K", "filed": "2025-02-01", "end": "2024-12-31"},
                ]}},
                "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [
                    {"val": 20_000_000, "form": "10-K", "filed": "2025-02-01", "end": "2024-12-31"},
                ]}},
                "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
                    {"val": 1_000_000_000, "form": "10-K", "filed": "2025-02-01",
                     "start": "2024-01-01", "end": "2024-12-31"},
                ]}},
            }
        }
    }
    result = screen_symbol("TESTCO", "SOFTWARE", market_cap=1_000_000_000, sec_client=fake_client, as_of_date=date(2026, 1, 1))

    assert result["compliant"] is True
    assert result["business_activity"]["passed"] is True
    assert result["financial_ratios"]["passed"] is True


def test_screen_symbol_degrades_gracefully_when_sec_unavailable():
    """SEC EDGAR unreachable/misconfigured must never crash the screen —
    same graceful-degradation contract as the rest of this project's
    external-data integrations (signals/signal_engine.py::_earnings_timing)."""
    fake_client = MagicMock()
    fake_client.company_facts.side_effect = RuntimeError("no user agent configured")

    result = screen_symbol("TESTCO", "SOFTWARE", market_cap=1_000_000_000, sec_client=fake_client, as_of_date=date(2026, 1, 1))

    assert result["fully_determined"] is False
    assert result["note"] is not None


def test_screen_symbol_warns_strongly_for_finance_adjacent_zero_coverage():
    """Regression test for a real finding: Goldman Sachs/BlackRock/Visa/
    Mastercard all came back with EVERY ratio unknown (bank-style XBRL tags
    this screen's concept list doesn't recognize) — that specific case must
    get a stronger warning than the generic 'one or more ratios unknown'
    note, since 'compliant: True' there reflects total absence of data."""
    fake_client = MagicMock()
    fake_client.company_facts.return_value = {"facts": {"us-gaap": {}}}  # nothing extractable at all

    result = screen_symbol("GS", "ASSET_MANAGEMENT", market_cap=140_000_000_000, sec_client=fake_client, as_of_date=date(2026, 1, 1))

    assert result["compliant"] is True  # no KNOWN failure -> not blocked
    assert result["fully_determined"] is False
    assert "bank-style XBRL tags" in result["note"]
    assert "verify independently" in result["note"]


def test_screen_symbol_ordinary_industry_gets_generic_warning_not_strong_one():
    """The stronger warning is specific to ASSET_MANAGEMENT/PAYMENTS — an
    ordinary industrial/tech company with genuinely missing data (e.g. a
    small/young filer) should get the plain, less alarming note."""
    fake_client = MagicMock()
    fake_client.company_facts.return_value = {"facts": {"us-gaap": {}}}

    result = screen_symbol("TESTCO", "SOFTWARE", market_cap=1_000_000_000, sec_client=fake_client, as_of_date=date(2026, 1, 1))

    assert "bank-style XBRL tags" not in result["note"]
    assert "could not be computed" in result["note"]
