"""
Tests the earnings-risk wiring added to signals/signal_engine.py (spec
section 40). Mocks SecEdgarClient rather than hitting the network, so this
runs offline/deterministically like the rest of the suite — the SEC client
itself is already tested against live data in tests/test_fundamentals.py's
sibling script (scripts/fetch_sec_fundamentals.py) and doesn't need
re-verifying here.
"""

from datetime import date
from unittest.mock import MagicMock, patch

from signals.signal_engine import _earnings_timing


def _fake_submissions(filed_dates: list[str], forms: list[str]) -> dict:
    return {"filings": {"recent": {"form": forms, "filingDate": filed_dates}}}


def test_earnings_timing_returns_data_when_sec_available():
    fake_client = MagicMock()
    fake_client.submissions.return_value = _fake_submissions(
        ["2024-02-14", "2024-05-01", "2024-08-01", "2024-11-01"],
        ["10-K", "10-Q", "10-Q", "10-Q"],
    )
    with patch("ingestion.sec_edgar.SecEdgarClient", return_value=fake_client):
        result = _earnings_timing("AAPL", date(2025, 1, 20))

    assert result["days_since_earnings"] is not None
    assert result["days_to_earnings"] is not None
    assert result["days_to_earnings"] >= 0


def test_earnings_timing_degrades_gracefully_when_sec_unavailable():
    """No SEC_EDGAR_USER_AGENT configured -> SecEdgarClient() raises ValueError
    at construction. This must never break signal generation — earnings
    timing is a risk enhancement, not a hard requirement."""
    with patch("ingestion.sec_edgar.SecEdgarClient", side_effect=ValueError("no user agent configured")):
        result = _earnings_timing("AAPL", date(2025, 1, 20))

    assert result == {
        "days_since_earnings": None,
        "days_to_earnings": None,
        "days_to_earnings_is_estimate": None,
        "next_earnings_date": None,
    }


def test_earnings_timing_degrades_gracefully_on_network_error():
    fake_client = MagicMock()
    fake_client.submissions.side_effect = RuntimeError("connection timed out")
    with patch("ingestion.sec_edgar.SecEdgarClient", return_value=fake_client):
        result = _earnings_timing("AAPL", date(2025, 1, 20))

    assert result["days_since_earnings"] is None
    assert result["days_to_earnings"] is None
