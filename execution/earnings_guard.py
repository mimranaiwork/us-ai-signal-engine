"""
Earnings guard for the execution layer — reuses the engine's own SEC EDGAR
earnings timing (features/earnings/earnings.py), the same source
signals/signal_engine.py's _earnings_timing() uses for its earnings HOLD
override. Same documented proxy caveat applies: dates come from 10-K/10-Q
filing cadence, typically a few days to two weeks after the press release,
so the window is deliberately generous.

Needs SEC_EDGAR_USER_AGENT; without it (or offline) the answer is None
("unknown"), never a guessed number.
"""

from __future__ import annotations

from datetime import date


def days_to_earnings(symbol: str, as_of: date, client=None) -> int | None:
    try:
        from features.earnings.earnings import build_filing_events, earnings_snapshot
        if client is None:
            from ingestion.sec_edgar import SecEdgarClient
            client = SecEdgarClient()
        snap = earnings_snapshot(build_filing_events(client.submissions(symbol)), as_of)
        d = snap.get("days_to_earnings")
        return int(d) if d is not None else None
    except Exception:
        return None


def blocks_entry(days: int | None, window: int, block_if_unknown: bool = False) -> bool:
    if days is None:
        return block_if_unknown
    return 0 <= days <= window
