"""
Earnings-event features (spec section 19), derived from SEC filing history
(10-K/10-Q filing dates) rather than a dedicated earnings-calendar feed,
which the project doesn't have access to. This is a documented
approximation: a company's actual earnings *press release* typically comes a
few days to a couple weeks BEFORE its 10-Q/10-K is filed, so `days_to/since
_earnings` computed from filing dates run slightly late relative to the true
earnings-reaction window. Point-in-time correctness itself is exact (filing
dates are real, public timestamps) — it's the proxy-vs-true-event-date gap
that's the approximation.

True EPS/revenue "surprise" (actual vs analyst consensus) needs a paid
analyst-estimates feed the project doesn't have; `eps_growth_yoy`/
`revenue_growth_yoy` (fundamental momentum, from features/fundamentals) are
provided as the honestly-derivable substitute rather than fabricating a
consensus number.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

EARNINGS_FORMS = {"10-K", "10-Q"}


def build_filing_events(submissions: dict) -> pd.DataFrame:
    """submissions: raw SEC submissions.json for one company. Returns a
    DataFrame of (filed date, form) for every 10-K/10-Q, sorted ascending —
    the proxy "earnings event" calendar."""
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    filed_dates = recent.get("filingDate", [])
    rows = [
        {"filed": pd.to_datetime(d).date(), "form": f}
        for f, d in zip(forms, filed_dates)
        if f in EARNINGS_FORMS
    ]
    df = pd.DataFrame(rows).drop_duplicates().sort_values("filed").reset_index(drop=True)
    return df


def days_to_next_and_since_last(events: pd.DataFrame, as_of_date: date) -> dict:
    """days_since_earnings: from the most recent filing ON OR BEFORE as_of_date.
    days_to_earnings: to the next KNOWN filing after as_of_date — only
    meaningful if there's a later filing already in the (historical) dataset;
    for live/production use this would come from a forward filing-date
    estimate instead (companies file on a fairly regular quarterly cadence),
    which isn't implemented here."""
    if events.empty:
        return {"days_since_earnings": None, "days_to_earnings": None, "next_earnings_date": None}

    past = events[events["filed"] <= as_of_date]
    future = events[events["filed"] > as_of_date]

    days_since = (as_of_date - past["filed"].iloc[-1]).days if not past.empty else None
    next_date = future["filed"].iloc[0] if not future.empty else None
    days_to = (next_date - as_of_date).days if next_date is not None else None

    return {"days_since_earnings": days_since, "days_to_earnings": days_to, "next_earnings_date": next_date}


def earnings_risk_flag(days_to_earnings: int | None, threshold_days: int = 5) -> bool:
    """spec section 40: flag elevated event risk when an earnings filing is
    imminent. Configurable threshold — not hardcoded permanently, per the
    spec's instruction to validate this via event backtesting before relying
    on it (Phase 4's backtesting engine can be pointed at this once there's
    a real earnings-date feed to backtest against). Default widened to 5
    (not 1) trading days for LIVE use via estimate_next_filing_date(), which
    is itself an estimate with some slack — 1 day of margin on an estimate
    would almost never fire.
    """
    return days_to_earnings is not None and days_to_earnings <= threshold_days


def estimate_next_filing_date(events: pd.DataFrame, as_of_date: date) -> date | None:
    """
    Live-use estimate of the NEXT filing date, since (unlike backtesting)
    there's no future filing already in the dataset to look at. Uses the
    median interval between the last few 10-Q/10-K filings (companies file
    quarterly/annually on a fairly regular cadence) added to the most recent
    filing on/before as_of_date. This is explicitly an estimate, not a real
    earnings-calendar feed — see the module docstring.
    """
    past = events[events["filed"] <= as_of_date].sort_values("filed")
    if len(past) < 2:
        return None
    intervals = past["filed"].diff().dropna().apply(lambda d: d.days)
    recent_intervals = intervals.tail(4)  # last few gaps, so a 10-K -> 10-Q short gap doesn't skew it
    if recent_intervals.empty:
        return None
    median_interval = int(recent_intervals.median())
    if median_interval <= 0:
        return None

    # Project forward from the last known filing until the estimate is
    # actually in the future — a stale/overdue single-cycle estimate would
    # otherwise sit in the past and produce a negative days_to_earnings,
    # which would incorrectly read as "imminent" forever. Cap extrapolation
    # at 2 cycles: beyond that the cadence assumption is unreliable (a company
    # may have stopped filing regularly, or there's a data gap) — treat as unknown.
    estimate = past["filed"].iloc[-1] + pd.Timedelta(days=median_interval)
    cycles = 1
    while estimate <= as_of_date:
        if cycles >= 2:
            return None
        estimate += pd.Timedelta(days=median_interval)
        cycles += 1
    return estimate


def earnings_snapshot(events: pd.DataFrame, as_of_date: date) -> dict:
    """Combines days_since_earnings (real, from filing history) with a live-
    usable days_to_earnings (estimated via estimate_next_filing_date when no
    actual future filing is already known — the backtesting case where one
    IS known still takes priority, since that's real not estimated)."""
    timing = days_to_next_and_since_last(events, as_of_date)
    if timing["days_to_earnings"] is None:
        estimated = estimate_next_filing_date(events, as_of_date)
        if estimated is not None:
            timing["days_to_earnings"] = (estimated - as_of_date).days
            timing["next_earnings_date"] = estimated
            timing["days_to_earnings_is_estimate"] = True
        else:
            timing["days_to_earnings_is_estimate"] = None
    else:
        timing["days_to_earnings_is_estimate"] = False
    return timing
