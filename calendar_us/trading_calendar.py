"""
Self-contained NYSE/Nasdaq trading-session calendar.

Deliberately dependency-free (stdlib + pandas only, no `pandas_market_calendars`)
so Phase 1 has no external calendar-data package to fail to install or go stale.
Holidays are computed programmatically from the actual NYSE holiday rules, not
hardcoded per year, so the calendar remains correct for any year in range.

NYSE full-day holidays:
    New Year's Day       (Jan 1, weekend-observed)
    Martin Luther King   (3rd Monday of January)
    Washington's Birthday (3rd Monday of February)
    Good Friday          (Friday before Easter Sunday)
    Memorial Day         (last Monday of May)
    Juneteenth           (Jun 19, weekend-observed; NYSE holiday since 2022)
    Independence Day     (Jul 4, weekend-observed)
    Labor Day            (1st Monday of September)
    Thanksgiving         (4th Thursday of November)
    Christmas            (Dec 25, weekend-observed)

Early-close (half-day) sessions, 1:00pm ET close instead of 4:00pm:
    Day after Thanksgiving
    Christmas Eve (Dec 24), when it falls on a weekday
    July 3rd, when it falls on a weekday and July 4th is observed on its actual date

These are the standard rules; exchange-announced ad-hoc closures (e.g. September
11, 2001, hurricanes) are NOT modeled here — extend `EXTRA_CLOSURES` if needed.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

# Ad-hoc closures not derivable from the standard holiday rules above.
EXTRA_CLOSURES: set[date] = {
    date(2001, 9, 11), date(2001, 9, 12), date(2001, 9, 13), date(2001, 9, 14),
    date(2012, 10, 29),  # Hurricane Sandy
    date(2012, 10, 30),  # Hurricane Sandy
    date(2018, 12, 5),   # National day of mourning, George H.W. Bush
}


def _easter_sunday(year: int) -> date:
    """Anonymous Gregorian algorithm (Meeus/Jones/Butcher)."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """weekday: Monday=0 .. Sunday=6. n=1 -> first occurrence, n=-1 -> last."""
    if n > 0:
        d = date(year, month, 1)
        offset = (weekday - d.weekday()) % 7
        d += timedelta(days=offset + 7 * (n - 1))
        return d
    # last occurrence: start from the last day of the month and walk backward
    next_month = date(year + (month == 12), month % 12 + 1, 1)
    d = next_month - timedelta(days=1)
    offset = (d.weekday() - weekday) % 7
    return d - timedelta(days=offset)


def _observed(d: date) -> date:
    """Weekend-observance rule: Saturday -> preceding Friday, Sunday -> following Monday."""
    if d.weekday() == 5:  # Saturday
        return d - timedelta(days=1)
    if d.weekday() == 6:  # Sunday
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=None)
def _holidays_for_year(year: int) -> frozenset[date]:
    holidays = {
        _observed(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),          # MLK — 3rd Monday Jan
        _nth_weekday(year, 2, 0, 3),          # Washington's Birthday — 3rd Monday Feb
        _easter_sunday(year) - timedelta(days=2),  # Good Friday
        _nth_weekday(year, 5, 0, -1),         # Memorial Day — last Monday May
        _nth_weekday(year, 9, 0, 1),          # Labor Day — 1st Monday Sep
        _nth_weekday(year, 11, 3, 4),         # Thanksgiving — 4th Thursday Nov
        _observed(date(year, 7, 4)),
        _observed(date(year, 12, 25)),
    }
    if year >= 2022:  # NYSE observed Juneteenth starting 2022
        holidays.add(_observed(date(year, 6, 19)))
    return frozenset(holidays)


@lru_cache(maxsize=None)
def _early_close_for_year(year: int) -> frozenset[date]:
    thanksgiving = _nth_weekday(year, 11, 3, 4)
    early = {thanksgiving + timedelta(days=1)}
    dec24 = date(year, 12, 24)
    if dec24.weekday() < 5:
        early.add(dec24)
    jul3 = date(year, 7, 3)
    if jul3.weekday() < 5 and date(year, 7, 4).weekday() < 5:
        early.add(jul3)
    return frozenset(early)


def is_holiday(d: date) -> bool:
    return d in _holidays_for_year(d.year) or d in EXTRA_CLOSURES


def is_early_close(d: date) -> bool:
    return d in _early_close_for_year(d.year)


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and not is_holiday(d)


def trading_sessions(start: date, end: date) -> list[date]:
    """All trading sessions in [start, end], inclusive, ascending order."""
    if end < start:
        return []
    days = []
    d = start
    one_day = timedelta(days=1)
    while d <= end:
        if is_trading_day(d):
            days.append(d)
        d += one_day
    return days


def add_sessions(d: date, n: int) -> date:
    """
    The trading session `n` sessions after (n>0) or before (n<0) `d`.
    `d` itself does not need to be a trading day. n=0 returns the next
    trading session on/after d (or on/before d if n==0 and d isn't a session
    — callers computing horizons should treat T as already a valid session).
    """
    step = 1 if n >= 0 else -1
    remaining = abs(n)
    cur = d
    # Land on a trading day first if d itself isn't one.
    while not is_trading_day(cur):
        cur += timedelta(days=step)
        if n == 0:
            return cur
    while remaining > 0:
        cur += timedelta(days=step)
        if is_trading_day(cur):
            remaining -= 1
    return cur


def sessions_between(d1: date, d2: date) -> int:
    """Number of trading sessions strictly between d1 (exclusive) and d2 (inclusive)."""
    if d2 < d1:
        return -sessions_between(d2, d1)
    return len(trading_sessions(d1 + timedelta(days=1), d2))


# Standard horizon definitions, in trading sessions (not calendar days).
HORIZON_SESSIONS: dict[str, int] = {
    "1D": 1,
    "1W": 5,
    "2W": 10,
    "1M": 21,
    "2M": 42,
    "3M": 63,
    "6M": 126,
}
