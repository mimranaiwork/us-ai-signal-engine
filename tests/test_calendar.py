from datetime import date

from calendar_us.trading_calendar import (
    HORIZON_SESSIONS,
    add_sessions,
    is_early_close,
    is_holiday,
    is_trading_day,
    sessions_between,
    trading_sessions,
)


def test_weekends_are_not_trading_days():
    assert not is_trading_day(date(2024, 1, 6))   # Saturday
    assert not is_trading_day(date(2024, 1, 7))   # Sunday


def test_known_fixed_holidays_2024():
    assert is_holiday(date(2024, 1, 1))    # New Year's Day
    assert is_holiday(date(2024, 7, 4))    # Independence Day
    assert is_holiday(date(2024, 12, 25))  # Christmas


def test_known_floating_holidays_2024():
    assert is_holiday(date(2024, 1, 15))   # MLK Day (3rd Monday Jan)
    assert is_holiday(date(2024, 2, 19))   # Washington's Birthday (3rd Monday Feb)
    assert is_holiday(date(2024, 3, 29))   # Good Friday
    assert is_holiday(date(2024, 5, 27))   # Memorial Day (last Monday May)
    assert is_holiday(date(2024, 9, 2))    # Labor Day (1st Monday Sep)
    assert is_holiday(date(2024, 11, 28))  # Thanksgiving (4th Thursday Nov)


def test_juneteenth_observed_from_2022():
    assert is_holiday(date(2022, 6, 20))   # Juneteenth 2022 fell on Sunday -> observed Monday
    assert not is_holiday(date(2021, 6, 18))  # Not yet an NYSE holiday in 2021


def test_weekend_observance_shifts():
    # July 4, 2021 was a Sunday -> observed Monday July 5.
    assert is_holiday(date(2021, 7, 5))
    assert not is_holiday(date(2021, 7, 4))  # the actual Sunday isn't itself a session anyway


def test_early_close_day_after_thanksgiving():
    assert is_early_close(date(2024, 11, 29))


def test_trading_sessions_excludes_holidays_and_weekends():
    sessions = trading_sessions(date(2024, 12, 23), date(2024, 12, 27))
    # Dec 23 (Mon), Dec 24 (Tue, early close but still a session), Dec 26 (Thu).
    # Dec 25 (Wed, Christmas) and Dec 28-29 (weekend) excluded.
    assert date(2024, 12, 25) not in sessions
    assert date(2024, 12, 24) in sessions
    assert date(2024, 12, 26) in sessions


def test_add_sessions_round_trip():
    start = date(2024, 3, 1)  # a Friday, trading day
    forward = add_sessions(start, 5)
    back = add_sessions(forward, -5)
    assert back == start


def test_sessions_between_matches_add_sessions():
    start = date(2024, 3, 1)
    end = add_sessions(start, 21)
    assert sessions_between(start, end) == 21


def test_horizon_sessions_match_spec():
    assert HORIZON_SESSIONS == {
        "1D": 1, "1W": 5, "2W": 10, "1M": 21, "2M": 42, "3M": 63, "6M": 126,
    }
