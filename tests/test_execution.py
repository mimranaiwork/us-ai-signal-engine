"""Execution layer: breakout rules, sizing, exits, Shariah universe,
earnings guard, AI filter, and one full daily cycle against a fake broker.
No network, no Alpaca account, no trained models needed."""

from datetime import date, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from execution import ai_filter, breakout, earnings_guard
from execution.shariah_universe import (
    build_universe, diy_non_compliant, parse_holdings_csv, read_symbol_file, FALLBACK_PATH,
)

CFG = dict(breakout.DEFAULTS)
UP = list(np.linspace(50, 100, 220))                       # long uptrend
BASE = [100 + 3 * np.sin(i / 2) for i in range(30)]        # ~7% deep base, top ~103.5


def frame(closes, last_vol=None, last_range=None, end="2026-10-30"):
    c = np.array(closes, float)
    v = np.full(len(c), 1e6)
    if last_vol:
        v[-1] = last_vol
    hi, lo = c * 1.005, c * 0.995
    if last_range:
        hi[-1], lo[-1] = c[-1] + last_range[1], c[-1] - last_range[0]
    idx = pd.date_range(end=pd.Timestamp(end + " 20:00", tz="UTC"), periods=len(c), freq="B")
    return pd.DataFrame({"open": c, "high": hi, "low": lo, "close": c, "volume": v}, idx)


def enriched(*a, **k):
    return breakout.add_breakout_indicators(frame(*a, **k), CFG)


# ── scan ──────────────────────────────────────────────────────────────────────
def test_breakout_on_volume_detected_with_stop_below_entry():
    info = breakout.classify(enriched(UP + BASE + [105.5], last_vol=2.5e6, last_range=(1.0, 0.3)), CFG)
    assert info["status"] == "breakout"
    assert info["stop"] < info["close"]
    assert (info["close"] - info["stop"]) / info["close"] <= CFG["max_stop_pct"]


@pytest.mark.parametrize("closes,vol,why", [
    (UP + BASE + [105.5], 1.1e6, "volume too low"),
    (UP + BASE + [112.0], 3e6, "too extended"),
    (list(np.linspace(150, 100, 250)) + [101], 3e6, "downtrend"),
    (UP + [100 + 12 * np.sin(i / 2) for i in range(30)] + [113], 3e6, "base too loose"),
])
def test_non_setups_rejected(closes, vol, why):
    assert breakout.classify(enriched(closes, last_vol=vol), CFG) is None, why


def test_near_breakout_goes_to_watchlist():
    assert breakout.classify(enriched(UP + BASE + [102.0]), CFG)["status"] == "near"


def test_breakout_level_uses_only_prior_sessions():
    """No leakage: today's own high must not be part of today's breakout level."""
    df = enriched(UP + BASE + [130.0], last_vol=3e6)
    assert df["breakout_level"].iloc[-1] < 110


def test_market_filter():
    assert breakout.market_filter_ok(enriched(list(np.linspace(40, 50, 250))))
    assert not breakout.market_filter_ok(enriched(list(np.linspace(50, 40, 250))))
    assert not breakout.market_filter_ok(None)


# ── sizing ────────────────────────────────────────────────────────────────────
def test_sizing_risk_cap_and_cash_only():
    assert breakout.position_size(105.5, 100, 10_000, 10_000, CFG) == 18   # $100 risk / $5.5
    assert breakout.position_size(105.5, 104, 10_000, 10_000, CFG) == 18   # 20% cap binds
    assert breakout.position_size(105.5, 100, 10_000, 500, CFG) == 4       # cash binds, no margin
    assert breakout.position_size(105.5, 100, 10_000, -50, CFG) == 0       # negative cash never borrows
    assert breakout.position_size(100, 101, 10_000, 10_000, CFG) == 0      # stop above entry


# ── exits ─────────────────────────────────────────────────────────────────────
def _pos(df, sessions_ago=2):
    return {"entry": 105.5, "stop": 100.0, "risk": 5.5, "breakout_level": 103.5,
            "date": str(df.index[-1 - sessions_ago].date()), "high_close": 105.5,
            "partial": False, "earnings_cut": False}


def test_failed_breakout_exits_everything():
    df = enriched(UP + BASE + [105.5, 104, 99.5])
    actions, _ = breakout.manage(_pos(df), df, 18, CFG)
    assert actions == [("sell", 18, actions[0][2])] and "failed" in actions[0][2]


def test_partial_at_2r_then_trailing_stop_only_rises():
    df = enriched(UP + BASE + [105.5, 110, 117])
    actions, p = breakout.manage(_pos(df), df, 18, CFG)
    assert actions[0][1] == 9 and p["partial"] and p["stop"] >= 105.5
    df2 = enriched(UP + BASE + [105.5, 110, 117, 125])
    actions2, p2 = breakout.manage(p, df2, 9, CFG)
    assert actions2 == [] and p2["stop"] > p["stop"]
    df3 = enriched(UP + BASE + [105.5, 110, 117, 125, 120])
    _, p3 = breakout.manage(p2, df3, 9, CFG)
    assert p3["stop"] >= p2["stop"]


def test_time_stop_after_flat_period():
    df = enriched(UP + BASE + [105.5] + [106] * 16)
    actions, _ = breakout.manage(_pos(df, sessions_ago=16), df, 18, CFG)
    assert "time stop" in actions[0][2]


def test_earnings_cut_happens_once():
    df = enriched(UP + BASE + [105.5, 106, 107])
    actions, p = breakout.manage(_pos(df), df, 18, CFG, days_to_earnings=3, earnings_window=5)
    assert actions[0][1] == 9 and p["earnings_cut"]
    actions, _ = breakout.manage(p, df, 9, CFG, days_to_earnings=2, earnings_window=5)
    assert actions == []


# ── Shariah universe ──────────────────────────────────────────────────────────
ISSUER_CSV = ("Date,Account,StockTicker,CUSIP,SecurityName,Shares\n"
              "10/30/2026,SPUS,NVDA,1,NVIDIA,10\n10/30/2026,SPUS,BRK/B,2,x,1\n"
              "10/30/2026,SPUS,003654100CVR,3,cvr,1\n10/30/2026,SPUS,Cash&Other,,cash,1\n")


def test_issuer_csv_parsing_drops_non_equities():
    assert parse_holdings_csv(ISSUER_CSV) == ["NVDA", "BRK.B"]


def test_bundled_fallback_list_is_usable():
    syms = read_symbol_file(FALLBACK_PATH)
    assert len(syms) > 200 and "NVDA" in syms


def test_broken_download_falls_back_and_diy_failures_are_removed(tmp_path, monkeypatch):
    fb = tmp_path / "u.txt"
    fb.write_text("# test\nAAPL MSFT XOM\n")
    report = tmp_path / "r.json"
    report.write_text('{"results": [{"symbol": "XOM", "compliant": false}, {"symbol": "AAPL", "compliant": true}]}')
    monkeypatch.setattr("execution.shariah_universe.DIY_REPORT_PATH", str(report))
    assert diy_non_compliant(str(report)) == {"XOM"}

    def boom(url):
        raise OSError("offline")
    syms, source, removed = build_universe(exclude=["MSFT"], fetch=boom, fallback_path=str(fb))
    assert syms == ["AAPL"] and removed == ["XOM"] and "download failed" in source


# ── earnings guard (reuses features/earnings with a fake SEC client) ─────────
def test_earnings_guard_uses_engine_earnings_timing():
    client = SimpleNamespace(submissions=lambda s: {"filings": {"recent": {
        "form": ["10-Q", "10-Q", "10-K", "10-Q"],
        "filingDate": ["2026-02-01", "2026-05-03", "2026-08-01", "2026-11-05"]}}})
    d = earnings_guard.days_to_earnings("X", date(2026, 11, 2), client=client)
    assert d == 3
    assert earnings_guard.blocks_entry(d, 5)
    assert not earnings_guard.blocks_entry(20, 5)


def test_earnings_unknown_is_none_not_a_guess():
    bad = SimpleNamespace(submissions=lambda s: (_ for _ in ()).throw(RuntimeError("no UA")))
    assert earnings_guard.days_to_earnings("X", date(2026, 11, 2), client=bad) is None
    assert not earnings_guard.blocks_entry(None, 5)
    assert earnings_guard.blocks_entry(None, 5, block_if_unknown=True)


# ── AI filter ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("mode,signal,allowed", [
    ("off", {"signal": "SELL"}, True),
    ("veto_sell", {"signal": "SELL"}, False),
    ("veto_sell", {"signal": "HOLD"}, True),
    ("veto_sell", None, True),
    ("require_buy", {"signal": "HOLD"}, False),
    ("require_buy", {"signal": "BUY", "confidence": 0.6}, True),
    ("require_buy", None, False),
])
def test_ai_filter_modes(mode, signal, allowed):
    assert ai_filter.decide(signal, mode)[0] is allowed


def test_ai_filter_unreachable_api_returns_none():
    assert ai_filter.fetch_signal("AAPL", None) is None
    assert ai_filter.fetch_signal("AAPL", "http://127.0.0.1:9") is None


# ── full daily cycle ──────────────────────────────────────────────────────────
class FakeBroker:
    live = False

    def __init__(self, bars):
        self.bars_data, self.pos, self.stops = bars, {}, {}

    def bars(self, syms):
        return {k: v for k, v in self.bars_data.items() if k in syms}

    def account(self):
        return {"equity": 10_000.0, "cash": 10_000.0, "last_equity": 10_000.0}

    def positions(self):
        return dict(self.pos)

    def cancel(self, s):
        self.stops.pop(s, None)

    def market(self, s, side, q):
        px = float(self.bars_data[s]["close"].iloc[-1])
        n = self.pos.get(s, (0, px))[0] + (q if side == "buy" else -q)
        if n:
            self.pos[s] = (n, px)
        else:
            self.pos.pop(s, None)
        return px

    def set_stop(self, s, q, p):
        self.stops[s] = (q, p)
        return True


def test_full_cycle_buy_then_failed_breakout_exit(tmp_path):
    from scripts.run_trading_bot import run
    bars = {"AAA": frame(UP + BASE + [105.5], last_vol=2.5e6, last_range=(1.0, 0.3)),
            "BBB": frame(UP + BASE + [102.0]),
            "CCC": frame(UP + BASE + [105.5], last_vol=2.5e6, last_range=(1.0, 0.3)),
            "SPUS": frame(list(np.linspace(40, 50, 250)))}
    fb = FakeBroker(bars)
    cfg = {**CFG, "benchmark": "SPUS", "exclude": [], "ai_filter_mode": "veto_sell",
           "earnings_window": 5, "block_if_earnings_unknown": False}
    universe = (["AAA", "BBB", "CCC"], "test", [])
    dte = {"AAA": None, "CCC": 2}   # CCC reports in 2 days -> skipped
    now = datetime(2026, 11, 2, 10)

    out = run(fb, cfg, now, dry_run=False, universe=universe, days_to_earnings=lambda s, d: dte.get(s),
              signal_fetcher=lambda s: {"signal": "HOLD"}, state_dir=str(tmp_path))
    assert set(fb.pos) == {"AAA"} and fb.stops["AAA"][0] == 18
    assert any("CCC" in a and "earnings" in a for a in out["acted"])
    assert [s for s, _ in out["near"]] == ["BBB"]
    assert (tmp_path / "report.md").exists()

    fb.bars_data["AAA"] = frame(UP + BASE + [105.5, 99.0], end="2026-11-02")
    out = run(fb, cfg, datetime(2026, 11, 3, 10), dry_run=False, universe=universe,
              days_to_earnings=lambda s, d: dte.get(s), signal_fetcher=lambda s: None, state_dir=str(tmp_path))
    assert fb.pos == {} and out["state"]["positions"] == {}
    assert out["state"]["closed"][0]["symbol"] == "AAA"


def test_dry_run_places_no_orders_and_saves_no_state(tmp_path):
    from scripts.run_trading_bot import run
    bars = {"AAA": frame(UP + BASE + [105.5], last_vol=2.5e6, last_range=(1.0, 0.3)),
            "SPUS": frame(list(np.linspace(40, 50, 250)))}
    fb = FakeBroker(bars)
    cfg = {**CFG, "benchmark": "SPUS", "exclude": [], "ai_filter_mode": "off",
           "earnings_window": 5, "block_if_earnings_unknown": False}
    out = run(fb, cfg, datetime(2026, 11, 2, 10), dry_run=True, universe=(["AAA"], "t", []),
              days_to_earnings=lambda s, d: None, state_dir=str(tmp_path))
    assert fb.pos == {} and not (tmp_path / "state.json").exists()
    assert any("would BUY" in a for a in out["acted"])


def test_partial_fraction_zero_keeps_position_but_still_trails():
    df = enriched(UP + BASE + [105.5, 110, 117])
    cfg = {**CFG, "partial_fraction": 0.0}
    actions, p = breakout.manage(_pos(df), df, 18, cfg)
    assert actions == [] and p["partial"] and p["stop"] >= 105.5
