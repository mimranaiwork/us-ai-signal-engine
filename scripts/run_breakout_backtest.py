"""
Backtest the execution layer's breakout strategy on real daily prices.

Downloads split/dividend-adjusted daily bars via yfinance for the Shariah
universe (data/universe/shariah_spus.txt) plus SPUS and SPY, then runs
backtesting/breakout_backtest.py twice: with the SPUS market filter (what
the live bot does) and without it, to show what the filter contributes.

Writes backtesting/breakout_backtest_report.md (+ .json, trades .csv).
Needs internet access to Yahoo Finance: runs on GitHub Actions
(.github/workflows/breakout-backtest.yml), not inside sandboxes that block it.

Usage: python scripts/run_breakout_backtest.py --start 2019-01-01
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd
import yaml

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from backtesting.breakout_backtest import run, trades_frame, yearly_returns  # noqa: E402
from execution import breakout  # noqa: E402
from execution.shariah_universe import FALLBACK_PATH, read_symbol_file  # noqa: E402

OUT = os.path.join(ROOT, "backtesting")


def load_cfg() -> dict:
    with open(os.path.join(ROOT, "config", "settings.yaml")) as f:
        t = (yaml.safe_load(f) or {}).get("trading", {}) or {}
    return {**breakout.DEFAULTS, **(t.get("rules") or {})}


def download(symbols: list[str], start: str) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out: dict[str, pd.DataFrame] = {}
    ymap = {s: s.replace(".", "-") for s in symbols}
    batch = list(ymap.values())
    for i in range(0, len(batch), 50):
        chunk = batch[i:i + 50]
        for attempt in range(3):
            try:
                raw = yf.download(chunk, start=start, auto_adjust=True, group_by="ticker",
                                  threads=True, progress=False)
                break
            except Exception as exc:
                print(f"download retry {attempt + 1}: {exc}")
                time.sleep(5)
        else:
            continue
        for s, ys in ymap.items():
            if ys not in chunk:
                continue
            try:
                df = raw[ys] if isinstance(raw.columns, pd.MultiIndex) else raw
            except KeyError:
                continue
            df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
            if len(df) > 250:
                out[s] = df
    return out


def pct(x):
    return f"{x * 100:.1f}%"


def section(name, res):
    m, b, s = res["metrics"], res["benchmark"] or {}, res["summary"]
    L = [f"### {name}", "",
         "| | Strategy | " + (f"Buy & hold {b.get('symbol', '')}" if b else "Benchmark") + " |", "|---|---|---|",
         f"| Total return | {pct(m.get('total_return', 0))} | {pct(b.get('total_return', 0))} |",
         f"| CAGR | {pct(m.get('cagr', 0))} | {pct(b.get('cagr', 0))} |",
         f"| Max drawdown | {pct(m.get('max_drawdown', 0))} | {pct(b.get('max_drawdown', 0))} |",
         f"| Sharpe | {m.get('sharpe', 0):.2f} | {b.get('sharpe', 0):.2f} |",
         f"| Annual volatility | {pct(m.get('annualized_volatility', 0))} | {pct(b.get('annualized_volatility', 0))} |", "",
         f"Trades: **{s['trades']}** · win rate **{pct(s['win_rate'])}** · avg win **{s['avg_win_r']:+.2f}R** · "
         f"avg loss **{s['avg_loss_r']:+.2f}R** · expectancy **{s['expectancy_r']:+.2f}R per trade** · "
         f"avg hold {s['avg_hold_sessions']:.0f} sessions",
         f"Time with ≥1 position: {pct(s['time_in_market'])} · avg positions held {s['avg_positions']:.1f} · "
         f"market filter on {pct(s['market_filter_on_pct'])} of days", "",
         "Exit reasons: " + ", ".join(f"{k} {v}" for k, v in s["exit_reasons"].items()), "",
         "| Year | Strategy | " + (b.get("symbol", "Benchmark")) + " |", "|---|---|---|"]
    yr = yearly_returns(res["equity"]["equity"])
    by = res.get("bench_yearly")
    for y, v in yr.items():
        L.append(f"| {y} | {pct(v)} | {pct(by.get(y, float('nan'))) if by is not None else '—'} |")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01-01", help="first trading day of the test")
    ap.add_argument("--end", default=None)
    ap.add_argument("--capital", type=float, default=100_000)
    args = ap.parse_args()

    cfg = load_cfg()
    universe = read_symbol_file(FALLBACK_PATH)
    warmup = (pd.Timestamp(args.start) - pd.Timedelta(days=420)).strftime("%Y-%m-%d")
    t0 = time.time()
    frames = download(universe + ["SPUS", "SPY"], warmup)
    print(f"downloaded {len(frames)} symbols in {time.time() - t0:.0f}s")
    end = args.end or str(max(df.index.max() for df in frames.values()).date())

    results = {}
    for name, use_filter in [("With SPUS market filter (live bot setting)", True),
                             ("Without market filter", False)]:
        t0 = time.time()
        res = run(frames, [s for s in universe if s in frames], cfg, args.start, end,
                  initial_capital=args.capital, use_market_filter=use_filter)
        bsym = (res["benchmark"] or {}).get("symbol")
        if bsym:
            bc = frames[bsym]["close"]
            bc.index = (bc.index.tz_localize(None) if bc.index.tz is not None else bc.index).normalize()
            res["bench_yearly"] = yearly_returns(bc.loc[res["start"]:res["end"]])
        results[name] = res
        print(f"{name}: {res['summary']['trades']} trades in {time.time() - t0:.0f}s")

    missing = sorted(set(universe) - set(frames))
    L = ["# Breakout strategy backtest", "",
         f"Period **{results[next(iter(results))]['start']} → {results[next(iter(results))]['end']}**, "
         f"${args.capital:,.0f} starting capital, {len(universe) - len(missing)} of {len(universe)} Shariah (SPUS) "
         f"stocks with data. Same rule functions as the live bot (`execution/breakout.py`), signals at the close, "
         f"fills at the next open, stops checked against each day's low, 3 bps cost per side.", "",
         "**Read with care:** the universe is *today's* Shariah list applied to the past (survivorship bias: "
         "stocks that dropped out, often after falling, are missing), which flatters absolute returns. "
         "Earnings guard, AI filter and daily-loss brake are not simulated. "
         "Past results don't predict future returns. Not financial advice.", ""]
    for name, res in results.items():
        L += section(name, res) + [""]
    if missing:
        L += [f"No price data for: {', '.join(missing)}", ""]
    with open(os.path.join(OUT, "breakout_backtest_report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(OUT, "breakout_backtest_report.json"), "w") as f:
        json.dump({n: {"metrics": r["metrics"], "benchmark": r["benchmark"], "summary": r["summary"],
                       "start": r["start"], "end": r["end"]} for n, r in results.items()}, f, indent=2, default=str)
    trades_frame(results[next(iter(results))]["trades"]).to_csv(
        os.path.join(OUT, "breakout_backtest_trades.csv"), index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
