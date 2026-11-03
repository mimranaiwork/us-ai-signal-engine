"""
Compare a small, pre-declared set of breakout-strategy variants without
fitting to the past:

  1. Selection period (2020-01-01 → 2023-12-31): every variant is run; the one
     with the best Sharpe ratio is chosen. Nothing else is tuned.
  2. Test period (2024-01-01 → latest): the chosen variant is then judged on
     data it was never selected on, against the baseline and buy-and-hold SPUS.

The variant list is fixed here, before seeing any results, so the study
can't quietly grow until something looks good.

Writes backtesting/breakout_variants_report.md (+ .json).
Usage: python scripts/run_breakout_variants.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from backtesting.breakout_backtest import run  # noqa: E402
from execution.shariah_universe import FALLBACK_PATH, read_symbol_file  # noqa: E402
from scripts.run_breakout_backtest import coverage_lines, download, get_universe, load_cfg  # noqa: E402

OUT = os.path.join(ROOT, "backtesting")
SELECT = ("2020-01-01", "2023-12-31")
TEST_START = "2024-01-01"

NO_TIME_STOP = {"time_stop_days": 10_000}
LET_RUN = {"partial_fraction": 0.0, "trail_atr": 4.0, **NO_TIME_STOP}
BIGGER = {"risk_per_trade": 0.015, "max_position_pct": 0.33, "max_positions": 4}

VARIANTS = {
    "A. Baseline (live bot today)":            ({}, None),
    "B. No time stop":                          (NO_TIME_STOP, None),
    "C. Let winners run":                       (LET_RUN, None),
    "D. Bigger positions":                      (BIGGER, None),
    "E. Let winners run + bigger positions":    ({**LET_RUN, **BIGGER}, None),
    "F. Baseline + idle cash in SPUS":          ({}, "SPUS"),
    "G. E + idle cash in SPUS":                 ({**LET_RUN, **BIGGER}, "SPUS"),
}
DESCRIPTIONS = {
    "A": "Current settings: sell half at +2R, trail 3×ATR, exit after 15 days under +0.5R, 1% risk, ≤20% per stock, 5 slots.",
    "B": "As A, but no 15-day time exit.",
    "C": "No partial sale, wider 4×ATR trailing stop, no time exit — full position rides the trend.",
    "D": "As A, but 1.5% risk per trade, ≤33% per stock, 4 slots.",
    "E": "C and D combined.",
    "F": "As A, but uninvested cash is held in SPUS instead of sitting idle.",
    "G": "E with uninvested cash held in SPUS.",
}


def pct(x):
    return "—" if x is None or pd.isna(x) else f"{x * 100:.1f}%"


def row(name, r):
    m, s = r["metrics"], r["summary"]
    return (f"| {name} | {pct(m.get('cagr'))} | {pct(m.get('max_drawdown'))} | {m.get('sharpe', 0):.2f} | "
            f"{s['trades']} | {pct(s['win_rate'])} | {s['expectancy_r']:+.2f}R |")


def bench_row(r):
    b = r["benchmark"] or {}
    return f"| Buy & hold {b.get('symbol', 'SPUS')} | {pct(b.get('cagr'))} | {pct(b.get('max_drawdown'))} | {b.get('sharpe', 0):.2f} | — | — | — |"


HEAD = ["| Variant | CAGR | Max drawdown | Sharpe | Trades | Win rate | Avg per trade |", "|---|---|---|---|---|---|---|"]


def main():
    base_cfg = load_cfg()
    universe, membership, uni_label = get_universe()
    t0 = time.time()
    frames = download(universe + ["SPUS", "SPY"], "2018-06-01")
    uni = [s for s in universe if s in frames]
    end = str(max(df.index.max() for df in frames.values()).date())
    print(f"downloaded {len(frames)} symbols in {time.time() - t0:.0f}s; end {end}")

    def go(cfg_over, idle, start, stop):
        return run(frames, uni, {**base_cfg, **cfg_over}, start, stop, idle_cash_in=idle, membership=membership)

    sel = {}
    for name, (over, idle) in VARIANTS.items():
        t0 = time.time()
        sel[name] = go(over, idle, *SELECT)
        print(f"select  {name}: sharpe {sel[name]['metrics'].get('sharpe', 0):.2f} ({time.time() - t0:.0f}s)")
    chosen = max(sel, key=lambda n: sel[n]["metrics"].get("sharpe", float("-inf")))
    baseline = next(iter(VARIANTS))

    test = {}
    for name, (over, idle) in VARIANTS.items():   # all reported for transparency; only `chosen` was picked
        test[name] = go(over, idle, TEST_START, end)
        print(f"test    {name}: sharpe {test[name]['metrics'].get('sharpe', 0):.2f}")

    full = {n: go(*VARIANTS[n], "2020-01-01", end) for n in dict.fromkeys([baseline, chosen])}

    L = ["# Breakout strategy: variant study", "",
         f"Seven variants fixed in advance. Each was run on the **selection period ({SELECT[0]} → {SELECT[1]})**; "
         f"the best Sharpe ratio was chosen; that choice was then checked on the **test period ({TEST_START} → {end})**, "
         "which played no part in choosing it. Same simulator, costs and caveats as `breakout_backtest_report.md`. "
         f"Universe: **{uni_label}**. Earnings guard and AI filter not simulated. Not financial advice.", ""] + coverage_lines(membership, frames, SELECT[0]) + [
         "## Variants", ""] + [f"- **{n}**: {DESCRIPTIONS[n[0]]}" for n in VARIANTS] + [
         "", f"## Selection period ({SELECT[0]} → {SELECT[1]})", ""] + HEAD + [row(n, r) for n, r in sel.items()] + [
         bench_row(sel[baseline]), "", f"**Chosen on Sharpe: {chosen}**", "",
         f"## Test period ({TEST_START} → {end}): unseen data", ""] + HEAD + [
         row(("**" + n + " ← chosen**") if n == chosen else n, r) for n, r in test.items()] + [
         bench_row(test[baseline]), "",
         f"## Whole period (2020-01-01 → {end})", ""] + HEAD + [row(n, r) for n, r in full.items()] + [
         bench_row(full[baseline]), ""]
    with open(os.path.join(OUT, "breakout_variants_report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(OUT, "breakout_variants_report.json"), "w") as f:
        json.dump({"chosen": chosen, "selection_period": SELECT, "test_start": TEST_START, "end": end,
                   "selection": {n: {"metrics": r["metrics"], "summary": r["summary"]} for n, r in sel.items()},
                   "test": {n: {"metrics": r["metrics"], "summary": r["summary"], "benchmark": r["benchmark"]}
                            for n, r in test.items()}}, f, indent=2, default=str)
    print("\n".join(L))


if __name__ == "__main__":
    main()
