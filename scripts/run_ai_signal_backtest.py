"""
Walk-forward trading backtest of the engine's XGBoost signals on the Shariah
universe — see backtesting/ai_signal_backtest.py for the method.

Downloads real daily bars via the project's own YFinanceSource, builds
features/labels with the Phase 1 builders, retrains yearly 2020..latest, and
writes backtesting/ai_signal_backtest_report.md (+ .json).

Runs on GitHub Actions (.github/workflows/ai-signal-backtest.yml); needs
internet access to Yahoo Finance and roughly 30–90 minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from backtesting.ai_signal_backtest import (  # noqa: E402
    composite, decile_spread, purged_splits, rank_ic, simulate, target_weights, weekly_rebalance_days,
)
from calendar_us.trading_calendar import HORIZON_SESSIONS  # noqa: E402
from execution.shariah_universe import FALLBACK_PATH, read_symbol_file  # noqa: E402
from features.market.market_features import BENCHMARKS, add_relative_strength, compute_all_benchmark_features  # noqa: E402
from features.price.price_features import add_price_features  # noqa: E402
from features.technical.indicators import add_all_technical_indicators  # noqa: E402
from labels.horizons import LabelConfig, generate_all_horizons  # noqa: E402
from models.common import feature_columns  # noqa: E402
from models.xgboost.model import XGBoostMultiHorizonModel  # noqa: E402

OUT = os.path.join(ROOT, "backtesting")
PERIODS = {"2020–2023": ("2020-01-01", "2023-12-31"), "2024–latest": ("2024-01-01", None), "Whole period": ("2020-01-01", None)}
TOP_N = 10


def settings():
    with open(os.path.join(ROOT, "config", "settings.yaml")) as f:
        s = yaml.safe_load(f)
    return s["horizons"]["weights"], float(s["signal_thresholds"]["buy"]), s["labels"]["method"]


def fetch(symbols, start, end):
    from ingestion.ohlcv_source import YFinanceSource
    src, out = YFinanceSource(), {}
    for i, s in enumerate(symbols):
        for attempt in range(3):
            try:
                df = src.get_bars(s.replace(".", "-"), start, end)
                break
            except Exception as exc:
                print(f"  {s} retry {attempt + 1}: {exc}")
                time.sleep(3)
        else:
            continue
        if len(df) >= 300:
            out[s] = df.reset_index(drop=True)
        if i % 25 == 0:
            print(f"  fetched {i + 1}/{len(symbols)}")
        time.sleep(0.2)
    return out


def build_frames(bars, universe, label_method):
    bench_feats = compute_all_benchmark_features({b: bars[b] for b in BENCHMARKS})
    feats, labels = [], []
    cfg = LabelConfig(method=label_method)
    for s in universe:
        if s not in bars:
            continue
        df = add_all_technical_indicators(add_price_features(bars[s]))
        for b in BENCHMARKS:
            df = add_relative_strength(df, bench_feats[b], prefix=b)
        lb = generate_all_horizons(df, cfg)[["date", "horizon", "label", "target_session_date"]]
        df["symbol"], lb["symbol"] = s, s
        feats.append(df)
        labels.append(lb)
    F = pd.concat(feats, ignore_index=True)
    L = pd.concat(labels, ignore_index=True)
    F["date"], L["date"] = pd.to_datetime(F["date"]), pd.to_datetime(L["date"])
    L["target_session_date"] = pd.to_datetime(L["target_session_date"])
    return F, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-year", type=int, default=2020)
    ap.add_argument("--data-start", default="2015-01-01")
    ap.add_argument("--max-symbols", type=int, default=0, help="debug: limit universe size")
    args = ap.parse_args()

    weights, buy_threshold, label_method = settings()
    universe = read_symbol_file(FALLBACK_PATH)
    if args.max_symbols:
        universe = universe[: args.max_symbols]
    t0 = time.time()
    bars = fetch(universe + BENCHMARKS + ["SPUS"], date.fromisoformat(args.data_start), date.today())
    print(f"fetched {len(bars)} symbols in {time.time() - t0:.0f}s")

    t0 = time.time()
    F, L = build_frames(bars, universe, label_method)
    feat_cols = [c for c in feature_columns(F) if c not in {"symbol"}]
    # Real data can produce ±inf (e.g. a ratio over a zero-volume day); XGBoost
    # rejects those. Treat them as missing, like warm-up NaNs, and say how many.
    inf_mask = np.isinf(F[feat_cols].to_numpy(dtype=float)).any(axis=1)
    print(f"rows with infinite feature values dropped: {int(inf_mask.sum()):,} of {len(F):,}")
    F = F[~inf_mask]
    F = F.dropna(subset=feat_cols).reset_index(drop=True)
    key = pd.MultiIndex.from_arrays([F["date"], F["symbol"]])
    for h in HORIZON_SESSIONS:   # align each horizon's label + label-known date to the feature rows
        Lh = L[L["horizon"] == h].drop_duplicates(["date", "symbol"]).set_index(["date", "symbol"])
        F[f"label_{h}"] = Lh["label"].reindex(key).to_numpy()
        F[f"target_{h}"] = Lh["target_session_date"].reindex(key).to_numpy()
    del L
    print(f"features: {len(F):,} rows x {len(feat_cols)} cols in {time.time() - t0:.0f}s")

    last_year = int(F["date"].max().year)
    score_parts = []
    fit_log = []
    for year in range(args.first_year, last_year + 1):
        test_rows = F[F["date"].dt.year == year]
        if test_rows.empty:
            continue
        per_h = {}
        for h in HORIZON_SESSIONS:
            meta = pd.DataFrame({"date": F["date"], "label": F[f"label_{h}"],
                                 "target_session_date": pd.to_datetime(F[f"target_{h}"])})
            train, val = purged_splits(meta, year)
            if len(train) < 1000 or len(val) < 200:
                print(f"  {year} {h}: not enough data, skipped")
                continue
            m = XGBoostMultiHorizonModel()
            m.feature_cols = feat_cols
            t1 = time.time()
            rep = m.train_horizon(F.loc[train.index, feat_cols].assign(label=train["label"]),
                                  F.loc[val.index, feat_cols].assign(label=val["label"]), h)
            p = m.predict_proba(test_rows, h)                 # columns: SELL, HOLD, BUY
            per_h[h] = pd.Series(p[:, 2] - p[:, 0], index=test_rows.index)
            fit_log.append({"year": year, "horizon": h, "n_train": rep["n_train"], "n_val": rep["n_val"],
                            "val_accuracy": rep["val_accuracy"], "max_train_label_date": str(train["target_session_date"].max().date())})
            print(f"  {year} {h}: train {rep['n_train']:,} val {rep['n_val']:,} acc {rep['val_accuracy']:.3f} ({time.time() - t1:.0f}s)")
        if per_h:
            sc = composite(per_h, weights)
            score_parts.append(pd.DataFrame({"date": test_rows["date"], "symbol": test_rows["symbol"], "score": sc}))

    scores = pd.concat(score_parts).pivot_table(index="date", columns="symbol", values="score")
    prices = pd.DataFrame({s: bars[s].set_index(pd.to_datetime(bars[s]["date"]))["adj_close"]
                           for s in scores.columns}).sort_index()
    spus = bars["SPUS"].set_index(pd.to_datetime(bars["SPUS"]["date"]))["adj_close"].sort_index()
    prices = prices.loc[prices.index >= scores.index.min() - pd.Timedelta(days=10)]
    reb = weekly_rebalance_days(scores.index)

    W_model = target_weights(scores, TOP_N, buy_threshold, reb)
    have_price = prices.reindex(reb).notna()
    W_equal = have_price.div(have_price.sum(axis=1), axis=0).fillna(0.0)
    end = str(scores.index.max().date())
    ic = rank_ic(scores, prices)

    def m(res):
        return res.metrics

    L_out = ["# AI signal backtest (walk-forward, XGBoost)", "",
             f"{scores.shape[1]} Shariah (SPUS) stocks, {scores.index.min().date()} → {end}. The model is retrained every January "
             f"on purged past data only, scores combine the 7 horizons with the repo's weights, and the portfolio holds the "
             f"**top {TOP_N} stocks scoring ≥ {buy_threshold:+.2f}** (the BUY threshold), rebalanced weekly, traded one day after the "
             "signal, 3 bps per side. Nothing was tuned on these results.", "",
             "**Read with care:** today's Shariah list applied to the past (survivorship bias); XGBoost only (no LightGBM/LSTM/"
             "Transformer); no conservative-HOLD overrides; no sector/macro/fundamental features. Not financial advice.", ""]
    summary = {}
    for pname, (ps, pe) in PERIODS.items():
        pe = pe or end
        rows = {
            f"AI top {TOP_N} (idle cash)": simulate(W_model, prices, ps, pe),
            f"AI top {TOP_N} (idle cash in SPUS)": simulate(W_model, prices, ps, pe, idle_prices=spus),
            "Equal-weight all Shariah stocks": simulate(W_equal, prices, ps, pe),
        }
        bh = spus.loc[ps:pe]
        from backtesting.metrics import trading_metrics
        bhm = trading_metrics(bh / bh.iloc[0] * 100_000)
        L_out += [f"## {pname} ({ps} → {pe})", "", "| Portfolio | CAGR | Max drawdown | Sharpe | Avg invested |",
                  "|---|---|---|---|---|"]
        for n, r in rows.items():
            L_out.append(f"| {n} | {m(r)['cagr']*100:.1f}% | {m(r)['max_drawdown']*100:.1f}% | {m(r)['sharpe']:.2f} | {r.avg_invested*100:.0f}% |")
        L_out.append(f"| Buy & hold SPUS | {bhm['cagr']*100:.1f}% | {bhm['max_drawdown']*100:.1f}% | {bhm['sharpe']:.2f} | 100% |")
        icp = ic.loc[ps:pe]
        L_out += ["", f"Signal quality: average rank IC (score vs next-21-session return) **{icp.mean():+.3f}** "
                      f"(positive on {(icp > 0).mean()*100:.0f}% of days); top-minus-bottom decile 21-session return "
                      f"**{decile_spread(scores.loc[ps:pe], prices)*100:+.2f}%**.", ""]
        summary[pname] = {n: {"metrics": r.metrics, "avg_invested": r.avg_invested, "turnover": r.turnover} for n, r in rows.items()}
        summary[pname]["Buy & hold SPUS"] = {"metrics": bhm}
        summary[pname]["rank_ic_mean"] = float(icp.mean())
    days_with_picks = (W_model.sum(axis=1) > 0).mean()
    L_out += [f"Weeks with at least one stock above the BUY threshold: {days_with_picks*100:.0f}%. "
              f"Average holdings when invested: {(W_model.gt(0).sum(axis=1)[W_model.sum(axis=1) > 0]).mean():.1f} stocks.", "",
              "Purging check: every model's latest training label date is before the year it trades "
              "(see `fit_log` in the JSON).", ""]
    with open(os.path.join(OUT, "ai_signal_backtest_report.md"), "w") as f:
        f.write("\n".join(L_out) + "\n")
    with open(os.path.join(OUT, "ai_signal_backtest_report.json"), "w") as f:
        json.dump({"summary": summary, "fit_log": fit_log, "top_n": TOP_N, "buy_threshold": buy_threshold,
                   "horizon_weights": weights}, f, indent=2, default=str)
    bad = [r for r in fit_log if pd.Timestamp(r["max_train_label_date"]).year >= r["year"]]
    print("\n".join(L_out))
    if bad:
        raise SystemExit(f"PURGING FAILED: {bad}")


if __name__ == "__main__":
    main()
