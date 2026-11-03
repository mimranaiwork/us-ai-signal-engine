"""
Walk-forward trading backtest of the engine's own XGBoost signals.

backtesting/walk_forward.py measures classification accuracy per fold; this
asks the trading question instead: if you held the stocks the model scores
highest, would you have beaten simply holding SPUS, or simply holding every
Shariah stock equally?

Method (fixed in advance; nothing here is tuned on results):
  * Features and labels: the same Phase 1 builders as
    scripts/build_phase1_dataset.py (price + technical + SPY/QQQ/IWM/DIA
    relative strength; volatility-scaled labels for all seven horizons).
  * Yearly walk-forward retrain: to trade year Y, one XGBoostMultiHorizonModel
    per horizon is trained on rows whose label was fully known before
    Y-1-01-01 and validated (early stopping) on rows whose label was known
    before Y-01-01. Labels whose future window reaches past a cut-off are
    PURGED, so no training label peeks into the year being traded.
  * Score: per-horizon P(BUY) - P(SELL), combined with config/settings.yaml's
    horizon weights — the same composite signals/scoring.py builds.
  * Portfolio: every Friday close, hold the top N stocks whose score clears the
    BUY threshold (config signal_thresholds.buy), equal-weighted; trade at the
    next session's close (one-day lag); 3 bps per side on turnover. Empty
    slots stay in cash, or (second variant) in SPUS.

Not simulated: LightGBM/LSTM/Transformer (only XGBoost is walked forward,
as in the repo's own harness), the engine's conservative-HOLD overrides,
sector/industry/macro/fundamental features (need per-symbol reference data
the SPUS list doesn't carry). Survivorship bias as in the breakout study.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtesting.metrics import trading_metrics

EXTRA_EXCLUDE = {"symbol", "label", "target_session_date"}


def purged_splits(frame: pd.DataFrame, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """frame: one horizon's rows with date, target_session_date, label + features.
    train: labels fully known before Y-1-01-01; val: rows dated in Y-1 whose
    labels are known before Y-01-01. Nothing whose label needs year-Y prices."""
    cut_train, cut_val = pd.Timestamp(year - 1, 1, 1), pd.Timestamp(year, 1, 1)
    known = frame["label"].notna() & frame["target_session_date"].notna()
    train = frame[known & (frame["target_session_date"] < cut_train)]
    val = frame[known & (frame["date"] >= cut_train) & (frame["target_session_date"] < cut_val)]
    return train, val


def composite(scores: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    """Weighted mean of per-horizon (P(BUY) - P(SELL)) — signals/scoring.composite_score, vectorised."""
    total = sum(weights.get(h, 0.0) for h in scores)
    out = sum(s * weights.get(h, 0.0) for h, s in scores.items())
    return out / total if total else out


def target_weights(score_wide: pd.DataFrame, top_n: int, threshold: float,
                   rebalance_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Equal weights (1/top_n each) for the top_n scores >= threshold on each
    rebalance day; rows only on rebalance days."""
    rows = {}
    for d in rebalance_days:
        if d not in score_wide.index:
            continue
        s = score_wide.loc[d].dropna()
        picks = s[s >= threshold].nlargest(top_n).index
        w = pd.Series(0.0, index=score_wide.columns)
        w[picks] = 1.0 / top_n
        rows[d] = w
    return pd.DataFrame(rows).T.sort_index()


def weekly_rebalance_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each calendar week."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.isocalendar().year, index.isocalendar().week]).max().values)


@dataclass
class SimResult:
    equity: pd.Series
    turnover: float
    avg_invested: float
    metrics: dict


def simulate(weights_on_rebalance: pd.DataFrame, prices: pd.DataFrame, start, end,
             cost_per_side: float = 0.0003, idle_prices: pd.Series | None = None,
             initial_capital: float = 100_000.0) -> SimResult:
    """Signals at close t are traded at close t+1 and earn returns from t+2:
    the effective weight row for date t+2 is the rebalance target from t."""
    rets = prices.pct_change().fillna(0.0)
    w = weights_on_rebalance.reindex(prices.index).ffill().fillna(0.0)
    w_eff = w.shift(2).fillna(0.0)                   # one-day trade lag + returns start next day
    w_trade = w.shift(1).fillna(0.0)                 # the day the trade happens
    turnover = w_trade.diff().abs().sum(axis=1).fillna(w_trade.abs().sum(axis=1))
    idle = (1.0 - w_eff.sum(axis=1)).clip(lower=0.0)
    daily = (w_eff * rets[w_eff.columns]).sum(axis=1)
    cost = turnover * cost_per_side
    if idle_prices is not None:
        ir = idle_prices.reindex(prices.index).pct_change().fillna(0.0)
        daily = daily + idle * ir
        idle_trade = (1.0 - w_trade.sum(axis=1)).clip(lower=0.0)
        cost = cost + idle_trade.diff().abs().fillna(0.0) * cost_per_side
    daily = (daily - cost).loc[pd.Timestamp(start):pd.Timestamp(end)]
    eq = initial_capital * (1 + daily).cumprod()
    span = slice(pd.Timestamp(start), pd.Timestamp(end))
    return SimResult(eq, float(turnover.loc[span].sum()), float(w_eff.sum(axis=1).loc[span].mean()),
                     trading_metrics(eq) if len(eq) > 2 else {})


def rank_ic(score_wide: pd.DataFrame, prices: pd.DataFrame, horizon: int = 21) -> pd.Series:
    """Daily Spearman correlation between today's score and the next `horizon`
    sessions' return across stocks. Positive = higher scores, higher returns."""
    fwd = prices.shift(-horizon) / prices - 1
    out = {}
    for d in score_wide.index:
        if d not in fwd.index:
            continue
        a, b = score_wide.loc[d], fwd.loc[d]
        m = a.notna() & b.notna()
        if m.sum() >= 20:
            out[d] = a[m].rank().corr(b[m].rank())
    return pd.Series(out, dtype=float)


def decile_spread(score_wide: pd.DataFrame, prices: pd.DataFrame, horizon: int = 21) -> float:
    """Mean forward return of the top score decile minus the bottom decile."""
    fwd = prices.shift(-horizon) / prices - 1
    diffs = []
    for d in score_wide.index[::5]:
        if d not in fwd.index:
            continue
        a, b = score_wide.loc[d], fwd.loc[d]
        m = a.notna() & b.notna()
        if m.sum() < 30:
            continue
        q = a[m].rank(pct=True)
        diffs.append(b[m][q >= 0.9].mean() - b[m][q <= 0.1].mean())
    return float(np.nanmean(diffs)) if diffs else float("nan")
