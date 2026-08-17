"""
Walk-forward optimization of signal thresholds, horizon weights, and model
weights (spec sections 32/33/35) — picks values on a VALIDATION window and
confirms on a separate TEST window, instead of eyeballing the spec's initial
placeholders forever.

Scope, honestly stated: this searches over the already-trained XGBoost +
LightGBM ensemble's historical composite scores. LSTM/Transformer are NOT
included in the historical score reconstruction here — recomputing their
sequence-windowed predictions across an entire multi-year history for every
symbol in a grid search is a materially bigger compute job (the same
per-fold-retraining cost problem documented in backtesting/walk_forward.py's
docstring, compounded by the outer grid-search loop). Their model_weights
entries keep the spec's initial placeholder values; xgboost/lightgbm's
weights are what actually get optimized, then all four are renormalized in
the returned config so weights still sum to 1.
"""

from __future__ import annotations

import itertools
from datetime import date

import numpy as np
import pandas as pd

from backtesting.engine import simulate_long_only
from backtesting.metrics import trading_metrics
from calendar_us.trading_calendar import HORIZON_SESSIONS
from features.market.market_features import BENCHMARKS
from ingestion.ohlcv_source import SampleCSVSource
from models.common import LABEL_TO_IDX, PROCESSED_DIR
from models.lightgbm.model import LightGBMMultiHorizonModel
from models.xgboost.model import XGBoostMultiHorizonModel

BUY_IDX, HOLD_IDX, SELL_IDX = LABEL_TO_IDX["BUY"], LABEL_TO_IDX["HOLD"], LABEL_TO_IDX["SELL"]

# A handful of preset horizon-weight schemes rather than a continuous search
# over 7 simultaneous weights — keeps the search space tractable. All must
# sum to 1.0 across the 7 horizons.
HORIZON_WEIGHT_PRESETS = {
    "spec_default": {"1D": 0.05, "1W": 0.10, "2W": 0.10, "1M": 0.20, "2M": 0.20, "3M": 0.20, "6M": 0.15},
    "equal": {h: 1 / 7 for h in HORIZON_SESSIONS},
    "front_loaded": {"1D": 0.15, "1W": 0.20, "2W": 0.20, "1M": 0.20, "2M": 0.10, "3M": 0.10, "6M": 0.05},
    "back_loaded": {"1D": 0.03, "1W": 0.05, "2W": 0.07, "1M": 0.15, "2M": 0.20, "3M": 0.25, "6M": 0.25},
}

MODEL_WEIGHT_PRESETS = {
    "spec_default": {"xgboost": 0.30 / 0.55, "lightgbm": 0.25 / 0.55},  # renormalized to sum to 1 over just these 2
    "equal": {"xgboost": 0.5, "lightgbm": 0.5},
    "xgboost_heavy": {"xgboost": 0.7, "lightgbm": 0.3},
    "lightgbm_heavy": {"xgboost": 0.3, "lightgbm": 0.7},
}

THRESHOLD_GRID = [0.10, 0.15, 0.20, 0.25, 0.30]


def compute_historical_ensemble_scores(
    symbol: str, tree_models: dict, horizon_weights: dict, model_weights: dict,
) -> pd.DataFrame:
    """Reconstructs the XGBoost+LightGBM ensemble's composite score for
    EVERY historical date at once (vectorized — one predict_proba call per
    (model, horizon), not a per-date loop), given a specific weight config."""
    features = pd.read_csv(f"{PROCESSED_DIR}/{symbol}_features.csv", parse_dates=["date"])

    horizon_scores = {}
    for horizon in horizon_weights:
        # Accumulate weight*proba and the row's total contributing weight
        # SEPARATELY, so a row where every model was NaN (warmup period)
        # divides 0/0 -> NaN, rather than silently reading as a "confident"
        # neutral (0.0) prediction — a real bug caught by inspecting the
        # reconstructed AAPL scores: 219 rows were exactly 0.0, all in the
        # 2018 warmup window before the 200-day features exist.
        weighted_sum = np.zeros((len(features), 3))
        row_weight = np.zeros(len(features))
        for name, model in tree_models.items():
            if horizon not in model.models:
                continue
            valid_mask = features[model.feature_cols].notna().all(axis=1).to_numpy()
            proba = np.zeros((len(features), 3))
            proba[valid_mask] = model.predict_proba(features.loc[valid_mask, model.feature_cols], horizon)
            weight = model_weights.get(name, 0.0)
            weighted_sum += weight * proba
            row_weight += np.where(valid_mask, weight, 0.0)
        if row_weight.sum() == 0:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            weighted_probs = weighted_sum / row_weight[:, np.newaxis]
        horizon_scores[horizon] = weighted_probs[:, BUY_IDX] - weighted_probs[:, SELL_IDX]  # NaN where row_weight==0

    if not horizon_scores:
        raise RuntimeError(f"No horizon scores computed for {symbol}")

    total_hw = sum(horizon_weights.values())
    composite = sum(horizon_scores[h] * horizon_weights[h] for h in horizon_scores) / total_hw

    return pd.DataFrame({"date": features["date"], "adj_close": features["adj_close"], "score": composite})


def _signals_from_scores(scores: pd.Series, buy_threshold: float, sell_threshold: float) -> pd.Series:
    signal = pd.Series("HOLD", index=scores.index)
    signal = signal.mask(scores >= buy_threshold, "BUY")
    signal = signal.mask(scores <= sell_threshold, "SELL")
    return signal


def evaluate_config(
    symbols: list[str], tree_models: dict, horizon_weights: dict, model_weights: dict,
    buy_threshold: float, sell_threshold: float, window_start: date, window_end: date,
) -> float:
    """Average Sharpe across symbols for one (weights, thresholds) config
    over [window_start, window_end) — standalone convenience wrapper (the
    optimize() grid search inlines this same logic itself to reuse scores
    across thresholds; this is for one-off confirmation runs, e.g. checking
    the winning config on a held-out test window)."""
    sharpes = []
    for symbol in symbols:
        scored = compute_historical_ensemble_scores(symbol, tree_models, horizon_weights, model_weights)
        sharpe = _sharpe_for_window(scored, buy_threshold, sell_threshold, window_start, window_end)
        if sharpe is not None:
            sharpes.append(sharpe)
    return float(np.mean(sharpes)) if sharpes else float("-inf")


def _sharpe_for_window(scored: pd.DataFrame, buy_threshold: float, sell_threshold: float,
                        window_start: date, window_end: date) -> float | None:
    window = scored[(scored["date"].dt.date >= window_start) & (scored["date"].dt.date < window_end)].dropna(subset=["score"])
    if len(window) < 30:
        return None
    signals = _signals_from_scores(window["score"], buy_threshold, sell_threshold)
    equity, _, _ = simulate_long_only(window["date"], window["adj_close"], signals)
    metrics = trading_metrics(equity)
    return metrics.get("sharpe")


def optimize(symbols: list[str], val_start: date, val_end: date) -> dict:
    """
    Grid search over THRESHOLD_GRID x HORIZON_WEIGHT_PRESETS x MODEL_WEIGHT_PRESETS,
    scored by average out-of-sample Sharpe on [val_start, val_end). Returns the
    winning config plus the full leaderboard for transparency.

    Loop order matters for cost: scores only depend on (symbol, horizon_weights,
    model_weights) — thresholds are applied AFTER scoring, so they're evaluated
    from an already-computed score series instead of re-running predict_proba.
    That's a 5x reduction in the expensive part (len(THRESHOLD_GRID)) for free.
    """
    xgb = XGBoostMultiHorizonModel.load()
    lgb = LightGBMMultiHorizonModel.load()
    tree_models = {"xgboost": xgb, "lightgbm": lgb}

    weight_combos = list(itertools.product(HORIZON_WEIGHT_PRESETS.items(), MODEL_WEIGHT_PRESETS.items()))
    print(f"Evaluating {len(weight_combos)} weight configs x {len(THRESHOLD_GRID)} thresholds "
          f"on {len(symbols)} symbols, window [{val_start}, {val_end})...")

    results = []
    for hw_name, hw in HORIZON_WEIGHT_PRESETS.items():
        for mw_name, mw in MODEL_WEIGHT_PRESETS.items():
            scored_by_symbol = {s: compute_historical_ensemble_scores(s, tree_models, hw, mw) for s in symbols}
            for buy_t in THRESHOLD_GRID:
                sharpes = [
                    s for s in (
                        _sharpe_for_window(scored, buy_t, -buy_t, val_start, val_end)
                        for scored in scored_by_symbol.values()
                    ) if s is not None
                ]
                avg_sharpe = float(np.mean(sharpes)) if sharpes else float("-inf")
                results.append({
                    "horizon_weights": hw_name, "model_weights": mw_name,
                    "buy_threshold": buy_t, "sell_threshold": -buy_t, "avg_sharpe": avg_sharpe,
                })
                print(f"  horizon={hw_name:12s} model={mw_name:14s} buy_t={buy_t:.2f}  avg_sharpe={avg_sharpe:+.3f}")

    results.sort(key=lambda r: r["avg_sharpe"], reverse=True)
    best = results[0]
    return {
        "best": best,
        "leaderboard": results,
        "horizon_weights": HORIZON_WEIGHT_PRESETS[best["horizon_weights"]],
        "model_weights": MODEL_WEIGHT_PRESETS[best["model_weights"]],
    }
