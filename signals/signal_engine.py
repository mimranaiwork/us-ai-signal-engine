"""
The final BUY/HOLD/SELL signal engine (spec sections 34-39, 48) — ties
together model predictions (Phase 2), ensemble scoring/consensus (this
phase), market regime, confidence, conservative HOLD overrides, and
SHAP-derived reasons/risks into the single API-shaped response.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import yaml

from calendar_us.trading_calendar import HORIZON_SESSIONS
from features.earnings.earnings import build_filing_events, earnings_risk_flag, earnings_snapshot
from features.market.market_breadth import compute_market_breadth, latest_breadth_snapshot
from features.market.market_features import BENCHMARKS, compute_all_benchmark_features
from features.sentiment.news_sentiment import live_sentiment_snapshot
from features.technical.pivot_points import breakout_assessment
from ingestion.ohlcv_source import SampleCSVSource
from models.common import PROCESSED_DIR, default_universe_symbols
from models.predict import ModelRegistry
from models.sequence_common import SEQUENCE_FEATURE_COLUMNS
from signals.confidence import DataQuality, compute_confidence
from signals.market_regime import detect_market_regime, latest_benchmark_row
from signals.scoring import build_composite, score_to_label
from signals.stock_regime import detect_stock_regime
from universe.universe_loader import Universe

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")

MIN_HISTORY_ROWS = 260
LIQUIDITY_HOLD_MESSAGE = "Average daily dollar volume below the configured liquidity floor"

_breadth_cache: dict = {"as_of": None, "snapshot": None}


def _clean_num(v) -> float | None:
    """NaN-safe float coercion for response fields sourced straight off a
    features-CSV row (e.g. volume/volume_sma_20 on a symbol's warmup rows,
    or any column a particular row happens to be missing)."""
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def _cached_market_breadth_score(as_of_date) -> float | None:
    """Market breadth (spec section 14) requires scanning EVERY symbol's
    processed features, unlike every other input to a single symbol's
    signal — too expensive to recompute per request, so it's cached
    in-process and only recomputed when as_of_date changes (i.e. once per
    new trading session, not once per symbol)."""
    if _breadth_cache["as_of"] == as_of_date and _breadth_cache["snapshot"] is not None:
        return _breadth_cache["snapshot"].get("market_breadth_score")
    try:
        breadth = compute_market_breadth(default_universe_symbols())
        snapshot = latest_breadth_snapshot(breadth)
    except Exception as exc:
        logger.info("Market breadth unavailable: %s", exc)
        snapshot = None
    _breadth_cache["as_of"] = as_of_date
    _breadth_cache["snapshot"] = snapshot
    return snapshot.get("market_breadth_score") if snapshot else None


def _earnings_timing(symbol: str, as_of_date) -> dict:
    """spec section 40: earnings-aware risk. Best-effort — SEC EDGAR needs
    SEC_EDGAR_USER_AGENT configured and network access; this degrades to
    "unknown" rather than failing the whole signal if either is unavailable,
    since earnings timing is a risk *enhancement*, not a hard requirement."""
    try:
        from ingestion.sec_edgar import SecEdgarClient
        client = SecEdgarClient()
        submissions = client.submissions(symbol)
        events = build_filing_events(submissions)
        return earnings_snapshot(events, as_of_date)
    except Exception as exc:
        logger.info("Earnings timing unavailable for %s: %s", symbol, exc)
        return {
            "days_since_earnings": None, "days_to_earnings": None,
            "days_to_earnings_is_estimate": None, "next_earnings_date": None,
        }


def _news_snapshot(symbol: str) -> dict:
    """Live news sentiment (features/sentiment/news_sentiment.py) — real
    current headlines via yfinance, display-only (see that module's
    docstring for why this can't feed the trained models). Best-effort, same
    degrade-gracefully pattern as _earnings_timing: news is an enrichment,
    never a reason to fail signal generation."""
    try:
        return live_sentiment_snapshot(symbol)
    except Exception as exc:
        logger.info("News sentiment unavailable for %s: %s", symbol, exc)
        return {"available": False, "overall_sentiment": None, "headlines": []}


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _load_symbol_features(symbol: str) -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, f"{symbol}_features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No processed features for {symbol}. Run scripts/build_phase1_dataset.py first."
        )
    return pd.read_csv(path, parse_dates=["date"])


def _data_quality(features: pd.DataFrame, latest_row: pd.Series, min_dollar_volume: float) -> DataQuality:
    sufficient_history = len(features) >= MIN_HISTORY_ROWS
    avg_dollar_volume = latest_row.get("avg_dollar_volume_20d", 0.0)
    sufficient_liquidity = bool(avg_dollar_volume is not None and avg_dollar_volume >= min_dollar_volume)
    key_cols = ["adj_close", "rsi_14", "atr_percent", "volatility_20d"]
    missing_ratio = float(latest_row[key_cols].isna().mean())
    return DataQuality(sufficient_history, sufficient_liquidity, missing_ratio)


def _risk_level(latest_row: pd.Series, regime: str, data_quality: DataQuality) -> str:
    vol = latest_row.get("volatility_20d", np.nan)
    if not data_quality.sufficient_liquidity or regime == "HIGH_VOLATILITY":
        return "HIGH"
    if pd.notna(vol) and vol >= 0.45:
        return "HIGH"
    if pd.notna(vol) and vol >= 0.28:
        return "MEDIUM"
    return "LOW"


def generate_signal(symbol: str, registry: ModelRegistry, config: dict | None = None) -> dict:
    config = config or _load_config()
    universe = Universe()
    stock = universe.get(symbol)

    features = _load_symbol_features(symbol)
    valid_rows = features.dropna(subset=["adj_close", "rsi_14"])
    if valid_rows.empty:
        raise ValueError(f"No usable feature rows for {symbol}")
    latest_row = valid_rows.iloc[-1]

    # Sequence input for LSTM/Transformer: trailing seq_len rows of the curated columns.
    seq_len = registry.sequence_models["lstm"].seq_len if "lstm" in registry.sequence_models else 60
    seq_slice = features.dropna(subset=SEQUENCE_FEATURE_COLUMNS).tail(seq_len)
    feature_sequence = None
    if len(seq_slice) == seq_len:
        feature_sequence = seq_slice[SEQUENCE_FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    predictions = registry.predict_all(latest_row, feature_sequence)
    if not predictions:
        raise RuntimeError("No trained models available — run training/train.py first")

    model_weights = config.get("model_weights", {})
    horizon_weights = config["horizons"]["weights"]
    composite = build_composite(
        symbol, predictions, model_weights, horizon_weights,
        stacker=registry.stacker, stacking_horizons=registry.stacking_horizons,
    )

    # Market regime — recomputed from the benchmark ETFs' own processed features.
    source = SampleCSVSource()
    start, end = features["date"].min().date(), features["date"].max().date()
    benchmark_bars = {b: source.get_bars(b, start, end) for b in BENCHMARKS}
    benchmark_features = compute_all_benchmark_features(benchmark_bars)
    regime_inputs = {b: latest_benchmark_row(df) for b, df in benchmark_features.items()}
    as_of_for_breadth = features["date"].max().date()
    breadth_score = _cached_market_breadth_score(as_of_for_breadth)
    regime = detect_market_regime(regime_inputs, market_breadth_score=breadth_score)

    # Stock-specific regime (spec section 24) — ADX/SMA200/momentum/volatility
    # for THIS stock, distinct from the market-wide regime above. sector_trend
    # isn't wired in yet (needs features/sector/, still a stub); SPY's own
    # trend stands in for "market trend" since it's the most standard single
    # benchmark reference.
    market_trend = regime_inputs.get("SPY", {}).get("trend")
    stock_regime = detect_stock_regime(latest_row, market_trend=market_trend)

    liquidity_floor = config.get("liquidity", {}).get("min_avg_dollar_volume_20d", 0)
    data_quality = _data_quality(features, latest_row, liquidity_floor)

    all_probs = [p.probs for p in predictions]
    confidence = compute_confidence(
        composite.model_consensus, composite.horizon_consensus, all_probs,
        regime.confidence, data_quality,
    )

    thresholds = config.get("signal_thresholds", {"buy": 0.25, "sell": -0.25})
    raw_signal = score_to_label(composite.score, thresholds["buy"], thresholds["sell"])

    as_of_date = latest_row["date"].date() if hasattr(latest_row["date"], "date") else latest_row["date"]
    earnings_timing = _earnings_timing(symbol, as_of_date)
    earnings_days_threshold = config.get("earnings", {}).get("risk_window_days", 5)
    earnings_imminent = earnings_risk_flag(earnings_timing["days_to_earnings"], earnings_days_threshold)

    # ── Conservative HOLD overrides (spec section 36) ──────────────────────
    hold_reasons = []
    if composite.model_consensus < 0.4:
        hold_reasons.append("High disagreement between models")
    if composite.horizon_consensus < 0.4:
        hold_reasons.append("High disagreement between time horizons")
    if confidence < 0.35:
        hold_reasons.append("Low overall confidence")
    if regime.regime == "HIGH_VOLATILITY" and regime.confidence < 0.5:
        hold_reasons.append("Market regime uncertain")
    if not data_quality.sufficient_history:
        hold_reasons.append("Insufficient price history")
    if not data_quality.sufficient_liquidity:
        hold_reasons.append(LIQUIDITY_HOLD_MESSAGE)
    if earnings_imminent:
        days = earnings_timing["days_to_earnings"]
        estimate_tag = " (estimated)" if earnings_timing.get("days_to_earnings_is_estimate") else ""
        hold_reasons.append(f"Earnings expected within {days} day(s){estimate_tag}")

    final_signal = "HOLD" if hold_reasons else raw_signal

    risk = "HIGH" if earnings_imminent else _risk_level(latest_row, regime.regime, data_quality)

    reasons, risks = [], []
    if "xgboost" in registry.tree_models and "1M" in registry.tree_models["xgboost"].models:
        from explainability.shap import top_reasons_and_risks
        model = registry.tree_models["xgboost"].models["1M"]
        feat_cols = registry.tree_models["xgboost"].feature_cols
        X_row = pd.DataFrame([latest_row[feat_cols]])
        reasons, risks = top_reasons_and_risks(model, X_row, buy_class_index=2)
    if hold_reasons:
        risks = hold_reasons + risks

    levels = breakout_assessment(latest_row)
    news = _news_snapshot(symbol)

    return {
        "symbol": symbol,
        "as_of": latest_row["date"].strftime("%Y-%m-%d") if hasattr(latest_row["date"], "strftime") else str(latest_row["date"]),
        "signal": final_signal,
        "score": round(composite.score, 4),
        "confidence": round(confidence, 4),
        "model_consensus": round(composite.model_consensus, 4),
        "horizon_consensus": round(composite.horizon_consensus, 4),
        "market_regime": regime.regime,
        "stock_regime": stock_regime.regime,
        "sector": stock.sector_code,
        "industry": stock.industry_code,
        "company_name": stock.company_name,
        "market_cap": stock.market_cap,
        "volume": _clean_num(latest_row.get("volume")),
        "avg_volume_20d": _clean_num(latest_row.get("volume_sma_20")),
        "risk": risk,
        "horizons": {
            h: {"signal": score_to_label(s, thresholds["buy"], thresholds["sell"]), "score": round(s, 4)}
            for h, s in composite.horizon_scores.items()
        },
        "reasons": reasons,
        "risks": risks,
        "earnings": earnings_timing,
        "stacking_used": composite.stacking_used,
        "levels": levels,
        "news": news,
    }
