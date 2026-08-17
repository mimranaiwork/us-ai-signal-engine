"""
US market regime engine (spec section 38). Uses SPY/QQQ/IWM trend +
realized volatility to classify BULL/BEAR/SIDEWAYS/HIGH_VOLATILITY, with
market breadth (features/market/market_breadth.py) as an optional
CONFIRMING/DISCONFIRMING signal — the standard technical-analysis
interpretation of breadth: an index-level BULL read while most individual
stocks are actually declining ("narrow market") is a real warning sign, so
breadth divergence reduces confidence rather than silently being ignored or
flipping the regime label outright on its own. VIX, DXY, and 10Y Treasury
are still not wired in — those need a macro data pipeline (see
features/macro/, still a stub pending a FRED/yfinance macro ingestion pass).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

REGIME_BENCHMARKS = ["SPY", "QQQ", "IWM"]

# Annualized 20D volatility above this is "elevated" — a rough, documented
# placeholder threshold; Phase 4 backtesting can replace it with a
# percentile-of-history threshold once there's enough data to compute one.
HIGH_VOLATILITY_THRESHOLD = 0.35


@dataclass
class RegimeResult:
    regime: str  # BULL | BEAR | SIDEWAYS | HIGH_VOLATILITY
    confidence: float  # how unanimous the benchmarks are, in [0, 1]
    detail: dict


def _breadth_confidence_adjustment(regime: str, market_breadth_score: float | None) -> tuple[float, bool]:
    """Returns (multiplicative confidence adjustment, is_divergence). Breadth
    strongly opposing the regime direction is a divergence (narrow-market
    warning) and cuts confidence; breadth agreeing leaves confidence alone
    rather than artificially inflating it (the benchmark trend vote already
    IS the primary signal)."""
    if market_breadth_score is None or regime not in ("BULL", "BEAR"):
        return 1.0, False
    expected_sign = 1 if regime == "BULL" else -1
    if market_breadth_score * expected_sign < -0.15:  # breadth meaningfully opposes the regime
        return 0.6, True
    return 1.0, False


def detect_market_regime(
    benchmark_features_at_date: dict[str, dict], market_breadth_score: float | None = None,
) -> RegimeResult:
    """
    benchmark_features_at_date: {"SPY": {"trend": "UP", "volatility_20d": 0.18, "return_1d": ...}, "QQQ": {...}, "IWM": {...}}
    — i.e. one row (as of a given date) per benchmark, already computed by
    features/market/market_features.py.
    market_breadth_score: optional, from features/market/market_breadth.py —
    see module docstring for how it's used (confidence adjustment, not a
    regime override).
    """
    available = {b: v for b, v in benchmark_features_at_date.items() if b in REGIME_BENCHMARKS}
    if not available:
        return RegimeResult("SIDEWAYS", 0.0, {"reason": "no benchmark data available"})

    trends = [v["trend"] for v in available.values() if v.get("trend") is not None]
    vols = [v["volatility_20d"] for v in available.values() if v.get("volatility_20d") is not None]
    avg_vol = sum(vols) / len(vols) if vols else None

    if avg_vol is not None and avg_vol >= HIGH_VOLATILITY_THRESHOLD:
        # High volatility overrides trend — spec section 38 lists it as its
        # own regime, and a choppy high-vol tape makes trend classification
        # unreliable regardless of which way SMA50/200 currently point.
        up_share = trends.count("UP") / len(trends) if trends else 0.5
        return RegimeResult("HIGH_VOLATILITY", 1.0 - abs(up_share - 0.5) * 2,
                             {"avg_volatility_20d": avg_vol, "trends": trends, "market_breadth_score": market_breadth_score})

    if not trends:
        return RegimeResult("SIDEWAYS", 0.0, {"reason": "no trend data available"})

    up_count = trends.count("UP")
    down_count = trends.count("DOWN")
    n = len(trends)

    if up_count / n >= 0.66:
        regime = "BULL"
    elif down_count / n >= 0.66:
        regime = "BEAR"
    else:
        regime = "SIDEWAYS"

    majority_share = max(up_count, down_count, n - up_count - down_count) / n
    breadth_adjustment, is_divergence = _breadth_confidence_adjustment(regime, market_breadth_score)
    confidence = majority_share * breadth_adjustment

    detail = {"avg_volatility_20d": avg_vol, "trends": trends, "market_breadth_score": market_breadth_score}
    if is_divergence:
        detail["breadth_divergence"] = True

    return RegimeResult(regime, confidence, detail)


def latest_benchmark_row(benchmark_df: pd.DataFrame) -> dict:
    """Pulls the most recent non-NaN row of a compute_benchmark_features() frame as a dict."""
    valid = benchmark_df.dropna(subset=["trend", "volatility_20d"])
    if valid.empty:
        return {}
    return valid.iloc[-1].to_dict()
