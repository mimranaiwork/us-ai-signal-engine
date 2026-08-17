"""
Classification metrics (per horizon — spec section 44) and trading/risk
metrics (spec section 44) computed from a simulated equity curve. Both are
plain functions over numpy/pandas so they're usable from the walk-forward
harness, ad-hoc notebooks, or the API alike.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.preprocessing import label_binarize

TRADING_DAYS_PER_YEAR = 252


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> dict:
    """y_true/y_pred: integer class labels (0=SELL,1=HOLD,2=BUY per models.common.LABEL_ORDER).
    y_proba: optional (n, 3) probability array for ROC-AUC/PR-AUC."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }
    if y_proba is not None:
        try:
            y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
            metrics["roc_auc_macro"] = float(roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr"))
        except ValueError:
            metrics["roc_auc_macro"] = None  # e.g. a class missing entirely from this fold
    return metrics


def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return float(drawdown.min())


def trading_metrics(
    equity_curve: pd.Series, trade_returns: pd.Series | None = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    """
    equity_curve: portfolio value indexed by date, starting at some base (e.g. 1.0 or 1,000,000).
    trade_returns: optional per-trade returns (for win rate / profit factor / avg trade);
        if omitted, those three metrics are computed from daily returns instead.
    """
    equity_curve = equity_curve.dropna()
    if len(equity_curve) < 2:
        return {"error": "insufficient data for trading metrics"}

    daily_returns = equity_curve.pct_change().dropna()
    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
    n_years = len(equity_curve) / periods_per_year
    cagr = float((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / n_years) - 1) if n_years > 0 else 0.0

    ann_vol = float(daily_returns.std() * np.sqrt(periods_per_year))
    sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(periods_per_year)) if daily_returns.std() > 0 else 0.0

    downside = daily_returns[daily_returns < 0]
    downside_vol = float(downside.std() * np.sqrt(periods_per_year)) if len(downside) > 1 else 0.0
    sortino = float(daily_returns.mean() * periods_per_year / downside_vol) if downside_vol > 0 else 0.0

    max_dd = _max_drawdown(equity_curve)
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0

    returns_for_trade_stats = trade_returns if trade_returns is not None else daily_returns
    wins = returns_for_trade_stats[returns_for_trade_stats > 0]
    losses = returns_for_trade_stats[returns_for_trade_stats < 0]
    win_rate = float(len(wins) / len(returns_for_trade_stats)) if len(returns_for_trade_stats) else 0.0
    profit_factor = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf") if wins.sum() > 0 else 0.0
    avg_trade = float(returns_for_trade_stats.mean()) if len(returns_for_trade_stats) else 0.0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "average_trade_return": avg_trade,
        "n_periods": len(equity_curve),
    }
