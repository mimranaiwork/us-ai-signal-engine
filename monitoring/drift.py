"""
Data drift (feature distributions shifting vs. training time) and model
drift (prediction/label distribution shifting) detection — spec section
"Phase 7 — Production: ... Data drift, Model drift".

Uses the Population Stability Index (PSI), the standard, interpretable
metric for this: bins a reference and current distribution the same way and
sums (current% - reference%) * ln(current% / reference%) over bins. Rule of
thumb thresholds (industry-standard, not something specific to this
project): PSI < 0.1 no significant shift, 0.1-0.25 moderate shift worth
watching, > 0.25 significant shift — the model's training-time assumptions
about this feature/prediction distribution may no longer hold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PSI_WATCH_THRESHOLD = 0.1
PSI_ALERT_THRESHOLD = 0.25


@dataclass
class DriftResult:
    feature: str
    psi: float
    status: str  # "STABLE" | "WATCH" | "ALERT"


def _psi_bins(reference: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Quantile bin edges from the reference distribution — current data is
    binned against these SAME edges, which is what makes PSI meaningful."""
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 3:
        edges = np.linspace(reference.min(), reference.max(), n_bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def population_stability_index(reference: pd.Series, current: pd.Series, n_bins: int = 10) -> float:
    reference = reference.dropna().to_numpy()
    current = current.dropna().to_numpy()
    if len(reference) < n_bins or len(current) < 1:
        return 0.0

    edges = _psi_bins(reference, n_bins)
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), 1e-6, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _status_for_psi(psi: float) -> str:
    if psi >= PSI_ALERT_THRESHOLD:
        return "ALERT"
    if psi >= PSI_WATCH_THRESHOLD:
        return "WATCH"
    return "STABLE"


def detect_feature_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, feature_cols: list[str]) -> list[DriftResult]:
    """reference_df: features from the training window. current_df: recent
    (e.g. last 20-60 sessions) features for the same columns."""
    results = []
    for col in feature_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        psi = population_stability_index(reference_df[col], current_df[col])
        results.append(DriftResult(feature=col, psi=psi, status=_status_for_psi(psi)))
    return results


def detect_prediction_drift(reference_scores: pd.Series, current_scores: pd.Series) -> DriftResult:
    """Same PSI machinery, applied to the ensemble's composite score
    distribution instead of a raw feature — catches the case where every
    individual feature looks stable but the model's overall output behavior
    has shifted (e.g. from a subtle multi-feature interaction change)."""
    psi = population_stability_index(reference_scores, current_scores)
    return DriftResult(feature="composite_score", psi=psi, status=_status_for_psi(psi))


def summarize_drift(results: list[DriftResult]) -> dict:
    alerts = [r.feature for r in results if r.status == "ALERT"]
    watches = [r.feature for r in results if r.status == "WATCH"]
    return {
        "n_features_checked": len(results),
        "n_alert": len(alerts),
        "n_watch": len(watches),
        "alert_features": alerts,
        "watch_features": watches,
        "max_psi": max((r.psi for r in results), default=0.0),
    }
