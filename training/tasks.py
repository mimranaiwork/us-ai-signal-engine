"""Celery tasks: scheduled data refresh + scheduled retraining. Kept
intentionally thin: all the real logic lives in the scripts/ pipeline
(fetch_real_data, fetch_macro_data, build_phase1_dataset, augment_features)
and training/train.py, so calling these tasks does exactly what running them
by hand does, just on a schedule instead of by hand."""

from __future__ import annotations

from datetime import date

from training.celery_app import app
from training.train import train_sequence_models, train_tree_models
from models.common import default_universe_symbols


@app.task(name="training.tasks.refresh_data")
def refresh_data() -> dict:
    """The actual "keep prices/RSI/Bollinger/pivot levels/screener from
    going stale" step (retrain_all below assumes fresh data already exists
    on disk — it never fetched anything itself, a gap this task closes):
    re-fetch OHLCV + macro series via yfinance, rebuild every symbol's
    features/labels, re-run the sector/industry/macro augmentation, then
    clear the cached signal/screener responses so the next request
    recomputes against the new data instead of serving stale cache for up
    to its remaining TTL.

    Daily is the right cadence, not more: yfinance's OHLCV is daily-bar
    data, it doesn't change intraday, so refreshing more often than once
    per session wouldn't produce different results — see
    scripts/fetch_real_data.py's module docstring."""
    import scripts.augment_features as augment_features
    import scripts.build_phase1_dataset as build_phase1_dataset
    import scripts.fetch_macro_data as fetch_macro_data
    import scripts.fetch_real_data as fetch_real_data
    from monitoring.cache import clear_pattern

    start, end = date(2018, 1, 1), date.today()

    fetch_real_data.fetch_all(start, end)
    fetch_macro_data.fetch_all(start, end)
    build_phase1_dataset.build(start, end, data_dir=build_phase1_dataset.REAL_DIR)
    augment_features.main()

    cleared = clear_pattern("signal:*") + clear_pattern("screener:*")

    return {"as_of": end.isoformat(), "cache_keys_cleared": cleared}


@app.task(name="training.tasks.retrain_all")
def retrain_all(symbols: list[str] | None = None, skip_sequence_models: bool = False) -> dict:
    symbols = symbols or default_universe_symbols()
    report = {}

    tree_report, tree_models = train_tree_models(symbols)
    report["tree_models"] = tree_report
    trained_models = dict(tree_models)

    if not skip_sequence_models:
        seq_report, seq_models = train_sequence_models(symbols)
        report["sequence_models"] = seq_report
        trained_models.update(seq_models)

    from training.mlflow_tracking import log_training_run
    run_id = log_training_run(
        {"n_symbols": len(symbols), "trigger": "celery_scheduled"}, report, trained_models
    )
    report["mlflow_run_id"] = run_id
    return report


@app.task(name="training.tasks.nightly_pipeline")
def nightly_pipeline(skip_sequence_models: bool = False) -> dict:
    """The single nightly beat_schedule entry: refresh_data must complete
    BEFORE retrain_all runs (retrain_all trains on whatever's already on
    disk — it never fetches anything itself), so these run in-process here
    rather than as two independent cron entries that could race or drift
    out of order."""
    refresh_result = refresh_data()
    retrain_result = retrain_all(skip_sequence_models=skip_sequence_models)
    return {"refresh": refresh_result, "retrain": retrain_result}
