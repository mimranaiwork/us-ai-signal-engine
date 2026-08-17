"""
MLflow experiment tracking + model registry (spec Phase 7). Uses MLflow's
local file-store backend by default (`mlruns/`, gitignored) — genuinely
functional tracking with zero extra infrastructure; point `MLFLOW_TRACKING_URI`
at a real server (e.g. the one implied by adding an `mlflow` service to
docker-compose.yml) when you have one running, and the exact same calls
start writing there instead.
"""

from __future__ import annotations

import os

import mlflow
import mlflow.lightgbm
import mlflow.pytorch
import mlflow.tracking
import mlflow.xgboost

EXPERIMENT_NAME = "us-ai-signal-engine"


def _configure():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_training_run(params: dict, report: dict, models: dict) -> str:
    """
    params: run configuration (n_symbols, epochs, hidden_size, seq_len, ...).
    report: the same nested dict train.py already writes to training_report.json.
    models: {"xgboost": XGBoostMultiHorizonModel, "lightgbm": ..., "lstm": ..., "transformer": ...}
            — whichever were actually trained this run.
    Returns the MLflow run ID.
    """
    _configure()
    with mlflow.start_run() as run:
        mlflow.log_params(params)

        for arch, horizons in report.get("tree_models", {}).items():
            for horizon, metrics in horizons.items():
                mlflow.log_metric(f"{arch}_{horizon}_val_accuracy", metrics["val_accuracy"])

        for arch, metrics in report.get("sequence_models", {}).items():
            if arch == "note" or not isinstance(metrics, dict):
                continue
            for horizon, acc in metrics.get("val_acc", {}).items():
                mlflow.log_metric(f"{arch}_{horizon}_val_accuracy", acc)

        if "xgboost" in models:
            for horizon, booster in models["xgboost"].models.items():
                mlflow.xgboost.log_model(booster, name=f"xgboost_{horizon}")
        if "lightgbm" in models:
            for horizon, booster in models["lightgbm"].models.items():
                mlflow.lightgbm.log_model(booster, name=f"lightgbm_{horizon}")
        # serialization_format="pickle": mlflow's default ("pt2", torch.export-based)
        # requires torch>=2.4; this project is pinned to torch==2.2.2 for Intel Mac
        # ABI compatibility (see requirements.txt), so the default silently crashes
        # log_training_run — hit for real when this ran end-to-end for the first
        # time via the scheduled Celery pipeline (training/tasks.py::retrain_all).
        if "lstm" in models:
            mlflow.pytorch.log_model(models["lstm"].net, name="lstm", serialization_format=mlflow.pytorch.SERIALIZATION_FORMAT_PICKLE)
        if "transformer" in models:
            mlflow.pytorch.log_model(models["transformer"].net, name="transformer", serialization_format=mlflow.pytorch.SERIALIZATION_FORMAT_PICKLE)

        return run.info.run_id


def latest_run_metrics() -> dict | None:
    """Convenience lookup for monitoring/CLI use — the most recent run's metrics."""
    _configure()
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return None
    runs = client.search_runs([experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
    if not runs:
        return None
    return dict(runs[0].data.metrics)
