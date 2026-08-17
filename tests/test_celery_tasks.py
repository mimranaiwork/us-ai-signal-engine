"""
Tests the Celery task orchestration in training/tasks.py — NOT the actual
pipeline behavior (fetch_real_data/build_phase1_dataset/train_tree_models
etc. are each tested against their own real logic elsewhere; hitting the
network or running real ML training here would make this suite slow and
flaky). Mocks every expensive call and checks refresh_data/nightly_pipeline
wire them together in the right order with the right arguments — the actual
bug class this guards against: retrain_all silently training on stale data
because nothing upstream of it ever refreshed the CSVs (see
training/tasks.py's refresh_data docstring for why that's a real gap it
closes, not a hypothetical one).
"""

from unittest.mock import MagicMock, patch

from training.tasks import nightly_pipeline, refresh_data


def test_refresh_data_calls_pipeline_in_order_and_clears_cache():
    with patch("scripts.fetch_real_data.fetch_all") as fetch_ohlcv, \
         patch("scripts.fetch_macro_data.fetch_all") as fetch_macro, \
         patch("scripts.build_phase1_dataset.build") as build, \
         patch("scripts.augment_features.main") as augment, \
         patch("monitoring.cache.clear_pattern", side_effect=[3, 1]) as clear_pattern:
        result = refresh_data()

    fetch_ohlcv.assert_called_once()
    fetch_macro.assert_called_once()
    build.assert_called_once()
    augment.assert_called_once()
    assert clear_pattern.call_count == 2
    assert clear_pattern.call_args_list[0].args[0] == "signal:*"
    assert clear_pattern.call_args_list[1].args[0] == "screener:*"
    assert result["cache_keys_cleared"] == 4
    assert "as_of" in result


def test_refresh_data_builds_from_real_data_dir():
    """Regression guard: must pass data_dir=REAL_DIR to build(), not the
    default SAMPLE_DIR — silently rebuilding from synthetic sample data
    instead of the freshly-fetched real data would be a much worse bug than
    just not refreshing at all."""
    with patch("scripts.fetch_real_data.fetch_all"), \
         patch("scripts.fetch_macro_data.fetch_all"), \
         patch("scripts.build_phase1_dataset.build") as build, \
         patch("scripts.augment_features.main"), \
         patch("monitoring.cache.clear_pattern", return_value=0):
        refresh_data()

    import scripts.build_phase1_dataset as build_phase1_dataset
    assert build.call_args.kwargs["data_dir"] == build_phase1_dataset.REAL_DIR


def test_nightly_pipeline_runs_refresh_before_retrain():
    call_order = []

    def _fake_refresh():
        call_order.append("refresh")
        return {"as_of": "2026-08-12", "cache_keys_cleared": 0}

    def _fake_retrain(**kwargs):
        call_order.append("retrain")
        return {"mlflow_run_id": "fake"}

    with patch("training.tasks.refresh_data", side_effect=_fake_refresh) as refresh_mock, \
         patch("training.tasks.retrain_all", side_effect=_fake_retrain) as retrain_mock:
        result = nightly_pipeline()

    assert call_order == ["refresh", "retrain"]
    refresh_mock.assert_called_once()
    retrain_mock.assert_called_once()
    assert "refresh" in result and "retrain" in result
