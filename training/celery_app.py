"""
Celery app for scheduled data refresh + retraining (spec Phase 7). Uses the
same Redis instance docker-compose.yml already runs as both broker and
result backend — no extra infrastructure beyond what Phase 7's Redis caching
already needs.

Run a worker:  celery -A training.celery_app worker --loglevel=info
Run the beat scheduler (periodic refresh+retrain, see beat_schedule below):
                celery -A training.celery_app beat --loglevel=info
Trigger manually, without waiting for the schedule:
    python -c "from training.tasks import refresh_data; refresh_data.delay()"
    python -c "from training.tasks import nightly_pipeline; nightly_pipeline.delay()"

worker_pool="solo" is required, not a style choice: Celery's default
prefork pool forks a child process per task. On macOS, forking a process
after PyTorch has touched its Metal/MPS backend crashes the child with
SIGABRT (`+[MPSGraphObject initialize] may have been in progress in
another thread when fork() was called`) — hit for real running
nightly_pipeline's LSTM/Transformer retrain under prefork during this
project's own testing, not a theoretical concern. "solo" runs tasks
in-process with no forking, which sidesteps the whole class of bug; the
tradeoff is no intra-worker task concurrency, which costs nothing here
since beat only ever schedules one job at a time.
"""

from __future__ import annotations

import os

from celery import Celery
from celery.schedules import crontab

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = os.environ.get("REDIS_PORT", "6379")
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"

app = Celery("us_ai_signal_engine", broker=REDIS_URL, backend=REDIS_URL, include=["training.tasks"])

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="America/New_York",  # US market hours reference
    task_track_started=True,
    worker_pool="solo",  # see module docstring — avoids the macOS PyTorch/fork crash
)

# Refresh data + retrain nightly after the US market closes (4pm ET) — 6pm ET
# gives data vendors time to finalize the day's adjusted close before this
# runs. One entry, not two independent crons: retrain_all trains on whatever
# CSVs are already on disk, so it must run AFTER refresh_data's fetch
# completes, not on its own overlapping schedule — training.tasks.nightly_pipeline
# runs both in the correct order.
app.conf.beat_schedule = {
    "nightly-refresh-and-retrain": {
        "task": "training.tasks.nightly_pipeline",
        "schedule": crontab(hour=18, minute=0),
    },
}
