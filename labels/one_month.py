"""21-trading-session-ahead (~1 calendar month) label. See labels/horizons.py."""

from labels.horizons import LabelConfig, generate_labels

HORIZON = "1M"


def generate(df, config: LabelConfig | None = None, benchmark_future_return=None):
    return generate_labels(df, HORIZON, config, benchmark_future_return)
