"""1-trading-session-ahead label. Thin wrapper around labels.horizons for the
per-horizon file layout the spec calls for; all real logic lives there."""

from labels.horizons import LabelConfig, generate_labels

HORIZON = "1D"


def generate(df, config: LabelConfig | None = None, benchmark_future_return=None):
    return generate_labels(df, HORIZON, config, benchmark_future_return)
