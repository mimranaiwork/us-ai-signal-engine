from datetime import date

import numpy as np
import pandas as pd
import pytest

from features.options.options_features import SyntheticOptionsSource, add_options_features
from features.sentiment.news_sentiment import (
    classify_topic,
    fetch_recent_news,
    live_sentiment_snapshot,
    score_articles,
    score_text,
    sentiment_features_as_of,
)
from ingestion.ohlcv_source import SampleCSVSource

# ── Options ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def options_df():
    src = SampleCSVSource()
    bars = src.get_bars("AAPL", date(2018, 1, 1), date(2026, 8, 8))
    snapshots = SyntheticOptionsSource().get_daily_snapshots("AAPL", bars)
    realized_vol = bars["adj_close"].pct_change().rolling(20).std() * np.sqrt(252)
    return add_options_features(snapshots, realized_vol)


def test_put_call_ratio_positive(options_df):
    valid = options_df["put_call_ratio"].dropna()
    assert (valid > 0).all()


def test_iv_rank_bounded_0_1(options_df):
    valid = options_df["iv_rank"].dropna()
    assert (valid >= 0).all() and (valid <= 1).all()


def test_iv_percentile_bounded_0_1(options_df):
    valid = options_df["iv_percentile"].dropna()
    assert (valid >= 0).all() and (valid <= 1).all()


def test_iv_positive(options_df):
    valid = options_df["iv"].dropna()
    assert (valid > 0).all()


def test_options_sentiment_roughly_bounded(options_df):
    valid = options_df["options_sentiment"].dropna()
    assert valid.abs().max() <= 1.01  # allow tiny float slop at the clip boundary


def test_iv_deterministic_per_symbol():
    """Same symbol -> same synthetic IV path (seeded), for reproducibility."""
    src = SampleCSVSource()
    bars = src.get_bars("MSFT", date(2018, 1, 1), date(2020, 1, 1))
    run1 = SyntheticOptionsSource().get_daily_snapshots("MSFT", bars)
    run2 = SyntheticOptionsSource().get_daily_snapshots("MSFT", bars)
    pd.testing.assert_series_equal(run1["iv"], run2["iv"])


def test_different_symbols_get_different_iv_paths():
    src = SampleCSVSource()
    bars_a = src.get_bars("AAPL", date(2018, 1, 1), date(2020, 1, 1))
    bars_b = src.get_bars("MSFT", date(2018, 1, 1), date(2020, 1, 1))
    iv_a = SyntheticOptionsSource().get_daily_snapshots("AAPL", bars_a)["iv"]
    iv_b = SyntheticOptionsSource().get_daily_snapshots("MSFT", bars_b)["iv"]
    assert not iv_a.equals(iv_b)


# ── News sentiment ───────────────────────────────────────────────────────────
# Illustrative test sentences only — not real news, not presented as such.

def test_score_text_positive():
    assert score_text("Company beats earnings expectations, shares surge on strong growth") > 0.5


def test_score_text_negative():
    assert score_text("Company misses earnings, shares plunge after weak guidance and layoffs") < -0.5


def test_score_text_neutral_when_no_sentiment_words():
    assert score_text("The quarterly report was released on schedule") == 0.0


def test_score_text_mixed_sentiment():
    # Equal positive and negative signal -> should net toward zero.
    result = score_text("Company beats revenue estimates but issues a warning on weak demand")
    assert -0.5 < result < 0.5


def test_classify_topic_earnings():
    assert classify_topic("Company reports quarterly earnings, beats EPS estimates") == "earnings"


def test_classify_topic_m_and_a():
    assert classify_topic("Company announces acquisition of smaller rival") == "m_and_a"


def test_classify_topic_legal():
    assert classify_topic("Company faces lawsuit over patent dispute") == "legal"


def test_classify_topic_falls_back_to_other():
    assert classify_topic("The weather was pleasant today") == "other"


def test_score_articles_adds_columns():
    articles = pd.DataFrame({
        "date": [date(2024, 1, 1), date(2024, 1, 2)],
        "headline": ["Company beats earnings estimates", "Company announces lawsuit settlement"],
    })
    scored = score_articles(articles)
    assert "sentiment" in scored.columns
    assert "topic" in scored.columns
    assert scored.iloc[0]["sentiment"] > 0
    assert scored.iloc[1]["topic"] == "legal"


def test_sentiment_features_as_of_no_future_leakage():
    articles = pd.DataFrame({
        "date": [date(2024, 1, 1), date(2024, 6, 1)],
        "headline": ["Company beats earnings estimates", "Company misses guidance badly"],
    })
    scored = score_articles(articles)

    # As of a date before the second (negative) article, only the first (positive) counts.
    early = sentiment_features_as_of(scored, date(2024, 2, 1), windows=(60,))
    assert early["news_sentiment_60d"] > 0

    # As of a date after both, the negative article is now visible too.
    late = sentiment_features_as_of(scored, date(2024, 6, 2), windows=(200,))
    assert late["news_sentiment_200d"] < early["news_sentiment_60d"]


def test_sentiment_features_empty_before_any_article():
    articles = pd.DataFrame({"date": [date(2024, 6, 1)], "headline": ["Company beats earnings estimates"]})
    scored = score_articles(articles)
    result = sentiment_features_as_of(scored, date(2024, 1, 1))
    assert result["news_sentiment_1d"] == 0.0
    assert result["positive_news_count"] == 0


# ── Live news (yfinance) ─────────────────────────────────────────────────────

class _FakeTicker:
    def __init__(self, news):
        self.news = news


def _yf_item(title, pub_date="2026-08-10T13:56:05Z", url="https://finance.yahoo.com/story"):
    return {"content": {"title": title, "pubDate": pub_date, "canonicalUrl": {"url": url}}}


def test_fetch_recent_news_parses_real_yfinance_shape(monkeypatch):
    import yfinance as yf

    items = [_yf_item("Company beats earnings estimates"), _yf_item("Company plunges on weak guidance")]
    monkeypatch.setattr(yf, "Ticker", lambda symbol: _FakeTicker(items))

    result = fetch_recent_news("AAPL")
    assert len(result) == 2
    assert result.iloc[0]["headline"] == "Company beats earnings estimates"
    assert result.iloc[0]["url"] == "https://finance.yahoo.com/story"


def test_fetch_recent_news_skips_items_without_a_title(monkeypatch):
    import yfinance as yf

    items = [_yf_item("Real headline"), {"content": {"title": None}}]
    monkeypatch.setattr(yf, "Ticker", lambda symbol: _FakeTicker(items))

    result = fetch_recent_news("AAPL")
    assert len(result) == 1


def test_fetch_recent_news_returns_empty_df_on_fetch_failure(monkeypatch):
    import yfinance as yf

    def _raise(symbol):
        raise ConnectionError("simulated network failure")

    monkeypatch.setattr(yf, "Ticker", _raise)
    result = fetch_recent_news("AAPL")
    assert result.empty
    assert list(result.columns) == ["date", "headline", "url"]


def test_live_sentiment_snapshot_scores_real_headlines(monkeypatch):
    import yfinance as yf

    items = [_yf_item("Company beats earnings, stock surges"), _yf_item("Analyst downgrade cites weak growth")]
    monkeypatch.setattr(yf, "Ticker", lambda symbol: _FakeTicker(items))

    snapshot = live_sentiment_snapshot("AAPL")
    assert snapshot["available"] is True
    assert len(snapshot["headlines"]) == 2
    assert snapshot["positive_count"] + snapshot["negative_count"] + snapshot["neutral_count"] == 2
    assert -1 <= snapshot["overall_sentiment"] <= 1


def test_live_sentiment_snapshot_unavailable_when_no_news(monkeypatch):
    import yfinance as yf

    monkeypatch.setattr(yf, "Ticker", lambda symbol: _FakeTicker([]))
    snapshot = live_sentiment_snapshot("ZZZZ")
    assert snapshot["available"] is False
    assert snapshot["overall_sentiment"] is None
    assert snapshot["headlines"] == []
