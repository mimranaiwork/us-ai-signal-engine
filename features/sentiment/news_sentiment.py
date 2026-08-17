"""
News sentiment (spec section 22): a lexicon-based scorer and topic
classifier that operate on whatever text a real news source hands them.

This module deliberately does NOT ship any "sample news data" — fabricating
plausible-looking financial headlines and presenting them as sample data
would be manufacturing fake news content, which is a different (and worse)
problem than the "synthetic OHLCV" approach used elsewhere in this project
(price paths are obviously synthetic math; invented headlines read as real
text). `fetch_recent_news`/`live_sentiment_snapshot` pull REAL current
headlines via yfinance's free `.news` endpoint (no API key) — the live
signal card's news panel; a full historical news API (NewsAPI, Benzinga,
Alpha Vantage News, etc.) would still be needed for point-in-time
training/backtesting features, since yfinance's news is current-only, no
archive. tests/test_sentiment.py exercises the scorer/classifier against a
handful of illustrative sentences written for testing, not represented as
real news.

Approach is a compact finance-oriented positive/negative word list (in the
spirit of the Loughran-McDonald finance sentiment dictionary — not a
reproduction of it, a small original list scoped to what's needed here),
not an ML model — transparent, dependency-free, and auditable: every score
traces back to specific matched words rather than an opaque black box.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

POSITIVE_WORDS = {
    "beat", "beats", "beating", "surge", "surged", "surging", "rally", "rallied",
    "outperform", "outperformed", "upgrade", "upgraded", "raise", "raised",
    "record", "strong", "strength", "growth", "profit", "profitable", "gain", "gains",
    "exceed", "exceeded", "exceeds", "bullish", "optimistic", "expansion", "accelerate",
    "accelerated", "breakthrough", "approval", "approved", "buyback", "dividend increase",
}

NEGATIVE_WORDS = {
    "miss", "missed", "misses", "plunge", "plunged", "plunging", "slump", "slumped",
    "downgrade", "downgraded", "cut", "cuts", "layoff", "layoffs", "weak", "weakness",
    "decline", "declined", "loss", "losses", "lawsuit", "investigation", "recall",
    "bearish", "pessimistic", "contraction", "delay", "delayed", "warning", "warns",
    "bankruptcy", "fraud", "restatement", "resign", "resigned", "scandal",
}

TOPIC_KEYWORDS: dict[str, set[str]] = {
    "earnings": {"earnings", "eps", "quarterly", "revenue", "guidance"},
    "guidance": {"guidance", "forecast", "outlook"},
    "m_and_a": {"acquire", "acquires", "acquisition", "merger", "merge", "takeover", "buyout"},
    "legal": {"lawsuit", "litigation", "settlement", "sue", "sued"},
    "regulatory": {"regulation", "regulatory", "sec", "ftc", "antitrust", "compliance"},
    "product": {"launch", "launches", "product", "release", "unveil", "unveils"},
    "management": {"ceo", "cfo", "executive", "resign", "resigned", "appoint", "appointed"},
    "analyst": {"analyst", "upgrade", "downgrade", "price target", "rating"},
    "macro": {"fed", "inflation", "rates", "gdp", "unemployment", "recession"},
}


def _tokenize(text: str) -> set[str]:
    return set("".join(c.lower() if c.isalnum() else " " for c in text).split())


def score_text(text: str) -> float:
    """Returns a score in [-1, 1]: (positive_hits - negative_hits) / total_hits,
    0.0 if no sentiment words are present at all."""
    tokens = _tokenize(text)
    pos = len(tokens & POSITIVE_WORDS)
    neg = len(tokens & NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def classify_topic(text: str) -> str:
    tokens = _tokenize(text)
    best_topic, best_overlap = "other", 0
    for topic, keywords in TOPIC_KEYWORDS.items():
        overlap = len(tokens & keywords)
        if overlap > best_overlap:
            best_topic, best_overlap = topic, overlap
    return best_topic


def score_articles(articles: pd.DataFrame) -> pd.DataFrame:
    """articles: DataFrame with columns `date` (publish date) and `headline`
    (or `text`) — from a real news source. Adds `sentiment` and `topic`."""
    df = articles.copy()
    text_col = "headline" if "headline" in df.columns else "text"
    df["sentiment"] = df[text_col].apply(score_text)
    df["topic"] = df[text_col].apply(classify_topic)
    return df


def fetch_recent_news(symbol: str, max_items: int = 8) -> pd.DataFrame:
    """Real, CURRENT headlines via yfinance — free, no API key. Only recent
    (whatever Yahoo currently has, typically the last few days), no
    historical archive — so this powers a live display feature only, NOT
    point-in-time training/backtesting data (unlike OHLCV, which yfinance
    does provide historically). Returns DataFrame(date, headline, url),
    empty (not fabricated) if the symbol has no recent news or the fetch
    fails for any reason (network, rate limit, yfinance schema change)."""
    import yfinance as yf

    try:
        raw = yf.Ticker(symbol).news or []
    except Exception:
        return pd.DataFrame(columns=["date", "headline", "url"])

    rows = []
    for item in raw[:max_items]:
        content = item.get("content", item)  # yfinance's news payload shape has changed before; tolerate both
        title = content.get("title")
        if not title:
            continue
        url = (content.get("canonicalUrl") or {}).get("url") or (content.get("clickThroughUrl") or {}).get("url")
        rows.append({"date": content.get("pubDate"), "headline": title, "url": url})
    return pd.DataFrame(rows, columns=["date", "headline", "url"])


def live_sentiment_snapshot(symbol: str, max_items: int = 8) -> dict:
    """Live sentiment snapshot for the signal card: scores each recent real
    headline with score_text/classify_topic and returns an overall average
    plus the scored headlines for display. `available=False` (not a
    fabricated neutral score) when there's no recent news to score."""
    news = fetch_recent_news(symbol, max_items=max_items)
    if news.empty:
        return {"available": False, "overall_sentiment": None, "headlines": []}

    scored = score_articles(news)
    return {
        "available": True,
        "overall_sentiment": round(float(scored["sentiment"].mean()), 3),
        "positive_count": int((scored["sentiment"] > 0.1).sum()),
        "negative_count": int((scored["sentiment"] < -0.1).sum()),
        "neutral_count": int((scored["sentiment"].abs() <= 0.1).sum()),
        "headlines": [
            {
                "headline": row["headline"],
                "date": row["date"],
                "sentiment": round(row["sentiment"], 3),
                "topic": row["topic"],
                "url": row["url"],
            }
            for _, row in scored.iterrows()
        ],
    }


def sentiment_features_as_of(scored_articles: pd.DataFrame, as_of_date: date, windows: tuple[int, ...] = (1, 3, 7, 30)) -> dict:
    """
    Point-in-time news features (no future leakage — only articles published
    on/before as_of_date are considered, mirroring
    features/fundamentals/fundamentals.py's as_of pattern).
    """
    available = scored_articles[pd.to_datetime(scored_articles["date"]).dt.date <= as_of_date]
    if available.empty:
        return {f"news_sentiment_{w}d": 0.0 for w in windows} | {
            "positive_news_count": 0, "negative_news_count": 0, "neutral_news_count": 0,
            "sentiment_momentum": 0.0,
        }

    result = {}
    for w in windows:
        window_start = as_of_date - pd.Timedelta(days=w)
        window_articles = available[pd.to_datetime(available["date"]).dt.date > window_start]
        result[f"news_sentiment_{w}d"] = float(window_articles["sentiment"].mean()) if len(window_articles) else 0.0

    recent = available[pd.to_datetime(available["date"]).dt.date > as_of_date - pd.Timedelta(days=7)]
    result["positive_news_count"] = int((recent["sentiment"] > 0.1).sum())
    result["negative_news_count"] = int((recent["sentiment"] < -0.1).sum())
    result["neutral_news_count"] = int((recent["sentiment"].abs() <= 0.1).sum())

    short = result.get("news_sentiment_3d", 0.0)
    long = result.get("news_sentiment_30d", 0.0)
    result["sentiment_momentum"] = short - long

    return result
