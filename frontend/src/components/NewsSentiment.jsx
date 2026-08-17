function sentimentLabel(score) {
  if (score > 0.1) return { text: "Positive", cls: "news-pill--pos" };
  if (score < -0.1) return { text: "Negative", cls: "news-pill--neg" };
  return { text: "Neutral", cls: "news-pill--neu" };
}

function timeAgo(isoDate) {
  if (!isoDate) return "";
  const diffMs = Date.now() - new Date(isoDate).getTime();
  const hours = Math.floor(diffMs / 3_600_000);
  if (hours < 1) return "just now";
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

function Headline({ item }) {
  const { text, cls } = sentimentLabel(item.sentiment);
  return (
    <li className="news-item">
      <a href={item.url} target="_blank" rel="noreferrer" className="news-item__title">
        {item.headline}
      </a>
      <div className="news-item__meta">
        <span className={`news-pill ${cls}`}>{text}</span>
        {item.topic !== "other" && <span className="news-pill news-pill--topic">{item.topic.replace(/_/g, " ")}</span>}
        <span className="news-item__time">{timeAgo(item.date)}</span>
      </div>
    </li>
  );
}

export default function NewsSentiment({ news }) {
  if (!news || !news.available) {
    return (
      <>
        <p className="section-label">News &amp; sentiment</p>
        <p className="rr-empty">No recent news available for this symbol.</p>
      </>
    );
  }

  const overall = sentimentLabel(news.overall_sentiment);

  return (
    <>
      <div className="sr-header">
        <p className="section-label" style={{ margin: 0 }}>
          News &amp; sentiment
        </p>
        <span className={`news-pill ${overall.cls}`}>
          Overall: {overall.text} ({news.overall_sentiment.toFixed(2)})
        </span>
      </div>
      <ul className="news-list">
        {news.headlines.map((item, i) => (
          <Headline item={item} key={i} />
        ))}
      </ul>
      <p className="footer-note">
        Live headlines via Yahoo Finance, scored with a lexicon-based sentiment model — current only, not used to train
        the signal models.
      </p>
    </>
  );
}
