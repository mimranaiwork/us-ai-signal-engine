export default function ScoreGauge({ score }) {
  const clamped = Math.max(-1, Math.min(1, score));
  const pct = ((clamped + 1) / 2) * 100;

  return (
    <div>
      <div className="hero-score__label">
        Composite score <span className="gauge__value">{score.toFixed(3)}</span>
      </div>
      <div className="gauge">
        <div className="gauge__marker" style={{ left: `${pct}%` }} />
      </div>
    </div>
  );
}
