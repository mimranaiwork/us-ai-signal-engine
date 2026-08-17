const HORIZON_ORDER = ["1D", "1W", "2W", "1M", "2M", "3M", "6M"];

const SIGNAL_CLASS = {
  BUY: "horizon-chip__signal--buy",
  SELL: "horizon-chip__signal--sell",
  HOLD: "horizon-chip__signal--hold",
};

export default function HorizonBreakdown({ horizons, stackingUsed }) {
  const stackingSet = new Set(stackingUsed || []);
  return (
    <>
      <p className="section-label">Horizons</p>
      <div className="horizon-row">
        {HORIZON_ORDER.filter((h) => horizons[h]).map((h) => {
          const { signal, score } = horizons[h];
          return (
            <div className="horizon-chip" key={h}>
              <div className="horizon-chip__label">{h}</div>
              <div className={`horizon-chip__signal ${SIGNAL_CLASS[signal] || ""}`}>{signal}</div>
              <div className="horizon-chip__score">{score.toFixed(3)}</div>
              {stackingSet.has(h) && <div className="horizon-chip__stack">stacked</div>}
            </div>
          );
        })}
      </div>
    </>
  );
}
