function formatMarketCap(v) {
  if (v == null) return "—";
  if (v >= 1e12) return `$${(v / 1e12).toFixed(2)}T`;
  if (v >= 1e9) return `$${(v / 1e9).toFixed(2)}B`;
  if (v >= 1e6) return `$${(v / 1e6).toFixed(2)}M`;
  return `$${v.toLocaleString()}`;
}

function formatVolume(v) {
  if (v == null) return "—";
  if (v >= 1e9) return `${(v / 1e9).toFixed(2)}B`;
  if (v >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(1)}K`;
  return v.toLocaleString();
}

function titleCase(s) {
  if (!s) return "—";
  return s
    .toLowerCase()
    .split("_")
    .map((w) => w[0].toUpperCase() + w.slice(1))
    .join(" ");
}

function Stat({ label, value, sub }) {
  return (
    <div className="key-stat">
      <span className="key-stat__label">{label}</span>
      <span className="key-stat__value">{value}</span>
      {sub && <span className="key-stat__sub">{sub}</span>}
    </div>
  );
}

export default function KeyStats({ data }) {
  const earnings = data.earnings || {};
  const nextEarnings = earnings.next_earnings_date;
  const estimateTag = earnings.days_to_earnings_is_estimate ? " (est.)" : "";
  const volumeRatio = data.volume && data.avg_volume_20d ? data.volume / data.avg_volume_20d : null;

  return (
    <>
      <p className="section-label">Key stats</p>
      <div className="key-stats-grid">
        <Stat label="Sector" value={titleCase(data.sector)} sub={titleCase(data.industry)} />
        <Stat label="Market cap" value={formatMarketCap(data.market_cap)} />
        <Stat
          label="Volume"
          value={formatVolume(data.volume)}
          sub={data.avg_volume_20d ? `avg ${formatVolume(data.avg_volume_20d)}${volumeRatio ? ` (${volumeRatio.toFixed(2)}x)` : ""}` : null}
        />
        <Stat
          label="Next earnings"
          value={nextEarnings ? nextEarnings : "Unknown"}
          sub={nextEarnings ? `${earnings.days_to_earnings} day(s)${estimateTag}` : null}
        />
      </div>
    </>
  );
}
