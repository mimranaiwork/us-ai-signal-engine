const CLASS_BY_SIGNAL = {
  BUY: "signal-badge--buy",
  SELL: "signal-badge--sell",
  HOLD: "signal-badge--hold",
};

export default function SignalBadge({ signal }) {
  const cls = CLASS_BY_SIGNAL[signal] || "signal-badge--hold";
  return <div className={`signal-badge ${cls}`}>{signal}</div>;
}
