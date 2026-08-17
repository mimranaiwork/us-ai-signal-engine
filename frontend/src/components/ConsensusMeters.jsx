function Meter({ label, value }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div>
      <div className="meter__label">
        <span>{label}</span>
        <span>{Math.round(pct)}%</span>
      </div>
      <div className="meter__track">
        <div className="meter__fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export default function ConsensusMeters({ confidence, modelConsensus, horizonConsensus }) {
  return (
    <div className="meters">
      <Meter label="Confidence" value={confidence} />
      <Meter label="Model consensus" value={modelConsensus} />
      <Meter label="Horizon consensus" value={horizonConsensus} />
    </div>
  );
}
