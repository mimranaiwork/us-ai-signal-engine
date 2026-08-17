import SignalBadge from "./SignalBadge.jsx";
import ScoreGauge from "./ScoreGauge.jsx";
import ConsensusMeters from "./ConsensusMeters.jsx";
import HorizonBreakdown from "./HorizonBreakdown.jsx";
import KeyStats from "./KeyStats.jsx";
import PriceChart from "./PriceChart.jsx";
import SupportResistance from "./SupportResistance.jsx";
import NewsSentiment from "./NewsSentiment.jsx";
import ReasonsRisks from "./ReasonsRisks.jsx";

function titleCase(s) {
  if (!s) return "";
  return s
    .toLowerCase()
    .split("_")
    .map((w) => w[0].toUpperCase() + w.slice(1))
    .join(" ");
}

const RISK_TAG_CLASS = {
  LOW: "tag--risk-low",
  MEDIUM: "tag--risk-medium",
  HIGH: "tag--risk-high",
};

function EarningsBanner({ earnings, risk }) {
  if (!earnings || earnings.days_to_earnings == null) return null;
  const estimateTag = earnings.days_to_earnings_is_estimate ? " (estimated)" : "";
  return (
    <div className="earnings-banner">
      ⚠ Earnings expected in {earnings.days_to_earnings} day(s){estimateTag}
      {risk === "HIGH" ? " — forcing conservative risk assessment." : "."}
    </div>
  );
}

export default function SignalCard({ data }) {
  return (
    <div className="signal-card">
      <EarningsBanner earnings={data.earnings} risk={data.risk} />

      <div className="signal-card__hero">
        <SignalBadge signal={data.signal} />
        <div className="hero-score">
          <ScoreGauge score={data.score} />
        </div>
      </div>

      <div className="tag-row">
        <span className={`tag ${data.market_regime === "BULL" ? "tag--regime-bull" : data.market_regime === "BEAR" ? "tag--regime-bear" : ""}`}>
          Market: {titleCase(data.market_regime)}
        </span>
        <span className="tag">Stock regime: {titleCase(data.stock_regime)}</span>
        <span className="tag">{titleCase(data.sector)}</span>
        <span className="tag">{titleCase(data.industry)}</span>
        <span className={`tag ${RISK_TAG_CLASS[data.risk] || ""}`}>Risk: {data.risk}</span>
        {data.cached && <span className="tag">cached</span>}
      </div>

      <KeyStats data={data} />

      <ConsensusMeters
        confidence={data.confidence}
        modelConsensus={data.model_consensus}
        horizonConsensus={data.horizon_consensus}
      />

      <HorizonBreakdown horizons={data.horizons} stackingUsed={data.stacking_used} />

      <PriceChart symbol={data.symbol} levels={data.levels} />

      <SupportResistance levels={data.levels} />

      <NewsSentiment news={data.news} />

      <ReasonsRisks reasons={data.reasons} risks={data.risks} />

      <p className="footer-note">As of {data.as_of}</p>
    </div>
  );
}
