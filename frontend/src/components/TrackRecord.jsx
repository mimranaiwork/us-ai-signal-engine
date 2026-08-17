import { useEffect, useState } from "react";
import { fetchTrackRecord } from "../api.js";

const HORIZON_ORDER = ["1D", "1W", "2W", "1M", "2M", "3M", "6M"];

function pct(v, digits = 1) {
  return v == null ? "—" : `${(v * 100).toFixed(digits)}%`;
}

function Section({ title, subtitle, children }) {
  return (
    <div className="signal-card" style={{ marginBottom: 20 }}>
      <p className="section-label" style={{ fontSize: 13, marginBottom: 2 }}>
        {title}
      </p>
      {subtitle && <p className="footer-note" style={{ marginBottom: 14 }}>{subtitle}</p>}
      {children}
    </div>
  );
}

function TreeModelWalkForward({ data }) {
  if (!data.available) return <p className="rr-empty">No walk-forward report found — run scripts/run_backtest.py.</p>;
  const horizons = Object.keys(data.by_horizon).sort((a, b) => HORIZON_ORDER.indexOf(a) - HORIZON_ORDER.indexOf(b));

  return (
    <div className="screener__table-wrap">
      <table className="screener__table">
        <thead>
          <tr>
            <th>Horizon</th>
            <th>Folds</th>
            <th className="screener__th--numeric">Avg accuracy</th>
            <th className="screener__th--numeric">Avg F1 (macro)</th>
          </tr>
        </thead>
        <tbody>
          {horizons.map((h) => {
            const folds = data.by_horizon[h];
            const avgAcc = folds.reduce((s, f) => s + f.accuracy, 0) / folds.length;
            const avgF1 = folds.reduce((s, f) => s + f.f1_macro, 0) / folds.length;
            return (
              <tr key={h}>
                <td className="screener__symbol">{h}</td>
                <td>{folds.length}</td>
                <td className="screener__td--numeric">{pct(avgAcc)}</td>
                <td className="screener__td--numeric">{avgF1.toFixed(3)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function TradingSimulation({ data }) {
  if (!data.available || data.results.length === 0) return <p className="rr-empty">No trading simulation results found.</p>;
  return (
    <div className="screener__table-wrap">
      <table className="screener__table">
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Horizon</th>
            <th className="screener__th--numeric">Strategy return</th>
            <th className="screener__th--numeric">Buy &amp; hold return</th>
            <th className="screener__th--numeric">Sharpe</th>
            <th className="screener__th--numeric">Max drawdown</th>
            <th className="screener__th--numeric">Win rate</th>
            <th className="screener__th--numeric">Trades</th>
          </tr>
        </thead>
        <tbody>
          {data.results.map((r, i) => (
            <tr key={i}>
              <td className="screener__symbol">{r.symbol}</td>
              <td>{r.horizon}</td>
              <td className={`screener__td--numeric ${r.strategy_return >= 0 ? "sr-target__profit--pos" : "sr-target__profit--neg"}`}>
                {pct(r.strategy_return)}
              </td>
              <td className="screener__td--numeric">{pct(r.buy_and_hold_return)}</td>
              <td className="screener__td--numeric">{r.strategy_sharpe?.toFixed(2) ?? "—"}</td>
              <td className="screener__td--numeric">{pct(r.max_drawdown)}</td>
              <td className="screener__td--numeric">{pct(r.win_rate)}</td>
              <td className="screener__td--numeric">{r.n_trades}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function StackingComparison({ data }) {
  if (!data.available) return <p className="rr-empty">No stacking report found — run scripts/train_stacking.py.</p>;
  const horizons = Object.keys(data.by_horizon).sort((a, b) => HORIZON_ORDER.indexOf(a) - HORIZON_ORDER.indexOf(b));
  return (
    <>
      <p className="footer-note" style={{ marginBottom: 14 }}>
        Stacking wins {data.summary.stacking_wins} of {data.summary.total_horizons} horizons (by macro-F1 on held-out
        test data).
      </p>
      <div className="screener__table-wrap">
        <table className="screener__table">
          <thead>
            <tr>
              <th>Horizon</th>
              <th className="screener__th--numeric">Stacking F1</th>
              <th className="screener__th--numeric">Weighted-avg F1</th>
              <th>Winner</th>
            </tr>
          </thead>
          <tbody>
            {horizons.map((h) => {
              const row = data.by_horizon[h];
              return (
                <tr key={h}>
                  <td className="screener__symbol">{h}</td>
                  <td className="screener__td--numeric">{row.stacking.f1_macro.toFixed(3)}</td>
                  <td className="screener__td--numeric">{row.weighted_average.f1_macro.toFixed(3)}</td>
                  <td>
                    <span className={`tag ${row.winner === "stacking" ? "tag--breakout" : ""}`}>{row.winner.replace("_", " ")}</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}

function SequenceWalkForward({ data }) {
  if (!data.available) return <p className="rr-empty">No sequence walk-forward report found — run scripts/run_sequence_walk_forward.py.</p>;
  return (
    <>
      <p className="footer-note" style={{ marginBottom: 14 }}>{data.note}</p>
      <div className="screener__table-wrap">
        <table className="screener__table">
          <thead>
            <tr>
              <th>Architecture</th>
              <th>Horizon</th>
              <th>Test window</th>
              <th className="screener__th--numeric">Accuracy</th>
              <th className="screener__th--numeric">F1 (macro)</th>
            </tr>
          </thead>
          <tbody>
            {data.results
              .filter((r) => !r.error)
              .map((r, i) => (
                <tr key={i}>
                  <td className="screener__symbol">{r.architecture}</td>
                  <td>{r.horizon}</td>
                  <td>{r.fold.val_end} → {r.fold.test_end}</td>
                  <td className="screener__td--numeric">{pct(r.test_classification.accuracy)}</td>
                  <td className="screener__td--numeric">{r.test_classification.f1_macro.toFixed(3)}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

export default function TrackRecord() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchTrackRecord()
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <p className="state-message">Loading track record...</p>;
  if (error) return <p className="state-message state-message--error">{error}</p>;
  if (!data) return null;

  return (
    <div>
      <Section
        title="Tree-model walk-forward accuracy"
        subtitle="XGBoost/LightGBM retrained per expanding-window fold, scored on each fold's held-out test window — real out-of-sample results, not a single static split."
      >
        <TreeModelWalkForward data={data.tree_model_walk_forward} />
      </Section>

      <Section
        title="Trading simulation vs. Buy & Hold"
        subtitle="Simulated long-only strategy following the model's signals, transaction costs included, compared against simply buying and holding the same symbol."
      >
        <TradingSimulation data={data.trading_simulation} />
      </Section>

      <Section
        title="Stacking vs. weighted-average ensemble"
        subtitle="Per-horizon comparison of the logistic-regression stacking meta-model against the fixed-weight average, both scored on the same held-out test data."
      >
        <StackingComparison data={data.stacking} />
      </Section>

      <Section
        title="LSTM / Transformer walk-forward"
        subtitle="Sequence models genuinely retrained per fold (not just re-scored on a single split) — a representative subset, see the note below for exactly what's covered."
      >
        <SequenceWalkForward data={data.sequence_walk_forward} />
      </Section>
    </div>
  );
}
