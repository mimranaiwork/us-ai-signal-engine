import { useEffect, useMemo, useState } from "react";
import { fetchScreener } from "../api.js";

const COLUMNS = [
  { key: "symbol", label: "Symbol" },
  { key: "company", label: "Company" },
  { key: "sector", label: "Sector" },
  { key: "signal", label: "Signal" },
  { key: "price", label: "Price", numeric: true },
  { key: "target", label: "Target", numeric: true },
  { key: "expected_gain_pct", label: "Gain %", numeric: true },
  { key: "confidence", label: "Confidence", numeric: true },
  { key: "risk", label: "Risk" },
  { key: "near_breakout", label: "Breakout" },
  { key: "sharia_compliant", label: "Sharia*" },
];

const SIGNAL_FILTERS = ["ALL", "BUY", "HOLD", "SELL"];

function ShariaBadge({ row }) {
  if (row.sharia_compliant == null) return <span className="footer-note">—</span>;
  const label = row.sharia_compliant ? "Compliant" : "Not compliant";
  const cls = row.sharia_compliant ? "sharia-badge--pass" : "sharia-badge--fail";
  return (
    <span className={`sharia-badge ${cls}`} title={row.sharia_note || undefined}>
      {label}
      {row.sharia_note && <sup>†</sup>}
    </span>
  );
}
const AUTO_REFRESH_MS = 2 * 60 * 1000; // checks in more often than the backend's 10-min screener cache so fresh data shows up soon after it's ready, without forcing a recompute

function titleCase(s) {
  if (!s) return "";
  return s
    .toLowerCase()
    .split("_")
    .map((w) => w[0].toUpperCase() + w.slice(1))
    .join(" ");
}

export default function Screener({ onSelectSymbol }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [sortKey, setSortKey] = useState("expected_gain_pct");
  const [sortDir, setSortDir] = useState("desc");
  const [signalFilter, setSignalFilter] = useState("ALL");
  const [breakoutOnly, setBreakoutOnly] = useState(false);
  const [shariaOnly, setShariaOnly] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const load = (refresh = false, silent = false) => {
    if (!silent) setLoading(true);
    setError(null);
    fetchScreener(refresh)
      .then(setData)
      .catch((e) => {
        if (!silent) setError(e.message); // a failed background poll shouldn't blow away a working table
      })
      .finally(() => {
        if (!silent) setLoading(false);
      });
  };

  useEffect(() => {
    load(false);
  }, []);

  useEffect(() => {
    if (!autoRefresh) return;
    const id = setInterval(() => load(false, true), AUTO_REFRESH_MS);
    return () => clearInterval(id);
  }, [autoRefresh]);

  const handleSort = (key) => {
    if (key === sortKey) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  };

  const rows = useMemo(() => {
    if (!data) return [];
    let results = data.results;
    if (signalFilter !== "ALL") results = results.filter((r) => r.signal === signalFilter);
    if (breakoutOnly) results = results.filter((r) => r.near_breakout);
    if (shariaOnly) results = results.filter((r) => r.sharia_compliant === true);

    const sorted = [...results].sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      if (typeof av === "string") return sortDir === "asc" ? av.localeCompare(bv) : bv.localeCompare(av);
      return sortDir === "asc" ? av - bv : bv - av;
    });
    return sorted;
  }, [data, sortKey, sortDir, signalFilter, breakoutOnly, shariaOnly]);

  return (
    <div className="screener">
      <div className="screener__toolbar">
        <div className="screener__filters">
          {SIGNAL_FILTERS.map((f) => (
            <button
              key={f}
              className={`filter-chip ${signalFilter === f ? "filter-chip--active" : ""}`}
              onClick={() => setSignalFilter(f)}
            >
              {f}
            </button>
          ))}
          <button
            className={`filter-chip ${breakoutOnly ? "filter-chip--active" : ""}`}
            onClick={() => setBreakoutOnly((v) => !v)}
          >
            Near breakout only
          </button>
          <button
            className={`filter-chip ${shariaOnly ? "filter-chip--active" : ""}`}
            onClick={() => setShariaOnly((v) => !v)}
            title="Filters to symbols this screen marks compliant — see the disclaimer below the table"
          >
            Sharia compliant only*
          </button>
        </div>
        <div className="screener__meta">
          {data && <span className="footer-note">{rows.length} of {data.count} symbols{data.cached ? " · cached" : ""}</span>}
          <button
            className={`filter-chip ${autoRefresh ? "filter-chip--active" : ""}`}
            onClick={() => setAutoRefresh((v) => !v)}
            title="Check for fresh data every 2 minutes"
          >
            Auto-refresh: {autoRefresh ? "ON" : "OFF"}
          </button>
          <button className="refresh-btn" onClick={() => load(true)} disabled={loading}>
            {loading ? "Scanning..." : "Refresh"}
          </button>
        </div>
      </div>

      {loading && !data && <p className="state-message">Scanning the tracked universe (first run can take ~30s)...</p>}
      {error && <p className="state-message state-message--error">{error}</p>}

      {data && (
        <div className="screener__table-wrap">
          <table className="screener__table">
            <thead>
              <tr>
                {COLUMNS.map((col) => (
                  <th
                    key={col.key}
                    className={col.numeric ? "screener__th--numeric" : ""}
                    onClick={() => handleSort(col.key)}
                  >
                    {col.label}
                    {sortKey === col.key && <span className="screener__sort-arrow">{sortDir === "asc" ? " ▲" : " ▼"}</span>}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.symbol} onClick={() => onSelectSymbol(r.symbol)} className="screener__row">
                  <td className="screener__symbol">{r.symbol}</td>
                  <td>{r.company}</td>
                  <td>{titleCase(r.sector)}</td>
                  <td>
                    <span className={`tag screener__signal-tag--${r.signal.toLowerCase()}`}>{r.signal}</span>
                  </td>
                  <td className="screener__td--numeric">{r.price != null ? r.price.toFixed(2) : "—"}</td>
                  <td className="screener__td--numeric">{r.target != null ? r.target.toFixed(2) : "—"}</td>
                  <td className={`screener__td--numeric ${r.expected_gain_pct >= 0 ? "sr-target__profit--pos" : "sr-target__profit--neg"}`}>
                    {r.expected_gain_pct != null ? `${r.expected_gain_pct >= 0 ? "+" : ""}${r.expected_gain_pct.toFixed(2)}%` : "—"}
                  </td>
                  <td className="screener__td--numeric">{Math.round(r.confidence * 100)}%</td>
                  <td>{r.risk}</td>
                  <td>{r.near_breakout ? <span className="tag tag--breakout">Yes</span> : ""}</td>
                  <td><ShariaBadge row={r} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {data && (
        <p className="footer-note screener__sharia-disclaimer">
          * Unofficial, mechanical approximation (AAOIFI-style business-activity exclusion + debt/liquid-assets/interest-income
          ratio thresholds computed from public SEC filings) — <strong>not a certified Sharia compliance determination</strong>.
          No qualified Islamic finance board reviewed these results. Hover a badge for details; † marks a result with an extra
          caveat (e.g. ratios that couldn't be computed from available data). Verify independently before relying on this.
        </p>
      )}
    </div>
  );
}
