import { useCallback, useEffect, useState } from "react";
import Sidebar from "./components/Sidebar.jsx";
import SignalCard from "./components/SignalCard.jsx";
import Screener from "./components/Screener.jsx";
import TrackRecord from "./components/TrackRecord.jsx";
import { fetchSignal, fetchUniverse } from "./api.js";
import "./App.css";

const DEFAULT_WATCHLIST = ["NVDA", "AAPL", "MSFT", "TSLA", "JPM"];
const STORAGE_KEY = "signal-engine-watchlist";

function loadWatchlist() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : DEFAULT_WATCHLIST;
  } catch {
    return DEFAULT_WATCHLIST;
  }
}

const AUTO_REFRESH_MS = 5 * 60 * 1000; // matches the backend's per-signal cache TTL (monitoring/cache.py) — polling faster would just re-fetch the same cached response

export default function App() {
  const [view, setView] = useState("watchlist"); // "watchlist" | "screener"
  const [universe, setUniverse] = useState([]);
  const [watchlist, setWatchlist] = useState(loadWatchlist);
  const [activeSymbol, setActiveSymbol] = useState(watchlist[0] || null);
  const [signalsBySymbol, setSignalsBySymbol] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  useEffect(() => {
    fetchUniverse()
      .then((res) => setUniverse(res.symbols || []))
      .catch(() => setUniverse([]));
  }, []);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(watchlist));
  }, [watchlist]);

  const loadSignal = useCallback(async (symbol) => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchSignal(symbol);
      setSignalsBySymbol((prev) => ({ ...prev, [symbol]: data }));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (activeSymbol && !signalsBySymbol[activeSymbol]) {
      loadSignal(activeSymbol);
    }
  }, [activeSymbol, signalsBySymbol, loadSignal]);

  useEffect(() => {
    if (!autoRefresh || !activeSymbol || view !== "watchlist") return;
    const id = setInterval(() => loadSignal(activeSymbol), AUTO_REFRESH_MS);
    return () => clearInterval(id);
  }, [autoRefresh, activeSymbol, view, loadSignal]);

  const handleSelect = (symbol) => {
    setActiveSymbol(symbol);
    setWatchlist((prev) => (prev.includes(symbol) ? prev : [...prev, symbol]));
  };

  const handleSelectFromScreener = (symbol) => {
    handleSelect(symbol);
    setView("watchlist");
  };

  const handleRefresh = () => {
    if (activeSymbol) loadSignal(activeSymbol);
  };

  const activeData = activeSymbol ? signalsBySymbol[activeSymbol] : null;

  return (
    <div className="app">
      <Sidebar
        universe={universe}
        watchlist={watchlist}
        signalsBySymbol={signalsBySymbol}
        activeSymbol={view === "watchlist" ? activeSymbol : null}
        onSelect={handleSelectFromScreener}
        view={view}
        onViewChange={setView}
      />

      <div className="main">
        {view === "watchlist" && (
          <>
            {activeSymbol && (
              <div className="main__topbar">
                <div className="main__topbar-left">
                  <h1 className="symbol-heading">{activeSymbol}</h1>
                  {activeData && <span className="as-of">as of {activeData.as_of}</span>}
                </div>
                <div className="main__topbar-actions">
                  <button
                    className={`filter-chip ${autoRefresh ? "filter-chip--active" : ""}`}
                    onClick={() => setAutoRefresh((v) => !v)}
                    title="Auto-refresh every 5 minutes"
                  >
                    Auto-refresh: {autoRefresh ? "ON" : "OFF"}
                  </button>
                  <button className="refresh-btn" onClick={handleRefresh} disabled={loading}>
                    {loading ? "Loading..." : "Refresh"}
                  </button>
                </div>
              </div>
            )}

            {!activeSymbol && <p className="state-message">Search for a symbol to see its signal.</p>}
            {activeSymbol && loading && !activeData && <p className="state-message">Loading signal...</p>}
            {activeSymbol && error && !activeData && <p className="state-message state-message--error">{error}</p>}
            {activeData && <SignalCard data={activeData} />}
          </>
        )}

        {view === "screener" && (
          <>
            <div className="main__topbar">
              <div className="main__topbar-left">
                <h1 className="symbol-heading">Screener</h1>
                <span className="as-of">tracked universe, ranked by breakout target</span>
              </div>
            </div>
            <Screener onSelectSymbol={handleSelectFromScreener} />
          </>
        )}

        {view === "track-record" && (
          <>
            <div className="main__topbar">
              <div className="main__topbar-left">
                <h1 className="symbol-heading">Track Record</h1>
                <span className="as-of">real, historical walk-forward validation results</span>
              </div>
            </div>
            <TrackRecord />
          </>
        )}
      </div>
    </div>
  );
}
