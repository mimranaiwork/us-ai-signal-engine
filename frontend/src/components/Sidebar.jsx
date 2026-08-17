import { useEffect, useMemo, useState } from "react";
import { fetchQuotes } from "../api.js";

const BADGE_CLASS = {
  BUY: "signal-badge--buy",
  SELL: "signal-badge--sell",
  HOLD: "signal-badge--hold",
};

const QUOTES_REFRESH_MS = 2 * 60 * 1000;

export default function Sidebar({ universe, watchlist, signalsBySymbol, activeSymbol, onSelect, view, onViewChange }) {
  const [query, setQuery] = useState("");
  const [quotesBySymbol, setQuotesBySymbol] = useState({});

  useEffect(() => {
    if (watchlist.length === 0) return;
    const load = () => {
      fetchQuotes(watchlist)
        .then((res) => {
          const bySymbol = {};
          for (const q of res.quotes) bySymbol[q.symbol] = q;
          setQuotesBySymbol((prev) => ({ ...prev, ...bySymbol }));
        })
        .catch(() => {}); // quotes are a nice-to-have on the sidebar, never block the rest of the UI
    };
    load();
    const id = setInterval(load, QUOTES_REFRESH_MS);
    return () => clearInterval(id);
  }, [watchlist.join(",")]);

  const filtered = useMemo(() => {
    const q = query.trim().toUpperCase();
    if (!q) return [];
    return universe.filter((s) => s.startsWith(q)).slice(0, 8);
  }, [query, universe]);

  const handleSubmit = (e) => {
    e.preventDefault();
    const symbol = query.trim().toUpperCase();
    if (symbol) {
      onSelect(symbol);
      setQuery("");
    }
  };

  return (
    <div className="sidebar">
      <div className="sidebar__header">
        <p className="sidebar__title">US AI SIGNAL ENGINE</p>
        <div className="nav-tabs">
          <button
            className={`nav-tab ${view === "watchlist" ? "nav-tab--active" : ""}`}
            onClick={() => onViewChange("watchlist")}
          >
            Watchlist
          </button>
          <button
            className={`nav-tab ${view === "screener" ? "nav-tab--active" : ""}`}
            onClick={() => onViewChange("screener")}
          >
            Screener
          </button>
          <button
            className={`nav-tab ${view === "track-record" ? "nav-tab--active" : ""}`}
            onClick={() => onViewChange("track-record")}
          >
            Track Record
          </button>
        </div>
        <form className="search-form" onSubmit={handleSubmit}>
          <input
            className="search-input"
            placeholder="Search symbol..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            list="universe-symbols"
          />
          <button type="submit" className="search-submit" aria-label="Add symbol">
            Go
          </button>
        </form>
        {filtered.length > 0 && (
          <datalist id="universe-symbols">
            {filtered.map((s) => (
              <option value={s} key={s} />
            ))}
          </datalist>
        )}
      </div>

      <div className="watchlist">
        {watchlist.length === 0 && <p className="watchlist__empty">Search a symbol to get started.</p>}
        {watchlist.map((symbol) => {
          const signal = signalsBySymbol[symbol]?.signal;
          const quote = quotesBySymbol[symbol];
          return (
            <div
              key={symbol}
              className={`watchlist__row ${symbol === activeSymbol ? "watchlist__row--active" : ""}`}
              onClick={() => onSelect(symbol)}
            >
              <div className="watchlist__row-top">
                <span className="watchlist__symbol">{symbol}</span>
                {signal && <span className={`watchlist__badge ${BADGE_CLASS[signal] || ""}`}>{signal}</span>}
              </div>
              {quote && quote.available && (
                <div className="watchlist__row-bottom">
                  <span className="watchlist__price">{quote.price.toFixed(2)}</span>
                  <span className={quote.change >= 0 ? "watchlist__change--pos" : "watchlist__change--neg"}>
                    {quote.change >= 0 ? "+" : ""}
                    {quote.change?.toFixed(2)} ({quote.change_pct >= 0 ? "+" : ""}
                    {quote.change_pct?.toFixed(2)}%)
                  </span>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
