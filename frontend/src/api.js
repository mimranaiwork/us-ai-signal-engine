const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

async function getJSON(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

export function fetchSignal(symbol) {
  return getJSON(`/api/v1/signals/${encodeURIComponent(symbol)}`);
}

export function fetchUniverse() {
  return getJSON(`/api/v1/universe`);
}

export function fetchScreener(refresh = false) {
  return getJSON(`/api/v1/screener${refresh ? "?refresh=true" : ""}`);
}

export function fetchPrices(symbol, days = 120) {
  return getJSON(`/api/v1/prices/${encodeURIComponent(symbol)}?days=${days}`);
}

export function fetchTrackRecord() {
  return getJSON(`/api/v1/track-record`);
}

export function fetchQuotes(symbols) {
  if (!symbols.length) return Promise.resolve({ count: 0, quotes: [] });
  return getJSON(`/api/v1/quotes?symbols=${symbols.map(encodeURIComponent).join(",")}`);
}

export function fetchHealth() {
  return getJSON(`/health`);
}
