"""
SEC EDGAR ingestion (spec section 20) — real, free, no-API-key-required data:

  https://www.sec.gov/files/company_tickers.json          ticker -> CIK mapping
  https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json all reported XBRL
                                                            facts for a filer

SEC requires a descriptive User-Agent identifying the requester (name +
contact email) on every request — see https://www.sec.gov/os/webmaster-faq#developers.
Set it via `SEC_EDGAR_USER_AGENT` env var or pass user_agent= explicitly;
the client refuses to make requests with the default placeholder so nobody
accidentally ships requests SEC can't trace back to a real contact.

Rate-limited to well under SEC's stated ~10 req/sec fair-access guidance.
Every response is cached to disk (data/sec/cache/) so re-running features
doesn't re-hit the network — and so the point-in-time correctness tests in
tests/test_fundamentals.py can run offline against a real, frozen filing set.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sec", "cache")
DEFAULT_USER_AGENT_PLACEHOLDER = "REPLACE_ME your-name your-email@example.com"
MIN_REQUEST_INTERVAL_SECONDS = 0.15  # ~6-7 req/sec, comfortably under SEC's fair-access guidance


class SecEdgarClient:
    def __init__(self, user_agent: str | None = None, cache_dir: str = CACHE_DIR):
        self.user_agent = user_agent or os.environ.get("SEC_EDGAR_USER_AGENT", DEFAULT_USER_AGENT_PLACEHOLDER)
        if self.user_agent == DEFAULT_USER_AGENT_PLACEHOLDER:
            raise ValueError(
                "Set SEC_EDGAR_USER_AGENT (e.g. 'Your Name your-email@example.com') before "
                "calling SEC EDGAR — they require a traceable contact on every request."
            )
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._last_request_time = 0.0
        self._ticker_to_cik: dict[str, str] | None = None

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
            time.sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)
        self._last_request_time = time.monotonic()

    def _get_json(self, url: str, cache_key: str) -> dict:
        cache_path = os.path.join(self.cache_dir, cache_key)
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                return json.load(f)

        self._throttle()
        # No Accept-Encoding header: urllib doesn't auto-decompress gzip/deflate
        # the way `requests` does, so asking for compression here would just
        # hand back bytes json.loads can't parse. Let the server send plain text.
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"SEC EDGAR request failed ({exc.code}) for {url}: {exc.reason}") from exc

        with open(cache_path, "w") as f:
            json.dump(data, f)
        return data

    def ticker_to_cik(self, ticker: str) -> str:
        if self._ticker_to_cik is None:
            raw = self._get_json("https://www.sec.gov/files/company_tickers.json", "company_tickers.json")
            self._ticker_to_cik = {
                entry["ticker"].upper(): str(entry["cik_str"]).zfill(10) for entry in raw.values()
            }
        cik = self._ticker_to_cik.get(ticker.upper())
        if cik is None:
            raise KeyError(f"No CIK found for ticker '{ticker}' in SEC's company_tickers.json")
        return cik

    def company_facts(self, ticker: str) -> dict:
        """Raw XBRL companyfacts JSON — every reported concept (us-gaap:Revenues,
        us-gaap:EarningsPerShareDiluted, etc.), each with a list of {val, fy, fp,
        form, filed, end, start} records. `filed` is the point-in-time-correct
        publication timestamp — see features/fundamentals/fundamentals.py."""
        cik = self.ticker_to_cik(ticker)
        return self._get_json(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", f"companyfacts_{cik}.json"
        )

    def submissions(self, ticker: str) -> dict:
        """Filing history (10-K/10-Q/8-K list with filing dates) — used as the
        earnings-event proxy in features/earnings/earnings.py."""
        cik = self.ticker_to_cik(ticker)
        return self._get_json(f"https://data.sec.gov/submissions/CIK{cik}.json", f"submissions_{cik}.json")
