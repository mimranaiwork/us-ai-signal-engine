"""
Phase 5 demonstration: pulls REAL fundamentals from SEC EDGAR for a handful
of symbols and prints a point-in-time snapshot, proving the ingestion +
point-in-time logic work against actual filings, not just synthetic
fixtures. Requires SEC_EDGAR_USER_AGENT to be set (see ingestion/sec_edgar.py).

Not wired into scripts/build_phase1_dataset.py's per-symbol loop across the
full universe — SEC's fair-access rate limit means fetching+parsing
companyfacts for all 40 symbols would take a couple of minutes and pull a
few hundred MB of JSON, which is fine to do once but isn't run automatically
on every pipeline build. Run this script directly when you want fundamentals
data refreshed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from features.earnings.earnings import build_filing_events, days_to_next_and_since_last  # noqa: E402
from features.fundamentals.fundamentals import build_fundamentals_long, fundamentals_snapshot  # noqa: E402
from ingestion.sec_edgar import SecEdgarClient  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="*", default=["AAPL", "MSFT", "JPM"])
    parser.add_argument("--as-of", default=date.today().isoformat())
    args = parser.parse_args()

    client = SecEdgarClient()
    as_of_date = date.fromisoformat(args.as_of)

    for symbol in args.symbols:
        print(f"\n=== {symbol} as of {as_of_date} ===")
        facts = client.company_facts(symbol)
        long_df = build_fundamentals_long(facts)
        snapshot = fundamentals_snapshot(long_df, as_of_date)
        print(json.dumps(snapshot, indent=2, default=str))

        submissions = client.submissions(symbol)
        events = build_filing_events(submissions)
        earnings = days_to_next_and_since_last(events, as_of_date)
        print("earnings timing:", earnings)


if __name__ == "__main__":
    main()
