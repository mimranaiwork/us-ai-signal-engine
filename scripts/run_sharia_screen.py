"""
Runs the DIY Sharia-compliance screen (features/compliance/sharia_screen.py)
across the tracked universe and writes a report — NOT computed live per API
request, since it needs one SEC EDGAR company_facts() call per symbol.
Fundamentals change quarterly at most, so re-run this on demand (or wire it
into the nightly Celery pipeline if you want it to auto-refresh), not on
every page load.

See features/compliance/sharia_screen.py's module docstring for the full
methodology and honesty caveats — this is an unofficial mechanical
approximation, not a certified Sharia compliance determination.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from features.compliance.sharia_screen import screen_symbol  # noqa: E402
from ingestion.sec_edgar import SecEdgarClient  # noqa: E402
from universe.universe_loader import Universe  # noqa: E402

REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "saved", "sharia_screen_report.json")


def main():
    universe = Universe()
    symbols = universe.all_symbols(active_only=True)
    client = SecEdgarClient()
    as_of = date.today()

    results = []
    for i, symbol in enumerate(symbols):
        stock = universe.get(symbol)
        result = screen_symbol(symbol, stock.industry_code, stock.market_cap, client, as_of)
        result["company_name"] = stock.company_name
        result["industry"] = stock.industry_code
        results.append(result)
        status = "COMPLIANT" if result["compliant"] else "NOT COMPLIANT"
        determined = "" if result["fully_determined"] else " (provisional — some data unknown)"
        print(f"  [{i+1}/{len(symbols)}] {symbol}: {status}{determined}")

    compliant_count = sum(1 for r in results if r["compliant"])
    report = {
        "as_of": as_of.isoformat(),
        "methodology_note": (
            "Unofficial DIY approximation of AAOIFI-style screening — NOT a certified "
            "Sharia compliance determination. See features/compliance/sharia_screen.py."
        ),
        "summary": {"total": len(results), "compliant": compliant_count, "not_compliant": len(results) - compliant_count},
        "results": results,
    }

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{compliant_count}/{len(results)} symbols pass this screen.")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
