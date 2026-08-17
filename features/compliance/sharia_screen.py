"""
DIY Sharia-compliance screen — an unofficial, mechanical approximation of
the methodology real index providers use (S&P Shariah, Dow Jones Islamic
Market, MSCI Islamic, AAOIFI-based screens like Zoya/Musaffa). This is NOT
a certified Sharia compliance determination: no qualified Islamic finance
board reviewed these results, real index providers can and do disagree on
borderline names, and thresholds vary by methodology (AAOIFI vs. S&P vs.
Dow Jones all differ slightly). Treat this as one transparent, reproducible
mechanical input — not a religious ruling.

Two-part screen, both must pass for "compliant":

1. Business activity — sector/industry exclusion of companies whose core
   business is inherently non-compliant (conventional interest-based
   banking/lending, conventional insurance, tobacco, alcohol, gambling,
   pork, adult entertainment, conventional weapons in some methodologies).
   This project's universe only has representatives of banking, insurance,
   and tobacco, so only those are wired into EXCLUDED_INDUSTRIES — the
   others are listed for completeness, not because they're needed today.

2. Financial ratios (AAOIFI-style; thresholds are the commonly-cited
   industry convention, not backtested/optimized by this project — same
   honesty class as features/technical/pivot_points.py's
   BREAKOUT_PROXIMITY_PCT):
     debt / market_cap                          < 33%
     (cash + short-term investments) / market_cap < 33%
     interest income / revenue                    < 5%

   Real SEC XBRL data backs these where available. Interest income in
   particular is inconsistently tagged — many non-financial filers don't
   break it out at all — so a ratio that can't be computed is reported as
   `None` ("unknown"), never silently assumed to pass.
"""

from __future__ import annotations

from datetime import date

from features.fundamentals.fundamentals import as_of_fundamentals, build_fundamentals_long

# Industries whose core business model is inherently non-compliant
# regardless of financial ratios (interest-based lending/insurance IS the
# business, not an incidental income source). Only BANKS/INSURANCE/TOBACCO
# currently have members in this project's tracked universe; the rest are
# listed for completeness in case the universe grows into them.
EXCLUDED_INDUSTRIES = {
    "BANKS", "INSURANCE", "TOBACCO",
    # not currently present in the universe, included for completeness:
    "ALCOHOL", "GAMBLING", "ADULT_ENTERTAINMENT", "PORK_PRODUCTS",
}

# Industries adjacent to conventional finance that are NOT auto-excluded
# (asset managers earn fees, not interest; payment networks earn transaction
# fees, not interest — real Islamic screens do let both proceed to ratio
# screening rather than blanket-excluding them) but whose SEC filings use
# bank-style XBRL tags (deposits, trading liabilities, etc.) that this
# screen's concept list doesn't recognize — found for real running this
# screen against Goldman Sachs/BlackRock/Visa/Mastercard, all of which came
# back with every ratio "unknown" rather than a real computed number. A
# "provisional compliant" for one of these carries materially less
# information than for an industrial/tech company with clean data, and is
# flagged with an extra warning rather than reading the same as a real pass.
FINANCE_ADJACENT_LOW_COVERAGE_INDUSTRIES = {"ASSET_MANAGEMENT", "PAYMENTS"}

DEBT_TO_MARKET_CAP_MAX = 0.33
LIQUID_ASSETS_TO_MARKET_CAP_MAX = 0.33
INTEREST_INCOME_TO_REVENUE_MAX = 0.05


def business_activity_screen(industry_code: str) -> dict:
    passed = industry_code not in EXCLUDED_INDUSTRIES
    return {
        "passed": passed,
        "reason": None if passed else f"Industry '{industry_code}' is inherently interest-based or otherwise excluded by business activity",
    }


def financial_ratio_screen(values: dict[str, float], market_cap: float | None) -> dict:
    """values: as_of_fundamentals() output — expects (best-effort)
    long_term_debt, short_term_debt, short_term_investments, cash, revenue,
    interest_income; any of these may be absent from SEC filings.
    market_cap: from universe reference data (approximate, see
    universe/universe_loader.py)."""

    def ratio(numerator_keys: list[str], denominator: float | None) -> tuple[float | None, bool]:
        """Returns (ratio_or_None, all_inputs_known). If every numerator
        component is missing, the ratio is unknown, not assumed zero —
        SEC filers with genuinely zero debt just don't tag the field, same
        as filers who simply weren't extracted; the two can't be told apart
        from this data alone."""
        present = [values[k] for k in numerator_keys if values.get(k) is not None]
        if not present or denominator is None or denominator == 0:
            return None, False
        return sum(present) / denominator, True

    debt_ratio, debt_known = ratio(["long_term_debt", "short_term_debt"], market_cap)
    liquid_ratio, liquid_known = ratio(["cash", "short_term_investments"], market_cap)

    interest_income = values.get("interest_income")
    revenue = values.get("revenue")
    if interest_income is not None and revenue:
        interest_ratio = interest_income / revenue
        interest_known = True
    else:
        interest_ratio, interest_known = None, False

    checks = {
        "debt_to_market_cap": {
            "value": debt_ratio, "threshold": DEBT_TO_MARKET_CAP_MAX, "known": debt_known,
            "passed": (debt_ratio < DEBT_TO_MARKET_CAP_MAX) if debt_known else None,
        },
        "liquid_assets_to_market_cap": {
            "value": liquid_ratio, "threshold": LIQUID_ASSETS_TO_MARKET_CAP_MAX, "known": liquid_known,
            "passed": (liquid_ratio < LIQUID_ASSETS_TO_MARKET_CAP_MAX) if liquid_known else None,
        },
        "interest_income_to_revenue": {
            "value": interest_ratio, "threshold": INTEREST_INCOME_TO_REVENUE_MAX, "known": interest_known,
            "passed": (interest_ratio < INTEREST_INCOME_TO_REVENUE_MAX) if interest_known else None,
        },
    }
    # Only a KNOWN failing ratio fails the overall screen — an unknown ratio
    # is flagged, not treated as a failure (see module docstring).
    any_known_failure = any(c["passed"] is False for c in checks.values())
    all_known = all(c["known"] for c in checks.values())

    return {
        "passed": not any_known_failure,
        "fully_determined": all_known,
        "checks": checks,
    }


def screen_symbol(symbol: str, industry_code: str, market_cap: float | None, sec_client, as_of_date: date) -> dict:
    """Full screen for one symbol: business activity + financial ratios.
    `sec_client`: an ingestion.sec_edgar.SecEdgarClient instance (injected
    so callers can share one client/cache across a batch, and so this stays
    mockable in tests without hitting the network)."""
    activity = business_activity_screen(industry_code)
    if not activity["passed"]:
        return {
            "symbol": symbol, "compliant": False, "fully_determined": True,
            "business_activity": activity, "financial_ratios": None,
            "note": "Excluded on business activity — financial ratios not evaluated (moot).",
        }

    try:
        facts = sec_client.company_facts(symbol)
        long_df = build_fundamentals_long(facts)
        values = as_of_fundamentals(long_df, as_of_date)
    except Exception:
        values = {}

    ratios = financial_ratio_screen(values, market_cap)

    note = None
    if not ratios["fully_determined"]:
        note = "One or more ratios could not be computed from available SEC filing data — treat 'compliant' as provisional."
        no_ratios_known = all(not c["known"] for c in ratios["checks"].values())
        if industry_code in FINANCE_ADJACENT_LOW_COVERAGE_INDUSTRIES and no_ratios_known:
            note = (
                f"'{industry_code}' companies commonly file using bank-style XBRL tags this screen doesn't "
                "recognize — NONE of the three ratios could be computed, not just one. This 'compliant' result "
                "reflects an absence of data, not a real pass; verify independently before relying on it."
            )

    return {
        "symbol": symbol,
        "compliant": activity["passed"] and ratios["passed"],
        "fully_determined": ratios["fully_determined"],
        "business_activity": activity,
        "financial_ratios": ratios,
        "note": note,
    }
