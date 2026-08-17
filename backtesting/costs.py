"""
Transaction cost model (spec section 43): commission, bid/ask spread, and
slippage, all configurable rather than assumed zero. Defaults are rough,
documented placeholders for liquid US large-caps — tune them per broker/
venue before trusting the backtest's net-of-cost numbers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TransactionCostModel:
    commission_per_share: float = 0.0035   # e.g. IBKR-style per-share commission
    commission_minimum: float = 0.35
    spread_bps: float = 2.0                # half-spread paid on entry AND exit, in basis points
    slippage_bps: float = 1.0              # additional market-impact/slippage, in basis points

    def round_trip_cost_pct(self) -> float:
        """Approximate cost of one entry + one exit, as a fraction of trade value
        (excludes the flat commission, which is share-count- not price-dependent)."""
        return 2 * (self.spread_bps + self.slippage_bps) / 10_000

    def commission(self, shares: float) -> float:
        return max(self.commission_minimum, abs(shares) * self.commission_per_share)

    def entry_or_exit_cost_pct(self) -> float:
        """Spread + slippage cost of ONE side of a trade (entry or exit), as a
        fraction of trade value — use round_trip_cost_pct() for a full trip."""
        return (self.spread_bps + self.slippage_bps) / 10_000
