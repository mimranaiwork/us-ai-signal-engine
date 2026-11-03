"""
Daily Shariah breakout trading run (Alpaca; PAPER unless ALPACA_LIVE=yes).

  1. Universe: SPUS Shariah holdings, minus anything this project's own DIY
     screen marks non-compliant (execution/shariah_universe.py)
  2. Manage open trades: exits per execution/breakout.manage, earnings
     risk cut via the engine's SEC earnings timing, broker-side GTC stop
  3. Scan for breakouts / near-breakouts (execution/breakout.py)
  4. For each breakout, best first: earnings guard, optional ensemble
     confirmation (execution/ai_filter.py), risk-based size, buy, stop
  5. Write trading_state/report.md, save state, optional Telegram summary

Long only, cash only (no margin), no shorting, no options.
Not financial advice and not a religious ruling.

Usage:
  python scripts/run_trading_bot.py --dry-run   # scan + report, no orders
  python scripts/run_trading_bot.py             # trade (market hours)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import yaml

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from execution import ai_filter, breakout, earnings_guard  # noqa: E402
from execution.shariah_universe import build_universe  # noqa: E402

NY = ZoneInfo("America/New_York")
STATE_DIR = os.path.join(ROOT, "trading_state")
logger = logging.getLogger("trading_bot")


def load_config() -> dict:
    with open(os.path.join(ROOT, "config", "settings.yaml")) as f:
        settings = yaml.safe_load(f) or {}
    t = settings.get("trading", {}) or {}
    cfg = {**breakout.DEFAULTS, **(t.get("rules") or {})}
    cfg.update({
        "benchmark": t.get("benchmark", "SPUS"),
        "exclude": t.get("exclude", []) or [],
        "ai_filter_mode": t.get("ai_filter_mode", "veto_sell"),
        "earnings_window": int((settings.get("earnings") or {}).get("risk_window_days", 5)),
        "block_if_earnings_unknown": bool(t.get("block_if_earnings_unknown", False)),
    })
    return cfg


def load_state(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return {"positions": {}, "closed": [], **json.load(f)}
    return {"positions": {}, "closed": []}


def notify(msg: str) -> None:
    logger.info(msg)
    token, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat):
        return
    try:
        import urllib.request
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=json.dumps({"chat_id": chat, "text": msg[:4000]}).encode(),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as exc:
        logger.warning("telegram failed: %s", exc)


def write_report(path, title, source, removed, regime_ok, breakouts, near, state, acted):
    L = [f"# {title}", "",
         f"Universe: {source}." + (f" Removed by this project's DIY screen: {', '.join(removed)}." if removed else ""),
         f"Market filter: {'ON — new buys allowed' if regime_ok else 'OFF — no new buys'}", "", "## Actions"]
    L += [f"- {a}" for a in acted] or ["- none"]
    L += ["", "## Open positions", "| Symbol | Entry | Stop | Breakout level | Since |", "|---|---|---|---|---|"]
    L += [f"| {s} | {p['entry']:.2f} | {p['stop']:.2f} | {p['breakout_level']:.2f} | {p['date']} |"
          for s, p in state["positions"].items()] or ["| — | | | | |"]
    L += ["", "## Breakouts", "| Symbol | Close | Level | Volume × avg | Base depth | Stop |", "|---|---|---|---|---|---|"]
    L += [f"| {s} | {i['close']:.2f} | {i['breakout_level']:.2f} | {i['vol_ratio']:.1f}× | "
          f"{i['base_depth']*100:.0f}% | {i['stop']:.2f} |" for s, i in breakouts] or ["| — | | | | | |"]
    L += ["", "## Near breakout (watchlist)", "| Symbol | Close | Level | Distance | Base depth |", "|---|---|---|---|---|"]
    L += [f"| {s} | {i['close']:.2f} | {i['breakout_level']:.2f} | {i['dist_pct']:.1f}% | "
          f"{i['base_depth']*100:.0f}% |" for s, i in near] or ["| — | | | | |"]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def run(broker, cfg: dict, now: datetime, dry_run: bool, universe=None,
        days_to_earnings=earnings_guard.days_to_earnings, signal_fetcher=None,
        state_dir: str = STATE_DIR) -> dict:
    """One daily cycle. Dependencies are injectable so tests can run the
    whole cycle against a fake broker with no network."""
    os.makedirs(state_dir, exist_ok=True)
    state_path = os.path.join(state_dir, "state.json")
    state = load_state(state_path)
    today = now.date()
    trading = not dry_run
    mode = ("LIVE" if getattr(broker, "live", False) else "PAPER") + ("" if trading else " dry-run")
    api_url = os.environ.get("SIGNAL_API_URL")
    fetch_signal = signal_fetcher or (lambda s: ai_filter.fetch_signal(s, api_url))

    symbols, source, removed = universe or build_universe(exclude=cfg["exclude"], today=str(today))
    held = broker.positions()
    wanted = list(dict.fromkeys(symbols + list(state["positions"]) + [cfg["benchmark"]]))
    frames = {s: breakout.add_breakout_indicators(d, cfg) for s, d in broker.bars(wanted).items() if len(d) > 60}
    acted: list[str] = []

    # 1) trades closed at the broker (stop hit / manual) since last run
    for sym in list(state["positions"]):
        if sym not in held:
            p = state["positions"].pop(sym)
            state["closed"].append({**p, "symbol": sym, "exit": "stop hit or closed manually", "closed_on": str(today)})
            acted.append(f"{sym}: position closed (stop hit or closed manually)")

    # 2) manage open trades
    for sym, pos in list(state["positions"].items()):
        if sym not in frames:
            continue
        qty = held[sym][0]
        dte = days_to_earnings(sym, today)
        actions, newpos = breakout.manage(pos, frames[sym], qty, cfg, dte, cfg["earnings_window"])
        if not trading:
            acted += [f"{sym}: would {a[0]} {a[1]} ({a[2]})" for a in actions]
            continue
        try:
            if actions:
                broker.cancel(sym)
            for side, q, why in actions:
                px = broker.market(sym, side, q)
                acted.append(f"{sym}: SELL {q} @ {px:.2f} — {why}")
            remaining = broker.positions().get(sym, (0, 0.0))[0]
            if remaining == 0:
                state["closed"].append({**newpos, "symbol": sym, "exit": actions[-1][2] if actions else "",
                                        "closed_on": str(today)})
                del state["positions"][sym]
            else:
                if broker.set_stop(sym, remaining, newpos["stop"]) and newpos["stop"] != pos["stop"]:
                    acted.append(f"{sym}: stop raised to {newpos['stop']:.2f}")
                state["positions"][sym] = newpos
        except Exception as exc:
            acted.append(f"{sym}: ERROR managing position: {exc}")

    # 3) scan
    breakouts, near = breakout.scan(frames, symbols, cfg)
    regime_ok = breakout.market_filter_ok(frames.get(cfg["benchmark"]))

    # 4) entries
    acct = broker.account()
    equity, cash = acct["equity"], acct["cash"]
    daily_ok = acct["equity"] - acct["last_equity"] > -equity * cfg["max_daily_loss_pct"]
    slots = cfg["max_positions"] - len(broker.positions() if trading else held)
    if breakouts and not regime_ok:
        acted.append(f"Market filter off ({cfg['benchmark']} below 50-day avg): no new buys")
    elif breakouts and not daily_ok:
        acted.append("Daily loss limit hit: no new buys")
    elif breakouts:
        for sym, info in breakouts:
            if slots <= 0:
                acted.append("Max positions reached")
                break
            if sym in held or sym in state["positions"]:
                continue
            dte = days_to_earnings(sym, today)
            if earnings_guard.blocks_entry(dte, cfg["earnings_window"], cfg["block_if_earnings_unknown"]):
                acted.append(f"{sym}: breakout skipped — earnings {'unknown' if dte is None else f'in ~{dte} days'}")
                continue
            ok, ai_note = ai_filter.decide(
                fetch_signal(sym) if cfg["ai_filter_mode"] != "off" else None, cfg["ai_filter_mode"])
            if not ok:
                acted.append(f"{sym}: breakout skipped — {ai_note}")
                continue
            q = breakout.position_size(info["close"], info["stop"], equity, cash, cfg)
            if q < 1:
                acted.append(f"{sym}: breakout but size < 1 share, skipped")
                continue
            if not trading:
                acted.append(f"{sym}: would BUY {q} near {info['close']:.2f}, stop {info['stop']:.2f} [{ai_note}]")
                slots -= 1
                continue
            try:
                px = broker.market(sym, "buy", q)
                stop = min(info["stop"], round(px * 0.99, 2))
                broker.set_stop(sym, q, stop)
                state["positions"][sym] = {
                    "entry": px, "stop": stop, "risk": round(px - stop, 4),
                    "breakout_level": info["breakout_level"], "date": str(today),
                    "high_close": px, "partial": False, "earnings_cut": False}
                acted.append(f"{sym}: BUY {q} @ {px:.2f}, stop {stop:.2f} "
                             f"(volume {info['vol_ratio']:.1f}×, base {info['base_depth']*100:.0f}%) [{ai_note}]")
                cash -= q * px
                slots -= 1
            except Exception as exc:
                acted.append(f"{sym}: ERROR buying: {exc}")

    if trading:
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)
    write_report(os.path.join(state_dir, "report.md"),
                 f"Shariah breakout scan — {now:%Y-%m-%d %H:%M} New York ({mode})",
                 source, removed, regime_ok, breakouts, near[:25], state, acted)
    notify(f"[{mode}] Shariah scan: {len(breakouts)} breakouts, {len(near)} near, "
           f"{len(state['positions'])} open.\n" + ("\n".join(acted) or "No actions.")
           + ("\nWatchlist: " + ", ".join(s for s, _ in near[:10]) if near else ""))
    return {"acted": acted, "breakouts": breakouts, "near": near, "state": state, "regime_ok": regime_ok}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="scan and report, place no orders")
    ap.add_argument("--force", action="store_true", help="trade even if the market is closed")
    args = ap.parse_args()
    os.makedirs(STATE_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(os.path.join(STATE_DIR, "bot.log")), logging.StreamHandler()])

    if os.path.exists(os.path.join(ROOT, "STOP")):
        notify("Kill switch file STOP found. Trading bot did nothing.")
        return
    from execution.broker_alpaca import AlpacaBroker
    broker = AlpacaBroker()
    if not args.dry_run and not args.force and not broker.market_open():
        logger.info("Market closed. Run during market hours, or use --dry-run.")
        return
    run(broker, load_config(), datetime.now(NY), args.dry_run)


if __name__ == "__main__":
    main()
