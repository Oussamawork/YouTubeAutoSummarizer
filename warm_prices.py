"""Weekly price-cache warmer (runs Sundays from warm-prices.yml).

The price provider's free tier allows 8 requests/minute, and the signals
dataset spans hundreds of distinct symbols — too slow to do inline in the
Monday pulse or the Friday scorecard without blowing their timeouts. This job
exists to be the slow one: it tops up `data/prices.json` ahead of both, so
they read history instead of fetching it.

Only the *uncovered* ranges cost anything. Closes are immutable, so a symbol
whose calls have all elapsed is fetched once and never again; a steady week
asks only for new tickers and for calls whose horizon is still open.
"""
import argparse
import sys
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

import channel_scorecard as cs
import market_pulse as mp
import price_cache
import ticker_resolver
from log import log_info, log_warn, log_error
from market_pulse import (
    SIGNALS_FILE, load_signals, aggregate_assets, _in_window,
)

load_dotenv('.env')

# How far past a call the scorecard ever looks (its longest horizon), plus a
# few days of slack so a horizon landing on a holiday still finds a close.
HORIZON_SLACK_DAYS = max(cs.HORIZONS) + cs.MAX_PRICE_LAG_DAYS
# The pulse annotates targets with the latest close; it looks back this far.
LATEST_PRICE_LOOKBACK_DAYS = 10
# The pulse's window, mirrored here so the warm job covers what it will ask for.
PULSE_WINDOW_DAYS = 7


def needed_ranges(records, today):
    """
    {symbol: (start, end)} covering every price lookup the weekly jobs will
    make: each directional call's symbol from its call date through its longest
    horizon, plus the current pulse window's price-target symbols.

    `end` is capped at today — future closes don't exist, and asking for them
    would leave the range permanently "uncovered".
    """
    ranges = {}

    def want(symbol, start, end):
        end = min(end, today)
        if start > end:
            return
        if symbol in ranges:
            known_start, known_end = ranges[symbol]
            ranges[symbol] = (min(known_start, start), max(known_end, end))
        else:
            ranges[symbol] = (start, end)

    for signal_date, _, _, symbol, _ in cs._directional_calls(records):
        want(symbol, signal_date, signal_date + timedelta(days=HORIZON_SLACK_DAYS))

    window_start = today - timedelta(days=PULSE_WINDOW_DAYS)
    current = [r for r in records if _in_window(r, window_start, today)]
    for entry in aggregate_assets(current).values():
        if not entry["targets"] or not entry.get("ticker"):
            continue
        symbol = cs.symbol_for({"ticker": entry["ticker"], "type": entry["type"]})
        if symbol:
            want(symbol, today - timedelta(days=LATEST_PRICE_LOOKBACK_DAYS), today)

    return ranges


def warm(records, today, cache, fetcher=None):
    """Fetch every range the cache doesn't already cover. Returns
    (fetched, skipped, failed) counts."""
    fetcher = fetcher or cs.fetch_prices_live
    ranges = needed_ranges(records, today)
    fetched = skipped = failed = 0
    for symbol, (start, end) in sorted(ranges.items()):
        entry = cache.get(symbol)
        if price_cache.covered(entry, start, end):
            skipped += 1
            continue
        # Widen to whatever is already known, so one request replaces the
        # union rather than leaving a hole between two disjoint ranges.
        if entry:
            start, end = min(start, entry["from"]), max(end, entry["to"])
        prices = fetcher(symbol, start, end)
        if prices:
            price_cache.remember(cache, symbol, start, end, prices)
            fetched += 1
        else:
            failed += 1
    return fetched, skipped, failed


def main():
    parser = argparse.ArgumentParser(description="Warm the daily-close cache")
    parser.add_argument("--signals", default=SIGNALS_FILE)
    parser.add_argument("--cache", default=price_cache.CACHE_FILE)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be fetched without calling the provider")
    parser.add_argument("--resolve", action="store_true",
                        help="Find assets whose ticker does not price, work out the real "
                             "one (provider search first, LLM suggestion verified against "
                             "the catalogue second) and save data/ticker_map.json.")
    parser.add_argument("--find", metavar="NAMES",
                        help="Look up company names in the provider's symbol search and "
                             "print the real ticker, exchange and currency for each. Use "
                             "this to correct a ticker with evidence instead of a guess.")
    parser.add_argument("--probe", metavar="SYMBOLS", nargs="?", const="nvda.us",
                        help="Fetch these comma-separated symbols live and report what "
                             "came back, without touching the cache. Answers 'is the "
                             "price provider working, and does this ticker resolve?' in "
                             "seconds instead of a full warm run.")
    args = parser.parse_args()

    if args.resolve:
        if not cs.twelvedata_key():
            log_error("No price API key configured (TWELVEDATA_API).")
            return 1
        key = cs.twelvedata_key()
        learned = ticker_resolver.load()
        # One entry per (name, recorded ticker) that currently prices as
        # nothing. Already-resolved keys are skipped, so this is cheap to
        # re-run and only pays for genuinely new names.
        unresolved = {}
        for _, asset in mp._iter_assets(records):
            name = (asset.get("name") or "").strip()
            recorded = (asset.get("ticker") or "").strip().upper()
            ticker = mp.canonical_ticker(asset)
            key_name = recorded or name.upper()
            if not name or key_name in learned:
                continue
            if ticker and cs.symbol_for(asset):
                continue  # already priceable
            unresolved.setdefault(key_name, (name, recorded))
        if not unresolved:
            log_info("Every asset already resolves to a priceable ticker.")
            return 0
        log_info(f"Resolving {len(unresolved)} unpriceable asset(s).")
        for key_name, (name, recorded) in sorted(unresolved.items()):
            entry = ticker_resolver.resolve(name, recorded, key)
            learned[key_name] = entry
            status = entry["ticker"] or "no US listing"
            log_info(f"  {name[:34]:36} {recorded or '-':10} -> {status} ({entry['via']})")
        ticker_resolver.save(learned)
        return 0

    if args.find:
        if not cs.twelvedata_key():
            log_error("No price API key configured (TWELVEDATA_API).")
            return 1
        key = cs.twelvedata_key()
        for name in [n.strip() for n in args.find.split(",") if n.strip()]:
            rows = cs.search_symbols(name, key)
            if not rows:
                log_warn(f"{name}: no listings found.")
                continue
            log_info(f"{name}:")
            for row in rows:
                log_info(f"    {row.get('symbol','?'):12} {row.get('exchange','?'):10} "
                         f"{row.get('mic_code','?'):6} {row.get('currency','?'):4} "
                         f"{row.get('country','?'):16} {row.get('instrument_name','?')[:44]}")
        return 0

    if args.probe:
        if not cs.twelvedata_key():
            log_error("No price API key configured (TWELVEDATA_API).")
            return 1
        today = datetime.now(timezone.utc).date()
        symbols = [s.strip() for s in args.probe.split(",") if s.strip()]
        failed = []
        for symbol in symbols:
            prices = cs.fetch_prices_live(symbol, today - timedelta(days=10), today)
            if prices:
                latest = max(prices)
                log_info(f"Probe OK: {symbol} -> {len(prices)} closes, "
                         f"latest {latest} = {prices[latest]}.")
            else:
                failed.append(symbol)
                log_warn(f"Probe FAILED: {symbol} returned no closes.")
        if failed:
            log_error(f"{len(failed)}/{len(symbols)} symbols unavailable: "
                      f"{', '.join(failed)}")
            return 1
        log_info(f"All {len(symbols)} probed symbols resolved.")
        return 0

    records = load_signals(args.signals)
    if not records:
        log_info("No signal records yet; nothing to warm.")
        return 0

    today = datetime.now(timezone.utc).date()
    cache = price_cache.reset(price_cache.load(args.cache))

    if args.dry_run:
        ranges = needed_ranges(records, today)
        missing = [s for s, (start, end) in sorted(ranges.items())
                   if not price_cache.covered(cache.get(s), start, end)]
        log_info(f"{len(ranges)} symbols needed, {len(missing)} uncovered: "
                 f"{', '.join(missing[:20])}{' …' if len(missing) > 20 else ''}")
        return 0

    if not cs.twelvedata_key():
        log_error("No price API key configured (TWELVEDATA_API); cannot warm the cache.")
        return 1

    fetched, skipped, failed = warm(records, today, cache)
    log_info(f"Price cache warm: {fetched} fetched, {skipped} already covered, "
             f"{failed} unavailable.")
    if fetched and not price_cache.save(cache, args.cache, today=today):
        return 1
    if failed and not fetched:
        log_warn("No symbol could be priced; leaving the cache untouched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
