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
import os
import sys
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

import channel_scorecard as cs
import price_cache
import signals_data as mp
import ticker_resolver
from helpers import env_int
from log import log_info, log_warn, log_error
from signals_data import (
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
# How many assets one --resolve run will work through. Resolution is paced
# (several catalogue lookups per asset at 6/minute), so an unbounded pass over
# the backlog runs for hours and dies on the workflow timeout having saved
# nothing. Bounded runs chip away at it instead, most-discussed first, and
# results are cached so no asset is paid for twice.
MAX_RESOLVE_PER_RUN = env_int("MAX_RESOLVE_PER_RUN", 40)


def resolvable_assets(records, cache, learned):
    """
    Assets worth spending a lookup on, most-mentioned first.

    Only two things make an asset scoreable: a directional call (which the
    scorecard grades) or a price target (which the upside chart measures). A
    neutral name-drop of a company nobody made a call on buys nothing, and the
    dataset is full of them — 324 assets in all, against 166 with any stake and
    far fewer that matter. Anything already priced, already looked up, or of a
    type with no tradable symbol is skipped.
    """
    mentions, first_seen = {}, {}
    for _, asset in mp._iter_assets(records):
        name = (asset.get("name") or "").strip()
        recorded = (asset.get("ticker") or "").strip().upper()
        key = recorded or name.upper()
        if not name or key in learned:
            continue
        if asset.get("type") not in ("stock", "etf", "crypto"):
            continue
        # A hand-curated "no US listing" decision outranks anything the
        # resolver might find. Without this, OpenAI — deliberately marked
        # unpriceable — would resolve to a pre-IPO tracking instrument the
        # catalogue happens to carry, and the learned entry would then
        # override the curated one downstream.
        if (recorded in mp.UNPRICEABLE_TICKERS
                or name.upper() in mp.UNPRICEABLE_TICKERS):
            continue
        target = asset.get("price_target")
        stake = (asset.get("stance") in ("bullish", "bearish")
                 or (isinstance(target, (int, float)) and not isinstance(target, bool)))
        if not stake:
            continue
        symbol = cs.symbol_for(asset)
        if symbol and symbol in cache:
            continue
        mentions[key] = mentions.get(key, 0) + 1
        first_seen.setdefault(key, (name, recorded))
    ranked = sorted(mentions, key=lambda k: (-mentions[k], k))
    return [(key, *first_seen[key]) for key in ranked]


def needed_ranges(records, today, claims=None):
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

    # The canonical scorecard and pulse: every testable headline forecast's
    # instrument from the day before publication through its evaluation
    # window, its resolved benchmark over the same range, and the current
    # window's price-target symbols.
    for symbol, start, end in canonical_ranges(claims if claims is not None else _load_claims(), today):
        want(symbol, start, end)

    return ranges


def _load_claims():
    import canonical_claims
    try:
        return canonical_claims.load_canonical_claims()
    except Exception as e:  # warming must never fail on a research-ledger problem
        log_warn(f"Canonical claims unavailable for warming: {e}")
        return []


def canonical_ranges(claims, today):
    """[(symbol, start, end)] the canonical analytics will look up."""
    import canonical_claims
    import scorecard_pricing as sp
    from datetime import date as _date
    out = []
    import instruments
    for c in canonical_claims.headline_claims(claims):
        inst = instruments.resolve_instrument(c.get("ticker"), c.get("canonical_entity_name") or c.get("subject_mention"),
                                              c.get("asset_type"))
        symbol = inst.symbol if inst else canonical_claims.priceable_symbol(c)
        if not symbol:
            continue
        pub = canonical_claims.claim_date(c)
        if c.get("testable") and c.get("forecast_end_date") and pub:
            try:
                end = _date.fromisoformat(c["forecast_end_date"])
            except ValueError:
                continue
            start, stop = pub - timedelta(days=1), end + timedelta(days=cs.MAX_PRICE_LAG_DAYS)
            out.append((symbol, start, stop))
            bench, _ = instruments.benchmark_for(inst, sp.exchange_for_instrument(inst), symbol)
            if bench:
                out.append((bench, start, stop))
    window_start = today - timedelta(days=PULSE_WINDOW_DAYS)
    current = [c for c in claims if canonical_claims.in_window(c, window_start, today)]
    for entry in canonical_claims.aggregate_views(current).values():
        symbol = canonical_claims.priceable_symbol(entry)
        if symbol and entry["targets"]:
            out.append((symbol, today - timedelta(days=LATEST_PRICE_LOOKBACK_DAYS), today))
    return out


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
    parser.add_argument("--ticker-map", default=ticker_resolver.MAP_FILE,
                        dest="ticker_map")
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
        records = load_signals(args.signals)
        if not records:
            log_info("No signal records yet; nothing to resolve.")
            return 0
        learned = ticker_resolver.load(args.ticker_map)
        # What counts as "needs resolving" is a symbol the warm job could not
        # actually price — not merely one we failed to build. A ticker like
        # APPLE forms the symbol apple.us perfectly well and still 404s, so
        # the cache (which only holds symbols that really returned closes) is
        # the honest signal here.
        cache = price_cache.load(args.cache)
        if not cache:
            log_warn("Price cache is empty, so every asset looks unresolved; "
                     "run a warm first to make this selective.")
        backlog = resolvable_assets(records, cache, learned)
        if not backlog:
            log_info("Every scoreable asset already resolves to a priceable ticker.")
            return 0
        batch = backlog[:MAX_RESOLVE_PER_RUN]
        log_info(f"Resolving {len(batch)} of {len(backlog)} unpriced asset(s) "
                 f"with a call or a target, most-discussed first.")
        saved = 0
        for key_name, name, recorded in batch:
            entry = ticker_resolver.resolve(name, recorded, key)
            if entry["via"] == "unchecked":
                # The catalogue was unreachable, so this is not a verdict.
                log_warn(f"  {name[:34]:36} could not be checked; leaving for next run.")
                break
            learned[key_name] = entry
            saved += 1
            status = entry["ticker"] or "no US listing"
            log_info(f"  {name[:34]:36} {recorded or '-':10} -> {status} ({entry['via']})")
        if saved:
            ticker_resolver.save(learned, args.ticker_map)
        remaining = len(backlog) - saved
        if remaining > 0:
            log_info(f"{remaining} asset(s) left for the next run.")
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
