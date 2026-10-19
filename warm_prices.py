"""Price-cache warmer (Sundays and Fridays at 06:00 UTC, from warm-prices.yml).

The price provider's free tier allows 8 requests/minute, and the dataset spans
hundreds of distinct symbols — too slow to do inline in the Monday pulse or the
Friday report without blowing their timeouts. This job exists to be the slow
one: it tops up `data/prices.json` ahead of both, so they read history instead
of fetching it. The Sunday run serves the Monday pulse (Friday's close), the
Friday run the 16:00 UTC Friday report (Thursday's close). The
creators-vs-market tracker in both reads the cache only
(creator_tracker.cache_fetcher), so every range it needs
(creator_tracker.needed_price_ranges) is warmed here.

Only the *uncovered* ranges cost anything. Closes are immutable, so a symbol
whose calls have all elapsed is fetched once and never again; a steady week
asks only for new tickers and for calls still open (the tracker's planner
ends a closed call's range shortly after its closing close). No request
reaches past the last completed session (last_complete_day): the warm day's
own bar is still printing, and a Sunday warm once stored btcusd's in-progress
candle as a close. An open range is refetched up to that day on the next run,
which also replaces any partial bar an earlier warm stored. Cache entries are
never deleted here (ticker_audit lists orphans).

A run is bounded: requests go out in priority order (warm_order: the market
context and benchmarks first, then the tracker's symbols, stalest first, then
the rest), none starts after WARM_DEADLINE_MINUTES, and the cache is saved
every SAVE_EVERY fetched symbols — so a run that hits the workflow timeout
still keeps what it fetched, and the next run picks up the stalest symbols
first.
"""
import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone

from dotenv import load_dotenv

import channel_scorecard as cs
import creator_tracker
import price_cache
import scorecard_pricing as sp
import signals_data as mp
import ticker_resolver
from helpers import env_int
from log import describe_error, log_info, log_warn, log_error
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
# No new request starts after this many minutes, so the cache is saved and
# the workflow's Persist step runs inside its 90-minute timeout. At the
# provider pace (channel_scorecard.TWELVEDATA_PER_MINUTE, 6) that is ~450
# requests; whatever is left goes first next run (warm_order).
WARM_DEADLINE_MINUTES = env_int("WARM_DEADLINE_MINUTES", 75)
# The cache is written after every this many fetched symbols (atomically).
SAVE_EVERY = 50


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
    for record, asset in mp._iter_assets(records):
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
        symbol = cs.symbol_for(asset, record.get("channel_name"))
        if symbol and symbol in cache:
            continue
        mentions[key] = mentions.get(key, 0) + 1
        first_seen.setdefault(key, (name, recorded))
    ranked = sorted(mentions, key=lambda k: (-mentions[k], k))
    return [(key, *first_seen[key]) for key in ranked]


def needed_ranges(records, today, claims=None):
    """
    {symbol: (start, end)} covering every price lookup the weekly jobs will
    make:
      - legacy: each directional call's symbol from its call date through its
        longest horizon, plus the legacy pulse window's price-target symbols;
      - canonical (canonical_ranges): every testable headline forecast and
        its benchmark, plus the pulse's price-target symbols;
      - the creators-vs-market tracker (tracker_ranges): every call's symbol
        from the day before its first call, the S&P 500 and Bitcoin
        benchmarks, and the market context (SPY, QQQ, BTC, ETH), through the
        last completed session.
    `claims` are the canonical claims, loaded once here when not given.

    `end` is capped at today — future closes don't exist, and asking for them
    would leave the range permanently "uncovered". warm() caps each request
    further, at the last completed session (last_complete_day).
    """
    claims = _load_claims() if claims is None else claims
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
    for symbol, start, end in canonical_ranges(claims, today):
        want(symbol, start, end)

    # The creators-vs-market tracker (Friday report, Monday pulse): it reads
    # the cache only, so whatever it will ask for has to be here.
    for symbol, start, end in tracker_ranges(claims, today):
        want(symbol, start, end)

    return ranges


def _load_claims():
    import canonical_claims
    try:
        return canonical_claims.load_canonical_claims()
    except Exception as e:  # warming must never fail on a research-ledger problem
        log_warn(f"Canonical claims unavailable for warming: {e}")
        return []


def tracker_ranges(claims, today):
    """
    [(symbol, start, end)] the creators-vs-market tracker reads for a report
    on `today`: creator_tracker.needed_price_ranges — every call's symbol
    from the day before its first call, the S&P 500 and Bitcoin benchmarks,
    and the market context (spy.us, qqq.us, btcusd, ethusd), each through
    creator_tracker.default_as_of(today). None without claims: no calls means
    no report to price, not even its market context.
    """
    if not claims:
        return []
    try:
        return list(creator_tracker.needed_price_ranges(claims, today))
    except Exception as e:  # noqa: BLE001 - warming must never fail on a research-ledger problem
        log_warn(f"Creators-vs-market ranges unavailable for warming: {describe_error(e)}")
        return []


def canonical_ranges(claims, today):
    """
    [(symbol, start, end)] the canonical analytics will look up:
      - the strict scorecard: every testable headline forecast's instrument
        (instruments.resolve_instrument, else canonical_claims.priceable_symbol)
        from the day before publication through its evaluation window, and
        its resolved benchmark over the same range;
      - the pulse's price targets: canonical_claims.aggregate_views_by_asset
        (the pulse's own aggregation, one vote per creator) over the pulse
        window, priced as the pulse prices them
        (canonical_claims.priceable_symbol, dollar quotes only) over
        [as_of - LATEST_PRICE_LOOKBACK_DAYS, as_of], where as_of =
        creator_tracker.default_as_of(today): the close the pulse annotates
        targets with.
    """
    import canonical_claims
    import instruments
    from datetime import date as _date
    out = []
    for c in canonical_claims.headline_claims(claims):
        pub = canonical_claims.claim_date(c)
        if not (c.get("testable") and c.get("forecast_end_date") and pub):
            continue
        inst = instruments.resolve_instrument(c.get("ticker"), c.get("canonical_entity_name") or c.get("subject_mention"),
                                              c.get("asset_type"))
        symbol = inst.symbol if inst else canonical_claims.priceable_symbol(c)
        if not symbol:
            continue
        try:
            end = _date.fromisoformat(c["forecast_end_date"])
        except ValueError:
            continue
        start, stop = pub - timedelta(days=1), end + timedelta(days=cs.MAX_PRICE_LAG_DAYS)
        out.append((symbol, start, stop))
        bench, _ = instruments.benchmark_for(inst, sp.exchange_for_instrument(inst), symbol)
        if bench:
            out.append((bench, start, stop))
    as_of = creator_tracker.default_as_of(today)
    window_start = today - timedelta(days=PULSE_WINDOW_DAYS)
    current = [c for c in claims if canonical_claims.in_window(c, window_start, today)]
    for entry in canonical_claims.aggregate_views_by_asset(current).values():
        if not entry["targets"]:
            continue
        symbol = canonical_claims.priceable_symbol(entry)
        if symbol and cs.quotes_in_usd(symbol):
            out.append((symbol, as_of - timedelta(days=LATEST_PRICE_LOOKBACK_DAYS), as_of))
    return out


def last_complete_day(symbol, today):
    """
    The newest day whose close has finished printing when the warm job runs
    on `today` (UTC): yesterday for a 24/7 crypto pair (an internal symbol
    with no venue suffix, ending in "usd": btcusd), the last completed NYSE
    session (creator_tracker.default_as_of) for everything else. A later bar
    is still in progress, and cached it would read as history forever.
    """
    symbol = (symbol or "").strip().lower()
    if "." not in symbol and symbol.endswith("usd"):
        return today - timedelta(days=1)
    return creator_tracker.default_as_of(today)


def _request_for(symbol, start, end, entry, today):
    """
    (start, end) to fetch for a needed range, or None when the cache already
    holds it through the symbol's last completed session (or no session in it
    has completed yet). The request is widened to its union with the cached
    range, so one request replaces both rather than leaving a hole between
    them, and then capped at last_complete_day: an open range is refetched up
    to that day, which also replaces a partial last bar an earlier warm
    stored.
    """
    last = last_complete_day(symbol, today)
    if price_cache.covered(entry, start, min(end, last), sp.adjustment_label()):
        return None
    if entry:
        start, end = min(start, entry["from"]), max(end, entry["to"])
    end = min(end, last)
    return (start, end) if start <= end else None


def _through(prices, end):
    """The fetched bars dated on or before `end` — (closes, highs, lows) — so
    a provider answering past the requested end cannot slip an in-progress
    bar into the cache. Takes a {date: close} map (channel_scorecard.Bars
    carries highs/lows beside it) or a scorecard_pricing.PriceSeries."""
    def cut(values):
        return {d: v for d, v in values.items() if d <= end} if isinstance(values, dict) else {}
    closes = getattr(prices, "closes", prices)
    return cut(closes), cut(getattr(prices, "highs", None)), cut(getattr(prices, "lows", None))


def warm_order(ranges, cache, tracker_symbols=()):
    """
    [(symbol, (start, end))] in the order a warm run requests them: the
    market context and the benchmarks first (the tracker's as-of is the last
    S&P 500 close), then the creators-vs-market symbols, then everything
    else; within each tier the stalest first (never cached, then the oldest
    cached end), so a run cut short by its deadline leaves the freshest
    symbols for the next one and every symbol is reached in turn.
    """
    anchors = [creator_tracker.MARKET_SYMBOL, creator_tracker.CRYPTO_MARKET_SYMBOL] + [
        s for s, _ in creator_tracker.CONTEXT_SYMBOLS]
    rank = {}
    for i, symbol in enumerate(anchors):
        rank.setdefault(symbol, i)
    tracker_symbols = set(tracker_symbols)

    def key(item):
        symbol = item[0]
        if symbol in rank:               # spy.us first: it sets every report's as-of
            return 0, rank[symbol], date.min, symbol
        entry = cache.get(symbol)
        return 1 if symbol in tracker_symbols else 2, 0, entry["to"] if entry else date.min, symbol

    return sorted(ranges.items(), key=key)


def warm(records, today, cache, fetcher=None, claims=None, save=None, deadline_minutes=None, clock=None):
    """Fetch every needed range the cache doesn't already hold through the
    last completed session, in warm_order, starting no request after
    `deadline_minutes` (WARM_DEADLINE_MINUTES) and calling `save(cache)`
    after every SAVE_EVERY fetched symbols. Returns (fetched, skipped,
    failed) counts; skipped means already covered, or nothing complete to
    fetch yet. Ranges left by the deadline are logged, not counted."""
    fetcher = fetcher or cs.fetch_prices_live
    clock = clock or time.monotonic
    limit = 60 * (WARM_DEADLINE_MINUTES if deadline_minutes is None else deadline_minutes)
    adjustment = sp.adjustment_label()
    claims = _load_claims() if claims is None else claims
    ranges = needed_ranges(records, today, claims)
    tracker = {symbol for symbol, _, _ in tracker_ranges(claims, today)}
    fetched = skipped = failed = 0
    deferred, started = [], clock()
    for symbol, (start, end) in warm_order(ranges, cache, tracker):
        request = _request_for(symbol, start, end, cache.get(symbol), today)
        if request is None:
            skipped += 1
            continue
        if clock() - started >= limit:
            deferred.append(symbol)
            continue
        start, end = request
        closes, highs, lows = _through(fetcher(symbol, start, end) or {}, end)
        if closes:
            price_cache.remember(cache, symbol, start, end, closes, adjustment, highs=highs, lows=lows)
            fetched += 1
            if save and fetched % SAVE_EVERY == 0:
                save(cache)
        else:
            failed += 1
    if deferred:
        log_warn(f"Warm deadline ({limit // 60} minutes) reached: {len(deferred)} symbol(s) left for the next "
                 f"run, stalest first ({_listing(deferred, 10)}).")
    return fetched, skipped, failed


def _listing(symbols, limit=None):
    shown = symbols if limit is None else symbols[:limit]
    return ", ".join(shown) + (" …" if limit is not None and len(symbols) > limit else "")


def dry_run_report(records, claims, today, cache):
    """
    Log what a warm run on `today` would fetch, without calling the provider:
    the needed symbols, those with something to fetch (open or uncovered
    ranges, capped at the last completed session), and how many of the needed
    symbols the creators-vs-market tracker accounts for — listed in full,
    with the ones never cached. Returns the same as a dict of sorted lists:
    needed, to_fetch, tracker, tracker_to_fetch, never_cached.
    """
    ranges = needed_ranges(records, today, claims)
    to_fetch = [s for s, (start, end) in sorted(ranges.items())
                if _request_for(s, start, end, cache.get(s), today) is not None]
    tracker = sorted({symbol for symbol, _, _ in tracker_ranges(claims, today)})
    pending = set(to_fetch)
    tracker_to_fetch = [s for s in tracker if s in pending]
    never_cached = [s for s in tracker if s not in cache]
    context = [s for s, _ in creator_tracker.CONTEXT_SYMBOLS if s in tracker]
    minutes = len(to_fetch) / max(1, cs.TWELVEDATA_PER_MINUTE)
    log_info(f"{len(ranges)} symbols needed, {len(to_fetch)} to fetch (through the last completed session: "
             f"{creator_tracker.default_as_of(today)} for listings, {today - timedelta(days=1)} for crypto): "
             f"{_listing(to_fetch, 20)}")
    log_info(f"Projected: about {minutes:.0f} minutes at {cs.TWELVEDATA_PER_MINUTE}/minute against a "
             f"{WARM_DEADLINE_MINUTES}-minute deadline"
             + ("." if minutes <= WARM_DEADLINE_MINUTES else
                " — the stalest symbols go first and the rest wait for the next run."))
    if tracker:
        log_info(f"Creators-vs-market tracker: {len(tracker)} of the needed symbols (every call symbol, the S&P 500 "
                 f"and Bitcoin benchmarks, market context {', '.join(context)}); {len(tracker_to_fetch)} to fetch, "
                 f"{len(never_cached)} never cached{': ' + _listing(never_cached) if never_cached else ''}.")
        log_info(f"Tracker symbols: {_listing(tracker)}")
    else:
        log_info("Creators-vs-market tracker: no calls, nothing to warm for it.")
    return {"needed": sorted(ranges), "to_fetch": to_fetch, "tracker": tracker,
            "tracker_to_fetch": tracker_to_fetch, "never_cached": never_cached}


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
    claims = _load_claims()
    if not records and not claims:
        log_info("No signal records or canonical claims yet; nothing to warm.")
        return 0

    today = datetime.now(timezone.utc).date()
    cache = price_cache.reset(price_cache.load(args.cache))

    if args.dry_run:
        dry_run_report(records, claims, today, cache)
        return 0

    if not cs.twelvedata_key():
        log_error("No price API key configured (TWELVEDATA_API); cannot warm the cache.")
        return 1

    # Never trim history a needed range still starts at: a long-horizon
    # forecast's entry close stays cached until it matures.
    keep_from = {symbol: start for symbol, (start, _) in needed_ranges(records, today, claims).items()}

    def save(current):
        return price_cache.save(current, args.cache, today=today, keep_from=keep_from)

    fetched, skipped, failed = warm(records, today, cache, claims=claims, save=save)
    log_info(f"Price cache warm: {fetched} fetched, {skipped} already covered (or no session "
             f"completed yet), {failed} unavailable.")
    if fetched and not save(cache):
        return 1
    if failed and not fetched:
        log_warn("No symbol could be priced; leaving the cache untouched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
