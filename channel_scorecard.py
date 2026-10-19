import argparse
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

import instruments
import price_cache
import scorecard_pricing as sp
from helpers import env_int
from log import describe_error, log_info, log_warn, log_error
from sendToTelegram import send_telegram_photo_album, send_telegram_text
from signals_data import (
    SIGNALS_FILE, load_signals, _parse_date, _iter_assets, canonical_ticker,
    UNPRICEABLE_TICKERS,
)

# The Friday report (weekly-scorecard.yml, Fridays 16:00 UTC).
#
# PRODUCTION PATH (main -> generate_weekly_report), in reading order:
#   1. Creators vs the market (creator_tracker): every creator's calls since
#      they were made, marked to the last completed session and held against
#      the S&P 500 (crypto: Bitcoin) over the same days. Labelled
#      "mark-to-market, not a verdict", never ranked. It reads the committed
#      price cache only (creator_tracker.cache_fetcher) and never spends a
#      live request: warm_prices covers creator_tracker.needed_price_ranges,
#      Sundays and Fridays at 06:00 UTC.
#   2. Its two charts (pulse_charts.render_tracker_charts), best-effort.
#   3. The strict scorecard (generate_canonical_scorecard): canonical claims
#      whose dated forecast has matured, scored by research_analytics.scorecard
#      under scorecard_pricing (exchange-aware next-close entry, split-adjusted
#      prices with recorded provenance, per-claim benchmarks or an honest
#      null), plus the conditional calls of the last CONDITIONAL_WINDOW_DAYS.
#      Below the sample minimum it is compact: one plain line per scored
#      forecast, when the next ones mature, and a one-line conditional count.
#      The verdict layer; it stays EXPERIMENTAL and unranked until
#      SCORECARD_RANKINGS=true. Its matured forecasts are the only live price
#      requests the job makes (cache first, live for gaps,
#      TWELVEDATA_MAX_REQUESTS per run).
#
# LEGACY PATH (evaluate / build_scorecard / generate_scorecard, --legacy):
# the original fixed 7/30-day directional hit rate over data/signals.jsonl —
# kept as a compatibility consumer of the legacy view and for
# market_pulse's track-record weighting, never as the headline scorecard.
# Hit rates over tiny samples are noise: the report always shows sample sizes,
# and remains research input, not investment advice.

load_dotenv('.env')

REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
RETRY_BACKOFF = 2

# Twelve Data is the price source; TWELVEDATA_API must be set for any price
# lookup to work. It is built for server-side use, and its free tier (800
# requests/day, 8 per minute) comfortably covers a weekly pulse and scorecard.
# The keyless predecessor, Stooq, sat behind a JavaScript browser check from
# 2026-08-17 and silently returned nothing from the first scheduled run
# (2026-07-27) until Twelve Data replaced it; it has been removed rather than
# kept as a fallback that only ever produced an empty result.
TWELVEDATA_URL = "https://api.twelvedata.com/time_series"
TWELVEDATA_SEARCH_URL = "https://api.twelvedata.com/symbol_search"
# The plan allows 8 credits/minute; we pace below it on purpose. Measured
# 2026-08-17: pacing at exactly 8 tripped 429s, because the limit is enforced
# on the provider's rolling minute across *all* our runs while the pacer below
# only knows about this process — back-to-back runs start out already spent.
TWELVEDATA_PER_MINUTE = 6
# A 429 means "this minute is spent", so the only useful wait is the rest of
# the window. Short exponential backoffs just burned paced slots and collapsed
# throughput to ~1.5 symbols/minute in the same measurement.
TWELVEDATA_RATE_LIMIT_COOLDOWN = 62
# A per-run ceiling on price requests. At 8/minute an unbounded run is a
# wall-clock problem, not a quota one: the dataset already spans ~180 distinct
# symbols, which would pace out to ~27 minutes and blow the workflow timeout.
# The cap keeps a run bounded (120 -> ~15 minutes) and degrades the way the
# rest of this module does — callers that get {} just skip those assets.
# Callers spend it in priority order, so the visible features are funded first.
TWELVEDATA_MAX_REQUESTS = env_int("TWELVEDATA_MAX_REQUESTS", 120)
# A signal dated on a weekend/holiday uses the next trading day's close, up to
# this many days later; beyond that the price point is treated as missing.
MAX_PRICE_LAG_DAYS = 5
HORIZONS = (7, 30)
# The scorecard waits until the dataset spans at least this many days.
MIN_DATASET_AGE_DAYS = 7

DISCLAIMER = (
    "Mark-to-market since each call, and a strict scorecard of matured forecasts. "
    "Small samples are noise. Research input, not investment advice."
)
# The legacy --legacy report's own footer: it measures something else (a
# fixed 7/30-day directional hit rate), so it keeps the words that say so.
LEGACY_DISCLAIMER = (
    "⚠️ Directional hit rate vs price after the call. Small samples are noise. "
    "Research input, not investment advice."
)

# The strict block's conditional-forecast summary covers claims published in
# this many trailing days. All-time it listed 445 rows nobody could act on.
CONDITIONAL_WINDOW_DAYS = env_int("CONDITIONAL_WINDOW_DAYS", 30)
STRICT_HEADER = "🎯 Strict scorecard — matured, dated forecasts only"
DATA_LINE = "Data: canonical claims (data/research/claims.jsonl, active runs only)"
# Below the scorecard's sample minimum the strict block is compact: one plain
# line per scored forecast, when the next ones mature, and a one-line
# conditional count (the research report keeps the full list).
COMPACT_DATA_LINE = ("The creators' own dated price forecasts matured by the {close} close, from the canonical "
                     "claims (next-close entry, split-adjusted prices).")
_BENCHMARK_NAMES = {"spy.us": "S&P 500", "btcusd": "Bitcoin"}
# Where --dry-run (and the send path) writes the tracker chart PNGs; charts/
# is gitignored.
CHARTS_DIR = "charts/scorecard"
CHART_ALBUM_CAPTION = ("Creators vs the market in charts — how to read each one is written on "
                       "the image.")

# The shape of a US listing's ticker: letters, optionally a share class
# (BRK.B, MOG.A). Typed "stock", anything else — a KRX code (000660), a
# Taiwan or German listing (2454.TW, SAP.DE) — would become a ".us" symbol
# that 404s on every warm run, so the type rule does not price it.
_US_TICKER = re.compile(r"[A-Z]{1,6}(?:[.-][A-C])?")
_COIN_TICKER = re.compile(r"[A-Z0-9]{1,10}")


def symbol_for(asset, channel=None):
    """
    Map an asset entry to an internal symbol, or None when it isn't priceable.
    Uses the canonical ticker (recorded, else the curated alias table), so a
    call on "Chevron" scores even though the speaker never said "CVX" — a
    third of directional calls were ticker-less and invisible before this.
    Given the row's `channel`, that channel's own spelling comes first
    (claims.CHANNEL_ALIASES, as canonical_claims.resolved_ticker applies it
    to claims): Couch Investor's "New" is Nu Holdings, never new.us.

    The instrument registry decides first (instruments.resolve_instrument,
    given no claim type): the model-written type is the least reliable field
    a legacy row has. "Bitcoin" typed stock prices as btcusd — never btc.us,
    the Grayscale Bitcoin Mini Trust — and SpaceX or CoreWeave typed
    "other" as spcx.us / crwv.us. Only for a ticker that is both a coin and a
    US listing (canonical_claims.COLLIDING_TICKERS) does an explicit type
    choose the namespace, as in canonical_claims.priceable_symbol. An
    unregistered ticker falls back to the type rule: stock / etf ->
    "<t>.us" (US-shaped tickers only), crypto -> "<t>usd"; index, commodity
    and macro symbols vary too much per provider. UNPRICEABLE_TICKERS give
    None.
    """
    ticker = (_channel_ticker(asset, channel) or canonical_ticker(asset) or "").upper()
    if not ticker or ticker in UNPRICEABLE_TICKERS:
        return None
    asset_type = asset.get("type")
    inst = instruments.resolve_instrument(ticker, asset.get("name"))
    if inst is not None and not _type_overrides_registry(ticker, asset_type, inst):
        return inst.symbol
    if asset_type in ("stock", "etf"):
        return f"{ticker.lower()}.us" if _US_TICKER.fullmatch(ticker) else None
    if asset_type == "crypto":
        return f"{ticker.lower()}usd" if _COIN_TICKER.fullmatch(ticker) else None
    return None  # index/commodity/macro symbols vary too much per provider


def _channel_ticker(asset, channel):
    """The ticker `channel`'s own spelling of the asset's name stands for
    (claims.channel_ticker), or None — always None without a channel."""
    if not channel:
        return None
    import claims  # lazily, like the rest of this module's research imports
    return claims.channel_ticker(asset.get("name"), channel)


def _type_overrides_registry(ticker, asset_type, inst):
    """True for a COLLIDING_TICKERS ticker (a coin AND a US listing) whose
    explicit stock / etf / crypto type names the other namespace than the
    registry's row: Interlink typed stock is not Chainlink."""
    import canonical_claims  # lazily, like the rest of this module's research imports
    explicit = {"stock": "listing", "etf": "listing", "crypto": "coin"}.get(asset_type)
    kind = "coin" if inst.asset_type == "crypto" else "listing"
    return ticker in canonical_claims.COLLIDING_TICKERS and explicit is not None and explicit != kind


def twelvedata_key():
    """The Twelve Data key from the environment, or "" when unset. Accepts the
    repo's secret name (TWELVEDATA_API) and the conventional one."""
    return (os.getenv("TWELVEDATA_API") or os.getenv("TWELVEDATA_API_KEY") or "").strip()


# Internal symbols carry their trading venue as a suffix ("nvda.us",
# "000660.krx"), and each venue maps to the parameter the provider needs to
# resolve a bare ticker unambiguously. Without one, a ticker listed on several
# exchanges comes back as a 400 asking which is meant.
VENUES = {
    "us": {"country": "United States"},
    "sse": {"exchange": "SSE"},         # Shanghai, incl. the STAR board
    "szse": {"exchange": "SZSE"},       # Shenzhen
    "krx": {"exchange": "KRX"},         # Korea
    "xetra": {"exchange": "XETR"},      # Germany
    "ams": {"exchange": "Euronext"},    # Amsterdam
    "hkex": {"exchange": "HKEX"},       # Hong Kong
    "lse": {"exchange": "LSE"},         # London
    "tse": {"exchange": "TSE"},         # Tokyo
}

# What each venue quotes in. A foreign listing still scores fine on the
# scorecard, which only reads the direction of a move — but its close must
# never be compared against a price target the speaker gave in dollars, so
# the implied-upside annotation and the price-target chart take USD only.
VENUE_CURRENCY = {
    "us": "USD", "sse": "CNY", "szse": "CNY", "krx": "KRW",
    "xetra": "EUR", "ams": "EUR", "hkex": "HKD", "lse": "GBP", "tse": "JPY",
}


def quotes_in_usd(symbol):
    """True when the symbol's venue quotes in dollars (crypto pairs are
    explicitly /USD, so they qualify)."""
    symbol = (symbol or "").strip().lower()
    if "." not in symbol:
        return symbol.endswith("usd")
    return VENUE_CURRENCY.get(symbol.rsplit(".", 1)[-1]) == "USD"


def twelvedata_symbol(symbol):
    """
    Translate our internal symbol to Twelve Data's spelling: "nvda.us" ->
    "NVDA", "000660.krx" -> "000660", "btcusd" -> "BTC/USD". Keeping the
    internal symbol unchanged means symbol_for() and its callers stay
    provider-agnostic.
    """
    symbol = (symbol or "").strip().lower()
    if "." in symbol:
        base, suffix = symbol.rsplit(".", 1)
        if suffix in VENUES and base:
            return base.upper()
        return None
    if symbol.endswith("usd"):
        return f"{symbol[:-3].upper()}/USD" if symbol[:-3] else None
    return None


def venue_params(symbol):
    """The exchange/country parameter that pins an internal symbol's venue."""
    symbol = (symbol or "").strip().lower()
    suffix = symbol.rsplit(".", 1)[-1] if "." in symbol else ""
    return dict(VENUES.get(suffix, {}))


class SearchUnavailable(Exception):
    """The catalogue could not be consulted (rate-limited out). Distinct from
    'the catalogue has no such listing' — callers must not record a company as
    unlisted on the strength of a lookup that never happened."""


def search_symbols(query, api_key, limit=8):
    """
    Candidate listings for a company name, from the provider's own symbol
    search. This is how a wrong ticker gets corrected with evidence instead of
    a guess: it returns the real symbol, exchange, currency and instrument
    name. Returns [] on any failure (logged, never raises).

    Paced and retried like the price endpoint, and for a sharper reason: an
    empty result here reads downstream as "this company has no listing", so a
    rate-limited search would quietly record a real company as unlisted. A 429
    must never be mistaken for an answer.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        _twelvedata_pace()
        rows = _search_once(query, api_key, limit)
        if rows is not None:
            return rows
        if attempt < MAX_RETRIES:
            time.sleep(TWELVEDATA_RATE_LIMIT_COOLDOWN)
    raise SearchUnavailable(
        f"symbol search for {query!r} gave up after {MAX_RETRIES} rate-limited attempts")


def _search_once(query, api_key, limit):
    """One symbol-search call. Returns the rows, or None when the answer was
    'rate limited' — which the caller must retry rather than treat as empty."""
    try:
        resp = requests.get(TWELVEDATA_SEARCH_URL,
                            params={"symbol": query, "outputsize": limit,
                                    "apikey": api_key},
                            timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        log_warn(f"Twelve Data symbol search failed for {query!r}: {e}")
        return []
    if resp.status_code == 429:
        log_warn(f"Twelve Data symbol search rate-limited for {query!r}.")
        return None  # retryable: not an answer
    if resp.status_code != 200:
        log_warn(f"Twelve Data symbol search returned {resp.status_code} for {query!r}.")
        return []
    try:
        payload = resp.json()
    except ValueError:
        log_warn(f"Twelve Data symbol search sent a non-JSON body for {query!r}.")
        return []
    return [row for row in (payload.get("data") or []) if isinstance(row, dict)][:limit]


def _twelvedata_pace(now=None, _calls=[]):
    """
    Block until another request fits inside the free tier's per-minute budget.
    The plan allows TWELVEDATA_PER_MINUTE requests per rolling 60s (800/day),
    and the scorecard asks for dozens of symbols in one run, so pacing here is
    what keeps a run from turning into a wall of 429s. Sleeps only when the
    budget is actually spent.
    """
    now = now if now is not None else time.monotonic()
    cutoff = now - 60.0
    while _calls and _calls[0] <= cutoff:
        _calls.pop(0)
    if len(_calls) >= TWELVEDATA_PER_MINUTE:
        wait = _calls[0] + 60.0 - now
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
            cutoff = now - 60.0
            while _calls and _calls[0] <= cutoff:
                _calls.pop(0)
    _calls.append(now)


class Bars(dict):
    """A {date: close} dict that also carries the day's high and low
    (`.highs`, `.lows`, {date: value}) when the provider returned bars. Every
    existing caller keeps reading closes; the scorecard's price-target
    methods read the range (scorecard_pricing.evaluate_target)."""

    def __init__(self, closes=None, highs=None, lows=None):
        super().__init__(closes or {})
        self.highs = dict(highs or {})
        self.lows = dict(lows or {})


def _parse_twelvedata(payload, symbol):
    """Daily closes (a Bars dict, highs/lows beside them) from a Twelve Data
    time_series body. The API reports its own errors inside a 200 body
    (status="error"), so that is checked before the values are read."""
    if not isinstance(payload, dict):
        log_warn(f"Unexpected Twelve Data response for {symbol}.")
        return {}
    if payload.get("status") == "error":
        log_warn(f"Twelve Data error for {symbol}: "
                 f"{payload.get('code')} {payload.get('message', '')[:120]}")
        return {}
    prices, highs, lows = {}, {}, {}
    for row in payload.get("values") or []:
        day = _parse_date((row.get("datetime") or "")[:10])
        try:
            close = float(row.get("close"))
        except (TypeError, ValueError):
            continue
        if day:
            prices[day] = close
            try:
                highs[day], lows[day] = float(row.get("high")), float(row.get("low"))
            except (TypeError, ValueError):
                highs.pop(day, None), lows.pop(day, None)
    if not prices:
        log_warn(f"No usable price data from Twelve Data for {symbol}.")
        return {}
    return Bars(prices, highs, lows)


def _spend_request_budget(_state=[0]):
    """True while this run may still make a price request. Logs once at the
    cap so a truncated run is visible rather than looking like missing data."""
    if _state[0] >= TWELVEDATA_MAX_REQUESTS:
        if _state[0] == TWELVEDATA_MAX_REQUESTS:
            _state[0] += 1  # log the ceiling once, not once per skipped symbol
            log_warn(f"Price request budget spent ({TWELVEDATA_MAX_REQUESTS} this "
                     "run); remaining symbols go unpriced. Raise "
                     "TWELVEDATA_MAX_REQUESTS if the workflow has time for more.")
        return False
    _state[0] += 1
    return True


def fetch_prices_twelvedata(symbol, start, end, api_key):
    """Daily closes for `symbol` from Twelve Data as {date: close}. Returns {}
    on any failure (logged, never raises), like every other fetcher here."""
    td_symbol = twelvedata_symbol(symbol)
    if not td_symbol:
        return {}
    if not _spend_request_budget():
        return {}
    params = {
        "symbol": td_symbol, "interval": "1day",
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "order": "ASC", "format": "JSON", "outputsize": 5000, "apikey": api_key,
        # Split-adjusted at minimum ("splits", the provider default); "all"
        # folds dividends in for a total-return series. Recorded with the
        # closes so a cached series is never mistaken for the other kind.
        "adjust": sp.PRICE_ADJUSTMENT,
    }
    # A bare ticker listed on several exchanges is rejected with a 400 asking
    # for disambiguation — that is what NU (Nu Holdings), AMTM (Amentum) and
    # ECG (Everus) hit, all of which are perfectly real US listings. The
    # venue suffix carries the answer.
    params.update(venue_params(symbol))
    for attempt in range(1, MAX_RETRIES + 1):
        _twelvedata_pace()
        try:
            resp = requests.get(TWELVEDATA_URL, params=params, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            log_warn(f"Twelve Data request error for {td_symbol} "
                     f"(attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
                continue
            return {}
        # 429 means the provider's rolling minute is spent — wait it out rather
        # than retrying into the same closed window.
        if resp.status_code == 429:
            log_warn(f"Twelve Data rate limit hit for {td_symbol} "
                     f"(attempt {attempt}/{MAX_RETRIES}); waiting for the window.")
            if attempt < MAX_RETRIES:
                time.sleep(TWELVEDATA_RATE_LIMIT_COOLDOWN)
                continue
            return {}
        if resp.status_code != 200:
            log_warn(f"Twelve Data returned {resp.status_code} for {td_symbol}.")
            return {}
        try:
            payload = resp.json()
        except ValueError:
            log_warn(f"Twelve Data sent a non-JSON body for {td_symbol}.")
            return {}
        return _parse_twelvedata(payload, td_symbol)
    return {}


def fetch_prices_live(symbol, start, end, _warned=[]):
    """
    Daily closes straight from the provider. Returns {} on any failure
    (logged, never raises). Without a Twelve Data key there is no provider:
    that is said once per run, and market_pulse.fetch_latest_prices then
    reports the resulting "no prices for any ticker" in one loud line.
    """
    api_key = twelvedata_key()
    if api_key:
        return fetch_prices_twelvedata(symbol, start, end, api_key)
    if not _warned:
        _warned.append(True)
        log_warn("No price API key configured (TWELVEDATA_API); price lookups are off.")
    return {}


def fetch_prices(symbol, start, end, cache=None):
    """
    Daily closes for `symbol` as {date: close}, served from the persistent
    cache when it already covers the range and fetched live otherwise.

    Closes are immutable history, so a covered range never needs the network
    again — which is the whole reason the weekly jobs fit their timeouts at 8
    requests/minute. A live fetch widens the cache in memory; only the warm job
    (price_cache.save) persists it. When a live fetch fails but the cache holds
    part of the range, the partial data is returned: some history scores more
    calls than none.
    """
    cache = price_cache.active() if cache is None else cache
    entry = cache.get(symbol)
    adjustment = sp.adjustment_label()
    if price_cache.covered(entry, start, end, adjustment):
        return price_cache.slice_range(entry, start, end)

    prices = fetch_prices_live(symbol, start, end)
    if prices:
        price_cache.remember(cache, symbol, start, end, prices, adjustment)
        return price_cache.slice_range(cache[symbol], start, end)
    if entry and (entry.get("adjustment") or "split_adjusted") != adjustment:
        return {}  # never serve a differently adjusted series as this one
    return price_cache.slice_range(entry, start, end)


# What every series served by fetch_prices is: the scorecard records it per
# scored claim and refuses series whose adjustment it does not know.
fetch_prices.price_provenance = {
    "provider": "twelvedata", "adjustment": sp.adjustment_label(),
    "corporate_action_status": sp.adjustment_label(),
}


def fetch_price_series(symbol, start, end, cache=None):
    """
    The scorecard's price fetcher: a scorecard_pricing.PriceSeries with the
    closes from fetch_prices (cache first, live for gaps) plus the daily
    highs and lows the cache holds for the same days, so price targets can
    be tested by intraday touch where bars exist and are reported as
    closes-only where they do not.
    """
    cache = price_cache.active() if cache is None else cache
    closes = fetch_prices(symbol, start, end, cache)
    entry = cache.get(symbol) or {}
    return sp.PriceSeries(
        symbol=symbol, closes=dict(closes), provider="twelvedata", adjustment=sp.adjustment_label(),
        corporate_action_status=sp.adjustment_label(), requested_start=start, requested_end=end,
        highs=price_cache.slice_range(entry, start, end, "highs") or None,
        lows=price_cache.slice_range(entry, start, end, "lows") or None,
    )


fetch_price_series.price_provenance = fetch_prices.price_provenance


def price_on_or_after(prices, day, max_lag=MAX_PRICE_LAG_DAYS):
    """Close on `day` or the next available trading day within `max_lag` days."""
    for offset in range(max_lag + 1):
        candidate = day + timedelta(days=offset)
        if candidate in prices:
            return prices[candidate]
    return None


def _directional_calls(records):
    """Yield evaluable calls: (date, channel, ticker, symbol, stance)."""
    for record, asset in _iter_assets(records):
        stance = asset.get("stance")
        if stance not in ("bullish", "bearish"):
            continue
        channel = record.get("channel_name")
        symbol = symbol_for(asset, channel)
        signal_date = _parse_date(record.get("date"))
        if symbol and signal_date and channel:
            yield signal_date, channel, asset.get("ticker"), symbol, stance


def evaluate(records, today, price_fetcher=fetch_prices):
    """
    Score every directional call whose horizon has elapsed. Returns
    {channel: {horizon: {"hits": int, "total": int, "dir_return_sum": float}}}
    where dir_return is the price move in the called direction (positive =
    the call was right by that much).
    """
    calls = list(_directional_calls(records))
    if not calls:
        return {}

    # One price fetch per symbol, covering its full needed range.
    ranges = {}
    for signal_date, _, _, symbol, _ in calls:
        start, end = ranges.get(symbol, (signal_date, signal_date))
        ranges[symbol] = (min(start, signal_date), max(end, signal_date))
    prices = {
        symbol: price_fetcher(symbol, start, today)
        for symbol, (start, _) in ranges.items()
    }

    stats = defaultdict(lambda: {h: {"hits": 0, "total": 0, "dir_return_sum": 0.0} for h in HORIZONS})
    for signal_date, channel, ticker, symbol, stance in calls:
        series = prices.get(symbol) or {}
        entry = price_on_or_after(series, signal_date)
        if entry is None or entry == 0:
            continue
        for horizon in HORIZONS:
            target_day = signal_date + timedelta(days=horizon)
            if target_day > today:
                continue  # horizon not elapsed yet
            later = price_on_or_after(series, target_day)
            if later is None:
                continue
            ret = (later - entry) / entry
            dir_return = ret if stance == "bullish" else -ret
            bucket = stats[channel][horizon]
            bucket["total"] += 1
            bucket["dir_return_sum"] += dir_return
            if dir_return > 0:
                bucket["hits"] += 1
    return {channel: dict(h) for channel, h in stats.items()}


def _format_channel_line(channel, horizons):
    parts = [f"• {channel}"]
    for horizon in HORIZONS:
        bucket = horizons.get(horizon, {"total": 0})
        if bucket["total"]:
            pct = 100.0 * bucket["hits"] / bucket["total"]
            avg = 100.0 * bucket["dir_return_sum"] / bucket["total"]
            parts.append(
                f"{horizon}d: {bucket['hits']}/{bucket['total']} ({pct:.0f}%) avg {avg:+.1f}%"
            )
    return " — ".join(parts) if len(parts) > 1 else ""


def build_scorecard(stats, today):
    """Plain-text report; "" when nothing was evaluable."""
    lines_by_channel = {
        channel: line
        for channel, horizons in stats.items()
        if (line := _format_channel_line(channel, horizons))
    }
    if not lines_by_channel:
        return ""

    total_calls = sum(
        bucket["total"] for horizons in stats.values() for bucket in horizons.values()
    )
    lines = [
        f"🎯 Channel Scorecard — as of {today.isoformat()}",
        f"Directional calls evaluated: {total_calls} (across {len(HORIZONS)} horizons)",
        "",
    ]
    # Rank by 7-day hit rate, most active first on ties.
    def rank(channel):
        bucket = stats[channel].get(HORIZONS[0], {"hits": 0, "total": 0})
        rate = bucket["hits"] / bucket["total"] if bucket["total"] else 0.0
        return (-rate, -bucket["total"])

    for channel in sorted(lines_by_channel, key=rank):
        lines.append(lines_by_channel[channel])
    lines.append("")
    lines.append(LEGACY_DISCLAIMER)
    return "\n".join(lines)


def generate_scorecard(today=None, path=SIGNALS_FILE, price_fetcher=fetch_prices):
    """Load the dataset and build the scorecard. Returns "" while the dataset
    is younger than MIN_DATASET_AGE_DAYS or nothing is evaluable yet."""
    today = today or datetime.now(timezone.utc).date()
    records = load_signals(path)
    dates = [d for d in (_parse_date(r.get("date")) for r in records) if d is not None]
    if not dates:
        log_info("No signal records yet; scorecard will start once data accumulates.")
        return ""
    if min(dates) > today - timedelta(days=MIN_DATASET_AGE_DAYS):
        log_info(
            f"Dataset spans less than {MIN_DATASET_AGE_DAYS} days; "
            "waiting for more history before scoring."
        )
        return ""
    stats = evaluate(records, today, price_fetcher=price_fetcher)
    return build_scorecard(stats, today)


def _session_label(day):
    """'Thu Oct 15' for a date."""
    return f"{day:%a %b} {day.day}"


def generate_canonical_scorecard(today=None, price_fetcher=fetch_price_series, claims=None):
    """
    The strict scorecard block — the Friday report's verdict layer, sent
    after the creators-vs-market report (generate_weekly_report). Canonical
    claims (active runs, no legacy rows, no repeats) whose dated, testable
    forecast has matured, scored by research_analytics.scorecard under
    scorecard_pricing, then the conditional calls published in the last
    CONDITIONAL_WINDOW_DAYS. With fewer scored forecasts than the sample
    minimum the block is compact (_compact_scorecard_lines: one plain line
    per forecast, when the next ones mature, a one-line conditional count);
    from the minimum on it carries research_analytics' full source lines
    and the conditional examples.

    Forecasts count as matured by the last COMPLETED session
    (creator_tracker.default_as_of(today)): the Friday job runs at 16:00 UTC,
    mid-session, so a forecast ending today would be judged on a bar still
    printing. The block is returned even when nothing has matured — a reader
    should see that the verdict layer is still empty, and why; "" only when
    there are no canonical claims. Experimental and unranked unless
    SCORECARD_RANKINGS=true. `price_fetcher` is cache first, live for gaps:
    the job's only live price requests.
    """
    import canonical_claims
    import creator_tracker
    import research_analytics as ra
    today = today or datetime.now(timezone.utc).date()
    claims = canonical_claims.load_canonical_claims() if claims is None else claims
    if not claims:
        log_info("No canonical claims yet; the scorecard starts once research data accumulates.")
        return ""
    as_of = creator_tracker.default_as_of(today)
    sc = ra.scorecard(claims, as_of, price_fetcher)
    excluded, min_sample = sc.pop("_excluded"), sc.pop("_min_sample")
    rankings_on = sc.pop("_rankings_enabled")
    if not sc:
        log_info(f"No matured, scorable forecasts yet (excluded: {excluded}); "
                 "the strict block says so instead of being skipped.")
    rankable = ra.rankable_sources(sc, min_sample)
    label = "ranked" if rankings_on and rankable else sp.SCORECARD_EXPERIMENTAL_LABEL
    window_start = today - timedelta(days=CONDITIONAL_WINDOW_DAYS)
    window = [c for c in claims if canonical_claims.in_window(c, window_start, today)]
    conditionals = ra.conditional_forecast_report(window, today)
    scored = sum(v["n"] for v in sc.values())
    if scored >= min_sample:
        lines = [f"{STRICT_HEADER} ({label})",
                 f"{DATA_LINE} · forecasts matured by the {_session_label(as_of)} close", "",
                 ra.format_scorecard_lines(sc, excluded, min_sample, rankings_on, rankable)]
        conditional = ra.format_conditional_report(conditionals)
        if conditional:
            lines += ["", conditional]
    else:
        lines = [f"{STRICT_HEADER} ({label})", COMPACT_DATA_LINE.format(close=_session_label(as_of))]
        lines += _compact_scorecard_lines(sc, excluded, min_sample, claims, as_of)
        conditional = _conditional_count_line(conditionals)
        if conditional:
            lines += ["", conditional]
    lines += ["", DISCLAIMER]
    return "\n".join(lines)


def _mon_d(value):
    """'Sep 17' from an ISO date string or a date; '?' otherwise."""
    try:
        day = value if hasattr(value, "strftime") else datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return "?"
    return f"{day:%b} {day.day}"


def _forecast_line(source, s):
    """One scored forecast in plain words: 'More Crypto Online — ETH ▼ Sep 17
    → Oct 8: +1.1% vs Bitcoin +7.0% — wrong direction'."""
    label = s.get("instrument_ticker") or s.get("symbol") or "?"
    arrow = "▲" if s.get("direction") == "bullish" else "▼"
    ret = s.get("raw_return") or 0.0
    line = (f"• {source} — {label} {arrow} {_mon_d(s.get('entry_resolved_trading_date'))} → "
            f"{_mon_d(s.get('evaluation_resolved_trading_date'))}: {100 * ret:+.1f}%")
    bench = s.get("benchmark_symbol")
    if bench and s.get("benchmark_return") is not None:
        name = _BENCHMARK_NAMES.get(bench) or bench.split(".")[0].upper()
        line += f" vs {name} {100 * s['benchmark_return']:+.1f}%"
    if not ret:
        line += " — no move"
    else:
        line += " — right direction" if (ret > 0) == (s.get("direction") == "bullish") else " — wrong direction"
    if s.get("target_test_method"):
        line += f"; target {s.get('target_value')} {'reached' if s.get('target_reached') else 'not reached'}"
    return line


def _next_maturity(claims, as_of):
    """The earliest end date after `as_of` among the dated, testable price
    forecasts still waiting to mature, or None."""
    import canonical_claims
    ends = []
    for c in canonical_claims.headline_claims(claims):
        if not c.get("testable") or c.get("condition") or c.get("forecast_metric") not in (None, "price"):
            continue
        try:
            end = datetime.strptime(str(c.get("forecast_end_date"))[:10], "%Y-%m-%d").date()
        except ValueError:
            continue
        if end > as_of:
            ends.append(end)
    return min(ends) if ends else None


def _compact_scorecard_lines(sc, excluded, min_sample, claims, as_of):
    """The strict block below the sample minimum: one plain line per scored
    forecast (unranked), then how many dated price forecasts mature later
    and when the next one does."""
    scored = sum(v["n"] for v in sc.values())
    lines = [f"{scored} scored so far (unranked) — a comparison needs {min_sample}." if scored
             else "• no matured forecasts yet (unranked)."]
    for source in sorted(sc):
        lines += [_forecast_line(source, s) for s in sc[source].get("scored") or []]
    waiting = (excluded or {}).get("not_matured", 0)
    if waiting:
        nxt = _next_maturity(claims, as_of)
        lines.append(f"{waiting:,} dated price forecast{' matures' if waiting == 1 else 's mature'} later"
                     + (f" (next: {_mon_d(nxt)})." if nxt else "."))
    return lines


def _conditional_count_line(rows):
    """One line for the creators' conditional calls in the window (they wait
    for their condition and stay out of the track record); the research
    report lists them."""
    import research_analytics as ra
    own = [r for r in rows or [] if r.get("own_view")]
    if not own:
        return ""
    evaluated = sum(1 for r in own if r.get("status") in ra._EVALUATED_STATUSES)
    return (f"🔀 {len(own)} own conditional call{'' if len(own) == 1 else 's'} in the last {CONDITIONAL_WINDOW_DAYS} "
            f"days (\"buy if it drops to …\") stay out of the track record; "
            + (f"{evaluated} evaluated so far." if evaluated else "none evaluated yet.")
            + " The research report lists them.")


def generate_weekly_report(today=None, claims=None, price_fetcher=fetch_price_series,
                           tracker_fetcher=None, charts_dir=None):
    """
    The Friday report, in reading order:

      1. creators vs the market — creator_tracker.track_record at
         creator_tracker.default_as_of(today) (the Thursday close for the
         Friday run), formatted by creator_tracker.format_report. Read
         through `tracker_fetcher`, by default creator_tracker.cache_fetcher():
         the committed price cache only, never a live request;
      2. its charts, when `charts_dir` is given: pulse_charts'
         render_tracker_charts over build_tracker_chart_data(record), drawn
         from the same record as the text. Best-effort: any failure is logged
         and leaves no charts, never a broken report;
      3. the strict scorecard block, generate_canonical_scorecard(today,
         price_fetcher, claims) — the only part that may fetch live.

    Returns {"messages": [str], "charts": [path], "record": dict}. The
    messages are the tracker's (none when no call is marked yet) followed by
    the strict block, which is always last and present whenever there are
    claims. With no claims: no messages, no charts and an empty record.
    """
    import canonical_claims
    import creator_tracker
    today = today or datetime.now(timezone.utc).date()
    claims = canonical_claims.load_canonical_claims() if claims is None else claims
    if not claims:
        log_info("No canonical claims yet; the Friday report starts once research data accumulates.")
        return {"messages": [], "charts": [], "record": {}}
    as_of = creator_tracker.default_as_of(today)
    record = creator_tracker.track_record(claims, as_of, tracker_fetcher or creator_tracker.cache_fetcher())
    stats = record["stats"]
    source = "the given fetcher" if tracker_fetcher else "the price cache only, no live requests"
    log_info(f"Creators vs the market: {stats['calls']} calls from {stats['video_calls']} video calls, "
             f"{stats['scored']} marked to market at {record['as_of']} (requested {as_of}; read from {source}).")
    worklist = creator_tracker.unpriced_assets(record)
    if worklist:
        # The ticker gaps behind "no ticker" / "not in the instrument
        # registry", for the maintainer (ticker_audit has the full list).
        log_info("Calls with no instrument (the ticker worklist): "
                 + ", ".join(f"{asset} {n}" for asset, n in worklist))
    if record["as_of"] < as_of:
        # Maintainer detail for the workflow log; the report itself warns
        # readers once the prices are more than a week old.
        log_warn(f"The newest S&P 500 close in the price cache is {record['as_of']}, before the requested "
                 f"{as_of}: calls are marked to that close. Has the Friday 06:00 UTC warm-prices run "
                 "succeeded?")
    messages = list(creator_tracker.format_report(record))
    charts = _tracker_charts(record, charts_dir) if charts_dir else []
    strict = generate_canonical_scorecard(today, price_fetcher, claims=claims)
    if strict:
        messages.append(strict)
    return {"messages": messages, "charts": charts, "record": record}


def _tracker_charts(record, out_dir):
    """The creators-vs-market PNGs for `record`, or [] — never raises."""
    try:
        import pulse_charts
        paths = pulse_charts.render_tracker_charts(pulse_charts.build_tracker_chart_data(record), out_dir)
        return [p for p in (paths or []) if p]
    except Exception as e:  # noqa: BLE001 - charts illustrate the text and must never block it
        log_warn(f"Creators-vs-market charts unavailable this week: {describe_error(e)}")
        return []


def _legacy_main(dry_run):
    """--legacy: the 7/30-day hit rate over data/signals.jsonl, unchanged."""
    scorecard = generate_scorecard()
    if not scorecard:
        log_info("Nothing to score this week; skipping the scorecard.")
        return 0

    if dry_run:
        print(scorecard)
        return 0

    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHANNEL_ID")
    if not token or not chat_id:
        log_error("TELEGRAM_TOKEN and TELEGRAM_CHANNEL_ID must be set to send the scorecard.")
        return 1

    if send_telegram_text(token, chat_id, scorecard):
        log_info("Weekly channel scorecard sent.")
        return 0
    return 1


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Friday report: creators vs the market (mark-to-market from the price cache), "
                    "its charts, and the strict scorecard of matured forecasts"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print every message and write the charts locally instead of sending")
    parser.add_argument("--legacy", action="store_true",
                        help="Use the legacy 7/30-day scorecard over data/signals.jsonl (compatibility only)")
    parser.add_argument("--charts-dir", default=CHARTS_DIR,
                        help=f"Directory the chart PNGs are written to (default: {CHARTS_DIR})")
    parser.add_argument("--no-charts", action="store_true", help="Text only, without the chart album")
    args = parser.parse_args(argv)

    if args.legacy:
        return _legacy_main(args.dry_run)

    report = generate_weekly_report(charts_dir=None if args.no_charts else args.charts_dir)
    messages, charts = report["messages"], report["charts"]
    if not messages:
        log_info("Nothing to report this week; skipping the Friday report.")
        return 0

    if args.dry_run:
        print("\n\n".join(messages))
        if charts:
            log_info(f"Charts written: {', '.join(charts)}")
        return 0

    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHANNEL_ID")
    if not token or not chat_id:
        log_error("TELEGRAM_TOKEN and TELEGRAM_CHANNEL_ID must be set to send the scorecard.")
        return 1

    # The strict block is always the last message (generate_weekly_report).
    *tracker_messages, strict = messages
    for i, text in enumerate(tracker_messages, 1):
        if not send_telegram_text(token, chat_id, text):
            log_error(f"Creators-vs-market message {i}/{len(tracker_messages)} was not sent; "
                      "stopping so no later part arrives without it.")
            return 1
    # The charts illustrate the text; failing to send them must not fail the run.
    if charts and not send_telegram_photo_album(token, chat_id, charts, caption=CHART_ALBUM_CAPTION):
        log_warn("Chart album failed to send; the text report goes out without it.")
    if not send_telegram_text(token, chat_id, strict):
        log_error("The strict scorecard block was not sent.")
        return 1
    log_info(f"Friday report sent: {len(tracker_messages)} creators-vs-market message(s), "
             f"{len(charts)} chart(s) and the strict scorecard.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
