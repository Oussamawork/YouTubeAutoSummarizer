import argparse
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

import price_cache
from log import log_info, log_warn, log_error
from market_pulse import (
    SIGNALS_FILE, load_signals, _parse_date, _iter_assets, canonical_ticker,
    UNPRICEABLE_TICKERS,
)
from sendToTelegram import send_telegram_text

# Weekly per-channel accuracy scorecard: joins the directional calls recorded
# in data/signals.jsonl with free daily price data (Stooq, keyless CSV) and
# measures each channel's hit rate — was the price higher after a bullish call,
# lower after a bearish one — at 7- and 30-day horizons. Runs every Friday from
# weekly-scorecard.yml; silently skips until the dataset spans at least a week.
# Hit rates over tiny samples are noise: the report always shows sample sizes,
# and remains research input, not investment advice.

load_dotenv('.env')

STOOQ_URL = "https://stooq.com/q/d/l/?s={symbol}&d1={d1}&d2={d2}&i=d"
# Stooq is behind a JavaScript browser check and is NOT usable server-side as
# of 2026-08-17, measured directly:
#   - default python-requests UA  -> 404 for every symbol
#   - browser UA (these headers)  -> 200 whose body is the JS challenge page,
#                                    not CSV, so the parser still yields nothing
# The 404 masqueraded as "no such ticker" in the logs, which is how this went
# unnoticed from the first scheduled run (2026-07-27) onward: every price lookup
# has failed since, silently disabling implied-upside annotations, track-record
# weighting, the scorecard and the price-target chart. These headers are kept
# because they are correct for a CSV client and cost nothing if Stooq drops the
# challenge, but restoring prices needs a different provider — do not read their
# presence as "prices work".
STOOQ_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
    "Accept": "text/csv,text/plain,*/*",
}
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
RETRY_BACKOFF = 2

# Twelve Data is the price source once TWELVEDATA_API is set: unlike Stooq it
# is built for server-side use, and its free tier (800 requests/day, 8 per
# minute) comfortably covers a weekly pulse and scorecard.
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
TWELVEDATA_MAX_REQUESTS = int(os.getenv("TWELVEDATA_MAX_REQUESTS", "120") or 120)
# A signal dated on a weekend/holiday uses the next trading day's close, up to
# this many days later; beyond that the price point is treated as missing.
MAX_PRICE_LAG_DAYS = 5
HORIZONS = (7, 30)
# The scorecard waits until the dataset spans at least this many days.
MIN_DATASET_AGE_DAYS = 7

DISCLAIMER = (
    "⚠️ Directional hit rate vs price after the call. Small samples are noise. "
    "Research input, not investment advice."
)


def symbol_for(asset):
    """
    Map an asset entry to a Stooq symbol, or None when it isn't priceable.
    Uses the canonical ticker (recorded, else the curated alias table), so a
    call on "Chevron" scores even though the speaker never said "CVX" — a
    third of directional calls were ticker-less and invisible before this.
    """
    ticker = (canonical_ticker(asset) or "").upper()
    if not ticker or ticker in UNPRICEABLE_TICKERS:
        return None
    ticker = ticker.lower()
    asset_type = asset.get("type")
    if asset_type in ("stock", "etf"):
        return f"{ticker}.us"
    if asset_type == "crypto":
        return f"{ticker}usd"
    return None  # index/commodity/macro naming on Stooq is too inconsistent


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
    log_warn(f"Symbol search for {query!r} gave up after {MAX_RETRIES} rate-limited attempts.")
    return []


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


def _parse_twelvedata(payload, symbol):
    """{date: close} from a Twelve Data time_series body. The API reports its
    own errors inside a 200 body (status="error"), so that is checked before
    the values are read."""
    if not isinstance(payload, dict):
        log_warn(f"Unexpected Twelve Data response for {symbol}.")
        return {}
    if payload.get("status") == "error":
        log_warn(f"Twelve Data error for {symbol}: "
                 f"{payload.get('code')} {payload.get('message', '')[:120]}")
        return {}
    prices = {}
    for row in payload.get("values") or []:
        day = _parse_date((row.get("datetime") or "")[:10])
        try:
            close = float(row.get("close"))
        except (TypeError, ValueError):
            continue
        if day:
            prices[day] = close
    if not prices:
        log_warn(f"No usable price data from Twelve Data for {symbol}.")
    return prices


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


def fetch_prices_live(symbol, start, end):
    """
    Daily closes straight from the provider: Twelve Data when a key is
    configured, Stooq otherwise. Returns {} on any failure (logged, never
    raises).

    Stooq is the keyless legacy path and has been unusable server-side since
    2026-08-17 (see STOOQ_HEADERS) — without a Twelve Data key this returns
    nothing, which market_pulse.fetch_latest_prices reports as one loud line.
    """
    api_key = twelvedata_key()
    if api_key:
        return fetch_prices_twelvedata(symbol, start, end, api_key)
    return fetch_prices_stooq(symbol, start, end)


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
    if price_cache.covered(entry, start, end):
        return price_cache.slice_range(entry, start, end)

    prices = fetch_prices_live(symbol, start, end)
    if prices:
        price_cache.remember(cache, symbol, start, end, prices)
        return price_cache.slice_range(cache[symbol], start, end)
    return price_cache.slice_range(entry, start, end)


def fetch_prices_stooq(symbol, start, end):
    """Daily closes for `symbol` from Stooq as {date: close}; {} on failure."""
    url = STOOQ_URL.format(
        symbol=symbol, d1=start.strftime("%Y%m%d"), d2=end.strftime("%Y%m%d")
    )
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=STOOQ_HEADERS, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            log_warn(f"Stooq request error for {symbol} (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
                continue
            return {}
        if resp.status_code != 200:
            log_warn(f"Stooq returned {resp.status_code} for {symbol}.")
            return {}
        return _parse_stooq_csv(resp.text, symbol)
    return {}


def _parse_stooq_csv(text, symbol):
    lines = (text or "").strip().splitlines()
    if len(lines) < 2 or not lines[0].lower().startswith("date"):
        log_warn(f"No usable price data from Stooq for {symbol}.")
        return {}
    prices = {}
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < 5:
            continue
        day = _parse_date(parts[0])
        try:
            close = float(parts[4])
        except ValueError:
            continue
        if day is not None:
            prices[day] = close
    return prices


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
        symbol = symbol_for(asset)
        signal_date = _parse_date(record.get("date"))
        channel = record.get("channel_name")
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
    lines.append(DISCLAIMER)
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


def main():
    parser = argparse.ArgumentParser(
        description="Per-channel accuracy scorecard from data/signals.jsonl + Stooq prices"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print instead of sending")
    args = parser.parse_args()

    scorecard = generate_scorecard()
    if not scorecard:
        log_info("Nothing to score this week; skipping the scorecard.")
        return 0

    if args.dry_run:
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


if __name__ == "__main__":
    sys.exit(main())
