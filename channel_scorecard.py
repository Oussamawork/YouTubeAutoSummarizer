import argparse
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

from log import log_info, log_warn, log_error
from market_pulse import (
    SIGNALS_FILE, load_signals, _parse_date, _iter_assets, canonical_ticker,
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
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
RETRY_BACKOFF = 2
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
    ticker = (canonical_ticker(asset) or "").lower()
    if not ticker:
        return None
    asset_type = asset.get("type")
    if asset_type in ("stock", "etf"):
        return f"{ticker}.us"
    if asset_type == "crypto":
        return f"{ticker}usd"
    return None  # index/commodity/macro naming on Stooq is too inconsistent


def fetch_prices(symbol, start, end):
    """
    Daily closes for `symbol` from Stooq as {date: close}. Returns {} on any
    failure (logged, never raises); the caller just skips those assets.
    """
    url = STOOQ_URL.format(
        symbol=symbol, d1=start.strftime("%Y%m%d"), d2=end.strftime("%Y%m%d")
    )
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
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
