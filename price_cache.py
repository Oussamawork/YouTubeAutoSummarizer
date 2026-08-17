"""Persistent daily-close cache for the price-dependent analytics.

Daily closes are immutable history: once a trading day's close is known it
never changes. Refetching them every run is what made the price layer
expensive — the free Twelve Data tier allows 8 requests/minute, and the signals
dataset already spans ~180 distinct symbols, so a cold run paces out past any
sensible workflow timeout.

So closes are cached in `data/prices.json` and topped up by a separate weekly
job (`warm-prices.yml`, Sundays) that has the time to be slow. The pulse and the
scorecard then read the cache and only reach the network for what it is missing.

Cache shape, keyed by our internal (Stooq-style, provider-independent) symbol:

    {"version": 1, "symbols": {
        "nvda.us": {"from": "2026-07-01", "to": "2026-08-16",
                    "closes": {"2026-07-01": 180.5, ...}}}}

`from`/`to` record the range actually *requested*, which is what makes coverage
decidable: closes alone can't distinguish "no trading that day" (a weekend)
from "never fetched".
"""
import json
import os
from datetime import datetime, timedelta

from log import log_info, log_warn

CACHE_FILE = "data/prices.json"
CACHE_VERSION = 1
# Closes older than this are dropped when the cache is saved. The longest
# horizon the scorecard scores is 30 days, so a year of history is already
# generous; the bound stops the committed file growing without limit.
MAX_HISTORY_DAYS = 400


def _parse_day(value):
    try:
        return datetime.strptime((value or "").strip(), "%Y-%m-%d").date()
    except (ValueError, AttributeError):
        return None


def load(path=CACHE_FILE):
    """Read the cache. A missing, unreadable or unrecognised file is not an
    error — it just means a cold start, and the callers refetch."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError) as e:
        log_warn(f"Could not read the price cache at {path}: {e}")
        return {}
    if not isinstance(payload, dict) or payload.get("version") != CACHE_VERSION:
        log_warn(f"Ignoring price cache at {path}: unrecognised format.")
        return {}
    cache = {}
    for symbol, entry in (payload.get("symbols") or {}).items():
        if not isinstance(entry, dict):
            continue
        start, end = _parse_day(entry.get("from")), _parse_day(entry.get("to"))
        if not start or not end:
            continue
        closes = {}
        for day, close in (entry.get("closes") or {}).items():
            parsed = _parse_day(day)
            if parsed is None:
                continue
            try:
                closes[parsed] = float(close)
            except (TypeError, ValueError):
                continue
        cache[symbol] = {"from": start, "to": end, "closes": closes}
    return cache


def save(cache, path=CACHE_FILE, today=None):
    """Write the cache, trimming history beyond MAX_HISTORY_DAYS. Returns True
    on success; a write failure is logged, never raised."""
    today = today or datetime.utcnow().date()
    cutoff = today - timedelta(days=MAX_HISTORY_DAYS)
    symbols = {}
    for symbol, entry in sorted(cache.items()):
        closes = {d.isoformat(): c for d, c in sorted(entry["closes"].items())
                  if d >= cutoff}
        if not closes:
            continue
        symbols[symbol] = {
            "from": max(entry["from"], cutoff).isoformat(),
            "to": entry["to"].isoformat(),
            "closes": closes,
        }
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": CACHE_VERSION, "symbols": symbols}, f,
                      indent=1, sort_keys=True)
            f.write("\n")
    except OSError as e:
        log_warn(f"Could not write the price cache to {path}: {e}")
        return False
    total = sum(len(entry["closes"]) for entry in symbols.values())
    log_info(f"Price cache saved: {len(symbols)} symbols, {total} closes.")
    return True


def covered(entry, start, end):
    """True when the cached range spans [start, end] — i.e. the request can be
    answered without touching the network."""
    return bool(entry) and entry["from"] <= start and entry["to"] >= end


def slice_range(entry, start, end):
    """The cached closes falling inside [start, end]."""
    if not entry:
        return {}
    return {d: c for d, c in entry["closes"].items() if start <= d <= end}


def remember(cache, symbol, start, end, closes):
    """Merge a freshly fetched range into the cache, widening the symbol's
    covered range. `start`/`end` are what was *asked for*, so a symbol that
    simply had no trading days in the window still counts as covered and is
    not asked for again."""
    entry = cache.get(symbol)
    if entry:
        entry["from"] = min(entry["from"], start)
        entry["to"] = max(entry["to"], end)
        entry["closes"].update(closes)
    else:
        cache[symbol] = {"from": start, "to": end, "closes": dict(closes)}
    return cache[symbol]


# The process-wide cache. Loaded once on first use so every caller in a run
# shares it (and so a run never refetches a symbol it already fetched).
_ACTIVE = None


def active(path=CACHE_FILE):
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = load(path)
    return _ACTIVE


def reset(cache=None):
    """Replace the process-wide cache. Tests use this to stay isolated; the
    warm job uses it to load from an explicit path."""
    global _ACTIVE
    _ACTIVE = cache
    return _ACTIVE
