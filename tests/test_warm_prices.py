"""The price-cache warmer (warm_prices.py) as the creators-vs-market tracker
needs it: every range the tracker reads is warmed, no request reaches past
the last completed session, and a partial bar an earlier warm stored is
replaced on the next refetch. Offline: every fetcher here is a stub."""
import sys
from datetime import date, datetime, timedelta, timezone

import pytest

import canonical_claims
import channel_scorecard as cs
import creator_tracker
import price_cache
import scorecard_pricing as sp
import warm_prices
from tests.test_canonical_analytics import _claim

US = sp.EXCHANGES["us"]
SUNDAY = date(2026, 10, 18)          # the Sunday 06:00 UTC warm
FRIDAY = date(2026, 10, 16)          # the Friday 06:00 UTC warm; also default_as_of(SUNDAY)
THURSDAY = date(2026, 10, 15)        # default_as_of(FRIDAY)
SATURDAY = date(2026, 10, 17)
TARGET = {"claim_type": "price_target", "target_kind": "absolute_value", "currency": "USD", "target_unit": "USD"}


def _rec(day, assets):
    return {"date": day, "channel_name": "Chan",
            "signals": {"assets": assets, "market_sentiment": "neutral", "topics": []}}


def _asset(name, ticker, asset_type, stance="bullish"):
    return {"name": name, "ticker": ticker, "stance": stance, "type": asset_type, "action": "none",
            "conviction": "medium", "horizon": "unspecified", "catalysts": [], "price_target": None}


def _calls():
    """An NVDA call, a Bitcoin call and a gold-theme call (priced via GLD)."""
    return [_claim("Alpha", "NVDA", "bullish", "unspecified", published="2026-09-08"),
            _claim("Beta", "BTC", "bearish", "unspecified", published="2026-09-10", asset_type="crypto", n=1),
            _claim("Gamma", "Gold", "bullish", "unspecified", published="2026-09-15", ticker=None,
                   asset_type="commodity", n=2)]


def _history(symbol, start=date(2026, 8, 1), end=SUNDAY):
    """A provider's full daily history for `symbol` (every day for crypto,
    NYSE sessions otherwise), including the warm day's own bar."""
    crypto = warm_prices.last_complete_day(symbol, SUNDAY) == SATURDAY
    out, day, i = {}, start, 0
    while day <= end:
        if crypto or sp.is_trading_day(US, day):
            out[day] = 100.0 + i
            i += 1
        day += timedelta(days=1)
    return out


def test_needed_ranges_include_tracker_symbols_and_market_context():
    ranges = warm_prices.needed_ranges([], SUNDAY, _calls())
    as_of = creator_tracker.default_as_of(SUNDAY)
    assert as_of == FRIDAY
    assert ranges["nvda.us"] == (date(2026, 9, 7), as_of)          # from the day before the call
    assert ranges["btcusd"] == (date(2026, 9, 7), as_of)            # the call, and the market context
    assert ranges["gld.us"] == (date(2026, 9, 14), as_of)           # the GOLD theme's proxy ETF
    # The benchmarks and the market context (S&P 500, Nasdaq-100, Bitcoin, Ether).
    for symbol in ("spy.us", "qqq.us", "ethusd"):
        assert ranges[symbol] == (date(2026, 9, 7), as_of), symbol
    # Exactly what the tracker will ask for, nothing it will not.
    assert {s: (a, b) for s, a, b in creator_tracker.needed_price_ranges(_calls(), SUNDAY)} == ranges


def test_tracker_ranges_need_claims():
    """No claims, no tracker report: not even its market context is warmed
    (the legacy-only warm keeps asking for the legacy symbols alone)."""
    assert warm_prices.tracker_ranges([], SUNDAY) == []
    assert warm_prices.needed_ranges([], SUNDAY, claims=[]) == {}
    legacy = warm_prices.needed_ranges([_rec("2026-10-14", [_asset("Nvidia", "NVDA", "stock")])], SUNDAY, claims=[])
    assert set(legacy) == {"nvda.us"}


def test_needed_ranges_loads_the_claims_once(monkeypatch):
    loads = []
    monkeypatch.setattr(warm_prices, "_load_claims", lambda: loads.append(1) or _calls())
    ranges = warm_prices.needed_ranges([], SUNDAY)
    assert loads == [1] and {"nvda.us", "qqq.us", "gld.us"} <= set(ranges)


def test_last_complete_day():
    assert warm_prices.last_complete_day("btcusd", SUNDAY) == SATURDAY
    assert warm_prices.last_complete_day("nvda.us", SUNDAY) == FRIDAY
    assert warm_prices.last_complete_day("nvda.us", FRIDAY) == THURSDAY
    assert warm_prices.last_complete_day("ethusd", FRIDAY) == THURSDAY
    # The Tuesday after Labor Day: the last session is the Friday before it.
    assert warm_prices.last_complete_day("spy.us", date(2026, 9, 8)) == date(2026, 9, 4)
    assert warm_prices.last_complete_day("solusd", date(2026, 9, 8)) == date(2026, 9, 7)


def test_warm_caps_requests_at_the_last_completed_session():
    # Legacy ranges run to `today` (needed_ranges' contract); the requests do not.
    records = [_rec("2026-10-14", [_asset("Nvidia", "NVDA", "stock"), _asset("Bitcoin", "BTC", "crypto")])]
    assert warm_prices.needed_ranges(records, SUNDAY, claims=[])["btcusd"] == (date(2026, 10, 14), SUNDAY)
    asked = {}

    def fetcher(symbol, start, end):
        asked[symbol] = (start, end)
        return {d: v for d, v in _history(symbol).items() if start <= d <= end}

    cache = {}
    assert warm_prices.warm(records, SUNDAY, cache, fetcher=fetcher, claims=[]) == (2, 0, 0)
    assert asked == {"nvda.us": (date(2026, 10, 14), FRIDAY),       # Sunday: US to Friday ...
                     "btcusd": (date(2026, 10, 14), SATURDAY)}      # ... crypto to Saturday
    assert cache["btcusd"]["to"] == SATURDAY and max(cache["btcusd"]["closes"]) == SATURDAY

    # Coverage is tested against the capped end: a second Sunday run is free.
    asked.clear()
    assert warm_prices.warm(records, SUNDAY, cache, fetcher=fetcher, claims=[]) == (0, 2, 0)
    assert asked == {}

    # Friday 06:00 UTC: Thursday's close for both, the warm day's never.
    asked.clear()
    warm_prices.warm(records, FRIDAY, {}, fetcher=fetcher, claims=[])
    assert asked == {"nvda.us": (date(2026, 10, 14), THURSDAY), "btcusd": (date(2026, 10, 14), THURSDAY)}

    # A call from the warm day itself has no completed close yet: nothing to fetch.
    asked.clear()
    today_only = [_rec(FRIDAY.isoformat(), [_asset("Bitcoin", "BTC", "crypto")])]
    assert warm_prices.warm(today_only, FRIDAY, {}, fetcher=fetcher, claims=[]) == (0, 1, 0)
    assert asked == {}


def test_open_range_refetch_replaces_a_partial_last_bar():
    # The Sunday 2026-10-11 warm stored that day's bar while it was printing.
    days = [date(2026, 10, 1) + timedelta(days=i) for i in range(11)]
    closes = {d: 80000.0 + 100 * i for i, d in enumerate(days)}
    closes[date(2026, 10, 11)] = 86000.0                               # partial
    cache = {}
    price_cache.remember(cache, "btcusd", days[0], days[-1], closes,
                         highs={date(2026, 10, 11): 86100.0}, lows={date(2026, 10, 11): 85500.0})
    records = [_rec("2026-10-09", [_asset("Bitcoin", "BTC", "crypto")])]    # an open legacy range
    final = {d: 80000.0 + 100 * i for i, d in enumerate(days)}
    final.update({date(2026, 10, 11): 87000.0, date(2026, 10, 12): 87500.0, date(2026, 10, 15): 88000.0,
                  FRIDAY: 90000.0})                                          # FRIDAY: still printing
    asked = []

    def provider(symbol, start, end):
        asked.append((symbol, start, end))
        # A provider that answers past the requested end must not slip the
        # in-progress bar in either.
        return cs.Bars(final, highs={d: v + 500 for d, v in final.items()},
                       lows={d: v - 500 for d, v in final.items()})

    assert warm_prices.warm(records, FRIDAY, cache, fetcher=provider, claims=[]) == (1, 0, 0)
    assert asked == [("btcusd", date(2026, 10, 1), THURSDAY)]               # the union, capped
    entry = cache["btcusd"]
    assert entry["closes"][date(2026, 10, 11)] == 87000.0                   # repaired
    assert entry["highs"][date(2026, 10, 11)] == 87500.0 and entry["lows"][date(2026, 10, 11)] == 86500.0
    assert FRIDAY not in entry["closes"] and FRIDAY not in entry["highs"]
    assert entry["to"] == THURSDAY and max(entry["closes"]) == THURSDAY


def test_pulse_target_ranges_follow_aggregate_views_by_asset_and_end_at_as_of(monkeypatch):
    in_window = [
        # Two creators' NVDA targets in different horizon buckets: one asset,
        # one range (the per-bucket aggregation made one per bucket).
        _claim("Alpha", "NVDA", "bullish", "short", published="2026-10-14", target_value=250.0, **TARGET),
        _claim("Beta", "NVDA", "bullish", "long", published="2026-10-15", target_value=300.0, n=1, **TARGET),
        _claim("Gamma", "AAPL", "bullish", "unspecified", published="2026-10-14", n=2),   # no target
    ]
    old = _claim("Delta", "MSFT", "bullish", "unspecified", published="2026-10-01", target_value=600.0,
                 n=3, **TARGET)                                             # before the pulse window
    seen = []
    real = canonical_claims.aggregate_views_by_asset

    def spy(claims, *args, **kwargs):
        seen.append(sorted(c["claim_id"] for c in claims))
        return real(claims, *args, **kwargs)

    monkeypatch.setattr(canonical_claims, "aggregate_views_by_asset", spy)
    ranges = warm_prices.canonical_ranges(in_window + [old], SUNDAY)
    as_of = creator_tracker.default_as_of(SUNDAY)
    assert seen == [sorted(c["claim_id"] for c in in_window)]              # the pulse window only
    assert ranges == [("nvda.us", as_of - timedelta(days=warm_prices.LATEST_PRICE_LOOKBACK_DAYS), as_of)]
    assert as_of == FRIDAY                                                  # the close the pulse shows

    # A target the pulse cannot annotate (a non-dollar listing) is not warmed.
    monkeypatch.setattr(canonical_claims, "priceable_symbol", lambda entry: "nvda.xetra")
    assert warm_prices.canonical_ranges(in_window, SUNDAY) == []


def test_warm_then_track_record_reads_everything_from_the_cache(monkeypatch):
    monkeypatch.setattr(cs, "fetch_prices_live", lambda *a, **k: pytest.fail("the tracker never goes live"))
    claims = _calls()

    def provider(symbol, start, end):
        return {d: v for d, v in _history(symbol).items() if start <= d <= end}

    cache = {}
    fetched, skipped, failed = warm_prices.warm([], SUNDAY, cache, fetcher=provider, claims=claims)
    assert (fetched, failed) == (len(cache), 0) and fetched >= 6
    # The tracker marks crypto on the stocks' as-of (Friday), so that is as
    # far as its Bitcoin range goes; the warm day's own bar is never stored.
    assert max(cache["btcusd"]["closes"]) == FRIDAY and SUNDAY not in cache["btcusd"]["closes"]

    price_cache.reset(cache)
    fetch = creator_tracker.cache_fetcher()
    record = creator_tracker.track_record(claims, creator_tracker.default_as_of(SUNDAY), fetch)
    assert record["as_of"] == FRIDAY and record["market_data"]
    assert record["stats"]["scored"] == record["stats"]["calls"] == 3
    assert {r["symbol"] for r in record["calls"]} == {"nvda.us", "btcusd", "gld.us"}
    for symbol, start, end in fetch.requests:                               # every read is covered
        assert price_cache.covered(cache[symbol], start, end), (symbol, start, end)
    assert {s for s, _, _ in fetch.requests} >= {"spy.us", "qqq.us", "btcusd", "ethusd"}


def test_dry_run_report_lists_the_tracker_symbols(monkeypatch):
    logged = []
    monkeypatch.setattr(warm_prices, "log_info", lambda m: logged.append(m))
    cache = {}
    price_cache.remember(cache, "nvda.us", date(2026, 9, 1), FRIDAY, {date(2026, 9, 2): 1.0})
    report = warm_prices.dry_run_report([], _calls(), SUNDAY, cache)
    assert {"nvda.us", "btcusd", "gld.us", "spy.us", "qqq.us", "ethusd"} == set(report["tracker"])
    assert "nvda.us" not in report["to_fetch"]                              # covered through Friday
    assert "qqq.us" in report["to_fetch"] and "qqq.us" in report["never_cached"]
    assert "nvda.us" not in report["never_cached"]
    assert any(m.startswith("Creators-vs-market tracker: 6 of the needed symbols") for m in logged)
    tracker_line = next(m for m in logged if m.startswith("Tracker symbols: "))
    for symbol in ("qqq.us", "spy.us", "btcusd", "ethusd", "nvda.us", "gld.us"):
        assert symbol in tracker_line


def test_main_dry_run_never_fetches(monkeypatch, tmp_path):
    """`python warm_prices.py --dry-run` with canonical claims and no legacy
    rows: it reports, needs no key and makes no request."""
    today = datetime.now(timezone.utc).date()
    claims = [_claim("Alpha", "NVDA", "bullish", "unspecified",
                     published=(today - timedelta(days=30)).isoformat())]
    monkeypatch.setattr(warm_prices, "_load_claims", lambda: claims)
    monkeypatch.setattr(warm_prices, "load_signals", lambda path: [])
    monkeypatch.setattr(cs, "fetch_prices_live", lambda *a, **k: pytest.fail("a dry run never fetches"))
    monkeypatch.delenv("TWELVEDATA_API", raising=False)
    monkeypatch.delenv("TWELVEDATA_API_KEY", raising=False)
    logged = []
    monkeypatch.setattr(warm_prices, "log_info", lambda m: logged.append(m))
    monkeypatch.setattr(sys, "argv", ["warm_prices.py", "--dry-run", "--cache", str(tmp_path / "prices.json")])
    assert warm_prices.main() == 0
    tracker_line = next(m for m in logged if m.startswith("Tracker symbols: "))
    assert all(s in tracker_line for s in ("nvda.us", "spy.us", "qqq.us", "btcusd", "ethusd"))


def test_warm_goes_context_first_then_tracker_stalest_first():
    cache = {}
    price_cache.remember(cache, "nvda.us", date(2026, 9, 1), date(2026, 10, 1), {date(2026, 9, 2): 1.0})
    price_cache.remember(cache, "aapl.us", date(2026, 9, 1), date(2026, 10, 8), {date(2026, 9, 2): 1.0})
    ranges = {s: (date(2026, 9, 1), FRIDAY) for s in ("aapl.us", "nvda.us", "zzz.us", "spy.us", "btcusd", "msft.us")}
    order = [s for s, _ in warm_prices.warm_order(ranges, cache, tracker_symbols={"aapl.us", "nvda.us",
                                                                                  "msft.us"})]
    # Context and benchmarks; then the tracker's symbols, never cached first, then the stalest;
    # then everything else.
    assert order == ["spy.us", "btcusd", "msft.us", "nvda.us", "aapl.us", "zzz.us"]


def test_warm_stops_at_its_deadline_with_the_market_fetched_and_saves_as_it_goes(monkeypatch):
    claims = _calls()
    ticks = iter(range(0, 10_000, 60))                       # every clock read is a minute later
    requested, saves = [], []

    def provider(symbol, start, end):
        requested.append(symbol)
        return {d: v for d, v in _history(symbol).items() if start <= d <= end}

    monkeypatch.setattr(warm_prices, "SAVE_EVERY", 2)
    cache = {}
    fetched, skipped, failed = warm_prices.warm([], SUNDAY, cache, fetcher=provider, claims=claims,
                                                save=lambda c: saves.append(len(c)), deadline_minutes=3,
                                                clock=lambda: next(ticks))
    assert requested == ["spy.us", "btcusd"]                    # the as-of anchor and the crypto benchmark first
    assert fetched == 2 and failed == 0                         # no request starts after 3 minutes
    assert saves == [2]                                         # saved after every SAVE_EVERY symbols
    assert set(cache) == {"btcusd", "spy.us"}


def test_save_keeps_history_a_needed_range_still_starts_at(tmp_path):
    path = str(tmp_path / "prices.json")
    today = date(2026, 8, 17)
    old = today - timedelta(days=price_cache.MAX_HISTORY_DAYS + 10)
    cache = {}
    for symbol in ("nvda.us", "aapl.us"):
        price_cache.remember(cache, symbol, old, today, {old: 1.0, today: 2.0})
    price_cache.save(cache, path, today=today, keep_from={"nvda.us": old})
    loaded = price_cache.load(path)
    assert list(loaded["nvda.us"]["closes"]) == [old, today] and loaded["nvda.us"]["from"] == old
    assert list(loaded["aapl.us"]["closes"]) == [today]          # nothing needs it: trimmed as before
