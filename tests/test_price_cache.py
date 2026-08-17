"""Tests for the persistent daily-close cache and the Sunday warm job."""
import json
from datetime import date, timedelta

import pytest

import channel_scorecard as cs
import price_cache
import warm_prices


def _rec(day, channel="Chan", assets=None):
    return {
        "date": day, "channel_name": channel,
        "signals": {"assets": assets or [], "market_sentiment": "neutral", "topics": []},
    }


def _asset(name="Nvidia", ticker="NVDA", stance="bullish", price_target=None):
    return {"name": name, "ticker": ticker, "stance": stance, "action": "none",
            "price_target": price_target, "catalysts": [], "type": "stock",
            "conviction": "medium", "horizon": "unspecified"}


def test_roundtrip_survives_save_and_load(tmp_path):
    path = str(tmp_path / "prices.json")
    cache = {}
    price_cache.remember(cache, "nvda.us", date(2026, 8, 10), date(2026, 8, 16),
                         {date(2026, 8, 14): 180.5})
    assert price_cache.save(cache, path, today=date(2026, 8, 17))
    loaded = price_cache.load(path)
    assert loaded["nvda.us"]["closes"] == {date(2026, 8, 14): 180.5}
    assert loaded["nvda.us"]["from"] == date(2026, 8, 10)
    assert loaded["nvda.us"]["to"] == date(2026, 8, 16)


def test_load_tolerates_missing_and_corrupt_files(tmp_path):
    assert price_cache.load(str(tmp_path / "absent.json")) == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert price_cache.load(str(bad)) == {}
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"version": 999, "symbols": {}}), encoding="utf-8")
    assert price_cache.load(str(wrong)) == {}


def test_save_trims_history_beyond_the_bound(tmp_path):
    path = str(tmp_path / "prices.json")
    today = date(2026, 8, 17)
    old = today - timedelta(days=price_cache.MAX_HISTORY_DAYS + 10)
    cache = {}
    price_cache.remember(cache, "nvda.us", old, today,
                         {old: 1.0, today: 2.0})
    price_cache.save(cache, path, today=today)
    loaded = price_cache.load(path)
    assert list(loaded["nvda.us"]["closes"]) == [today]  # the stale close is gone


def test_covered_needs_the_whole_requested_range():
    entry = {"from": date(2026, 8, 10), "to": date(2026, 8, 16), "closes": {}}
    assert price_cache.covered(entry, date(2026, 8, 11), date(2026, 8, 15))
    assert not price_cache.covered(entry, date(2026, 8, 9), date(2026, 8, 15))
    assert not price_cache.covered(entry, date(2026, 8, 11), date(2026, 8, 17))
    assert not price_cache.covered(None, date(2026, 8, 11), date(2026, 8, 15))


def test_covered_range_is_served_without_touching_the_network(monkeypatch):
    cache = price_cache.active()
    price_cache.remember(cache, "nvda.us", date(2026, 8, 1), date(2026, 8, 16),
                         {date(2026, 8, 14): 180.5, date(2026, 7, 1): 99.0})
    monkeypatch.setattr(cs, "fetch_prices_live",
                        lambda *a: pytest.fail("cached range must not refetch"))
    prices = cs.fetch_prices("nvda.us", date(2026, 8, 10), date(2026, 8, 16))
    assert prices == {date(2026, 8, 14): 180.5}  # sliced to the request


def test_uncovered_range_fetches_and_widens_the_cache(monkeypatch):
    calls = []

    def fake_live(symbol, start, end):
        calls.append((symbol, start, end))
        return {date(2026, 8, 14): 180.5}

    monkeypatch.setattr(cs, "fetch_prices_live", fake_live)
    cs.fetch_prices("nvda.us", date(2026, 8, 10), date(2026, 8, 16))
    assert len(calls) == 1
    # The same range is now served from the cache.
    cs.fetch_prices("nvda.us", date(2026, 8, 10), date(2026, 8, 16))
    assert len(calls) == 1


def test_partial_cache_is_returned_when_the_live_fetch_fails(monkeypatch):
    """Some history scores more calls than none — a dead provider must not
    throw away what was already cached."""
    cache = price_cache.active()
    price_cache.remember(cache, "nvda.us", date(2026, 8, 10), date(2026, 8, 14),
                         {date(2026, 8, 13): 175.0})
    monkeypatch.setattr(cs, "fetch_prices_live", lambda *a: {})
    prices = cs.fetch_prices("nvda.us", date(2026, 8, 10), date(2026, 8, 16))
    assert prices == {date(2026, 8, 13): 175.0}


def test_needed_ranges_spans_horizon_and_caps_at_today():
    records = [_rec("2026-08-14", assets=[_asset()])]
    today = date(2026, 8, 17)
    ranges = warm_prices.needed_ranges(records, today)
    start, end = ranges["nvda.us"]
    assert start == date(2026, 8, 14)
    assert end == today  # the 30-day horizon hasn't elapsed; don't ask for the future


def test_needed_ranges_includes_current_window_target_symbols():
    records = [_rec("2026-08-16", assets=[_asset("Reddit", "RDDT", "neutral",
                                                 price_target=203)])]
    ranges = warm_prices.needed_ranges(records, date(2026, 8, 17))
    # Neutral, so it is not a directional call — it is here for the price target.
    assert "rddt.us" in ranges


def test_warm_skips_covered_symbols_and_counts_outcomes():
    records = [
        _rec("2026-07-01", assets=[_asset("Nvidia", "NVDA")]),
        _rec("2026-07-01", assets=[_asset("Apple", "AAPL")]),
    ]
    today = date(2026, 8, 17)
    cache = {}
    # NVDA's full needed range is already known; AAPL's is not.
    nvda_start, nvda_end = warm_prices.needed_ranges(records, today)["nvda.us"]
    price_cache.remember(cache, "nvda.us", nvda_start, nvda_end, {nvda_start: 1.0})

    asked = []

    def fetcher(symbol, start, end):
        asked.append(symbol)
        return {start: 2.0}

    fetched, skipped, failed = warm_prices.warm(records, today, cache, fetcher=fetcher)
    assert (fetched, skipped, failed) == (1, 1, 0)
    assert asked == ["aapl.us"]  # the covered symbol was never requested


def test_warm_counts_unavailable_symbols_without_caching_them():
    records = [_rec("2026-07-01", assets=[_asset()])]
    cache = {}
    fetched, skipped, failed = warm_prices.warm(
        records, date(2026, 8, 17), cache, fetcher=lambda *a: {})
    assert (fetched, skipped, failed) == (0, 0, 1)
    assert cache == {}  # a failed fetch must not record a false "covered"


def test_warm_second_run_is_free_once_horizons_have_elapsed():
    """The point of the cache: a fully elapsed call is fetched once, ever."""
    records = [_rec("2026-06-01", assets=[_asset()])]
    today = date(2026, 8, 17)  # well past the 30-day horizon
    cache = {}
    calls = []

    def fetcher(symbol, start, end):
        calls.append(symbol)
        return {start: 1.0}

    warm_prices.warm(records, today, cache, fetcher=fetcher)
    warm_prices.warm(records, today, cache, fetcher=fetcher)
    assert len(calls) == 1
