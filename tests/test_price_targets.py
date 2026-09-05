"""Hardening item 5: price-target success is defined per method, with
intraday touches from daily bars and unknown (not false) without them."""
from datetime import date, timedelta

import channel_scorecard as cs
import price_cache
import research_analytics as ra
import scorecard_pricing as sp

from tests.test_scorecard_pricing import _claim, _fetch

START = date(2026, 7, 1)


def _days(n):
    out, d = [], START
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


DAYS = _days(30)
# Closes never reach 110; the high on day 6 does; the close on day 8 does
# not; the horizon close (day 12) is back at 104.
CLOSES = {d: 100.0 + (i % 5) for i, d in enumerate(DAYS)}
HIGHS = {d: CLOSES[d] + (12.0 if i == 6 else 1.0) for i, d in enumerate(DAYS)}
LOWS = {d: CLOSES[d] - (7.0 if i == 9 else 1.0) for i, d in enumerate(DAYS)}


def _ohlc(symbol="nvda.us"):
    return sp.PriceSeries(symbol, dict(CLOSES), provider="test", adjustment="split_adjusted",
                          corporate_action_status="split_adjusted", highs=dict(HIGHS), lows=dict(LOWS))


def test_intraday_touch_daily_close_and_horizon_close_are_separate_verdicts():
    entry, end = DAYS[1], DAYS[12]
    rec = sp.evaluate_target(_ohlc(), 110.0, "bullish", entry, end)
    assert rec["target_test_method"] == "intraday_touch" and rec["price_data_granularity"] == "daily_ohlc"
    assert rec["target_reached"] is True and rec["target_reached_within_window"] is True
    assert rec["target_first_reached_date"] == DAYS[6].isoformat()
    assert rec["intraday_target_reached"] is True and rec["intraday_first_reached_date"] == DAYS[6].isoformat()
    assert rec["daily_close_target_reached"] is False and rec["daily_close_first_reached_date"] is None
    assert rec["horizon_close_target_met"] is False   # the horizon close moved away: still reached
    # Bearish targets read the lows.
    rec = sp.evaluate_target(_ohlc(), 97.0, "bearish", entry, end)   # day 9 low is 97, no close is
    assert rec["target_reached"] and rec["target_first_reached_date"] == DAYS[9].isoformat()


def test_closes_only_series_reports_intraday_reach_as_unknown_not_false():
    closes_only = sp.PriceSeries("nvda.us", dict(CLOSES), provider="test", adjustment="split_adjusted")
    assert closes_only.granularity == "daily_close"
    rec = sp.evaluate_target(closes_only, 110.0, "bullish", DAYS[1], DAYS[12])
    assert rec["target_test_method"] == "daily_close" and rec["target_reached"] is False
    assert rec["intraday_target_reached"] is None and rec["intraday_first_reached_date"] is None
    assert rec["daily_close_target_reached"] is False and rec["horizon_close_target_met"] is False
    # A close that reaches the target is a daily_close reach with its date.
    reached = dict(CLOSES)
    reached[DAYS[4]] = 111.0
    rec = sp.evaluate_target(sp.PriceSeries("x", reached, adjustment="split_adjusted"), 110.0, "bullish",
                             DAYS[1], DAYS[12])
    assert rec["target_reached"] and rec["target_first_reached_date"] == DAYS[4].isoformat()
    assert rec["horizon_close_target_met"] is False


def test_scorecard_records_the_target_method_per_claim():
    claim = _claim("2026-07-02T14:00:00+00:00", end=DAYS[12].isoformat(), target_kind="absolute_value",
                   target_value=110)

    def fetch(symbol, start, end):
        return _ohlc(symbol) if symbol == "nvda.us" else sp.PriceSeries(symbol, dict(CLOSES), provider="test",
                                                                       adjustment="split_adjusted")
    sc = ra.scorecard([claim], date(2026, 9, 1), fetch, min_sample=1)
    scored = sc["A"]["scored"][0]
    assert scored["target_test_method"] == "intraday_touch" and scored["target_reached"] is True
    assert scored["horizon_close_target_met"] is False and scored["price_data_granularity"] == "daily_ohlc"
    assert sc["A"]["target_hits"] == 1 and sc["A"]["target_test_methods"] == {"intraday_touch": 1}
    # The same claim with closes only: daily_close, and intraday unknown.
    sc = ra.scorecard([claim], date(2026, 9, 1), _fetch({"nvda.us": CLOSES, "xlk.us": CLOSES}), min_sample=1)
    scored = sc["A"]["scored"][0]
    assert scored["target_test_method"] == "daily_close" and scored["target_reached"] is False
    assert scored["intraday_target_reached"] is None and sc["A"]["target_test_methods"] == {"daily_close": 1}
    assert "intraday_touch" in ra.format_scorecard_lines({"A": sc["A"]}, {}, 1, False, []) or \
        "daily_close 1" in ra.format_scorecard_lines({"A": sc["A"]}, {}, 1, False, [])


def test_provider_bars_flow_through_the_cache_into_the_series(monkeypatch, tmp_path):
    payload = {"status": "ok", "values": [
        {"datetime": "2026-08-14", "close": "180.5", "high": "185.0", "low": "178.0"},
        {"datetime": "2026-08-17", "close": "182.25", "high": "183.0", "low": "181.0"},
    ]}
    bars = cs._parse_twelvedata(payload, "NVDA")
    assert dict(bars) == {date(2026, 8, 14): 180.5, date(2026, 8, 17): 182.25}
    assert bars.highs[date(2026, 8, 14)] == 185.0 and bars.lows[date(2026, 8, 17)] == 181.0
    monkeypatch.setattr(cs, "fetch_prices_live", lambda s, a, b: bars)
    series = cs.fetch_price_series("nvda.us", date(2026, 8, 10), date(2026, 8, 18))
    assert series.granularity == "daily_ohlc" and series.highs[date(2026, 8, 14)] == 185.0
    assert series.provenance()["price_data_granularity"] == "daily_ohlc"
    # Bars survive a save/load of the cache; entries without them stay closes-only.
    path = str(tmp_path / "prices.json")
    price_cache.save(price_cache.active(), path, today=date(2026, 8, 18))
    loaded = price_cache.load(path)
    assert loaded["nvda.us"]["highs"][date(2026, 8, 14)] == 185.0
    price_cache.remember(loaded, "old.us", date(2026, 8, 10), date(2026, 8, 18), {date(2026, 8, 14): 1.0})
    price_cache.reset(loaded)
    old = cs.fetch_price_series("old.us", date(2026, 8, 10), date(2026, 8, 18))
    assert old.granularity == "daily_close" and old.highs is None
