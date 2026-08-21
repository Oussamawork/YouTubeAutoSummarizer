"""Tests for the per-channel accuracy scorecard (prices mocked, no network)."""
import json
from datetime import date, timedelta

import pytest

import channel_scorecard as cs


def _rec(day, channel, assets):
    return {
        "date": day,
        "channel_name": channel,
        "signals": {"assets": assets, "market_sentiment": "neutral", "topics": []},
    }


def _asset(ticker="TSLA", stance="bullish", asset_type="stock"):
    return {"name": ticker, "ticker": ticker, "stance": stance, "type": asset_type,
            "action": "none", "conviction": "medium", "horizon": "unspecified",
            "catalysts": [], "price_target": None}


def test_symbol_mapping():
    assert cs.symbol_for(_asset("TSLA", asset_type="stock")) == "tsla.us"
    assert cs.symbol_for(_asset("SPY", asset_type="etf")) == "spy.us"
    assert cs.symbol_for(_asset("BTC", asset_type="crypto")) == "btcusd"
    assert cs.symbol_for(_asset("SPX", asset_type="index")) is None
    assert cs.symbol_for({"ticker": None, "type": "stock"}) is None


def test_parse_stooq_csv():
    csv = "Date,Open,High,Low,Close,Volume\n2026-07-01,10,11,9,10.5,1000\nbad,line\n2026-07-02,10.5,12,10,11.0,900\n"
    prices = cs._parse_stooq_csv(csv, "x.us")
    assert prices == {date(2026, 7, 1): 10.5, date(2026, 7, 2): 11.0}
    assert cs._parse_stooq_csv("No data", "x.us") == {}
    assert cs._parse_stooq_csv("", "x.us") == {}


def test_price_on_or_after_skips_weekend():
    prices = {date(2026, 7, 6): 100.0}  # Monday
    assert cs.price_on_or_after(prices, date(2026, 7, 4)) == 100.0  # Saturday signal
    assert cs.price_on_or_after(prices, date(2026, 7, 10)) is None  # beyond lag


def _fetcher(series):
    return lambda symbol, start, end: series.get(symbol, {})


def test_evaluate_bullish_hit_and_bearish_hit():
    today = date(2026, 7, 24)
    d0 = date(2026, 7, 10)
    series = {"tsla.us": {d0: 100.0, d0 + timedelta(days=7): 110.0},
              "btcusd": {d0: 50000.0, d0 + timedelta(days=7): 45000.0}}
    records = [
        _rec("2026-07-10", "Bulls", [_asset("TSLA", "bullish")]),
        _rec("2026-07-10", "Bears", [_asset("BTC", "bearish", "crypto")]),
    ]
    stats = cs.evaluate(records, today, price_fetcher=_fetcher(series))
    assert stats["Bulls"][7]["hits"] == 1 and stats["Bulls"][7]["total"] == 1
    assert stats["Bulls"][7]["returns"] == pytest.approx([0.1])
    bears = stats["Bears"][7]
    assert bears["hits"] == 1 and bears["total"] == 1
    assert bears["returns"][0] > 0  # bearish call, price fell -> positive directional return
    # 30-day horizon has not elapsed -> no entries
    assert stats["Bulls"][30]["total"] == 0


def test_evaluate_miss_and_unelapsed_horizon():
    today = date(2026, 7, 24)
    d0 = date(2026, 7, 10)
    series = {"tsla.us": {d0: 100.0, d0 + timedelta(days=7): 90.0}}
    records = [
        _rec("2026-07-10", "Chan", [_asset("TSLA", "bullish")]),
        _rec("2026-07-23", "Chan", [_asset("TSLA", "bullish")]),  # 7d not elapsed
    ]
    stats = cs.evaluate(records, today, price_fetcher=_fetcher(series))
    assert stats["Chan"][7]["hits"] == 0 and stats["Chan"][7]["total"] == 1
    assert stats["Chan"][7]["returns"] == pytest.approx([-0.1])


def test_evaluate_skips_neutral_and_missing_prices():
    today = date(2026, 7, 24)
    records = [
        _rec("2026-07-10", "Chan", [_asset("TSLA", "neutral")]),
        _rec("2026-07-10", "Chan", [_asset("NOPE", "bullish")]),
    ]
    stats = cs.evaluate(records, today, price_fetcher=_fetcher({}))
    assert stats == {} or stats["Chan"][7]["total"] == 0


def test_build_scorecard_report():
    stats = {
        "Couch Investor": {7: {"hits": 2, "total": 3, "returns": [0.05, 0.02, -0.01]},
                           30: {"hits": 1, "total": 1, "returns": [0.2]}},
        "More Crypto Online": {7: {"hits": 0, "total": 2, "returns": [-0.02, -0.02]},
                               30: {"hits": 0, "total": 0, "returns": []}},
    }
    text = cs.build_scorecard(stats, date(2026, 7, 24))
    assert "Channel Scorecard" in text
    assert "1. Couch Investor" in text
    assert "   after 1 week: 2 of 3 right (67%) · typical move +2.0%" in text
    assert "   after 1 month: 1 of 1 right (100%) · typical move +20.0%" in text
    assert "2. More Crypto Online" in text
    assert "   after 1 week: 0 of 2 right (0%) · typical move -2.0%" in text
    # A horizon with nothing scored is omitted rather than shown as 0 of 0.
    assert text.count("   after 1 month") == 1
    assert cs.LEGEND in text
    assert cs.DISCLAIMER in text
    # Ranked: Couch Investor (67%) above More Crypto Online (0%)
    assert text.index("Couch Investor") < text.index("More Crypto Online")


def test_build_scorecard_counts_calls_per_horizon_not_doubled():
    stats = {
        "A": {7: {"hits": 2, "total": 4, "returns": [0.1, 0.1, -0.1, -0.1]},
              30: {"hits": 1, "total": 2, "returns": [0.1, -0.1]}},
        "B": {7: {"hits": 1, "total": 3, "returns": [0.1, -0.1, -0.1]},
              30: {"hits": 0, "total": 1, "returns": [-0.1]}},
    }
    text = cs.build_scorecard(stats, date(2026, 7, 24))
    # 7 calls reached the 1-week mark and 3 of those also reached 1 month;
    # the header must not report their sum (10) as if they were distinct calls.
    assert "Calls scored: 7 after 1 week, 3 after 1 month" in text
    assert "10" not in text.splitlines()[1]


def test_build_scorecard_flags_small_samples():
    stats = {
        "Thin": {7: {"hits": 3, "total": 4, "returns": [0.1, 0.1, 0.1, -0.1]},
                 30: {"hits": 0, "total": 0, "returns": []}},
        "Thick": {7: {"hits": 6, "total": 12, "returns": [0.1] * 6 + [-0.1] * 6},
                  30: {"hits": 0, "total": 0, "returns": []}},
    }
    text = cs.build_scorecard(stats, date(2026, 7, 24))
    assert "1. Thin  (small sample)" in text
    assert "2. Thick\n" in text
    # Flagged, not demoted: 75% on 4 calls still outranks 50% on 12.
    assert text.index("Thin") < text.index("Thick")


def test_horizon_label_falls_back_for_unknown_horizons():
    assert cs.horizon_label(7) == "after 1 week"
    assert cs.horizon_label(30) == "after 1 month"
    assert cs.horizon_label(90) == "after 90 days"


def test_build_scorecard_empty():
    assert cs.build_scorecard({}, date(2026, 7, 24)) == ""


def test_generate_scorecard_waits_for_a_week_of_data(tmp_path):
    path = tmp_path / "signals.jsonl"
    path.write_text(json.dumps(_rec("2026-07-22", "Chan", [_asset()])) + "\n", encoding="utf-8")
    out = cs.generate_scorecard(today=date(2026, 7, 24), path=str(path),
                                price_fetcher=_fetcher({}))
    assert out == ""  # dataset only 2 days old


def test_generate_scorecard_end_to_end(tmp_path):
    d0 = date(2026, 7, 10)
    path = tmp_path / "signals.jsonl"
    path.write_text(json.dumps(_rec("2026-07-10", "Chan", [_asset("TSLA", "bullish")])) + "\n",
                    encoding="utf-8")
    series = {"tsla.us": {d0: 100.0, d0 + timedelta(days=7): 105.0}}
    out = cs.generate_scorecard(today=date(2026, 7, 24), path=str(path),
                                price_fetcher=_fetcher(series))
    assert "1. Chan  (small sample)" in out
    assert "   after 1 week: 1 of 1 right (100%) · typical move +5.0%" in out


def test_symbol_for_scores_ticker_less_calls_via_alias():
    # A third of directional calls arrived ticker-less (speakers say "Chevron",
    # not "CVX") and were invisible to the scorecard.
    assert cs.symbol_for({"ticker": None, "name": "Chevron", "type": "stock"}) == "cvx.us"
    assert cs.symbol_for({"ticker": None, "name": "Bitcoin", "type": "crypto"}) == "btcusd"
    # No alias and no ticker -> still honestly unpriceable.
    assert cs.symbol_for({"ticker": None, "name": "SpaceX", "type": "stock"}) is None


# --- Twelve Data price source -------------------------------------------------

class _Resp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def test_twelvedata_symbol_translation():
    assert cs.twelvedata_symbol("nvda.us") == "NVDA"
    assert cs.twelvedata_symbol("btcusd") == "BTC/USD"
    assert cs.twelvedata_symbol("^spx") is None  # indices stay unpriceable


def test_twelvedata_key_accepts_both_names(monkeypatch):
    monkeypatch.delenv("TWELVEDATA_API_KEY", raising=False)
    monkeypatch.setenv("TWELVEDATA_API", " abc ")
    assert cs.twelvedata_key() == "abc"
    monkeypatch.delenv("TWELVEDATA_API", raising=False)
    monkeypatch.setenv("TWELVEDATA_API_KEY", "def")
    assert cs.twelvedata_key() == "def"


def test_fetch_prices_uses_twelvedata_when_key_set(monkeypatch):
    monkeypatch.setenv("TWELVEDATA_API", "tok")
    monkeypatch.setattr(cs.time, "sleep", lambda s: None)
    seen = {}

    def fake_get(url, params=None, timeout=None):
        seen.update({"url": url, "params": params})
        return _Resp(payload={"status": "ok", "values": [
            {"datetime": "2026-08-14", "close": "180.5"},
            {"datetime": "2026-08-15", "close": "182.25"},
        ]})

    monkeypatch.setattr(cs.requests, "get", fake_get)
    monkeypatch.setattr(cs, "fetch_prices_stooq",
                        lambda *a: pytest.fail("Stooq must not be called with a key set"))
    prices = cs.fetch_prices("nvda.us", date(2026, 8, 10), date(2026, 8, 16))
    assert prices == {date(2026, 8, 14): 180.5, date(2026, 8, 15): 182.25}
    assert seen["params"]["symbol"] == "NVDA"
    assert seen["params"]["apikey"] == "tok"


def test_fetch_prices_falls_back_to_stooq_without_key(monkeypatch):
    monkeypatch.delenv("TWELVEDATA_API", raising=False)
    monkeypatch.delenv("TWELVEDATA_API_KEY", raising=False)
    monkeypatch.setattr(cs, "fetch_prices_stooq",
                        lambda symbol, start, end: {date(2026, 8, 14): 1.0})
    assert cs.fetch_prices("nvda.us", date(2026, 8, 10), date(2026, 8, 16))


def test_twelvedata_reports_error_body_and_yields_nothing(monkeypatch):
    """The API answers 200 with status=error; that must not parse as prices."""
    monkeypatch.setenv("TWELVEDATA_API", "tok")
    monkeypatch.setattr(cs.time, "sleep", lambda s: None)
    monkeypatch.setattr(cs.requests, "get", lambda *a, **k: _Resp(
        payload={"status": "error", "code": 404, "message": "symbol not found"}))
    assert cs.fetch_prices("zzzz.us", date(2026, 8, 10), date(2026, 8, 16)) == {}


def test_twelvedata_retries_then_gives_up_on_rate_limit(monkeypatch):
    monkeypatch.setenv("TWELVEDATA_API", "tok")
    slept, calls = [], []
    monkeypatch.setattr(cs.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(cs.requests, "get",
                        lambda *a, **k: (calls.append(1), _Resp(status_code=429))[1])
    assert cs.fetch_prices("nvda.us", date(2026, 8, 10), date(2026, 8, 16)) == {}
    assert len(calls) == cs.MAX_RETRIES  # retried, then gave up quietly
    # A 429 means the provider's minute is spent, so the wait is the rest of
    # that window — short backoffs would just retry into a closed window.
    assert slept == [cs.TWELVEDATA_RATE_LIMIT_COOLDOWN] * (cs.MAX_RETRIES - 1)


def test_twelvedata_pacer_only_sleeps_once_the_budget_is_spent(monkeypatch):
    slept = []
    monkeypatch.setattr(cs.time, "sleep", lambda s: slept.append(s))
    window = []
    for i in range(cs.TWELVEDATA_PER_MINUTE):
        cs._twelvedata_pace(now=100.0 + i, _calls=window)
    assert not slept  # under the per-minute budget, no waiting
    cs._twelvedata_pace(now=100.0 + cs.TWELVEDATA_PER_MINUTE, _calls=window)
    assert slept and slept[0] > 0  # budget spent -> waits for the window to roll


def test_request_budget_stops_fetching_after_the_cap(monkeypatch):
    """The cap bounds wall-clock time (8 req/min), so it must actually stop
    issuing requests — and say so once, not once per skipped symbol."""
    monkeypatch.setenv("TWELVEDATA_API", "tok")
    monkeypatch.setattr(cs.time, "sleep", lambda s: None)
    monkeypatch.setattr(cs, "TWELVEDATA_MAX_REQUESTS", 2)
    warned, calls = [], []
    monkeypatch.setattr(cs, "log_warn", lambda m: warned.append(m))
    monkeypatch.setattr(cs.requests, "get", lambda *a, **k: (
        calls.append(1), _Resp(payload={"status": "ok", "values": [
            {"datetime": "2026-08-14", "close": "1.0"}]}))[1])
    budget = [0]
    monkeypatch.setattr(cs, "_spend_request_budget",
                        lambda: budget[0] < 2 and (budget.__setitem__(0, budget[0] + 1) or True))

    for symbol in ("a.us", "b.us", "c.us", "d.us"):
        cs.fetch_prices(symbol, date(2026, 8, 10), date(2026, 8, 16))
    assert len(calls) == 2  # stopped at the cap rather than pacing on forever


def test_spend_request_budget_logs_the_ceiling_once(monkeypatch):
    monkeypatch.setattr(cs, "TWELVEDATA_MAX_REQUESTS", 1)
    warned = []
    monkeypatch.setattr(cs, "log_warn", lambda m: warned.append(m))
    state = [0]
    assert cs._spend_request_budget(_state=state) is True
    assert cs._spend_request_budget(_state=state) is False
    assert cs._spend_request_budget(_state=state) is False
    assert len(warned) == 1  # one line at the ceiling, not one per symbol


def test_name_recorded_as_ticker_folds_to_the_real_one():
    """Speakers say the company name and the extractor files it as the ticker.
    Unfolded, one asset splits across two buckets and prices as nothing."""
    for name, recorded, expected in [
        ("Apple", "APPLE", "aapl.us"), ("Google", "GOOGLE", "googl.us"),
        ("Nvidia", "NVIDIA", "nvda.us"), ("Nebius", "NEBL", "nbis.us"),
        ("Pan American Silver", "PAS", "paas.us"),
    ]:
        assert cs.symbol_for({"name": name, "ticker": recorded, "type": "stock"}) == expected


def test_unpriceable_tickers_are_skipped_before_any_request():
    for ticker in ("OPENAI", "ANTHROPIC", "CXMT", "WAYMO"):
        assert cs.symbol_for({"name": ticker, "ticker": ticker, "type": "stock"}) is None


def test_spacex_is_priceable_via_the_curated_ticker():
    """Verified live 2026-08-17: spcx.us returns closes, so SpaceX is no
    longer written off as private."""
    assert cs.symbol_for({"name": "SpaceX", "ticker": "SPACEX",
                          "type": "stock"}) == "spcx.us"


def test_us_symbols_disambiguate_by_country(monkeypatch):
    """A bare ticker on several exchanges is rejected with a 400 asking which
    one — NU, AMTM and ECG are all real US listings that hit exactly that."""
    monkeypatch.setenv("TWELVEDATA_API", "tok")
    monkeypatch.setattr(cs.time, "sleep", lambda s: None)
    seen = {}

    def fake_get(url, params=None, timeout=None):
        seen.update(params)
        return _Resp(payload={"status": "ok", "values": [
            {"datetime": "2026-08-14", "close": "12.0"}]})

    monkeypatch.setattr(cs.requests, "get", fake_get)
    cs.fetch_prices_live("nu.us", date(2026, 8, 10), date(2026, 8, 16))
    assert seen["country"] == "United States"

    seen.clear()
    cs.fetch_prices_live("btcusd", date(2026, 8, 10), date(2026, 8, 16))
    assert "country" not in seen  # crypto is not exchange-listed


def test_quotes_in_usd_by_venue():
    assert cs.quotes_in_usd("nvda.us")
    assert cs.quotes_in_usd("btcusd")       # crypto pairs are explicitly /USD
    assert not cs.quotes_in_usd("000660.krx")   # KRW
    assert not cs.quotes_in_usd("bas.xetra")    # EUR
    assert not cs.quotes_in_usd("688825.sse")   # CNY


def test_symbol_search_retries_rate_limits_instead_of_reporting_no_listings(monkeypatch):
    """An empty search result reads downstream as 'this company has no
    listing', so a 429 must be retried, never returned as an answer."""
    monkeypatch.setattr(cs.time, "sleep", lambda s: None)
    responses = [_Resp(status_code=429),
                 _Resp(payload={"data": [{"symbol": "RBRK",
                                          "instrument_name": "Rubrik Inc"}]})]
    monkeypatch.setattr(cs.requests, "get", lambda *a, **k: responses.pop(0))
    rows = cs.search_symbols("Rubrik", "tok")
    assert [r["symbol"] for r in rows] == ["RBRK"]  # the retry won


def test_symbol_search_raises_rather_than_claiming_no_listings(monkeypatch):
    """Giving up must be distinguishable from 'no such company', or a
    rate-limited run would cache a real company as unlisted forever."""
    monkeypatch.setattr(cs.time, "sleep", lambda s: None)
    monkeypatch.setattr(cs.requests, "get", lambda *a, **k: _Resp(status_code=429))
    with pytest.raises(cs.SearchUnavailable):
        cs.search_symbols("Rubrik", "tok")


def test_call_date_prefers_publish_time_over_run_date():
    # `date` is stamped when the summariser ran, up to a week after publication.
    assert cs._call_date({"date": "2026-08-18",
                          "published_at": "2026-08-11T22:28:04+00:00"}) == date(2026, 8, 11)
    # Both shapes the scraper writes.
    assert cs._call_date({"date": "2026-08-18",
                          "published_at": "2026-08-11T22:28:04Z"}) == date(2026, 8, 11)
    # Falls back to the run date when absent or unparseable.
    assert cs._call_date({"date": "2026-08-18"}) == date(2026, 8, 18)
    assert cs._call_date({"date": "2026-08-18", "published_at": "nonsense"}) == date(2026, 8, 18)


def test_horizon_starts_at_publication_not_at_the_run():
    today = date(2026, 7, 24)
    d_pub, d_run = date(2026, 7, 10), date(2026, 7, 13)
    # Flat after the run date, so a call scores only if it starts at publication.
    series = {"tsla.us": {d_pub: 100.0, d_run: 120.0,
                          d_pub + timedelta(days=7): 110.0,
                          d_run + timedelta(days=7): 120.0}}
    rec = _rec("2026-07-13", "Chan", [_asset("TSLA", "bullish")])
    rec["published_at"] = "2026-07-10T09:00:00+00:00"
    stats = cs.evaluate([rec], today, price_fetcher=_fetcher(series))
    assert stats["Chan"][7]["returns"] == pytest.approx([0.1])  # 100 -> 110, not 120 -> 120


def test_repeat_mentions_of_one_position_score_once():
    today = date(2026, 7, 24)
    d0 = date(2026, 7, 10)
    series = {"tsla.us": {d0: 100.0, d0 + timedelta(days=7): 110.0}}
    # The same channel repeating the same view is not three predictions.
    records = [_rec("2026-07-10", "Chan", [_asset("TSLA", "bullish")]) for _ in range(3)]
    stats = cs.evaluate(records, today, price_fetcher=_fetcher(series))
    assert stats["Chan"][7]["total"] == 1
    # A different channel's identical call is still its own call.
    records.append(_rec("2026-07-10", "Other", [_asset("TSLA", "bullish")]))
    stats = cs.evaluate(records, today, price_fetcher=_fetcher(series))
    assert stats["Chan"][7]["total"] == 1 and stats["Other"][7]["total"] == 1


def test_a_reopened_position_counts_again():
    calls = [
        (date(2026, 7, 1), "Chan", "TSLA", "tsla.us", "bullish"),
        (date(2026, 7, 5), "Chan", "TSLA", "tsla.us", "bullish"),   # still open
        (date(2026, 8, 20), "Chan", "TSLA", "tsla.us", "bullish"),  # window closed
    ]
    kept = cs._dedupe_calls(calls)
    assert [c[0] for c in kept] == [date(2026, 7, 1), date(2026, 8, 20)]
    # The opposite stance is a separate position, not a repeat.
    calls.append((date(2026, 7, 1), "Chan", "NVDA", "nvda.us", "bearish"))
    assert len(cs._dedupe_calls(calls)) == 3


def test_same_day_contradiction_is_not_a_call():
    today = date(2026, 7, 24)
    d0 = date(2026, 7, 10)
    series = {"tsla.us": {d0: 100.0, d0 + timedelta(days=7): 110.0}}
    # Saying both things on one day guarantees one hit and one miss, which drags
    # every hit rate toward 50% while looking like two real calls.
    records = [
        _rec("2026-07-10", "Chan", [_asset("TSLA", "bullish")]),
        _rec("2026-07-10", "Chan", [_asset("TSLA", "bearish")]),
    ]
    stats = cs.evaluate(records, today, price_fetcher=_fetcher(series))
    assert stats == {} or stats["Chan"][7]["total"] == 0


def test_todays_horizon_is_not_scored_against_an_open_bar():
    # The job runs at 16:00 UTC Friday, hours before the US close, so a bar
    # dated today is still in progress.
    today = date(2026, 7, 17)
    d0 = date(2026, 7, 10)
    series = {"tsla.us": {d0: 100.0, today: 110.0}}
    records = [_rec("2026-07-10", "Chan", [_asset("TSLA", "bullish")])]
    stats = cs.evaluate(records, today, price_fetcher=_fetcher(series))
    assert stats == {} or stats["Chan"][7]["total"] == 0
    # One day later the bar is closed and the call scores.
    stats = cs.evaluate(records, date(2026, 7, 18), price_fetcher=_fetcher(series))
    assert stats["Chan"][7]["total"] == 1


def test_frozen_series_is_unscored_rather_than_a_miss():
    today = date(2026, 7, 24)
    d0 = date(2026, 7, 10)
    # A delisted/frozen listing never moves; every call on it would return 0.0
    # and be counted as a miss.
    series = {"tsla.us": {d0: 0.862, d0 + timedelta(days=7): 0.862}}
    records = [_rec("2026-07-10", "Chan", [_asset("TSLA", "bullish")])]
    stats = cs.evaluate(records, today, price_fetcher=_fetcher(series))
    assert stats == {} or stats["Chan"][7]["total"] == 0


def test_baseline_counts_each_symbol_day_once():
    today = date(2026, 7, 24)
    d0 = date(2026, 7, 10)
    later = d0 + timedelta(days=7)
    series = {"tsla.us": {d0: 100.0, later: 110.0},   # rose
              "nvda.us": {d0: 100.0, later: 90.0}}    # fell
    records = [
        # Three channels on the same rising name must not make the market look
        # like it rose three times.
        _rec("2026-07-10", "A", [_asset("TSLA", "bullish")]),
        _rec("2026-07-10", "B", [_asset("TSLA", "bullish")]),
        _rec("2026-07-10", "C", [_asset("TSLA", "bearish")]),
        _rec("2026-07-10", "D", [_asset("NVDA", "bullish")]),
    ]
    _, baseline = cs.score_calls(records, today, price_fetcher=_fetcher(series))
    assert baseline == (0.5, 2)


def test_scorecard_shows_the_baseline_line():
    stats = {"A": {7: {"hits": 6, "total": 10, "returns": [0.01] * 6 + [-0.01] * 4},
                   30: {"hits": 0, "total": 0, "returns": []}}}
    text = cs.build_scorecard(stats, date(2026, 7, 24), baseline=(0.669, 228))
    assert "For scale: 67% of these 7-day windows rose on their own" in text
    assert "(228 windows)" in text
    # Absent baseline simply omits the line.
    assert "For scale" not in cs.build_scorecard(stats, date(2026, 7, 24))


def test_typical_move_is_the_median_not_the_mean():
    # One outlier must not set the headline number: mean here is +10.9%.
    stats = {"A": {7: {"hits": 4, "total": 5,
                       "returns": [0.02, 0.03, 0.01, -0.01, 0.49]},
                   30: {"hits": 0, "total": 0, "returns": []}}}
    text = cs.build_scorecard(stats, date(2026, 7, 24))
    assert "typical move +2.0%" in text
