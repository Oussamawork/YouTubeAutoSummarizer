"""Tests for the Friday report — creators vs the market, its charts and the
strict scorecard — and the legacy per-channel scorecard (prices mocked or
read from a test cache, never the network)."""
import json
import os
from datetime import date, timedelta

import pytest

import canonical_claims
import channel_scorecard as cs
import creator_tracker
import price_cache
import pulse_charts
import scorecard_pricing as sp
from tests.test_canonical_analytics import _claim


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
    assert stats["Bulls"][7] == {"hits": 1, "total": 1, "dir_return_sum": 0.1}
    bears = stats["Bears"][7]
    assert bears["hits"] == 1 and bears["total"] == 1
    assert bears["dir_return_sum"] > 0  # bearish call, price fell -> positive directional return
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
    assert stats["Chan"][7] == {"hits": 0, "total": 1, "dir_return_sum": -0.1}


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
        "Couch Investor": {7: {"hits": 2, "total": 3, "dir_return_sum": 0.06},
                           30: {"hits": 1, "total": 1, "dir_return_sum": 0.2}},
        "More Crypto Online": {7: {"hits": 0, "total": 2, "dir_return_sum": -0.04},
                               30: {"hits": 0, "total": 0, "dir_return_sum": 0.0}},
    }
    text = cs.build_scorecard(stats, date(2026, 7, 24))
    assert "Channel Scorecard" in text
    assert "Couch Investor — 7d: 2/3 (67%) avg +2.0% — 30d: 1/1 (100%) avg +20.0%" in text
    assert "More Crypto Online — 7d: 0/2 (0%) avg -2.0%" in text
    # The legacy report measures a fixed-horizon hit rate and keeps its own footer.
    assert cs.LEGACY_DISCLAIMER in text and cs.DISCLAIMER not in text
    # Ranked: Couch Investor (67%) above More Crypto Online (0%)
    assert text.index("Couch Investor") < text.index("More Crypto Online")


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
    assert "Chan — 7d: 1/1 (100%) avg +5.0%" in out


def test_symbol_for_scores_ticker_less_calls_via_alias():
    # A third of directional calls arrived ticker-less (speakers say "Chevron",
    # not "CVX") and were invisible to the scorecard.
    assert cs.symbol_for({"ticker": None, "name": "Chevron", "type": "stock"}) == "cvx.us"
    assert cs.symbol_for({"ticker": None, "name": "Bitcoin", "type": "crypto"}) == "btcusd"
    # SpaceX by name now resolves through the curated name alias (SPACEX -> SPCX).
    assert cs.symbol_for({"ticker": None, "name": "SpaceX", "type": "stock"}) == "spcx.us"
    # No alias and no ticker -> still honestly unpriceable.
    assert cs.symbol_for({"ticker": None, "name": "Zorblax Industries", "type": "stock"}) is None


def test_symbol_for_is_registry_first():
    """The instrument registry outranks the row's model-written type: a
    'stock' Bitcoin is the coin, never btc.us (the Grayscale Bitcoin Mini
    Trust), and a SpaceX typed private/other is the SPCX listing."""
    assert cs.symbol_for({"name": "Bitcoin", "ticker": "BTC", "type": "stock"}) == "btcusd"
    assert cs.symbol_for({"name": "Ethereum", "ticker": None, "type": "stock"}) == "ethusd"
    assert cs.symbol_for({"name": "SpaceX", "ticker": "SPCX", "type": "private_company"}) == "spcx.us"
    assert cs.symbol_for({"name": "SpaceX", "ticker": None, "type": "other"}) == "spcx.us"
    # An unregistered ticker keeps the type rule ...
    assert cs.symbol_for({"name": "Zorblax", "ticker": "ZRBX", "type": "stock"}) == "zrbx.us"
    assert cs.symbol_for({"name": "Zorblax", "ticker": "ZRBX", "type": "other"}) is None
    # ... and an index (or a market theme) is never priced from its type.
    assert cs.symbol_for({"name": "S&P 500", "ticker": "SPX", "type": "index"}) is None
    # A foreign listing code typed stock would 404 as '.us' on every warm run.
    assert cs.symbol_for({"name": "Foreign Co", "ticker": "123456", "type": "stock"}) is None
    assert cs.symbol_for({"name": "Foreign Co", "ticker": "ABC.DE", "type": "stock"}) is None
    assert cs.symbol_for({"name": "Share class", "ticker": "ZRBX.B", "type": "stock"}) == "zrbx.b.us"
    # Unpriceable decisions still win.
    assert cs.symbol_for({"name": "OpenAI", "ticker": "OPENAI", "type": "other"}) is None


def test_symbol_for_applies_the_channels_own_spelling(monkeypatch):
    """A legacy row is priced with its channel's own spelling first, as
    canonical claims are (Couch Investor's "New" is Nu Holdings)."""
    import claims as claims_module
    monkeypatch.setitem(claims_module.CHANNEL_ALIASES, "Zorb Channel", {"ZORB": "NVDA"})
    row = {"name": "Zorb", "ticker": None, "type": "stock", "stance": "bullish"}
    assert cs.symbol_for(row, "Zorb Channel") == "nvda.us"
    assert cs.symbol_for(row) is None and cs.symbol_for(row, "Another Channel") is None
    calls = list(cs._directional_calls([_rec("2026-09-01", "Zorb Channel", [row])]))
    assert [symbol for _, _, _, symbol, _ in calls] == ["nvda.us"]


def test_symbol_for_lets_an_explicit_type_pick_the_namespace_of_a_colliding_ticker(monkeypatch):
    """Only for a ticker that is both a coin and a US listing does the type
    choose (canonical_claims.COLLIDING_TICKERS, as in priceable_symbol)."""
    ticker = next(t for t in sorted(canonical_claims.COLLIDING_TICKERS)
                  if (inst := cs.instruments.REGISTRY.get(t)) is not None and inst.asset_type == "crypto")
    assert cs.symbol_for({"name": ticker, "ticker": ticker, "type": "crypto"}) == f"{ticker.lower()}usd"
    assert cs.symbol_for({"name": ticker, "ticker": ticker, "type": None}) == f"{ticker.lower()}usd"
    assert cs.symbol_for({"name": ticker, "ticker": ticker, "type": "stock"}) == f"{ticker.lower()}.us"


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
    prices = cs.fetch_prices("nvda.us", date(2026, 8, 10), date(2026, 8, 16))
    assert prices == {date(2026, 8, 14): 180.5, date(2026, 8, 15): 182.25}
    assert seen["params"]["symbol"] == "NVDA"
    assert seen["params"]["apikey"] == "tok"


def test_fetch_prices_without_key_is_empty_and_never_hits_the_network(monkeypatch):
    monkeypatch.delenv("TWELVEDATA_API", raising=False)
    monkeypatch.delenv("TWELVEDATA_API_KEY", raising=False)
    monkeypatch.setattr(cs.requests, "get",
                        lambda *a, **k: pytest.fail("no provider without a key"))
    assert cs.fetch_prices("nvda.us", date(2026, 8, 10), date(2026, 8, 16)) == {}


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
    """SpaceX is the SPCX listing, by evidence in the data rather than by a
    series merely existing: creators titled their videos "SPCX Stock
    Analysis", and on 2026-09-14 one said SpaceX was "now trading at $150 per
    share, about 10% above its IPO price" against a cached spcx.us close of
    148.15 that day."""
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


# --- The Friday report: creators vs the market, charts, strict scorecard ------

@pytest.fixture(autouse=True)
def _charts_in_tmp(tmp_path, monkeypatch):
    """main() writes chart PNGs to CHARTS_DIR by default: never into the checkout."""
    monkeypatch.setattr(cs, "CHARTS_DIR", str(tmp_path / "default-charts"))


US = sp.EXCHANGES["us"]
FRIDAY = date(2026, 10, 16)                  # the weekly-scorecard.yml run (16:00 UTC, mid-session)
THURSDAY = date(2026, 10, 15)                # creator_tracker.default_as_of(FRIDAY)
CREATORS = (("Alpha", "2026-09-08"), ("Beta", "2026-09-10"), ("Gamma", "2026-09-14"))


def _sessions(start, end):
    days, day = [], start
    while day <= end:
        if sp.is_trading_day(US, day):
            days.append(day)
        day += timedelta(days=1)
    return days


def _line(first, step, end=THURSDAY):
    return {d: round(first + i * step, 4) for i, d in enumerate(_sessions(date(2026, 8, 24), end))}


def _install_cache(**series):
    """A process-wide price cache holding these {symbol: {date: close}} series."""
    price_cache.reset({s: {"from": min(c), "to": max(c), "closes": dict(c), "adjustment": "split_adjusted"}
                       for s, c in series.items()})


def _market_cache(end=THURSDAY):
    _install_cache(**{"spy.us": _line(500, 1, end), "nvda.us": _line(100, 2, end),
                      "aapl.us": _line(200, -1, end), "msft.us": _line(300, 0.5, end)})


def _calls():
    """Three creators, each bullish on NVDA, AAPL and MSFT in one video:
    nine calls, three per creator, three assets all three agree on."""
    return [_claim(creator, asset, "bullish", "unspecified", published=day, video=f"{creator}-{day}", n=n)
            for creator, day in CREATORS for n, asset in enumerate(("NVDA", "AAPL", "MSFT"))]


def _no_live_prices(monkeypatch):
    monkeypatch.setattr(cs, "fetch_prices_live",
                        lambda *a, **k: pytest.fail("the Friday report must not fetch prices live here"))
    monkeypatch.setattr(cs.requests, "get", lambda *a, **k: pytest.fail("no network in tests"))


def test_weekly_report_leads_with_the_tracker_and_keeps_the_strict_scorecard(monkeypatch):
    _market_cache()
    _no_live_prices(monkeypatch)
    report = cs.generate_weekly_report(today=FRIDAY, claims=_calls())
    messages, record = report["messages"], report["record"]
    assert set(report) == {"messages", "charts", "record"} and report["charts"] == []
    # Marked to the last completed session (Thursday for the Friday run).
    assert record["requested_as_of"] == THURSDAY and record["as_of"] == THURSDAY
    assert record["stats"]["scored"] == 9
    assert messages[0].startswith("📈 Creators vs the market")
    assert "marked to Thu Oct 15 close" in messages[0] and creator_tracker.REPORT_LABEL in messages[0]
    assert "right 2/3" in messages[0]                    # every rate carries its denominator
    # The tracker's messages come first and are its own, in order ...
    assert messages[:-1] == creator_tracker.format_report(record)
    assert any("Where creators agreed" in m for m in messages[:-1])
    # ... and the strict scorecard closes the report, experimental and unranked.
    strict = messages[-1]
    assert strict.startswith("🎯 Strict scorecard — matured, dated forecasts only (EXPERIMENTAL — unranked)")
    assert "canonical claims" in strict and strict.rstrip().endswith(cs.DISCLAIMER)
    assert all(len(m) <= creator_tracker.MAX_MESSAGE_CHARS for m in messages[:-1])


def test_weekly_report_tracker_never_goes_live(monkeypatch):
    # The cache stops a week before the as-of and lacks AAPL entirely: a
    # live-capable fetcher would fill both gaps; the tracker must not try.
    _install_cache(**{"spy.us": _line(500, 1), "nvda.us": _line(100, 2, end=date(2026, 10, 8))})
    _no_live_prices(monkeypatch)
    claims = [_claim("Alpha", "NVDA", "bullish", "unspecified", published="2026-09-08"),
              _claim("Alpha", "AAPL", "bullish", "unspecified", published="2026-09-08", n=1)]
    report = cs.generate_weekly_report(today=FRIDAY, claims=claims)
    calls = {r["asset"]: r for r in report["record"]["calls"]}
    assert calls["NVDA"]["reason"] == "stale_mark" and calls["AAPL"]["reason"] == "no_prices"
    assert report["messages"] and report["messages"][-1].startswith("🎯 Strict scorecard")

    # The default tracker fetcher is the cache fetcher: what it was asked for
    # is exactly the tracker's planned requests, all ending at the as-of.
    seen = []
    real = creator_tracker.cache_fetcher

    def recording_cache_fetcher(cache=None):
        fetch = real(cache)
        seen.append(fetch)
        return fetch

    monkeypatch.setattr(creator_tracker, "cache_fetcher", recording_cache_fetcher)
    cs.generate_weekly_report(today=FRIDAY, claims=claims)
    assert len(seen) == 1
    assert sorted(seen[0].requests) == sorted(creator_tracker.needed_price_ranges(claims, FRIDAY))
    assert {end for _, _, end in seen[0].requests} == {THURSDAY}


def test_strict_block_is_sent_when_nothing_matured(monkeypatch):
    _market_cache()
    _no_live_prices(monkeypatch)
    claims = _calls()                                   # views, but no dated forecast
    strict = cs.generate_canonical_scorecard(FRIDAY, cs.fetch_price_series, claims=claims)
    assert strict.startswith("🎯 Strict scorecard — matured, dated forecasts only (EXPERIMENTAL — unranked)")
    assert "no matured forecasts yet" in strict and "canonical claims" in strict
    assert "forecasts matured by the Thu Oct 15 close" in strict
    assert cs.generate_canonical_scorecard(FRIDAY, cs.fetch_price_series, claims=[]) == ""

    # And the send path delivers it, last.
    sent = _send_mode(monkeypatch, claims)
    assert cs.main([]) == 0
    texts = [payload for kind, payload in sent if kind == "text"]
    assert texts[-1] == strict and texts[0].startswith("📈 Creators vs the market")


def test_strict_block_counts_forecasts_matured_by_the_last_completed_session():
    """A forecast ending on the run day is judged next week: at 16:00 UTC on a
    Friday that day's close is still printing."""
    fetched = []

    def fetch(symbol, start, end):
        fetched.append((symbol, start, end))
        return {}
    fetch.price_provenance = {"provider": "test", "adjustment": "split_adjusted",
                              "corporate_action_status": "split_adjusted"}
    ending_today = _claim("Alpha", "NVDA", "bullish", "short", published="2026-10-01", testable=True,
                          testability_type="unconditional_testable", forecast_end_date=FRIDAY.isoformat())
    strict = cs.generate_canonical_scorecard(FRIDAY, fetch, claims=[ending_today])
    assert "no matured forecasts yet" in strict
    assert fetched == []                                # nothing matured, nothing fetched
    ended_thursday = dict(ending_today, forecast_end_date=THURSDAY.isoformat(), claim_id="thu")
    cs.generate_canonical_scorecard(FRIDAY, fetch, claims=[ended_thursday])
    assert fetched and max(end for _, _, end in fetched) <= THURSDAY


def test_strict_block_lists_conditionals_from_the_trailing_window_only():
    recent = _claim("Alpha", "NVDA", "bullish", "unspecified", published="2026-10-10",
                    condition="if it drops under 150", condition_observable=True, n=1)
    old = _claim("Beta", "AAPL", "bullish", "unspecified", published="2026-08-01",
                 condition="if it drops under 150", condition_observable=True, n=2)
    strict = cs.generate_canonical_scorecard(FRIDAY, cs.fetch_price_series, claims=[recent, old])
    # Below the sample minimum the Friday block only counts them (the research
    # report lists them): Beta's is 76 days old, outside the 30-day window.
    assert "🔀 1 own conditional call in the last 30 days" in strict and "none evaluated yet" in strict


def test_chart_failure_never_blocks_the_text(monkeypatch, tmp_path):
    _market_cache()
    _no_live_prices(monkeypatch)

    def broken(*a, **k):
        raise RuntimeError("renderer exploded")

    monkeypatch.setattr(pulse_charts, "render_tracker_charts", broken)
    report = cs.generate_weekly_report(today=FRIDAY, claims=_calls(), charts_dir=str(tmp_path))
    assert report["charts"] == []
    assert report["messages"][0].startswith("📈 Creators vs the market")
    assert report["messages"][-1].startswith("🎯 Strict scorecard")

    monkeypatch.setattr(pulse_charts, "build_tracker_chart_data", broken)
    assert cs.generate_weekly_report(today=FRIDAY, claims=_calls(), charts_dir=str(tmp_path))["charts"] == []


def test_weekly_report_draws_the_tracker_charts_from_the_same_record(monkeypatch, tmp_path):
    _market_cache()
    _no_live_prices(monkeypatch)
    drawn = []
    real_build = pulse_charts.build_tracker_chart_data

    def spy_build(record):
        drawn.append(record)
        return real_build(record)

    monkeypatch.setattr(pulse_charts, "build_tracker_chart_data", spy_build)
    report = cs.generate_weekly_report(today=FRIDAY, claims=_calls(), charts_dir=str(tmp_path / "charts"))
    assert drawn == [report["record"]]
    assert [os.path.basename(p) for p in report["charts"]] == ["tracker-1-creators.png", "tracker-2-consensus.png"]
    assert all(os.path.getsize(p) > 0 for p in report["charts"])


def test_weekly_report_without_claims_is_empty(monkeypatch):
    monkeypatch.setattr(canonical_claims, "load_canonical_claims", lambda *a, **k: [])
    assert cs.generate_weekly_report(today=FRIDAY) == {"messages": [], "charts": [], "record": {}}


_REAL_WEEKLY_REPORT = cs.generate_weekly_report


def _patch_report(monkeypatch, claims, charts=None):
    """main() builds the report for `claims` on FRIDAY; chart rendering is
    replaced by files written to the requested directory (or by `charts`)."""
    def report(**kwargs):
        return _REAL_WEEKLY_REPORT(today=FRIDAY, claims=claims, **kwargs)

    def fake_render(data, out_dir):
        if charts is not None:
            return charts
        os.makedirs(out_dir, exist_ok=True)
        paths = [os.path.join(out_dir, name) for name in ("tracker-1-creators.png", "tracker-2-consensus.png")]
        for path in paths:
            with open(path, "wb") as f:
                f.write(b"png")
        return paths

    monkeypatch.setattr(cs, "generate_weekly_report", report)
    monkeypatch.setattr(pulse_charts, "render_tracker_charts", fake_render)


def _send_mode(monkeypatch, claims, text_ok=lambda text: True, album_ok=True):
    """Send-mode main() with Telegram replaced by a recorder: [(kind, payload)]."""
    _patch_report(monkeypatch, claims)
    monkeypatch.setenv("TELEGRAM_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHANNEL_ID", "chat")
    sent = []

    def text(token, chat_id, body):
        sent.append(("text", body))
        return text_ok(body)

    def album(token, chat_id, paths, caption=None):
        sent.append(("album", (list(paths), caption)))
        return album_ok

    monkeypatch.setattr(cs, "send_telegram_text", text)
    monkeypatch.setattr(cs, "send_telegram_photo_album", album)
    return sent


def test_dry_run_prints_tracker_then_strict(monkeypatch, capsys, tmp_path):
    _market_cache()
    _no_live_prices(monkeypatch)
    _patch_report(monkeypatch, _calls())
    logged = []
    monkeypatch.setattr(cs, "log_info", lambda m: logged.append(m))
    monkeypatch.setattr(cs, "send_telegram_text", lambda *a, **k: pytest.fail("a dry run never sends"))
    monkeypatch.setattr(cs, "send_telegram_photo_album", lambda *a, **k: pytest.fail("a dry run never sends"))
    charts_dir = tmp_path / "charts"
    assert cs.main(["--dry-run", "--charts-dir", str(charts_dir)]) == 0
    out = capsys.readouterr().out
    tracker, strict = out.index("📈 Creators vs the market"), out.index("🎯 Strict scorecard")
    assert tracker < out.index("Where creators agreed") < strict
    assert "\n\n🎯 Strict scorecard" in out              # messages separated by a blank line
    written = sorted(os.listdir(charts_dir))
    assert written == ["tracker-1-creators.png", "tracker-2-consensus.png"]
    assert any(m.startswith("Charts written: ") and str(charts_dir) in m for m in logged)


def test_send_mode_sends_tracker_then_album_then_strict(monkeypatch):
    _market_cache()
    _no_live_prices(monkeypatch)
    sent = _send_mode(monkeypatch, _calls())
    assert cs.main([]) == 0
    kinds = [kind for kind, _ in sent]
    assert kinds[-2:] == ["album", "text"] and set(kinds[:-2]) == {"text"} and len(kinds) >= 4
    assert sent[0][1].startswith("📈 Creators vs the market")
    paths, caption = sent[-2][1]
    assert caption == cs.CHART_ALBUM_CAPTION and len(paths) == 2
    assert sent[-1][1].startswith("🎯 Strict scorecard")


def test_album_failure_is_only_logged_but_a_text_failure_fails_the_run(monkeypatch):
    _market_cache()
    _no_live_prices(monkeypatch)
    sent = _send_mode(monkeypatch, _calls(), album_ok=False)
    assert cs.main([]) == 0                               # the album is an illustration
    assert sent[-1][1].startswith("🎯 Strict scorecard")

    sent = _send_mode(monkeypatch, _calls(), text_ok=lambda body: not body.startswith("📈"))
    assert cs.main([]) == 1                               # a tracker message failed: stop there
    assert [kind for kind, _ in sent] == ["text"]

    sent = _send_mode(monkeypatch, _calls(), text_ok=lambda body: not body.startswith("🎯"))
    assert cs.main([]) == 1                               # the strict block failed
    assert sent[-1][1].startswith("🎯 Strict scorecard")


def test_main_skips_without_claims_and_keeps_the_legacy_path(monkeypatch):
    monkeypatch.setattr(canonical_claims, "load_canonical_claims", lambda *a, **k: [])
    monkeypatch.setenv("TELEGRAM_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHANNEL_ID", "chat")
    monkeypatch.setattr(cs, "send_telegram_text", lambda *a, **k: pytest.fail("nothing to send"))
    assert cs.main([]) == 0
    # --legacy is the old 7/30-day report, untouched by the new layout.
    monkeypatch.setattr(cs, "generate_scorecard", lambda: "legacy report")
    sent = []
    monkeypatch.setattr(cs, "send_telegram_text", lambda token, chat, text: sent.append(text) or True)
    assert cs.main(["--legacy"]) == 0 and sent == ["legacy report"]


def test_strict_block_below_the_sample_minimum_is_plain_and_compact():
    matured = _claim("Alpha", "NVDA", "bullish", "short", published="2026-07-01", testable=True,
                     testability_type="unconditional_testable", forecast_end_date="2026-07-15")
    later = dict(matured, claim_id="later", forecast_end_date="2026-11-20", video_id="v-later")
    prices = {"nvda.us": {date(2026, 7, 1) + timedelta(days=i): 100.0 + i for i in range(40)},
              "xlk.us": {date(2026, 7, 1) + timedelta(days=i): 200.0 for i in range(40)}}

    def fetch(symbol, start, end):
        return {d: v for d, v in prices.get(symbol, {}).items() if start <= d <= end}
    fetch.price_provenance = {"provider": "test", "adjustment": "split_adjusted",
                              "corporate_action_status": "split_adjusted"}
    strict = cs.generate_canonical_scorecard(date(2026, 8, 1), fetch, claims=[matured, later])
    lines = strict.splitlines()
    assert lines[0] == "🎯 Strict scorecard — matured, dated forecasts only (EXPERIMENTAL — unranked)"
    assert lines[2] == "1 scored so far (unranked) — a comparison needs 20."
    assert lines[3].startswith("• Alpha — NVDA ▲ Jul 1 → Jul 15: +") and lines[3].endswith("right direction")
    assert " vs XLK +0.0%" in lines[3]
    assert "1 dated price forecast matures later (next: Nov 20)." in strict
    for jargon in ("benchmark country", "data/research", "n/a", "excess vs benchmark"):
        assert jargon not in strict, jargon
