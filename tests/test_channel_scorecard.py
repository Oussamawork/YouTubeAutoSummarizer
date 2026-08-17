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
    assert cs.DISCLAIMER in text
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
