"""Tests for the per-channel accuracy scorecard (prices mocked, no network)."""
import json
from datetime import date, timedelta

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
