"""Tests for the weekly market-pulse aggregation (pure logic, no network)."""
import json
from datetime import date

import market_pulse as mp


def _rec(day, channel="Chan", assets=None, sentiment="neutral"):
    return {
        "date": day,
        "channel_name": channel,
        "signals": None if assets is None else {
            "assets": assets, "market_sentiment": sentiment, "topics": [],
        },
    }


def _asset(name="Tesla", ticker="TSLA", stance="bullish", action="none",
           price_target=None):
    return {"name": name, "ticker": ticker, "stance": stance, "action": action,
            "price_target": price_target, "catalysts": [], "type": "stock",
            "conviction": "medium", "horizon": "unspecified"}


def test_load_signals_missing_file_is_empty(tmp_path):
    assert mp.load_signals(str(tmp_path / "nope.jsonl")) == []


def test_load_signals_skips_malformed_lines(tmp_path):
    path = tmp_path / "signals.jsonl"
    path.write_text(
        json.dumps(_rec("2026-07-20")) + "\nnot json\n\n" + json.dumps(_rec("2026-07-21")) + "\n",
        encoding="utf-8",
    )
    records = mp.load_signals(str(path))
    assert len(records) == 2


def test_aggregate_assets_counts_and_channels():
    records = [
        _rec("2026-07-20", "A", [_asset(stance="bullish", action="buy", price_target=500)]),
        _rec("2026-07-21", "B", [_asset(stance="bullish")]),
        _rec("2026-07-22", "A", [_asset(stance="bearish")]),
        _rec("2026-07-22", "A", None),  # signals=None row must be ignored
    ]
    stats = mp.aggregate_assets(records)
    entry = stats["TSLA"]
    assert entry["mentions"] == 3
    assert entry["channels"] == {"A", "B"}
    assert (entry["bull"], entry["bear"], entry["neutral"]) == (2, 1, 0)
    assert entry["actions"]["buy"] == 1
    assert entry["targets"] == [500]
    assert 0 < mp.net_stance(entry) < 1


def test_asset_key_falls_back_to_name():
    stats = mp.aggregate_assets([
        _rec("2026-07-20", "A", [{"name": "Some Startup", "ticker": None, "stance": "neutral"}]),
    ])
    assert "SOME STARTUP" in stats


def test_find_flips_only_on_sign_change():
    cur = mp.aggregate_assets([
        _rec("2026-07-20", "A", [_asset(stance="bearish"), _asset(name="N", ticker="NVDA", stance="neutral")]),
    ])
    prev = mp.aggregate_assets([
        _rec("2026-07-13", "A", [_asset(stance="bullish"), _asset(name="N", ticker="NVDA", stance="bullish")]),
    ])
    flips = mp.find_flips(cur, prev)
    labels = [f["label"] for f in flips]
    assert labels == ["TSLA"]  # bullish -> bearish flips; bullish -> mixed does not
    assert flips[0]["from"] == "bullish" and flips[0]["to"] == "bearish"


def test_find_new_assets():
    cur = mp.aggregate_assets([_rec("2026-07-20", "A", [_asset(ticker="SOFI", name="SoFi")])])
    older = [_rec("2026-07-01", "A", [_asset(ticker="TSLA")])]
    new = mp.find_new_assets(cur, older)
    assert list(new) == ["SOFI"]
    # Already-seen assets are not "new".
    cur2 = mp.aggregate_assets([_rec("2026-07-20", "A", [_asset(ticker="TSLA")])])
    assert mp.find_new_assets(cur2, older) == {}


def test_build_pulse_contains_sections_and_disclaimer():
    current = [
        _rec("2026-07-20", "A", [_asset(stance="bullish", action="buy")], sentiment="bullish"),
        _rec("2026-07-22", "B", [_asset(ticker="SOFI", name="SoFi", stance="bullish")], sentiment="mixed"),
    ]
    previous = [_rec("2026-07-13", "A", [_asset(stance="bearish")])]
    older = previous
    text = mp.build_pulse(current, previous, older, date(2026, 7, 17), date(2026, 7, 24))
    assert "Weekly Market Pulse" in text
    assert "Videos analyzed: 2" in text
    assert "TSLA" in text and "SOFI" in text
    assert "Consensus flips" in text          # TSLA bearish -> bullish
    assert "New on the radar" in text          # SOFI absent from older records
    assert mp.DISCLAIMER in text


def test_build_pulse_empty_window_returns_empty():
    assert mp.build_pulse([], [], [], date(2026, 7, 17), date(2026, 7, 24)) == ""


def test_generate_pulse_windows(tmp_path, monkeypatch):
    path = tmp_path / "signals.jsonl"
    rows = [
        _rec("2026-07-23", "A", [_asset(stance="bullish")]),   # current window
        _rec("2026-07-12", "A", [_asset(stance="bearish")]),   # previous window
        _rec("2026-05-01", "A", [_asset(ticker="OLD1234")]),   # outside lookback
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    text = mp.generate_pulse(days=7, today=date(2026, 7, 24), path=str(path))
    assert "Videos analyzed: 1" in text
    assert "Consensus flips" in text  # bearish (prev) -> bullish (current)
    assert "OLD1234" not in text


def test_generate_pulse_no_data(tmp_path):
    assert mp.generate_pulse(days=7, today=date(2026, 7, 24), path=str(tmp_path / "none.jsonl")) == ""
