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
    # The legacy view is a compatibility path: it runs only when asked for.
    monkeypatch.setattr(mp, "PULSE_DATA_SOURCE", "legacy")
    text = mp.generate_pulse(days=7, today=date(2026, 7, 24), path=str(path),
                             price_fetcher=lambda symbol, start, end: {})
    assert "Videos analyzed: 1" in text
    assert "Consensus flips" in text  # bearish (prev) -> bullish (current)
    assert "OLD1234" not in text


def test_generate_pulse_no_data(tmp_path):
    assert mp.generate_pulse(days=7, today=date(2026, 7, 24), path=str(tmp_path / "none.jsonl")) == ""


# --- Track-record weighting and implied upside ---


def test_weighted_consensus_changes_direction():
    records = [
        _rec("2026-07-20", "Good", [_asset(stance="bullish")]),
        _rec("2026-07-21", "Bad", [_asset(stance="bearish")]),
    ]
    unweighted = mp.aggregate_assets(records)
    assert mp._direction(mp.net_stance(unweighted["TSLA"])) == "mixed"
    weighted = mp.aggregate_assets(records, {"Good": 1.5, "Bad": 0.5})
    assert mp._direction(mp.net_stance(weighted["TSLA"])) == "bullish"
    # Raw display counts stay unweighted.
    assert weighted["TSLA"]["bull"] == 1 and weighted["TSLA"]["bear"] == 1


def test_channel_weights_from_track_record():
    from datetime import timedelta
    d0 = date(2026, 7, 1)
    series = {"tsla.us": {d0: 100.0, d0 + timedelta(days=7): 110.0}}
    # 5 evaluated calls for "Proven" (meets MIN_TRACK_CALLS), 1 for "Rookie".
    records = [_rec("2026-07-01", "Proven", [_asset(stance="bullish")]) for _ in range(5)]
    records.append(_rec("2026-07-01", "Rookie", [_asset(stance="bullish")]))
    weights, details = mp.channel_weights_from_track_record(
        records, date(2026, 7, 24), price_fetcher=lambda s, a, b: series.get(s, {})
    )
    assert weights == {"Proven": 1.5}  # 5/5 hits -> 0.5 + 1.0
    assert details == {"Proven": (5, 5)}
    assert "Rookie" not in weights  # below MIN_TRACK_CALLS


def test_asset_line_implied_upside():
    entry = mp.aggregate_assets([
        _rec("2026-07-20", "A", [_asset(stance="bullish", price_target=110)]),
    ])["TSLA"]
    line = mp._format_asset_line(entry, latest_price=100.0)
    assert "avg target 110 (+10% implied)" in line
    assert "implied" not in mp._format_asset_line(entry)  # no price -> no annotation


def test_build_pulse_weight_footer():
    current = [_rec("2026-07-20", "A", [_asset(stance="bullish")])]
    text = mp.build_pulse(
        current, [], [], date(2026, 7, 17), date(2026, 7, 24),
        channel_weights={"A": 1.25}, weight_details={"A": (6, 8)},
    )
    assert "⚖️ Consensus weighted by 7d track record: A 1.25 (6/8)" in text
    # No details -> no footer line.
    text2 = mp.build_pulse(current, [], [], date(2026, 7, 17), date(2026, 7, 24))
    assert "⚖️" not in text2


def test_fetch_latest_prices_only_for_targeted_tickers():
    entries = mp.aggregate_assets([
        _rec("2026-07-20", "A", [_asset(stance="bullish", price_target=110)]),
        _rec("2026-07-20", "A", [_asset(name="No Target", ticker="NT", stance="bullish")]),
    ])
    calls = []

    def fetcher(symbol, start, end):
        calls.append(symbol)
        return {date(2026, 7, 23): 100.0}

    prices = mp.fetch_latest_prices(entries, price_fetcher=fetcher, today=date(2026, 7, 24))
    assert calls == ["tsla.us"]  # NT has no price target -> not fetched
    assert prices == {"TSLA": 100.0}


def test_canonical_ticker_prefers_recorded_then_alias_table():
    # The extractor is forbidden from guessing tickers, so "Chevron" arrives
    # ticker-less; canonicalization is code's job, deterministic and auditable.
    assert mp.canonical_ticker({"ticker": "nvda", "name": "whatever"}) == "NVDA"
    assert mp.canonical_ticker({"ticker": None, "name": "Chevron"}) == "CVX"
    assert mp.canonical_ticker({"ticker": "", "name": "  bitcoin "}) == "BTC"
    # Private companies stay ticker-less by design.
    assert mp.canonical_ticker({"ticker": None, "name": "SpaceX"}) is None


def test_ticker_variants_fold_to_one_key():
    # Dual share classes / renames must not split one asset across two keys.
    assert mp._asset_key({"ticker": "GOOG", "name": "Alphabet"}) == "GOOGL"
    assert mp._asset_key({"ticker": None, "name": "Google"}) == "GOOGL"
    assert mp._asset_key({"ticker": "SQ", "name": "Block"}) == "XYZ"


def test_name_and_ticker_records_aggregate_together():
    # BTC (ticker recorded) and Bitcoin (name only) are the same asset; split
    # keys dilute consensus and make flip detection blind.
    recs = [
        {"date": "2026-07-28", "channel_name": "a",
         "signals": {"assets": [{"name": "Bitcoin", "ticker": None,
                                 "type": "crypto", "stance": "bearish"}]}},
        {"date": "2026-07-28", "channel_name": "b",
         "signals": {"assets": [{"name": "BTC", "ticker": "BTC",
                                 "type": "crypto", "stance": "bearish"}]}},
    ]
    stats = mp.aggregate_assets(recs)
    assert set(stats) == {"BTC"}
    assert stats["BTC"]["mentions"] == 2 and stats["BTC"]["bear"] == 2


def _vote(stance, conviction="unspecified", name="Nvidia", ticker="NVDA"):
    return {"name": name, "ticker": ticker, "type": "stock",
            "stance": stance, "conviction": conviction}


def _vrec(assets, channel="chan", date="2026-07-28"):
    return {"date": date, "channel_name": channel, "signals": {"assets": assets}}


def test_neutral_mentions_are_breadth_not_dilution():
    # 3 bullish calls + 5 neutral name-drops is a bullish consensus with wide
    # radar coverage — not "mixed". Neutrals stay out of the denominator.
    recs = [_vrec([_vote("bullish")], f"c{i}") for i in range(3)]
    recs += [_vrec([_vote("neutral")], f"n{i}") for i in range(5)]
    stats = mp.aggregate_assets(recs)
    assert mp.net_stance(stats["NVDA"]) == 1.0
    assert mp._direction(mp.net_stance(stats["NVDA"])) == "bullish"
    assert stats["NVDA"]["neutral"] == 5          # breadth is still recorded


def test_conviction_weighs_directional_votes():
    # One table-pounding bearish call (1.5) vs two hedged bullish leans
    # (0.75 each): dead heat -> mixed, where unweighted counting said bullish.
    recs = [_vrec([_vote("bullish", "low")], "a"),
            _vrec([_vote("bullish", "unspecified")], "b"),
            _vrec([_vote("bearish", "high")], "c")]
    stats = mp.aggregate_assets(recs)
    assert abs(mp.net_stance(stats["NVDA"])) <= mp.NET_THRESHOLD
    assert mp._direction(mp.net_stance(stats["NVDA"])) == "mixed"


def test_ranking_prefers_directional_over_name_drops():
    # A megacap name-dropped neutrally everywhere must not outrank an asset
    # with real calls on it.
    recs = [_vrec([_vote("neutral", name="Microsoft", ticker="MSFT")], f"n{i}")
            for i in range(6)]
    recs += [_vrec([_vote("bullish", "high", name="Chevron", ticker="CVX")], f"c{i}")
             for i in range(2)]
    text = mp.build_pulse(recs, [], [], date(2026, 7, 21), date(2026, 7, 28))
    top = [l for l in text.splitlines() if l.startswith("• ")]
    assert top[0].startswith("• CVX")
    assert any(l.startswith("• MSFT") for l in top)  # still reported, as breadth


def test_neutral_flood_cannot_fake_a_flip():
    # Same 2-0 bullish consensus both weeks; this week adds 5 neutral
    # name-drops. Under the old dilution math the direction collapsed to
    # "mixed"; either way a flip alert requires a real sign change.
    prev = [_vrec([_vote("bullish")], f"p{i}", "2026-07-15") for i in range(2)]
    cur = [_vrec([_vote("bullish")], f"c{i}") for i in range(2)]
    cur += [_vrec([_vote("neutral")], f"n{i}") for i in range(5)]
    flips = mp.find_flips(mp.aggregate_assets(cur), mp.aggregate_assets(prev))
    assert flips == []


def test_asset_line_separates_direction_from_breadth():
    recs = [_vrec([_vote("bullish")], "a"), _vrec([_vote("neutral")], "b")]
    stats = mp.aggregate_assets(recs)
    line = mp._format_asset_line(stats["NVDA"])
    assert "net bullish (1↑/0↓, 1 neutral)" in line
    assert "2 mentions across 2 channels" in line


def test_fetch_latest_prices_warns_when_every_lookup_fails(monkeypatch):
    """All-tickers-failed is a source outage, not a per-symbol miss; it must
    surface as one loud line rather than only per-symbol warnings."""
    warnings = []
    monkeypatch.setattr(mp, "log_warn", lambda msg: warnings.append(msg))
    entries = mp.aggregate_assets([
        _rec("2026-08-10", "A", [_asset(price_target=500)]),
        _rec("2026-08-10", "A", [_asset("Nvidia", "NVDA", price_target=200)]),
    ])
    prices = mp.fetch_latest_prices(entries, price_fetcher=lambda *a: {},
                                    today=date(2026, 8, 16))
    assert prices == {}
    assert any("any of the 2 tickers" in w for w in warnings)


def test_fetch_latest_prices_quiet_on_partial_success(monkeypatch):
    warnings = []
    monkeypatch.setattr(mp, "log_warn", lambda msg: warnings.append(msg))
    entries = mp.aggregate_assets([
        _rec("2026-08-10", "A", [_asset(price_target=500)]),
        _rec("2026-08-10", "A", [_asset("Nvidia", "NVDA", price_target=200)]),
    ])

    def fetcher(symbol, start, end):
        return {end.isoformat(): 100.0} if symbol == "tsla.us" else {}

    prices = mp.fetch_latest_prices(entries, price_fetcher=fetcher,
                                    today=date(2026, 8, 16))
    assert prices == {"TSLA": 100.0}
    assert not any("any of the" in w for w in warnings)


def test_latest_prices_skip_non_usd_listings(monkeypatch):
    """A euro close against a dollar price target is arithmetic on two
    different units — the implied-move annotation must skip it."""
    import channel_scorecard as cs
    monkeypatch.setattr(cs, "symbol_for", lambda asset: "bas.xetra")
    asked = []

    def fetcher(symbol, start, end):
        asked.append(symbol)
        return {end.isoformat(): 45.0}

    entries = mp.aggregate_assets([
        _rec("2026-08-10", "A", [_asset("BASF", "BASF", price_target=60)]),
    ])
    prices = mp.fetch_latest_prices(entries, price_fetcher=fetcher,
                                    today=date(2026, 8, 16))
    assert prices == {} and asked == []  # never even requested
