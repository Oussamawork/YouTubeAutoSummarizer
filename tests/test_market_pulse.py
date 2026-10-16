"""Tests for the weekly market-pulse aggregation (pure logic, no network)."""
import json
from datetime import date, timedelta

import pytest

import canonical_claims as cc
import channel_scorecard as cs
import creator_tracker as ct
import instruments
import market_pulse as mp
import price_cache
import pulse_charts

from tests.test_canonical_analytics import _claim


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
    # Private companies stay ticker-less by design (SpaceX listed as SPCX in
    # 2026, so it is no longer the example).
    assert mp.canonical_ticker({"ticker": None, "name": "OpenAI"}) is None


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
    monkeypatch.setattr(cc, "priceable_symbol", lambda entry: "bas.xetra")
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


# --- The canonical Monday pulse: the market side ---
#
# Every price is a close from one as-of: the last completed US session before
# the run (the Friday close for a Monday run), read through creator_tracker.

MONDAY = date(2026, 10, 12)          # a Monday run
FRIDAY = date(2026, 10, 9)           # its as-of: the last completed NYSE session


def _steps(points, start=date(2026, 9, 8), end=FRIDAY, every_day=False):
    """Closes on every weekday (every day for crypto) from start to end, held
    at each (date, close) step until the next one."""
    points = sorted(points)
    closes, day = {}, start
    while day <= end:
        if every_day or day.weekday() < 5:
            closes[day] = next((c for d, c in reversed(points) if d <= day), points[0][1])
        day += timedelta(days=1)
    return closes


def _section(text, title):
    """The lines of one pulse section, from its header to the next blank line."""
    return text.split(title, 1)[1].split("\n\n", 1)[0]


def _market_fetcher(**extra):
    """S&P 500 +1.2% over the week to Fri Oct 9 and +1.9% over four weeks;
    Bitcoin −2.2% / +7.0%; Ether −6.8% / −1.1%; a Nasdaq-100 series that
    stops in September (stale, so it must not be printed as this week's)."""
    series = {
        "spy.us": _steps([(date(2026, 9, 11), 100.0), (date(2026, 10, 2), 100.69), (FRIDAY, 101.9)]),
        "qqq.us": _steps([(date(2026, 9, 11), 100.0), (date(2026, 9, 25), 103.0)], end=date(2026, 9, 25)),
        "btcusd": _steps([(date(2026, 9, 11), 100.0), (date(2026, 10, 2), 109.41), (FRIDAY, 107.0)], every_day=True),
        "ethusd": _steps([(date(2026, 9, 11), 100.0), (date(2026, 10, 2), 106.12), (FRIDAY, 98.9)], every_day=True),
    }
    series.update(extra)
    return ct.static_fetcher(series)


def _two_bulls(asset="NVDA", published="2026-10-07", **over):
    return [_claim("A", asset, "bullish", "unspecified", published=published, video=f"a-{asset}", **over),
            _claim("B", asset, "bullish", "unspecified", published=published, video=f"b-{asset}", **over)]


def test_market_block_reports_week_and_four_week_moves():
    inputs = mp._canonical_pulse_inputs(7, MONDAY, _market_fetcher(), claims=_two_bulls())
    assert inputs["as_of"] == FRIDAY and inputs["market"] == inputs["track"]["market"]
    text = mp.build_canonical_pulse(inputs, MONDAY)
    block = _section(text, "Market · ")
    assert block.splitlines() == ["Fri Oct 2 → Fri Oct 9 close",
                                  "S&P 500 +1.2% · Bitcoin −2.2% · Ether −6.8%",
                                  "4 weeks: S&P 500 +1.9% · Bitcoin +7.0% · Ether −1.1%"]
    # Right after the takeaway, which says the same in words; a symbol
    # without current data (the stale Nasdaq-100 series) is left out, never "n/a".
    assert text.splitlines()[2:4] == ["", "Market · Fri Oct 2 → Fri Oct 9 close"]
    assert text.index("Market · ") < text.index("Where creators agree")
    assert "In the week to Fri Oct 9, the S&P 500 rose 1.2% and Bitcoin fell 2.2%." in text.splitlines()[1]
    assert "stale" not in text                       # the Monday run has Friday's close
    assert "Nasdaq" not in text and "n/a" not in text
    html = mp.generate_pulse(today=MONDAY, inputs=inputs, html=True)
    assert "<b>Market</b> · Fri Oct 2 → Fri Oct 9 close" in html
    assert "S&amp;P 500 +1.2% · Bitcoin −2.2%" in html and "S&P" not in html


def test_agree_row_shows_the_move_since_the_first_call_vs_the_market():
    # A called NVDA, ETH and BTC on Mon Sep 14 and repeated it in this week's
    # window; B's calls are two days old. The row runs from the first call.
    # Both say "buy" on ETH; only A says it on NVDA, which is A's opinion.
    claims = []
    for asset, kind, a_says, b_says in (("NVDA", "stock", "buy", "none"), ("ETH", "crypto", "buy", "buy"),
                                        ("BTC", "crypto", "none", "none")):
        claims += [_claim("A", asset, "bullish", "unspecified", published="2026-09-14", video="a1", asset_type=kind),
                   _claim("A", asset, "bullish", "unspecified", published="2026-10-08", video="a2", asset_type=kind,
                          recommendation_action=a_says),
                   _claim("B", asset, "bullish", "unspecified", published="2026-10-07", video="b1", asset_type=kind,
                          recommendation_action=b_says)]
    fetch = ct.static_fetcher({
        "nvda.us": _steps([(date(2026, 9, 14), 100.0), (date(2026, 10, 7), 104.0), (FRIDAY, 110.0)]),
        "spy.us": _steps([(date(2026, 9, 14), 500.0), (date(2026, 10, 7), 505.0), (FRIDAY, 510.0)]),
        "ethusd": _steps([(date(2026, 9, 14), 2000.0), (FRIDAY, 2200.0)], every_day=True),
        "btcusd": _steps([(date(2026, 9, 14), 100000.0), (FRIDAY, 105000.0)], every_day=True),
    })
    inputs = mp._canonical_pulse_inputs(7, MONDAY, fetch, claims=claims)
    text = mp.build_canonical_pulse(inputs, MONDAY)
    agree = _section(text, "Where creators agree")
    assert agree.splitlines()[0] == " · move since the first call (vs S&P 500, crypto vs Bitcoin)"
    rows = {line.split(" — ")[0][2:]: line for line in agree.splitlines()[1:]}
    assert rows["NVDA"] == "🟢 NVDA — 2 of 2 creators bullish · since Sep 14: +10.0% (S&P +2.0%)"
    assert rows["ETH"] == "🟢 ETH — 2 of 2 creators bullish · since Sep 14: +10.0% (Bitcoin +5.0%) · 2 say buy"
    # Bitcoin is its own benchmark: the move stands alone.
    assert rows["BTC"] == "🟢 BTC — 2 of 2 creators bullish · since Sep 14: +5.0%"


def test_new_call_without_a_close_reads_new_call_no_close_yet():
    # Weekend calls after the Friday close: nothing to measure yet.
    claims = [_claim("A", "NVDA", "bullish", "unspecified", published="2026-10-10", video="a1"),
              _claim("B", "NVDA", "bullish", "unspecified", published="2026-10-11", video="b1"),
              # AMD: C's call is old but its prices stop in September (stale),
              # D's is new — so the row is not "new" and has no move either.
              _claim("C", "AMD", "bullish", "unspecified", published="2026-09-14", video="c1"),
              _claim("C", "AMD", "bullish", "unspecified", published="2026-10-08", video="c2"),
              _claim("D", "AMD", "bullish", "unspecified", published="2026-10-10", video="d1")]
    fetch = _market_fetcher(**{"nvda.us": _steps([(date(2026, 9, 8), 100.0)]),
                               "amd.us": _steps([(date(2026, 9, 8), 100.0)], end=date(2026, 9, 25))})
    inputs = mp._canonical_pulse_inputs(7, MONDAY, fetch, claims=claims)
    agree = _section(mp.build_canonical_pulse(inputs, MONDAY), "Where creators agree")
    assert "🟢 NVDA — 2 of 2 creators bullish · new call, no close yet" in agree
    assert "🟢 AMD — 2 of 2 creators bullish\n" in agree + "\n"


def test_target_fragment_needs_two_consensus_side_creators():
    def target(source, stance, level, video):
        return _claim(source, "NVDA", stance, "unspecified", published="2026-10-06", video=video,
                      claim_type="price_target", forecast_metric="price", target_kind="absolute_value",
                      target_value=level, currency="USD")

    fetch = _market_fetcher(**{"nvda.us": _steps([(date(2026, 10, 6), 100.0), (FRIDAY, 110.0)])})
    # Two bulls with targets, one bear (the dissenter) with a $60 level.
    claims = [target("A", "bullish", 150.0, "a1"), target("B", "bullish", 140.0, "b1"),
              target("C", "bearish", 60.0, "c1")]
    inputs = mp._canonical_pulse_inputs(7, MONDAY, fetch, claims=claims)
    row = _section(mp.build_canonical_pulse(inputs, MONDAY), "Where creators agree").splitlines()[-1]
    assert row.startswith("🟢 NVDA — 2 of 3 creators bullish · since Oct 6: +10.0%")
    assert row.endswith(" · avg target $145 (+32% vs $110)") and "$60" not in row
    # One bull's target is that creator's opinion, not the row's.
    claims = [target("A", "bullish", 150.0, "a1"),
              _claim("B", "NVDA", "bullish", "unspecified", published="2026-10-06", video="b1"),
              target("C", "bearish", 60.0, "c1")]
    inputs = mp._canonical_pulse_inputs(7, MONDAY, fetch, claims=claims)
    row = _section(mp.build_canonical_pulse(inputs, MONDAY), "Where creators agree").splitlines()[-1]
    assert "target" not in row and "$" not in row


def test_creators_vs_market_line_is_pooled_labelled_and_needs_ten_calls():
    def book(creators):
        claims = [_claim(f"Creator{i}", "NVDA", "bullish", "unspecified", published="2026-09-14", video=f"v{i}")
                  for i in range(creators)]
        return claims + _two_bulls(published="2026-10-07")     # this week's window
    fetch = _market_fetcher(**{"nvda.us": _steps([(date(2026, 9, 14), 100.0), (FRIDAY, 110.0)])})
    text = mp.build_canonical_pulse(mp._canonical_pulse_inputs(7, MONDAY, fetch, claims=book(10)), MONDAY)
    section = _section(text, "Creators vs the market\n")
    # 10 calls from Sep 14 (A and B, two days old, are not marked yet).
    assert section == ("Calls since Sep 14, marked to Fri Oct 9: 10 calls · 10/10 moving the called way · "
                       "10/10 beat the S&P 500 in the called direction over the same days (crypto vs Bitcoin). "
                       "Mark-to-market, not a verdict — per-creator detail in Friday's scorecard.")
    assert "Creator" not in section                  # pooled: never per creator, never ranked
    assert text.index("Where creators agree") < text.index("Creators vs the market")
    text = mp.build_canonical_pulse(mp._canonical_pulse_inputs(7, MONDAY, fetch, claims=book(9)), MONDAY)
    assert "Creators vs the market" not in text


def test_last_weeks_picks_use_claims_as_sent():
    # Last week's run: Mon Oct 5 08:00 UTC over (Sep 28, Oct 5].
    def pair(asset, stance, s1, s2, published="2026-10-01", **over):
        return [_claim(s1, asset, stance, "unspecified", published=published, video=f"{s1}-{asset}", **over),
                _claim(s2, asset, stance, "unspecified", published=published, video=f"{s2}-{asset}", **over)]

    late = {"published": "2026-10-04", "extracted_at": "2026-10-05T09:00:00+00:00"}
    claims = (pair("NVDA", "bullish", "A", "B") + pair("AAPL", "bullish", "C", "D")
              + pair("TSLA", "bearish", "E", "F") + pair("BTC", "bullish", "M", "N", asset_type="crypto")
              + pair(None, "bullish", "G", "H", subject_mention="Oil", canonical_entity_name="Oil",
                     asset_type="commodity")
              # Extracted after that run: no reader saw these. They would have
              # split NVDA and added an MSFT pick.
              + pair("NVDA", "bearish", "I", "J", **late) + pair("MSFT", "bullish", "K", "L", **late)
              + _two_bulls("AMD"))                                         # this week's window
    week = lambda move: _steps([(date(2026, 10, 2), 100.0), (date(2026, 10, 5), 100.0), (FRIDAY, 100.0 + move)])
    # Bitcoin is its own benchmark: a rise is a win, whatever the S&P 500 did.
    fetch = ct.static_fetcher({"spy.us": week(2.0), "nvda.us": week(10.0), "aapl.us": week(1.0),
                               "tsla.us": week(8.0), "msft.us": week(20.0),
                               "btcusd": _steps([(date(2026, 10, 5), 100.0), (FRIDAY, 101.0)], every_day=True)})
    inputs = mp._canonical_pulse_inputs(7, MONDAY, fetch, claims=claims)
    picks = inputs["picks"]
    assert [r["label"] for r in picks["rows"]] == ["AAPL", "BTC", "NVDA", "TSLA"] and picks["unpriced"] == ["Oil"]
    text = mp.build_canonical_pulse(inputs, MONDAY)
    section = _section(text, "Last week's picks, one week on")
    # The average is in the called direction (TSLA's rise counts against it);
    # the S&P 500's own move stands beside it, not as a like-for-like comparison.
    assert section.splitlines() == [
        " · Mon Oct 5 → Fri Oct 9",
        "➖ AAPL +1.0% · ✅ BTC +1.0% · ✅ NVDA +10.0% · ❌ TSLA +8.0% (bearish)",
        "3 of 4 went the creators' way · together +1.0% in the called direction · S&P 500 +2.0% · not priced: Oil",
        "✅ went their way and beat the market · ➖ went their way, lagged the market · ❌ went against them"]
    assert "MSFT" not in section
    assert text.splitlines()[1].endswith("Last week's picks: 3 of 4 went the creators' way.")
    # With nothing priced the section is left out.
    inputs = mp._canonical_pulse_inputs(7, MONDAY, lambda *a: {}, claims=claims)
    assert "Last week's picks" not in mp.build_canonical_pulse(inputs, MONDAY)


def test_pulse_prices_end_at_as_of_and_never_go_live(monkeypatch):
    live = []
    monkeypatch.setattr(cs, "fetch_prices_live", lambda *a, **k: live.append(a) or 1 / 0)
    requests = []
    real_cache_fetcher = ct.cache_fetcher

    def recording(cache=None):
        fetch = real_cache_fetcher(cache)

        def wrapped(symbol, start, end):
            requests.append((symbol, start, end))
            return fetch(symbol, start, end)
        wrapped.price_provenance, wrapped.requests = fetch.price_provenance, fetch.requests
        return wrapped
    monkeypatch.setattr(ct, "cache_fetcher", recording)
    cache = price_cache.active()
    for symbol, closes in (
            ("spy.us", _steps([(date(2026, 9, 8), 500.0), (FRIDAY, 510.0)])),
            ("nvda.us", _steps([(date(2026, 9, 8), 100.0), (FRIDAY, 110.0)])),
            # The Sunday warm stored a partial Sunday bar: never "now".
            ("btcusd", _steps([(date(2026, 9, 8), 100000.0), (FRIDAY, 105000.0), (date(2026, 10, 11), 90000.0)],
                              end=date(2026, 10, 11), every_day=True))):
        price_cache.remember(cache, symbol, min(closes), max(closes), closes)
    level = dict(claim_type="price_target", forecast_metric="price", target_kind="absolute_value", currency="USD")
    # AMD is not cached at all: it goes without a price, never to the network.
    claims = (_two_bulls("NVDA", target_value=150.0, **level) + _two_bulls("AMD", target_value=300.0, **level)
              + _two_bulls("BTC", target_value=150000.0, asset_type="crypto", **level))
    inputs = mp._canonical_pulse_inputs(7, MONDAY, None, claims=claims)
    assert inputs["as_of"] == FRIDAY
    assert inputs["latest_prices"] == {"NVDA": 110.0, "BTC": 105000.0}
    assert ("amd.us", FRIDAY - timedelta(days=10), FRIDAY) in requests
    assert ("btcusd", FRIDAY - timedelta(days=10), FRIDAY) in requests
    assert requests and all(end <= FRIDAY for _, _, end in requests)
    assert [r["as_of"] for r in inputs["market"]] == [FRIDAY, FRIDAY]
    text = mp.build_canonical_pulse(inputs, MONDAY)
    assert "Prices: daily closes to Fri Oct 9." in text and "90,000" not in text
    # The one as-of is the last S&P 500 close the cache holds: when that is
    # Thursday, every price the pulse prints is Thursday's.
    price_cache.reset({})
    for symbol, closes in (
            ("spy.us", _steps([(date(2026, 9, 8), 500.0)], end=date(2026, 10, 8))),
            ("nvda.us", _steps([(date(2026, 9, 8), 100.0), (date(2026, 10, 8), 108.0), (FRIDAY, 110.0)])),
            ("btcusd", _steps([(date(2026, 9, 8), 100000.0), (date(2026, 10, 8), 104000.0), (FRIDAY, 105000.0)],
                              end=date(2026, 10, 11), every_day=True))):
        price_cache.remember(price_cache.active(), symbol, min(closes), max(closes), closes)
    inputs = mp._canonical_pulse_inputs(7, MONDAY, None, claims=claims)
    assert inputs["as_of"] == date(2026, 10, 8)
    assert inputs["latest_prices"] == {"NVDA": 108.0, "BTC": 104000.0}
    assert "Prices: daily closes to Thu Oct 8." in mp.build_canonical_pulse(inputs, MONDAY)
    assert live == []


def test_removed_sections_are_gone_and_footer_names_the_price_date():
    owned = {"P1": ("ADBE", "Adobe"), "P2": ("MSFT", "Microsoft"), "P3": ("AAPL", "Apple"),
             "P4": ("AMZN", "Amazon"), "P5": ("NFLX", "Netflix")}
    disclosures = [_claim(source, ticker, "not_applicable", "unspecified", published="2026-10-07",
                          video=f"d-{source}", claim_type="portfolio_disclosure",
                          portfolio_disclosure="owns_unspecified", evidence_text=f"I own {name} stock.",
                          subject_mention=name, canonical_entity_name=name)
                   for source, (ticker, name) in owned.items()]
    # Last week one creator talked about AMD at length: the old "Attention
    # vs last week" section would have fired on it.
    last_week = [_claim("C", "AMD", "bullish", "unspecified", published="2026-10-01", video="c1", n=i)
                 for i in range(5)]
    inputs = mp._canonical_pulse_inputs(7, MONDAY, _market_fetcher(),
                                        claims=_two_bulls() + disclosures + last_week)
    text = mp.build_canonical_pulse(inputs, MONDAY)
    assert "Most discussed" not in text and "Attention vs last week" not in text
    assert not hasattr(mp, "_attention_movers") and not hasattr(mp, "MOVER_MIN_CLAIMS")
    assert text.splitlines()[-2:] == [
        "Prices: daily closes to Fri Oct 9. Based on 2 opinions in 2 videos from 2 creators; "
        "an asset needs 2+ creators to be listed.", mp.DISCLAIMER]
    assert "time horizon" not in text
    owners = _section(text, "What creators say they own\n").splitlines()
    assert len(owners) == 5 and owners[-1] == "+1 more creator"


def test_labels_use_display_names_in_text_and_chart_data(monkeypatch, tmp_path):
    label_of = getattr(instruments, "display_label", None)
    if label_of is None:
        pytest.skip("instruments.display_label is not available: labels stay tickers")
    expected = label_of("SPCX")
    assert expected != "SPCX" and "SPCX" in expected          # "SpaceX (SPCX)"
    claims = [_claim("A", "SPCX", "bearish", "unspecified", video="v1"),
              _claim("B", "SPCX", "bearish", "unspecified", video="v2")]
    inputs = mp._canonical_pulse_inputs(7, date(2026, 7, 12), lambda *a: {}, claims=claims)
    assert inputs["current_views"]["SPCX"]["label"] == expected
    assert f"🔴 {expected} — 2 of 2 creators bearish" in mp.build_canonical_pulse(inputs, date(2026, 7, 12))
    drawn = []
    monkeypatch.setattr(pulse_charts, "render_charts", lambda data, out_dir: drawn.append(data) or [])
    mp.generate_charts(today=date(2026, 7, 12), inputs=inputs, out_dir=str(tmp_path))
    assert drawn and drawn[0]["consensus"][0]["label"] == expected


def test_takeaway_leads_with_the_creator_level_lean():
    claims = [_claim("A", "NVDA", "bullish", "unspecified", video="a1"),
              _claim("A", "AMD", "bullish", "unspecified", video="a1", n=1),
              _claim("B", "NVDA", "bullish", "unspecified", video="b1"),
              _claim("C", "TSLA", "bearish", "unspecified", video="c1"),       # C: one up, one down: mixed
              _claim("C", "MSFT", "bullish", "unspecified", video="c1", n=1),
              _claim("D", "INTC", "bearish", "unspecified", video="d1")]
    today = date(2026, 7, 12)
    inputs = mp._canonical_pulse_inputs(7, today, lambda *a: {}, claims=claims)
    assert mp.build_canonical_pulse(inputs, today).splitlines()[1] == (
        "2 of 4 creators leaned bullish, 1 bearish. Strongest agreement: 2 of 2 creators bullish on NVDA.")
    inputs = mp._canonical_pulse_inputs(7, today, lambda *a: {}, claims=claims[:-1])
    assert mp.build_canonical_pulse(inputs, today).splitlines()[1].startswith(
        "2 of 3 creators leaned bullish, none bearish.")


# --- Review fixes ---------------------------------------------------------------------------------


def _views(rows):
    """Per-asset views from (asset, {creator: vote}) rows, in the
    aggregate_views_by_asset shape the ranking reads."""
    claims = []
    for asset, votes in rows:
        for creator, vote in votes.items():
            if vote == "mixed":
                claims += [_claim(creator, asset, "bullish", "unspecified", video=f"{creator}-{asset}", n=1),
                           _claim(creator, asset, "bearish", "unspecified", video=f"{creator}-{asset}", n=2)]
            else:
                claims.append(_claim(creator, asset, vote, "unspecified", video=f"{creator}-{asset}"))
    return cc.aggregate_views_by_asset(claims)


def test_agreement_ranks_by_share_and_keeps_a_row_for_the_strongest_bearish_agreement():
    views = _views([("NVDA", {"A": "bullish", "B": "bullish", "C": "bullish", "D": "mixed"}),
                    ("MU", {"A": "bullish", "B": "bullish", "C": "bullish", "D": "bullish"})])
    for creator in "ABCD":                         # NVDA has more opinions: the old tie-break favoured it
        views["NVDA"]["mentions"] += 5
    assert [k for k, _ in mp.agree_rows(views)] == ["MU", "NVDA"]           # 4 of 4 before 3 of 4
    bulls = _views([(f"B{i}", {"A": "bullish", "B": "bullish", "C": "bullish"}) for i in range(7)]
                   + [("SPCX", {"A": "bearish", "B": "bearish"})])
    rows = mp.agree_rows(bulls)
    assert len(rows) == mp.MAX_AGREEMENT_ROWS and rows[-1][0] == "SPCX"    # a bullish week never hides it
    assert mp.agree_rows(bulls, limit=1) == [rows[0]]


def test_two_channels_of_one_house_are_not_an_agreement():
    claims = [_claim("HKCM", "DXY", "bullish", "unspecified", published="2026-10-07", video="h1",
                     asset_type="index"),
              _claim("Phantom by HKCM", "DXY", "bullish", "unspecified", published="2026-10-07", video="p1",
                     asset_type="index")] + _two_bulls("NVDA")
    inputs = mp._canonical_pulse_inputs(7, MONDAY, _market_fetcher(), claims=claims)
    assert inputs["houses"]["Phantom by HKCM"] == "HKCM"
    text = mp.build_canonical_pulse(inputs, MONDAY)
    assert "DXY" not in text and "🟢 NVDA — 2 of 2 creators bullish" in text


def test_text_and_upside_chart_use_the_same_targets():
    def target(source, stance, level, video, published="2026-10-06"):
        return _claim(source, "NVDA", stance, "unspecified", published=published, video=video,
                      claim_type="price_target", forecast_metric="price", target_kind="absolute_value",
                      target_value=level, currency="USD")

    fetch = _market_fetcher(**{"nvda.us": _steps([(date(2026, 9, 1), 100.0), (FRIDAY, 110.0)],
                                                  start=date(2026, 9, 1))})
    claims = [target("A", "bullish", 150.0, "a1"), target("B", "bullish", 130.0, "b1"),
              target("C", "bearish", 60.0, "c1"),
              # A's older, higher level from before the window is not this week's target.
              target("A", "bullish", 300.0, "a0", published="2026-09-02")]
    inputs = mp._canonical_pulse_inputs(7, MONDAY, fetch, claims=claims)
    row = _section(mp.build_canonical_pulse(inputs, MONDAY), "Where creators agree").splitlines()[-1]
    assert row.endswith(" · avg target $140 (+27% vs $110)")
    upside = pulse_charts.upside_rows(inputs["current_views"], inputs["latest_prices"])
    assert [(r["label"], r["target"], r["n_targets"], r["price"]) for r in upside] == [("NVDA", 140.0, 2, 110.0)]


def test_a_stale_cache_names_its_week_and_says_the_prices_are_stale():
    # A Friday-16:00 run wants Thursday Oct 15's close; the cache stops at Fri Oct 9.
    friday = date(2026, 10, 16)
    inputs = mp._canonical_pulse_inputs(7, friday, _market_fetcher(), claims=_two_bulls("NVDA", "2026-10-14"))
    assert inputs["as_of"] == FRIDAY and inputs["requested_as_of"] == date(2026, 10, 15)
    text = mp.build_canonical_pulse(inputs, friday)
    assert "In the week to Fri Oct 9, the S&P 500 rose 1.2%" in text and "Last week the" not in text
    assert ("⚠️ Prices are stale: the newest close in the data is Fri Oct 9, not Thu Oct 15; every move here "
            "ends there.") in _section(text, "Market · ")
