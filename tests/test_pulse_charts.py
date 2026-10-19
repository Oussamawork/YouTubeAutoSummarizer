"""Tests for the weekly pulse charts and the creators-vs-market tracker
charts: pure data prep plus rendering and send-wiring smoke tests
(matplotlib is a runtime dependency, so rendering runs in CI too; only the
network stays mocked, and tracker prices come from in-memory fetchers)."""
import inspect
import json
import os
import sys
from datetime import date, timedelta

import pytest

import creator_tracker as ct
import market_pulse as mp
import pulse_charts as pc
import scorecard_pricing as sp


def _rec(day, channel="Chan", assets=None, sentiment="neutral"):
    return {
        "date": day,
        "channel_name": channel,
        "signals": None if assets is None else {
            "assets": assets, "market_sentiment": sentiment, "topics": [],
        },
    }


def _asset(name="Tesla", ticker="TSLA", stance="bullish", price_target=None):
    return {"name": name, "ticker": ticker, "stance": stance, "action": "none",
            "price_target": price_target, "catalysts": [], "type": "stock",
            "conviction": "medium", "horizon": "unspecified"}


def _agg(records):
    return mp.aggregate_assets(records)


def test_window_label_same_and_cross_month():
    assert pc.window_label(date(2026, 8, 9), date(2026, 8, 16)) == "Aug 10 - 16, 2026"
    assert pc.window_label(date(2026, 7, 26), date(2026, 8, 2)) == "Jul 27 - Aug 2, 2026"


def test_consensus_rows_rank_and_skip_undirectional():
    current = _agg([
        _rec("2026-08-10", "A", [_asset("Nvidia", "NVDA", "bullish"),
                                 _asset("Apple", "AAPL", "neutral")]),
        _rec("2026-08-11", "B", [_asset("Nvidia", "NVDA", "bullish")]),
        _rec("2026-08-12", "C", [_asset("Tesla", "TSLA", "bearish")]),
    ])
    rows = pc.consensus_rows(current)
    # NVDA (2 directional calls) ranks above TSLA (1); AAPL (neutral only) is skipped.
    assert [r["label"] for r in rows] == ["NVDA", "TSLA"]
    assert rows[0] == {"label": "NVDA", "bull": 2, "bear": 0, "neutral": 0, "channels": 2}


def test_flip_rows_detects_reversal_with_evidence():
    previous = _agg([
        _rec("2026-08-03", "A", [_asset(stance="bullish")]),
        _rec("2026-08-04", "B", [_asset(stance="bullish")]),
    ])
    current = _agg([
        _rec("2026-08-10", "A", [_asset(stance="bearish")]),
        _rec("2026-08-11", "B", [_asset(stance="bearish")]),
    ])
    rows = pc.flip_rows(current, previous)
    assert len(rows) == 1
    row = rows[0]
    assert row["label"] == "TSLA"
    assert row["from_score"] > 0 > row["to_score"]
    assert (row["from_votes"], row["to_votes"]) == (2, 2)


def test_flip_rows_ignores_move_to_mixed():
    previous = _agg([_rec("2026-08-03", "A", [_asset(stance="bullish")])])
    current = _agg([
        _rec("2026-08-10", "A", [_asset(stance="bullish")]),
        _rec("2026-08-11", "B", [_asset(stance="bearish")]),
    ])
    assert pc.flip_rows(current, previous) == []


def test_tone_weeks_buckets_and_partial_flag():
    records = [
        _rec("2026-08-14", assets=[], sentiment="bullish"),
        _rec("2026-08-15", assets=[], sentiment="bearish"),
        _rec("2026-08-05", assets=[], sentiment="bullish"),
    ]
    weeks = pc.tone_weeks(records, date(2026, 8, 16))
    assert [w["n"] for w in weeks] == [1, 2]
    # Oldest week first; the dataset starts mid-window, so it is partial.
    assert weeks[0]["partial"] is True
    assert weeks[1]["partial"] is False
    assert weeks[1]["bullish"] == 1 and weeks[1]["bearish"] == 1
    assert "2026" not in weeks[0]["label"]


def test_spread_weeks_signed_percentage_points():
    records = [
        # Current week: 3 bullish, 1 bearish of 4 -> (75 - 25) = +50.
        _rec("2026-08-14", assets=[], sentiment="bullish"),
        _rec("2026-08-14", assets=[], sentiment="bullish"),
        _rec("2026-08-15", assets=[], sentiment="bullish"),
        _rec("2026-08-15", assets=[], sentiment="bearish"),
        # Prior week: 1 bullish, 3 bearish of 4 -> -50.
        _rec("2026-08-05", assets=[], sentiment="bullish"),
        _rec("2026-08-05", assets=[], sentiment="bearish"),
        _rec("2026-08-06", assets=[], sentiment="bearish"),
        _rec("2026-08-06", assets=[], sentiment="bearish"),
    ]
    rows = pc.spread_weeks(records, date(2026, 8, 16))
    assert [round(r["spread"]) for r in rows] == [-50, 50]  # oldest first
    # weeks_ago lets the chart tell a skipped week from a consecutive one.
    assert [r["weeks_ago"] for r in rows] == [1, 0]


def test_spread_weeks_marks_gaps_by_weeks_ago():
    records = [
        _rec("2026-08-15", assets=[], sentiment="bullish"),   # current window
        _rec("2026-07-27", assets=[], sentiment="bearish"),   # window two weeks back
    ]
    rows = pc.spread_weeks(records, date(2026, 8, 16))
    # The silent week between is dropped, but the jump in weeks_ago (2 -> 0)
    # is what tells the renderer to break the line instead of bridging it.
    assert [r["weeks_ago"] for r in rows] == [2, 0]


def test_spread_chart_skipped_until_three_weeks(tmp_path):
    data = {"window": "Aug 10 - 16, 2026", "videos": 1, "channels": 1,
            "consensus": [], "flips": [], "tone": [], "map": [], "upside": [],
            "spread": [{"label": "Aug 10 - Aug 16", "short_label": "Aug 16",
                        "weeks_ago": 0, "n": 5, "spread": 20.0, "partial": False}]}
    assert pc.render_charts(data, str(tmp_path)) == []


def test_conviction_points_threshold_and_bearish_kept_past_cap():
    records = []
    # 16 bullish assets with descending call counts crowd the cap...
    for i in range(16):
        for n in range(3 + (16 - i)):
            records.append(_rec("2026-08-10", f"C{n}", [_asset(f"Bull{i:02d}", f"B{i:02d}")]))
    # ...and one bearish asset with the minimum 3 calls would fall off the end.
    for n in range(3):
        records.append(_rec("2026-08-10", f"C{n}", [_asset("Bear", "BEAR", "bearish")]))
    points = pc.conviction_points(_agg(records))
    labels = {p["label"] for p in points}
    assert "BEAR" in labels  # non-bullish outliers survive the cap
    assert len(points) == pc.MAX_MAP_POINTS + 1


def test_upside_rows_math_and_order():
    current = _agg([
        _rec("2026-08-10", "A", [_asset("Reddit", "RDDT", price_target=203),
                                 _asset("Broadcom", "AVGO", price_target=505),
                                 _asset("Lululemon", "LULU", price_target=400),
                                 _asset("Apple", "AAPL")]),  # no target -> excluded
        _rec("2026-08-11", "B", [_asset("Broadcom", "AVGO", price_target=390)]),
    ])
    rows = pc.upside_rows(current, {"RDDT": 177.0, "AVGO": 400.0, "LULU": 94.5})
    # Shared targets first, then the implied move: a lone +323% target never
    # outranks the one two creators back.
    assert [r["label"] for r in rows] == ["AVGO", "LULU", "RDDT"]
    assert [r["n_targets"] for r in rows] == [2, 1, 1]
    assert round(rows[2]["pct"], 1) == 14.7
    # AVGO averages its two targets and keeps the low-high range for the chart —
    # including a low target sitting below the last close (negative percent).
    avgo = rows[0]
    assert (avgo["t_lo"], avgo["t_hi"], avgo["n_targets"]) == (390, 505, 2)
    assert round(avgo["lo"], 1) == -2.5 and round(avgo["hi"], 1) == 26.2
    assert round(avgo["pct"], 1) == 11.9


def _full_chart_data():
    week = lambda i, n, bull, bear: {"label": f"W{i}", "short_label": f"W{i}", "weeks_ago": i, "n": n,
                                     "bullish": bull, "bearish": bear, "mixed": n - bull - bear, "neutral": 0,
                                     "partial": False}
    return {
        "window": "Aug 10 - 16, 2026", "videos": 40, "channels": 6,
        "consensus": [{"label": l, "bull": b, "bear": r, "neutral": 0, "channels": b + r}
                      for l, b, r in (("NVDA", 4, 0), ("MU", 2, 2), ("INTC", 0, 2))],
        "tone": [week(1, 30, 18, 4), week(0, 40, 26, 5)],
        "upside": [{"label": "AVGO", "price": 400.0, "target": 505.0, "t_lo": 390.0, "t_hi": 620.0,
                    "pct": 26.0, "lo": -2.5, "hi": 55.0, "n_targets": 2}],
        "flips": [{"label": l, "from_score": 1, "to_score": -1, "from_votes": 2, "to_votes": 3}
                  for l in ("AMD", "SK Hynix")],
        "map": [{"label": l, "net": n, "directional": d} for l, n, d in
                (("NVDA", 1, 6), ("MU", 0, 4), ("INTC", -1, 3), ("META", 0.3, 4), ("AMZN", 1, 3), ("SOL", 0, 3))],
        "spread": [{"label": f"W{i}", "short_label": f"W{i}", "weeks_ago": i, "n": 40, "spread": 30.0 + i,
                    "partial": False} for i in range(pc.MIN_SPREAD_WEEKS - 1, -1, -1)],
    }


def test_render_charts_writes_pngs(tmp_path):
    paths = pc.render_charts(_full_chart_data(), str(tmp_path))
    assert [p.split("/")[-1] for p in paths] == [
        "1-consensus.png", "2-tone.png", "3-upside.png", "4-flips.png", "5-map.png", "6-spread.png"]
    for path in paths:
        assert (tmp_path / path.split("/")[-1]).stat().st_size > 0


def test_thin_sections_are_left_out_of_the_album(tmp_path):
    data = _full_chart_data()
    data["consensus"] = data["consensus"][:2]            # < 3 assets with 2+ creators
    data["flips"][1]["from_votes"] = 1                   # only one well-attended reversal
    data["map"] = [p for p in data["map"] if p["net"] > 0.2]  # nothing bearish or contested
    data["spread"] = data["spread"][-3:]                 # 3 weeks repeat the tone chart
    paths = pc.render_charts(data, str(tmp_path))
    assert [p.split("/")[-1] for p in paths] == ["2-tone.png", "3-upside.png"]


def test_chart_titles_state_the_takeaway():
    data = _full_chart_data()
    assert pc.consensus_title(data["consensus"]) == "Bullish on NVDA; bearish on INTC"
    assert pc.tone_title(data["tone"]) == "65% of this week's videos expect a rise (up from 60%)"
    assert pc.upside_title(data["upside"]) == "Best-supported target: AVGO +26% (2 creators)"
    # The upside headline is the best-supported target, never the boldest one:
    # a lone multi-year target does not headline, whatever its size ...
    lone = {"label": "LULU", "price": 94.5, "target": 400.0, "t_lo": 400.0, "t_hi": 400.0,
            "pct": 323.0, "lo": 323.0, "hi": 323.0, "n_targets": 1}
    assert pc.upside_title([lone] + data["upside"]) == "Best-supported target: AVGO +26% (2 creators)"
    # ... more creators beat a bigger move, and without a shared target there is no headline number.
    three = dict(data["upside"][0], label="NVDA", pct=9.0, n_targets=3)
    assert pc.upside_title(data["upside"] + [three]) == "Best-supported target: NVDA +9% (3 creators)"
    assert pc.upside_title([lone]) == "Creator price targets vs the last close"
    # A longer label drops the count rather than overflow the header; one too
    # long even for that gets the plain title.
    named = dict(data["upside"][0], label="Micron (MU)", pct=12.0)
    assert pc.upside_title([named]) == "Best-supported target: Micron (MU) +12%"
    very_long = dict(data["upside"][0], label="Taiwan Semiconductor (TSM)")
    assert pc.upside_title([very_long]) == "Creator price targets vs the last close"
    titles = [pc.upside_title(rows) for rows in ([named], [very_long], [lone], data["upside"])]
    assert all(len(t) <= pc.UPSIDE_TITLE_MAX_CHARS for t in titles)
    assert "Boldest" not in inspect.getsource(pc)


def test_render_charts_skips_empty_sections(tmp_path):
    data = {"window": "Aug 10 - 16, 2026", "videos": 1, "channels": 1,
            "consensus": [], "flips": [], "tone": [], "spread": [], "map": [],
            "upside": []}
    assert pc.render_charts(data, str(tmp_path)) == []


def test_generate_charts_never_raises(tmp_path, monkeypatch):
    # Even a corrupt dataset path must degrade to "no charts", not an error.
    monkeypatch.setattr(pc, "build_chart_data", lambda *a, **k: 1 / 0)
    path = tmp_path / "signals.jsonl"
    path.write_text(json.dumps(_rec("2026-08-10", "A", [_asset()])) + "\n", encoding="utf-8")
    assert mp.generate_charts(today=date(2026, 8, 16), path=str(path),
                              price_fetcher=lambda *a: {}, out_dir=str(tmp_path)) == []


# --- Creators vs the market (tracker charts) ----------------------------------------------

AS_OF = date(2026, 10, 9)   # a Friday
RANKING_WORDS = ("best", "worst", "rank", "top ", "#1", "winner", "leader", "most accurate")


def _closes(first, step, start=date(2026, 8, 31), end=AS_OF, crypto=False):
    """Closes on every NYSE session (every day for crypto), moving `step` a day."""
    out, day, i = {}, start, 0
    while day <= end:
        if crypto or sp.is_trading_day(sp.EXCHANGES["us"], day):
            out[day] = round(first + i * step, 6)
            i += 1
        day += timedelta(days=1)
    return out


TRACKER_PRICES = {
    "spy.us": _closes(500, 0.5), "btcusd": _closes(80000, 50, crypto=True), "ethusd": _closes(2000, -10, crypto=True),
    "nvda.us": _closes(100, 1.5), "meta.us": _closes(600, -2), "intc.us": _closes(20, 0.2),
    "amzn.us": _closes(200, 0.5), "msft.us": _closes(400, 1), "aapl.us": _closes(220, -0.5),
    "tsla.us": _closes(300, -2), "googl.us": _closes(180, 1),
}
# (creator, ticker, stance, published): Alpha 5 calls, Beta 4, Gamma 3, Delta 2, Crypto 1 (BTC
# itself, which has no benchmark). Three creators agree on NVDA, two on META and INTC; TSLA is split.
TRACKER_CALLS = [
    ("Alpha", "NVDA", "bullish", "2026-09-08"), ("Alpha", "META", "bullish", "2026-09-09"),
    ("Alpha", "INTC", "bearish", "2026-09-10"), ("Alpha", "AMZN", "bullish", "2026-09-11"),
    ("Alpha", "MSFT", "bullish", "2026-09-14"),
    ("Beta", "NVDA", "bullish", "2026-09-09"), ("Beta", "META", "bullish", "2026-09-10"),
    ("Beta", "INTC", "bearish", "2026-09-11"), ("Beta", "AAPL", "bullish", "2026-09-14"),
    ("Gamma", "NVDA", "bullish", "2026-09-10"), ("Gamma", "TSLA", "bearish", "2026-09-11"),
    ("Gamma", "ETH", "bullish", "2026-09-14"),
    ("Delta", "TSLA", "bullish", "2026-09-15"), ("Delta", "GOOGL", "bullish", "2026-09-16"),
    ("Crypto", "BTC", "bullish", "2026-09-15"),
]


def _tracker_record(calls=TRACKER_CALLS):
    """A real creator_tracker record over canonical-claim fixtures and in-memory prices."""
    from tests.test_canonical_analytics import _claim   # pinned fixture; imported lazily
    claims = [_claim(creator, ticker, stance, "unspecified", published=day, video=f"v-{creator}-{ticker}",
                     asset_type="crypto" if ticker in ("BTC", "ETH") else "stock")
              for creator, ticker, stance, day in calls]
    return ct.track_record(claims, AS_OF, ct.static_fetcher(TRACKER_PRICES))


def _summary(calls, median, bench_n=None, beat=0, small=None, guests=False):
    """A creator_tracker CreatorSummary, reduced to what the charts read."""
    return {"calls": calls, "bench_n": calls if bench_n is None else bench_n, "beat": beat,
            "median_dir_excess": median, "small_sample": calls < 10 if small is None else small,
            "mostly_guests": guests}


def _asset_summary(label, consensus, bulls=(), bears=(), move=None, bench=None, bench_label="S&P 500",
                   since=date(2026, 9, 8), right=None):
    """A creator_tracker AssetSummary, reduced to what the charts read."""
    priced = move is not None
    return {"asset": label, "label": label, "current": {}, "bulls": list(bulls), "bears": list(bears),
            "consensus": consensus, "since": since if priced else None, "move": move,
            "benchmark_label": bench_label if priced else None, "bench_move": bench, "right": right}


def _rich_tracker_data():
    creators = [
        {"label": "Parkev Tatevosian, CFA", "calls": 62, "bench_n": 62, "beat": 21, "median_dir_excess": -2.6,
         "small_sample": False, "mostly_guests": False},
        {"label": "TheStreet", "calls": 36, "bench_n": 34, "beat": 14, "median_dir_excess": -1.2,
         "small_sample": False, "mostly_guests": True},
        {"label": "Couch Investor", "calls": 12, "bench_n": 12, "beat": 6, "median_dir_excess": 0.03,
         "small_sample": False, "mostly_guests": False},
        {"label": "Joseph Carlson After Hours", "calls": 4, "bench_n": 4, "beat": 4, "median_dir_excess": 13.7,
         "small_sample": True, "mostly_guests": False},
    ]
    outcome = lambda label, direction, n, move, bench, bench_label="S&P 500", right=True: {
        "label": label, "direction": direction, "n_creators": n, "move": move, "bench_move": bench,
        "benchmark_label": bench_label, "since": date(2026, 9, 8), "right": right}
    outcomes = [outcome("META", "bullish", 4, 17.1, 1.6), outcome("NFLX", "bullish", 4, -8.4, 1.6, right=False),
                outcome("ETH", "bearish", 2, -9.7, -4.1, "Bitcoin"),
                outcome("S&P 500 (SPY)", "bullish", 2, 1.6, None, None)]   # the market itself: no tick
    return {"as_of": AS_OF, "since": date(2026, 9, 5), "stale": True, "creators": creators,
            "not_charted": [{"label": "HKCM GLOBAL", "calls": 1, "reason": "too_few"},
                            {"label": "Bitcoin Only", "calls": 5, "reason": "no_benchmark"}],
            "outcomes": outcomes}


def _names(paths):
    return [os.path.basename(p) for p in paths]


def test_creators_vs_market_rows_order_by_calls_not_results():
    record = {"creators": {
        "Hot Hand": _summary(4, 0.12, beat=4),                # few calls, by far the best result
        "Steady": _summary(25, -0.01, beat=11),
        "Middle": _summary(12, 0.02, beat=7),
        "Also Middle": _summary(12, -0.05, beat=3),          # same count as Middle: the name decides
        "Benchmark Only": _summary(5, None, bench_n=0),       # only BTC / SPY calls: nothing to compare
        "No Comparison": _summary(6, 0.04, bench_n=0),        # a median without compared calls is no bar
        "One Call": _summary(1, 0.30, beat=1),               # too few calls to chart
    }}
    rows = pc.creators_vs_market_rows(record)
    assert [r["label"] for r in rows] == ["Steady", "Also Middle", "Middle", "Hot Hand"]
    assert [r["median_dir_excess"] for r in rows] == pytest.approx([-1.0, -5.0, 2.0, 12.0])   # percentage points
    assert pc.creators_vs_market_rows(record, limit=2) == rows[:2]
    # The minimum follows the text report's 'too few calls' line, and can be lowered.
    assert "One Call" in [r["label"] for r in pc.creators_vs_market_rows(record, min_calls=1)]
    data = pc.build_tracker_chart_data(dict(record, calls=[{}], market_data=True))
    assert data["not_charted"] == [{"label": "No Comparison", "calls": 6, "reason": "no_benchmark"},
                                   {"label": "Benchmark Only", "calls": 5, "reason": "no_benchmark"},
                                   {"label": "One Call", "calls": 1, "reason": "too_few"}]
    assert pc.creators_vs_market_rows(None) == [] and pc.creators_vs_market_rows({}) == []


def test_small_sample_flag_and_benchmark_counts_are_carried():
    record = {"creators": {"A": _summary(30, 0.015, bench_n=28, beat=17, guests=True),
                           "B": _summary(6, -0.02, bench_n=4, beat=1)}}
    a, b = pc.creators_vs_market_rows(record)
    assert a == {"label": "A", "calls": 30, "bench_n": 28, "beat": 17, "median_dir_excess": pytest.approx(1.5),
                 "small_sample": False, "mostly_guests": True}
    assert (b["calls"], b["bench_n"], b["beat"], b["small_sample"]) == (6, 4, 1, True)
    # The tracker's own flag is carried, not recomputed here.
    record["creators"]["A"]["small_sample"] = True
    assert pc.creators_vs_market_rows(record)[0]["small_sample"] is True


def test_consensus_outcome_rows_keep_only_agreed_assets():
    record = {"assets": {
        "META": _asset_summary("META", "bullish", bulls="ABC", move=0.171, bench=0.016, right=True),
        "ETH": _asset_summary("ETH", "bearish", bears="AB", move=-0.097, bench=-0.041, bench_label="Bitcoin",
                              since=date(2026, 9, 22), right=True),
        "NFLX": _asset_summary("NFLX", "bullish", bulls="BC", move=-0.084, bench=0.016, right=False),
        "S&P 500": _asset_summary("S&P 500 (SPY)", "bullish", bulls="AD", move=0.016, bench=None, bench_label=None,
                                  right=True),                                  # the market itself
        "NVDA": _asset_summary("NVDA", "split", bulls="AB", bears="CD", move=0.02, bench=0.01),   # they disagree
        "AAPL": _asset_summary("AAPL", "bullish", bulls="A", move=0.05, bench=0.01, right=True),   # one creator
        "SPCX": _asset_summary("SPCX", None),                                     # nothing scored yet
    }}
    rows = pc.consensus_outcome_rows(record)
    assert [(r["label"], r["direction"], r["n_creators"]) for r in rows] == [
        ("META", "bullish", 3), ("ETH", "bearish", 2), ("NFLX", "bullish", 2), ("S&P 500 (SPY)", "bullish", 2)]
    meta, eth, nflx, spx = rows
    assert set(meta) == {"label", "direction", "n_creators", "move", "bench_move", "benchmark_label", "since", "right"}
    assert (meta["move"], meta["bench_move"]) == (pytest.approx(17.1), pytest.approx(1.6))   # percent
    assert (meta["benchmark_label"], meta["since"], meta["right"]) == ("S&P 500", date(2026, 9, 8), True)
    assert (eth["benchmark_label"], eth["since"], eth["right"]) == ("Bitcoin", date(2026, 9, 22), True)
    assert nflx["right"] is False and nflx["move"] == pytest.approx(-8.4)
    assert spx["bench_move"] is None and spx["benchmark_label"] is None
    assert pc.consensus_outcome_rows(record, limit=2) == rows[:2]
    assert pc.consensus_outcome_rows(None) == []
    # Two channels of one house (HKCM and Phantom by HKCM) are one opinion, not a shared call.
    record["assets"]["ETH"]["houses"] = 1
    assert "ETH" not in [r["label"] for r in pc.consensus_outcome_rows(record)]


def test_tracker_titles_state_the_takeaway_without_ranking():
    row = lambda value, small=False: {"label": "C", "median_dir_excess": value, "small_sample": small}
    five = [row(1.0), row(-0.5), row(0.2), row(-2.0), row(3.0), row(0.6)]
    titles = [pc.creators_vs_market_title(five + [row(9.0, small=True)])]   # faded rows are not counted
    assert titles[-1] == "3 of 6 with 10+ calls ahead of the market"   # +0.2 points is level, not ahead
    titles.append(pc.creators_vs_market_title([row(0.0), row(-1.0), row(5.0, small=True)]))
    assert titles[-1] == "0 of 2 with 10+ calls ahead of the market"
    titles.append(pc.creators_vs_market_title([row(5.0, small=True)] * 3))
    assert titles[-1] == "Creators' calls vs the market so far"       # no full sample: no count
    titles.append(pc.creators_vs_market_title([row(1.0)] * 10))
    assert titles[-1] == "10 of 10 with 10+ calls ahead"               # shortened to fit
    outcomes = [{"right": True}, {"right": False}, {"right": True}]
    titles.append(pc.consensus_outcomes_title(outcomes))
    assert titles[-1] == "2 of 3 shared calls moving their way"
    titles.append(pc.consensus_outcomes_title(outcomes[:1]))
    assert titles[-1] == "1 of 1 shared call moving their way"
    titles.append(pc.consensus_outcomes_title([]))
    data = _rich_tracker_data()
    titles += [pc.creators_vs_market_title(data["creators"]), pc.consensus_outcomes_title(data["outcomes"])]
    # Couch Investor's +0.03 points is level with the market, not ahead of it.
    assert titles[-2:] == ["0 of 3 with 10+ calls ahead of the market", "3 of 4 shared calls moving their way"]
    for title in titles:
        assert len(title) <= pc.TITLE_MAX_CHARS == 42, title
        for word in RANKING_WORDS:
            assert word not in title.lower(), (word, title)


def test_build_tracker_chart_data_from_a_real_record(monkeypatch):
    monkeypatch.setattr(ct, "MIN_CREATOR_SAMPLE", 4)        # Alpha and Beta have full samples, Gamma does not
    record = _tracker_record()
    assert record["stats"]["scored"] == len(TRACKER_CALLS)
    data = pc.build_tracker_chart_data(record)
    assert set(data) == {"as_of", "since", "stale", "creators", "not_charted", "outcomes"}
    assert (data["as_of"], data["since"], data["stale"]) == (AS_OF, date(2026, 9, 8), False)
    assert [r["label"] for r in data["creators"]] == ["Alpha", "Beta", "Gamma"]
    for row in data["creators"]:
        summary = record["creators"][row["label"]]
        assert (row["calls"], row["bench_n"], row["beat"]) == (summary["calls"], summary["bench_n"], summary["beat"])
        assert row["median_dir_excess"] == pytest.approx(100 * summary["median_dir_excess"])
        assert row["small_sample"] == summary["small_sample"]
    assert [r["small_sample"] for r in data["creators"]] == [False, False, True]
    assert data["not_charted"] == [{"label": "Delta", "calls": 2, "reason": "too_few"},
                                   {"label": "Crypto", "calls": 1, "reason": "too_few"}]
    labels = {a: record["assets"][a]["label"] for a in ("NVDA", "INTC", "META")}
    expected = sorted([(labels["NVDA"], "bullish", 3), (labels["INTC"], "bearish", 2), (labels["META"], "bullish", 2)],
                      key=lambda t: (-t[2], t[0]))
    assert [(r["label"], r["direction"], r["n_creators"]) for r in data["outcomes"]] == expected
    nvda = next(r for r in data["outcomes"] if r["label"] == labels["NVDA"])
    summary = record["assets"]["NVDA"]
    assert nvda["move"] == pytest.approx(100 * summary["move"]) and nvda["since"] == summary["since"] == date(2026, 9, 8)
    assert nvda["bench_move"] == pytest.approx(100 * summary["bench_move"]) and nvda["benchmark_label"] == "S&P 500"
    intc = next(r for r in data["outcomes"] if r["label"] == labels["INTC"])
    assert intc["move"] > 0 and intc["right"] is False                # INTC rose against the bearish call
    ahead = sum(r["median_dir_excess"] > pc.LEVEL_BAND_POINTS for r in data["creators"][:2])
    assert pc.creators_vs_market_title(data["creators"]) == f"{ahead} of 2 with 4+ calls ahead of the market"
    # The tracker's stale-price rule: over a week behind the requested as-of, or no S&P 500 close at all.
    assert pc.build_tracker_chart_data(dict(record, requested_as_of=AS_OF + timedelta(days=10)))["stale"] is True
    assert pc.build_tracker_chart_data(dict(record, market_data=False))["stale"] is True
    assert "(stale prices)" in pc._creators_meta(dict(data, stale=True))
    empty = pc.build_tracker_chart_data(None)
    assert empty["creators"] == [] and empty["outcomes"] == [] and empty["stale"] is False


def test_render_tracker_charts_writes_two_pngs(tmp_path, monkeypatch):
    monkeypatch.setattr(ct, "MIN_CREATOR_SAMPLE", 4)
    paths = pc.render_tracker_charts(pc.build_tracker_chart_data(_tracker_record()), str(tmp_path / "real"))
    assert _names(paths) == ["tracker-1-creators.png", "tracker-2-consensus.png"]
    paths += pc.render_tracker_charts(_rich_tracker_data(), str(tmp_path / "rich"))
    assert len(paths) == 4
    for path in paths:
        assert os.path.getsize(path) > 0
    # The pulse album is a separate list, left as it was.
    assert [f for _, f, _ in pc.CHARTS] == ["1-consensus.png", "2-tone.png", "3-upside.png", "4-flips.png",
                                            "5-map.png", "6-spread.png"]


def test_thin_tracker_data_is_left_out(tmp_path):
    thin = [("A", "NVDA", "bullish", "2026-09-08"), ("A", "META", "bullish", "2026-09-09"),
            ("A", "AMZN", "bullish", "2026-09-10"), ("B", "NVDA", "bullish", "2026-09-09"),
            ("B", "META", "bullish", "2026-09-10"), ("B", "MSFT", "bullish", "2026-09-11")]
    data = pc.build_tracker_chart_data(_tracker_record(thin))
    assert (len(data["creators"]), len(data["outcomes"])) == (2, 2)   # 2 creators, 2 shared calls
    assert pc.render_tracker_charts(data, str(tmp_path / "thin")) == []
    assert not (tmp_path / "thin").exists()                            # nothing to draw, nothing written
    # Each chart is gated on its own section.
    rich = _rich_tracker_data()
    assert _names(pc.render_tracker_charts(dict(rich, outcomes=rich["outcomes"][:2]), str(tmp_path / "a"))) == [
        "tracker-1-creators.png"]
    assert _names(pc.render_tracker_charts(dict(rich, creators=rich["creators"][:2]), str(tmp_path / "b"))) == [
        "tracker-2-consensus.png"]
    assert pc.render_tracker_charts(pc.build_tracker_chart_data({}), str(tmp_path / "c")) == []


def test_render_tracker_charts_never_raises(tmp_path, monkeypatch):
    warnings = []
    monkeypatch.setattr(pc, "log_warn", warnings.append)
    data = _rich_tracker_data()
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "matplotlib", None)                    # the plotting stack is missing
        assert pc.render_tracker_charts(data, str(tmp_path / "none")) == []
    assert "matplotlib unavailable" in warnings[-1]

    def broken(plt, data, path):
        raise RuntimeError("bad render")

    monkeypatch.setattr(pc, "TRACKER_CHARTS", [("creators", "tracker-1-creators.png", broken), pc.TRACKER_CHARTS[1]])
    paths = pc.render_tracker_charts(data, str(tmp_path / "one"))
    assert _names(paths) == ["tracker-2-consensus.png"]                # one failure never blocks the other
    assert "bad render" in warnings[-1]
    assert pc.render_tracker_charts(None, str(tmp_path)) == []
    malformed = {"creators": [{"label": "x"}] * 3, "outcomes": "junk"}
    assert pc.render_tracker_charts(malformed, str(tmp_path / "bad")) == []
