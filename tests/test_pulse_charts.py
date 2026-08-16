"""Tests for the weekly pulse charts: pure data prep plus a rendering and
send-wiring smoke test (matplotlib is a runtime dependency, so rendering runs
in CI too; only the network stays mocked)."""
import json
from datetime import date

import market_pulse as mp
import pulse_charts as pc


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
                                 _asset("Apple", "AAPL")]),  # no target -> excluded
    ])
    rows = pc.upside_rows(current, {"RDDT": 177.0, "AVGO": 390.0})
    assert [r["label"] for r in rows] == ["AVGO", "RDDT"]  # biggest implied move first
    assert round(rows[1]["pct"], 1) == 14.7


def test_render_charts_writes_pngs(tmp_path):
    records = [
        _rec("2026-08-10", "A", [_asset(stance="bullish", price_target=500)], "bullish"),
        _rec("2026-08-11", "B", [_asset(stance="bullish")], "bullish"),
        _rec("2026-08-12", "C", [_asset(stance="bearish")], "bearish"),
        _rec("2026-08-04", "A", [_asset(stance="bearish")], "bearish"),
        _rec("2026-08-05", "B", [_asset(stance="bearish")], "neutral"),
    ]
    window_start = date(2026, 8, 9)
    today = date(2026, 8, 16)
    current = _agg([r for r in records if mp._in_window(r, window_start, today)])
    previous = _agg([r for r in records if mp._in_window(r, window_start.replace(day=2), window_start)])
    data = pc.build_chart_data(records, current, previous, window_start, today)
    data["upside"] = pc.upside_rows(current, {"TSLA": 400.0})
    paths = pc.render_charts(data, str(tmp_path))
    assert len(paths) == 5  # all five charts had data
    for path in paths:
        assert path.endswith(".png")
        assert (tmp_path / path.split("/")[-1]).stat().st_size > 0


def test_render_charts_skips_empty_sections(tmp_path):
    data = {"window": "Aug 10 - 16, 2026", "videos": 1, "channels": 1,
            "consensus": [], "flips": [], "tone": [], "map": [], "upside": []}
    assert pc.render_charts(data, str(tmp_path)) == []


def test_generate_charts_never_raises(tmp_path, monkeypatch):
    # Even a corrupt dataset path must degrade to "no charts", not an error.
    monkeypatch.setattr(pc, "build_chart_data", lambda *a, **k: 1 / 0)
    path = tmp_path / "signals.jsonl"
    path.write_text(json.dumps(_rec("2026-08-10", "A", [_asset()])) + "\n", encoding="utf-8")
    assert mp.generate_charts(today=date(2026, 8, 16), path=str(path),
                              price_fetcher=lambda *a: {}, out_dir=str(tmp_path)) == []
