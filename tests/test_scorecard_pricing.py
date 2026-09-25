"""Item 4: the scorecard's entry rule, calendars, price provenance and benchmarks."""
from datetime import date, timedelta

import pytest

import research_analytics as ra
import scorecard_pricing as sp

US = sp.EXCHANGES["us"]


def _closes(start, days, step=1.0, skip_weekends=True, holidays=()):
    out, d, i = {}, start, 0
    while len(out) < days:
        if not (skip_weekends and d.weekday() >= 5) and d not in holidays:
            out[d] = 100.0 + step * i
            i += 1
        d += timedelta(days=1)
    return out


def _claim(published, end="2026-08-14", **over):
    c = {"claim_id": "c", "video_id": "v", "channel_name": "A", "published_at": published,
         "extracted_at": published, "ticker": "NVDA", "subject_mention": "Nvidia", "stance": "bullish",
         "horizon_bucket": "short", "review_required": False, "coverage_status": "full",
         "evidence_start_character": 1, "attribution_type": "speaker_personal_view", "repeat_of_claim_id": None,
         "is_forward_looking": True, "testable": True, "testability_type": "unconditional_testable",
         "claim_type": "forecast", "forecast_direction": "increase", "forecast_end_date": end, "asset_type": "stock",
         "schema_version": "2", "sector": None}
    c.update(over)
    return c


def _fetch(prices, adjustment="split_adjusted"):
    def fetch(symbol, start, end):
        return prices.get(symbol, {})
    fetch.price_provenance = {"provider": "test", "adjustment": adjustment, "corporate_action_status": adjustment}
    return fetch


JULY = _closes(date(2026, 7, 1), 40, holidays={date(2026, 7, 3)})


def test_video_published_before_the_close_takes_that_days_close():
    # 14:00 UTC on Thursday 2 July = 10:00 New York, during the session.
    e = sp.entry_point(US, "2026-07-02T14:00:00+00:00", JULY)
    assert e["session_relation"] == "during_session" and e["resolved_trading_date"] == "2026-07-02"
    assert e["price"] == JULY[date(2026, 7, 2)]


def test_video_published_after_the_close_never_uses_that_days_close():
    # 21:00 UTC = 17:00 New York, after the 16:00 close: that close printed
    # before the video went up, so the next session's close is the entry.
    e = sp.entry_point(US, "2026-07-02T21:00:00+00:00", JULY)
    assert e["session_relation"] == "after_close"
    assert e["resolved_trading_date"] == "2026-07-06"  # Fri 3 July is the Independence Day observance
    assert e["price"] != JULY[date(2026, 7, 2)]


def test_weekend_publication_rolls_to_the_next_trading_day():
    e = sp.entry_point(US, "2026-07-11T15:00:00+00:00", JULY)  # a Saturday
    assert e["session_relation"] == "non_trading_day" and e["resolved_trading_date"] == "2026-07-13"


def test_market_holiday_publication_rolls_past_the_holiday():
    assert date(2026, 7, 3) in sp.nyse_holidays(2026)
    e = sp.entry_point(US, "2026-07-03T15:00:00+00:00", JULY)
    assert e["session_relation"] == "non_trading_day" and e["resolved_trading_date"] == "2026-07-06"
    # A close the series happens to carry ON a holiday is never used either.
    with_bad_row = {**JULY, date(2026, 7, 3): 1.0}
    assert sp.entry_point(US, "2026-07-03T15:00:00+00:00", with_bad_row)["resolved_trading_date"] == "2026-07-06"


def test_publication_without_a_time_of_day_takes_the_next_trading_day_conservatively():
    e = sp.entry_point(US, "2026-07-02", JULY)
    assert e["session_relation"] == "time_unknown" and e["resolved_trading_date"] == "2026-07-06"
    assert "publication time unknown" in e["convention"]


def test_stock_split_unadjusted_series_is_refused_and_split_adjusted_scores():
    # A 10:1 split on 8 July: unadjusted closes fall from 1000 to 100.
    unadjusted = {d: (1000.0 + i if d < date(2026, 7, 8) else 100.0 + i) for i, d in enumerate(sorted(JULY))}
    adjusted = {d: 100.0 + i for i, d in enumerate(sorted(JULY))}
    claim = _claim("2026-07-02T14:00:00+00:00", end="2026-07-20")
    sc = ra.scorecard([claim], date(2026, 8, 1), _fetch({"nvda.us": unadjusted, "spy.us": adjusted}, "unadjusted"),
                      min_sample=1)
    assert sc["_excluded"] == {"unadjusted_or_unknown_prices": 1}
    sc = ra.scorecard([claim], date(2026, 8, 1), _fetch({"nvda.us": adjusted, "spy.us": adjusted}), min_sample=1)
    scored = sc["A"]["scored"][0]
    assert sc["A"]["direction_hits"] == 1 and scored["adjustment_type"] == "split_adjusted"
    assert scored["price_provider"] == "test" and scored["currency"] == "USD"
    assert scored["entry_requested_date"] == "2026-07-02" and scored["entry_resolved_trading_date"] == "2026-07-02"


def test_dividend_adjusted_series_is_recorded_as_total_return():
    claim = _claim("2026-07-02T14:00:00+00:00", end="2026-07-20")
    series = {d: 100.0 + i for i, d in enumerate(sorted(JULY))}
    sc = ra.scorecard([claim], date(2026, 8, 1), _fetch({"nvda.us": series, "spy.us": series}, "total_return_adjusted"),
                      min_sample=1)
    assert sc["A"]["scored"][0]["adjustment_type"] == "total_return_adjusted"
    assert sp.adjustment_label("all") == "total_return_adjusted" and sp.adjustment_label("splits") == "split_adjusted"


def test_cryptocurrency_uses_the_utc_day_close_and_no_session():
    closes = {date(2026, 7, 4) + timedelta(days=i): 50000.0 + i for i in range(30)}  # 24/7: weekends included
    e = sp.entry_point(sp.CRYPTO, "2026-07-04T23:30:00+00:00", closes)  # a Saturday, late evening UTC
    assert e["resolved_trading_date"] == "2026-07-04" and e["session_relation"] == "during_session"
    claim = _claim("2026-07-04T23:30:00+00:00", end="2026-07-20", ticker="BTC", asset_type="crypto", subject_mention="Bitcoin")
    sc = ra.scorecard([claim], date(2026, 8, 1), _fetch({"btcusd": closes}), min_sample=1)
    scored = sc["A"]["scored"][0]
    assert scored["exchange"] == "CRYPTO" and scored["entry_resolved_trading_date"] == "2026-07-04"
    # BTC benchmarked against BTC is meaningless: raw only, benchmark null.
    assert scored["benchmark_method"] == "self_benchmark" and scored["excess_return"] is None
    assert sc["A"]["excess_returns"] == []


def test_unresolved_exchange_excludes_the_claim(monkeypatch):
    import instruments
    assert sp.resolve_exchange("nvda.xyz") is None and sp.resolve_exchange("", "stock") is None
    claim = _claim("2026-07-02T14:00:00+00:00", end="2026-07-20", asset_type="commodity", ticker="GOLD")
    sc = ra.scorecard([claim], date(2026, 8, 1), _fetch({}), min_sample=1)
    assert sc["_excluded"] == {"unresolved_instrument": 1}  # no instrument metadata at all
    sc = ra.scorecard([_claim("2026-07-02T14:00:00+00:00", end="2026-07-20", asset_type="stock", ticker="NVDA")],
                      date(2026, 8, 1), _fetch({}), min_sample=1)
    assert sc["_excluded"] == {"no_prices": 1}
    # A resolved instrument on a venue without session rules is refused explicitly.
    odd = instruments.Instrument("XXXX:NVDA", "NVDA", "nvda.unknownvenue", "unknownvenue", "XXXX", "stock",
                                 "ZZ", "ZZZ", None)
    monkeypatch.setattr(instruments, "resolve_instrument", lambda *a, **k: odd)
    sc = ra.scorecard([claim], date(2026, 8, 1), _fetch({}), min_sample=1)
    assert sc["_excluded"] == {"unresolved_exchange": 1}


def test_unresolved_benchmark_gives_raw_performance_and_null_benchmark_metrics():
    lse = sp.EXCHANGES["lse"]
    assert sp.resolve_benchmark("stock", lse) == (None, "no_benchmark_for_asset_type")
    assert sp.resolve_benchmark("stock", None) == (None, "unresolved_exchange")
    assert sp.resolve_benchmark("stock", US, "Technology") == ("xlk.us", "sector")
    assert sp.resolve_benchmark("stock", US, "made-up sector") == ("spy.us", "country")
    claim = _claim("2026-07-02T14:00:00+00:00", end="2026-07-20", asset_type="etf", ticker="SPY")
    series = {d: 100.0 + i for i, d in enumerate(sorted(JULY))}
    sc = ra.scorecard([claim], date(2026, 8, 1), _fetch({"spy.us": series}), min_sample=1)
    scored = sc["A"]["scored"][0]
    assert scored["raw_return"] > 0 and scored["excess_return"] is None and scored["benchmark_return"] is None
    assert scored["benchmark_symbol"] is None and scored["benchmark_method"] == "self_benchmark"


def test_benchmark_table_is_configurable(monkeypatch):
    monkeypatch.setenv("BENCHMARKS_JSON", '{"stock|GB|": "isf.lse"}')
    assert sp.resolve_benchmark("stock", sp.EXCHANGES["lse"]) == ("isf.lse", "country")


def test_sources_under_different_benchmark_methods_are_not_ranked_together(monkeypatch):
    monkeypatch.setattr(sp, "SCORECARD_RANKINGS_ENABLED", True)
    sc = {"A": {"n": 3, "benchmark_methods": {"country": 3}}, "B": {"n": 3, "benchmark_methods": {"none": 3}},
          "C": {"n": 3, "benchmark_methods": {"country": 3}}, "D": {"n": 1, "benchmark_methods": {"country": 1}}}
    assert ra.rankable_sources(sc, 2) == ["A", "C"]
    monkeypatch.setattr(sp, "SCORECARD_RANKINGS_ENABLED", False)
    assert ra.rankable_sources(sc, 2) == []


def test_rankings_are_disabled_by_default_and_the_report_says_experimental():
    assert sp.SCORECARD_RANKINGS_ENABLED is False
    claim = _claim("2026-07-02T14:00:00+00:00", end="2026-07-20")
    series = {d: 100.0 + i for i, d in enumerate(sorted(JULY))}
    text = ra.build_report([claim], [], {"videos": {}}, [], date(2026, 6, 30), date(2026, 8, 1), today=date(2026, 8, 1),
                           price_fetcher=_fetch({"nvda.us": series, "spy.us": series}))
    assert sp.SCORECARD_EXPERIMENTAL_LABEL in text and "(unranked)" in text
    assert ra.SCORECARD_RULES["status"].startswith("experimental")


def test_conditional_forecasts_stay_out_of_the_unconditional_scorecard():
    series = {d: 100.0 + i for i, d in enumerate(sorted(JULY))}
    cond = _claim("2026-07-02T14:00:00+00:00", end="2026-07-20", testability_type="conditional_testable",
                  condition="if the Fed cuts", condition_status="met")
    sc = ra.scorecard([cond], date(2026, 8, 1), _fetch({"nvda.us": series, "spy.us": series}), min_sample=1)
    assert sc["_excluded"] == {"conditional": 1}
    sc = ra.scorecard([cond], date(2026, 8, 1), _fetch({"nvda.us": series, "spy.us": series}), min_sample=1,
                      include_conditional=True)
    assert sc["A"]["n"] == 1
    unmet = dict(cond, condition_status="not_met")
    sc = ra.scorecard([unmet], date(2026, 8, 1), _fetch({"nvda.us": series}), min_sample=1, include_conditional=True)
    assert sc["_excluded"] == {"condition_not_met": 1}
