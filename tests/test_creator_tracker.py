"""Creators vs the market (creator_tracker.py): every creator's calls marked
to market against what the market did over the same days.

Offline throughout: prices come from static_fetcher (or a price cache the
test installs), never the network."""
import json
import os
import re
import subprocess
import sys
from datetime import date, timedelta

import pytest

import canonical_claims
import creator_tracker as ct
import price_cache
import scorecard_pricing as sp

US = sp.EXCHANGES["us"]
START, END = date(2026, 8, 24), date(2026, 10, 9)                  # Monday .. Friday
BEFORE_OPEN, AFTER_CLOSE = "T11:45:00+00:00", "T21:00:00+00:00"    # 07:45 / 17:00 New York (EDT)

RECORD_KEYS = {"requested_as_of", "as_of", "market_data", "since", "calls", "excluded", "claim_exclusions",
               "stats", "creators", "pooled", "assets", "market"}
CALL_KEYS = {
    "creator", "asset", "label", "ticker", "symbol", "asset_class", "resolution",
    "direction", "call_date", "published_at", "video_id", "title", "videos", "claim_ids", "guest_share", "status",
    "entry_date", "entry_price", "mark_date", "mark_price", "days", "ret", "dir_ret", "right",
    "benchmark", "benchmark_label", "bench_ret", "excess", "dir_excess", "beat",
    "w28_dir_ret", "w28_dir_excess",
    "target", "target_date", "target_progress", "target_reached", "target_reached_date", "target_method",
    "scored", "reason",
}
CREATOR_KEYS = {"calls", "bull", "bear", "right", "bench_n", "beat", "median_dir_ret", "median_dir_excess",
                "mean_dir_excess", "median_days", "w28_n", "w28_right", "w28_bench_n", "w28_beat", "best", "worst",
                "small_sample", "mostly_guests", "unscored"}
POOLED_KEYS = {"calls", "right", "bench_n", "beat", "w28_n", "w28_right", "w28_bench_n", "w28_beat", "market_return"}
ASSET_KEYS = {"asset", "label", "symbol", "current", "bulls", "bears", "consensus", "houses", "since", "move",
              "benchmark_label", "bench_move", "dir_excess", "right", "beat"}
MARKET_KEYS = {"symbol", "label", "as_of", "close", "week", "week_from", "four_weeks", "since", "since_date"}
OUTCOME_KEYS = {"asset", "label", "symbol", "direction", "creators", "pending", "since", "entry_price", "move",
                "benchmark_label", "bench_move", "dir_excess", "right", "beat", "targets"}
REASONS = {"no_ticker", "unresolved_instrument", "no_prices", "too_recent", "no_entry_close",
           "no_post_call_close", "stale_mark", "younger_than_min_age", "reversed_within_min_age"}


def _days(start=START, end=END, crypto=False):
    out, day = [], start
    while day <= end:
        if crypto or sp.is_trading_day(US, day):
            out.append(day)
        day += timedelta(days=1)
    return out


def _line(first, step, start=START, end=END, crypto=False):
    """Closes on every session (every day for crypto), moving `step` a day."""
    return {d: round(first + i * step, 6) for i, d in enumerate(_days(start, end, crypto))}


def _market(**extra):
    """spy.us rising 1 a session from 500; btcusd and ethusd every day."""
    data = {"spy.us": _line(500, 1), "btcusd": _line(80000, 100, crypto=True),
            "ethusd": _line(2000, -5, crypto=True)}
    data.update(extra)
    return data


def _claim(cid, video, day, stance, *, at=BEFORE_OPEN, channel="Chan", ticker="NVDA", name="Nvidia",
           ctype="recommendation", metric=None, condition=None, target=None, end=None, asset_type="stock",
           review=False, attribution="speaker_personal_view", position=1, title="Stock analysis"):
    return {
        "claim_id": cid, "video_id": video, "published_at": f"{day}{at}", "extracted_at": f"{day}T23:00:00+00:00",
        "channel_name": channel, "video_title": title, "stance": stance, "ticker": ticker,
        "canonical_entity_name": name, "subject_mention": name, "asset_type": asset_type,
        "claim_type": ctype, "forecast_metric": metric, "condition": condition,
        "attribution_type": attribution, "coverage_status": "full", "evidence_start_character": position,
        "review_required": review, "review_reasons": ["test"] if review else [], "schema_version": "2",
        "repeat_of_claim_id": None, "forecast_direction": None, "forecast_end_date": end,
        "target_kind": "absolute_value" if target else None, "target_value": target,
        "currency": "USD", "target_unit": "USD" if target else None, "evidence_text": "e",
    }


def _record(claims, as_of=END, prices=None, highs=None, lows=None, **kw):
    prices = prices if prices is not None else _market(**{"nvda.us": _line(100, 2)})
    fetcher = ct.static_fetcher(prices, highs, lows)
    return ct.track_record(claims, as_of, fetcher, **kw), fetcher


def _row(record, **match):
    rows = [r for r in record["calls"] if all(r[k] == v for k, v in match.items())]
    assert len(rows) == 1, rows
    return rows[0]


def _cache(prices):
    return {s: {"from": min(c), "to": max(c), "closes": dict(c), "adjustment": "split_adjusted"}
            for s, c in prices.items()}


# --- Calendar and which claims make calls ---------------------------------------------------


def test_default_as_of_is_the_last_completed_nyse_session():
    assert ct.default_as_of(date(2026, 10, 19)) == date(2026, 10, 16)   # Monday run -> Friday
    assert ct.default_as_of(date(2026, 10, 16)) == date(2026, 10, 15)   # Friday 16:00 UTC run -> Thursday
    assert ct.default_as_of(date(2026, 10, 18)) == date(2026, 10, 16)   # Sunday warm -> Friday
    assert ct.default_as_of(date(2026, 9, 8)) == date(2026, 9, 4)       # Labor Day Monday is no session


def test_video_call_is_the_two_thirds_lean_and_mixed_videos_make_no_call():
    mixed = [_claim("a", "v1", "2026-09-08", "bullish"),
             _claim("b", "v1", "2026-09-08", "bearish", ctype="price_target", target=80, position=2)]
    calls, excluded = ct.video_calls(mixed)
    assert calls == [] and excluded["video_mixed"] == 1
    two_thirds = mixed + [_claim("c", "v1", "2026-09-08", "bullish", ctype="valuation_view", position=3)]
    earlier = [_claim("x", "v0", "2026-09-07", "bearish", ticker="META", name="Meta")]
    calls, excluded = ct.video_calls(two_thirds + earlier)
    assert excluded["video_mixed"] == 0 and [c["video_id"] for c in calls] == ["v0", "v1"]   # published order
    call = calls[1]
    assert set(call) == {"creator", "asset", "video_id", "published_at", "date", "title", "direction",
                         "claim_ids", "horizon_end", "guest_share"}
    assert (call["creator"], call["asset"], call["direction"]) == ("Chan", "NVDA", "bullish")
    assert call["claim_ids"] == ["a", "b", "c"] and call["date"] == date(2026, 9, 8)
    assert call["published_at"] == "2026-09-08T11:45:00+00:00" and call["guest_share"] == 0
    # 3 of 5 is not two thirds: still mixed
    five = two_thirds + [_claim("d", "v1", "2026-09-08", "bearish", position=4)]
    assert ct.video_calls(five)[1]["video_mixed"] == 1


def test_fundamental_forecasts_risks_and_trigger_conditions_are_not_calls_but_qualifiers_are():
    claims = [
        _claim("rev", "v1", "2026-09-08", "bearish", ctype="forecast", metric="revenue"),   # a business metric
        _claim("risk", "v2", "2026-09-09", "bearish", ctype="risk"),
        _claim("trig", "v3", "2026-09-10", "bullish", condition="if it drops to $600"),
        _claim("qual", "v4", "2026-09-11", "bullish", condition="as long as $76,220 holds"),
        _claim("flat", "v5", "2026-09-12", "neutral"),
    ]
    calls, excluded = ct.video_calls(claims)
    assert [c["video_id"] for c in calls] == ["v4"]
    assert excluded["not_a_security_view"] == 2 and excluded["trigger_condition"] == 1
    assert excluded["not_directional"] == 1 and excluded["no_asset"] == 0
    assert ct.is_security_view({"claim_type": "forecast", "forecast_metric": "price"})
    assert ct.is_security_view({"claim_type": "opinion", "forecast_metric": "market_cap"})
    assert ct.is_security_view({"claim_type": "price_target", "forecast_metric": None})
    assert not ct.is_security_view({"claim_type": "forecast", "forecast_metric": "earnings"})
    assert not ct.is_security_view({"claim_type": "stance", "forecast_metric": "margin"})
    assert not ct.is_security_view({"claim_type": "catalyst", "forecast_metric": None})
    for qualifier in ("as long as support holds", "for investors with a high risk tolerance",
                      "unless something major happens"):
        assert not ct.is_trigger_condition({"condition": qualifier}), qualifier
    assert ct.is_trigger_condition({"condition": "on a pullback to 150"})
    assert not ct.is_trigger_condition({"condition": None}) and not ct.is_trigger_condition({})


def test_review_required_claims_that_flip_the_video_lean_drop_the_call():
    flipped = [_claim("a", "v1", "2026-09-08", "bearish"),
               _claim("b", "v1", "2026-09-08", "bullish", ctype="price_target", target=150, review=True, position=2),
               _claim("c", "v1", "2026-09-08", "bullish", ctype="price_target", target=160, review=True, position=3)]
    calls, excluded = ct.video_calls(flipped)
    assert calls == [] and excluded["video_not_review_robust"] == 1
    # A review-required claim that agrees with the vote leaves it standing ...
    agreeing = [_claim("d", "v2", "2026-09-09", "bullish"), _claim("e", "v2", "2026-09-09", "bullish", position=2),
                _claim("f", "v2", "2026-09-09", "bearish", review=True, position=3)]
    # ... and a quoted third party is not the creator's own view, flagged or not.
    quoted = [_claim("g", "v3", "2026-09-10", "bearish"),
              _claim("h", "v3", "2026-09-10", "bullish", review=True, attribution="speaker_quoting_third_party",
                     position=2),
              _claim("i", "v3", "2026-09-10", "bullish", review=True, attribution="speaker_quoting_third_party",
                     position=3)]
    calls, excluded = ct.video_calls(agreeing + quoted)
    assert [(c["video_id"], c["direction"]) for c in calls] == [("v2", "bullish"), ("v3", "bearish")]
    assert excluded["video_not_review_robust"] == 0
    assert calls[0]["claim_ids"] == ["d", "e"]          # review-required claims never join the call


# --- Episodes, entry and mark ----------------------------------------------------------------------


def test_reiteration_keeps_the_entry_and_reversal_closes_at_the_reversal_close():
    claims = [_claim("a", "v1", "2026-09-08", "bullish"),                    # before the open: Sep 8 close
              _claim("b", "v2", "2026-09-09", "bullish", at=AFTER_CLOSE),   # a reiteration
              _claim("c", "v3", "2026-09-17", "bearish", at=AFTER_CLOSE)]   # a reversal: Sep 18 close
    record, _ = _record(claims)
    nvda = _line(100, 2)
    first, second = record["calls"]
    assert record["stats"]["reiterations"] == 1 and record["stats"]["reversals"] == 1
    assert record["stats"]["video_calls"] == 3 and record["stats"]["calls"] == 2
    assert first["status"] == "reversed" and first["videos"] == 2 and first["claim_ids"] == ["a", "b"]
    assert first["entry_date"] == date(2026, 9, 8) and first["entry_price"] == nvda[date(2026, 9, 8)]
    assert first["mark_date"] == date(2026, 9, 18) and first["mark_price"] == nvda[date(2026, 9, 18)]
    assert first["days"] == 10 and first["scored"]
    assert first["ret"] == pytest.approx(nvda[date(2026, 9, 18)] / nvda[date(2026, 9, 8)] - 1) and first["right"]
    # The reversal opens a bearish call from that same close, open at the as-of.
    assert second["direction"] == "bearish" and second["status"] == "open" and second["videos"] == 1
    assert second["entry_date"] == date(2026, 9, 18) and second["mark_date"] == END
    assert second["dir_ret"] == pytest.approx(-(nvda[END] / nvda[date(2026, 9, 18)] - 1))
    assert second["right"] is False and second["scored"]
    # Reversed within MIN_CALL_AGE_DAYS: scenario talk, priced but never aggregated.
    quick = claims[:2] + [_claim("c", "v3", "2026-09-10", "bearish", at=AFTER_CLOSE)]   # Sep 11 close
    record, _ = _record(quick)
    first = record["calls"][0]
    assert first["status"] == "reversed" and first["days"] == 3 and first["mark_date"] == date(2026, 9, 11)
    assert first["reason"] == "reversed_within_min_age" and not first["scored"] and first["ret"] is not None
    assert record["excluded"] == {"reversed_within_min_age": 1} and record["stats"]["scored"] == 1
    assert "reversed within 7 days 1" in "\n".join(ct.format_report(record))


def test_matured_horizon_closes_the_call_at_the_evaluation_close():
    claims = [_claim("a", "v1", "2026-09-08", "bullish", ctype="price_target", target=150, end="2026-09-26"),
              _claim("b", "v2", "2026-10-01", "bullish")]     # after the horizon: a new call, not a reiteration
    record, _ = _record(claims)
    nvda = _line(100, 2)
    matured, fresh = record["calls"]
    assert matured["status"] == "matured" and matured["videos"] == 1 and matured["scored"]
    assert matured["mark_date"] == date(2026, 9, 28)           # the horizon ends on a Saturday: Monday's close
    assert matured["mark_price"] == nvda[date(2026, 9, 28)]
    assert fresh["status"] == "open" and fresh["entry_date"] == date(2026, 10, 1)
    assert record["stats"]["matured"] == 1 and record["stats"]["reiterations"] == 0
    # Before the horizon ends the call is open, and a video after the as-of is too recent.
    early, _ = _record(claims, as_of=date(2026, 9, 25))
    first, second = early["calls"]
    assert first["status"] == "open" and first["mark_date"] == date(2026, 9, 25) and first["scored"]
    assert second["reason"] == "too_recent" and early["stats"]["matured"] == 0


def test_entry_is_the_next_close_after_publication_and_crypto_uses_the_utc_close():
    cases = {   # creator: (published, entry close)
        "Morning": ("2026-09-09T11:45:00+00:00", date(2026, 9, 9)),   # before the open: that day's close
        "Midday": ("2026-09-09T17:00:00+00:00", date(2026, 9, 9)),    # during the session: that day's close
        "Evening": ("2026-09-09T21:00:00+00:00", date(2026, 9, 10)),  # after the close: the next one
        "Weekend": ("2026-09-12T15:00:00+00:00", date(2026, 9, 14)),  # Saturday: Monday's close
        "Holiday": ("2026-09-04T21:00:00+00:00", date(2026, 9, 8)),   # Friday evening before Labor Day
    }
    claims = [dict(_claim(name, name, "2026-09-01", "bullish", channel=name), published_at=published)
              for name, (published, _) in cases.items()]
    claims.append(dict(_claim("btc", "vb", "2026-09-12", "bullish", channel="Crypto", ticker="BTC", name="Bitcoin",
                              asset_type="crypto"), published_at="2026-09-12T23:30:00+00:00"))
    record, _ = _record(claims)
    nvda = _line(100, 2)
    for name, (_, entry) in cases.items():
        row = _row(record, creator=name)
        assert (row["entry_date"], row["entry_price"]) == (entry, nvda[entry]), name
    btc = _row(record, creator="Crypto")   # a Saturday-night video: that UTC day's close
    assert btc["entry_date"] == date(2026, 9, 12)
    assert btc["entry_price"] == _market()["btcusd"][date(2026, 9, 12)]


def test_effective_as_of_is_the_last_spy_close_and_crypto_is_marked_on_the_same_date():
    prices = _market(btcusd=_line(80000, 100, end=date(2026, 10, 11), crypto=True))   # crypto runs to Sunday
    claims = [_claim("b", "v1", "2026-09-10", "bullish", ticker="BTC", name="Bitcoin", asset_type="crypto")]
    record, _ = _record(claims, as_of=date(2026, 10, 16), prices=prices)
    assert record["requested_as_of"] == date(2026, 10, 16)
    assert record["as_of"] == date(2026, 10, 9) and record["market_data"] is True
    btc = record["calls"][0]
    assert btc["mark_date"] == date(2026, 10, 9) and btc["scored"]   # not Sunday's in-progress candle
    assert {row["as_of"] for row in record["market"]} == {date(2026, 10, 9)}
    # No S&P 500 close in the 14 days before the as-of: no market date, the requested one stands.
    stale = _market(**{"spy.us": _line(500, 1, end=date(2026, 9, 25))})
    record, _ = _record(claims, as_of=date(2026, 10, 16), prices=stale)
    assert record["market_data"] is False and record["as_of"] == date(2026, 10, 16)
    assert record["calls"][0]["reason"] == "stale_mark"


def test_too_recent_stale_and_young_calls_are_counted_not_aggregated():
    prices = _market(**{"nvda.us": _line(100, 2), "amzn.us": _line(200, 1),
                        "meta.us": _line(600, 3, end=date(2026, 9, 25))})
    claims = [
        _claim("old", "v1", "2026-09-08", "bullish"),                                    # scored
        _claim("young", "v2", "2026-10-06", "bullish", ticker="AMZN", name="Amazon"),    # 3 days old at the as-of
        _claim("stale", "v3", "2026-09-08", "bullish", ticker="META", name="Meta"),      # prices stop on Sep 25
        _claim("late", "v4", "2026-10-09", "bearish", ticker="AMZN", name="Amazon", channel="Other",
               at=AFTER_CLOSE),                                                          # after the last close
        _claim("future", "v5", "2026-10-14", "bullish", channel="Other"),                # after the as-of
    ]
    record, fetcher = _record(claims, prices=prices)
    young = _row(record, asset="AMZN", creator="Chan")
    assert young["reason"] == "younger_than_min_age" and young["scored"] is False
    assert young["days"] == 3 and young["ret"] is not None and young["dir_excess"] is not None   # keeps its returns
    stale = _row(record, asset="META")
    assert stale["reason"] == "stale_mark" and stale["mark_date"] == date(2026, 9, 25) and stale["ret"] is None
    assert _row(record, asset="AMZN", creator="Other")["reason"] == "too_recent"
    assert _row(record, asset="NVDA", creator="Other")["reason"] == "too_recent"
    assert record["excluded"] == {"too_recent": 2, "stale_mark": 1, "younger_than_min_age": 1}
    assert sum(record["excluded"].values()) == record["stats"]["calls"] - record["stats"]["scored"]
    assert record["stats"]["scored"] == 1 and record["pooled"]["calls"] == 1
    assert record["creators"]["Chan"]["calls"] == 1 and record["creators"]["Chan"]["unscored"] == 2
    assert "Other" not in record["creators"]
    # A call published after the as-of is never priced: no request starts after it.
    assert all(start <= END for _, start, _ in fetcher.requests)


# --- Benchmarks, instruments and targets ---------------------------------------------------------------


def test_benchmark_on_identical_dates():
    prices = _market(**{"nvda.us": _line(100, 2), "gld.us": _line(300, -1)})
    claims = [
        _claim("n", "v1", "2026-09-08", "bullish", channel="A"),
        _claim("e", "v2", "2026-09-08", "bearish", channel="A", ticker="ETH", name="Ethereum", asset_type="crypto"),
        _claim("b", "v3", "2026-09-08", "bullish", channel="A", ticker="BTC", name="Bitcoin", asset_type="crypto"),
        _claim("s", "v4", "2026-09-08", "bullish", channel="A", ticker="SPY", name="SPY", asset_type="etf"),
        _claim("x", "v5", "2026-09-08", "bearish", channel="A", ticker=None, name="S&P 500", asset_type="index"),
        _claim("g", "v6", "2026-09-08", "bullish", channel="A", ticker=None, name="Gold", asset_type="commodity"),
    ]
    record, _ = _record(claims, prices=prices)
    spy, btc = prices["spy.us"], prices["btcusd"]
    nvda = _row(record, asset="NVDA")
    assert (nvda["benchmark"], nvda["benchmark_label"]) == ("spy.us", "S&P 500")
    assert nvda["bench_ret"] == pytest.approx(spy[nvda["mark_date"]] / spy[nvda["entry_date"]] - 1)
    assert nvda["excess"] == pytest.approx(nvda["ret"] - nvda["bench_ret"])
    assert nvda["dir_excess"] == pytest.approx(nvda["excess"]) and nvda["beat"] == (nvda["dir_excess"] > 0)
    eth = _row(record, asset="ETH")
    assert (eth["benchmark"], eth["benchmark_label"]) == ("btcusd", "Bitcoin")
    assert eth["bench_ret"] == pytest.approx(btc[eth["mark_date"]] / btc[eth["entry_date"]] - 1)
    assert eth["dir_excess"] == pytest.approx(-(eth["ret"] - eth["bench_ret"]))   # bearish: signed like dir_ret
    for asset in ("BTC", "SPY", "S&P 500"):                    # the market itself: raw return only
        row = _row(record, asset=asset)
        assert row["scored"] and row["benchmark"] is None and row["bench_ret"] is None, asset
        assert row["dir_excess"] is None and row["beat"] is None, asset
    sp500 = _row(record, asset="S&P 500")
    assert (sp500["symbol"], sp500["resolution"], sp500["label"]) == ("spy.us", "proxy", "S&P 500 (SPY)")
    assert sp500["right"] is False                            # bearish while SPY rose
    gold = _row(record, asset="GOLD")
    assert (gold["symbol"], gold["resolution"], gold["benchmark"]) == ("gld.us", "proxy", "spy.us")
    assert gold["bench_ret"] == pytest.approx(spy[gold["mark_date"]] / spy[gold["entry_date"]] - 1)
    # A missing benchmark close leaves the comparison empty; the call still counts.
    del prices["spy.us"][date(2026, 9, 8)]
    record, _ = _record(claims, prices=prices)
    nvda = _row(record, asset="NVDA")
    assert nvda["scored"] and nvda["benchmark"] == "spy.us" and nvda["bench_ret"] is None and nvda["beat"] is None


def test_unregistered_ticker_is_unresolved_even_if_a_series_exists():
    prices = _market(**{"zorb.us": _line(10, 0.1)})
    claims = [_claim("z", "v1", "2026-09-08", "bullish", ticker="ZORB", name="Zorb"),
              _claim("y", "v2", "2026-09-08", "bullish", ticker=None, name="Zorblax Industries")]
    record, fetcher = _record(claims, prices=prices)
    zorb = _row(record, asset="ZORB")
    assert zorb["reason"] == "unresolved_instrument" and zorb["symbol"] is None and zorb["ticker"] == "ZORB"
    assert _row(record, asset="ZORBLAX INDUSTRIES")["reason"] == "no_ticker"
    assert {s for s, _, _ in fetcher.requests} == {s for s, _ in ct.CONTEXT_SYMBOLS}    # nothing else fetched
    assert ct.instrument_for("ZORB", "ZORB", "Zorb", "stock") is None


def test_instrument_for_prices_proxies_and_the_registry_only():
    gold = ct.instrument_for("GOLD")
    assert gold == {"symbol": "gld.us", "label": "Gold (GLD)", "ticker": None, "asset_class": "etf",
                    "exchange": US, "source": "proxy", "instrument": None}
    nvda = ct.instrument_for("NVDA", "NVDA", "Nvidia", "stock")
    assert (nvda["symbol"], nvda["ticker"], nvda["asset_class"], nvda["source"]) == ("nvda.us", "NVDA", "stock",
                                                                                     "registry")
    assert nvda["exchange"] is US and "NVDA" in nvda["label"]
    btc = ct.instrument_for("BTC", "BTC", "Bitcoin", "crypto")
    assert (btc["symbol"], btc["asset_class"], btc["exchange"]) == ("btcusd", "crypto", sp.CRYPTO)
    assert ct.instrument_for("BTC", "BTC", "Bitcoin", "stock") is None     # never btc.us from a claim's type
    assert ct.instrument_for("DAX", None, "DAX", "index") is None          # a market key without a proxy


def test_targets_on_the_call_side_within_range_touch_after_entry_day_never_for_proxies():
    nvda = _line(100, 1)                       # Sep 8 (the entry) closes at 110, Oct 9 (the mark) at 133
    entry = date(2026, 9, 8)
    assert (nvda[entry], nvda[END]) == (110, 133)
    highs = {"nvda.us": {d: v + 1 for d, v in nvda.items()}}
    lows = {"nvda.us": {d: v - 1 for d, v in nvda.items()}}
    highs["nvda.us"][entry] = 500.0            # the entry day's high may have printed before the video
    prices = _market(**{"nvda.us": nvda, "meta.us": _line(600, 2)})

    def target(cid, video, level, channel, stance="bullish", **kw):
        return _claim(cid, video, kw.pop("day", "2026-09-08"), stance, channel=channel, ctype="price_target",
                      target=level, **kw)

    claims = [
        # A: 105 is below the entry and 1,000 is 9x it; 140 is the first level that qualifies, 160 a bull case.
        target("a1", "va", 105, "A", position=1), target("a2", "va", 1000, "A", position=2),
        target("a3", "va", 140, "A", position=3), target("a4", "va", 160, "A", position=4),
        # B: the latest video's target (125) wins, touched by the daily high on Sep 28.
        target("b1", "vb1", 120, "B"), target("b2", "vb2", 125, "B", day="2026-09-21"),
        # C: bearish; 150 is on the wrong side, 90 is tracked and moving away.
        target("c1", "vc", 150, "C", stance="bearish", position=1),
        target("c2", "vc", 90, "C", stance="bearish", position=2),
        # D: closes only (no bars for META): reached by a close.
        target("d1", "vd", 650, "D", ticker="META", name="Meta"),
        # E: the S&P 500 is priced through SPY, and a proxy never gets a target.
        target("e1", "ve", 900, "E", ticker=None, name="S&P 500", asset_type="index"),
    ]
    record, _ = _record(claims, prices=prices, highs=highs, lows=lows)
    a = _row(record, creator="A")
    assert a["target"] == 140 and a["target_progress"] == pytest.approx((133 - 110) / (140 - 110))
    assert a["target_reached"] is False and a["target_method"] == "intraday_touch"   # entry-day high not counted
    b = _row(record, creator="B")
    assert b["target"] == 125 and b["videos"] == 2
    assert b["target_reached"] is True and b["target_reached_date"] == date(2026, 9, 28)
    c = _row(record, creator="C")
    assert c["target"] == 90 and c["target_progress"] == pytest.approx((133 - 110) / (90 - 110))
    assert c["target_reached"] is False
    d = _row(record, creator="D")
    meta = prices["meta.us"]
    assert d["target_method"] == "daily_close" and d["target_reached"] is True
    assert d["target_reached_date"] == min(day for day, v in meta.items() if v >= 650 and day > entry)
    e = _row(record, creator="E")
    assert e["scored"] and e["resolution"] == "proxy" and e["target"] is None and e["target_progress"] is None


# --- Aggregates ----------------------------------------------------------------------------------------


def test_creator_summary_orders_by_calls_tags_small_samples_and_guest_channels(monkeypatch):
    monkeypatch.setattr(ct, "MIN_CREATOR_SAMPLE", 3)
    monkeypatch.setattr(sp, "SCORECARD_RANKINGS_ENABLED", True)    # the tracker never ranks, whatever it says
    prices = _market(**{"nvda.us": _line(100, -1), "meta.us": _line(600, -2), "amzn.us": _line(200, -1),
                        "msft.us": _line(400, 3)})
    claims = [   # Many: three losing calls; Few: one winning call; Guests: two calls, mostly guests' views
        _claim("m1", "a1", "2026-09-08", "bullish", channel="Many"),
        _claim("m2", "a2", "2026-09-08", "bullish", channel="Many", ticker="META", name="Meta"),
        _claim("m3", "a3", "2026-09-08", "bullish", channel="Many", ticker="AMZN", name="Amazon"),
        _claim("f1", "b1", "2026-09-08", "bullish", channel="Few", ticker="MSFT", name="Microsoft"),
        _claim("g1", "c1", "2026-09-08", "bullish", channel="Guests", ticker="MSFT", name="Microsoft",
               attribution="guest_personal_view"),
        _claim("g2", "c2", "2026-09-08", "bearish", channel="Guests", attribution="guest_personal_view"),
    ]
    record, _ = _record(claims, prices=prices)
    assert list(record["creators"]) == ["Many", "Guests", "Few"]       # by number of calls, never by results
    many, guests, few = (record["creators"][c] for c in ("Many", "Guests", "Few"))
    assert set(many) == CREATOR_KEYS
    assert (many["calls"], many["bull"], many["right"], many["bench_n"]) == (3, 3, 0, 3)
    assert not many["small_sample"] and not many["mostly_guests"] and many["unscored"] == 0
    assert many["best"]["dir_ret"] >= many["worst"]["dir_ret"] and many["median_dir_excess"] is not None
    assert few["right"] == 1 and few["small_sample"]
    assert guests["mostly_guests"] and guests["small_sample"] and (guests["bull"], guests["bear"]) == (1, 1)
    monkeypatch.setattr(ct, "MIN_CREATOR_CALLS_SHOWN", 2)
    text = "\n".join(ct.format_report(record))
    assert "• Guests (mostly guests) (small sample) — 2 calls (1▲ 1▼) · right 2/2 · beat " in text
    assert "• Many — 3 calls (3▲ 0▼) · right 0/3 · beat 0/3 · median vs market " in text
    assert "   best " in text and " · worst " in text                 # samples of MIN_CREATOR_SAMPLE or more
    assert "Too few calls to judge yet: Few 1" in text
    assert text.index("• Many") < text.index("• Guests")


def test_assets_agreed_vs_split_and_asset_outcome_filters_creators_direction_pending():
    prices = _market(**{"nvda.us": _line(100, 2), "meta.us": _line(600, 1), "amzn.us": _line(200, 1)})
    claims = [
        _claim("a1", "va1", "2026-09-08", "bullish", channel="A"),
        _claim("b1", "vb1", "2026-09-10", "bullish", channel="B"),
        _claim("c1", "vc1", "2026-10-09", "bullish", channel="C", at=AFTER_CLOSE),     # no entry close yet
        _claim("a2", "va2", "2026-09-08", "bullish", channel="A", ticker="META", name="Meta"),
        _claim("b2", "vb2", "2026-09-14", "bearish", channel="B", ticker="META", name="Meta"),
        _claim("a3", "va3", "2026-09-08", "bullish", channel="A", ticker="AMZN", name="Amazon"),
        _claim("a4", "va4", "2026-10-06", "bearish", channel="A", ticker="AMZN", name="Amazon"),   # reversed, young
    ]
    record, _ = _record(claims, prices=prices)
    nvda, meta, amzn = (record["assets"][a] for a in ("NVDA", "META", "AMZN"))
    assert set(nvda) == ASSET_KEYS and set(nvda["current"]) == {"A", "B", "C"}
    assert nvda["consensus"] == "bullish" and (nvda["bulls"], nvda["bears"]) == (["A", "B"], [])
    first = nvda["current"]["A"]
    assert nvda["since"] == first["entry_date"] == date(2026, 9, 8) and nvda["move"] == first["ret"]
    assert nvda["bench_move"] == first["bench_ret"] and nvda["benchmark_label"] == "S&P 500"
    assert nvda["right"] is True and nvda["beat"] == first["beat"]
    assert meta["consensus"] == "split" and (meta["bulls"], meta["bears"]) == (["A"], ["B"])
    assert meta["since"] is None and meta["move"] is None
    # A creator's current call is their latest: the young reversal replaces the scored bullish call.
    assert amzn["current"]["A"]["direction"] == "bearish" and amzn["consensus"] is None
    out = ct.asset_outcome(record, "NVDA")
    assert set(out) == OUTCOME_KEYS
    assert out["direction"] == "bullish" and out["creators"] == ["A", "B"] and out["pending"] == ["C"]
    assert (out["since"], out["move"], out["entry_price"]) == (date(2026, 9, 8), first["ret"], first["entry_price"])
    only_b = ct.asset_outcome(record, "NVDA", creators=["B"])
    assert only_b["since"] == nvda["current"]["B"]["entry_date"] and only_b["pending"] == []
    assert ct.asset_outcome(record, "META") is None                      # the current calls disagree
    bears = ct.asset_outcome(record, "META", direction="bearish")
    assert bears["creators"] == ["B"] and bears["right"] == (bears["move"] < 0)
    young = ct.asset_outcome(record, "AMZN")
    assert young["direction"] == "bearish" and young["creators"] == ["A"] and young["move"] is not None
    assert ct.asset_outcome(record, "AAPL") is None
    assert ct.asset_outcome(record, "NVDA", creators=["Nobody"]) is None


def test_market_context_week_four_weeks_since_and_missing_symbols_omitted():
    prices = _market(btcusd=_line(80000, 100, end=date(2026, 10, 11), crypto=True),
                     ethusd=_line(2000, -5, start=date(2026, 10, 5), crypto=True))
    fetcher = ct.static_fetcher(prices)
    rows = ct.market_context(date(2026, 10, 12), fetcher, since=date(2026, 9, 5))
    assert [r["symbol"] for r in rows] == ["spy.us", "btcusd", "ethusd"]       # qqq.us has no data
    assert all(set(r) == MARKET_KEYS for r in rows)
    spy, btc, eth = rows
    s, b = prices["spy.us"], prices["btcusd"]
    assert (spy["label"], spy["as_of"], spy["close"]) == ("S&P 500", END, s[END])
    assert spy["week_from"] == date(2026, 10, 2) and spy["week"] == pytest.approx(s[END] / s[date(2026, 10, 2)] - 1)
    assert spy["four_weeks"] == pytest.approx(s[END] / s[date(2026, 9, 11)] - 1)
    assert spy["since_date"] == date(2026, 9, 8) and spy["since"] == pytest.approx(s[END] / s[date(2026, 9, 8)] - 1)
    assert btc["as_of"] == END and btc["close"] == b[END]      # capped at the S&P 500's date, not Sunday
    assert btc["since_date"] == date(2026, 9, 5) and btc["week"] == pytest.approx(b[END] / b[date(2026, 10, 2)] - 1)
    assert eth["week"] is None and eth["four_weeks"] is None and eth["since_date"] == date(2026, 10, 5)
    assert fetcher.requests == [(symbol, date(2026, 9, 4), date(2026, 10, 12)) for symbol, _ in ct.CONTEXT_SYMBOLS]
    lines = ct.format_market_lines(rows, since=date(2026, 9, 5))
    assert re.fullmatch(r"• S&P 500 \(SPY\): week \+\d\.\d% · 4 weeks \+\d\.\d% · since Sep 5 \+\d\.\d%", lines[0])
    assert lines[1].startswith("• Bitcoin: week +") and " · since Sep 5 " in lines[1]
    assert re.fullmatch(r"• Ether: since Oct 5 -\d\.\d%", lines[2])   # missing parts omitted, own start named
    assert ct.market_context(date(2026, 10, 12), ct.static_fetcher({})) == []
    assert ct.format_market_lines([]) == [] and ct.format_market_lines(None) == []


def test_price_move_first_close_on_or_after_start_to_last_close_on_or_before_end():
    prices = {"nvda.us": _line(100, 2)}
    n = prices["nvda.us"]
    fetcher = ct.static_fetcher(prices)
    move = ct.price_move("nvda.us", date(2026, 9, 5), date(2026, 9, 13), fetcher)   # Saturday .. Sunday
    assert move == {"symbol": "nvda.us", "start_date": date(2026, 9, 8), "start_price": n[date(2026, 9, 8)],
                    "end_date": date(2026, 9, 11), "end_price": n[date(2026, 9, 11)],
                    "ret": pytest.approx(n[date(2026, 9, 11)] / n[date(2026, 9, 8)] - 1)}
    assert fetcher.requests == [("nvda.us", date(2026, 9, 5), date(2026, 9, 13))]
    assert ct.price_move("nvda.us", date(2026, 9, 12), date(2026, 9, 13), fetcher) is None   # a weekend
    assert ct.price_move("nvda.us", date(2026, 9, 11), date(2026, 9, 13), fetcher) is None   # one close
    assert ct.price_move("nvda.us", date(2026, 9, 13), date(2026, 9, 5), fetcher) is None    # reversed range
    assert ct.price_move("zzz.us", date(2026, 9, 5), date(2026, 9, 13), fetcher) is None


# --- Prices: the plan, the cache, refused series ----------------------------------------------------------


def test_needed_price_ranges_cover_every_request_track_record_makes():
    claims = [
        _claim("a", "v1", "2026-09-01", "bullish"),
        _claim("b", "v2", "2026-09-20", "bearish", ticker="ETH", name="Ethereum", asset_type="crypto"),
        _claim("c", "v3", "2026-09-14", "bullish", ticker=None, name="Gold", asset_type="commodity"),
        _claim("d", "v4", "2026-09-15", "bullish", ticker="ZORB", name="Zorb"),     # unresolved: never fetched
        _claim("e", "v5", "2026-10-14", "bullish", ticker="META", name="Meta"),     # after the as-of: not fetched
    ]
    today = date(2026, 10, 13)                        # a Tuesday: the as-of is Monday Oct 12
    as_of = ct.default_as_of(today)
    needed = ct.needed_price_ranges(claims, today)
    fetcher = ct.static_fetcher(_market(**{"nvda.us": _line(100, 2)}))
    ct.track_record(claims, as_of, fetcher)
    assert sorted(fetcher.requests) == needed          # exactly the plan: one request per symbol
    for symbol, start, end in fetcher.requests:
        assert any(s == symbol and a <= start and end <= b for s, a, b in needed)
    assert as_of == date(2026, 10, 12) and all(end == as_of for _, _, end in needed)
    starts = {symbol: start for symbol, start, _ in needed}
    assert set(starts) == {"nvda.us", "ethusd", "gld.us", "spy.us", "btcusd", "qqq.us"}
    assert starts["nvda.us"] == date(2026, 8, 31) and starts["gld.us"] == date(2026, 9, 13)
    for symbol, _ in ct.CONTEXT_SYMBOLS:              # market context: since the first call (or 35 days)
        assert starts[symbol] == date(2026, 8, 31)


def test_cache_fetcher_never_goes_live(monkeypatch):
    import channel_scorecard
    live = []

    def must_not_run(*args, **kwargs):
        live.append(args)
        raise AssertionError("the tracker must never fetch prices live")

    monkeypatch.setattr(channel_scorecard, "fetch_prices_live", must_not_run)
    spy = _line(500, 1)
    cache = _cache({"nvda.us": _line(100, 2, end=date(2026, 9, 30)), "spy.us": spy})   # nvda.us stops early
    cache["spy.us"]["highs"] = {d: v + 1 for d, v in spy.items()}
    cache["spy.us"]["lows"] = {d: v - 1 for d, v in spy.items()}
    price_cache.reset(cache)
    record = ct.track_record([_claim("a", "v1", "2026-09-08", "bullish")], END)    # default: the cache
    assert live == [] and record["as_of"] == END
    assert record["calls"][0]["reason"] == "stale_mark"        # the gap is reported, not filled live
    fetcher = ct.cache_fetcher()
    series = fetcher("spy.us", date(2026, 9, 1), date(2026, 9, 4))
    assert isinstance(series, sp.PriceSeries) and sorted(series.closes) == [date(2026, 9, d) for d in (1, 2, 3, 4)]
    assert series.adjustment == "split_adjusted" and series.granularity == "daily_ohlc"
    assert fetcher("qqq.us", date(2026, 9, 1), date(2026, 9, 4)).closes == {}
    assert fetcher.requests == [("spy.us", date(2026, 9, 1), date(2026, 9, 4)),
                                ("qqq.us", date(2026, 9, 1), date(2026, 9, 4))]
    assert fetcher.price_provenance["provider"] == "price_cache"
    explicit = ct.cache_fetcher({"spy.us": cache["spy.us"]})          # an explicit dict instead of the cache
    assert explicit("spy.us", START, START).closes == {START: spy[START]}
    assert live == []


def test_plain_dict_fetcher_without_provenance_is_refused_as_no_prices(monkeypatch):
    warnings = []
    monkeypatch.setattr(ct, "log_warn", warnings.append)
    prices = _market(**{"nvda.us": _line(100, 2)})
    claims = [_claim("a", "v1", "2026-09-08", "bullish")]

    def plain(symbol, start, end):
        return {d: v for d, v in (prices.get(symbol) or {}).items() if start <= d <= end}

    record = ct.track_record(claims, END, plain)
    assert record["calls"][0]["reason"] == "no_prices" and record["market_data"] is False
    assert record["market"] == [] and record["stats"]["scored"] == 0
    assert len(warnings) == 1 and "unknown adjustment" in warnings[0]      # four series, one line
    plain.price_provenance = {"provider": "test", "adjustment": "split_adjusted"}
    record = ct.track_record(claims, END, plain)
    assert record["calls"][0]["scored"] and record["market_data"] is True and len(warnings) == 1

    def broken(symbol, start, end):
        raise RuntimeError("provider down")

    record = ct.track_record(claims, END, broken)              # never raises: no prices
    assert record["calls"][0]["reason"] == "no_prices" and record["stats"]["scored"] == 0
    assert len(warnings) == 2 and "provider down" in warnings[1]


# --- Record shape and formatting --------------------------------------------------------------------------


def test_record_shape_and_reason_vocabulary_match_the_contract():
    prices = _market(**{"nvda.us": _line(100, 2)})
    claims = [_claim("a", "v1", "2026-09-08", "bullish"),
              _claim("b", "v2", "2026-09-08", "bullish", ticker=None, name="Zorblax Industries"),
              _claim("c", "v3", "2026-10-08", "bullish", channel="Other")]
    record, _ = _record(claims, prices=prices)
    assert set(record) == RECORD_KEYS
    assert set(record["stats"]) == {"video_calls", "calls", "reiterations", "reversals", "matured", "scored"}
    assert set(record["claim_exclusions"]) == {"not_directional", "not_a_security_view", "trigger_condition",
                                               "no_asset", "not_an_instrument", "video_mixed",
                                               "video_not_review_robust"}
    assert set(record["pooled"]) == POOLED_KEYS and record["since"] == date(2026, 9, 8)
    assert all(set(r) == CALL_KEYS for r in record["calls"])
    assert all(r["scored"] == (r["reason"] is None) for r in record["calls"])
    assert set(record["excluded"]) <= REASONS
    assert [(r["call_date"], r["creator"], r["asset"]) for r in record["calls"]] == sorted(
        (r["call_date"], r["creator"], r["asset"]) for r in record["calls"])
    assert record["pooled"]["market_return"] == pytest.approx(prices["spy.us"][END] / prices["spy.us"][date(2026, 9, 8)] - 1)
    assert set(ct.TRACKER_RULES) == {"unit", "video_direction", "dedup", "entry", "mark", "as_of", "benchmark",
                                     "metrics", "min_age", "targets", "exclusions", "prices"}


def _many_creators_record(as_of=END, spy_end=END):
    tickers = [("NVDA", "Nvidia", 2), ("META", "Meta", -1), ("AMZN", "Amazon", 0.5),
               ("MSFT", "Microsoft", 1), ("AAPL", "Apple", -0.5), ("TSLA", "Tesla", 3)]
    prices = _market(**{f"{t.lower()}.us": _line(200, step) for t, _, step in tickers})
    prices["spy.us"] = _line(500, 1, end=spy_end)
    claims = []
    for i in range(12):
        for j, (ticker, name, _) in enumerate(tickers):
            stance = "bullish" if j < 2 or (i + j) % 3 else "bearish"     # agreement on two, splits elsewhere
            claims.append(_claim(f"c{i}-{j}", f"v{i}-{j}", date(2026, 9, 8) + timedelta(days=j), stance,
                                 channel=f"Creator {i:02d}", ticker=ticker, name=name))
    return _record(claims, as_of=as_of, prices=prices)[0]


def test_format_report_messages_fit_telegram_carry_denominators_and_the_label(monkeypatch):
    record = _many_creators_record()
    messages = ct.format_report(record)
    assert len(messages) >= 2 and all(len(m) <= ct.MAX_MESSAGE_CHARS for m in messages)
    head = messages[0].splitlines()[0]
    assert head == ("📈 Creators vs the market — calls since Sep 8, marked to Fri Oct 9 close "
                    "(mark-to-market, not a verdict)")
    text = "\n".join(messages)
    assert "not a verdict" in head and "Market:" in text and "• S&P 500 (SPY): week +" in text
    assert ("▲ bullish ▼ bearish · right = moved the called way · beat = beat the S&P 500 in the called "
            "direction over the same days: a ▲ call rose more (or fell less), a ▼ call fell more (or rose less)"
            in text)
    assert re.search(r"• Creator 00 \(small sample\) — 6 calls \(\d▲ \d▼\) · right \d/6 · beat \d/6 · "
                     r"median vs market [+-]\d+\.\d%", text)
    assert "Same-length check — the " in text and "read beat" in text   # the S&P 500 rose
    assert "🤝 Where creators agreed" in text and "⚔️ Where they disagreed:" in text
    assert "• NVDA ▲ 12 creators — " in text
    coverage = next(line for line in text.splitlines() if line.startswith("Coverage: "))
    assert coverage.startswith(f"Coverage: {record['stats']['calls']} calls on tradable assets (one per creator, "
                               f"asset and change of view) → {record['stats']['scored']} marked to market.")
    assert messages[-1].endswith("Research input, not investment advice.")
    for word in ("best creator", "#1", "rank", "winner", "top creator"):
        assert word not in text.lower(), word
    # Over the limit, a report splits at line boundaries only.
    lines = [line for m in messages for line in m.splitlines()]
    monkeypatch.setattr(ct, "MAX_MESSAGE_CHARS", 400)
    small = ct.format_report(record)
    assert len(small) > len(messages) and all(len(m) <= 400 for m in small)
    assert [line for m in small for line in m.splitlines() if line] == [line for line in lines if line]
    # Prices more than a week behind the requested as-of are flagged.
    stale = _many_creators_record(as_of=date(2026, 10, 16), spy_end=date(2026, 10, 7))
    assert stale["as_of"] == date(2026, 10, 7)
    assert "⚠️ The prices are stale" in "\n".join(ct.format_report(stale))
    assert "stale" not in "\n".join(ct.format_report(record))


def test_format_summary_lines_and_empty_record_return_lists():
    record = _many_creators_record()
    lines = ct.format_summary_lines(record, max_creators=2)
    assert isinstance(lines, list) and ct.REPORT_LABEL in lines[0] and "Fri Oct 9" in lines[0]
    assert re.fullmatch(r"72 calls · right \d+/72 · beat \d+/\d+ · S&P 500 \+\d\.\d% over the period", lines[1])
    assert lines[2].startswith("▲ bullish ▼ bearish · right = moved the called way · beat = beat the S&P 500")
    assert [line[:13] for line in lines[3:5]] == ["• Creator 00 ", "• Creator 01 "]
    # Every creator left out is named with their calls, and the base rate is said.
    assert lines[5].startswith("+10 more creators (Creator 02 6, Creator 03 6, ") and "Friday's report" in lines[5]
    assert lines[6].startswith("In a rising market (S&P 500 +") and lines[6].endswith("read beat.")
    empty = ct.track_record([], END, ct.static_fetcher(_market()))
    assert set(empty) == RECORD_KEYS and empty["calls"] == [] and empty["since"] is None
    assert empty["stats"]["scored"] == 0 and empty["creators"] == {} and empty["assets"] == {}
    assert ct.format_report(empty) == [] and ct.format_summary_lines(empty) == []
    assert ct.format_report(None) == [] and ct.format_summary_lines({}) == []


# --- Module hygiene and the CLI -------------------------------------------------------------------------------


def test_import_pulls_in_no_network_code():
    code = ("import sys, creator_tracker; "
            "bad = [m for m in ('requests', 'channel_scorecard', 'sendToTelegram', 'market_pulse', "
            "'research_analytics', 'pulse_charts', 'warm_prices') if m in sys.modules]; "
            "assert not bad, bad")
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=repo)
    assert done.returncode == 0, done.stderr


def test_cli_prints_the_report_and_writes_the_record_as_json(monkeypatch, tmp_path, capsys):
    claims = [_claim("a", "v1", "2026-09-08", "bullish"), _claim("b", "v2", "2026-09-10", "bearish", channel="B",
                                                                 ticker="META", name="Meta")]
    monkeypatch.setattr(canonical_claims, "load_canonical_claims", lambda *a, **k: claims)
    price_cache.reset(_cache(_market(**{"nvda.us": _line(100, 2), "meta.us": _line(600, 1)})))
    out = tmp_path / "record.json"
    assert ct.main(["--as-of", "2026-10-12", "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "marked to Fri Oct 9 close (mark-to-market, not a verdict)" in printed
    data = json.loads(out.read_text())
    assert set(data) == RECORD_KEYS
    assert (data["requested_as_of"], data["as_of"], data["since"]) == ("2026-10-12", "2026-10-09", "2026-09-08")
    assert data["calls"][0]["entry_date"] == "2026-09-08" and data["calls"][0]["scored"] is True


# --- Review fixes: hindsight, staleness, houses, themes, one fund, entry gaps -----------------------------


def test_a_target_restated_later_is_tested_from_its_statement_and_never_with_hindsight():
    nvda = _line(100, 1)                       # Sep 8 (the entry) 110 ... Sep 21 119 ... Oct 9 133
    highs = {"nvda.us": {d: v + 1 for d, v in nvda.items()}}
    lows = {"nvda.us": {d: v - 1 for d, v in nvda.items()}}
    prices = _market(**{"nvda.us": nvda})

    def target(cid, video, level, day, channel):
        return _claim(cid, video, day, "bullish", channel=channel, ctype="price_target", target=level)

    claims = [
        # A: a $118 level restated on Sep 21, after the price had passed it (Sep 21 closes at 119):
        # not a target; the first video's $140 is the call's target.
        target("a1", "va1", 140, "2026-09-08", "A"), target("a2", "va2", 118, "2026-09-21", "A"),
        # B: $125 restated on Sep 14 (close 114) is reached later, dated from the restatement.
        _claim("b1", "vb1", "2026-09-08", "bullish", channel="B"), target("b2", "vb2", 125, "2026-09-14", "B"),
        # C: a level stated after the mark (Oct 12, the as-of is Oct 9) is not known yet.
        _claim("c1", "vc1", "2026-09-08", "bullish", channel="C"),
        target("c2", "vc2", 150, "2026-10-12", "C"),
    ]
    record, _ = _record(claims, prices=prices, highs=highs, lows=lows)
    a = _row(record, creator="A")
    assert a["target"] == 140 and a["target_date"] == date(2026, 9, 8) and a["target_reached"] is False
    b = _row(record, creator="B")
    assert b["target"] == 125 and b["target_date"] == date(2026, 9, 14)
    assert b["target_progress"] == pytest.approx((133 - 110) / (125 - 110))
    assert b["target_reached"] is True and b["target_reached_date"] > b["target_date"]
    c = _row(record, creator="C")
    assert c["videos"] == 2 and c["target"] is None and c["target_date"] is None
    text = "\n".join(ct.format_report(record))
    assert "125 by B (call Sep 8, target stated Sep 14)" in text


def test_no_current_sp500_close_marks_nothing_and_says_why():
    claims = [_claim("a", "v1", "2026-09-08", "bullish"),
              _claim("b", "v2", "2026-09-21", "bearish", at=AFTER_CLOSE)]     # a reversed call, 2 weeks long
    stale = _market(**{"spy.us": _line(500, 1, end=date(2026, 9, 25)), "nvda.us": _line(100, 2)})
    record, _ = _record(claims, as_of=date(2026, 10, 16), prices=stale)
    assert record["market_data"] is False and record["stats"]["scored"] == 0 and record["pooled"]["calls"] == 0
    assert record["excluded"] == {"stale_mark": 2} and record["creators"] == {}
    # The context rows stop at the newest S&P 500 close, crypto included (no Sunday candle).
    assert {r["as_of"] for r in record["market"]} == {date(2026, 9, 25)}
    assert ct.asset_outcome(record, "NVDA")["move"] is None          # no move from a stale book
    notice = ct.format_report(record)
    assert notice == ct.format_summary_lines(record) and len(notice) == 1
    assert notice[0].startswith("📈 Creators vs the market — not marked this week: the prices are stale (the "
                                "newest S&P 500 close in the data is Fri Sep 25")
    assert "2 calls wait for fresh closes." in notice[0]


def test_two_channels_of_one_house_are_one_opinion(monkeypatch):
    houses = ct.creator_houses(["HKCM", "Phantom by HKCM", "HKCM GLOBAL", "FinVid", "Couch Investor"])
    assert houses == {"HKCM": "HKCM", "Phantom by HKCM": "HKCM", "HKCM GLOBAL": "HKCM", "FinVid": "FinVid",
                      "Couch Investor": "Couch Investor"}
    assert ct.creator_houses(["Phantom by HKCM", "HKCM GLOBAL"]) == {"Phantom by HKCM": "Phantom by HKCM",
                                                                     "HKCM GLOBAL": "HKCM GLOBAL"}
    monkeypatch.setattr(ct, "CREATOR_HOUSES", {"Couch Investor": "Couches", "FinVid": "Couches"})
    assert ct.creator_houses(["FinVid", "Couch Investor"]) == {"FinVid": "Couches", "Couch Investor": "Couches"}
    monkeypatch.setattr(ct, "CREATOR_HOUSES", {})
    prices = _market(**{"nvda.us": _line(100, 2), "meta.us": _line(600, 1)})
    claims = [_claim("h1", "vh1", "2026-09-08", "bullish", channel="HKCM"),
              _claim("p1", "vp1", "2026-09-08", "bullish", channel="Phantom by HKCM"),
              _claim("h2", "vh2", "2026-09-08", "bullish", channel="HKCM", ticker="META", name="Meta"),
              _claim("f2", "vf2", "2026-09-08", "bullish", channel="FinVid", ticker="META", name="Meta")]
    record, _ = _record(claims, prices=prices)
    assert record["assets"]["NVDA"]["houses"] == 1 and record["assets"]["META"]["houses"] == 2
    text = "\n".join(ct.format_report(record))
    assert "• META ▲ 2 creators" in text and "• NVDA ▲" not in text      # one house's two channels: no consensus


def test_themes_and_markets_without_a_fund_are_not_calls_but_ticker_gaps_are():
    claims = [
        _claim("s", "v1", "2026-09-08", "bullish", ticker=None, name="Aerospace", asset_type="sector"),
        _claim("o", "v2", "2026-09-08", "bullish", ticker=None, name="Value investing", asset_type="other"),
        _claim("d", "v3", "2026-09-08", "bearish", ticker=None, name="DAX", asset_type="index"),
        _claim("p", "v4", "2026-09-08", "bullish", ticker=None, name="Jersey Mike's", asset_type="private_company"),
        _claim("g", "v5", "2026-09-08", "bullish", ticker=None, name="Zorblax Industries"),   # a stock: a gap
        _claim("t", "v6", "2026-09-08", "bullish", ticker=None, name="Technology", asset_type="sector"),  # XLK
    ]
    record, _ = _record(claims, prices=_market(**{"xlk.us": _line(200, 1)}))
    assert record["claim_exclusions"]["not_an_instrument"] == 4
    assert {r["asset"] for r in record["calls"]} == {"ZORBLAX INDUSTRIES", "TECHNOLOGY"}
    assert _row(record, asset="ZORBLAX INDUSTRIES")["reason"] == "no_ticker"
    assert _row(record, asset="TECHNOLOGY")["scored"]
    coverage = next(line for line in ct.format_report(record)[-1].splitlines() if line.startswith("Coverage: "))
    assert coverage.endswith("Views on themes, macro and markets without a fund (4 opinions) are not calls.")
    assert ct.unpriced_assets(record) == [("ZORBLAX INDUSTRIES", 1)]
    # The research report's video calls keep every market and theme.
    assert ct.video_calls(claims)[1]["not_an_instrument"] == 0 and len(ct.video_calls(claims)[0]) == 6
    # A sector with a fund is priced through it; a plural group ("AI stocks") is no instrument.
    assert ct.instrument_for("HEALTHCARE")["symbol"] == "xlv.us" and ct.instrument_for("BANKS")["symbol"] == "kbe.us"
    group = [_claim("ai", "v7", "2026-09-08", "bullish", ticker=None, name="AI stocks")]
    assert ct.track_record(group, END, ct.static_fetcher(_market()))["claim_exclusions"]["not_an_instrument"] == 1


def test_an_untickered_name_is_never_read_as_a_ticker():
    for asset, name in (("GM", "GM"), ("LOW", "Low"), ("TEAM", "Team")):
        assert ct.instrument_for(asset, None, name, "stock") is None, asset
    assert ct.instrument_for("LOW", "LOW", "Lowe's", "stock")["symbol"] == "low.us"
    claims = [_claim("g", "v1", "2026-09-08", "bearish", ticker=None, name="GM")]
    record, fetcher = _record(claims, prices=_market(**{"gm.us": _line(50, 1)}))
    assert record["calls"][0]["reason"] == "no_ticker"
    assert "gm.us" not in {s for s, _, _ in fetcher.requests}


def test_spellings_of_one_fund_are_one_asset_and_a_minority_horizon_never_matures_the_call():
    claims = [_claim("a", "v1", "2026-09-08", "bullish", ticker=None, name="the S&P 500", asset_type="index"),
              _claim("b", "v2", "2026-09-21", "bullish", ticker=None, name="the US stock market",
                     asset_type="index")]
    record, _ = _record(claims)
    assert record["stats"]["calls"] == 1 and record["stats"]["reiterations"] == 1
    assert record["calls"][0]["asset"] == "S&P 500" and record["calls"][0]["videos"] == 2
    # Two long-dated bullish claims and one bearish week-long remark: the call's horizon is its own.
    minority = [_claim("x", "v3", "2026-09-08", "bullish", end="2027-11-03", channel="M"),
                _claim("y", "v3", "2026-09-08", "bullish", end="2027-11-03", channel="M", position=2),
                _claim("z", "v3", "2026-09-08", "bearish", end="2026-09-15", channel="M", position=3)]
    calls, _ = ct.video_calls(minority)
    assert calls[0]["direction"] == "bullish" and calls[0]["horizon_end"] == date(2027, 11, 3)
    record, _ = _record(minority)
    assert record["calls"][0]["status"] == "open" and record["calls"][0]["scored"]


def test_an_entry_is_never_shifted_to_a_later_close_when_the_series_starts_late():
    # Published Thu Sep 17 during the session: the rule's entry is Sep 17's close, which the series lacks.
    late = _line(20, 0.1, start=date(2026, 9, 18))
    claims = [_claim("c", "v1", "2026-09-17", "bearish", at="T18:35:00+00:00", ticker="CCL", name="Carnival")]
    record, _ = _record(claims, prices=_market(**{"ccl.us": late}))
    row = record["calls"][0]
    assert row["reason"] == "no_entry_close" and row["entry_price"] is None and not row["scored"]
    full = _line(20, 0.1, start=date(2026, 9, 16))
    record, _ = _record(claims, prices=_market(**{"ccl.us": full}))
    assert record["calls"][0]["entry_date"] == date(2026, 9, 17) and record["calls"][0]["scored"]


def test_a_bearish_call_beats_the_market_in_its_direction_and_the_words_say_so():
    # NVDA rises 2.0% while the S&P 500 rises more: the bearish call was wrong but beat the market.
    prices = _market(**{"nvda.us": _line(1000, 0.5)})
    record, _ = _record([_claim("n", "v1", "2026-09-08", "bearish")], prices=prices)
    row = record["calls"][0]
    assert row["ret"] > 0 and row["right"] is False and row["bench_ret"] > row["ret"] and row["beat"] is True
    assert "a ▼ call fell more (or rose less)" in "\n".join(ct.format_report(record))


def test_closed_calls_stop_costing_warm_requests():
    claims = [_claim("a", "v1", "2026-09-08", "bullish", ticker="META", name="Meta"),
              _claim("b", "v2", "2026-09-14", "bearish", ticker="META", name="Meta"),        # reversed: META open
              _claim("c", "v3", "2026-09-01", "bullish", end="2026-09-11")]                 # matured Sep 11
    today = date(2026, 10, 13)
    needed = {s: (start, end) for s, start, end in ct.needed_price_ranges(claims, today)}
    assert needed["nvda.us"] == (date(2026, 8, 31), date(2026, 9, 21))      # horizon end + 10 days
    assert needed["meta.us"][1] == ct.default_as_of(today)                    # the reversal's call is open
    assert needed["spy.us"][1] == ct.default_as_of(today)                     # benchmarks and context run on
    fetcher = ct.static_fetcher(_market(**{"nvda.us": _line(100, 2), "meta.us": _line(600, 1)}))
    record = ct.track_record(claims, ct.default_as_of(today), fetcher)
    assert sorted(fetcher.requests) == ct.needed_price_ranges(claims, today)
    assert _row(record, asset="NVDA")["status"] == "matured" and _row(record, asset="NVDA")["scored"]


def test_nyse_half_days_close_at_one_so_a_later_video_gets_the_next_close():
    assert sp.nyse_early_closes(2026) == {date(2026, 11, 27), date(2026, 12, 24)}   # July 4, 2026 is a Saturday
    assert date(2027, 7, 2) not in sp.nyse_early_closes(2027) and date(2025, 7, 3) in sp.nyse_early_closes(2025)
    closes = {d: 100.0 + i for i, d in enumerate(_days(date(2026, 11, 20), date(2026, 12, 31)))}
    after_half_day = sp.entry_point(US, "2026-11-27T19:30:00+00:00", closes)     # 14:30 New York
    assert after_half_day["resolved_trading_date"] == "2026-11-30"
    assert sp.entry_point(US, "2026-11-27T17:30:00+00:00", closes)["resolved_trading_date"] == "2026-11-27"
    assert sp.entry_point(US, "2026-12-24T20:00:00+00:00", closes)["resolved_trading_date"] == "2026-12-28"
    assert sp.entry_point(US, "2026-12-23T20:00:00+00:00", closes)["resolved_trading_date"] == "2026-12-23"
