"""Analytics over canonical claims: canonical identity, consensus, view
changes, the conditional block, the quality header, the strict scorecard and
the creators-vs-market block. Offline throughout: prices come from in-memory
fetchers or a price cache the test installs."""
from collections import Counter
from datetime import date, datetime, timedelta, timezone

import pytest

import canonical_claims
import creator_tracker
import price_cache
import research_analytics as ra
import scorecard_pricing as sp

US = sp.EXCHANGES["us"]


def _c(source, asset, stance, bucket="short", published="2026-07-10", **over):
    base = {
        "claim_id": f"{source}-{asset}-{stance}-{bucket}-{published}-{over.get('n', 0)}",
        "video_id": f"vid-{source}-{published}", "channel_name": source, "published_at": f"{published}T12:00:00+00:00",
        "extracted_at": f"{published}T13:00:00+00:00", "ticker": asset, "subject_mention": asset,
        "stance": stance, "horizon_bucket": bucket, "review_required": False, "coverage_status": "full",
        "evidence_start_character": 10, "attribution_type": "speaker_personal_view",
        "repeat_of_claim_id": None, "is_forward_looking": True, "testable": False, "claim_type": "stance",
        "certainty_level": "medium", "recommendation_action": "none", "catalysts": [], "risks": [],
        "assumptions": [], "segment_id": f"seg-{over.get('n', 0)}", "asset_type": "stock",
        "forecast_direction": "increase" if stance == "bullish" else "decrease",
        "evidence_text": "e", "schema_version": "1",
    }
    base.update(over)
    return base


@pytest.fixture(autouse=True)
def _seven_day_flips(monkeypatch):
    """The documented default, whatever RESEARCH_FLIP_MIN_GAP_DAYS says."""
    monkeypatch.setattr(ra, "MIN_FLIP_GAP_DAYS", 7)


def _sessions(start, end):
    out, day = [], start
    while day <= end:
        if sp.is_trading_day(US, day):
            out.append(day)
        day += timedelta(days=1)
    return out


def _rising(start, end, first=100.0, step=1.0):
    """A close on every NYSE session, moving `step` a session."""
    return {d: first + step * i for i, d in enumerate(_sessions(start, end))}


def test_consensus_counts_one_current_view_per_source_asset_bucket():
    claims = [
        _c("A", "NVDA", "bullish", n=1), _c("A", "NVDA", "bullish", n=2), _c("A", "NVDA", "bullish", n=3),
        _c("B", "NVDA", "bearish"),
        _c("C", "NVDA", "bullish", bucket="long"),
    ]
    cons = ra.consensus(claims)
    short = cons[("NVDA", "short")]
    assert short["bullish"] == 1 and short["bearish"] == 1 and short["sources"] == 2   # A counted once
    assert short["net_stance"] == 0.0
    assert ("NVDA", "long") in cons and cons[("NVDA", "long")]["bullish"] == 1     # never merged
    assert short["date_range"] == ("2026-07-10", "2026-07-10")


def test_latest_view_per_source_wins_and_neutral_is_breadth():
    claims = [_c("A", "NVDA", "bullish", published="2026-07-01"), _c("A", "NVDA", "bearish", published="2026-07-09"),
              _c("B", "NVDA", "neutral", forecast_direction=None)]
    e = ra.consensus(claims)[("NVDA", "short")]
    assert e["bearish_sources"] == ["A"] and e["neutral_sources"] == ["B"]
    assert e["net_stance"] == -1.0


def test_net_stance_is_none_on_zero_denominator():
    assert ra.net_stance(0, 0) is None
    assert ra.net_stance(3, 1) == 0.5


def test_excluded_claims_do_not_enter_consensus():
    claims = [_c("A", "NVDA", "bullish", review_required=True), _c("B", "NVDA", "bullish", coverage_status="partial"),
              _c("C", "NVDA", "bullish", attribution_type="speaker_quoting_third_party"),
              _c("D", "NVDA", "bullish", evidence_start_character=None),
              _c("E", "NVDA", "bullish", repeat_of_claim_id="x")]
    assert ra.consensus(claims) == {}


def test_research_keys_are_canonical():
    claims = [
        _c("A", "LINK", "bullish", asset_type="crypto"),
        _c("B", None, "bullish", subject_mention="Chainlink", asset_type="crypto"),
        _c("C", None, "bearish", subject_mention="Öl", asset_type="commodity"),
        _c("D", None, "bullish", subject_mention="oil", asset_type="commodity"),
        _c("E", None, "bullish", subject_mention="Alphabet stock"),
        _c("F", "GOOGL", "bullish"),
        # Placeholders, chart vocabulary and baskets are no asset at all.
        _c("G", None, "bearish", subject_mention="the market"),
        _c("H", None, "bullish", subject_mention="price", asset_type="crypto"),
        _c("I", None, "bearish", subject_mention="wave two or B wave", asset_type="crypto"),
        _c("J", None, "bullish", subject_mention="Google and Meta", subject_kind="basket"),
    ]
    cons = ra.consensus(claims)
    assert set(cons) == {("LINK", "short"), ("OIL", "short"), ("GOOGL", "short")}
    assert cons[("LINK", "short")]["bullish_sources"] == ["A", "B"]           # CHAINLINK and LINK fold
    assert (cons[("OIL", "short")]["bullish_sources"], cons[("OIL", "short")]["bearish_sources"]) == (["D"], ["C"])
    assert cons[("GOOGL", "short")]["bullish_sources"] == ["E", "F"]
    assert set(ra.latest_views(claims)) == {"LINK", "OIL", "GOOGL"}
    assert set(ra.descriptive(claims)["claims_by_asset"]) == {"LINK", "OIL", "GOOGL"}
    # The direction rule is canonical_claims.direction_of, neutral as None.
    assert ra._direction_of(_c("A", "NVDA", "neutral", forecast_direction=None)) is None
    assert ra._direction_of(_c("A", "NVDA", "mixed", forecast_direction=None)) is None
    assert ra._direction_of(_c("A", "NVDA", "not_applicable", forecast_direction="decline")) == "bearish"
    assert ra._source({"channel_id": "UC1"}) == canonical_claims.source_of({"channel_id": "UC1"}) == "UC1"


# --- View changes -------------------------------------------------------------------------


def test_flips_need_same_source_asset_and_seven_days():
    claims = [
        _c("A", "NVDA", "bullish", published="2026-06-01", catalysts=["earnings"]),
        _c("A", "NVDA", "bearish", published="2026-07-01", catalysts=["tariffs"], risks=["china"]),
        _c("A", "NVDA", "bullish", bucket="long", published="2026-07-02"),   # a day later: short-term, not a flip
        _c("B", "NVDA", "bearish", published="2026-07-03"),                  # other source: no flip
        _c("A", "TSLA", "bearish", published="2026-07-20"),                  # other asset: no flip
    ]
    flips = ra.find_flips(claims)
    assert len(flips) == 1
    f = flips[0]
    assert (f["source"], f["asset"], f["from"], f["to"], f["elapsed_days"]) == ("A", "NVDA", "bullish", "bearish", 30)
    assert f["previous_claim_id"] and f["new_claim_id"] and f["changed_catalysts"] == ["tariffs"]
    assert f["changed_risks"] == ["china"]
    assert (f["previous_date"], f["new_date"], f["horizon_bucket"]) == ("2026-06-01", "2026-07-01", "short")
    assert (f["previous_video_id"], f["new_video_id"]) == ("vid-A-2026-06-01", "vid-A-2026-07-01")
    assert ra.short_reversals(claims) == Counter({"A": 1})
    # The horizon is what the new video's claims name, joined.
    both = [_c("C", "AMD", "bullish", published="2026-06-01"),
            _c("C", "AMD", "bearish", published="2026-06-20", n=1),
            _c("C", "AMD", "bearish", bucket="long", published="2026-06-20", n=2)]
    assert ra.find_flips(both)[0]["horizon_bucket"] == "short/long"


def test_flips_ignore_same_video_and_short_gap_reversals():
    # Scenario talk inside one video ("a decent pullback in the next few
    # weeks" / "if we break above 2,487 ... higher"), alternating: the old
    # claim-level rule listed five 0-day flips here.
    scenario = [_c("A", "ETH", "bullish" if i % 2 else "bearish", published="2026-09-17", n=i,
                   asset_type="crypto", evidence_start_character=10 + i) for i in range(6)]
    assert ra.find_flips(scenario) == [] and ra.short_reversals(scenario) == Counter()
    # Two videos two days apart: a short-term reversal, counted, not listed.
    two_days = [_c("A", "ETH", "bullish", published="2026-09-17", asset_type="crypto"),
                _c("A", "ETH", "bearish", published="2026-09-19", asset_type="crypto")]
    assert ra.find_flips(two_days) == []
    assert ra.short_reversals(two_days) == Counter({"A": 1})
    # A week apart it is a view change.
    week = [two_days[0], _c("A", "ETH", "bearish", published="2026-09-24", asset_type="crypto")]
    [flip] = ra.find_flips(week)
    assert (flip["from"], flip["to"], flip["elapsed_days"]) == ("bullish", "bearish", 7)
    assert ra.short_reversals(week) == Counter()
    # A mixed (scenario) video between two calls neither flips nor resets.
    around = [_c("A", "ETH", "bullish", published="2026-09-01", asset_type="crypto")] + scenario + \
        [_c("A", "ETH", "bearish", published="2026-09-20", asset_type="crypto")]
    [flip] = ra.find_flips(around)
    assert (flip["previous_date"], flip["new_date"], flip["elapsed_days"]) == ("2026-09-01", "2026-09-20", 19)


def test_short_reversals_are_counted_per_creator():
    claims = [
        _c("A", "NVDA", "bullish", published="2026-09-01"), _c("A", "NVDA", "bearish", published="2026-09-03"),
        _c("A", "NVDA", "bullish", published="2026-09-05"),                       # two quick reversals
        _c("B", "TSLA", "bullish", published="2026-09-01"), _c("B", "TSLA", "bearish", published="2026-09-02"),
        _c("C", "AMD", "bullish", published="2026-09-01"), _c("C", "AMD", "bearish", published="2026-09-20"),
        _c("C", "AMD", "bearish", published="2026-09-22"),                        # a reiteration
    ]
    short = ra.short_reversals(claims)
    assert short == Counter({"A": 2, "B": 1})
    flips = ra.find_flips(claims)
    assert [(f["source"], f["asset"], f["elapsed_days"]) for f in flips] == [("C", "AMD", 19)]
    lines = ra.format_flip_lines(flips, short)
    assert lines == [
        "View changes (different videos ≥7 days apart; same-video scenarios ignored):",
        "• C on AMD [short]: bullish → bearish (2026-09-01 → 2026-09-20, 19d)",
        "Short-term reversals (<7 days apart, not counted as view changes): A 2 · B 1",
    ]
    # The gap is configurable.
    assert ra.short_reversals(claims, min_gap_days=2) == Counter({"B": 1})
    assert len(ra.find_flips(claims, min_gap_days=2)) == 3
    assert ra.format_flip_lines([], Counter()) == []
    # On a guest-panel channel a reversal is often two different guests.
    guests = [_c("G", "META", "bearish", published="2026-09-17", attribution_type="guest_personal_view"),
              _c("G", "META", "bullish", published="2026-10-07", attribution_type="guest_personal_view")]
    assert ra.format_flip_lines(ra.find_flips(guests), Counter())[1].endswith("(2026-09-17 → 2026-10-07, 20d; "
                                                                             "guests' views)")


def test_flips_count_only_directional_security_views():
    # A revenue forecast, a risk and a trigger-conditional claim say nothing
    # about the creator's view of the stock: none of them reverses it.
    claims = [
        _c("A", "AMZN", "bullish", published="2026-09-01", claim_type="recommendation"),
        _c("A", "AMZN", "bearish", published="2026-09-10", claim_type="forecast", forecast_metric="revenue"),
        _c("A", "AMZN", "bearish", published="2026-09-20", claim_type="risk"),
        _c("A", "AMZN", "bearish", published="2026-09-30", condition="if it breaks below $200"),
        _c("A", "AMZN", "bearish", published="2026-10-10", condition="as long as $200 holds"),   # a qualifier
    ]
    [flip] = ra.find_flips(claims)
    assert (flip["previous_date"], flip["new_date"]) == ("2026-09-01", "2026-10-10")


# --- Conditional claims -------------------------------------------------------------------


def _cond(source, asset, condition, stance="bullish", published="2026-07-01", **over):
    base = _c(source, asset, stance, published=published, n=over.pop("n", 0), claim_type="forecast",
              condition=condition, condition_observable=False, condition_kind="subjective",
              condition_status="not_evaluated", condition_data_source=None, testability_type="not_testable")
    base.update(over)
    return base


def _observable(source, asset, condition, published="2026-07-01", **over):
    return _cond(source, asset, condition, published=published, condition_observable=True,
                 condition_kind="price_level", condition_data_source="daily_close", **over)


def test_conditional_rows_mark_triggers_and_qualifiers():
    claims = {
        "drop": _cond("A", "META", "if it drops to $600", n=1),
        "hold": _cond("A", "BTC", "as long as $76,220 support holds", n=2, asset_type="crypto"),
        "risk": _cond("B", "UPST", "for investors with a high risk tolerance", n=3),
        "unless": _cond("B", "CRWV", "unless something major happens", n=4),
        "name": _cond("C", None, "if it breaks out", n=5, subject_mention="Chainlink", asset_type="crypto"),
        "placeholder": _cond("C", None, "if the market turns", n=6, subject_mention="the market"),
        "raw": _cond("C", None, "if it drops under $5", n=7, subject_mention="Zorblax Industries"),
        "review": _cond("D", "NVDA", "if it holds 150", n=8, review_required=True),
        "quoted": _cond("E", "NVDA", "if it holds 150", n=9, attribution_type="speaker_quoting_third_party"),
        "legacy": _cond("F", "NVDA", "if it holds 150", n=10, schema_version="legacy"),
    }
    rows = {r["claim_id"]: r for r in ra.conditional_forecast_report(list(claims.values()))}
    assert claims["legacy"]["claim_id"] not in rows and len(rows) == 9

    def row(name):
        return rows[claims[name]["claim_id"]]

    assert {"source", "asset", "condition", "observable", "kind", "status", "evaluation_date", "data_source",
            "testability_type", "direction", "forecast_end_date", "headline", "claim_id", "evidence",
            "own_view", "trigger", "resolved"} <= set(row("drop"))
    assert [row(n)["trigger"] for n in ("drop", "hold", "risk", "unless", "name")] == [True, False, False, False, True]
    assert [row(n)["own_view"] for n in ("drop", "review", "quoted")] == [True, False, False]
    assert (row("name")["asset"], row("name")["resolved"]) == ("LINK", True)           # the canonical key
    assert (row("placeholder")["asset"], row("placeholder")["resolved"]) == (None, False)
    assert (row("raw")["asset"], row("raw")["resolved"]) == ("ZORBLAX INDUSTRIES", False)
    assert row("drop")["status"] == "not_evaluated" and row("drop")["date"] == "2026-07-01"


def test_conditional_report_is_compact_lists_evaluated_rows_first_and_never_unresolved_names():
    claims = [
        # Own-view observable triggers from three creators.
        _observable("More", "BTC", "if price breaks above 87,250", published="2026-07-04", asset_type="crypto", n=1),
        _observable("More", "BTC", "if support breaks below 81,600", stance="bearish", published="2026-07-03",
                    asset_type="crypto", n=2),
        _observable("More", "ETH", "if it breaks above 2,487", published="2026-07-02", asset_type="crypto", n=3),
        _cond("More", "BTC", "as long as support holds", published="2026-07-01", asset_type="crypto", n=4),
        _observable("Couch", "META", "if it drops under $600", published="2026-07-03", n=5),
        _observable("Couch", "NBIS", "if it goes under $200", published="2026-07-01", n=6),
        _observable("Parkev", "QCOM", "if it falls to $150", published="2026-07-02", n=7),
        _cond("Parkev", "MSFT", "if management executes", published="2026-07-02", n=8),
        # Unresolved subjects: counted, never printed, even when newest.
        _observable("Couch", None, "if it drops under $5", published="2026-07-04", subject_mention="Zorblax", n=9),
        _observable("More", None, "if the market breaks 100", published="2026-07-05", subject_mention="the market",
                    n=10),
        # Not the creator's own reviewed view, but evaluated: listed first.
        _cond("Z", "NVDA", "if the Fed cuts in September", review_required=True, condition_observable=True,
              condition_kind="policy_rate_decision", condition_status="met", condition_data_source="fomc_decisions",
              condition_evaluation_date="2026-09-17", n=11),
        # Neither own view nor evaluated: never listed.
        _observable("Y", "NVDA", "if it holds 150", review_required=True, n=12),
    ]
    rows = ra.conditional_forecast_report(claims)
    text = ra.format_conditional_report(rows, max_examples=4)
    lines = text.split("\n")
    assert len(lines) <= 4 + 3
    assert lines[0] == ("🔀 Conditional calls (scenarios, kept out of the track record): 10 own-view of 12 — "
                        "More 5 · Couch 3 · Parkev 2; condition status not_evaluated 11, met 1")
    assert lines[1] == ("9 hinge on a trigger (e.g. buy under $600), 1 on a qualifier (as long as support holds); "
                        "8 are observable (8 on a price level).")
    assert lines[2] == ('• Z on NVDA [bullish] if "the Fed cuts in September" — observable via fomc_decisions; '
                        'condition met')
    # Then the observable own-view triggers, interleaved across creators, newest first.
    assert lines[3:6] == [
        '• More on BTC [bullish] if "price breaks above 87,250" — observable via daily_close; condition not_evaluated',
        '• Couch on META [bullish] if "it drops under $600" — observable via daily_close; condition not_evaluated',
        '• Parkev on QCOM [bullish] if "it falls to $150" — observable via daily_close; condition not_evaluated',
    ]
    assert lines[6] == "Not listed: 2 on unresolved subjects · 3 more observable triggers."
    assert "ZORBLAX" not in text.upper() and "THE MARKET" not in text.upper()
    assert "met 1" in text and "fomc_decisions" in text
    # The default shows six examples: nine lines at most.
    full = ra.format_conditional_report(rows).split("\n")
    assert len(full) == 9 and full[-1] == "Not listed: 2 on unresolved subjects · 1 more observable trigger."
    # Nothing own-view: one line, no dangling creator list.
    assert ra.format_conditional_report(ra.conditional_forecast_report([claims[-1]])) == (
        "🔀 Conditional calls (scenarios, kept out of the track record): 0 own-view of 1; "
        "condition status not_evaluated 1")
    assert ra.format_conditional_report([]) == ""


# --- Quality header and descriptive counts ---------------------------------------------------


def test_quality_header_shows_denominators_and_exclusions():
    claims = [_c("A", "NVDA", "bullish"), _c("B", "NVDA", "bullish", review_required=True, n=2),
              _c("C", "NVDA", "bullish", schema_version="legacy", review_required=True, n=3)]
    gate = [{"video_id": "v1", "outcome": "included", "decided_at": "2026-07-10T00:00:00+00:00"},
            {"video_id": "v2", "outcome": "title_filtered", "decided_at": "2026-07-10T00:00:00+00:00"}]
    state = {"videos": {"v1": {"research_status": "complete", "transcript_hash": "h", "published_at": "2026-07-10",
                                "transcript_source": "supadata"},
                        "v3": {"research_status": "quota_deferred", "transcript_hash": "h", "published_at": "2026-07-10",
                                "transcript_source": "gemini_video"}}}
    q = ra.quality_header(claims, gate, state, [], date(2026, 7, 1), date(2026, 7, 31))
    assert q["discovered_videos"] == 2 and q["included_videos"] == 1
    assert q["fully_processed_videos"] == 1 and q["awaiting_retry_videos"] == 1
    assert q["claims"] == 3 and q["headline_claims"] == 1 and q["excluded_claims"] == 2
    assert q["review_required_claims"] == 2 and q["legacy_claims"] == 1
    assert q["transcript_sources"] == {"supadata": 1, "gemini_video": 1}
    text = ra.format_quality_header(q)
    assert "included 1/2 (50%)" in text and "headline-eligible 1/3 (33%)" in text
    assert "Among videos selected by the configured title and duration filters" in text
    assert ra.pct(0, 0) == "0/0 (n/a)"


def test_descriptive_separates_units():
    claims = [_c("A", "NVDA", "bullish", n=1), _c("A", "NVDA", "bullish", n=2, segment_id="seg-1"),
              _c("B", "NVDA", "bullish", n=3, catalysts=["earnings"]), _c("B", "TSLA", "bearish", n=4, catalysts=["earnings"])]
    d = ra.descriptive(claims)
    assert d["claims_by_asset"]["NVDA"] == 3
    assert len(d["segments_by_asset"]["NVDA"]) == 2 and len(d["sources_by_asset"]["NVDA"]) == 2
    assert d["catalysts"]["earnings"] == 2 and d["claims_by_month"]["2026-07"] == 4


# --- The strict scorecard --------------------------------------------------------------------


def _series(start, days, step):
    return {start + timedelta(days=i): 100.0 + step * i for i in range(days)}


def _fetcher(prices, adjustment="split_adjusted"):
    """A price fetcher that declares its provenance, as the production one
    does; a fetcher without one is refused by the scorecard on purpose."""
    def fetch(symbol, start, end):
        return prices.get(symbol, {})
    fetch.price_provenance = {"provider": "test", "adjustment": adjustment,
                              "corporate_action_status": adjustment}
    return fetch


def test_scorecard_scores_only_matured_testable_forecasts():
    today = date(2026, 8, 1)
    matured = _c("A", "NVDA", "bullish", published="2026-07-01", testable=True, forecast_end_date="2026-07-15",
                 claim_type="price_target", target_kind="absolute_value", target_value=110)
    young = _c("A", "NVDA", "bullish", published="2026-07-20", testable=True, forecast_end_date="2026-09-30", n=2)
    untestable = _c("A", "NVDA", "bullish", published="2026-07-01", testable=False, n=3)
    prices = {"nvda.us": _series(date(2026, 7, 1), 40, 1.0), "xlk.us": _series(date(2026, 7, 1), 40, 0.5)}
    sc = ra.scorecard([matured, young, untestable], today, _fetcher(prices), min_sample=1)
    excluded = sc.pop("_excluded")
    assert excluded == {"not_matured": 1, "not_testable": 1}
    assert sc.pop("_rankings_enabled") is False
    a = sc["A"]
    assert a["n"] == 1 and a["direction_hits"] == 1 and a["target_hits"] == 1
    assert a["raw_returns"][0] > 0 and a["excess_returns"][0] > 0 and a["mfe"][0] >= a["raw_returns"][0]
    # Only a price level counts as a target: a stance's number is not one.
    stance = dict(matured, claim_type="stance", claim_id="s")
    assert ra.scorecard([stance], today, _fetcher(prices), min_sample=1)["A"]["target_n"] == 0


def test_scorecard_series_cache_is_per_window():
    # Production fetchers slice to the requested window, so a series cached
    # by symbol alone served the first forecast's window to the second
    # (no_prices) and to every later benchmark (no excess return).
    june = _c("A", "NVDA", "bullish", published="2026-06-02", testable=True, forecast_end_date="2026-06-16",
              claim_type="forecast", n=1)
    july = _c("A", "NVDA", "bullish", published="2026-07-06", testable=True, forecast_end_date="2026-07-20",
              claim_type="forecast", n=2)
    prices = {"nvda.us": _rising(date(2026, 5, 25), date(2026, 7, 31)),
              "xlk.us": _rising(date(2026, 5, 25), date(2026, 7, 31), first=200.0, step=0.5)}
    for claims in ([june, july], [july, june]):
        fetch = creator_tracker.static_fetcher(prices)
        sc = ra.scorecard(claims, date(2026, 8, 1), fetch, min_sample=1)
        assert sc["_excluded"] == {} and sc["A"]["n"] == 2
        assert all(s["benchmark_return"] is not None and s["benchmark_symbol"] == "xlk.us" for s in sc["A"]["scored"])
        assert len(sc["A"]["excess_returns"]) == 2
        lag = timedelta(days=ra.MAX_PRICE_LAG_DAYS)
        windows = {(start, end) for symbol, start, end in fetch.requests if symbol == "nvda.us"}
        assert windows == {(date(2026, 6, 1), date(2026, 6, 16) + lag), (date(2026, 7, 5), date(2026, 7, 20) + lag)}


def test_non_price_metric_forecasts_are_not_scored():
    # "I'm estimating Apple generates $144 billion in free cash flow": a
    # forecast with a 1.44e11 "target", never a share price.
    fcf = _c("A", "AAPL", "bullish", published="2026-07-01", testable=True, forecast_end_date="2026-07-15",
             claim_type="forecast", forecast_metric="other", target_kind="absolute_value", target_value=1.44e11,
             evidence_text="I'm estimating Apple generates $144 billion in free cash flow")
    others = [dict(fcf, claim_id=f"m-{m}", forecast_metric=m) for m in ("revenue", "earnings", "margin", "rate",
                                                                         "market_cap")]
    prices = {"aapl.us": _rising(date(2026, 6, 25), date(2026, 7, 31)),
              "xlk.us": _rising(date(2026, 6, 25), date(2026, 7, 31), first=200.0)}
    fetch = creator_tracker.static_fetcher(prices)
    sc = ra.scorecard([fcf] + others, date(2026, 8, 1), fetch, min_sample=1)
    assert sc["_excluded"] == {"non_price_metric": 6} and "A" not in sc
    assert fetch.requests == []                       # never priced, so never tested as a share price
    assert canonical_claims.price_target_of(fcf) is None
    # The same forecast about the price (or with no metric stated) is scored.
    price = dict(fcf, claim_id="p", forecast_metric="price", target_value=110.0)
    unstated = dict(fcf, claim_id="u", forecast_metric=None, target_value=None, target_kind=None)
    sc = ra.scorecard([price, unstated], date(2026, 8, 1), creator_tracker.static_fetcher(prices), min_sample=1)
    assert sc["A"]["n"] == 2 and sc["A"]["target_n"] == 1


def test_implausible_price_target_is_not_tested():
    base = dict(published="2026-07-01", testable=True, forecast_end_date="2026-07-15", claim_type="price_target",
                forecast_metric="price", target_kind="absolute_value")
    huge = _c("A", "NVDA", "bullish", target_value=1.44e11, n=1, **base)
    wrong_side = _c("A", "NVDA", "bullish", target_value=50.0, n=2, **base)   # below the entry: "reached" at once
    fine = _c("A", "NVDA", "bullish", target_value=110.0, n=3, **base)
    prices = {"nvda.us": _rising(date(2026, 6, 25), date(2026, 7, 31)),
              "xlk.us": _rising(date(2026, 6, 25), date(2026, 7, 31), first=200.0)}
    sc = ra.scorecard([huge, wrong_side, fine], date(2026, 8, 1), creator_tracker.static_fetcher(prices), min_sample=1)
    a = sc["A"]
    assert a["n"] == 3 and a["target_n"] == 1 and a["target_hits"] == 1
    scored = {s["claim_id"]: s for s in a["scored"]}
    assert scored[huge["claim_id"]]["target_skipped"] == "implausible_level"
    assert scored[huge["claim_id"]]["target_test_method"] is None and scored[huge["claim_id"]]["stated_target"] == 1.44e11
    assert scored[wrong_side["claim_id"]]["target_skipped"] == "wrong_side_of_entry"
    assert scored[fine["claim_id"]]["target_skipped"] is None and scored[fine["claim_id"]]["target_value"] == 110.0
    assert scored[fine["claim_id"]]["target_test_method"] == "daily_close"
    assert scored[fine["claim_id"]]["target_first_reached_date"] == "2026-07-10"


def test_target_touch_ignores_the_entry_day_range_printed_before_the_video():
    days = _sessions(date(2026, 7, 1), date(2026, 7, 31))
    closes = {d: 100.0 for d in days}
    highs = {d: 101.0 for d in days}
    lows = {d: 99.0 for d in days}
    highs[date(2026, 7, 2)] = 115.0                  # the entry day's high
    fetch = creator_tracker.static_fetcher({"nvda.us": closes, "xlk.us": closes},
                                           highs={"nvda.us": highs}, lows={"nvda.us": lows})
    base = dict(published="2026-07-02", testable=True, forecast_end_date="2026-07-20", claim_type="price_target",
                target_kind="absolute_value", target_value=110.0)
    # 14:00 UTC is 10:00 in New York: the video went up during the session,
    # so that day's high may have printed before it.
    during = _c("A", "NVDA", "bullish", n=1, **dict(base, published_at="2026-07-02T14:00:00+00:00"))
    # 12:00 UTC is 08:00 in New York, before the open: the whole session follows it.
    before = _c("A", "NVDA", "bullish", n=2, **base)
    sc = ra.scorecard([during, before], date(2026, 8, 1), fetch, min_sample=1)
    scored = {s["claim_id"]: s for s in sc["A"]["scored"]}
    assert scored[during["claim_id"]]["target_test_method"] == "intraday_touch"
    assert scored[during["claim_id"]]["target_reached"] is False
    assert scored[before["claim_id"]]["target_reached"] is True
    assert scored[before["claim_id"]]["target_first_reached_date"] == "2026-07-02"


def test_macro_subject_is_not_an_instrument():
    base = dict(published="2026-09-02", testable=True, forecast_end_date="2026-09-30", claim_type="forecast")
    claims = [
        _c("FinVid", None, "bullish", subject_mention="DRAM ASPs", asset_type="commodity", forecast_metric="price",
           n=1, **base),
        _c("Couch", None, "bullish", subject_mention="the Fed", asset_type="macro", n=2, **base),
        _c("X", None, "bearish", subject_mention="S&P 500", asset_type="index", n=3, **base),
        _c("X", None, "bullish", subject_mention="the stock market", subject_kind="market", n=4, **base),
        # A name the registry cannot resolve, and recorded tickers it cannot
        # resolve (GOLD typed commodity, TSLA typed crypto) are resolution
        # gaps, not "not an instrument".
        _c("Y", None, "bullish", subject_mention="Zorblax Industries", n=5, **base),
        _c("Z", "GOLD", "bullish", asset_type="commodity", n=6, **base),
        _c("Z", "TSLA", "bullish", asset_type="crypto", n=7, **base),
    ]
    fetch = creator_tracker.static_fetcher({})
    sc = ra.scorecard(claims, date(2026, 10, 16), fetch, min_sample=1)
    assert sc["_excluded"] == {"not_an_instrument": 4, "unresolved_instrument": 3}
    assert fetch.requests == []
    assert ra.format_exclusions(sc["_excluded"]) == (
        "Not scored: 3 with a ticker not in the instrument registry · "
        "4 not an instrument (macro, index, sector or commodity subject).")


def test_scorecard_lines_show_the_benchmark_symbol_and_per_claim_lines_for_small_n():
    prices = {"nvda.us": _rising(date(2026, 6, 25), date(2026, 8, 31)),
              "xlk.us": _rising(date(2026, 6, 25), date(2026, 8, 31), first=200.0, step=0.5)}
    base = dict(testable=True, claim_type="price_target", forecast_metric="price")
    up = _c("A", "NVDA", "bullish", published="2026-07-01", forecast_end_date="2026-07-15",
            target_kind="absolute_value", target_value=110.0, n=1, **base)
    down = _c("A", "NVDA", "bearish", published="2026-07-06", forecast_end_date="2026-07-20", n=2, **base)
    young = _c("A", "NVDA", "bullish", published="2026-07-20", forecast_end_date="2026-12-31", n=3, **base)
    many = [_c("B", "NVDA", "bullish", published=f"2026-07-{day:02d}", forecast_end_date=f"2026-08-{day:02d}",
               n=10 + day, **base) for day in (6, 7, 8, 9, 10)]
    sc = ra.scorecard([up, down, young] + many, date(2026, 8, 31), creator_tracker.static_fetcher(prices),
                      min_sample=1)
    excluded, min_sample, rankings = sc.pop("_excluded"), sc.pop("_min_sample"), sc.pop("_rankings_enabled")
    lines = ra.format_scorecard_lines(sc, excluded, min_sample, rankings, []).split("\n")
    a = lines.index(next(line for line in lines if line.startswith("• A:")))
    assert lines[a].endswith(" (unranked)") and "(daily_close 1)" in lines[a]
    assert "benchmark sector 2 vs xlk.us" in lines[a] and "n=2" in lines[a]
    assert lines[a + 1].startswith("   NVDA ↑ 2026-07-01→2026-07-15: +")
    assert " vs xlk.us +" in lines[a + 1]
    assert lines[a + 1].endswith(" — right direction; target 110.00 reached (daily_close)")
    assert lines[a + 2].startswith("   NVDA ↓ 2026-07-06→2026-07-20: +") and lines[a + 2].endswith(" — wrong direction")
    # Five or more scored forecasts: the rates speak, no per-forecast lines.
    b = lines.index(next(line for line in lines if line.startswith("• B:")))
    assert "n=5" in lines[b] and not lines[b + 1].startswith("   ")
    assert lines[-1] == "Not scored: 1 dated price forecast not matured yet."
    assert ra.format_exclusions({"not_headline": 1428, "not_testable": 1214, "not_matured": 83}) == (
        "Not scored: 83 dated price forecasts not matured yet · 1,214 not testable (no dated, measurable "
        "forecast) · 1,428 not the creator's own reviewed view.")
    assert ra.format_exclusions({"condition_not_met": 2, "not_testable": 1}) == (
        "Not scored: 2 conditional, condition not met · 1 not testable (no dated, measurable forecast).")


def test_report_renders_and_respects_min_sample():
    today = date(2026, 8, 1)
    claims = [_c("A", "NVDA", "bullish", published="2026-07-01", testable=True, forecast_end_date="2026-07-15")]
    prices = {"nvda.us": _series(date(2026, 7, 1), 40, 1.0)}
    text = ra.build_report(claims, [], {"videos": {}}, [], date(2026, 6, 30), today, today=today,
                           price_fetcher=_fetcher(prices))
    assert "Research data quality" in text
    # Rankings are off by default: the scorecard is labelled experimental
    # and every source is unranked, whatever its sample.
    assert "EXPERIMENTAL" in text and "(unranked)" in text
    assert "n=1" in text and "not investment advice" in text


# --- The creators-vs-market block -------------------------------------------------------------


def _tracker_inputs():
    claims = [_c(source, "NVDA", "bullish", bucket="unspecified", published=day, claim_type="recommendation")
              for source, day in (("A", "2026-09-08"), ("B", "2026-09-10"), ("C", "2026-09-14"))]
    prices = {"spy.us": _rising(date(2026, 9, 1), date(2026, 10, 9), first=500.0),
              "nvda.us": _rising(date(2026, 9, 1), date(2026, 10, 9), first=100.0, step=2.0)}
    return claims, prices


def test_build_report_includes_the_tracker_block_when_given():
    claims, prices = _tracker_inputs()
    record = creator_tracker.track_record(claims, date(2026, 10, 9), creator_tracker.static_fetcher(prices))
    assert record["stats"]["scored"] == 3
    start, end = date(2026, 9, 1), date(2026, 10, 9)
    plain = ra.build_report(claims, [], {"videos": {}}, [], start, end, today=end)
    text = ra.build_report(claims, [], {"videos": {}}, [], start, end, today=end, tracker_record=record)
    block = creator_tracker.format_summary_lines(record)
    assert block[0].startswith("📈 Creators vs the market") and "mark-to-market, not a verdict" in block[0]
    assert "Creators vs the market" not in plain and "\n".join(block) in text
    assert text.index("Consensus (") < text.index(block[0]) < text.index("not investment advice")
    assert "3 calls · right 3/3" in text
    # A record with nothing marked adds nothing.
    empty = creator_tracker.track_record([], date(2026, 10, 9), creator_tracker.static_fetcher(prices))
    assert ra.build_report(claims, [], {"videos": {}}, [], start, end, today=end, tracker_record=empty) == plain


def test_main_marks_the_tracker_block_from_the_price_cache(monkeypatch, capsys):
    claims, prices = _tracker_inputs()
    price_cache.reset({s: {"from": min(c), "to": max(c), "closes": dict(c), "adjustment": "split_adjusted"}
                       for s, c in prices.items()})
    monkeypatch.setattr(canonical_claims, "load_canonical_claims", lambda *a, **k: claims)
    import channel_scorecard
    monkeypatch.setattr(channel_scorecard, "fetch_price_series", creator_tracker.cache_fetcher())

    class _Monday(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(ra, "datetime", _Monday)
    assert ra.main(["--dry-run", "--days", "45"]) == 0
    out = capsys.readouterr().out
    assert "marked to Fri Oct 9 close (mark-to-market, not a verdict)" in out and "Scorecard (" in out
    built = []
    monkeypatch.setattr(creator_tracker, "track_record", lambda *a, **k: built.append(a))
    assert ra.main(["--dry-run", "--no-scorecard"]) == 0
    out = capsys.readouterr().out
    assert built == [] and "Creators vs the market" not in out and "Scorecard (" not in out
