"""Analytics over canonical claims: consensus, flips, quality header, scorecard."""
from datetime import date, timedelta

import research_analytics as ra


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


def test_flips_need_same_source_asset_and_bucket():
    claims = [
        _c("A", "NVDA", "bullish", published="2026-06-01", catalysts=["earnings"]),
        _c("A", "NVDA", "bearish", published="2026-07-01", catalysts=["tariffs"], risks=["china"]),
        _c("A", "NVDA", "bullish", bucket="long", published="2026-07-02"),   # other horizon: no flip
        _c("B", "NVDA", "bearish", published="2026-07-03"),                  # other source: no flip
    ]
    flips = ra.find_flips(claims)
    assert len(flips) == 1
    f = flips[0]
    assert (f["from"], f["to"], f["elapsed_days"]) == ("bullish", "bearish", 30)
    assert f["previous_claim_id"] and f["new_claim_id"] and f["changed_catalysts"] == ["tariffs"]
    assert f["changed_risks"] == ["china"]


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


def _series(start, days, step):
    return {start + timedelta(days=i): 100.0 + step * i for i in range(days)}


def test_scorecard_scores_only_matured_testable_forecasts():
    today = date(2026, 8, 1)
    matured = _c("A", "NVDA", "bullish", published="2026-07-01", testable=True, forecast_end_date="2026-07-15",
                 target_kind="absolute_value", target_value=110)
    young = _c("A", "NVDA", "bullish", published="2026-07-20", testable=True, forecast_end_date="2026-09-30", n=2)
    untestable = _c("A", "NVDA", "bullish", published="2026-07-01", testable=False, n=3)
    prices = {"nvda.us": _series(date(2026, 7, 1), 40, 1.0), "spy.us": _series(date(2026, 7, 1), 40, 0.5)}
    sc = ra.scorecard([matured, young, untestable], today, lambda s, a, b: prices.get(s, {}), min_sample=1)
    excluded = sc.pop("_excluded")
    assert excluded == {"not_matured": 1, "not_testable": 1}
    a = sc["A"]
    assert a["n"] == 1 and a["direction_hits"] == 1 and a["target_hits"] == 1
    assert a["raw_returns"][0] > 0 and a["excess_returns"][0] > 0 and a["mfe"][0] >= a["raw_returns"][0]


def test_report_renders_and_respects_min_sample():
    today = date(2026, 8, 1)
    claims = [_c("A", "NVDA", "bullish", published="2026-07-01", testable=True, forecast_end_date="2026-07-15")]
    prices = {"nvda.us": _series(date(2026, 7, 1), 40, 1.0)}
    text = ra.build_report(claims, [], {"videos": {}}, [], date(2026, 6, 30), today, today=today,
                           price_fetcher=lambda s, a, b: prices.get(s, {}))
    assert "Research data quality" in text
    assert "unranked: below sample minimum" in text
    assert "n=1" in text and "not investment advice" in text
