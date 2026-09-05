"""Atomic claims: evidence, attribution, entity, horizon and legacy rules."""
import json
from datetime import date

import claims as cm
import transcript_normalize as tn


def _nt(text):
    return tn.normalize_transcript(text, "v1")


def _ctx(published="2026-07-01T00:00:00+00:00", title="T"):
    return {"video_id": "v1", "channel_id": "c1", "channel_name": "Chan", "video_title": title,
            "published_at": published, "transcript_source": "supadata", "run_key": "rk",
            "extraction_model": "gemini-3.7-flash"}


def _validate(text, raw, **ctx_over):
    ctx = _ctx()
    ctx.update(ctx_over)
    nt = _nt(text)
    claims, warnings = cm.validate_claims(raw, nt, ctx)
    return claims, warnings


def _claim(**over):
    base = {"attribution_type": "speaker_personal_view", "claim_type": "forecast",
            "is_forward_looking": True, "subject_mention": "Nvidia", "stance": "bearish",
            "forecast_metric": "price", "forecast_direction": "decrease",
            "horizon_original": "over the next three months", "certainty_original": "expect",
            "certainty_level": "medium", "extraction_confidence": "high"}
    base.update(over)
    return base


# --- evidence ---

def test_evidence_must_be_locatable_normalization_aware():
    text = "I expect Nvidia — yes NVIDIA — to fall over the next three months, but I remain bullish over five years."
    ev = "i expect nvidia - yes nvidia - to fall over the next three months"
    assert cm.locate_evidence(ev, text) == (0, len("I expect Nvidia — yes NVIDIA — to fall over the next three months"))
    assert cm.locate_evidence("I expect Nvidia to fall soon", text) is None   # paraphrase
    assert cm.locate_evidence("short", text) is None


def test_missing_evidence_keeps_claim_out_of_primary_analytics():
    claims, _ = _validate("Nvidia looks fine today.", [_claim(evidence_text="Nvidia will crash tomorrow")])
    c = claims[0]
    assert c["review_required"] is True and "evidence_not_found" in c["review_reasons"]
    assert c["evidence_start_character"] is None and c["testable"] is False
    assert not cm.is_headline_claim(c)


def test_numbers_must_appear_in_evidence():
    text = "This stock could reach $100 this year and $150 by 2030."
    ok, _ = _validate(text, [_claim(claim_type="price_target", stance="bullish", target_kind="absolute_value",
                                    target_value=100, currency="USD", horizon_original="this year",
                                    subject_mention="This stock",
                                    evidence_text="This stock could reach $100 this year")])
    assert ok[0]["review_required"] is False
    bad, _ = _validate(text, [_claim(claim_type="price_target", stance="bullish", target_kind="absolute_value",
                                     target_value=120, horizon_original="this year", subject_mention="This stock",
                                     evidence_text="This stock could reach $100 this year")])
    assert bad[0]["review_required"] and "number_not_in_evidence:target_value" in bad[0]["review_reasons"]
    assert "unsupported_evidence" in bad[0]["testability_issues"]


def test_subject_named_in_the_segment_supports_a_pronoun_in_the_evidence():
    text = "Let me talk about Nvidia. The stock could reach $100 this year in my view."
    claims, _ = _validate(text, [_claim(claim_type="price_target", stance="bullish", target_kind="absolute_value",
                                        target_value=100, horizon_original="this year",
                                        evidence_text="The stock could reach $100 this year")])
    assert "subject_not_in_evidence" not in claims[0]["review_reasons"]
    claims, _ = _validate(text, [_claim(claim_type="price_target", stance="bullish", subject_mention="Broadcom",
                                        target_kind="absolute_value", target_value=100, horizon_original="this year",
                                        evidence_text="The stock could reach $100 this year")])
    assert "subject_not_in_evidence" in claims[0]["review_reasons"]


def test_number_support_handles_units_and_multipliers():
    assert cm.number_supported(150000, "between 150 and 180 thousand next year")
    assert cm.number_supported(150000, "150-180k next year")
    assert cm.number_supported(150, "between 150 and 180 thousand")
    assert cm.number_supported(5.25, "margins fall 5.25%")
    assert cm.number_supported(1200.5, "revenue of $1,200.50")
    assert cm.number_supported(2e9, "a $2 billion buyback")
    assert not cm.number_supported(200, "reach $100")


def test_two_targets_two_horizons_stay_two_claims():
    text = "This stock could reach $100 this year and $150 by 2030."
    claims, _ = _validate(text, [
        _claim(claim_type="price_target", stance="bullish", target_kind="absolute_value", target_value=100,
               currency="USD", horizon_original="this year", forecast_direction="increase",
               evidence_text="This stock could reach $100 this year", subject_mention="This stock"),
        _claim(claim_type="price_target", stance="bullish", target_kind="absolute_value", target_value=150,
               currency="USD", horizon_original="by 2030", forecast_direction="increase",
               evidence_text="$150 by 2030", subject_mention="This stock"),
    ])
    assert len(claims) == 2 and claims[1]["repeat_of_claim_id"] is None
    assert claims[0]["forecast_end_date"] == "2026-12-31" and claims[1]["forecast_end_date"] == "2030-12-31"
    assert claims[0]["horizon_bucket"] == "medium" and claims[1]["horizon_bucket"] == "long"


def test_short_bearish_and_long_bullish_are_two_claims_never_mixed():
    text = "I expect Nvidia to fall over the next three months, but I remain bullish over five years."
    claims, _ = _validate(text, [
        _claim(evidence_text="I expect Nvidia to fall over the next three months"),
        _claim(claim_type="stance", stance="bullish", forecast_direction=None, forecast_metric=None,
               horizon_original="over five years", evidence_text="I remain bullish over five years"),
    ])
    assert [c["stance"] for c in claims] == ["bearish", "bullish"]
    assert [c["horizon_bucket"] for c in claims] == ["short", "long"]
    assert claims[1]["repeat_of_claim_id"] is None
    assert all(c["ticker"] == "NVDA" and c["ticker_source"] == "curated_mapping" for c in claims)
    legacy = cm.claims_to_legacy_signals(claims)
    asset = legacy["assets"][0]
    assert asset["stance"] == "bearish"              # shortest horizon leads the legacy shape
    assert asset["reduced"].startswith("conflicting_horizons")
    assert set(asset["claim_ids"]) == {c["claim_id"] for c in claims}


def test_revenue_growth_and_margin_decline_are_two_claims():
    text = "For Micron, revenue should rise but margins may fall next quarter."
    claims, _ = _validate(text, [
        _claim(subject_mention="Micron", forecast_metric="revenue", forecast_direction="increase",
               stance="bullish", certainty_original="should", horizon_original="next quarter",
               evidence_text="revenue should rise"),
        _claim(subject_mention="Micron", forecast_metric="margin", forecast_direction="decrease",
               stance="bearish", certainty_original="may", certainty_level="low", horizon_original="next quarter",
               evidence_text="margins may fall next quarter"),
    ])
    assert len(claims) == 2 and {c["forecast_metric"] for c in claims} == {"revenue", "margin"}
    assert claims[1]["repeat_of_claim_id"] is None


# --- attribution rules ---

def test_a_question_is_not_a_forecast():
    text = "Could Apple fall 30 percent from here? Let's look."
    claims, _ = _validate(text, [_claim(subject_mention="Apple", forecast_direction="decrease",
                                        evidence_text="Could Apple fall 30 percent from here?")])
    c = claims[0]
    assert c["claim_type"] == "question" and c["is_forward_looking"] is False
    assert c["attribution_type"] == "interviewer_question" and c["testable"] is False


def test_quoted_analyst_target_is_not_the_hosts_view():
    text = "Goldman expects the stock to reach $200 next year."
    claims, _ = _validate(text, [_claim(claim_type="price_target", attribution_type="speaker_quoting_third_party",
                                        attributed_person_or_organization="Goldman", stance="not_applicable",
                                        target_kind="absolute_value", target_value=200, subject_mention="the stock",
                                        horizon_original="next year", evidence_text="Goldman expects the stock to reach $200")])
    c = claims[0]
    assert c["claim_type"] == "third_party_view" and c["attributed_person_or_organization"] == "Goldman"
    assert not cm.is_headline_claim(c)
    # If the model attributed it to the host anyway, the text gets it reviewed.
    claims, _ = _validate(text, [_claim(claim_type="price_target", stance="bullish", target_kind="absolute_value",
                                        target_value=200, subject_mention="the stock", horizon_original="next year",
                                        evidence_text="Goldman expects the stock to reach $200")])
    assert "possible_third_party_view" in claims[0]["review_reasons"]


def test_retrospective_claim_is_not_a_new_prediction():
    text = "Last year I said Bitcoin would double, and it did."
    claims, _ = _validate(text, [_claim(subject_mention="Bitcoin", stance="bullish", forecast_direction="increase",
                                        horizon_original="last year", evidence_text="Last year I said Bitcoin would double")])
    c = claims[0]
    assert c["claim_type"] == "historical_claim" and c["attribution_type"] == "retrospective_claim"
    assert c["is_forward_looking"] is False and c["forecast_end_date"] is None


def test_ownership_alone_is_not_a_buy_and_praise_alone_is_not_a_buy():
    text = "I own Tesla in my portfolio. Costco is a wonderful business."
    claims, w = _validate(text, [
        _claim(claim_type="portfolio_disclosure", subject_mention="Tesla", stance="neutral",
               recommendation_action="buy", is_forward_looking=False, evidence_text="I own Tesla in my portfolio"),
        _claim(claim_type="opinion", subject_mention="Costco", stance="bullish", recommendation_action="buy",
               is_forward_looking=False, evidence_text="Costco is a wonderful business"),
    ])
    assert claims[0]["recommendation_action"] == "none" and claims[0]["portfolio_disclosure"] == "owns_unspecified"
    assert claims[1]["recommendation_action"] == "none"
    assert len(w) == 2


def test_explicit_buy_recommendation_is_kept():
    text = "I would buy Palantir under $20, that is my plan."
    claims, _ = _validate(text, [_claim(claim_type="recommendation", subject_mention="Palantir", stance="bullish",
                                        recommendation_action="buy", condition="under $20", is_forward_looking=False,
                                        evidence_text="I would buy Palantir under $20")])
    assert claims[0]["recommendation_action"] == "buy" and claims[0]["ticker"] == "PLTR"
    assert claims[0]["condition"] == "under $20"


# --- entity resolution ---

def test_unsupported_ticker_is_dropped_and_curated_mapping_used():
    text = "I like Chevron a lot here."
    claims, _ = _validate(text, [_claim(subject_mention="Chevron", ticker_spoken="CVX", stance="bullish",
                                        forecast_direction="increase", evidence_text="I like Chevron a lot here")])
    c = claims[0]
    assert c["ticker_spoken"] == "CVX"
    assert c["ticker"] == "CVX" and c["ticker_source"] == "curated_mapping"   # via the table, not the model
    assert "ticker_not_in_evidence" in c["review_reasons"]


def test_spoken_ticker_and_title_ticker_are_confirmed():
    text = "I am buying more NVDA today."
    claims, _ = _validate(text, [_claim(subject_mention="NVDA", ticker_spoken="NVDA", stance="bullish",
                                        forecast_direction="increase", evidence_text="I am buying more NVDA today")])
    assert claims[0]["ticker_source"] == "spoken" and claims[0]["entity_resolution_status"] == "confirmed"
    claims, _ = _validate("the company looks strong now", [_claim(subject_mention="the company", ticker_spoken="MU",
                                                                  stance="bullish", forecast_direction="increase",
                                                                  evidence_text="the company looks strong now")],
                          video_title="Why $MU is my top pick")
    assert claims[0]["ticker_source"] == "title"


def test_unknown_and_ambiguous_names_stay_unresolved():
    claims, _ = _validate("Zorblax Industries will double this year.", [
        _claim(subject_mention="Zorblax Industries", stance="bullish", forecast_direction="increase",
               horizon_original="this year", evidence_text="Zorblax Industries will double this year")])
    assert claims[0]["ticker"] is None and claims[0]["ticker_source"] == "unresolved"
    assert claims[0]["testable"] is False and "unresolved_entity" in claims[0]["testability_issues"]
    claims, _ = _validate("Target should rally this year.", [
        _claim(subject_mention="Target", stance="bullish", forecast_direction="increase",
               horizon_original="this year", evidence_text="Target should rally this year")])
    assert claims[0]["entity_resolution_status"] == "ambiguous" and claims[0]["ticker"] is None
    assert claims[0]["review_required"] is True


# --- horizons ---

def test_vague_horizons_get_no_dates():
    P = date(2026, 7, 1)
    for vague in ("soon", "eventually", "long term", "in the future", "over time"):
        start, end, bucket, issue = cm.resolve_horizon(vague, P)
        assert (start, end, issue) == (None, None, "vague_horizon"), vague
    assert cm.resolve_horizon("long term", P)[2] == "long"
    assert cm.resolve_horizon("short term", P)[2] == "short"
    assert cm.resolve_horizon(None, P)[3] == "missing_horizon"
    assert cm.resolve_horizon("next year", None)[3] == "missing_publication_date"


def test_precise_horizons_follow_documented_rules():
    P = date(2026, 7, 1)
    assert cm.resolve_horizon("by the end of the year", P)[1] == date(2026, 12, 31)
    assert cm.resolve_horizon("next year", P)[1] == date(2027, 12, 31)
    assert cm.resolve_horizon("by 2030", P)[1] == date(2030, 12, 31)
    assert cm.resolve_horizon("next quarter", P)[1] == date(2026, 12, 31)
    assert cm.resolve_horizon("this quarter", P)[1] == date(2026, 9, 30)
    assert cm.resolve_horizon("over the next three months", P)[1] == date(2026, 10, 1)
    assert cm.resolve_horizon("within five years", P)[1] == date(2031, 7, 1)
    assert cm.resolve_horizon("in 6 weeks", P)[1] == date(2026, 8, 12)
    assert cm.resolve_horizon("Q1 2027", P)[1] == date(2027, 3, 31)
    assert cm.resolve_horizon("over the next three months", P)[2] == "short"
    assert cm.resolve_horizon("within five years", P)[2] == "long"


def test_conditional_forecast_keeps_condition_and_is_not_testable():
    text = "If the Fed cuts in September, small caps should rally into year end."
    claims, _ = _validate(text, [_claim(subject_mention="small caps", asset_type="sector", stance="bullish",
                                        forecast_direction="increase", condition="if the Fed cuts in September",
                                        horizon_original="into year end", certainty_original="should",
                                        evidence_text="If the Fed cuts in September, small caps should rally into year end")])
    c = claims[0]
    assert c["condition"] == "if the Fed cuts in September"
    assert "conditional_outcome_not_observable" in c["testability_issues"] and c["testable"] is False
    assert c["forecast_end_date"] == "2026-12-31"
    assert c["entity_resolution_status"] == "confirmed" and c["ticker"] is None  # a sector, not a security


def test_testable_forecast_has_everything():
    text = "I expect NVDA to reach $200 by the end of the year."
    claims, _ = _validate(text, [_claim(claim_type="price_target", subject_mention="NVDA", ticker_spoken="NVDA",
                                        stance="bullish", forecast_direction="increase", target_kind="absolute_value",
                                        target_value=200, horizon_original="by the end of the year",
                                        evidence_text="I expect NVDA to reach $200 by the end of the year")])
    c = claims[0]
    assert c["testable"] is True and c["testability_issues"] == []
    assert c["currency"] == "USD" and c["segment_id"] and c["claim_id"].startswith("clm_")
    assert c["schema_version"] == cm.SCHEMA_VERSION and c["extraction_prompt_version"] == cm.EXTRACTION_PROMPT_VERSION
    assert cm.is_headline_claim(c)


def test_partial_coverage_blocks_testability_and_headline():
    text = "I expect NVDA to reach $200 by the end of the year."
    ctx = _ctx()
    claims, _ = cm.validate_claims([_claim(claim_type="price_target", subject_mention="NVDA", ticker_spoken="NVDA",
                                           stance="bullish", forecast_direction="increase", target_kind="absolute_value",
                                           target_value=200, horizon_original="by the end of the year",
                                           evidence_text="I expect NVDA to reach $200 by the end of the year")],
                                   _nt(text), ctx, coverage_status="partial", chunk_id="c1")
    assert claims[0]["coverage_status"] == "partial" and claims[0]["chunk_id"] == "c1"
    assert "incomplete_transcript_coverage" in claims[0]["testability_issues"]
    assert not cm.is_headline_claim(claims[0])


def test_duplicates_from_overlapping_chunks_are_marked_once():
    text = "I expect NVDA to reach $200 by the end of the year."
    raw = _claim(claim_type="price_target", subject_mention="NVDA", ticker_spoken="NVDA", stance="bullish",
                 forecast_direction="increase", target_kind="absolute_value", target_value=200,
                 horizon_original="by the end of the year",
                 evidence_text="I expect NVDA to reach $200 by the end of the year")
    a, _ = cm.validate_claims([raw], _nt(text), _ctx(), "chunked_full", "c1")
    b, _ = cm.validate_claims([raw], _nt(text), _ctx(), "chunked_full", "c2")
    merged = cm.dedupe_across_chunks(a + b)
    primaries = [c for c in merged if not c["repeat_of_claim_id"]]
    assert len(primaries) == 1
    legacy = cm.claims_to_legacy_signals(merged)
    assert len(legacy["assets"]) == 1 and len(legacy["assets"][0]["claim_ids"]) == 1


def test_ids_are_deterministic_for_the_same_run():
    text = "I expect NVDA to reach $200 by the end of the year."
    raw = _claim(subject_mention="NVDA", stance="bullish", evidence_text="I expect NVDA to reach $200")
    a, _ = _validate(text, [raw])
    b, _ = _validate(text, [raw])
    assert a[0]["claim_id"] == b[0]["claim_id"]
    c, _ = _validate(text, [raw], run_key="other")
    assert c[0]["claim_id"] != a[0]["claim_id"]


# --- parsing and legacy ---

def test_raw_claims_distinguishes_missing_from_empty():
    assert cm.raw_claims_from({"claims": []}) == []
    assert cm.raw_claims_from({"claims": "oops"}) is None
    assert cm.raw_claims_from({"summary": "s"}) is None
    assert cm.raw_claims_from({"claims": [1, 2]}) is None
    assert cm.parse_json_object("```json\n{\"claims\": []}\n```") == {"claims": []}
    assert cm.parse_json_object("Here: {\"a\": 1} thanks") == {"a": 1}
    assert cm.parse_json_object("[1]") is None


def test_legacy_signals_shape_matches_consumers():
    text = "I like Nvidia and I am selling Tesla this week. Bitcoin between 150 and 180 thousand next year."
    claims, _ = _validate(text, [
        _claim(subject_mention="Nvidia", stance="bullish", forecast_direction="increase", asset_type="stock",
               recommendation_action="none", certainty_level="high", horizon_original=None,
               evidence_text="I like Nvidia"),
        _claim(subject_mention="Tesla", stance="bearish", forecast_direction="decrease", asset_type="stock",
               recommendation_action="sell", horizon_original="this week", evidence_text="I am selling Tesla this week"),
        _claim(claim_type="price_target", subject_mention="Bitcoin", stance="bullish", asset_type="crypto",
               target_kind="range", target_low=150000, target_high=180000, horizon_original="next year",
               evidence_text="Bitcoin between 150 and 180 thousand next year"),
    ])
    legacy = cm.claims_to_legacy_signals(claims)
    by = {a["ticker"]: a for a in legacy["assets"]}
    assert set(by) == {"NVDA", "TSLA", "BTC"}
    assert by["NVDA"]["conviction"] == "high" and by["NVDA"]["action"] == "none"
    assert by["TSLA"]["action"] == "sell" and by["TSLA"]["horizon"] == "short"
    assert by["BTC"]["price_target"] == 165000 and by["BTC"]["reduced"] == "range_midpoint"
    assert legacy["market_sentiment"] == "mixed"
    for a in legacy["assets"]:
        assert set(a) >= {"name", "ticker", "type", "stance", "conviction", "action", "catalysts",
                          "price_target", "horizon"}
    # The existing aggregator reads it unchanged.
    import signals_data
    stats = signals_data.aggregate_assets([{"channel_name": "Chan", "signals": legacy}])
    assert stats["NVDA"]["bull"] == 1 and stats["TSLA"]["bear"] == 1 and stats["BTC"]["targets"] == [165000]


def test_legacy_import_marks_evidence_unavailable():
    record = {"video_id": "old1", "channel_id": "c", "channel_name": "Chan", "video_title": "t",
              "published_at": "2026-07-24T17:00:15+00:00",
              "signals": {"assets": [{"name": "Bitcoin", "ticker": "BTC", "type": "crypto", "stance": "bullish",
                                      "conviction": "medium", "action": "watch", "price_target": 70000,
                                      "horizon": "short", "catalysts": ["cycle low"]}]}}
    out = cm.legacy_record_to_claims(record)
    assert len(out) == 1
    c = out[0]
    assert c["schema_version"] == "legacy" and c["review_required"] and c["evidence_text"] is None
    assert c["testable"] is False and not cm.is_headline_claim(c)
    assert cm.legacy_record_to_claims({"signals": None}) == []
    assert cm.legacy_record_to_claims(record)[0]["claim_id"] == c["claim_id"]  # deterministic
