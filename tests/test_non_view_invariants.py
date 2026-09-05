"""Hardening item 7: a reclassified non-view claim never keeps fields that
read as the speaker's own forecast."""
import claims as cm
import transcript_normalize as tn

FORECAST_SLOTS = ("forecast_direction", "target_value", "target_low", "target_high", "horizon_bucket",
                  "forecast_end_date", "forecast_start_date", "condition", "trigger")


def _ctx():
    return {"video_id": "v1", "channel_id": "c1", "channel_name": "Chan", "video_title": "T",
            "published_at": "2026-07-01T00:00:00+00:00", "transcript_source": "supadata", "run_key": "rk"}


def _validate(text, raw):
    claims, _ = cm.validate_claims([raw], tn.normalize_transcript(text, "v1"), _ctx())
    return claims[0]


def _forecast_like(evidence, subject="Nvidia", **over):
    raw = {"attribution_type": "speaker_personal_view", "claim_type": "forecast", "is_forward_looking": True,
           "subject_mention": subject, "asset_type": "stock", "stance": "bullish", "forecast_metric": "price",
           "forecast_direction": "increase", "target_kind": "absolute_value", "target_value": 200,
           "currency": "USD", "horizon_original": "next year", "recommendation_action": "buy",
           "condition": "if the Fed cuts in September", "evidence_text": evidence, "extraction_confidence": "high"}
    raw.update(over)
    return raw


def _assert_non_view(c):
    assert cm.non_view_invariant_violations(c) == [], cm.non_view_invariant_violations(c)
    for slot in FORECAST_SLOTS:
        assert c[slot] is None, slot
    assert c["target_kind"] == "none" and c["stance"] == "not_applicable"
    assert c["recommendation_action"] == "none"
    assert c["is_forward_looking"] is False and c["testable"] is False
    assert c["testability_type"] == "not_testable"
    assert not cm.carries_view(c)


def test_interviewer_question_keeps_no_forecast_direction():
    text = "Will Nvidia hit $200 next year if the Fed cuts in September?"
    c = _validate(text, _forecast_like("Will Nvidia hit $200 next year if the Fed cuts in September?"))
    assert c["claim_type"] == "question" and c["attribution_type"] == "interviewer_question"
    _assert_non_view(c)
    # Nothing the model said is lost: it moved out of the forecast slots.
    assert c["displaced_fields"]["forecast_direction"] == "increase"
    assert c["displaced_fields"]["target_value"] == 200 and c["displaced_fields"]["condition"]


def test_portfolio_disclosure_keeps_no_forecast_fields():
    text = "I own Nvidia in my portfolio."
    c = _validate(text, _forecast_like("I own Nvidia in my portfolio", claim_type="stance", forecast_direction=None,
                                       target_kind=None, target_value=None, recommendation_action="none",
                                       condition=None, horizon_original=None))
    assert c["claim_type"] == "portfolio_disclosure" and c["portfolio_disclosure"] == "owns_unspecified"
    _assert_non_view(c)
    assert not (c["displaced_fields"] or {}).get("forecast_direction")


def test_retrospective_claim_moves_its_old_target_out_of_the_forecast_slots():
    text = "Last year I said Nvidia would hit $200, and it did."
    c = _validate(text, _forecast_like("Last year I said Nvidia would hit $200", horizon_original="last year",
                                       condition=None, recommendation_action="none"))
    assert c["claim_type"] == "historical_claim" and c["attribution_type"] == "retrospective_claim"
    _assert_non_view(c)
    assert c["displaced_fields"]["target_value"] == 200 and c["displaced_fields"]["horizon_original"] == "last year"


def test_third_party_report_keeps_the_target_as_reported_not_as_a_forecast():
    text = "Nvidia looks strong. Goldman expects the stock to reach $200 next year if the Fed cuts in September."
    c = _validate(text, _forecast_like("Goldman expects the stock to reach $200 next year if the Fed cuts in September",
                                       subject="the stock", attribution_type="speaker_quoting_third_party",
                                       attributed_person_or_organization="Goldman", claim_type="third_party_view",
                                       recommendation_action="none"))
    assert c["claim_type"] == "third_party_view" and c["host_position"] == "neutral"
    _assert_non_view(c)
    assert c["reported_target_value"] == 200 and c["reported_target_kind"] == "absolute_value"
    assert c["reported_forecast_direction"] == "increase" and c["reported_horizon_bucket"] == "long"
    assert c["reported_condition"] == "if the Fed cuts in September"
    assert c["reported_stance"] == "bullish" and c["displaced_fields"] is None
    assert c["attributed_person_or_organization"] == "Goldman" and c["ticker"] == "NVDA"


def test_two_reported_targets_from_different_houses_are_not_duplicates():
    text = "Goldman expects Nvidia to reach $200. Morgan Stanley expects Nvidia to reach $250."
    raws = [_forecast_like("Goldman expects Nvidia to reach $200", attribution_type="speaker_quoting_third_party",
                           attributed_person_or_organization="Goldman", claim_type="third_party_view",
                           recommendation_action="none", condition=None, horizon_original=None),
            _forecast_like("Morgan Stanley expects Nvidia to reach $250", attribution_type="speaker_quoting_third_party",
                           attributed_person_or_organization="Morgan Stanley", claim_type="third_party_view",
                           target_value=250, recommendation_action="none", condition=None, horizon_original=None)]
    claims, _ = cm.validate_claims(raws, tn.normalize_transcript(text, "v1"), _ctx())
    assert [c["repeat_of_claim_id"] for c in claims] == [None, None]
    assert claims[0]["claim_id"] != claims[1]["claim_id"]
    assert [c["reported_target_value"] for c in claims] == [200, 250]


def test_hypothetical_scenario_is_kept_as_hypothetical_fields():
    text = "Imagine Nvidia at $200 next year; suppose it doubles from here."
    c = _validate(text, _forecast_like("Imagine Nvidia at $200 next year", attribution_type="hypothetical_example",
                                       claim_type="hypothetical", recommendation_action="none", condition=None))
    assert c["claim_type"] == "hypothetical"
    _assert_non_view(c)
    assert c["hypothetical_target_value"] == 200 and c["hypothetical_forecast_direction"] == "increase"
    assert c["hypothetical_horizon_bucket"] == "long"


def test_news_report_and_fact_carry_no_forecast_slots():
    text = "Reports say Micron raised guidance to $9 billion. Nvidia rose 5 percent today."
    news = _validate(text, _forecast_like("Reports say Micron raised guidance to $9 billion", subject="Micron",
                                          attribution_type="speaker_reporting_news", claim_type="news_report",
                                          forecast_metric="revenue", target_value=9e9, recommendation_action="none",
                                          condition=None, horizon_original=None))
    _assert_non_view(news)
    assert news["reported_target_value"] == 9e9 and news["reported_forecast_metric"] == "revenue"
    fact = _validate(text, _forecast_like("Nvidia rose 5 percent today", attribution_type="speaker_reporting_facts",
                                          claim_type="fact", target_kind="percentage_change", target_value=5,
                                          recommendation_action="none", condition=None, horizon_original=None))
    _assert_non_view(fact)


def test_view_claims_keep_their_forecast_fields_and_adopted_views_are_views():
    text = "Nvidia looks strong. Goldman expects the stock to reach $200 and I agree with that."
    own = _validate(text, _forecast_like("Nvidia looks strong", forecast_direction="increase", target_value=None,
                                         target_kind="direction_only", condition=None, recommendation_action="none"))
    assert cm.non_view_invariant_violations(own) == [] and own["forecast_direction"] == "increase"
    adopted = _validate(text, _forecast_like("Goldman expects the stock to reach $200 and I agree with that",
                                             subject="the stock", attribution_type="speaker_quoting_third_party",
                                             attributed_person_or_organization="Goldman",
                                             claim_type="third_party_view", recommendation_action="none",
                                             condition=None, horizon_original=None))
    assert adopted["claim_type"] == "price_target" and adopted["target_value"] == 200
    assert adopted["stance"] == "bullish" and "reported_target_value" not in adopted


def test_invariant_checker_flags_a_violating_record():
    bad = {"claim_type": "question", "forecast_direction": "increase", "target_value": 10, "stance": "bullish",
           "is_forward_looking": True, "testable": True, "testability_type": "unconditional_testable",
           "recommendation_action": "buy"}
    violations = cm.non_view_invariant_violations(bad)
    assert {"forecast_direction", "target_value", "stance", "recommendation_action", "is_forward_looking",
            "testability_type"} <= set(violations)
    assert cm.non_view_invariant_violations({"claim_type": "forecast", "forecast_direction": "increase"}) == []
