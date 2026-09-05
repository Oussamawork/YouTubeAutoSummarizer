"""Item 1: a portfolio disclosure is not a stance and never enters analytics."""
import claims as cm
import canonical_claims as cc
import research_analytics as ra
import transcript_normalize as tn


def _ctx(**over):
    ctx = {"video_id": "v1", "channel_id": "c1", "channel_name": "Chan", "video_title": "T",
           "published_at": "2026-07-01T00:00:00+00:00", "transcript_source": "supadata", "run_key": "rk",
           "extraction_model": "m"}
    ctx.update(over)
    return ctx


def _validate(text, raw, **over):
    return cm.validate_claims(raw, tn.normalize_transcript(text, "v1"), _ctx(**over))[0]


def _stanceable(claim):
    """Every field an analytics path could read a view from."""
    return (claim["claim_type"], claim["stance"], claim["recommendation_action"], claim["is_forward_looking"],
            cm.carries_view(claim))


def test_i_own_tesla_is_a_disclosure_with_no_stance_even_when_the_model_says_neutral():
    raw = {"attribution_type": "speaker_personal_view", "claim_type": "stance", "stance": "neutral",
           "subject_mention": "Tesla", "recommendation_action": "buy", "portfolio_disclosure": "long",
           "evidence_text": "I own Tesla by the way", "is_forward_looking": False}
    c = _validate("Anyway. I own Tesla by the way. Moving on.", [raw])[0]
    assert c["claim_type"] == "portfolio_disclosure"
    assert c["stance"] == "not_applicable" and c["stance_basis"] == "not_applicable"
    assert c["recommendation_action"] == "none"
    assert c["is_forward_looking"] is False and c["testable"] is False
    # "long" was the model's word, not the evidence's: only what the sentence
    # supports is kept.
    assert c["portfolio_disclosure"] == "owns_unspecified"
    assert cm.carries_view(c) is False and c["review_required"] is False


def test_supported_positions_are_kept_and_unsupported_ones_downgraded():
    long_ = _validate("I'm long Bitcoin here.", [{"claim_type": "portfolio_disclosure", "subject_mention": "Bitcoin",
                                                  "asset_type": "crypto", "evidence_text": "I'm long Bitcoin here"}])[0]
    short = _validate("And yes I'm short Tesla.", [{"claim_type": "portfolio_disclosure", "subject_mention": "Tesla",
                                                    "evidence_text": "I'm short Tesla"}])[0]
    none = _validate("I don't own Apple anymore.", [{"claim_type": "portfolio_disclosure", "subject_mention": "Apple",
                                                     "evidence_text": "I don't own Apple anymore"}])[0]
    assert (long_["portfolio_disclosure"], short["portfolio_disclosure"], none["portfolio_disclosure"]) == \
        ("long", "short", "no_position")
    assert {c["stance"] for c in (long_, short, none)} == {"not_applicable"}


def test_ownership_plus_a_forecast_keeps_the_forecast_and_notes_the_disclosure():
    raw = {"attribution_type": "speaker_personal_view", "claim_type": "forecast", "stance": "bullish",
           "subject_mention": "Tesla", "forecast_metric": "price", "forecast_direction": "increase",
           "horizon_original": "next year", "is_forward_looking": True,
           "evidence_text": "I own Tesla and I think it doubles next year"}
    c = _validate("I own Tesla and I think it doubles next year.", [raw])[0]
    assert c["claim_type"] == "forecast" and c["stance"] == "bullish"
    assert c["portfolio_disclosure"] == "owns_unspecified"
    assert cm.carries_view(c) is True


def test_ownership_never_becomes_a_view_in_any_analytics_path():
    text = "Host: could Apple fall 30 percent from here? I own Tesla by the way."
    raw = [{"attribution_type": "speaker_personal_view", "claim_type": "portfolio_disclosure",
            "subject_mention": "Tesla", "portfolio_disclosure": "owns_unspecified", "stance": "neutral",
            "evidence_text": "I own Tesla by the way"}]
    claims = _validate(text, raw)
    c = claims[0]
    assert c["ticker"] == "TSLA"
    # No stance counts, no net stance, no consensus, no sentiment, no flips,
    # no scorecard: every path that could read a view says nothing.
    assert ra.consensus(claims) == {}
    assert ra.find_flips(claims) == []
    assert ra.descriptive([k for k in claims if cm.carries_view(k)])["stances"] == {}
    legacy = cm.claims_to_legacy_signals(claims)
    assert legacy["assets"] == [] and legacy["market_sentiment"] == "neutral"
    assert cc.aggregate_views(claims) == {}
    assert cc.video_tone(claims) == {}
    sc = ra.scorecard(claims, __import__("datetime").date(2027, 1, 1), lambda *a: {})
    assert sc["_excluded"] == {"not_testable": 1}
    # ...but the disclosure IS reported, separately.
    rows = cc.portfolio_disclosures(claims)
    assert rows == [{"source": "Chan", "asset": "TSLA", "ticker": "TSLA", "position": "owns_unspecified",
                     "date": "2026-07-01", "evidence": "I own Tesla by the way", "claim_id": c["claim_id"],
                     "review_required": False}]
    assert "Portfolio disclosures" in cc.format_portfolio_disclosures(rows)


def test_a_disclosure_with_a_directional_stance_field_still_cannot_vote():
    # Even a hand-built record whose stance field says bullish is refused by
    # the view filter on claim_type alone.
    c = {"claim_type": "portfolio_disclosure", "stance": "bullish", "review_required": False,
         "coverage_status": "full", "evidence_start_character": 1, "attribution_type": "speaker_personal_view",
         "schema_version": "2"}
    assert cm.is_headline_claim(c) and not cm.carries_view(c)
    assert ra.latest_views([dict(c, ticker="TSLA", claim_id="x")]) == {}
