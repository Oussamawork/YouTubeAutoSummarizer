"""Item 6: third-party attribution positions and local transcript coreference."""
import claims as cm
import transcript_normalize as tn


def _ctx(title="T"):
    return {"video_id": "v1", "channel_id": "c1", "channel_name": "Chan", "video_title": title,
            "published_at": "2026-07-01T00:00:00+00:00", "transcript_source": "supadata", "run_key": "rk"}


def _third_party(text, evidence, subject="the stock", **over):
    raw = {"attribution_type": "speaker_quoting_third_party", "attributed_person_or_organization": "Goldman",
           "claim_type": "third_party_view", "is_forward_looking": True, "subject_mention": subject,
           "asset_type": "stock", "target_kind": "absolute_value", "target_value": 200, "currency": "USD",
           "stance": "not_applicable", "evidence_text": evidence, "extraction_confidence": "high"}
    raw.update(over)
    return cm.validate_claims([raw], tn.normalize_transcript(text, "v1"), _ctx())[0][0]


def test_one_clear_active_subject_resolves_the_stock_locally():
    text = "Nvidia looks expensive. Goldman expects the stock to reach $200."
    c = _third_party(text, "Goldman expects the stock to reach $200")
    assert c["subject_mention"] == "the stock"
    assert c["canonical_entity_name"] == "Nvidia" and c["ticker"] == "NVDA"
    assert c["entity_resolution_method"] == "local_coreference"
    assert c["entity_resolution_confidence"] == 0.7
    assert c["entity_resolution_status"] == "confirmed" and c["review_required"] is False


def test_two_possible_active_subjects_stay_unresolved():
    text = "Nvidia looks expensive and AMD looks cheap. Goldman expects the stock to reach $200."
    c = _third_party(text, "Goldman expects the stock to reach $200")
    assert c["ticker"] is None and c["entity_resolution_method"] == "unresolved"
    assert c["entity_resolution_status"] == "ambiguous" and c["entity_resolution_confidence"] == 0.0
    assert "ambiguous_coreference" in c["review_reasons"] and c["review_required"] is True


def test_coreference_looks_only_at_the_immediately_preceding_sentences():
    # Nvidia was the subject three sentences ago and nothing since names an
    # asset: too far back to be the active subject.
    text = ("Nvidia looks expensive. Rates matter here. Inflation is sticky too. "
            "The market is nervous. Goldman expects the stock to reach $200.")
    c = _third_party(text, "Goldman expects the stock to reach $200")
    assert c["ticker"] is None and c["entity_resolution_method"] == "unresolved"
    assert c["entity_resolution_status"] == "unresolved"


def test_third_party_target_explicitly_rejected_by_the_host():
    text = "Nvidia looks expensive. Goldman expects the stock to reach $200 but that is their call not mine."
    c = _third_party(text, "Goldman expects the stock to reach $200 but that is their call not mine")
    assert c["attribution_type"] == "speaker_quoting_third_party" and c["claim_type"] == "third_party_view"
    assert c["host_position"] == "rejected" and c["stance"] == "not_applicable"
    assert c["review_required"] is False and not cm.is_headline_claim(c)


def test_third_party_target_explicitly_adopted_by_the_host_becomes_their_own_view():
    text = "Nvidia looks expensive. Goldman expects the stock to reach $200 and I agree with that."
    c = _third_party(text, "Goldman expects the stock to reach $200 and I agree with that")
    assert c["host_position"] == "adopted"
    assert c["attribution_type"] == "speaker_personal_view" and c["claim_type"] == "price_target"
    assert c["third_party_origin"] == "Goldman" and c["target_value"] == 200
    assert c["ticker"] == "NVDA" and cm.is_headline_claim(c)


def test_third_party_target_reported_neutrally_needs_no_review_and_carries_no_view():
    text = "Nvidia looks expensive. Goldman expects the stock to reach $200."
    c = _third_party(text, "Goldman expects the stock to reach $200")
    assert c["host_position"] == "neutral" and c["review_required"] is False
    assert c["claim_type"] == "third_party_view" and c["stance"] == "not_applicable"
    assert not cm.is_headline_claim(c) and not cm.carries_view(c)
    # The model cannot promote a view on its own say-so: adoption must be in
    # the evidence.
    c2 = _third_party(text, "Goldman expects the stock to reach $200", host_position="adopted")
    assert c2["host_position"] == "neutral" and c2["attribution_type"] == "speaker_quoting_third_party"


def test_own_view_that_reads_like_a_quotation_still_goes_to_review():
    text = "Analysts expect Nvidia to reach $200."
    raw = {"attribution_type": "speaker_personal_view", "claim_type": "price_target", "is_forward_looking": True,
           "subject_mention": "Nvidia", "target_kind": "absolute_value", "target_value": 200,
           "evidence_text": "Analysts expect Nvidia to reach $200"}
    c = cm.validate_claims([raw], tn.normalize_transcript(text, "v1"), _ctx())[0][0]
    assert "possible_third_party_view" in c["review_reasons"] and c["review_required"]


def test_explicit_mention_and_title_resolution_methods_are_recorded():
    text = "I expect Nvidia to fall over the next three months."
    raw = {"attribution_type": "speaker_personal_view", "claim_type": "forecast", "subject_mention": "Nvidia",
           "forecast_direction": "decrease", "horizon_original": "over the next three months",
           "evidence_text": "I expect Nvidia to fall over the next three months"}
    c = cm.validate_claims([raw], tn.normalize_transcript(text, "v1"), _ctx())[0][0]
    assert c["entity_resolution_method"] == "explicit_mention" and c["entity_resolution_confidence"] == 1.0
    text2 = "It should fall over the next three months, that is my base case."
    raw2 = dict(raw, evidence_text="It should fall over the next three months")
    c2 = cm.validate_claims([raw2], tn.normalize_transcript(text2, "v1"), _ctx(title="Nvidia earnings preview"))[0][0]
    assert c2["entity_resolution_method"] == "title" and c2["ticker"] == "NVDA"
