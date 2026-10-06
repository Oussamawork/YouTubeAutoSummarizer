"""Review flags that kept valid creator views out of the weekly pulse (Oct 2026
audit: 501 of 1,183 own-view claims since Sep 7 were review-flagged, most of
them by the four rules below)."""
import canonical_claims as cc
import claims as cm
import research_state

from tests.test_claims import _claim, _validate


def test_evidence_elided_with_an_ellipsis_is_located_piece_by_piece():
    text = ("This company is selling at $179, and I calculated a fair value of $244. That gives it "
            "a nice margin of safety. This is another one of the companies I do not own yet, but I am "
            "interested in owning.")
    ev = "This company is selling at $179, and I calculated a fair value of $244... This is another one of the companies"
    span = cm.locate_evidence(ev, text)
    assert span == (0, text.index("of the companies") + len("of the companies"))
    assert cm.locate_evidence("This company is selling at $179 … interested in owning", text) is not None


def test_ellipsis_pieces_must_be_verbatim_in_order_and_close():
    text = "Alpha beta gamma delta epsilon. " + "filler words here. " * 200 + "Zeta eta theta iota kappa."
    assert cm.locate_evidence("Alpha beta gamma delta... zeta eta theta iota", text) is None    # too far apart
    assert cm.locate_evidence("Zeta eta theta iota... alpha beta gamma delta", text) is None    # out of order
    assert cm.locate_evidence("Alpha beta gamma delta... omega psi chi", "Alpha beta gamma delta and more") is None


def test_validated_claim_with_elided_evidence_is_not_sent_to_review():
    text = "I like Chevron a lot here. Their cash flow is great. I expect Chevron to rise this year."
    claims, _ = _validate(text, [_claim(subject_mention="Chevron", stance="bullish", forecast_direction="increase",
                                        evidence_text="I like Chevron a lot here... I expect Chevron to rise")])
    assert "evidence_not_found" not in claims[0]["review_reasons"]
    assert claims[0]["evidence_start_character"] == 0


def test_german_number_notation_supports_a_target():
    assert 5.84 in cm.numbers_in("Kursziel bei 5,84 US-Dollar")
    assert 18000.0 in cm.numbers_in("bis 18.000 Punkte")
    assert 1234.56 in cm.numbers_in("1.234,56 Euro")
    nums = cm.numbers_in("1,234 dollars and 3.12")
    assert 1234.0 in nums and 3.12 in nums and 1.234 not in nums


def test_according_to_my_own_estimate_is_not_a_third_party_view():
    text = "According to my estimates, Visa stock is worth $377 today."
    claims, _ = _validate(text, [_claim(subject_mention="Visa stock", stance="bullish", claim_type="valuation_view",
                                        target_kind="absolute_value", target_value=377, currency="USD",
                                        evidence_text="According to my estimates, Visa stock is worth $377 today")])
    assert "possible_third_party_view" not in claims[0]["review_reasons"]
    claims, _ = _validate("According to analysts, Visa is worth $400.", [_claim(
        subject_mention="Visa", stance="bullish", evidence_text="According to analysts, Visa is worth $400")])
    assert "possible_third_party_view" in claims[0]["review_reasons"]


def _stored(**over):
    base = {"claim_id": "c1", "video_id": "v1", "review_required": True, "evidence_text": "e",
            "subject_mention": "Chevron", "canonical_entity_name": "Chevron", "ticker": None,
            "asset_type": "stock"}
    base.update(over)
    return base


def test_stored_flags_the_current_rules_no_longer_raise_are_cleared():
    cleared = cm.revalidate_review(_stored(review_reasons=["ticker_not_in_evidence"]))
    assert cleared["review_required"] is False and cleared["review_reasons"] == []
    assert cleared["review_cleared"] == ["ticker_not_in_evidence"] and cleared["ticker"] == "CVX"
    german = cm.revalidate_review(_stored(review_reasons=["number_not_in_evidence:target_value"],
                                          target_value=5.84, evidence_text="Ziel 5,84 Dollar"))
    assert german["review_required"] is False


def test_revalidation_never_clears_what_the_rules_still_raise():
    unresolved = _stored(subject_mention="Iron", canonical_entity_name="Iron", review_reasons=["ticker_not_in_evidence"])
    assert cm.revalidate_review(unresolved) is unresolved
    partly = cm.revalidate_review(_stored(review_reasons=["ticker_not_in_evidence", "evidence_not_found"]))
    assert partly["review_required"] is True and partly["review_reasons"] == ["evidence_not_found"]
    wrong_number = _stored(review_reasons=["number_not_in_evidence:target_value"], target_value=120.0,
                           evidence_text="could reach $100")
    assert cm.revalidate_review(wrong_number)["review_required"] is True


def test_canonical_loader_applies_the_revalidation(tmp_path, monkeypatch):
    from tests.test_canonical_analytics import _claim as stored_claim
    monkeypatch.setattr(research_state, "RESEARCH_DIR", str(tmp_path))
    state = research_state.load_state()
    row = stored_claim("A", None, "bullish", "unspecified", subject_mention="Chevron",
                       canonical_entity_name="Chevron", review_required=True,
                       review_reasons=["ticker_not_in_evidence"])
    research_state.store_claims([row], state, row["video_id"], "rk1")
    loaded = cc.load_canonical_claims(state)
    assert loaded[0]["review_required"] is False and cc.view_claims(loaded)
    assert cc.asset_key(loaded[0]) == "CVX"
