"""
Validator behaviours that mislabelled correct claims, surfaced by the manual
extraction of 63 stored transcripts (data/research/candidates/, imported by
research_backfill.py --import-candidates). Every case uses the phrasing it
was found with; each one affected production Gemini extractions too.
"""
import claims as cm
import canonical_claims as cc
import transcript_normalize as tn


def _ctx(**over):
    ctx = {"video_id": "v1", "channel_id": "c1", "channel_name": "Chan", "video_title": "T",
           "published_at": "2026-07-01T00:00:00+00:00", "transcript_source": "supadata", "run_key": "rk",
           "extraction_model": "gemini-3.7-flash", "transcript_language": "en"}
    ctx.update(over)
    return ctx


def _validate(text, raws, **over):
    return cm.validate_claims(raws, tn.normalize_transcript(text, "v1"), _ctx(**over))[0]


def _disclosure(evidence, subject, position, **over):
    raw = {"speaker": "host", "attribution_type": "speaker_personal_view", "claim_type": "portfolio_disclosure",
           "subject_mention": subject, "portfolio_disclosure": position, "stance": "not_applicable",
           "recommendation_action": "none", "is_forward_looking": False, "evidence_text": evidence,
           "extraction_confidence": "high"}
    raw.update(over)
    return raw


# --- 1. A no-position disclosure is never turned into ownership --------------

MCDONALDS = ("I think all of this makes McDonald's stock a buying opportunity. It's the only one on this list "
             "that I don't already own in my portfolio, but I am interested in adding McDonald's stock here.")
MCDONALDS_EVIDENCE = ("It's the only one on this list that I don't already own in my portfolio, but I am "
                      "interested in adding McDonald's stock here.")


def test_negation_with_an_adverb_binds_to_the_holding_verb_not_to_the_portfolio_words():
    # Was owns_unspecified: "in my portfolio" matched the ownership cue and
    # "I don't already own" missed the exact "I don't own" wording.
    c = _validate(MCDONALDS, [_disclosure(MCDONALDS_EVIDENCE, "McDonald's", "no_position")])[0]
    assert c["claim_type"] == "portfolio_disclosure" and c["portfolio_disclosure"] == "no_position"
    assert c["review_required"] is False
    assert cc.portfolio_disclosures([c]) == []   # never "what creators say they own"


def test_i_dont_have_it_is_no_position():
    text = "Intel, I picked it up around 80s recently. Um, AMD, I don't have it, but AMD is up big. I have Nvidia."
    c = _validate(text, [_disclosure("AMD, I don't have it, but AMD is up big.", "AMD", "no_position",
                                     speaker="Annie Sen", attribution_type="guest_personal_view")])[0]
    assert c["portfolio_disclosure"] == "no_position" and c["ticker"] == "AMD"
    assert cc.portfolio_disclosures([c]) == []


def test_negated_portfolio_words_correct_a_model_that_read_them_as_ownership():
    # Couch Investor, 2026-10-21 (Gemini): stored as owns_unspecified and
    # reported as "owns SPCX".
    text = "And this is why right now this is not in my portfolio. Despite me really liking the business."
    c = _validate(text, [_disclosure("And this is why right now this is not in my portfolio.", "SpaceX",
                                     "owns_unspecified")])[0]
    assert c["portfolio_disclosure"] == "no_position"


def test_a_no_position_the_rules_cannot_read_is_kept_not_flipped_to_ownership():
    # No English holding cue at all: the model's no_position used to become
    # not_stated, and a disclosure without a position defaulted to ownership.
    c = _validate("I've stayed away from Palantir entirely.",
                  [_disclosure("I've stayed away from Palantir entirely.", "Palantir", "no_position")])[0]
    assert c["portfolio_disclosure"] == "no_position" and c["review_required"] is False
    # HKCM (offline set): stored as owns_unspecified by the old flip.
    wti = _validate("Also WTI sind wir ja raus aus dem Trade.",
                    [_disclosure("Also WTI sind wir ja raus aus dem Trade.", "WTI", "no_position",
                                 asset_type="commodity")], transcript_language="de")[0]
    assert wti["portfolio_disclosure"] == "no_position"


def test_evidence_that_reads_as_ownership_against_no_position_is_reviewed_not_flipped():
    c = _validate("I own a little Palantir, actually.",
                  [_disclosure("I own a little Palantir, actually.", "Palantir", "no_position")])[0]
    assert c["portfolio_disclosure"] == "no_position"
    assert "portfolio_position_conflict" in c["review_reasons"] and c["review_required"] is True


def test_a_negation_never_reaches_across_but_and_the_subject_picks_its_clause():
    text = "I own Nvidia, but AMD, I don't have it."
    nvda, amd = _validate(text, [_disclosure(text, "Nvidia", "owns_unspecified"),
                                 _disclosure(text, "AMD", "no_position")])
    assert (nvda["portfolio_disclosure"], amd["portfolio_disclosure"]) == ("owns_unspecified", "no_position")
    assert [r["ticker"] for r in cc.portfolio_disclosures([nvda, amd])] == ["NVDA"]


def test_position_reading_is_first_person_and_clause_bound():
    assert cm.portfolio_position(MCDONALDS_EVIDENCE, "McDonald's") == "no_position"
    assert cm.portfolio_position("Nvidia isn't in my portfolio yet") == "no_position"
    assert cm.portfolio_position("I've never owned Netflix") == "no_position"
    assert cm.portfolio_position("we're not invested in SpaceX") == "no_position"
    # The listener's position is not a disclosure; owning some is still owning.
    assert cm.portfolio_position("If you don't already own Uber, then I think it's a great buy") is None
    assert cm.portfolio_position("I don't own enough of it") == "owns_unspecified"
    assert cm.portfolio_position("I don't hold out much hope for Intel") is None
    assert cm.portfolio_position("I don't have a strong opinion on Tesla") is None
    assert cm.portfolio_position("I don't think it's overvalued, I own it") == "owns_unspecified"


def test_the_report_drops_a_stored_disclosure_whose_evidence_says_no_position():
    # Stored claims keep their extraction's position; the report re-reads
    # the evidence, so the stored "owns SPCX" row is gone without a re-run.
    row = {"claim_type": "portfolio_disclosure", "portfolio_disclosure": "owns_unspecified", "channel_name": "Couch",
           "subject_mention": "SpaceX", "ticker": "SPCX", "schema_version": "2", "claim_id": "x",
           "published_at": "2026-10-21T00:00:00+00:00",
           "evidence_text": "And this is why right now this is not in my portfolio. Despite me really liking the "
                            "business, the numbers to me are getting so big."}
    assert cc.portfolio_disclosures([row]) == []
    held = dict(row, evidence_text="Still have half of my position left for Netflix.", subject_mention="Netflix",
                ticker="NFLX")
    assert [r["ticker"] for r in cc.portfolio_disclosures([held])] == ["NFLX"]


# --- 2. "last year" as a comparison base is not a retrospective ---------------

GUIDANCE = ("In the other parts of its business, the company expects the rating side of the business to grow "
            "in the low-to-mid single digits despite difficult comps from the same time last year.")


def test_guidance_measured_against_last_year_stays_a_reported_forecast():
    raw = {"speaker": "host", "attribution_type": "speaker_quoting_third_party", "claim_type": "third_party_view",
           "attributed_person_or_organization": "the company", "host_position": "neutral",
           "is_forward_looking": True, "subject_mention": "the rating side of the business",
           "forecast_metric": "revenue", "forecast_direction": "increase", "evidence_text": GUIDANCE}
    c = _validate(GUIDANCE, [raw])[0]
    assert c["attribution_type"] == "speaker_quoting_third_party"   # was retrospective_claim
    assert c["claim_type"] == "third_party_view"
    assert (c["reported_forecast_metric"], c["reported_forecast_direction"]) == ("revenue", "increase")


def test_the_speakers_forecast_against_last_years_comps_stays_a_forecast():
    text = ("For S&P Global, I expect the ratings business to grow in the low-to-mid single digits next year "
            "despite difficult comps from the same time last year.")
    raw = {"speaker": "host", "attribution_type": "speaker_personal_view", "claim_type": "forecast",
           "is_forward_looking": True, "subject_mention": "S&P Global", "stance": "bullish",
           "forecast_metric": "revenue", "forecast_direction": "increase", "horizon_original": "next year",
           "evidence_text": text}
    c = _validate(text, [raw])[0]
    assert c["claim_type"] == "forecast" and c["attribution_type"] == "speaker_personal_view"
    assert c["is_forward_looking"] is True and c["forecast_end_date"] == "2027-12-31"


def test_a_claim_about_the_past_or_the_speakers_own_call_is_still_retrospective():
    past = _validate("Bitcoin doubled last year.", [{
        "attribution_type": "speaker_personal_view", "claim_type": "forecast", "is_forward_looking": True,
        "subject_mention": "Bitcoin", "stance": "bullish", "forecast_direction": "increase",
        "evidence_text": "Bitcoin doubled last year."}])[0]
    assert past["claim_type"] == "historical_claim" and past["attribution_type"] == "retrospective_claim"
    call = _validate("Last year I said Bitcoin would double, and it did.", [{
        "attribution_type": "speaker_personal_view", "claim_type": "forecast", "is_forward_looking": True,
        "subject_mention": "Bitcoin", "stance": "bullish", "forecast_direction": "increase",
        "evidence_text": "Last year I said Bitcoin would double"}])[0]
    assert call["claim_type"] == "historical_claim" and call["is_forward_looking"] is False


def test_only_past_time_references_count_and_forward_wording_wins():
    from datetime import date
    published = date(2026, 10, 1)

    def retro(evidence, horizon_end=None):
        return cm._is_retrospective(evidence, "speaker_personal_view", "forecast", horizon_end, published)

    assert retro("Net sales were up 5.3% compared to the same quarter last year.")
    assert retro("Back in 2021 the stock traded at 300.")
    assert retro("In 2021 I bought Tesla at 200.")
    assert not retro("But starting last year, foreign stocks are smoking U.S. stocks again. And so I think "
                     "we're going to be in another extended period of outperformance.")
    assert not retro("I do still think that this name is undervalued. Yes, I have initiated a position last week.")
    assert not retro("I would buy all three of these and look back in three years.")
    assert not retro("It shows Bitcoin is back in the game.")
    assert not retro("In 2027 I expect Nvidia at 300.")
    # No forward word, but a stated horizon that ends after publication.
    assert not retro("Q4 revenue up 10% to 15% despite tough comps from last year", date(2026, 12, 31))


# --- 3. Multiples ---------------------------------------------------------------

def test_multiples_are_read_whole():
    evidence = "So, from current levels, this would mean an easy 20X, probably 25, potentially even 30X."
    for value in (20, 25, 30):
        assert cm.number_supported(value, evidence)
    assert not cm.number_supported(2, evidence) and not cm.number_supported(3, evidence)   # was 2 and 3
    assert cm.number_supported(10, "I still think the median has to 10x at a minimum")
    assert not cm.number_supported(1, "I still think the median has to 10x at a minimum")
    assert cm.number_supported(4.8, "a 4.8x increase in total token throughput")
    assert cm.number_supported(12, "revenue increased by 12fold year-over-year")


def test_a_20_to_30x_range_passes_the_number_check():
    text = ("And if you know anything about Elliott Wave Theory, wave threes are the largest, most impulsive ones. "
            "So, from current levels, this would mean an easy 20X, probably 25, potentially even 30X.")
    raw = {"speaker": "Joe", "attribution_type": "speaker_personal_view", "claim_type": "forecast",
           "is_forward_looking": True, "subject_mention": "Crow", "stance": "bullish", "forecast_metric": "price",
           "forecast_direction": "increase", "target_kind": "multiple", "target_low": 20, "target_high": 30,
           "target_unit": "x",
           "evidence_text": "So, from current levels, this would mean an easy 20X, probably 25, potentially even 30X."}
    c = _validate(text, [raw])[0]
    assert not [r for r in c["review_reasons"] if r.startswith("number_not_in_evidence")]


# --- 4. Spelled-out numbers, German decimals, stray spaces ----------------------

JENSEN = ("And Jensen believes that AI infrastructure spending will reach three to 4 trillion annually by the "
          "end of the decade.")


def test_a_spelled_out_lower_bound_takes_the_ranges_scale():
    raw = {"speaker": "host", "attribution_type": "speaker_quoting_third_party", "claim_type": "third_party_view",
           "attributed_person_or_organization": "Jensen", "host_position": "neutral", "is_forward_looking": True,
           "subject_mention": "AI infrastructure spending", "asset_type": "macro", "target_kind": "range",
           "target_low": 3e12, "target_high": 4e12, "currency": "USD", "evidence_text": JENSEN}
    c = _validate(JENSEN, [raw])[0]
    assert "number_not_in_evidence:target_low" not in c["review_reasons"]
    assert "number_not_in_evidence:target_high" not in c["review_reasons"]
    assert cm.number_supported(4, "a four or five percent rate of return")
    assert cm.number_supported(5e11, "over half a trillion dollars in spending commitments")
    assert cm.number_supported(1e6, "Bitcoin could be on its way to a million dollars")
    assert cm.number_supported(250000, "two hundred and fifty thousand")


def test_german_decimal_commas_are_read_only_in_a_german_transcript():
    text = "und wir würden hier ohne Probleme immer noch den Bereich zwischen 386,48 und 339,69 anlaufen."
    raw = {"speaker": "guest", "attribution_type": "guest_personal_view", "claim_type": "forecast",
           "is_forward_looking": True, "subject_mention": "Gold", "asset_type": "commodity",
           "forecast_metric": "price", "target_kind": "range", "target_low": 339.69, "target_high": 386.48,
           "evidence_text": text}
    de = _validate(text, [raw], transcript_language="de")[0]
    assert not [r for r in de["review_reasons"] if r.startswith("number_not_in_evidence")]
    en = _validate(text, [raw], transcript_language="en")[0]
    assert "number_not_in_evidence:target_low" in en["review_reasons"]   # "1,690" is 1690 in English
    for value, evidence in ((1.69, "in Richtung 1,69"), (0.05, "in Richtung 0,05 Punkte hochlaufen"),
                            (4200, "in roundabout Richtung 4.200 $"), (100000, "die 100.000 US-Dollar"),
                            (60e6, "etwa 60 Millionen US-Dollar")):
        assert cm.number_supported(value, evidence, "de")
        assert not cm.number_supported(value, evidence, "en")


def test_a_space_before_the_thousands_comma_is_still_one_number():
    text = "if the price gets above the yellow line at 87 ,250 the door is open for 88 ,600 and 92 ,200."
    raw = {"speaker": "host", "attribution_type": "speaker_personal_view", "claim_type": "price_target",
           "is_forward_looking": True, "subject_mention": "Bitcoin", "asset_type": "crypto", "stance": "bullish",
           "forecast_metric": "price", "forecast_direction": "increase", "target_kind": "absolute_value",
           "target_value": 88600, "condition": "if the price gets above the yellow line at 87,250",
           "evidence_text": text}
    c = _validate(text, [raw])[0]
    assert "number_not_in_evidence:target_value" not in c["review_reasons"]
    assert cm.number_supported(92200, text) and cm.number_supported(87250, text)
    assert cm.number_supported(500, "Memory prices are up 3, 4, 500% compared")   # a list stays a list


# --- 5. Each guest's view is that guest's claim ----------------------------------

PANEL = ("I think Nvidia is still the best AI stock to own. >> I'd say Nvidia is the best AI stock to own too. "
         ">> And again, Nvidia is the best AI stock to own, I keep saying it.")


def _view(speaker, evidence):
    return {"speaker": speaker, "attribution_type": "guest_personal_view", "claim_type": "stance",
            "subject_mention": "Nvidia", "stance": "bullish", "evidence_text": evidence}


def test_two_guests_with_the_same_view_stay_two_claims_and_one_guest_repeating_folds():
    jim, steph, jim_again = _validate(PANEL, [
        _view("Jim", "I think Nvidia is still the best AI stock to own."),
        _view("Steph", "I'd say Nvidia is the best AI stock to own too."),
        _view("Jim", "And again, Nvidia is the best AI stock to own, I keep saying it."),
    ])
    assert jim["repeat_of_claim_id"] is None and steph["repeat_of_claim_id"] is None   # was a repeat of Jim's
    assert jim_again["repeat_of_claim_id"] == jim["claim_id"]
    assert [c["speaker"] for c in (jim, steph) if cm.carries_view(c)] == ["Jim", "Steph"]


def test_speaker_labels_compare_folded():
    a = {"speaker": "Parkev Tatevosian, CFA"}
    assert cm._speaker_key(a) == cm._speaker_key({"speaker": "parkev tatevosian cfa"})
    assert cm._speaker_key({"speaker": None}) == cm._speaker_key({}) == "unknown"


def test_overlapping_chunks_fold_one_excerpt_whatever_each_chunk_called_the_speaker():
    # A guest introduced by name in chunk 1 is "guest" in chunk 2, which never
    # saw the introduction: the same excerpt read twice is still one claim.
    text = "Dan Ives joins us. Nvidia is the best AI stock to own. >> Nvidia is the best AI stock to own too."
    nt = tn.normalize_transcript(text, "v1")
    first, _ = cm.validate_claims([_view("Dan Ives", "Nvidia is the best AI stock to own.")], nt, _ctx(),
                                  "chunked_full", "c1")
    second, _ = cm.validate_claims([_view("guest", "Nvidia is the best AI stock to own."),
                                    _view("Steph", "Nvidia is the best AI stock to own too.")], nt, _ctx(),
                                   "chunked_full", "c2")
    merged = cm.dedupe_across_chunks(first + second)
    primaries = [c["speaker"] for c in merged if not c["repeat_of_claim_id"]]
    assert sorted(primaries) == ["Dan Ives", "Steph"]


# --- 6. A dollar amount that is not a price is not a price level -----------------

BUYBACK = ("if you see something along the lines of a two-300 billion dollar buyback announced here over the "
           "next couple of months")


def test_a_buyback_size_over_the_next_months_is_not_a_price_level():
    assert cm.condition_observability(BUYBACK) == (False, "subjective", None)
    text = ("I think that will help rerating the stock. um if you see something along the lines of a two-300 "
            "billion dollar buyback announced here over the next couple of months um they will be allowed to um "
            "start buying back aggressively uh starting in December.")
    raw = {"speaker": "Angelo", "attribution_type": "guest_personal_view", "claim_type": "forecast",
           "is_forward_looking": True, "subject_mention": "Micron", "stance": "bullish", "forecast_metric": "price",
           "forecast_direction": "increase", "horizon_original": "over the next couple of months",
           "condition": BUYBACK, "evidence_text": text}
    c = _validate("Micron: " + text, [raw])[0]
    assert c["condition_observable"] is False and c["condition_kind"] == "subjective"
    assert "condition_not_objectively_observable" in c["testability_issues"]
    assert c["testability_type"] == "not_testable"


def test_amounts_and_time_are_never_levels_but_prices_still_are():
    for condition in ("if Nebus in 2029 is generating over $50 billion in revenue",
                      "if free cash flow reaches negative $50 billion for a trailing 12-month period",
                      "if Nvidia hits a $5 trillion market cap",
                      "if they buy back 5% of shares over the next year"):
        assert cm.condition_observability(condition)[1] != "price_level", condition
    for condition in ("if the price gets above the yellow line at 87,250", "if it drops under $600",
                      "if Bitcoin breaks $100k over the next couple of months", "if it holds the 50-day",
                      "if Nvidia trades above $200 in 2027", "under $20"):
        assert cm.condition_observability(condition)[1] == "price_level", condition
    assert cm.condition_observability("if revenue tops $50 billion")[1] == "earnings_release"
