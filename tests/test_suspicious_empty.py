"""Item 7: an empty claims array is only believed when the transcript is plausibly claim-free."""
import json

import claims as cm
import signals
import transcript_normalize as tn


def _nt(text):
    return tn.normalize_transcript(text, "v1")


def _ctx(text):
    return {"video_id": "v1", "channel_id": "c1", "channel_name": "Chan", "video_title": "T",
            "published_at": "2026-07-01T00:00:00+00:00", "normalized": _nt(text)}


EMPTY = {"claims": [], "extraction_metadata": {"warnings": []}}


def test_actual_no_claim_transcript_is_no_claims_found():
    text = ("Welcome back everyone. Today I want to show you how to set up a watchlist in the app and how "
            "a limit order works. Click the plus button, type the name, and save. Thanks for watching.")
    assert cm.suspicious_empty_check(text)["suspicious"] is False
    res = signals.build_research(EMPTY, _ctx(text))
    assert res["status"] == "no_claims_found" and res["failure_reason"] is None
    assert res["signals"] == {"assets": [], "market_sentiment": "neutral", "topics": [], "derived_from": "claims"}


def test_empty_result_for_a_clear_price_forecast_is_suspicious():
    text = "Let me be clear. I think Nvidia will hit $200 by the end of the year. That is my call."
    check = cm.suspicious_empty_check(text)
    assert check["suspicious"] and check["strong"]
    res = signals.build_research(EMPTY, _ctx(text))
    assert res["status"] == "failed_retryable" and res["failure_reason"] == "suspicious_empty_extraction"
    assert res["signals"] is None and res["claims"] == []
    assert any(w.startswith("suspicious_empty_extraction") for w in res["warnings"])
    # From a dedicated claims call the same result goes to a human instead.
    res = signals.build_research(EMPTY, _ctx(text), standalone=True)
    assert res["status"] == "needs_review" and res["failure_reason"] == "suspicious_empty_extraction"


def test_empty_result_for_a_buy_recommendation_is_suspicious():
    text = "Honestly, I'm buying Palantir here. This is a buy for me and I'd buy more under 20."
    assert cm.suspicious_empty_check(text)["suspicious"]
    assert signals.build_research(EMPTY, _ctx(text))["status"] == "failed_retryable"


def test_educational_transcript_with_percentages_is_not_a_false_alert():
    text = ("Let me explain what a P/E ratio is. A P/E of 20 means that you pay 20 dollars for every dollar of "
            "earnings. Historically the S&P 500 has returned about 10 percent per year, and in 2022 it fell about "
            "19 percent. For example, imagine Apple earning 5 dollars per share at 100 dollars: that is a P/E of 20.")
    check = cm.suspicious_empty_check(text)
    assert check["suspicious"] is False, check
    assert signals.build_research(EMPTY, _ctx(text))["status"] == "no_claims_found"


def test_news_report_with_a_quoted_third_party_target_is_suspicious():
    text = ("In analyst moves today, Morgan Stanley raised its price target on Micron to $150 from $120, "
            "citing memory pricing. Micron shares rose 3 percent on the news.")
    check = cm.suspicious_empty_check(text)
    assert check["suspicious"] and any("price target" in s for s in check["strong"])
    assert signals.build_research(EMPTY, _ctx(text))["failure_reason"] == "suspicious_empty_extraction"


def test_questions_and_all_caps_transcripts_do_not_trigger():
    assert cm.suspicious_empty_check("Will Nvidia hit $200 next year? Could Apple fall 30 percent?")["suspicious"] is False
    caps = "THE BOARD WILL MEET NEXT YEAR AND THE CEO WILL SPEAK ABOUT THE PLAN AND THE TEAM"
    assert cm.suspicious_empty_check(caps)["suspicious"] is False


def test_chunked_extraction_applies_the_same_check(monkeypatch, tmp_path):
    import summarizer
    monkeypatch.setattr(summarizer, "PARTIALS_DIR", str(tmp_path))
    monkeypatch.setattr(signals, "_research_chunk_tokens", lambda: 40000)
    monkeypatch.setattr(signals, "complete", lambda *a, **k: json.dumps(EMPTY))
    text = "I think Nvidia will hit $200 by the end of the year. That is my call."
    res = signals._extract_research_chunked(_nt(text), {"video_id": "v1"})
    assert res["status"] == "needs_review" and res["failure_reason"] == "suspicious_empty_extraction"
    assert res["signals"] is None
