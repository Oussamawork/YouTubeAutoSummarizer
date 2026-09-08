"""Hardening item 6: the suspicious-empty guard is language-aware. English
and German (the languages in channel_ids.txt) have rule sets; any other
language never gets an empty extraction certified as no_claims_found."""
import json

import claims as cm
import language_detect
import signals
import summarizer
import transcript_normalize as tn

EMPTY = {"claims": [], "extraction_metadata": {"warnings": []}}

EN_CLAIM = "Let me be clear. I think Nvidia will hit $200 by the end of the year. That is my call."
DE_CLAIM = ("Ich sage es ganz klar. Ich erwarte, dass Nvidia bis Jahresende 200 Dollar erreicht. "
            "Das ist mein Kursziel und ich bleibe hier investiert.")
DE_BUY = ("Ich habe heute Palantir nachgekauft, weil die Aktie für mich hier ein klarer Kauf ist. "
          "Wir sind bei dem Unternehmen noch nicht am Ende und ich werde die Position weiter aufstocken.")
DE_EDU = ("Lasst uns kurz erklären, was ein KGV ist. Ein KGV von 20 bedeutet, dass man 20 Euro für jeden "
          "Euro Gewinn zahlt. Historisch hat der DAX im Schnitt etwa 8 Prozent pro Jahr gebracht, und im Jahr "
          "2022 waren es minus 12 Prozent. Zum Beispiel: Apple mit 5 Dollar Gewinn bei 100 Dollar Kurs, das ist "
          "ein KGV von 20. Das war die Erklärung für heute.")
DE_QUESTIONS = "Wird Nvidia nächstes Jahr 200 Dollar erreichen? Könnte die Aktie um 30 Prozent fallen?"
FR_CLAIM = ("Je pense que Nvidia va atteindre 200 dollars avant la fin de l'année. C'est mon objectif et "
            "je reste investi dans cette entreprise pour les prochaines années.")
FR_EMPTY = ("Bonjour à tous et bienvenue dans cette nouvelle vidéo. Aujourd'hui nous allons parler de la "
            "manière dont je prépare mes journées et de quelques habitudes qui aident à rester concentré.")


def _ctx(text, language=None):
    return {"video_id": "v1", "channel_id": "c1", "channel_name": "Chan", "video_title": "T",
            "published_at": "2026-07-01T00:00:00+00:00", "normalized": tn.normalize_transcript(text, "v1"),
            "transcript_language": language}


# --- detection ---

def test_language_detection_is_deterministic_and_conservative():
    assert language_detect.detect_language(EN_CLAIM + " " + EN_CLAIM)["language"] == "en"
    assert language_detect.detect_language(DE_EDU)["language"] == "de"
    assert language_detect.detect_language(DE_BUY)["language"] == "de"
    assert language_detect.detect_language(FR_CLAIM)["language"] == "unknown"
    assert language_detect.detect_language("Nvidia $200")["language"] == "unknown"
    assert language_detect.detect_language("")["language"] == "unknown"
    assert language_detect.language_of("x", "de") == "de"          # declared metadata wins
    assert language_detect.language_of(FR_CLAIM, "fr") == "unknown"  # an unsupported declaration does not


# --- English ---

def test_english_claim_bearing_and_educational_transcripts():
    check = cm.suspicious_empty_check(EN_CLAIM)
    assert check["language"] == "en" and check["guard"] == "available" and check["suspicious"]
    assert signals.build_research(EMPTY, _ctx(EN_CLAIM))["failure_reason"] == "suspicious_empty_extraction"
    edu = ("Let me explain what a P/E ratio is. A P/E of 20 means that you pay 20 dollars for every dollar of "
           "earnings. Historically the S&P 500 has returned about 10 percent per year.")
    assert cm.suspicious_empty_check(edu)["suspicious"] is False
    assert signals.build_research(EMPTY, _ctx(edu))["status"] == "no_claims_found"


# --- German ---

def test_german_price_forecast_is_suspicious():
    check = cm.suspicious_empty_check(DE_CLAIM)
    assert check["language"] == "de" and check["guard"] == "available"
    assert check["suspicious"] and check["strong"], check
    res = signals.build_research(EMPTY, _ctx(DE_CLAIM))
    assert res["status"] == "failed_retryable" and res["failure_reason"] == "suspicious_empty_extraction"
    assert res["signals"] is None
    assert signals.build_research(EMPTY, _ctx(DE_CLAIM), standalone=True)["status"] == "needs_review"


def test_german_buy_recommendation_is_suspicious():
    check = cm.suspicious_empty_check(DE_BUY)
    assert check["suspicious"] and any("Kauf" in s or "nachgekauft" in s for s in check["strong"]), check


def test_german_educational_transcript_is_no_claims_found():
    check = cm.suspicious_empty_check(DE_EDU)
    assert check["language"] == "de" and check["suspicious"] is False, check
    res = signals.build_research(EMPTY, _ctx(DE_EDU))
    assert res["status"] == "no_claims_found" and res["failure_reason"] is None


def test_german_questions_alone_do_not_trigger():
    assert cm.suspicious_empty_check(DE_QUESTIONS, language="de")["suspicious"] is False


def test_declared_language_metadata_selects_the_rule_set():
    # Too short to detect, but the stored transcript says German.
    short = "Ich erwarte, dass Nvidia 200 Dollar erreicht."
    assert language_detect.detect_language(short)["language"] == "unknown"
    assert cm.suspicious_empty_check(short)["guard"] == "unavailable"
    assert cm.suspicious_empty_check(short, language="de")["suspicious"] is True
    assert signals.build_research(EMPTY, _ctx(short, "de"))["failure_reason"] == "suspicious_empty_extraction"


# --- unsupported / unknown language ---

def test_unknown_language_is_never_certified_as_no_claims():
    check = cm.suspicious_empty_check(FR_CLAIM)
    assert check["guard"] == "unavailable" and check["suspicious"] is None and check["asset_free"] is False
    res = signals.build_research(EMPTY, _ctx(FR_CLAIM))
    assert res["status"] == "needs_review"
    assert res["failure_reason"] == "empty_extraction_language_guard_unavailable"
    assert res["signals"] is None and res["claims"] == []
    assert any(w.startswith("empty_extraction_language_guard_unavailable") for w in res["warnings"])
    # The same from the standalone and the chunked paths.
    assert signals.build_research(EMPTY, _ctx(FR_CLAIM), standalone=True)["failure_reason"] == \
        "empty_extraction_language_guard_unavailable"


def test_unknown_language_asset_free_transcript_is_the_one_reliable_exception():
    check = cm.suspicious_empty_check(FR_EMPTY)
    assert check["guard"] == "unavailable" and check["asset_free"] is True
    res = signals.build_research(EMPTY, _ctx(FR_EMPTY))
    assert res["status"] == "no_claims_found" and res["failure_reason"] is None
    assert any("language_guard_unavailable" in w for w in res["warnings"])
    assert cm.asset_free("Rien de spécial, mais Nvidia est mentionné.") is False
    assert cm.asset_free("Le prix est de 30 dollars.") is False
    assert cm.asset_free("Une hausse de 5 % est possible.") is False


def test_chunked_extraction_applies_the_language_guard(monkeypatch, tmp_path):
    monkeypatch.setattr(summarizer, "PARTIALS_DIR", str(tmp_path))
    monkeypatch.setattr(signals, "_research_chunk_tokens", lambda: 40000)
    monkeypatch.setattr(signals, "complete", lambda *a, **k: json.dumps(EMPTY))
    res = signals._extract_research_chunked(tn.normalize_transcript(FR_CLAIM, "v1"), {"video_id": "v1"})
    assert res["status"] == "needs_review" and res["failure_reason"] == "empty_extraction_language_guard_unavailable"
    res = signals._extract_research_chunked(tn.normalize_transcript(DE_CLAIM, "v1"),
                                            {"video_id": "v1", "transcript_language": "de"})
    assert res["status"] == "needs_review" and res["failure_reason"] == "suspicious_empty_extraction"


def test_language_travels_with_the_transcript_and_the_claims(monkeypatch):
    import scraper
    import transcript_store
    stored = {}
    monkeypatch.setattr(scraper, "get_transcript_from_video", lambda url, languages=None: {"transcript": DE_CLAIM, "reason": "ok"})
    monkeypatch.setattr(scraper.transcript_store, "store_transcript",
                        lambda details, text, source, reason, language=None: stored.update(language=language) or {"stored": True})
    captured = {}

    def combined(transcript, title, compact=False, channel_name=None, context=None):
        captured.update(context)
        return "TL;DR\n\n• x", {"status": "no_claims_found", "claims": [], "signals": None, "warnings": [],
                                 "coverage_status": "full", "run_key": "rk", "telemetry": {}}
    monkeypatch.setattr(scraper, "summarize_with_signals", combined)
    details = {"video_id": "v1", "video_url": "https://www.youtube.com/watch?v=v1", "video_title": "T",
               "channel_name": "HKCM", "published_at": "2026-07-01T00:00:00+00:00"}
    scraper._summarize_video(details, want_signals=True)
    assert details["transcript_language"] == "de" and stored["language"] == "de"
    assert captured["transcript_language"] == "de"
    claims, _ = cm.validate_claims([{"attribution_type": "speaker_personal_view", "claim_type": "forecast",
                                     "subject_mention": "Nvidia", "forecast_direction": "increase",
                                     "evidence_text": "Ich erwarte, dass Nvidia bis Jahresende 200 Dollar erreicht"}],
                                   tn.normalize_transcript(DE_CLAIM, "v1"), dict(captured, run_key="rk"))
    assert claims[0]["transcript_language"] == "de"
