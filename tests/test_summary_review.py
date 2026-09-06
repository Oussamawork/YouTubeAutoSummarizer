import copy
import json

import pytest

import scraper
import summarizer
import summary_review as review
from transcript_normalize import normalize_transcript


SOURCE = "I rate Rubrik a hold with medium conviction. I would wait for a better opportunity to buy."
SUMMARY = "The speaker rates Rubrik a hold with medium conviction."


def verdict(supported=True):
    return {"checks": [{"line_id": 1, "verdict": "supported" if supported else "unsupported",
                        "evidence": ["I rate Rubrik a hold with medium conviction."],
                        "reason": "" if supported else "A rating is not ownership."}],
            "material_omissions": []}


def test_supported_line_retains_source_offsets_and_unknown_timestamp():
    nt = normalize_transcript(SOURCE)
    result = review.validate_review(verdict(), SUMMARY, nt)
    assert result["status"] == "approved"
    evidence = result["checks"][0]["evidence"][0]
    start, end = evidence["spans"][0]
    assert nt.text[start:end] == evidence["quote"]
    assert evidence["start_seconds"] is None


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(checks=[]),
    lambda d: d["checks"].append(copy.deepcopy(d["checks"][0])),
    lambda d: d["checks"][0].update(line_id=True),
    lambda d: d["checks"][0].update(line_id=2),
    lambda d: d["checks"][0].update(verdict=True),
    lambda d: d["checks"][0].update(evidence=[]),
    lambda d: d["checks"][0].update(evidence=[123]),
    lambda d: d["checks"][0].update(evidence=["I own Rubrik shares."]),
    lambda d: d["checks"][0].update(reason=None),
    lambda d: d.update(material_omissions=None),
    lambda d: d.update(material_omissions=[False]),
])
def test_malformed_or_ungrounded_review_cannot_approve(mutation):
    data = verdict()
    mutation(data)
    assert review.validate_review(data, SUMMARY, normalize_transcript(SOURCE))["status"] == "unavailable"


def test_all_nonblank_lines_need_a_verdict_in_order():
    assert review.validate_review(verdict(), SUMMARY + "\n\n• Extra unsupported claim.",
                                  normalize_transcript(SOURCE))["status"] == "unavailable"


def test_material_omission_or_uncertainty_rejects_even_with_real_evidence():
    data = verdict()
    data["material_omissions"] = ["The speaker says to wait for a better opportunity."]
    assert review.validate_review(data, SUMMARY, normalize_transcript(SOURCE))["status"] == "rejected"
    data = verdict()
    data["checks"][0]["verdict"] = "unclear"
    assert review.validate_review(data, SUMMARY, normalize_transcript(SOURCE))["status"] == "rejected"


def test_evidence_matching_preserves_negations_and_negative_numbers():
    assert not review.evidence_spans("Margins were 18.3%.", "Margins were -18.3%.")
    assert not review.evidence_spans("I own Rubrik.", "I do not own Rubrik.")
    assert review.evidence_spans("I rate Rubrik", "I rate\nRubrik") == [(0, 13)]


def test_duplicate_quotes_have_no_invented_unique_timestamp():
    result = review.validate_review(verdict(), SUMMARY, normalize_transcript(SOURCE + " " + SOURCE))
    quote = result["checks"][0]["evidence"][0]
    assert len(quote["spans"]) == 2
    assert "start_seconds" not in quote


def install(monkeypatch, responses):
    monkeypatch.setenv("SUMMARY_REVIEW_ENABLED", "true")
    calls = []
    values = iter(responses)
    def complete(system, user, **kwargs):
        calls.append((system, json.loads(user), kwargs))
        summarizer.LAST_CALL_TELEMETRY.update(model="reviewer")
        return next(values)
    monkeypatch.setattr(summarizer, "complete", complete)
    return calls


def test_repair_requires_fresh_review_and_preserves_generator_metadata(monkeypatch):
    calls = install(monkeypatch, [json.dumps(verdict(False)), SUMMARY, json.dumps(verdict())])
    summarizer.LAST_CALL_TELEMETRY.clear()
    summarizer.LAST_CALL_TELEMETRY.update(model="generator")
    text, report = review.review_summary("The speaker owns Rubrik.", normalize_transcript(SOURCE))
    assert text == SUMMARY and report["status"] == "approved"
    assert len(calls) == 3 and len(report["attempts"]) == 2
    assert all(call[1]["transcript"] == SOURCE for call in calls)
    assert summarizer.LAST_CALL_TELEMETRY == {"model": "generator"}
    assert report["attempts"][0]["telemetry"]["model"] == "reviewer"
    assert report["persisted"] is True


def test_failed_repair_is_never_delivered(monkeypatch):
    calls = install(monkeypatch, [json.dumps(verdict(False)), "Still wrong.", json.dumps(verdict(False))])
    text, report = review.review_summary("The speaker owns Rubrik.", normalize_transcript(SOURCE))
    assert text is None and report["status"] == "rejected" and len(calls) == 3


@pytest.mark.parametrize("failure", ["", "not JSON", summarizer.QUOTA_EXHAUSTED_SENTINEL,
                                     summarizer.INPUT_TOO_LARGE_SENTINEL, summarizer.TRUNCATED_SENTINEL])
def test_unavailable_review_never_approves_or_attempts_repair(monkeypatch, failure):
    calls = install(monkeypatch, [failure])
    text, report = review.review_summary(SUMMARY, normalize_transcript(SOURCE))
    assert text is None and report["status"] == "unavailable" and len(calls) == 1


def test_disabled_review_spends_no_requests(monkeypatch):
    monkeypatch.setenv("SUMMARY_REVIEW_ENABLED", "false")
    monkeypatch.setattr(summarizer, "complete", lambda *a, **k: pytest.fail("unexpected request"))
    assert review.review_summary(SUMMARY, normalize_transcript(SOURCE)) == (SUMMARY, {"status": "disabled"})


def test_empty_workflow_variable_keeps_review_enabled(monkeypatch):
    calls = install(monkeypatch, [json.dumps(verdict())])
    monkeypatch.setenv("SUMMARY_REVIEW_ENABLED", "")
    assert review.review_summary(SUMMARY, normalize_transcript(SOURCE))[1]["status"] == "approved"
    assert len(calls) == 1


@pytest.mark.parametrize("combined", [True, False])
def test_production_summary_paths_defer_unreviewed_text(monkeypatch, combined):
    install(monkeypatch, [summarizer.QUOTA_EXHAUSTED_SENTINEL])
    monkeypatch.setattr(scraper, "get_transcript_from_video", lambda *_: {"transcript": SOURCE})
    monkeypatch.setattr(scraper, "summarize_transcript", lambda *a, **k: SUMMARY)
    monkeypatch.setattr(scraper, "summarize_with_signals", lambda *a, **k: (SUMMARY, None))
    details = {"video_id": "test", "video_url": "https://youtube.com/watch?v=test",
               "video_title": "Test", "channel_name": "Test"}
    body, outcome, decided, _ = scraper._summarize_video(details, want_signals=combined)
    assert body is None and outcome == "summary_review_deferred" and not decided


def test_on_demand_review_deferral_reports_failure(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "test")
    monkeypatch.setenv("TELEGRAM_CHANNEL_ID", "test")
    monkeypatch.setattr(scraper, "_fetch_video_metadata", lambda *_: ("Title", "Channel"))
    monkeypatch.setattr(scraper, "_summarize_video", lambda *a, **k: (None, "summary_review_deferred", False, None))
    sent = []
    monkeypatch.setattr(scraper, "send_telegram_message", lambda *a: sent.append(a[-1]) or True)
    assert not scraper.summarize_on_demand("https://youtube.com/watch?v=test")
    assert "Try again later" in sent[0]


def test_review_retry_reuses_stored_source_and_draft(monkeypatch):
    install(monkeypatch, [json.dumps(verdict())])
    nt = normalize_transcript(SOURCE)
    monkeypatch.setattr(scraper.transcript_store, "load_transcript", lambda *_: {"raw_transcript": SOURCE})
    monkeypatch.setattr(scraper, "get_transcript_from_video", lambda *_: pytest.fail("paid refetch"))
    monkeypatch.setattr(scraper, "summarize_transcript", lambda *a, **k: pytest.fail("regenerated draft"))
    monkeypatch.setattr(scraper, "summarize_with_signals", lambda *a, **k: pytest.fail("regenerated combined draft"))
    details = {"video_id": "test", "video_url": "url", "video_title": "Title",
               "_summary_review_retry": {"draft": SUMMARY, "transcript_hash": nt.transcript_hash}}
    body, outcome, decided, _ = scraper._summarize_video(details, want_signals=True)
    assert body == SUMMARY and outcome == "sent" and decided


def test_review_retry_with_changed_transcript_regenerates_draft(monkeypatch):
    install(monkeypatch, [json.dumps(verdict())])
    monkeypatch.setattr(scraper.transcript_store, "load_transcript", lambda *_: {"raw_transcript": SOURCE})
    calls = []
    monkeypatch.setattr(scraper, "summarize_transcript", lambda *a, **k: calls.append(a[0]) or SUMMARY)
    details = {"video_id": "test", "video_url": "url", "video_title": "Title",
               "_summary_review_retry": {"draft": "stale", "transcript_hash": "changed"}}
    assert scraper._summarize_video(details)[0] == SUMMARY
    assert calls == [SOURCE]


def test_review_pending_survives_feed_expiry_and_is_not_redelivery():
    pending = {"v": {"channel_id": "c", "summary_review_retry": {
        "draft": SUMMARY, "channel_name": "C", "video_title": "T", "video_url": "url",
        "published_at": "2026-01-01T00:00:00+00:00"}}}
    assert scraper._evict_orphaned_pending(pending, "c", set()) == 0
    candidates = scraper._undelivered_candidates(pending, "c", set())
    assert candidates[0]["video_id"] == "v"
    assert candidates[0]["video_url"] == "url"
    assert scraper._undelivered(pending["v"]) is None


def test_scan_persists_review_retry_without_advancing_watermark(monkeypatch):
    from test_delivery import _harness, _vid
    def deferred(details, *args, **kwargs):
        details["summary_review"] = {"transcript_hash": "hash", "attempts": [{"summary": SUMMARY}]}
        return None, "summary_review_deferred", False, None
    calls = _harness(monkeypatch, [_vid("v1", "2026-07-03T00:00:00+00:00")], summarize=deferred)
    assert calls["sent"] == []
    assert calls["state"]["channels"]["c1"]["last_video_id"] == "seed"
    pending = calls["state"]["pending"]["v1"]
    assert pending["attempts"] == 0
    assert pending["summary_review_retry"]["draft"] == SUMMARY
    assert pending["last_attempt"]
