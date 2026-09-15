"""A video retried after a summary-quota / truncation / delivery deferral
reuses the transcript the store already holds instead of fetching it again.

On 2026-09-14/15, with Supadata dark and Gemini the one transcript source
left, three deferred videos were each transcribed a second time on the day
that quota was the bottleneck."""
import pytest

import scraper
import transcript_store

ENGLISH = ("Welcome back everyone. Today we are going to talk about the market and what I think "
           "happens next with the big technology names over the coming months. " * 6)
GERMAN = ("Willkommen zurück. Heute sprechen wir über den Markt und was ich in den nächsten Monaten "
          "bei den großen Technologiewerten erwarte, und warum das wichtig ist. " * 6)


def _details(video_id="v1", channel_id="UCx"):
    return {
        "video_id": video_id,
        "channel_id": channel_id,
        "channel_name": "Chan",
        "video_title": f"T-{video_id}",
        "video_url": f"https://www.youtube.com/watch?v={video_id}",
        "published_at": "2026-09-14T00:00:00+00:00",
    }


@pytest.fixture
def summary(monkeypatch):
    # The research path (want_signals) is what persists transcripts; the
    # combined call "fails" so the plain summary call answers.
    monkeypatch.setattr(scraper, "summarize_with_signals", lambda *a, **kw: None)
    monkeypatch.setattr(scraper, "summarize_transcript", lambda t, title, **kw: "TLDR\n\n• point")


@pytest.fixture
def fetch(monkeypatch):
    """The network fetch, recording every call; serves English."""
    calls = []

    def fake(url, languages=None):
        calls.append((url, languages))
        return {"transcript": ENGLISH, "reason": "gemini_ok", "language": "en"}

    monkeypatch.setattr(scraper, "get_transcript_from_video", fake)
    return calls


def _store(details, text, language, source="gemini_video"):
    rec = transcript_store.store_transcript(details, text, source, "gemini_ok", language=language)
    assert rec["stored"]
    return rec


def test_stored_transcript_is_reused_without_a_fetch(summary, fetch):
    details = _details()
    _store(details, ENGLISH, "en")

    body, outcome, decided, _ = scraper._summarize_video(details, languages=["en"], want_signals=True)

    assert (body, outcome, decided) == ("TLDR\n\n• point", "sent", True)
    assert fetch == []
    assert details["transcript_reason"] == "stored"
    assert details["transcript_source"] == "gemini_video"  # provenance is the original capture's
    assert details["transcript_language"] == "en"
    assert details["transcript_record"]["stored"] is True  # idempotent re-store, same file


def test_first_sighting_fetches_and_stores_then_the_retry_reuses(summary, fetch):
    details = _details("v2")
    assert scraper._summarize_video(details, languages=["en"], want_signals=True)[1] == "sent"
    assert len(fetch) == 1
    assert details["transcript_source"] == "gemini_video"

    retry = _details("v2")
    assert scraper._summarize_video(retry, languages=["en"], want_signals=True)[1] == "sent"
    assert len(fetch) == 1  # no second transcript request
    assert retry["transcript_reason"] == "stored"


def test_foreign_stored_transcript_is_not_reused(summary, fetch):
    # A German channel whose stored capture is English (the pre-#66 kind):
    # left for --reject-foreign-transcripts; the video is fetched fresh.
    details = _details("v3")
    _store(details, ENGLISH, "en")
    scraper._summarize_video(details, languages=["de"])
    assert len(fetch) == 1
    assert details["transcript_reason"] == "gemini_ok"


def test_stored_capture_with_unknown_language_is_verified_by_its_text(summary, fetch):
    # Older captures recorded "unknown" / no language; the text decides.
    details = _details("v4")
    _store(details, GERMAN, "unknown")
    assert scraper._summarize_video(details, languages=["de"])[1] == "sent"
    assert fetch == []
    assert details["transcript_language"] == "de"

    other = _details("v5")
    _store(other, GERMAN, None)
    scraper._summarize_video(other, languages=["en"])
    assert len(fetch) == 1  # German text, English channel: not reused


def test_persistence_off_or_no_video_id_means_a_normal_fetch(summary, fetch, monkeypatch):
    details = _details("v6")
    _store(details, ENGLISH, "en")
    monkeypatch.setattr(scraper, "PERSIST_TRANSCRIPTS", False)
    scraper._summarize_video(details, languages=["en"])
    assert len(fetch) == 1

    monkeypatch.setattr(scraper, "PERSIST_TRANSCRIPTS", True)
    anonymous = dict(_details("v7"), video_id=None)
    scraper._summarize_video(anonymous, languages=["en"])
    assert len(fetch) == 2


def test_rejected_capture_is_not_reused(summary, fetch):
    details = _details("v8")
    _store(details, ENGLISH, "en")
    assert transcript_store.reject_transcript("v8", "language_mismatch")
    scraper._summarize_video(details, languages=["en"])
    assert len(fetch) == 1


def test_a_broken_store_never_blocks_the_video(summary, fetch, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("disk")

    monkeypatch.setattr(transcript_store, "load_transcript", boom)
    assert scraper._summarize_video(_details("v9"), languages=["en"])[1] == "sent"
    assert len(fetch) == 1


def test_reused_transcript_counts_as_a_success_reason():
    import transcript
    assert "stored" in transcript.TRANSCRIPT_SUCCESS_REASONS
