"""research_backfill --adopt-stored: stored transcripts that never got research."""
import research_backfill as rb
import research_state as rs
import transcript_store as ts

ENGLISH = ("Nvidia is going to move higher from here and we think that "
           "the stock is a buy because of the data center demand. " * 20)
ARABIC = ("لا يزال مخطط البيتكوين يتداول دون المتوسط المتحرك البسيط لمدة 50 أسبوعاً. "
          "لم يحدث الكثير خلال عطلة نهاية الأسبوع، لكن حقيقة بقائنا تحت هذا المستوى تعني " * 12)
LANGS = {"c-en": ["en"]}


def _store(vid, text=ENGLISH, channel_id="c-en"):
    ts.store_transcript({"video_id": vid, "video_url": f"https://www.youtube.com/watch?v={vid}",
                         "channel_id": channel_id, "channel_name": "Chan", "video_title": f"Title {vid}",
                         "published_at": "2026-09-14T11:45:00+00:00", "duration_seconds": 600},
                        text, "supadata", "ok", language="en")


def test_adopts_only_orphaned_transcripts_in_the_channel_language():
    _store("o1")                                  # summary never went out: adopt
    _store("p1")                                  # still pending a summary: the daily run's
    _store("r1")                                  # already has research
    _store("ar1", ARABIC)                         # a translated track: never adopted
    _store("gone")
    ts.reject_transcript("gone", "language_mismatch:ar")   # retired capture: nothing to read
    state = rs.load_state()
    rs.update(state, "r1", delivery_status="sent", research_status="complete")
    rs.save_state(state)

    tally = rb.run_adopt_stored(pending={"p1": {"attempts": 1}}, channel_languages=LANGS)
    assert tally == {"adopted": 1, "pending_summary": 1, "foreign": 1, "missing": 1, "videos": ["o1"]}
    entry = rs.load_state()["videos"]["o1"]
    assert entry["research_status"] == "pending" and entry["delivery_status"] == "not_sent"
    assert entry["adopted_from"] == "stored_transcript" and entry["transcript_stored"] is True
    assert (entry["channel_id"], entry["channel_name"], entry["video_title"]) == ("c-en", "Chan", "Title o1")
    assert entry["published_at"] == "2026-09-14T11:45:00+00:00" and entry["transcript_language"] == "en"
    assert entry["transcript_hash"] == ts.load_transcript("o1")["transcript_hash"]
    assert rs.load_state()["videos"]["r1"]["research_status"] == "complete"     # untouched
    assert "o1" in rs.retry_candidates(rs.load_state())
    # Idempotent: an adopted video has a record now.
    again = rb.run_adopt_stored(pending={"p1": {"attempts": 1}}, channel_languages=LANGS)
    assert again["adopted"] == 0 and again["videos"] == []


def test_an_adopted_video_is_extracted_by_retry_into_the_canonical_claims(monkeypatch):
    monkeypatch.setattr(rb.research_budget, "remaining_requests", lambda providers=None: None)
    _store("o1")
    rb.run_adopt_stored(pending={}, channel_languages=LANGS)

    def extractor(nt, ctx):
        assert "Nvidia" in nt.text and ctx["published_at"] == "2026-09-14T11:45:00+00:00"
        claims = [{"claim_id": "clm-o1", "video_id": "o1", "schema_version": "2", "claim_type": "forecast",
                   "subject_mention": "Nvidia", "run_key": ctx["run_key"]}]
        return {"status": "complete", "claims": claims, "signals": None, "warnings": [],
                "coverage_status": "full", "run_key": ctx["run_key"], "extraction_model": "m", "telemetry": {}}

    assert rb.run_retry(extractor=extractor) == {"o1": "complete"}
    state = rs.load_state()
    assert state["videos"]["o1"]["research_status"] == "complete"
    assert state["videos"]["o1"]["delivery_status"] == "not_sent"               # nothing was sent
    assert [c["claim_id"] for c in rs.load_active_claims(state)] == ["clm-o1"]
