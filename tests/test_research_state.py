"""Independent research state, canonical stores, transcript persistence, backfill."""
import json

import claims as cm
import research_backfill as rb
import research_state as rs
import transcript_normalize as tn
import transcript_store as ts


def _details(vid="v1"):
    return {"video_id": vid, "video_url": f"https://youtu.be/{vid}", "channel_id": "c1",
            "channel_name": "Chan", "video_title": "T", "published_at": "2026-07-01T00:00:00+00:00",
            "duration_seconds": 600}


def test_transcript_is_stored_raw_compressed_and_idempotent():
    raw = "line one\r\nline   two with   spaces\n\n99% of $1,000"
    rec = ts.store_transcript(_details(), raw, "supadata", "ok")
    assert rec["stored"] is True and rec["raw_char_count"] == len(raw)
    loaded = ts.load_transcript("v1")
    assert loaded["raw_transcript"] == raw                    # never the cleaned version
    assert loaded["transcript_hash"] == rec["transcript_hash"] == tn.transcript_hash(raw)
    assert loaded["transcript_source"] == "supadata" and loaded["acquired_at"]
    again = ts.store_transcript(_details(), raw, "supadata", "ok")
    assert again["stored"] and again["path"] == rec["path"]
    index = [json.loads(l) for l in open(ts.TRANSCRIPT_INDEX, encoding="utf-8")]
    assert len(index) == 1                                    # not re-indexed
    other = ts.store_transcript(_details(), raw + " more", "gemini_video", "gemini_ok")
    assert other["path"] != rec["path"]                      # a different capture sits beside it
    assert ts.load_transcript("v1")["raw_transcript"] == raw  # original untouched


def test_transcript_persistence_failure_is_reported_not_hidden(monkeypatch):
    monkeypatch.setattr(ts, "TRANSCRIPTS_DIR", "/proc/nope/cannot-write")
    rec = ts.store_transcript(_details(), "text", "supadata", "ok")
    assert rec["stored"] is False and rec["error"]
    assert ts.store_transcript(_details(""), "text", "supadata", "ok")["stored"] is False


def test_state_roundtrip_and_status_guard():
    state = rs.load_state()
    rs.update(state, "v1", delivery_status="sent", research_status="failed_retryable")
    rs.update(state, "v2", research_status="bogus")
    rs.save_state(state)
    again = rs.load_state()
    assert again["videos"]["v1"]["research_status"] == "failed_retryable"
    assert again["videos"]["v1"]["delivery_status"] == "sent" and again["videos"]["v1"]["updated_at"]
    assert again["videos"]["v2"]["research_status"] == "failed_retryable"
    assert set(rs.retry_candidates(again)) == {"v1", "v2"}
    rs.update(again, "v1", next_eligible_at="2999-01-01T00:00:00+00:00")
    assert "v1" not in rs.retry_candidates(again)


def _claims(text, run_key="rk1", target=200):
    nt = tn.normalize_transcript(text, "v1")
    ctx = {"video_id": "v1", "channel_name": "Chan", "video_title": "T", "run_key": run_key,
           "published_at": "2026-07-01T00:00:00+00:00"}
    raw = [{"attribution_type": "speaker_personal_view", "claim_type": "price_target", "is_forward_looking": True,
            "subject_mention": "Nvidia", "stance": "bullish", "target_kind": "absolute_value",
            "target_value": target, "horizon_original": "this year",
            "evidence_text": f"Nvidia will hit ${target} this year", "extraction_confidence": "high"}]
    return cm.validate_claims(raw, nt, ctx)[0]


def test_storing_the_same_run_twice_appends_nothing():
    state = rs.load_state()
    claims = _claims("Nvidia will hit $200 this year.")
    assert rs.store_claims(claims, state, "v1", "rk1") == 1
    assert rs.store_claims(claims, state, "v1", "rk1") == 0
    assert rs.store_claims(_claims("Nvidia will hit $200 this year."), state, "v1", "rk1") == 0
    assert len(rs.load_claims("v1")) == 1
    assert state["videos"]["v1"]["active_run_key"] == "rk1"


def test_newer_run_supersedes_older_and_only_one_version_counts():
    state = rs.load_state()
    rs.store_claims(_claims("Nvidia will hit $200 this year."), state, "v1", "rk1")
    rs.store_claims(_claims("Nvidia will hit $200 this year.", run_key="rk2"), state, "v1", "rk2")
    assert len(rs.load_claims("v1")) == 2                   # history retained
    active = rs.load_active_claims(state)
    assert len(active) == 1 and active[0]["run_key"] == "rk2"
    assert state["videos"]["v1"]["superseded_run_keys"] == ["rk1"]


def test_review_queue_receives_flagged_claims():
    state = rs.load_state()
    nt = tn.normalize_transcript("Nvidia looks ok.", "v1")
    claims, _ = cm.validate_claims([{"subject_mention": "Nvidia", "stance": "bullish", "claim_type": "stance",
                                     "evidence_text": "text that is not there at all"}], nt,
                                   {"video_id": "v1", "run_key": "rk"})
    rs.store_claims(claims, state, "v1", "rk")
    queue = rs._read_jsonl(rs._paths()["review"])
    assert len(queue) == 1 and "evidence_not_found" in queue[0]["reasons"]


def test_gate_outcomes_and_segments_are_recorded():
    rs.record_gate_outcome("c1", {"video_id": "v9", "video_title": "t"}, "title_filtered",
                           channel_config={"only": ["btc"]}, discovery_source="rss", min_duration=90)
    rs.record_gate_outcome("c1", {"video_id": "v8"}, "not-a-real-outcome")
    rows = rs.load_gate_outcomes()
    assert rows[0]["outcome"] == "title_filtered" and rows[0]["title_filters"] == ["btc"]
    assert rows[0]["filter_config_version"] == rs.FILTER_CONFIG_VERSION
    assert rows[1]["outcome"] == "other"
    nt = tn.normalize_transcript("Some text about Nvidia. " * 20, "v1")
    assert rs.store_segments("v1", nt) and rs.store_segments("v1", nt)
    segs = rs._read_jsonl(rs._paths()["segments"])
    assert len(segs) == len(nt.segments)


# --- backfill ---

def _fake_extractor(status="complete", target=200):
    def extractor(nt, ctx):
        claims = _claims(nt.text, run_key=ctx["run_key"], target=target) if status in ("complete", "partial") else []
        return {"status": status, "failure_reason": None if status == "complete" else status,
                "claims": claims, "signals": cm.claims_to_legacy_signals(claims) if status == "complete" else None,
                "warnings": [], "coverage_status": "full" if status == "complete" else "partial",
                "run_key": ctx["run_key"], "extraction_model": "m", "telemetry": {}}
    return extractor


def test_retry_uses_stored_transcript_and_is_idempotent(monkeypatch, tmp_path):
    monkeypatch.setattr(rb, "SIGNALS_FILE", str(tmp_path / "signals.jsonl"))
    ts.store_transcript(_details(), "Nvidia will hit $200 this year.", "supadata", "ok")
    state = rs.load_state()
    rs.update(state, "v1", delivery_status="sent", research_status="failed_retryable",
              channel_name="Chan", video_title="T", published_at="2026-07-01T00:00:00+00:00")
    rs.save_state(state)
    calls = []

    def extractor(nt, ctx):
        calls.append(ctx["run_key"])
        return _fake_extractor()(nt, ctx)

    assert rb.run_retry(extractor=extractor) == {"v1": "complete"}
    state = rs.load_state()
    assert state["videos"]["v1"]["research_status"] == "complete"
    assert state["videos"]["v1"]["delivery_status"] == "sent"      # untouched
    assert state["videos"]["v1"]["active_run_key"] == calls[0]
    assert len(rs.load_claims("v1")) == 1
    rows = [json.loads(l) for l in open(tmp_path / "signals.jsonl", encoding="utf-8")]
    assert rows[0]["backfilled"] and rows[0]["signals"]["assets"][0]["ticker"] == "NVDA"
    # Same versions again: nothing to do, no model call, no duplicate claims.
    assert rb.run_retry(extractor=extractor) == {}
    assert rb.process_video(rs.load_state(), "v1", extractor) == "complete"
    assert len(calls) == 1 and len(rs.load_claims("v1")) == 1


def test_retry_quota_deferral_does_not_count_an_attempt_and_stops_the_run():
    for vid in ("v1", "v2"):
        ts.store_transcript(_details(vid), "Nvidia will hit $200 this year.", "supadata", "ok")
    state = rs.load_state()
    for vid in ("v1", "v2"):
        rs.update(state, vid, research_status="pending")
    rs.save_state(state)
    done = rb.run_retry(extractor=_fake_extractor("quota_deferred"))
    assert done == {"v1": "quota_deferred"}                # stopped after the first
    state = rs.load_state()
    assert state["videos"]["v1"]["attempt_count"] == 0
    assert state["videos"]["v1"]["research_status"] == "quota_deferred"
    assert state["videos"]["v2"]["research_status"] == "pending"


def test_retry_failure_counts_and_eventually_finalizes():
    ts.store_transcript(_details(), "Nvidia will hit $200 this year.", "supadata", "ok")
    state = rs.load_state()
    rs.update(state, "v1", research_status="pending", attempt_count=rs.MAX_RESEARCH_ATTEMPTS - 1)
    rs.save_state(state)
    assert rb.process_video(state, "v1", _fake_extractor("failed_retryable")) == "failed_final"
    assert rs.load_state()["videos"]["v1"]["research_status"] == "failed_final"


def test_missing_transcript_is_not_refetched():
    state = rs.load_state()
    rs.update(state, "v1", research_status="pending")
    assert rb.process_video(state, "v1", _fake_extractor()) == "failed_retryable"
    assert state["videos"]["v1"]["failure_reason"] == "transcript_not_stored"


def test_legacy_import_is_idempotent(tmp_path):
    path = tmp_path / "legacy.jsonl"
    row = {"date": "2026-07-24", "video_id": "old1", "channel_id": "c", "channel_name": "Chan",
           "video_title": "t", "published_at": "2026-07-24T17:00:15+00:00", "summary": "s",
           "signals": {"assets": [{"name": "Bitcoin", "ticker": "BTC", "type": "crypto", "stance": "bullish",
                                   "conviction": "medium", "action": "watch", "price_target": None,
                                   "horizon": "short", "catalysts": []}], "market_sentiment": "bullish", "topics": []}}
    path.write_text(json.dumps(row) + "\n" + json.dumps(dict(row, video_id="old2", signals=None)) + "\n")
    assert rb.import_legacy(str(path)) == 1
    assert rb.import_legacy(str(path)) == 0
    active = rs.load_active_claims()
    assert len(active) == 1 and active[0]["schema_version"] == "legacy" and active[0]["review_required"]
    assert rs.load_state()["videos"]["old2"]["transcript_stored"] is False


def test_reprocess_targets_stale_versions_only(monkeypatch, tmp_path):
    monkeypatch.setattr(rb, "SIGNALS_FILE", str(tmp_path / "signals.jsonl"))
    ts.store_transcript(_details(), "Nvidia will hit $200 this year.", "supadata", "ok")
    state = rs.load_state()
    rs.update(state, "v1", research_status="complete", extraction_prompt_version="0",
              schema_version=cm.SCHEMA_VERSION, normalization_version=tn.NORMALIZATION_VERSION,
              active_run_key="oldkey")
    rs.update(state, "v2", research_status="complete", extraction_prompt_version=cm.EXTRACTION_PROMPT_VERSION,
              schema_version=cm.SCHEMA_VERSION, normalization_version=tn.NORMALIZATION_VERSION)
    rs.save_state(state)
    done = rb.run_reprocess(extractor=_fake_extractor())
    assert done == {"v1": "complete"}
    entry = rs.load_state()["videos"]["v1"]
    assert entry["superseded_run_keys"] == ["oldkey"] and entry["extraction_prompt_version"] == cm.EXTRACTION_PROMPT_VERSION
