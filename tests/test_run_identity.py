"""Hardening item 2: the extraction run identity covers every material
processing input, so a changed model or chunking policy is a new run."""
import claims as cm
import research_backfill
import research_state
import signals
import summarizer
import transcript_normalize as tn
import transcript_store

TEXT = "I expect Nvidia to fall over the next three months. I remain bullish over five years."


def _gemini_env(monkeypatch, model):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODEL", model)
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", model)  # same model: deduplicated to one entry
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("SUMMARY_MODELS", "*")


def test_run_key_carries_versions_chunking_mode_and_policy(monkeypatch):
    _gemini_env(monkeypatch, "gemini-3.7-flash")
    nt = tn.normalize_transcript(TEXT, "v1")
    key = signals.run_key_for(nt, "standalone")
    parts = research_state.parse_run_key(key)
    assert parts["transcript_hash_prefix"] == nt.transcript_hash[:16]
    assert parts["normalization_version"] == nt.normalization_version
    assert parts["prompt_version"] == cm.EXTRACTION_PROMPT_VERSION
    assert parts["schema_version"] == cm.SCHEMA_VERSION
    assert parts["chunking_version"] == tn.CHUNKING_VERSION
    assert parts["extraction_mode"] == "standalone"
    identity = signals.extraction_identity("standalone")
    assert parts["policy_digest"] == research_state.policy_digest(identity)
    assert identity["model_policy"] == "gemini-3.7-flash:gemini-3.7-flash"
    assert "temperature=" in identity["model_config"] and "claims_max_output=" in identity["model_config"]
    assert identity["chunk_policy"] == "full"


def test_changing_the_extraction_model_changes_the_run_key(monkeypatch):
    nt = tn.normalize_transcript(TEXT, "v1")
    _gemini_env(monkeypatch, "gemini-3.7-flash")
    first = signals.run_key_for(nt, "standalone")
    _gemini_env(monkeypatch, "gemini-3.6-flash")
    second = signals.run_key_for(nt, "standalone")
    assert first != second
    # Same versions, same mode: only the policy digest differs.
    a, b = research_state.parse_run_key(first), research_state.parse_run_key(second)
    assert {k: v for k, v in a.items() if k != "policy_digest"} == {k: v for k, v in b.items() if k != "policy_digest"}
    # And a changed generation setting is a different extraction too.
    monkeypatch.setattr(summarizer, "LLM_TEMPERATURE", 0.9)
    assert signals.run_key_for(nt, "standalone") != second


def test_changing_the_chunking_version_or_chunk_policy_changes_the_run_key(monkeypatch):
    _gemini_env(monkeypatch, "gemini-3.7-flash")
    nt = tn.normalize_transcript(TEXT, "v1")
    standalone = signals.run_key_for(nt, "standalone")
    chunked = signals.run_key_for(nt, "chunked", 12000)
    smaller = signals.run_key_for(nt, "chunked", 6000)
    exhaustive = signals.run_key_for(nt, "exhaustive", 12000)
    combined = signals.run_key_for(nt, "combined")
    assert len({standalone, chunked, smaller, exhaustive, combined}) == 5
    assert research_state.parse_run_key(chunked)["extraction_mode"] == "chunked"
    monkeypatch.setattr(tn, "CHUNKING_VERSION", "99")
    assert signals.run_key_for(nt, "chunked", 12000) != chunked
    assert research_state.parse_run_key(signals.run_key_for(nt, "chunked", 12000))["chunking_version"] == "99"


def test_same_inputs_give_the_same_key_and_legacy_keys_still_parse(monkeypatch):
    _gemini_env(monkeypatch, "gemini-3.7-flash")
    nt = tn.normalize_transcript(TEXT, "v1")
    assert signals.run_key_for(nt, "standalone") == signals.run_key_for(nt, "standalone")
    legacy = research_state.run_key(nt.transcript_hash, "1", "2", "2")
    parts = research_state.parse_run_key(legacy)
    assert parts["prompt_version"] == "2" and parts["extraction_mode"] is None


def test_retry_after_a_model_change_is_a_new_run_that_supersedes_the_old(monkeypatch, tmp_path):
    monkeypatch.setattr(research_state, "RESEARCH_DIR", str(tmp_path / "research"))
    monkeypatch.setattr(transcript_store, "TRANSCRIPTS_DIR", str(tmp_path / "transcripts"))
    monkeypatch.setattr(research_backfill.research_budget, "check", lambda nt, **kw: {"allowed": True})
    video = {"video_id": "v1", "channel_name": "Chan", "video_title": "T", "published_at": "2026-07-01T00:00:00+00:00"}
    transcript_store.store_transcript(video, TEXT, "supadata", "ok")
    nt = tn.normalize_transcript(TEXT, "v1")
    raw = [{"attribution_type": "speaker_personal_view", "claim_type": "forecast", "subject_mention": "Nvidia",
            "stance": "bearish", "forecast_direction": "decrease", "horizon_original": "over the next three months",
            "evidence_text": "I expect Nvidia to fall over the next three months"}]

    def extractor(nt, ctx):
        validated, _ = cm.validate_claims(raw, nt, ctx)
        return {"status": "complete", "failure_reason": None, "claims": validated, "signals": None,
                "warnings": [], "coverage_status": "full", "run_key": ctx["run_key"],
                "run_identity": ctx.get("run_identity"), "extraction_model": ctx["run_identity"]["model_policy"] if ctx.get("run_identity") else None,
                "telemetry": {}}

    _gemini_env(monkeypatch, "gemini-3.7-flash")
    state = research_state.load_state()
    research_state.update(state, "v1", research_status="pending")
    assert research_backfill.process_video(state, "v1", extractor) == "complete"
    first_key = research_state.get(state, "v1")["active_run_key"]
    # Same model again: an identical extraction is a no-op.
    assert research_backfill.process_video(state, "v1", extractor) == "complete"
    assert research_state.get(state, "v1")["active_run_key"] == first_key
    assert len(research_state.load_runs()) == 1

    # A different model is a NEW run: it runs, supersedes the old one, and
    # only its claims are active.
    _gemini_env(monkeypatch, "gemini-3.6-flash")
    research_state.update(state, "v1", research_status="pending")
    assert research_backfill.process_video(state, "v1", extractor) == "complete"
    entry = research_state.get(state, "v1")
    assert entry["active_run_key"] != first_key
    assert entry["superseded_run_keys"] == [first_key]
    assert entry["run_identity"]["model_policy"] == "gemini-3.6-flash:gemini-3.6-flash"
    runs = research_state.load_runs()
    assert len(runs) == 2 and runs[-1]["identity"]["model_policy"] == "gemini-3.6-flash:gemini-3.6-flash"
    active = research_state.load_active_claims(state)
    assert len(active) == 1 and active[0]["run_key"] == entry["active_run_key"]
    assert len(research_state.load_claims()) == 2  # the old run's claims stay on disk, superseded
