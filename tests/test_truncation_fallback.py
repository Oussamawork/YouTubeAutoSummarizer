"""Item 2: a truncated combined response never blocks a valid summary.
Item 10: partial-cache records are reused only under identical inputs."""
import json

import pytest

import claims as cm
import partial_cache
import research_backfill
import research_state
import scraper
import signals
import summarizer
import transcript_normalize as tn
import transcript_store

TRANSCRIPT = ("Welcome back. I expect Nvidia to fall over the next three months, but I remain bullish over five "
              "years. I'd buy Palantir under $20. Micron is the cheapest memory name right now.")
CLAIM = {"attribution_type": "speaker_personal_view", "claim_type": "forecast", "is_forward_looking": True,
         "subject_mention": "Nvidia", "asset_type": "stock", "stance": "bearish", "forecast_metric": "price",
         "forecast_direction": "decrease", "horizon_original": "over the next three months",
         "certainty_level": "medium", "evidence_text": "I expect Nvidia to fall over the next three months",
         "extraction_confidence": "high"}


class _Resp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text

    def json(self):
        return self._payload


def _post_router(calls, claims_mode="quota"):
    """A fake provider: the combined envelope always ends with
    finish_reason=length (escalation included); the summary-only prompt
    completes; a standalone claims call either hits a day quota or answers."""
    def post(url, headers=None, json=None, timeout=None):
        system = json["messages"][0]["content"]
        calls.append({"max_tokens": json["max_tokens"], "system": system})
        if "OUTPUT ENVELOPE" in system:
            return _Resp(200, {"choices": [{"message": {"content": '{"summary": "TL;DR line\\n\\n• bullet", "claims": [{'},
                                            "finish_reason": "length"}]})
        if "EXTRACTION for a research dataset" in system:
            if claims_mode == "quota":
                return _Resp(429, {}, text='{"error": {"details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel"}]}]}}')
            return _Resp(200, {"choices": [{"message": {"content": json_mod.dumps(
                {"claims": [CLAIM], "extraction_metadata": {"warnings": []}})}, "finish_reason": "stop"}]})
        return _Resp(200, {"choices": [{"message": {"content": "TL;DR line\n\n• Nvidia may fall near term\n• Bullish five years"},
                                        "finish_reason": "stop"}]})
    return post


json_mod = json


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(research_state, "RESEARCH_DIR", str(tmp_path / "research"))
    monkeypatch.setattr(transcript_store, "TRANSCRIPTS_DIR", str(tmp_path / "transcripts"))
    monkeypatch.setattr(summarizer, "PARTIALS_DIR", str(tmp_path / "partials"))
    monkeypatch.setattr(research_backfill, "SIGNALS_FILE", str(tmp_path / "signals.jsonl"))
    monkeypatch.setattr(summarizer, "_provider_configs",
                        lambda: [{"name": "p", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                                  "api_key": "k", "model": "gemini-3.7-flash"}])
    monkeypatch.setattr(summarizer.time, "sleep", lambda *_: None)
    summarizer._EXHAUSTED_PROVIDERS.clear()
    monkeypatch.setattr(scraper, "get_transcript_from_video", lambda url: {"transcript": TRANSCRIPT, "reason": "ok"})
    rows = []
    monkeypatch.setattr(scraper, "append_jsonl", lambda path, rec: rows.append((path, rec)) or True)
    yield rows
    summarizer._EXHAUSTED_PROVIDERS.clear()


def _run_main(monkeypatch, calls, claims_mode):
    for name, value in {"YOUTUBE_API_KEY": "yt", "TELEGRAM_TOKEN": "tok", "TELEGRAM_CHANNEL_ID": "premium",
                        "MARKET_SIGNALS": "true"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("DAILY_DIGEST", raising=False)
    monkeypatch.setattr(summarizer.requests, "post", _post_router(calls, claims_mode))
    video = {"video_id": "v1", "channel_name": "Chan", "video_title": "Nvidia outlook",
             "video_url": "https://www.youtube.com/watch?v=v1", "published_at": "2026-07-01T14:00:00+00:00"}
    monkeypatch.setattr(scraper, "read_channels", lambda path: [{"channel_id": "c1", "digest": False, "max_per_run": 3}])
    state = {"channels": {}, "pending": {}}
    saved = []
    monkeypatch.setattr(scraper, "load_state", lambda path: state)
    monkeypatch.setattr(scraper, "save_state", lambda path, st: saved.append(json.loads(json.dumps(st))))
    monkeypatch.setattr(scraper, "get_recent_videos", lambda key, cid: [video])
    sent = []
    monkeypatch.setattr(scraper, "send_telegram_message", lambda *args: sent.append(args) or True)
    scraper.main()
    return sent, saved


def test_truncated_combined_call_still_delivers_the_summary_and_queues_claims(isolated, monkeypatch):
    rows = isolated
    calls = []
    sent, saved = _run_main(monkeypatch, calls, claims_mode="quota")

    # 1. The combined response ended with finish_reason=length and was
    #    escalated before being given up on.
    combined = [c for c in calls if "OUTPUT ENVELOPE" in c["system"]]
    assert len(combined) >= 2 and combined[1]["max_tokens"] > combined[0]["max_tokens"]
    # 2. The summary-only fallback succeeded and 3. Telegram delivery happened
    #    with THAT summary, not the half-written envelope.
    assert len(sent) == 1
    assert "Nvidia may fall near term" in sent[0][6] and "{" not in sent[0][6]
    # 6. The watermark advanced with the delivery.
    assert saved[-1]["channels"]["c1"]["last_video_id"] == "v1"
    # 5. The claims were tried separately (a standalone claims call went out
    #    AFTER the summary-only call) and 4. research stays pending on quota.
    kinds = ["combined" if "OUTPUT ENVELOPE" in c["system"] else "claims"
             if "EXTRACTION for a research dataset" in c["system"] else "summary" for c in calls]
    assert kinds.index("summary") > max(i for i, k in enumerate(kinds) if k == "combined")
    assert "claims" in kinds and kinds.index("claims") > kinds.index("summary")
    entry = research_state.get(research_state.load_state(), "v1")
    assert entry["delivery_status"] == "sent"
    assert entry["research_status"] == "quota_deferred"
    assert "v1" in research_state.retry_candidates(research_state.load_state())
    assert entry["attempt_count"] == 0  # a quota deferral never counts
    # 7. Never no_claims_found, and the compatibility row carries null signals.
    signal_rows = [r for p, r in rows if p == scraper.SIGNALS_FILE]
    assert signal_rows[0]["research_status"] == "quota_deferred" and signal_rows[0]["signals"] is None

    # The retry job later completes the research from the stored transcript,
    # without touching delivery state.
    def extractor(nt, ctx):
        validated, _ = cm.validate_claims([CLAIM], nt, ctx)
        return {"status": "complete", "failure_reason": None, "claims": validated,
                "signals": cm.claims_to_legacy_signals(validated), "warnings": [], "coverage_status": "full",
                "run_key": ctx["run_key"], "extraction_model": "m", "telemetry": {}}
    state = research_state.load_state()
    # The fake provider's daily budget is spent, so the retry job's own
    # summary-reserve check would (correctly) defer the video; the later run
    # this stands for has budget again.
    monkeypatch.setattr(research_backfill.research_budget, "remaining_requests", lambda providers=None: None)
    assert research_backfill.process_video(state, "v1", extractor) == "complete"
    assert research_state.get(state, "v1")["delivery_status"] == "sent"
    assert [c["stance"] for c in research_state.load_active_claims(state)] == ["bearish"]


def test_claims_are_extracted_separately_in_the_same_run_when_budget_permits(isolated, monkeypatch):
    calls = []
    sent, saved = _run_main(monkeypatch, calls, claims_mode="ok")
    assert len(sent) == 1 and saved[-1]["channels"]["c1"]["last_video_id"] == "v1"
    entry = research_state.get(research_state.load_state(), "v1")
    assert entry["research_status"] == "complete"
    assert entry["coverage_status"] == "chunked_full"  # chunked on purpose: the output was what overflowed
    assert [c["stance"] for c in research_state.load_active_claims()] == ["bearish"]


def test_complete_reports_truncation_rather_than_quota_or_empty(monkeypatch):
    monkeypatch.setattr(summarizer, "_provider_configs", lambda: [
        {"name": "a", "base_url": "u", "api_key": "k", "model": "m"},
        {"name": "b", "base_url": "u", "api_key": "k", "model": "m2"}])
    monkeypatch.setattr(summarizer, "_call_provider",
                        lambda p, *a, **k: summarizer.TRUNCATED_SENTINEL if p["name"] == "a" else summarizer.QUOTA_EXHAUSTED_SENTINEL)
    summarizer._EXHAUSTED_PROVIDERS.clear()
    assert summarizer.complete("s", "u") == summarizer.TRUNCATED_SENTINEL
    summarizer._EXHAUSTED_PROVIDERS.clear()


def test_summarize_with_signals_returns_summary_only_marker_on_truncation(monkeypatch):
    monkeypatch.setattr(signals, "complete", lambda *a, **k: summarizer.TRUNCATED_SENTINEL)
    nt = tn.normalize_transcript(TRANSCRIPT, "v1")
    summary, research = signals.summarize_with_signals("t", context={"video_id": "v1", "normalized": nt})
    assert summary is None
    assert research["status"] == "failed_retryable" and research["failure_reason"] == "combined_output_truncated"
    assert research["retry_separately"] is True and research["signals"] is None


def test_standalone_truncation_falls_back_to_smaller_chunks(monkeypatch, tmp_path):
    monkeypatch.setattr(summarizer, "PARTIALS_DIR", str(tmp_path))
    nt = tn.normalize_transcript(TRANSCRIPT, "v1")
    seen = []

    def fake_complete(system, user, **kw):
        seen.append(user)
        if len(seen) == 1:
            return summarizer.TRUNCATED_SENTINEL
        return json.dumps({"claims": [CLAIM], "extraction_metadata": {"warnings": []}})
    monkeypatch.setattr(signals, "complete", fake_complete)
    monkeypatch.setattr(signals, "_research_chunk_tokens", lambda: 40000)
    res = signals.extract_research(nt, {"video_id": "v1"})
    assert res["status"] == "complete" and res["coverage_status"] == "chunked_full"
    assert "part 1 of" in seen[1]


# --- Item 10: partial-cache versioning -------------------------------------


def _chunk(nt):
    import token_budget
    return tn.chunk_transcript(nt, 10_000, token_budget.estimate_tokens)[0]


def test_partial_is_reused_only_when_every_input_matches(monkeypatch, tmp_path):
    monkeypatch.setattr(summarizer, "PARTIALS_DIR", str(tmp_path))
    nt = tn.normalize_transcript(TRANSCRIPT, "v1")
    chunk = _chunk(nt)
    partial_cache.save("research_claims", nt, chunk, {"raw_claims": [CLAIM]}, "2", "2", "p:m")
    assert partial_cache.load("research_claims", nt, chunk, "2", "2", "p:m") == {"raw_claims": [CLAIM]}
    # A different prompt version, schema version, model policy, task type,
    # chunk boundary, normalization or chunking version each invalidates it.
    assert partial_cache.load("research_claims", nt, chunk, "3", "2", "p:m") is None
    assert partial_cache.load("research_claims", nt, chunk, "2", "3", "p:m") is None
    assert partial_cache.load("research_claims", nt, chunk, "2", "2", "q:other") is None
    assert partial_cache.load("summary_notes", nt, chunk, "2", "2", "p:m") is None
    moved = tn.Chunk(**dict(chunk.__dict__, end_character=chunk.end_character - 1))
    assert partial_cache.load("research_claims", nt, moved, "2", "2", "p:m") is None
    monkeypatch.setattr(tn, "CHUNKING_VERSION", "99")
    assert partial_cache.load("research_claims", nt, chunk, "2", "2", "p:m") is None
    monkeypatch.setattr(tn, "CHUNKING_VERSION", "1")
    other = tn.normalize_transcript(TRANSCRIPT + " extra words.", "v1")
    assert partial_cache.load("research_claims", other, _chunk(other), "2", "2", "p:m") is None


def test_changing_the_extraction_prompt_version_invalidates_an_old_claims_partial(monkeypatch, tmp_path):
    monkeypatch.setattr(summarizer, "PARTIALS_DIR", str(tmp_path))
    monkeypatch.setattr(summarizer, "_provider_configs", lambda: [{"name": "p", "model": "m", "base_url": "u", "api_key": "k"}])
    nt = tn.normalize_transcript(TRANSCRIPT, "v1")
    calls = []

    def fake_complete(system, user, **kw):
        calls.append(user)
        return json.dumps({"claims": [CLAIM], "extraction_metadata": {"warnings": []}})
    monkeypatch.setattr(signals, "complete", fake_complete)
    monkeypatch.setattr(signals, "_research_chunk_tokens", lambda: 40000)
    assert signals._extract_research_chunked(nt, {"video_id": "v1"})["status"] == "complete"
    assert signals._extract_research_chunked(nt, {"video_id": "v1"})["status"] == "complete"
    assert len(calls) == 1  # the second run resumed from the partial
    monkeypatch.setattr(cm, "EXTRACTION_PROMPT_VERSION", "999")
    assert signals._extract_research_chunked(nt, {"video_id": "v1"})["status"] == "complete"
    assert len(calls) == 2  # a new prompt version re-extracts; the stale partial is not reused


def test_summary_note_partials_carry_the_same_key(monkeypatch, tmp_path):
    monkeypatch.setattr(summarizer, "PARTIALS_DIR", str(tmp_path))
    monkeypatch.setattr(summarizer, "_provider_configs", lambda: [{"name": "p", "model": "m", "base_url": "u", "api_key": "k"}])
    nt = tn.normalize_transcript(TRANSCRIPT, "v1")
    chunk = _chunk(nt)
    summarizer._save_partial(nt, chunk, {"key_points": ["x"]}, {"model": "m"})
    assert summarizer._load_partial(nt, chunk) == {"notes": {"key_points": ["x"]}}
    monkeypatch.setattr(summarizer, "CHUNK_NOTES_PROMPT_VERSION", summarizer.CHUNK_NOTES_PROMPT_VERSION + "-changed")
    assert summarizer._load_partial(nt, chunk) is None
    with open(partial_cache.path_for(nt.transcript_hash, "summary_notes", chunk.chunk_id), encoding="utf-8") as f:
        key = json.load(f)["key"]
    assert set(key) >= {"task_type", "transcript_hash", "normalization_version", "chunking_version", "chunk_id",
                        "start_character", "end_character", "prompt_version", "schema_version", "model_policy"}
