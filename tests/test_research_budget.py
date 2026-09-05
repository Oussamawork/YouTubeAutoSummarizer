"""Hardening item 1: research extraction never spends the requests a later
video's summary needs — deferred past the delivery loop and held behind a
configurable summary reserve."""
import json

import pytest

import gemini_quota
import research_backfill
import research_budget
import research_state
import scraper
import signals
import summarizer
import transcript_normalize as tn
import transcript_store

GEMINI = "https://generativelanguage.googleapis.com/v1beta/openai"
SENTENCE = ("I expect Nvidia to fall over the next three months, but I remain bullish over five years, "
            "and I'd buy Palantir under $20 while Micron stays the cheapest memory name. ")
LONG = SENTENCE * 24          # ~3.3k chars: several research chunks at a small chunk size
SHORT = SENTENCE


class _Resp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text

    def json(self):
        return self._payload


def _kind(system):
    if "OUTPUT ENVELOPE" in system:
        return "combined"
    if "EXTRACTION for a research dataset" in system:
        return "claims"
    return "summary"


def _router(calls):
    """A fake provider whose combined envelope is always cut off
    (finish_reason=length), while summary-only and claims calls answer."""
    def post(url, headers=None, json=None, timeout=None):
        kind = _kind(json["messages"][0]["content"])
        calls.append(kind)
        if kind == "combined":
            return _Resp(200, {"choices": [{"message": {"content": '{"summary": "x", "claims": [{'},
                                            "finish_reason": "length"}]})
        if kind == "claims":
            return _Resp(200, {"choices": [{"message": {"content": json_mod.dumps({"claims": [], "extraction_metadata": {}})},
                                            "finish_reason": "stop"}]})
        return _Resp(200, {"choices": [{"message": {"content": "TL;DR\n\n• Nvidia may fall near term"},
                                        "finish_reason": "stop"}]})
    return post


json_mod = json


@pytest.fixture
def two_videos(tmp_path, monkeypatch):
    monkeypatch.setattr(research_state, "RESEARCH_DIR", str(tmp_path / "research"))
    monkeypatch.setattr(transcript_store, "TRANSCRIPTS_DIR", str(tmp_path / "transcripts"))
    monkeypatch.setattr(summarizer, "PARTIALS_DIR", str(tmp_path / "partials"))
    monkeypatch.setattr(summarizer, "_provider_configs",
                        lambda: [{"name": "gemini-3.7-flash", "base_url": GEMINI, "api_key": "k",
                                  "model": "gemini-3.7-flash"}])
    monkeypatch.setattr(summarizer.time, "sleep", lambda *_: None)
    monkeypatch.setattr(signals, "RESEARCH_CHUNK_TOKENS", 500)
    summarizer._EXHAUSTED_PROVIDERS.clear()
    transcripts = {"v1": LONG, "v2": SHORT}
    monkeypatch.setattr(scraper, "get_transcript_from_video",
                        lambda url: {"transcript": transcripts[url.rsplit("=", 1)[-1]], "reason": "ok"})
    rows = []
    monkeypatch.setattr(scraper, "append_jsonl", lambda path, rec: rows.append((path, rec)) or True)
    for name, value in {"YOUTUBE_API_KEY": "yt", "TELEGRAM_TOKEN": "tok", "TELEGRAM_CHANNEL_ID": "premium",
                        "MARKET_SIGNALS": "true"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("DAILY_DIGEST", raising=False)
    videos = [{"video_id": vid, "channel_name": "Chan", "video_title": f"Video {vid}",
               "video_url": f"https://www.youtube.com/watch?v={vid}", "published_at": published}
              for vid, published in (("v2", "2026-07-02T14:00:00+00:00"), ("v1", "2026-07-01T14:00:00+00:00"))]
    monkeypatch.setattr(scraper, "get_recent_videos", lambda key, cid: videos)
    monkeypatch.setattr(scraper, "read_channels", lambda path: [{"channel_id": "c1", "digest": False, "max_per_run": 0}])
    # A fresh dedup state per main() call, so a test can run the loop twice.
    monkeypatch.setattr(scraper, "load_state", lambda path: {
        "channels": {"c1": {"last_video_id": "v0", "last_published": "2026-06-01T00:00:00+00:00"}}, "pending": {}})
    monkeypatch.setattr(scraper, "save_state", lambda path, st: True)
    sent = []
    monkeypatch.setattr(scraper, "send_telegram_message", lambda *args: sent.append(args[3]) or True)
    yield {"rows": rows, "sent": sent}
    summarizer._EXHAUSTED_PROVIDERS.clear()


def _run(monkeypatch, calls, cap, reserve):
    monkeypatch.setattr(gemini_quota, "GEMINI_REQUESTS_PER_DAY", cap)
    monkeypatch.setattr(research_budget, "SUMMARY_REQUEST_RESERVE", reserve)
    monkeypatch.setattr(summarizer.requests, "post", _router(calls))
    scraper.main()


def test_research_runs_only_after_every_video_is_delivered(two_videos, monkeypatch):
    calls = []
    _run(monkeypatch, calls, cap=40, reserve=4)
    assert sorted(two_videos["sent"]) == ["Video v1", "Video v2"]
    # Every claims request comes after the last delivery request: the first
    # video's (multi-chunk) extraction waited for the second video's summary.
    last_delivery = max(i for i, k in enumerate(calls) if k != "claims")
    first_claims = min(i for i, k in enumerate(calls) if k == "claims")
    assert first_claims > last_delivery
    assert calls.count("claims") >= 3  # v1 alone needed several chunks
    st = research_state.load_state()
    assert research_state.get(st, "v1")["delivery_status"] == "sent"
    assert research_state.get(st, "v1")["research_status"] in ("no_claims_found", "needs_review", "complete")
    assert research_state.get(st, "v2")["delivery_status"] == "sent"


def test_large_extraction_for_the_first_video_cannot_starve_the_second_summary(two_videos, monkeypatch, tmp_path):
    # Measure what delivering both videos costs, then give the day exactly
    # that plus two spare requests. The first video's chunked extraction
    # alone needs more than the spare, so run immediately it would have
    # consumed the second video's summary requests.
    probe = []
    _run(monkeypatch, probe, cap=40, reserve=4)
    delivery_calls = [k for k in probe if k != "claims"]
    per_video = len(delivery_calls) // 2
    nt = tn.normalize_transcript(LONG, "v1")
    estimate = research_budget.estimate_research_requests(nt, prefer_chunked=True)
    assert estimate >= 3
    cap = len(delivery_calls) + 2
    assert cap - per_video - estimate < per_video, "the setup would not have starved v2"

    two_videos["sent"].clear()
    two_videos["rows"].clear()
    summarizer._EXHAUSTED_PROVIDERS.clear()
    monkeypatch.setattr(research_state, "RESEARCH_DIR", str(tmp_path / "research-second-day"))
    gemini_quota.save_usage({"day": gemini_quota.quota_day().isoformat(), "models": {}, "spent": {}})
    calls = []
    _run(monkeypatch, calls, cap=cap, reserve=4)
    # Both summaries went out; no claims request was made at all, because
    # after delivery fewer than reserve + estimate requests were left.
    assert sorted(two_videos["sent"]) == ["Video v1", "Video v2"]
    assert calls.count("claims") == 0
    st = research_state.load_state()
    for vid in ("v1", "v2"):
        entry = research_state.get(st, vid)
        assert entry["delivery_status"] == "sent"
        assert entry["research_status"] == "quota_deferred"
        assert entry["failure_reason"] == "summary_reserve_protected"
        assert entry["attempt_count"] == 0
        assert entry["budget_check"]["remaining"] - entry["budget_check"]["estimated"] < 4
    assert set(research_state.retry_candidates(st)) == {"v1", "v2"}
    # The compatibility rows say so too, once each.
    signal_rows = [r for p, r in two_videos["rows"] if p == scraper.SIGNALS_FILE]
    assert [r["research_status"] for r in signal_rows] == ["quota_deferred", "quota_deferred"]


def test_reserve_check_arithmetic(monkeypatch, tmp_path):
    providers = [{"name": "gemini-3.7-flash", "base_url": GEMINI, "api_key": "k", "model": "gemini-3.7-flash"},
                 {"name": "gemini-3.6-flash", "base_url": GEMINI, "api_key": "k", "model": "gemini-3.6-flash"}]
    monkeypatch.setattr(summarizer, "_provider_configs", lambda: providers)
    monkeypatch.setattr(gemini_quota, "GEMINI_REQUESTS_PER_DAY", 10)
    usage = gemini_quota.load_usage()
    usage["models"] = {"gemini-3.7-flash": 8, "gemini-3.6-flash": 5}
    gemini_quota.save_usage(usage)
    nt = tn.normalize_transcript(SHORT, "v")
    assert research_budget.remaining_requests() == 7
    assert research_budget.estimate_research_requests(nt) == 1
    ok = research_budget.check(nt, reserve=4)
    assert ok["allowed"] and ok["reason"] == "within_budget" and ok["remaining"] == 7
    assert research_budget.check(nt, reserve=7)["allowed"] is False
    assert research_budget.check(nt, reserve=4, workload=3)["allowed"] is False
    # An exhausted model contributes nothing; a written-off one neither.
    usage["models"]["gemini-3.6-flash"] = 10
    gemini_quota.save_usage(usage)
    assert research_budget.remaining_requests() == 2
    # An unmetered provider means no daily cap binds.
    monkeypatch.setattr(summarizer, "_provider_configs",
                        lambda: providers + [{"name": "groq", "base_url": "https://api.groq.com/openai/v1",
                                              "api_key": "k", "model": "llama"}])
    assert research_budget.remaining_requests() is None
    assert research_budget.check(nt, reserve=100)["allowed"]


def test_retry_job_applies_the_same_reserve(monkeypatch, tmp_path):
    monkeypatch.setattr(research_state, "RESEARCH_DIR", str(tmp_path / "research"))
    monkeypatch.setattr(transcript_store, "TRANSCRIPTS_DIR", str(tmp_path / "transcripts"))
    transcript_store.store_transcript({"video_id": "v1", "channel_name": "Chan", "video_title": "T",
                                       "published_at": "2026-07-01T00:00:00+00:00"}, SHORT, "supadata", "ok")
    state = research_state.load_state()
    research_state.update(state, "v1", research_status="failed_retryable", delivery_status="sent")
    called = []
    monkeypatch.setattr(research_budget, "check", lambda nt, **kw: {"allowed": False, "reason": "summary_reserve_protected",
                                                                    "remaining": 3, "estimated": 1, "reserve": 4, "workload": 0})
    assert research_backfill.process_video(state, "v1", lambda nt, ctx: called.append(1)) == "quota_deferred"
    assert not called
    entry = research_state.get(state, "v1")
    assert entry["failure_reason"] == "summary_reserve_protected" and entry["attempt_count"] == 0
    assert "v1" in research_state.retry_candidates(state)
