"""Complete-coverage chunked summarization and quota pre-checks."""
import json

import gemini_quota
import summarizer

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"


def _small(name="small", limit=4000):
    return {"name": name, "model": name, "base_url": "http://x", "api_key": "k"}


def _install(monkeypatch, responder, limit=8000, model="small"):
    monkeypatch.setenv("MODEL_CAPABILITIES_JSON",
                       json.dumps({model: {"input_token_limit": limit, "output_token_limit": 2000}}))
    import model_capabilities
    model_capabilities.reset_cache()
    sent = []

    class R:
        status_code = 200

        def __init__(self, content):
            self._c = content

        def json(self):
            return {"choices": [{"message": {"content": self._c}, "finish_reason": "stop"}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.append(json)
        return R(responder(json))

    monkeypatch.setattr(summarizer.requests, "post", fake_post)
    monkeypatch.setattr(summarizer.time, "sleep", lambda *_: None)
    return sent


def _responder(json_payload):
    user = json_payload["messages"][1]["content"]
    if "Transcript part" in user:
        return json.dumps({"key_points": [user[-40:]], "assets": [], "positions": [], "other": []})
    return "FINAL SUMMARY"


def test_over_limit_transcript_is_chunked_with_full_coverage_then_merged(monkeypatch):
    sent = _install(monkeypatch, _responder)
    text = " ".join(f"Sentence {i} says Nvidia goes to ${i} by 2030." for i in range(1200))
    marker = "Sentence 600 says Nvidia goes to $600 by 2030."
    out = summarizer._summarize_chunked(_small(), text, "Title", False, "hash1")
    assert out == "FINAL SUMMARY"
    chunk_calls = [p for p in sent if "Transcript part" in p["messages"][1]["content"]]
    merge_calls = [p for p in sent if "ORDERED set of notes" in p["messages"][1]["content"]]
    assert len(chunk_calls) >= 2 and len(merge_calls) == 1
    # Every character of the transcript went out in some chunk, in order.
    bodies = [p["messages"][1]["content"].split(":\n\n", 1)[1] for p in chunk_calls]
    assert any(marker in b for b in bodies)
    assert "".join(bodies).count("Sentence 0 says") >= 1 and "Sentence 1199 says" in bodies[-1]
    # All parts reached the merge, in sequence.
    merged = merge_calls[0]["messages"][1]["content"]
    for i in range(1, len(chunk_calls) + 1):
        assert f"--- Part {i} of {len(chunk_calls)} ---" in merged
    assert summarizer.LAST_CALL_TELEMETRY["coverage_status"] == "chunked_full"
    assert summarizer.LAST_CALL_TELEMETRY["chunks_failed"] == 0


def test_chunk_partials_persist_and_resume_without_rerunning(monkeypatch):
    text = " ".join(f"Sentence {i} says Nvidia goes to ${i} by 2030." for i in range(1200))
    calls = {"n": 0, "first_run": True}

    def flaky(payload):
        user = payload["messages"][1]["content"]
        if "Transcript part" in user:
            calls["n"] += 1
            if calls["first_run"] and "Transcript part 2 of" in user:
                return ""  # the second chunk fails on the first run
            return json.dumps({"key_points": ["k"], "assets": [], "positions": [], "other": []})
        return "FINAL"

    sent = _install(monkeypatch, flaky)
    assert summarizer._summarize_chunked(_small(), text, None, False, "hash2") == ""
    assert calls["n"] == 2                       # stopped at the failure, chunk 1 saved
    # Second run: chunk 1 is served from disk; only the missing ones run.
    calls.update(n=0, first_run=False)
    sent.clear()
    assert summarizer._summarize_chunked(_small(), text, None, False, "hash2") == "FINAL"
    parts = [p["messages"][1]["content"] for p in sent if "Transcript part" in p["messages"][1]["content"]]
    assert parts and not any("Transcript part 1 of" in p for p in parts)
    assert calls["n"] == len(parts)


def test_chunked_summary_never_merges_partial_coverage(monkeypatch):
    text = " ".join(f"Sentence {i} says Nvidia goes to ${i} by 2030." for i in range(1200))

    def truncating(payload):
        return json.dumps({"key_points": []})

    sent = _install(monkeypatch, truncating)
    # Make the second chunk hit the quota sentinel via a 429 per-day body.
    from test_gemini_quota import PER_DAY_BODY
    real_post = summarizer.requests.post
    seen = {"n": 0}

    class R429:
        status_code = 429
        text = PER_DAY_BODY
        headers = {}

    def post(url, headers=None, json=None, timeout=None):
        if "Transcript part" in json["messages"][1]["content"]:
            seen["n"] += 1
            if seen["n"] == 2:
                return R429()
        return real_post(url, headers=headers, json=json, timeout=timeout)

    monkeypatch.setattr(summarizer.requests, "post", post)
    out = summarizer._summarize_chunked(_small(), text, None, False, None)
    assert out == summarizer.QUOTA_EXHAUSTED_SENTINEL
    assert not any("ORDERED set of notes" in p["messages"][1]["content"] for p in sent)


def test_multi_call_summary_is_not_started_without_enough_quota(monkeypatch):
    # A Gemini-metered provider with 1 request left cannot finish a 3+ call
    # operation; it must defer up front instead of spending the request.
    model = "gemini-3.7-flash"
    provider = {"name": model, "model": model, "base_url": GEMINI_BASE, "api_key": "k"}
    sent = _install(monkeypatch, _responder, limit=8000, model=model)
    monkeypatch.setattr(gemini_quota, "GEMINI_REQUESTS_PER_DAY", 2)
    gemini_quota.record(model)  # 1 of 2 spent
    text = " ".join(f"Sentence {i} says Nvidia goes to ${i} by 2030." for i in range(1200))
    assert summarizer._summarize_chunked(provider, text, None, False, None) == summarizer.QUOTA_EXHAUSTED_SENTINEL
    assert sent == []


def test_summarize_transcript_routes_too_large_to_chunking(monkeypatch):
    sent = _install(monkeypatch, _responder)
    monkeypatch.setattr(summarizer, "_provider_configs", lambda: [_small()])
    text = " ".join(f"Sentence {i} says Nvidia goes to ${i} by 2030." for i in range(1200))
    assert summarizer.summarize_transcript(text, "T", cache_key="h3") == "FINAL SUMMARY"
    assert all("[transcript truncated]" not in p["messages"][1]["content"] for p in sent)


def test_complete_reports_too_large_when_no_model_can_read_it(monkeypatch):
    sent = _install(monkeypatch, _responder)
    monkeypatch.setattr(summarizer, "_provider_configs", lambda: [_small()])
    out = summarizer.complete("sys", "x " * 20000, json_mode=True)
    assert out == summarizer.INPUT_TOO_LARGE_SENTINEL and sent == []


def test_parse_json_object_tolerates_fences():
    assert summarizer._parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert summarizer._parse_json_object("nope") is None
