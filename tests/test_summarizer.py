"""Tests for the summarizer's pure logic and provider-orchestration routing."""
import pytest

import summarizer


@pytest.fixture(autouse=True)
def _clear_exhausted():
    # The exhausted-provider set is module-level run state; isolate each test.
    summarizer._EXHAUSTED_PROVIDERS.clear()
    yield
    summarizer._EXHAUSTED_PROVIDERS.clear()


class _Hdr:
    def __init__(self, headers):
        self.headers = headers


def test_truncate_short_unchanged():
    assert summarizer._truncate_transcript("short") == "short"


def test_truncate_long_appends_marker():
    long = "x" * (summarizer.LLM_MAX_TRANSCRIPT_CHARS + 100)
    out = summarizer._truncate_transcript(long)
    assert out.endswith(summarizer.TRANSCRIPT_TRUNCATION_MARKER)
    assert len(out) == summarizer.LLM_MAX_TRANSCRIPT_CHARS + len(
        summarizer.TRANSCRIPT_TRUNCATION_MARKER
    )


def test_build_user_message_with_title():
    msg = summarizer._build_user_message("BODY", "My Title")
    assert msg.startswith("Video title: My Title\n\n")
    assert "BODY" in msg


def test_build_user_message_without_title():
    msg = summarizer._build_user_message("BODY")
    assert not msg.startswith("Video title")
    assert "BODY" in msg


def test_extract_summary_ok():
    data = {"choices": [{"message": {"content": "  hi "}}]}
    assert summarizer._extract_summary(data) == "hi"


def test_extract_summary_bad_shapes():
    assert summarizer._extract_summary({}) == ""
    assert summarizer._extract_summary({"choices": []}) == ""
    assert summarizer._extract_summary(None) == ""


def test_was_truncated_detects_length_stop():
    assert summarizer._was_truncated({"choices": [{"finish_reason": "length"}]}) is True
    assert summarizer._was_truncated({"choices": [{"finish_reason": "MAX_TOKENS"}]}) is True
    assert summarizer._was_truncated({"choices": [{"finish_reason": "stop"}]}) is False
    # A missing/odd shape must not be read as truncation, or good summaries die.
    assert summarizer._was_truncated({"choices": [{}]}) is False
    assert summarizer._was_truncated({}) is False
    assert summarizer._was_truncated(None) is False


def _provider():
    return {"name": "p", "base_url": "http://x", "api_key": "k", "model": "m"}


def test_truncated_response_retries_with_a_bigger_budget(monkeypatch):
    # A response cut at the token cap ends mid-sentence but is a perfectly
    # ordinary 200, so without the finish_reason check it ships as a summary.
    caps = []

    class R:
        status_code = 200

        def __init__(self, payload):
            self._n = len(caps)

        def json(self):
            return {
                "choices": [{
                    "message": {"content": "half a summary that stops mid" if len(caps) == 1 else "complete."},
                    "finish_reason": "length" if len(caps) == 1 else "stop",
                }]
            }

    def fake_post(url, headers=None, json=None, timeout=None):
        caps.append(json["max_tokens"])
        return R(json)

    monkeypatch.setattr(summarizer.requests, "post", fake_post)
    monkeypatch.setattr(summarizer.time, "sleep", lambda *_: None)
    out = summarizer._call_provider(_provider(), "transcript text")
    assert out == "complete."
    assert caps[1] == caps[0] * 2      # budget doubled on the retry
    assert len(caps) == 2


def test_truncated_response_is_discarded_not_delivered(monkeypatch):
    # Still cut after escalating: drop it. Half a summary reads as a whole one
    # to the reader, and the video would be marked decided and never revisited.
    class R:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "cut off here and"},
                                 "finish_reason": "length"}]}

    monkeypatch.setattr(summarizer.requests, "post", lambda *a, **k: R())
    monkeypatch.setattr(summarizer.time, "sleep", lambda *_: None)
    assert summarizer._call_provider(_provider(), "transcript text") == ""


def test_untruncated_response_passes_through(monkeypatch):
    class R:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "a whole summary."},
                                 "finish_reason": "stop"}]}

    monkeypatch.setattr(summarizer.requests, "post", lambda *a, **k: R())
    assert summarizer._call_provider(_provider(), "transcript text") == "a whole summary."


def test_summarize_empty_transcript():
    assert summarizer.summarize_transcript("   ") == ""


def test_summarize_no_providers(monkeypatch):
    monkeypatch.setattr(summarizer, "_provider_configs", lambda: [])
    assert summarizer.summarize_transcript("some text") == ""


def _one_provider():
    return [{"name": "x", "model": "m", "base_url": "u", "api_key": "k"}]


def test_summarize_returns_summary(monkeypatch):
    monkeypatch.setattr(summarizer, "_provider_configs", _one_provider)
    monkeypatch.setattr(summarizer, "_call_provider", lambda p, t, title=None, **kw: "SUMMARY")
    assert summarizer.summarize_transcript("text", "title") == "SUMMARY"


def test_summarize_routes_sentinel(monkeypatch):
    monkeypatch.setattr(summarizer, "_provider_configs", _one_provider)
    monkeypatch.setattr(
        summarizer,
        "_call_provider",
        lambda p, t, title=None, **kw: summarizer.INSUFFICIENT_TRANSCRIPT_SENTINEL,
    )
    assert (
        summarizer.summarize_transcript("text")
        == summarizer.INSUFFICIENT_TRANSCRIPT_SENTINEL
    )


def test_summarize_falls_through_on_empty(monkeypatch):
    # First provider returns nothing, second returns a summary.
    monkeypatch.setattr(
        summarizer,
        "_provider_configs",
        lambda: [
            {"name": "a", "model": "m", "base_url": "u", "api_key": "k"},
            {"name": "b", "model": "m", "base_url": "u", "api_key": "k"},
        ],
    )
    calls = {"n": 0}

    def fake_call(p, t, title=None, **kw):
        calls["n"] += 1
        return "" if calls["n"] == 1 else "SECOND"

    monkeypatch.setattr(summarizer, "_call_provider", fake_call)
    assert summarizer.summarize_transcript("text") == "SECOND"


def test_parse_retry_after():
    assert summarizer._parse_retry_after(_Hdr({"Retry-After": "5"})) == 5
    assert summarizer._parse_retry_after(_Hdr({})) is None
    assert summarizer._parse_retry_after(_Hdr({"Retry-After": "notnum"})) is None
    # Capped so a hostile/huge value can't stall the run.
    assert (
        summarizer._parse_retry_after(_Hdr({"Retry-After": "99999"}))
        == summarizer.LLM_RETRY_AFTER_CAP
    )


def test_summarize_compact_uses_compact_prompt(monkeypatch):
    monkeypatch.setattr(summarizer, "_provider_configs", _one_provider)
    captured = {}

    def fake_call(p, t, title=None, system_prompt=None):
        captured["prompt"] = system_prompt
        return "S"

    monkeypatch.setattr(summarizer, "_call_provider", fake_call)
    assert summarizer.summarize_transcript("text", compact=True) == "S"
    assert captured["prompt"] == summarizer.COMPACT_SUMMARY_SYSTEM_PROMPT
    assert summarizer.summarize_transcript("text") == "S"
    assert captured["prompt"] == summarizer.SUMMARY_SYSTEM_PROMPT


def test_compact_prompt_keeps_sentinel_and_plaintext_rules():
    # The compact variant must keep the sentinel contract and plain-text rule,
    # or digest-channel refusals would be forwarded to Telegram as summaries.
    assert summarizer.INSUFFICIENT_TRANSCRIPT_SENTINEL in summarizer.COMPACT_SUMMARY_SYSTEM_PROMPT
    assert "plain text only" in summarizer.COMPACT_SUMMARY_SYSTEM_PROMPT


def test_summarize_quota_exhausted_routes_and_marks(monkeypatch):
    monkeypatch.setattr(summarizer, "_provider_configs", _one_provider)
    monkeypatch.setattr(
        summarizer, "_call_provider", lambda p, t, title=None, **kw: summarizer.QUOTA_EXHAUSTED_SENTINEL
    )
    assert summarizer.summarize_transcript("text") == summarizer.QUOTA_EXHAUSTED_SENTINEL
    assert "x" in summarizer._EXHAUSTED_PROVIDERS  # marked for the rest of the run


def test_summarize_skips_already_exhausted_provider(monkeypatch):
    summarizer._EXHAUSTED_PROVIDERS.add("x")
    monkeypatch.setattr(summarizer, "_provider_configs", _one_provider)
    calls = {"n": 0}

    def fake_call(p, t, title=None, **kw):
        calls["n"] += 1
        return "SUMMARY"

    monkeypatch.setattr(summarizer, "_call_provider", fake_call)
    # The only provider is already exhausted, so it's skipped without calling.
    assert summarizer.summarize_transcript("text") == summarizer.QUOTA_EXHAUSTED_SENTINEL
    assert calls["n"] == 0


def test_summarize_quota_then_success(monkeypatch):
    monkeypatch.setattr(
        summarizer,
        "_provider_configs",
        lambda: [
            {"name": "a", "model": "m", "base_url": "u", "api_key": "k"},
            {"name": "b", "model": "m", "base_url": "u", "api_key": "k"},
        ],
    )

    def fake_call(p, t, title=None, **kw):
        return summarizer.QUOTA_EXHAUSTED_SENTINEL if p["name"] == "a" else "OK"

    monkeypatch.setattr(summarizer, "_call_provider", fake_call)
    assert summarizer.summarize_transcript("text") == "OK"
    assert "a" in summarizer._EXHAUSTED_PROVIDERS


def test_json_mode_sets_response_format(monkeypatch):
    payloads = []

    class R:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        payloads.append(json)
        return R()

    monkeypatch.setattr(summarizer.requests, "post", fake_post)
    monkeypatch.setattr(summarizer, "_provider_configs", lambda: [
        {"name": "p", "base_url": "http://x", "api_key": "k", "model": "m"},
    ])
    monkeypatch.setattr(summarizer, "_EXHAUSTED_PROVIDERS", set())
    assert summarizer.complete("sys", "user", json_mode=True) == '{"ok": true}'
    assert payloads[0]["response_format"] == {"type": "json_object"}
    assert summarizer.complete("sys", "user") == '{"ok": true}'
    assert "response_format" not in payloads[1]


def test_provider_model_empty_env_falls_back(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODEL", "")  # unset repo variable arrives as ""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    providers = summarizer._provider_configs()
    # Preferred Gemini model first, proven 2.5-flash as same-key fallback.
    assert providers[0]["model"] == "gemini-3-flash-preview"
    assert providers[1]["model"] == "gemini-2.5-flash"


def test_gemini_no_duplicate_when_pinned_to_25_flash(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    providers = summarizer._provider_configs()
    assert [p["model"] for p in providers] == ["gemini-2.5-flash"]


def test_bad_preferred_model_falls_through_to_next_provider(monkeypatch):
    # Unknown model id -> non-transient 400 -> "" -> chain tries next entry.
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json["model"])
        r = type("R", (), {})()
        r.status_code = 400 if json["model"] == "bad-model" else 200
        r.text = "model not found"
        r.json = lambda: {"choices": [{"message": {"content": "SUMMARY"}}]}
        return r

    monkeypatch.setattr(summarizer.requests, "post", fake_post)
    monkeypatch.setattr(summarizer, "_provider_configs", lambda: [
        {"name": "gemini", "base_url": "http://x", "api_key": "k", "model": "bad-model"},
        {"name": "gemini-2.5-flash", "base_url": "http://x", "api_key": "k", "model": "gemini-2.5-flash"},
    ])
    monkeypatch.setattr(summarizer, "_EXHAUSTED_PROVIDERS", set())
    assert summarizer.summarize_transcript("some transcript") == "SUMMARY"
    assert calls == ["bad-model", "gemini-2.5-flash"]
