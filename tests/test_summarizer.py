"""Tests for the summarizer's pure logic and provider-orchestration routing."""
from datetime import datetime, timedelta, timezone

import pytest

import gemini_quota

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


def test_no_transcript_truncation_remains():
    # The 120,000-char head/tail cut is gone for good: nothing in the module
    # trims, caps or marks a transcript. A long video's middle — the second
    # asset, the target, the condition — must reach the model.
    for name in ("_truncate_transcript", "_head_and_tail", "_fit_to_provider",
                 "LLM_MAX_TRANSCRIPT_CHARS", "GEMINI_MAX_INPUT_CHARS",
                 "GROQ_MAX_INPUT_CHARS", "TRANSCRIPT_TRUNCATION_MARKER"):
        assert not hasattr(summarizer, name), name
    import inspect
    assert "[transcript truncated]" not in inspect.getsource(summarizer)


def _ok_post(sent):
    class R:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "ok."}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 3}}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.append(json)
        return R()
    return fake_post


def test_long_transcript_is_sent_in_full_when_it_fits(monkeypatch):
    # 150,000 chars — over the old cap — with a unique marker in the MIDDLE.
    # With a 1M-token model the complete request fits, so the marker must be
    # in the request that leaves the process.
    sent = []
    monkeypatch.setattr(summarizer.requests, "post", _ok_post(sent))
    marker = "UNIQUE-MIDDLE-MARKER-XYZ"
    transcript = "a" * 75000 + " " + marker + " " + "b" * 75000
    provider = {"name": "gemini-3.7-flash", "model": "gemini-3.7-flash",
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "api_key": "k"}
    assert summarizer._call_provider(provider, transcript, "Title") == "ok."
    body = sent[0]["messages"][1]["content"]
    assert marker in body
    assert transcript in body                      # verbatim, nothing cut
    assert "truncated" not in body
    assert summarizer.LAST_CALL_TELEMETRY["transcript_chars"] == len(transcript)
    assert summarizer.LAST_CALL_TELEMETRY["reported_input_tokens"] == 12
    assert summarizer.LAST_CALL_TELEMETRY["finish_reason"] == "stop"


def test_max_tokens_is_an_output_cap_not_an_input_limit(monkeypatch):
    # A tiny max_tokens must not shrink what the model reads.
    sent = []
    monkeypatch.setattr(summarizer.requests, "post", _ok_post(sent))
    transcript = "word " * 30000
    provider = {"name": "gemini-3.7-flash", "model": "gemini-3.7-flash",
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "api_key": "k"}
    summarizer._call_provider(provider, transcript, max_tokens=64)
    assert sent[0]["max_tokens"] == 64
    assert transcript.strip() in sent[0]["messages"][1]["content"]


def test_request_over_model_capacity_is_reported_not_trimmed(monkeypatch):
    # A model with a small window gets INPUT_TOO_LARGE, and no request is sent.
    sent = []
    monkeypatch.setattr(summarizer.requests, "post", _ok_post(sent))
    monkeypatch.setenv("MODEL_CAPABILITIES_JSON",
                       '{"tiny": {"input_token_limit": 4000, "output_token_limit": 1000}}')
    import model_capabilities
    model_capabilities.reset_cache()
    provider = {"name": "tiny", "model": "tiny", "base_url": "http://x", "api_key": "k"}
    out = summarizer._call_provider(provider, "x " * 20000)
    assert out == summarizer.INPUT_TOO_LARGE_SENTINEL
    assert sent == []


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
    # Not "": an empty summary is permanent (watermark advances, video lost),
    # while truncation varies run to run and must stay retryable.
    assert summarizer._call_provider(_provider(), "transcript text") == summarizer.TRUNCATED_SENTINEL


def test_escalation_is_not_consumed_by_transient_failures(monkeypatch):
    # A 5xx and a truncation are different problems. Sharing one counter let an
    # unrelated blip eat the escalation budget, so the ceiling was never reached.
    caps, codes = [], []

    class R:
        def __init__(self):
            self.status_code = 503 if len(codes) == 0 else 200
            codes.append(self.status_code)

        def json(self):
            truncating = len(caps) <= 3
            return {"choices": [{
                "message": {"content": "cut" if truncating else "done."},
                "finish_reason": "length" if truncating else "stop",
            }]}

    def fake_post(url, headers=None, json=None, timeout=None):
        caps.append(json["max_tokens"])
        return R()

    monkeypatch.setattr(summarizer.requests, "post", fake_post)
    monkeypatch.setattr(summarizer.time, "sleep", lambda *_: None)
    summarizer._call_provider(_provider(), "transcript text")
    # One 503, then escalation still climbs its full doubling path. Asserted
    # relative to the starting budget rather than against the constant, because
    # the effective ceiling is derived from wherever the call starts (see
    # _call_provider) — pinning the constant just re-broke when the default
    # budget changed, without anything being wrong.
    assert caps[-1] == caps[0] * 2 ** summarizer.LLM_MAX_ESCALATIONS


def test_budget_at_or_above_ceiling_still_escalates(monkeypatch):
    # Configuring a budget above the ceiling used to skip escalation AND
    # discard the response — raising the budget made things strictly worse.
    caps = []

    class R:
        status_code = 200

        def json(self):
            return {"choices": [{
                "message": {"content": "cut" if len(caps) == 1 else "done."},
                "finish_reason": "length" if len(caps) == 1 else "stop",
            }]}

    def fake_post(url, headers=None, json=None, timeout=None):
        caps.append(json["max_tokens"])
        return R()

    monkeypatch.setattr(summarizer.requests, "post", fake_post)
    monkeypatch.setattr(summarizer.time, "sleep", lambda *_: None)
    big = summarizer.LLM_MAX_TOKENS_CEILING * 2
    assert summarizer._call_provider(_provider(), "t", max_tokens=big) == "done."
    assert caps[1] == big * 2


def test_was_truncated_survives_hostile_shapes():
    # Must never raise: it runs inside the per-channel handler, so an exception
    # would kill that channel's whole batch.
    for data in ({"choices": [None]}, {"choices": ["str"]}, {"choices": "x"},
                 {"choices": [{"finish_reason": None}]}, []):
        assert summarizer._was_truncated(data) is False


def test_each_fallback_model_is_judged_on_its_own_window(monkeypatch):
    # The chain: a small-window model first, a 1M-token model second. The
    # small one must neither receive a trimmed transcript nor block the wide
    # one; the summary comes from the model that can read all of it.
    sent = []
    monkeypatch.setattr(summarizer.requests, "post", _ok_post(sent))
    monkeypatch.setenv("MODEL_CAPABILITIES_JSON",
                       '{"small": {"input_token_limit": 4000, "output_token_limit": 1000}}')
    import model_capabilities
    model_capabilities.reset_cache()
    small = {"name": "small", "model": "small", "base_url": "http://x", "api_key": "k"}
    wide = {"name": "gemini-3.7-flash", "model": "gemini-3.7-flash",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "api_key": "k"}
    monkeypatch.setattr(summarizer, "_provider_configs", lambda: [small, wide])
    # Bound the chunked path on the small model so the test proves the
    # per-model decision rather than the chunker (covered separately).
    monkeypatch.setattr(summarizer, "MAX_SUMMARY_CHUNKS", 0)
    transcript = "x " * 20000
    assert summarizer.summarize_transcript(transcript) == "ok."
    assert [p["model"] for p in sent] == ["gemini-3.7-flash"]
    assert transcript.strip() in sent[0]["messages"][1]["content"]


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


def test_summarize_no_providers_defers_instead_of_failing(monkeypatch):
    # Not "": an empty summary is a permanent outcome — the caller marks the
    # video decided and it is never revisited. A misconfiguration (no key, or
    # every provider disallowed by SUMMARY_MODELS) must cost a delay, not the
    # video, and the run's stalled-delivery alert then makes it visible.
    monkeypatch.setattr(summarizer, "_provider_configs", lambda: [])
    assert summarizer.summarize_transcript("some text") == summarizer.QUOTA_EXHAUSTED_SENTINEL


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
    monkeypatch.delenv("GEMINI_FALLBACK_MODELS", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    providers = summarizer._provider_configs()
    # Strongest Flash for the summary, then models with their own daily quota.
    assert [p["model"] for p in providers] == ["gemini-3.7-flash", "gemini-3.6-flash"]


def test_gemini_no_duplicate_when_pinned_to_a_fallback(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")  # also the fallback
    monkeypatch.delenv("GEMINI_FALLBACK_MODELS", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    models = [p["model"] for p in summarizer._provider_configs()]
    # Pinned model runs first and is not called twice.
    assert models == ["gemini-3.6-flash"]


def test_gemini_fallbacks_are_configurable(monkeypatch):
    # The chain builder itself; whether those models are *allowed* to summarize
    # is the separate guarantee covered by TestSummaryModelGuarantee.
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.7-flash")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "alpha, beta")
    assert summarizer._gemini_models() == ["gemini-3.7-flash", "alpha", "beta"]


def test_each_gemini_model_is_its_own_quota_bucket(monkeypatch):
    # A 429 on one model must not write off the others: each carries its own
    # 20-requests-per-day free-tier budget, which is the whole point of the
    # chain. Provider entries are therefore named per model.
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_FALLBACK_MODELS", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    names = [p["name"] for p in summarizer._provider_configs()]
    assert names == ["gemini-3.7-flash", "gemini-3.6-flash"]
    assert len(set(names)) == len(names)


def test_transcription_models_never_share_the_summary_budget(monkeypatch):
    # transcript.py spends ~123k tokens per call on its own models; if one of
    # them appeared here too, a day of transcripts would eat the summary quota.
    import transcript

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_FALLBACK_MODELS", raising=False)
    monkeypatch.delenv("GEMINI_TRANSCRIPT_MODELS", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    summary_models = {p["model"] for p in summarizer._provider_configs()}
    assert summary_models.isdisjoint(transcript._gemini_transcript_models())


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



class TestSummaryModelGuarantee:
    """Only 3.7 or 3.6 may write a summary. Defaults express the intent; this
    guard is what makes it true when configuration disagrees."""

    def _env(self, monkeypatch):
        for var in ("GEMINI_MODEL", "GEMINI_FALLBACK_MODELS", "SUMMARY_MODELS",
                    "LLM_API_KEY", "LLM_BASE_URL", "GROQ_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "k")

    def test_default_roster_is_37_and_36(self, monkeypatch):
        self._env(monkeypatch)
        assert summarizer._summary_allowlist() == {"gemini-3.7-flash", "gemini-3.6-flash"}

    def test_groq_cannot_write_a_summary(self, monkeypatch):
        # A configured GROQ_API_KEY would otherwise hand summaries to
        # llama-3.3-70b once both Gemini models were spent.
        self._env(monkeypatch)
        monkeypatch.setenv("GROQ_API_KEY", "g")
        models = [p["model"] for p in summarizer._provider_configs()]
        assert models == ["gemini-3.7-flash", "gemini-3.6-flash"]

    def test_custom_llm_endpoint_cannot_write_a_summary(self, monkeypatch):
        # The custom provider is first in the chain, so without the guard it
        # would take precedence over 3.7 for every summary.
        self._env(monkeypatch)
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("LLM_BASE_URL", "http://somewhere")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
        models = [p["model"] for p in summarizer._provider_configs()]
        assert "gpt-4o-mini" not in models
        assert models == ["gemini-3.7-flash", "gemini-3.6-flash"]

    def test_a_stale_gemini_model_variable_is_overruled(self, monkeypatch):
        # The exact trap this repo was in: GEMINI_MODEL left over from an
        # earlier default silently wins over the code's chain.
        self._env(monkeypatch)
        monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
        models = [p["model"] for p in summarizer._provider_configs()]
        assert "gemini-2.5-flash" not in models
        assert models == ["gemini-3.6-flash"]

    def test_unset_actions_variable_keeps_the_guarantee(self, monkeypatch):
        # An unconfigured repo variable arrives as "" — it must mean "use the
        # default roster", not "allow anything".
        self._env(monkeypatch)
        monkeypatch.setenv("SUMMARY_MODELS", "")
        monkeypatch.setenv("GROQ_API_KEY", "g")
        assert [p["model"] for p in summarizer._provider_configs()] == [
            "gemini-3.7-flash", "gemini-3.6-flash",
        ]

    def test_star_opts_out_explicitly(self, monkeypatch):
        self._env(monkeypatch)
        monkeypatch.setenv("SUMMARY_MODELS", "*")
        monkeypatch.setenv("GROQ_API_KEY", "g")
        assert "llama-3.3-70b-versatile" in [p["model"] for p in summarizer._provider_configs()]

    def test_transcript_models_cannot_write_summaries_either(self, monkeypatch):
        import transcript

        self._env(monkeypatch)
        allowed = summarizer._summary_allowlist()
        assert allowed.isdisjoint(transcript._gemini_transcript_models())


class TestRateLimitHandling:
    """
    A 429 must not cost a model its whole Pacific day unless the API says so.

    The free tier enforces a per-minute limit (5 requests) and a per-day one
    (20 requests) behind the same status code. Treating the first like the
    second retires the preferred summary model over a speed bump, and every
    later run that day falls to the fallback.
    """

    class _Resp:
        def __init__(self, status_code, body="", payload=None):
            self.status_code = status_code
            self.text = body
            self.headers = {}
            self._payload = payload

        def json(self):
            return self._payload

    _OK = {"choices": [{"message": {"content": "a summary"}, "finish_reason": "stop"}]}

    def _gemini(self):
        # base_url is what marks a provider as metered against the Gemini
        # free-tier counter, so it has to be the real host.
        return {
            "name": "gemini-3.7-flash",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "api_key": "k",
            "model": "gemini-3.7-flash",
        }

    def _responses(self, monkeypatch, sequence):
        calls = []

        def fake_post(url, headers=None, json=None, timeout=None):
            calls.append(json["model"])
            return sequence[min(len(calls) - 1, len(sequence) - 1)]

        monkeypatch.setattr(summarizer.requests, "post", fake_post)
        monkeypatch.setattr(summarizer.time, "sleep", lambda s: None)
        return calls

    def test_transient_errors_do_not_starve_the_rate_limit_retries(self, monkeypatch):
        # The 2026-08-15 16:35 outage, reproduced. Two unrelated 503s consumed
        # the retry budget, so the 429 behind them was never retried once and
        # gemini-3.7-flash was retired for the day at 13 of its 20 requests.
        import gemini_quota
        from test_gemini_quota import PER_MINUTE_BODY

        calls = self._responses(monkeypatch, [
            self._Resp(503),
            self._Resp(503),
            self._Resp(429, PER_MINUTE_BODY),
            self._Resp(200, payload=self._OK),
        ])
        result = summarizer._call_provider(self._gemini(), "transcript text")

        assert result == "a summary"
        assert len(calls) == 4  # the 429 got a retry of its own
        assert gemini_quota.is_exhausted("gemini-3.7-flash") is False

    def test_a_per_day_429_still_retires_the_model(self, monkeypatch):
        # The limit this counter exists for: no amount of waiting brings the
        # day's budget back, so retrying would only burn run time.
        import gemini_quota
        from test_gemini_quota import PER_DAY_BODY

        calls = self._responses(monkeypatch, [self._Resp(429, PER_DAY_BODY)])
        result = summarizer._call_provider(self._gemini(), "transcript text")

        assert result == summarizer.QUOTA_EXHAUSTED_SENTINEL
        assert len(calls) == 1
        assert gemini_quota.is_exhausted("gemini-3.7-flash") is True

    def test_an_unnamed_429_costs_the_run_not_the_day(self, monkeypatch):
        # Nothing identified the quota, so the model is skipped for this run
        # and left usable by the next one. Worst case that costs one wasted
        # request an hour from now; the old behavior cost seven summaries.
        import gemini_quota

        calls = self._responses(monkeypatch, [self._Resp(429, "Too Many Requests")])
        result = summarizer._call_provider(self._gemini(), "transcript text")

        assert result == summarizer.QUOTA_EXHAUSTED_SENTINEL
        assert len(calls) == summarizer.LLM_MAX_RATE_LIMIT_RETRIES
        assert gemini_quota.is_exhausted("gemini-3.7-flash") is False

    def test_googles_retry_delay_is_honored_over_a_blind_backoff(self, monkeypatch):
        from test_gemini_quota import PER_MINUTE_BODY

        waits = []
        monkeypatch.setattr(summarizer.time, "sleep", lambda s: waits.append(s))
        monkeypatch.setattr(summarizer.requests, "post", lambda *a, **k: self._Resp(
            429, PER_MINUTE_BODY))
        summarizer._call_provider(self._gemini(), "transcript text")

        assert waits and waits[0] == 27  # RetryInfo, not LLM_RETRY_BACKOFF

    def test_a_wait_is_capped_so_one_call_cannot_eat_the_run(self, monkeypatch):
        body = '{"quotaId":"PerMinute","retryDelay":"9999s"}'
        waits = []
        monkeypatch.setattr(summarizer.time, "sleep", lambda s: waits.append(s))
        monkeypatch.setattr(summarizer.requests, "post", lambda *a, **k: self._Resp(429, body))
        summarizer._call_provider(self._gemini(), "transcript text")

        assert waits and max(waits) == summarizer.LLM_RETRY_AFTER_CAP


def _gemini_provider():
    return [{
        "name": "gemini-3.7-flash",
        "model": "gemini-3.7-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key": "k",
    }]


def test_a_write_off_from_an_earlier_run_does_not_cost_the_whole_day(monkeypatch):
    # The failure this guards: gemini-3.7-flash 429'd once at 00:41 Pacific and
    # every later run of the day skipped it on that verdict, with 17 of its 20
    # free requests unspent — until summaries fell off the end of the chain and
    # videos were deferred for "quota reached".
    monkeypatch.setattr(summarizer, "_provider_configs", _gemini_provider)
    monkeypatch.setattr(summarizer, "_call_provider", lambda p, t, title=None, **kw: "SUMMARY")
    gemini_quota.mark_exhausted("gemini-3.7-flash")

    # A fresh verdict is still honored — the cooling-off period is the point.
    assert summarizer.summarize_transcript("text") == summarizer.QUOTA_EXHAUSTED_SENTINEL

    # Once it has passed, a later run tries the model again rather than
    # inheriting a decision made hours ago.
    usage = gemini_quota.load_usage()
    entry = usage["spent"]["gemini-3.7-flash"]
    aged = datetime.now(timezone.utc) - timedelta(
        minutes=gemini_quota.recheck_delay_minutes(entry) + 1
    )
    entry["at"] = aged.isoformat()
    gemini_quota.save_usage(usage)
    assert summarizer.summarize_transcript("text") == "SUMMARY"


def test_skip_reason_distinguishes_this_run_from_an_older_verdict():
    # Both cases used to log "quota exhausted earlier this run", which read as
    # normal traffic shaping and hid a model idling on a stale verdict.
    provider = _gemini_provider()[0]
    gemini_quota.mark_exhausted("gemini-3.7-flash")
    assert "budget spent" in summarizer._skip_reason(provider)
    summarizer._EXHAUSTED_PROVIDERS.add(provider["name"])
    assert "earlier this run" in summarizer._skip_reason(provider)
