"""Tests for the summarizer's pure logic and provider-orchestration routing."""
import summarizer


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


def test_summarize_empty_transcript():
    assert summarizer.summarize_transcript("   ") == ""


def test_summarize_no_providers(monkeypatch):
    monkeypatch.setattr(summarizer, "_provider_configs", lambda: [])
    assert summarizer.summarize_transcript("some text") == ""


def _one_provider():
    return [{"name": "x", "model": "m", "base_url": "u", "api_key": "k"}]


def test_summarize_returns_summary(monkeypatch):
    monkeypatch.setattr(summarizer, "_provider_configs", _one_provider)
    monkeypatch.setattr(summarizer, "_call_provider", lambda p, t, title=None: "SUMMARY")
    assert summarizer.summarize_transcript("text", "title") == "SUMMARY"


def test_summarize_routes_sentinel(monkeypatch):
    monkeypatch.setattr(summarizer, "_provider_configs", _one_provider)
    monkeypatch.setattr(
        summarizer,
        "_call_provider",
        lambda p, t, title=None: summarizer.INSUFFICIENT_TRANSCRIPT_SENTINEL,
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

    def fake_call(p, t, title=None):
        calls["n"] += 1
        return "" if calls["n"] == 1 else "SECOND"

    monkeypatch.setattr(summarizer, "_call_provider", fake_call)
    assert summarizer.summarize_transcript("text") == "SECOND"
