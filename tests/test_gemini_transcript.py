"""Tests for the Gemini video-transcript source (no network).

Gemini is the source that keeps the pipeline alive when Supadata's credits are
spent, so the cases that matter are: it is only used when the cheaper source
came back empty, a model that is out of daily quota rotates to the next one,
and "every model is out" defers the video rather than reporting it as having no
transcript (which would eventually write it off).
"""
import pytest

import transcript


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _payload(text):
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


LONG = "word " * 500  # comfortably over the minimum-length floor


@pytest.fixture
def gemini_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_TRANSCRIPT_MODELS", raising=False)


class TestModelList:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("GEMINI_TRANSCRIPT_MODELS", raising=False)
        assert transcript._gemini_transcript_models() == [
            "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3-flash-preview",
        ]

    def test_override(self, monkeypatch):
        monkeypatch.setenv("GEMINI_TRANSCRIPT_MODELS", "a, b ,c")
        assert transcript._gemini_transcript_models() == ["a", "b", "c"]

    def test_unset_actions_variable_falls_back(self, monkeypatch):
        # An unconfigured GitHub Actions variable arrives as "", which must not
        # become an empty model list (that would silently disable the source).
        monkeypatch.setenv("GEMINI_TRANSCRIPT_MODELS", "")
        assert transcript._gemini_transcript_models()[0] == "gemini-3.6-flash"


class TestTextExtraction:
    def test_joins_parts(self):
        data = {"candidates": [{"content": {"parts": [{"text": "a "}, {"text": "b"}]}}]}
        assert transcript._gemini_text_from_payload(data) == "a b"

    def test_missing_candidates(self):
        assert transcript._gemini_text_from_payload({}) == ""

    def test_not_a_dict(self):
        assert transcript._gemini_text_from_payload(None) == ""

    def test_blocked_response_without_parts(self):
        data = {"candidates": [{"finishReason": "SAFETY"}]}
        assert transcript._gemini_text_from_payload(data) == ""


class TestFetchGeminiTranscript:
    def test_first_model_serves(self, monkeypatch, gemini_key):
        calls = []

        def fake_post(url, **kwargs):
            calls.append(url)
            return FakeResponse(payload=_payload(LONG))

        monkeypatch.setattr(transcript.requests, "post", fake_post)
        text, exhausted, reason = transcript._fetch_gemini_transcript("vid00000001")
        assert text.startswith("word")
        assert (exhausted, reason) == (False, "gemini_ok")
        assert len(calls) == 1 and "gemini-3.6-flash" in calls[0]

    def test_rotates_past_a_model_out_of_quota(self, monkeypatch, gemini_key):
        seen = []

        def fake_post(url, **kwargs):
            seen.append(url)
            if "gemini-3.6-flash" in url:
                return FakeResponse(status_code=429, text="RESOURCE_EXHAUSTED")
            return FakeResponse(payload=_payload(LONG))

        monkeypatch.setattr(transcript.requests, "post", fake_post)
        text, exhausted, reason = transcript._fetch_gemini_transcript("vid00000001")
        assert text and reason == "gemini_ok" and exhausted is False
        assert len(seen) == 2 and "gemini-3.5-flash" in seen[1]

    def test_all_models_out_of_quota_defers(self, monkeypatch, gemini_key):
        monkeypatch.setattr(
            transcript.requests, "post",
            lambda url, **kwargs: FakeResponse(status_code=429, text="quota"),
        )
        text, exhausted, reason = transcript._fetch_gemini_transcript("vid00000001")
        # Deferral, not "this video has no transcript" — the distinction is what
        # keeps a video from eventually being written off during an outage.
        assert (text, exhausted, reason) == ("", True, "gemini_quota")

    def test_short_answer_is_not_accepted_as_a_transcript(self, monkeypatch, gemini_key):
        # A model that describes the video instead of transcribing it must not
        # have that paragraph summarized and shipped as if it were the video.
        monkeypatch.setattr(
            transcript.requests, "post",
            lambda url, **kwargs: FakeResponse(payload=_payload("This video is about stocks.")),
        )
        text, exhausted, reason = transcript._fetch_gemini_transcript("vid00000001")
        assert text == "" and exhausted is False and reason == "gemini_too_short"

    def test_no_key_is_skipped_quietly(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        called = []
        monkeypatch.setattr(transcript.requests, "post",
                            lambda *a, **k: called.append(1))
        assert transcript._fetch_gemini_transcript("vid00000001") == ("", False, "no_gemini_key")
        assert not called

    def test_http_error_does_not_defer(self, monkeypatch, gemini_key):
        # A 400 is a bug in our request, not an exhausted budget: it must not
        # masquerade as "try again later" forever.
        monkeypatch.setattr(
            transcript.requests, "post",
            lambda url, **kwargs: FakeResponse(status_code=400, text="bad request"),
        )
        _, exhausted, reason = transcript._fetch_gemini_transcript("vid00000001")
        assert exhausted is False and reason == "gemini_http_400"


class TestSourceOrder:
    def test_supadata_wins_and_gemini_is_not_called(self, monkeypatch, gemini_key):
        monkeypatch.setattr(transcript, "_fetch_supadata",
                            lambda vid: ("supadata text", False, "ok"))
        called = []
        monkeypatch.setattr(transcript, "_fetch_gemini_transcript",
                            lambda vid: called.append(1) or ("", False, "x"))
        result = transcript.get_transcript_from_video("https://youtu.be/dQw4w9WgXcQ")
        assert result["transcript"] == "supadata text"
        # Supadata costs ~9k tokens against Gemini's ~123k, so it must stay first.
        assert not called

    def test_gemini_used_when_supadata_is_out_of_credits(self, monkeypatch, gemini_key):
        monkeypatch.setattr(transcript, "_fetch_supadata",
                            lambda vid: ("", True, "no_credits"))
        monkeypatch.setattr(transcript, "_fetch_gemini_transcript",
                            lambda vid: (LONG, False, "gemini_ok"))
        result = transcript.get_transcript_from_video("https://youtu.be/dQw4w9WgXcQ")
        assert result["transcript"] == LONG
        assert result["reason"] == "gemini_ok"
        # A transcript arrived, so nothing is deferred even though Supadata was
        # out of credits.
        assert result["budget_exhausted"] is False

    def test_both_budgets_spent_defers(self, monkeypatch, gemini_key):
        monkeypatch.setattr(transcript, "_fetch_supadata",
                            lambda vid: ("", True, "no_credits"))
        monkeypatch.setattr(transcript, "_fetch_gemini_transcript",
                            lambda vid: ("", True, "gemini_quota"))
        monkeypatch.setattr(transcript, "_fetch_youtube_transcript_api", lambda vid: "")
        result = transcript.get_transcript_from_video("https://youtu.be/dQw4w9WgXcQ")
        assert result["budget_exhausted"] is True
        assert result["reason"] == "gemini_quota"

    def test_youtube_api_still_the_last_resort(self, monkeypatch, gemini_key):
        monkeypatch.setattr(transcript, "_fetch_supadata", lambda vid: ("", False, "empty_content"))
        monkeypatch.setattr(transcript, "_fetch_gemini_transcript",
                            lambda vid: ("", False, "gemini_too_short"))
        monkeypatch.setattr(transcript, "_fetch_youtube_transcript_api", lambda vid: "local text")
        result = transcript.get_transcript_from_video("https://youtu.be/dQw4w9WgXcQ")
        assert result["transcript"] == "local text"
        assert result["reason"] == "fallback_ok"
        assert result["budget_exhausted"] is False
