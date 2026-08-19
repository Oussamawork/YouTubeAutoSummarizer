"""Tests for the Gemini video-transcript source (no network).

Gemini is the source that keeps the pipeline alive when Supadata's credits are
spent, so the cases that matter are: it is only used when the cheaper source
came back empty, a model that is out of daily quota rotates to the next one,
and "every model is out" defers the video rather than reporting it as having no
transcript (which would eventually write it off).
"""
import pytest

import gemini_quota
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
            "gemini-3.5-flash", "gemini-3-flash-preview", "gemini-2.5-flash",
        ]

    def test_override(self, monkeypatch):
        monkeypatch.setenv("GEMINI_TRANSCRIPT_MODELS", "a, b ,c")
        assert transcript._gemini_transcript_models() == ["a", "b", "c"]

    def test_unset_actions_variable_falls_back(self, monkeypatch):
        # An unconfigured GitHub Actions variable arrives as "", which must not
        # become an empty model list (that would silently disable the source).
        monkeypatch.setenv("GEMINI_TRANSCRIPT_MODELS", "")
        assert transcript._gemini_transcript_models()[0] == "gemini-3.5-flash"


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
        assert len(calls) == 1 and "gemini-3.5-flash" in calls[0]

    def test_rotates_past_a_model_out_of_quota(self, monkeypatch, gemini_key):
        seen = []

        def fake_post(url, **kwargs):
            seen.append(url)
            if "gemini-3.5-flash" in url:
                return FakeResponse(status_code=429, text="RESOURCE_EXHAUSTED")
            return FakeResponse(payload=_payload(LONG))

        monkeypatch.setattr(transcript.requests, "post", fake_post)
        text, exhausted, reason = transcript._fetch_gemini_transcript("vid00000001")
        assert text and reason == "gemini_ok" and exhausted is False
        assert len(seen) == 2 and "gemini-3-flash-preview" in seen[1]

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

    def test_a_capped_model_is_not_asked_again(self, monkeypatch, gemini_key):
        # With ~9 videos in a run and 8 runs a day, re-asking a model that
        # already answered "out of quota" would waste a round trip per video
        # per model — and the cap lasts until the Pacific midnight reset, so the
        # count has to outlive the process.
        calls = []

        def fake_post(url, **kwargs):
            calls.append(url)
            return FakeResponse(status_code=429, text="quota")

        monkeypatch.setattr(transcript.requests, "post", fake_post)
        first = transcript._fetch_gemini_transcript("vid00000001")
        second = transcript._fetch_gemini_transcript("vid00000002")
        assert first == second == ("", True, "gemini_quota")
        # Every model asked once on the first video, none on the second.
        assert len(calls) == len(transcript._gemini_transcript_models())

    def _lapsed_write_offs(self):
        """
        Every transcript model written off by the API, with the verdict aged out
        so this run re-probes it. This is the state the committed counter file
        migrates into, so it is what the first run after a deploy actually sees.
        """
        for model in transcript._gemini_transcript_models():
            gemini_quota.mark_exhausted(model)
        usage = gemini_quota.load_usage()
        for entry in usage["spent"].values():
            entry["at"] = None
        gemini_quota.save_usage(usage)

    def test_a_lapsed_write_off_whose_re_probe_fails_still_defers(self, monkeypatch, gemini_key):
        # The models are out of budget; the video is fine. If the re-probe
        # failing for an unrelated reason made this look like "no transcript",
        # scraper.py would spend one of the video's give-up attempts on a quota
        # outage — and eight of those write a good video off for good.
        self._lapsed_write_offs()
        monkeypatch.setattr(
            transcript.requests, "post",
            lambda url, **kwargs: FakeResponse(status_code=400, text="bad request"),
        )
        text, exhausted, reason = transcript._fetch_gemini_transcript("vid00000001")
        assert (text, exhausted, reason) == ("", True, "gemini_quota")

    def test_a_failed_re_probe_is_not_repeated_for_every_video(self, monkeypatch, gemini_key):
        # One lapsed verdict must cost one probe per run, not one per video:
        # a failed probe leaves the persisted flag untouched, so without an
        # in-run guard the whole rotation is re-asked for every video.
        self._lapsed_write_offs()
        calls = []

        def fake_post(url, **kwargs):
            calls.append(url)
            return FakeResponse(status_code=400, text="bad request")

        monkeypatch.setattr(transcript.requests, "post", fake_post)
        first = transcript._fetch_gemini_transcript("vid00000001")
        second = transcript._fetch_gemini_transcript("vid00000002")
        assert first == second == ("", True, "gemini_quota")
        assert len(calls) == len(transcript._gemini_transcript_models())

    def test_a_re_probe_that_succeeds_clears_the_verdict(self, monkeypatch, gemini_key):
        # The other direction: the API answering is proof the verdict lapsed,
        # so the model must go back to serving rather than stay written off.
        self._lapsed_write_offs()
        monkeypatch.setattr(
            transcript.requests, "post",
            lambda url, **kwargs: FakeResponse(payload=_payload(LONG)),
        )
        text, exhausted, reason = transcript._fetch_gemini_transcript("vid00000001")
        assert text and exhausted is False and reason == "gemini_ok"
        first = transcript._gemini_transcript_models()[0]
        assert gemini_quota.written_off(first) is False
        assert gemini_quota.used(first) == 1

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


class TestRateLimitKind:
    """
    Only a per-day 429 may spend the day's budget.

    The rotation makes this cheap to get right: a model that is merely
    rate-limited for the minute is skipped in favor of the next one, and is
    still available to the next run an hour later.
    """

    def test_a_per_day_429_writes_the_model_off_for_the_day(self, monkeypatch, gemini_key):
        import gemini_quota
        from test_gemini_quota import PER_DAY_BODY

        monkeypatch.setattr(
            transcript.requests, "post",
            lambda url, **kwargs: FakeResponse(status_code=429, text=PER_DAY_BODY),
        )
        transcript._fetch_gemini_transcript("vid00000001")
        assert gemini_quota.is_exhausted("gemini-3.5-flash") is True

    def test_a_per_minute_429_leaves_the_day_intact(self, monkeypatch, gemini_key):
        import gemini_quota
        from test_gemini_quota import PER_MINUTE_BODY

        monkeypatch.setattr(
            transcript.requests, "post",
            lambda url, **kwargs: FakeResponse(status_code=429, text=PER_MINUTE_BODY),
        )
        text, exhausted, reason = transcript._fetch_gemini_transcript("vid00000001")
        # Still a deferral — every model refused — but tomorrow's counter is
        # not carrying a write-off that a minute would have cleared.
        assert (text, exhausted, reason) == ("", True, "gemini_quota")
        for model in transcript._gemini_transcript_models():
            assert gemini_quota.is_exhausted(model) is False

    def test_a_minute_limited_model_is_not_re_asked_this_run(self, monkeypatch, gemini_key):
        # Not written into the day's counter, so the skip has to be remembered
        # in the process — otherwise every video re-asks and is refused again.
        from test_gemini_quota import PER_MINUTE_BODY

        calls = []

        def fake_post(url, **kwargs):
            calls.append(url)
            return FakeResponse(status_code=429, text=PER_MINUTE_BODY)

        monkeypatch.setattr(transcript.requests, "post", fake_post)
        transcript._fetch_gemini_transcript("vid00000001")
        transcript._fetch_gemini_transcript("vid00000002")
        assert len(calls) == len(transcript._gemini_transcript_models())


class TestOversizedVideo:
    """
    A video too long for the context window is not a budget problem.

    Every model in the rotation shares the same 1,048,576-token window, so once
    one has rejected the video on size the others can only spend a request to
    say the same thing.
    """

    TOO_LARGE = (
        '{"error":{"code":400,"message":"The input token count exceeds the '
        'maximum number of tokens allowed 1048576.","status":"INVALID_ARGUMENT"}}'
    )

    def test_the_rotation_stops_at_the_first_size_rejection(self, monkeypatch, gemini_key):
        # Observed on ucrXJlTbB_w: three models, three 400s, ~90s of run time,
        # repeated on every retry of that video.
        calls = []

        def fake_post(url, **kwargs):
            calls.append(url)
            return FakeResponse(status_code=400, text=self.TOO_LARGE)

        monkeypatch.setattr(transcript.requests, "post", fake_post)
        text, exhausted, reason = transcript._fetch_gemini_transcript("vid00000001")

        assert (text, reason) == ("", "gemini_too_large")
        assert len(calls) == 1
        # Not a deferral on budget grounds: nothing here is out of quota, and a
        # Supadata credit can still transcribe this video later.
        assert exhausted is False

    def test_size_is_not_charged_to_the_daily_quota(self, monkeypatch, gemini_key):
        import gemini_quota

        monkeypatch.setattr(
            transcript.requests, "post",
            lambda url, **kwargs: FakeResponse(status_code=400, text=self.TOO_LARGE),
        )
        transcript._fetch_gemini_transcript("vid00000001")
        for model in transcript._gemini_transcript_models():
            assert gemini_quota.is_exhausted(model) is False

    def test_an_unrelated_400_still_rotates(self, monkeypatch, gemini_key):
        # Only the size rejection is model-independent; a bad request to one
        # model says nothing about the next.
        calls = []

        def fake_post(url, **kwargs):
            calls.append(url)
            return FakeResponse(status_code=400, text='{"error":{"message":"bad"}}')

        monkeypatch.setattr(transcript.requests, "post", fake_post)
        _, _, reason = transcript._fetch_gemini_transcript("vid00000001")
        assert len(calls) == len(transcript._gemini_transcript_models())
        assert reason == "gemini_http_400"


class TestSuccessReasonsAreShared:
    def test_every_success_path_is_declared_a_success(self, monkeypatch, gemini_key):
        # scraper.py filters its end-of-run failure breakdown on this set. It
        # used to hold a hand-written copy, so adding `gemini_ok` here silently
        # made every Gemini success show up as a reported failure:
        #   "Transcript failures by reason: gemini_ok=4, gemini_http_400=1"
        monkeypatch.setattr(transcript, "_fetch_supadata", lambda vid: ("", False, "empty"))
        monkeypatch.setattr(transcript, "_fetch_gemini_transcript",
                            lambda vid: (LONG, False, "gemini_ok"))
        result = transcript.get_transcript_from_video("https://youtu.be/dQw4w9WgXcQ")
        assert result["transcript"]
        assert result["reason"] in transcript.TRANSCRIPT_SUCCESS_REASONS

    def test_supadata_and_fallback_successes_are_declared_too(self, monkeypatch, gemini_key):
        monkeypatch.setattr(transcript, "_fetch_supadata", lambda vid: ("text", False, "ok"))
        assert transcript.get_transcript_from_video(
            "https://youtu.be/dQw4w9WgXcQ")["reason"] in transcript.TRANSCRIPT_SUCCESS_REASONS

        monkeypatch.setattr(transcript, "_fetch_supadata", lambda vid: ("", False, "empty"))
        monkeypatch.setattr(transcript, "_fetch_gemini_transcript",
                            lambda vid: ("", False, "gemini_too_short"))
        monkeypatch.setattr(transcript, "_fetch_youtube_transcript_api", lambda vid: "text")
        assert transcript.get_transcript_from_video(
            "https://youtu.be/dQw4w9WgXcQ")["reason"] in transcript.TRANSCRIPT_SUCCESS_REASONS
