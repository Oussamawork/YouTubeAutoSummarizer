"""Tests for transcript ID extraction and Supadata payload parsing (no network)."""
import transcript


class TestExtractVideoId:
    def test_bare_id(self):
        assert transcript._extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_watch_url(self):
        assert (
            transcript._extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            == "dQw4w9WgXcQ"
        )

    def test_youtu_be(self):
        assert transcript._extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_shorts(self):
        assert (
            transcript._extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ")
            == "dQw4w9WgXcQ"
        )

    def test_embed(self):
        assert (
            transcript._extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ")
            == "dQw4w9WgXcQ"
        )

    def test_empty(self):
        assert transcript._extract_video_id("") == ""

    def test_garbage(self):
        assert transcript._extract_video_id("not a url") == ""


class TestSupadataPayload:
    def test_string_content(self):
        assert transcript._supadata_text_from_payload({"content": "  hello  "}) == "hello"

    def test_list_content(self):
        data = {"content": [{"text": "a"}, {"text": "b"}]}
        assert transcript._supadata_text_from_payload(data) == "a b"

    def test_none(self):
        assert transcript._supadata_text_from_payload(None) == ""

    def test_missing_content(self):
        assert transcript._supadata_text_from_payload({}) == ""
