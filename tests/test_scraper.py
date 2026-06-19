"""Tests for YouTube response parsing and latest-video fetch (network mocked)."""
import scraper


class FakeResp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text
        self.url = "http://test"

    def json(self):
        return self._payload


def test_parse_latest_video():
    data = {
        "items": [
            {
                "id": {"videoId": "vid12345678"},
                "snippet": {
                    "channelTitle": "Chan",
                    "title": "Title",
                    "publishedAt": "2024-01-01T00:00:00Z",
                },
            }
        ]
    }
    out = scraper._parse_latest_video(data)
    assert out["video_id"] == "vid12345678"
    assert out["channel_name"] == "Chan"
    assert out["video_title"] == "Title"
    assert out["video_url"].endswith("vid12345678")


def test_parse_latest_video_empty():
    assert scraper._parse_latest_video({"items": []}) is None
    assert scraper._parse_latest_video({}) is None


def test_parse_latest_video_malformed_item():
    # Missing videoId or snippet must yield None, not raise (I7 hardening).
    assert scraper._parse_latest_video({"items": [{"snippet": {"title": "t"}}]}) is None
    assert scraper._parse_latest_video({"items": [{"id": {}, "snippet": {}}]}) is None


def test_get_latest_video_success(monkeypatch):
    payload = {
        "items": [
            {
                "id": {"videoId": "vid12345678"},
                "snippet": {
                    "channelTitle": "Chan",
                    "title": "Title",
                    "publishedAt": "2024-01-01T00:00:00Z",
                },
            }
        ]
    }
    monkeypatch.setattr(scraper.requests, "get", lambda *a, **k: FakeResp(200, payload))
    out = scraper.get_latest_video("KEY", "UCxxxxxxxxxxxxxxxxxxxxxx")
    assert out["video_id"] == "vid12345678"


def test_get_latest_video_non_transient_error(monkeypatch):
    monkeypatch.setattr(
        scraper.requests, "get", lambda *a, **k: FakeResp(404, {}, text="not found")
    )
    assert scraper.get_latest_video("KEY", "UCxxxxxxxxxxxxxxxxxxxxxx") is None
