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


# --- RSS feed parsing -------------------------------------------------------

RSS_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns:yt="http://www.youtube.com/xml/schemas/2015" xmlns="http://www.w3.org/2005/Atom">
 <title>Chan</title>
 <entry>
  <id>yt:video:vidCCCCCCC3</id>
  <yt:videoId>vidCCCCCCC3</yt:videoId>
  <title>Third</title>
  <author><name>Chan</name></author>
  <published>2026-06-03T00:00:00+00:00</published>
 </entry>
 <entry>
  <id>yt:video:vidBBBBBBB2</id>
  <yt:videoId>vidBBBBBBB2</yt:videoId>
  <title>Second</title>
  <author><name>Chan</name></author>
  <published>2026-06-02T00:00:00+00:00</published>
 </entry>
 <entry>
  <id>yt:video:vidAAAAAAA1</id>
  <yt:videoId>vidAAAAAAA1</yt:videoId>
  <title>First</title>
  <author><name>Chan</name></author>
  <published>2026-06-01T00:00:00+00:00</published>
 </entry>
</feed>
"""


def test_parse_rss_feed():
    videos = scraper._parse_rss_feed(RSS_SAMPLE)
    assert [v["video_id"] for v in videos] == ["vidCCCCCCC3", "vidBBBBBBB2", "vidAAAAAAA1"]
    assert videos[0]["channel_name"] == "Chan"
    assert videos[0]["video_title"] == "Third"
    assert videos[0]["video_url"].endswith("vidCCCCCCC3")
    assert videos[0]["published_at"] == "2026-06-03T00:00:00+00:00"


def test_parse_rss_feed_empty_feed_is_valid():
    xml = '<feed xmlns="http://www.w3.org/2005/Atom"><title>Chan</title></feed>'
    assert scraper._parse_rss_feed(xml) == []


def test_parse_rss_feed_invalid_xml_signals_fallback():
    assert scraper._parse_rss_feed("<feed") is None


def test_get_recent_videos_via_rss(monkeypatch):
    class RssResp:
        status_code = 200
        text = RSS_SAMPLE

    monkeypatch.setattr(scraper.requests, "get", lambda *a, **k: RssResp())
    videos = scraper.get_recent_videos("KEY", "UCxxxxxxxxxxxxxxxxxxxxxx")
    assert len(videos) == 3


def test_get_recent_videos_falls_back_to_api(monkeypatch):
    monkeypatch.setattr(scraper.requests, "get", lambda *a, **k: FakeResp(404, text="gone"))
    monkeypatch.setattr(scraper, "get_latest_video", lambda key, cid: {"video_id": "vidAPI000001"})
    assert scraper.get_recent_videos("KEY", "UCx") == [{"video_id": "vidAPI000001"}]


# --- Candidate selection ----------------------------------------------------

def _vid(video_id, published):
    return {
        "video_id": video_id,
        "channel_name": "Chan",
        "video_title": f"T-{video_id}",
        "video_url": f"https://www.youtube.com/watch?v={video_id}",
        "published_at": published,
    }


FEED = [  # newest first, like a real feed
    _vid("v5", "2026-06-05T00:00:00+00:00"),
    _vid("v4", "2026-06-04T00:00:00+00:00"),
    _vid("v3", "2026-06-03T00:00:00+00:00"),
    _vid("v2", "2026-06-02T00:00:00+00:00"),
    _vid("v1", "2026-06-01T00:00:00+00:00"),
]


def test_select_candidates_new_channel_takes_only_latest():
    # A newly added channel must not flood Telegram with its back catalog.
    out = scraper._select_candidates(FEED, None, {})
    assert [v["video_id"] for v in out] == ["v5"]


def test_select_candidates_watermark_returns_newer_oldest_first():
    state = {"last_video_id": "v3", "last_published": "2026-06-03T00:00:00+00:00"}
    out = scraper._select_candidates(FEED, state, {})
    assert [v["video_id"] for v in out] == ["v4", "v5"]


def test_select_candidates_watermark_up_to_date():
    state = {"last_video_id": "v5", "last_published": "2026-06-05T00:00:00+00:00"}
    assert scraper._select_candidates(FEED, state, {}) == []


def test_select_candidates_caps_at_limit_keeping_oldest():
    state = {"last_video_id": "v1", "last_published": "2026-06-01T00:00:00+00:00"}
    out = scraper._select_candidates(FEED, state, {}, limit=2)
    # Oldest first so delivery stays chronological; v4/v5 wait for the next run.
    assert [v["video_id"] for v in out] == ["v2", "v3"]


def test_select_candidates_v1_state_uses_feed_position():
    # Migrated v1 state has an id but no timestamp: take everything newer.
    out = scraper._select_candidates(FEED, {"last_video_id": "v4"}, {})
    assert [v["video_id"] for v in out] == ["v5"]


def test_select_candidates_v1_state_id_gone_resumes_from_latest():
    out = scraper._select_candidates(FEED, {"last_video_id": "vGONE"}, {})
    assert [v["video_id"] for v in out] == ["v5"]


def test_select_candidates_readds_pending_below_watermark():
    # v2 was deferred (e.g. captions not up yet) and the watermark moved past it.
    state = {"last_video_id": "v4", "last_published": "2026-06-04T00:00:00+00:00"}
    pending = {"v2": {"channel_id": "UCx", "attempts": 1}}
    out = scraper._select_candidates(FEED, state, pending)
    assert [v["video_id"] for v in out] == ["v2", "v5"]


# --- Watermark advance ------------------------------------------------------

def test_advance_channel_state_sets_watermark():
    channels = {}
    scraper._advance_channel_state(channels, "UCx", FEED[0])
    assert channels["UCx"] == {
        "last_video_id": "v5",
        "last_published": "2026-06-05T00:00:00+00:00",
    }


def test_advance_channel_state_never_regresses():
    # Deciding an older (previously deferred) video must not move the watermark back.
    channels = {"UCx": {"last_video_id": "v5", "last_published": "2026-06-05T00:00:00+00:00"}}
    scraper._advance_channel_state(channels, "UCx", FEED[3])  # v2, older
    assert channels["UCx"]["last_video_id"] == "v5"
    assert channels["UCx"]["last_published"] == "2026-06-05T00:00:00+00:00"


# --- Per-video summarization outcomes ---------------------------------------

def test_summarize_video_defers_missing_transcript(monkeypatch):
    monkeypatch.setattr(scraper, "get_transcript_from_video", lambda url: {"transcript": ""})
    body, outcome, decided = scraper._summarize_video(_vid("v1", ""), no_transcript_attempts=0)
    # Captions may still be processing: stay silent and retry next run.
    assert body is None
    assert outcome == "no_transcript_deferred"
    assert decided is False


def test_summarize_video_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr(scraper, "get_transcript_from_video", lambda url: {"transcript": ""})
    body, outcome, decided = scraper._summarize_video(
        _vid("v1", ""), no_transcript_attempts=scraper.NO_TRANSCRIPT_MAX_ATTEMPTS - 1
    )
    assert body is not None and "No transcript" in body
    assert outcome == "no_transcript"
    assert decided is True


def test_summarize_video_quota_deferral_not_decided(monkeypatch):
    monkeypatch.setattr(scraper, "get_transcript_from_video", lambda url: {"transcript": "words"})
    monkeypatch.setattr(scraper, "summarize_transcript", lambda t, title, **kw: scraper.QUOTA_EXHAUSTED_SENTINEL)
    body, outcome, decided = scraper._summarize_video(_vid("v1", ""))
    assert outcome == "quota_deferred"
    assert decided is False


def test_summarize_video_compact_flag_passthrough(monkeypatch):
    # Digest-mode channels must get compact TL;DR summaries.
    captured = {}

    def fake_summarize(t, title=None, compact=False):
        captured["compact"] = compact
        return "TLDR"

    monkeypatch.setattr(scraper, "get_transcript_from_video", lambda url: {"transcript": "words"})
    monkeypatch.setattr(scraper, "summarize_transcript", fake_summarize)
    scraper._summarize_video(_vid("v1", ""), compact=True)
    assert captured["compact"] is True
    scraper._summarize_video(_vid("v1", ""))
    assert captured["compact"] is False


def test_summarize_video_success(monkeypatch):
    monkeypatch.setattr(scraper, "get_transcript_from_video", lambda url: {"transcript": "words"})
    monkeypatch.setattr(scraper, "summarize_transcript", lambda t, title, **kw: "TLDR\n\n• point")
    details = _vid("v1", "")
    body, outcome, decided = scraper._summarize_video(details)
    assert body == "TLDR\n\n• point"
    assert outcome == "sent"
    assert decided is True
    assert details["summary"] == "TLDR\n\n• point"
