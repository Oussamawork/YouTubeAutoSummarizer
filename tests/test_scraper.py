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


# --- Free/premium teaser split (dual-send orchestration in main) ---


def _run_main(monkeypatch, free_channel=None, premium_url=None, outcome="sent",
              body="TL;DR line\n\n• detail 1\n• detail 2", market_signals=False):
    """Drive main() with everything mocked; return the recorded send calls."""
    for name, value in {
        "YOUTUBE_API_KEY": "yt", "TELEGRAM_TOKEN": "tok", "TELEGRAM_CHANNEL_ID": "premium",
    }.items():
        monkeypatch.setenv(name, value)
    for name, value in {
        "TELEGRAM_FREE_CHANNEL_ID": free_channel, "PREMIUM_INVITE_URL": premium_url,
    }.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    monkeypatch.delenv("DAILY_DIGEST", raising=False)
    # Recording defaults ON; tests opt out explicitly so the flag-off paths
    # stay covered.
    monkeypatch.setenv("MARKET_SIGNALS", "true" if market_signals else "false")

    video = {
        "video_id": "v1", "channel_name": "Chan", "video_title": "Title",
        "video_url": "http://u", "published_at": "2026-07-01T00:00:00+00:00",
    }
    monkeypatch.setattr(scraper, "read_channels", lambda path: [
        {"channel_id": "c1", "digest": False, "max_per_run": 3},
    ])
    monkeypatch.setattr(scraper, "load_state", lambda path: {"channels": {}, "pending": {}})
    monkeypatch.setattr(scraper, "save_state", lambda path, state: None)
    monkeypatch.setattr(scraper, "save_to_json", lambda results, filename: None)
    monkeypatch.setattr(scraper, "get_recent_videos", lambda key, cid: [video])
    monkeypatch.setattr(
        scraper, "_summarize_video",
        lambda details, attempts, compact=False: (body, outcome, True),
    )

    calls = {"message": [], "teaser": []}
    monkeypatch.setattr(
        scraper, "send_telegram_message",
        lambda *args: calls["message"].append(args) or True,
    )
    monkeypatch.setattr(
        scraper, "send_telegram_teaser",
        lambda *args: calls["teaser"].append(args) or True,
    )
    scraper.main()
    return calls


def test_main_sends_teaser_to_free_channel(monkeypatch):
    calls = _run_main(monkeypatch, free_channel="free", premium_url="https://t.me/+inv")
    # Full summary still goes to the premium channel.
    assert len(calls["message"]) == 1
    assert calls["message"][0][1] == "premium"
    # Teaser goes to the free channel: TL;DR first line + CTA url.
    assert len(calls["teaser"]) == 1
    token, chat_id, channel_name, title, url, teaser, cta = calls["teaser"][0]
    assert chat_id == "free"
    assert teaser == "TL;DR line"
    assert cta == "https://t.me/+inv"


def test_main_no_teaser_when_free_channel_unset(monkeypatch):
    calls = _run_main(monkeypatch, free_channel=None)
    assert len(calls["message"]) == 1
    assert calls["teaser"] == []


def test_main_no_teaser_for_warning_outcomes(monkeypatch):
    # Deferral/warning notices must never reach the public free channel.
    calls = _run_main(
        monkeypatch, free_channel="free",
        outcome="no_transcript", body="⚠️ No transcript available. Manual review needed.",
    )
    assert len(calls["message"]) == 1  # premium notice still sent
    assert calls["teaser"] == []


def test_main_teaser_failure_does_not_affect_state(monkeypatch):
    # A failing teaser send must not stop the watermark advance for the video.
    saved_states = []
    calls = _run_main(monkeypatch, free_channel="free")
    # Re-run with a teaser sender that fails and a state recorder.
    monkeypatch.setattr(
        scraper, "send_telegram_teaser", lambda *args: False,
    )
    monkeypatch.setattr(
        scraper, "save_state", lambda path, state: saved_states.append(
            {"channels": dict(state["channels"]), "pending": dict(state["pending"])}
        ),
    )
    scraper.main()
    assert saved_states  # state persisted
    assert saved_states[-1]["channels"].get("c1", {}).get("last_video_id") == "v1"
    assert saved_states[-1]["pending"] == {}


# --- Market-signal recording (MARKET_SIGNALS orchestration in main) ---


def test_main_records_signals_when_enabled(monkeypatch):
    records = []
    monkeypatch.setattr(
        scraper, "extract_signals",
        lambda summary, title=None, channel=None: {
            "assets": [], "market_sentiment": "bullish", "topics": []},
    )
    monkeypatch.setattr(
        scraper, "append_jsonl",
        lambda path, rec: records.append((path, rec)) or True,
    )
    _run_main(monkeypatch, market_signals=True)
    assert len(records) == 1
    path, rec = records[0]
    assert path == scraper.SIGNALS_FILE
    assert rec["video_id"] == "v1"
    assert rec["channel_id"] == "c1"
    assert rec["summary"].startswith("TL;DR line")
    assert rec["signals"]["market_sentiment"] == "bullish"
    assert rec["date"]


def test_main_no_signals_when_flag_off(monkeypatch):
    called = []
    monkeypatch.setattr(scraper, "extract_signals", lambda *a, **k: called.append(1))
    _run_main(monkeypatch)  # _run_main sets MARKET_SIGNALS=false by default
    assert called == []


def test_env_flag_defaults():
    assert scraper._env_flag("NO_SUCH_FLAG_XYZ") is False
    assert scraper._env_flag("NO_SUCH_FLAG_XYZ", default=True) is True


def test_main_signals_default_on(monkeypatch):
    # With MARKET_SIGNALS entirely unset, recording is enabled by default.
    records = []
    monkeypatch.setattr(
        scraper, "extract_signals",
        lambda *a, **k: {"assets": [], "market_sentiment": "neutral", "topics": []},
    )
    monkeypatch.setattr(scraper, "append_jsonl", lambda path, rec: records.append(rec) or True)
    _run_main(monkeypatch, market_signals=True)
    monkeypatch.delenv("MARKET_SIGNALS", raising=False)
    scraper.main()
    assert len(records) == 2  # once from _run_main, once from the unset-flag run


def test_main_no_signals_for_warning_outcomes(monkeypatch):
    called = []
    monkeypatch.setattr(scraper, "extract_signals", lambda *a, **k: called.append(1) or None)
    _run_main(monkeypatch, market_signals=True, outcome="no_transcript",
              body="⚠️ No transcript available. Manual review needed.")
    assert called == []


def test_main_signal_failure_never_breaks_delivery(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("LLM exploded")

    monkeypatch.setattr(scraper, "extract_signals", boom)
    calls = _run_main(monkeypatch, market_signals=True)
    assert len(calls["message"]) == 1  # summary still delivered to Telegram


def test_main_records_row_even_when_extraction_returns_none(monkeypatch):
    records = []
    monkeypatch.setattr(scraper, "extract_signals", lambda *a, **k: None)
    monkeypatch.setattr(scraper, "append_jsonl", lambda path, rec: records.append(rec) or True)
    _run_main(monkeypatch, market_signals=True)
    assert len(records) == 1
    assert records[0]["signals"] is None  # summary row still kept for the dataset
