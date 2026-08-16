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


def test_select_candidates_uncapped_takes_every_due_video():
    # limit=0 means no cap: the whole backlog goes out in this run instead of
    # the newest videos waiting for the next one.
    state = {"last_video_id": "v1", "last_published": "2026-06-01T00:00:00+00:00"}
    out = scraper._select_candidates(FEED, state, {}, limit=0)
    assert [v["video_id"] for v in out] == ["v2", "v3", "v4", "v5"]


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
    body, outcome, decided, _sig = scraper._summarize_video(_vid("v1", ""), no_transcript_attempts=0)
    # Captions may still be processing: stay silent and retry next run.
    assert body is None
    assert outcome == "no_transcript_deferred"
    assert decided is False


def test_summarize_video_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr(scraper, "get_transcript_from_video", lambda url: {"transcript": ""})
    body, outcome, decided, _sig = scraper._summarize_video(
        _vid("v1", ""), no_transcript_attempts=scraper.NO_TRANSCRIPT_MAX_ATTEMPTS - 1
    )
    assert body is not None and "No transcript" in body
    assert outcome == "no_transcript"
    assert decided is True


def test_summarize_video_keeps_trying_while_inside_the_time_window(monkeypatch):
    # Attempts alone is the wrong unit: a fast polling schedule burns through
    # the count in hours, while auto-captions can take most of a day to appear.
    monkeypatch.setattr(scraper, "get_transcript_from_video", lambda url: {"transcript": ""})
    body, outcome, decided, _sig = scraper._summarize_video(
        _vid("v1", ""),
        no_transcript_attempts=scraper.NO_TRANSCRIPT_MAX_ATTEMPTS + 5,
        hours_since_first=scraper.NO_TRANSCRIPT_MIN_HOURS - 1,
    )
    assert body is None and outcome == "no_transcript_deferred" and decided is False


def test_summarize_video_gives_up_once_both_gates_pass(monkeypatch):
    monkeypatch.setattr(scraper, "get_transcript_from_video", lambda url: {"transcript": ""})
    body, outcome, decided, _sig = scraper._summarize_video(
        _vid("v1", ""),
        no_transcript_attempts=scraper.NO_TRANSCRIPT_MAX_ATTEMPTS - 1,
        hours_since_first=scraper.NO_TRANSCRIPT_MIN_HOURS,
    )
    assert body is not None and outcome == "no_transcript" and decided is True


def test_summarize_video_time_window_alone_does_not_give_up(monkeypatch):
    # Long-waited but barely retried (e.g. runs were failing): keep trying.
    monkeypatch.setattr(scraper, "get_transcript_from_video", lambda url: {"transcript": ""})
    body, outcome, decided, _sig = scraper._summarize_video(
        _vid("v1", ""), no_transcript_attempts=0,
        hours_since_first=scraper.NO_TRANSCRIPT_MIN_HOURS * 10,
    )
    assert body is None and outcome == "no_transcript_deferred" and decided is False


def test_retry_backoff_holds_a_recently_tried_video(monkeypatch):
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(scraper, "PENDING_RETRY_MIN_HOURS", 3.0)
    recent = {"last_attempt": (now - timedelta(hours=1)).isoformat()}
    assert scraper._retry_wait_remaining(recent, now=now) == 2.0
    old = {"last_attempt": (now - timedelta(hours=5)).isoformat()}
    assert scraper._retry_wait_remaining(old, now=now) == 0.0


def test_retry_backoff_lets_untimestamped_and_budget_deferrals_through(monkeypatch):
    # Records written before timestamps existed, and budget deferrals (which
    # cost no credit and so are never stamped), must retry immediately.
    monkeypatch.setattr(scraper, "PENDING_RETRY_MIN_HOURS", 3.0)
    assert scraper._retry_wait_remaining({"attempts": 2}) == 0.0
    assert scraper._retry_wait_remaining({}) == 0.0
    assert scraper._retry_wait_remaining(None) == 0.0


def test_retry_backoff_disabled_by_zero(monkeypatch):
    from datetime import datetime, timedelta, timezone
    monkeypatch.setattr(scraper, "PENDING_RETRY_MIN_HOURS", 0)
    just_now = {"last_attempt": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()}
    assert scraper._retry_wait_remaining(just_now) == 0.0


def test_summarize_video_quota_deferral_not_decided(monkeypatch):
    monkeypatch.setattr(scraper, "get_transcript_from_video", lambda url: {"transcript": "words"})
    monkeypatch.setattr(scraper, "summarize_transcript", lambda t, title, **kw: scraper.QUOTA_EXHAUSTED_SENTINEL)
    body, outcome, decided, _sig = scraper._summarize_video(_vid("v1", ""))
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
    body, outcome, decided, _sig = scraper._summarize_video(details)
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
        lambda details, attempts, compact=False, want_signals=False, hours_since_first=None: (body, outcome, True, None),
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


# --- @handle resolution ---


def test_resolve_channel_handle_success(monkeypatch):
    monkeypatch.setattr(
        scraper.requests, "get",
        lambda *a, **k: FakeResp(200, {"items": [{"id": "UCabc123"}]}),
    )
    assert scraper.resolve_channel_handle("KEY", "@hkcm") == "UCabc123"


def test_resolve_channel_handle_sends_forhandle_param(monkeypatch):
    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured["url"], captured["params"] = url, params
        return FakeResp(200, {"items": [{"id": "UCabc123"}]})

    monkeypatch.setattr(scraper.requests, "get", fake_get)
    scraper.resolve_channel_handle("KEY", "@hkcm")
    assert captured["url"] == scraper.YOUTUBE_CHANNELS_URL
    assert captured["params"]["forHandle"] == "@hkcm"
    assert captured["params"]["part"] == "id"


def test_resolve_channel_handle_unknown_returns_none(monkeypatch):
    monkeypatch.setattr(scraper.requests, "get", lambda *a, **k: FakeResp(200, {"items": []}))
    assert scraper.resolve_channel_handle("KEY", "@nope") is None


def test_resolve_channel_handle_http_error_returns_none(monkeypatch):
    monkeypatch.setattr(scraper.requests, "get", lambda *a, **k: FakeResp(403, {}, text="forbidden"))
    assert scraper.resolve_channel_handle("KEY", "@x") is None


def test_main_resolves_handle_and_keys_state_by_id(monkeypatch):
    """A handle entry is resolved once, and dedup state uses the resolved id."""
    saved = []
    monkeypatch.setenv("YOUTUBE_API_KEY", "yt")
    monkeypatch.setenv("TELEGRAM_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHANNEL_ID", "premium")
    monkeypatch.delenv("TELEGRAM_FREE_CHANNEL_ID", raising=False)
    monkeypatch.delenv("DAILY_DIGEST", raising=False)
    monkeypatch.setenv("MARKET_SIGNALS", "false")

    video = {
        "video_id": "v1", "channel_name": "HKCM", "video_title": "T",
        "video_url": "http://u", "published_at": "2026-07-01T00:00:00+00:00",
    }
    calls = []
    monkeypatch.setattr(scraper, "read_channels", lambda p: [
        {"channel_id": "@hkcm", "digest": False, "max_per_run": 3},
        {"channel_id": "@hkcm", "digest": False, "max_per_run": 3},  # same handle twice
    ])
    monkeypatch.setattr(scraper, "load_state", lambda p: {"channels": {}, "pending": {}})
    monkeypatch.setattr(scraper, "save_state", lambda p, s: saved.append(
        {k: dict(v) for k, v in s["channels"].items()}))
    monkeypatch.setattr(scraper, "save_to_json", lambda r, f: None)
    monkeypatch.setattr(scraper, "resolve_channel_handle",
                        lambda key, handle: calls.append(handle) or "UCresolved")
    monkeypatch.setattr(scraper, "get_recent_videos", lambda key, cid: [video] if cid == "UCresolved" else [])
    monkeypatch.setattr(scraper, "_summarize_video", lambda d, a, compact=False, want_signals=False, hours_since_first=None: ("S", "sent", True, None))
    monkeypatch.setattr(scraper, "send_telegram_message", lambda *a: True)
    scraper.main()

    assert calls == ["@hkcm"]  # resolved once, cached for the second entry
    assert saved and "UCresolved" in saved[-1]  # state keyed by id, not handle


def test_main_skips_unresolvable_handle(monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "yt")
    monkeypatch.setenv("TELEGRAM_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHANNEL_ID", "premium")
    monkeypatch.setenv("MARKET_SIGNALS", "false")
    fetched = []
    monkeypatch.setattr(scraper, "read_channels", lambda p: [
        {"channel_id": "@bad", "digest": False, "max_per_run": 3},
    ])
    monkeypatch.setattr(scraper, "load_state", lambda p: {"channels": {}, "pending": {}})
    monkeypatch.setattr(scraper, "save_state", lambda p, s: None)
    monkeypatch.setattr(scraper, "save_to_json", lambda r, f: None)
    monkeypatch.setattr(scraper, "resolve_channel_handle", lambda key, handle: None)
    monkeypatch.setattr(scraper, "get_recent_videos", lambda key, cid: fetched.append(cid) or [])
    scraper.main()
    assert fetched == []  # never fetches with an unresolved handle


# --- Combined summarize+extract wiring (one LLM call instead of two) ---


def test_summarize_video_uses_combined_call(monkeypatch):
    monkeypatch.setattr(scraper, "get_transcript_from_video", lambda url: {"transcript": "text"})
    monkeypatch.setattr(
        scraper, "summarize_with_signals",
        lambda t, title=None, compact=False, channel_name=None: ("Summary", {"assets": []}),
    )
    monkeypatch.setattr(
        scraper, "summarize_transcript",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("second call should not happen")),
    )
    body, outcome, decided, sig = scraper._summarize_video(
        _vid("v1", ""), want_signals=True)
    assert (body, outcome, decided) == ("Summary", "sent", True)
    assert sig == {"assets": []}


def test_summarize_video_falls_back_when_combined_unusable(monkeypatch):
    monkeypatch.setattr(scraper, "get_transcript_from_video", lambda url: {"transcript": "text"})
    monkeypatch.setattr(
        scraper, "summarize_with_signals",
        lambda t, title=None, compact=False, channel_name=None: None,
    )
    monkeypatch.setattr(scraper, "summarize_transcript", lambda *a, **k: "Plain summary")
    body, outcome, decided, sig = scraper._summarize_video(_vid("v1", ""), want_signals=True)
    assert body == "Plain summary" and outcome == "sent" and sig is None


def test_summarize_video_skips_combined_when_signals_off(monkeypatch):
    monkeypatch.setattr(scraper, "get_transcript_from_video", lambda url: {"transcript": "text"})
    monkeypatch.setattr(
        scraper, "summarize_with_signals",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("combined call should not happen")),
    )
    monkeypatch.setattr(scraper, "summarize_transcript", lambda *a, **k: "Plain summary")
    body, _, _, sig = scraper._summarize_video(_vid("v1", ""))
    assert body == "Plain summary" and sig is None


def test_record_market_signals_reuses_combined_result(monkeypatch):
    rows = []
    monkeypatch.setattr(
        scraper, "extract_signals",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not re-extract")),
    )
    monkeypatch.setattr(scraper, "append_jsonl", lambda p, r: rows.append(r) or True)
    scraper._record_market_signals("c1", _vid("v1", ""), "S", {"assets": [], "market_sentiment": "neutral"})
    assert rows[0]["signals"]["market_sentiment"] == "neutral"


def test_record_market_signals_extracts_when_not_supplied(monkeypatch):
    rows, calls = [], []
    monkeypatch.setattr(scraper, "extract_signals", lambda *a, **k: calls.append(1) or {"assets": []})
    monkeypatch.setattr(scraper, "append_jsonl", lambda p, r: rows.append(r) or True)
    scraper._record_market_signals("c1", _vid("v1", ""), "S", None)
    assert calls == [1] and rows[0]["signals"] == {"assets": []}


# --- Transcript budget deferral and pending eviction ---


def test_summarize_video_budget_deferral_is_silent(monkeypatch):
    # Budget exhaustion is our choice, not the video's fault: no message, no
    # retry attempt consumed, watermark stays put.
    monkeypatch.setattr(
        scraper, "get_transcript_from_video",
        lambda url: {"transcript": "", "budget_exhausted": True},
    )
    body, outcome, decided, sig = scraper._summarize_video(_vid("v1", ""), no_transcript_attempts=2)
    assert body is None and outcome == "budget_deferred" and decided is False and sig is None


def test_summarize_video_missing_transcript_still_reports(monkeypatch):
    # Without the budget flag, the normal give-up path is unchanged.
    monkeypatch.setattr(
        scraper, "get_transcript_from_video",
        lambda url: {"transcript": "", "budget_exhausted": False},
    )
    body, outcome, decided, _ = scraper._summarize_video(
        _vid("v1", ""), no_transcript_attempts=scraper.NO_TRANSCRIPT_MAX_ATTEMPTS - 1)
    assert outcome == "no_transcript" and decided is True and "No transcript" in body


def test_evict_orphaned_pending_drops_videos_gone_from_feed():
    pending = {
        "gone": {"channel_id": "c1", "attempts": 2},      # not in feed -> evict
        "still": {"channel_id": "c1", "attempts": 1},     # in feed -> keep
        "other": {"channel_id": "c2", "attempts": 2},     # other channel -> keep
        "bad": "not a dict",
    }
    dropped = scraper._evict_orphaned_pending(pending, "c1", {"still", "new"})
    assert dropped == 1
    assert set(pending) == {"still", "other", "bad"}


def test_evict_orphaned_pending_noop_when_all_present():
    pending = {"a": {"channel_id": "c1", "attempts": 1}}
    assert scraper._evict_orphaned_pending(pending, "c1", {"a"}) == 0
    assert set(pending) == {"a"}


# --- Title filter wiring (only=) ---


def _filter_run(monkeypatch, only, titles):
    """Run main() over one channel whose feed has the given titles."""
    for name, value in {"YOUTUBE_API_KEY": "yt", "TELEGRAM_TOKEN": "tok",
                        "TELEGRAM_CHANNEL_ID": "premium"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("MARKET_SIGNALS", "false")
    monkeypatch.delenv("DAILY_DIGEST", raising=False)
    monkeypatch.delenv("TELEGRAM_FREE_CHANNEL_ID", raising=False)

    feed = [
        {"video_id": f"v{i}", "channel_name": "C", "video_title": t,
         "video_url": f"http://u/{i}", "published_at": f"2026-07-0{i+1}T00:00:00+00:00"}
        for i, t in enumerate(titles)
    ]
    summarized = []
    monkeypatch.setattr(scraper, "read_channels", lambda p: [
        {"channel_id": "c1", "digest": False, "max_per_run": 10, "only": only},
    ])
    monkeypatch.setattr(scraper, "load_state", lambda p: {
        "channels": {"c1": {"last_video_id": "seed", "last_published": "2026-01-01T00:00:00+00:00"}},
        "pending": {},
    })
    monkeypatch.setattr(scraper, "save_state", lambda p, s: None)
    monkeypatch.setattr(scraper, "save_to_json", lambda r, f: None)
    monkeypatch.setattr(scraper, "get_recent_videos", lambda k, c: feed)
    monkeypatch.setattr(
        scraper, "_summarize_video",
        lambda d, a, compact=False, want_signals=False, hours_since_first=None: summarized.append(d["video_title"]) or ("S", "sent", True, None),
    )
    monkeypatch.setattr(scraper, "send_telegram_message", lambda *a: True)
    scraper.main()
    return summarized


def test_main_title_filter_skips_before_transcript(monkeypatch):
    # Non-matching videos must never reach _summarize_video — that is where a
    # transcript credit would be spent.
    seen = _filter_run(
        monkeypatch, ["btc", "bitcoin"],
        ["XRP Price Analysis", "Has Bitcoin Started the Sell-off?", "HBAR Wave Count"],
    )
    assert seen == ["Has Bitcoin Started the Sell-off?"]


def test_main_no_filter_processes_everything(monkeypatch):
    seen = _filter_run(monkeypatch, [], ["XRP Price Analysis", "Bitcoin update"])
    assert len(seen) == 2


def test_main_filter_matching_nothing_is_a_clean_skip(monkeypatch):
    seen = _filter_run(monkeypatch, ["btc"], ["XRP Price Analysis", "HBAR Wave Count"])
    assert seen == []


# --- Duration gate (skip Shorts before spending a transcript credit) ---


def test_parse_iso_duration():
    assert scraper.parse_iso_duration("PT58S") == 58
    assert scraper.parse_iso_duration("PT4M13S") == 253
    assert scraper.parse_iso_duration("PT1H2M3S") == 3723
    assert scraper.parse_iso_duration("PT15M") == 900
    assert scraper.parse_iso_duration("P1DT2H") == 93600
    assert scraper.parse_iso_duration("bogus") is None
    assert scraper.parse_iso_duration("") is None
    assert scraper.parse_iso_duration(None) is None


def test_fetch_video_details_batches_by_fifty(monkeypatch, real_fetch_video_details):
    calls = []

    def fake_get(url, params=None, timeout=None):
        ids = params["id"].split(",")
        calls.append(len(ids))
        return FakeResp(200, {"items": [
            {"id": i, "contentDetails": {"duration": "PT5M", "caption": "true"}} for i in ids
        ]})

    monkeypatch.setattr(scraper.requests, "get", fake_get)
    details = real_fetch_video_details("KEY", [f"v{i}" for i in range(120)])
    assert calls == [50, 50, 20]      # 3 requests = 3 quota units for 120 videos
    assert len(details) == 120
    assert details["v0"] == {"duration_seconds": 300, "has_captions": True}


def test_fetch_video_details_http_error_returns_empty(monkeypatch, real_fetch_video_details):
    monkeypatch.setattr(scraper.requests, "get", lambda *a, **k: FakeResp(403, {}, text="denied"))
    assert real_fetch_video_details("KEY", ["v1"]) == {}


def test_fetch_video_details_parses_caption_flag(monkeypatch, real_fetch_video_details):
    monkeypatch.setattr(scraper.requests, "get", lambda *a, **k: FakeResp(200, {"items": [
        {"id": "a", "contentDetails": {"duration": "PT1M", "caption": "false"}},
        {"id": "b", "contentDetails": {"duration": "PT1M"}},           # missing flag
    ]}))
    details = real_fetch_video_details("KEY", ["a", "b"])
    assert details["a"]["has_captions"] is False
    assert details["b"]["has_captions"] is None


def test_filter_by_duration_skips_shorts_and_keeps_unknown():
    videos = [_vid("short", ""), _vid("long", ""), _vid("unknown", "")]
    details = {
        "short": {"duration_seconds": 45, "has_captions": None},
        "long": {"duration_seconds": 600, "has_captions": None},
        # "unknown" absent: metadata lookup failed for it
    }
    kept, short, uncaptioned = scraper.filter_by_duration(videos, details, 90)
    assert [v["video_id"] for v in kept] == ["long", "unknown"]  # fail open
    assert short == 1 and uncaptioned == 0
    assert kept[0]["duration_seconds"] == 600  # recorded for diagnostics


def test_filter_by_duration_caption_skip_is_opt_in():
    videos = [_vid("v1", "")]
    details = {"v1": {"duration_seconds": 600, "has_captions": False}}
    kept, _, uncaptioned = scraper.filter_by_duration(videos, details, 90)
    assert len(kept) == 1 and uncaptioned == 0          # off by default
    kept, _, uncaptioned = scraper.filter_by_duration(videos, details, 90, skip_uncaptioned=True)
    assert kept == [] and uncaptioned == 1


def test_filter_by_duration_disabled_with_zero():
    videos = [_vid("tiny", "")]
    details = {"tiny": {"duration_seconds": 5, "has_captions": None}}
    kept, short, _ = scraper.filter_by_duration(videos, details, 0)
    assert len(kept) == 1 and short == 0


def test_main_duration_gate_skips_before_transcript(monkeypatch):
    """A Short must never reach _summarize_video — that is where a credit goes."""
    summarized = []
    for name, value in {"YOUTUBE_API_KEY": "yt", "TELEGRAM_TOKEN": "tok",
                        "TELEGRAM_CHANNEL_ID": "premium"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("MARKET_SIGNALS", "false")
    monkeypatch.delenv("DAILY_DIGEST", raising=False)
    monkeypatch.delenv("TELEGRAM_FREE_CHANNEL_ID", raising=False)

    feed = [_vid("shorty", "2026-07-02T00:00:00+00:00"), _vid("real", "2026-07-03T00:00:00+00:00")]
    monkeypatch.setattr(scraper, "read_channels", lambda p: [
        {"channel_id": "c1", "digest": False, "max_per_run": 10, "only": []},
    ])
    monkeypatch.setattr(scraper, "load_state", lambda p: {
        "channels": {"c1": {"last_video_id": "seed", "last_published": "2026-01-01T00:00:00+00:00"}},
        "pending": {},
    })
    monkeypatch.setattr(scraper, "save_state", lambda p, s: None)
    monkeypatch.setattr(scraper, "save_to_json", lambda r, f: None)
    monkeypatch.setattr(scraper, "get_recent_videos", lambda k, c: feed)
    monkeypatch.setattr(scraper, "fetch_video_details", lambda key, ids: {
        "shorty": {"duration_seconds": 40, "has_captions": None},
        "real": {"duration_seconds": 900, "has_captions": None},
    })
    monkeypatch.setattr(
        scraper, "_summarize_video",
        lambda d, a, compact=False, want_signals=False, hours_since_first=None: summarized.append(d["video_id"]) or ("S", "sent", True, None),
    )
    monkeypatch.setattr(scraper, "send_telegram_message", lambda *a: True)
    scraper.main()
    assert summarized == ["real"]


def test_summarize_video_records_transcript_reason(monkeypatch):
    monkeypatch.setattr(
        scraper, "get_transcript_from_video",
        lambda url: {"transcript": "", "budget_exhausted": False, "reason": "empty_content"},
    )
    details = _vid("v1", "")
    scraper._summarize_video(details, no_transcript_attempts=0)
    assert details["transcript_reason"] == "empty_content"


class TestDeliveryStalledAlert:
    """A run that defers everything used to be indistinguishable from a quiet
    day: both exit green. These cover the one case worth interrupting for."""

    def _sent(self, monkeypatch):
        sent = []
        monkeypatch.setattr(scraper, "send_telegram_text",
                            lambda token, chat_id, text: sent.append(text))
        return sent

    def test_alerts_when_everything_deferred_and_nothing_sent(self, monkeypatch):
        sent = self._sent(monkeypatch)
        fired = scraper._alert_delivery_stalled(
            "tok", "chat", {"budget_deferred": 37}, {"no_credits": 37},
        )
        assert fired is True and len(sent) == 1
        assert "37 video(s) deferred" in sent[0]
        assert "no_credits=37" in sent[0]
        # Must not read as data loss: the videos are queued, not dropped.
        assert "Nothing is lost" in sent[0]

    def test_silent_when_summaries_went_out(self, monkeypatch):
        sent = self._sent(monkeypatch)
        fired = scraper._alert_delivery_stalled(
            "tok", "chat", {"sent": 2, "budget_deferred": 5}, {"no_credits": 5},
        )
        assert fired is False and sent == []

    def test_silent_on_an_ordinary_quiet_run(self, monkeypatch):
        # No new videos at all is the normal overnight case, not an outage.
        sent = self._sent(monkeypatch)
        assert scraper._alert_delivery_stalled("tok", "chat", {}, {}) is False
        assert sent == []

    def test_no_transcript_deferrals_are_not_an_outage(self, monkeypatch):
        # Captions not published yet is the video's problem and resolves itself.
        sent = self._sent(monkeypatch)
        fired = scraper._alert_delivery_stalled(
            "tok", "chat", {"no_transcript_deferred": 3}, {"empty_content": 3},
        )
        assert fired is False and sent == []

    def test_without_telegram_config_it_does_not_crash_the_run(self, monkeypatch):
        sent = self._sent(monkeypatch)
        assert scraper._alert_delivery_stalled("", "", {"budget_deferred": 1}, {}) is False
        assert sent == []


def _main_harness(monkeypatch, feed, summarize):
    """Minimal main() wiring: one channel, no network, no Telegram."""
    for name, value in {"YOUTUBE_API_KEY": "yt", "TELEGRAM_TOKEN": "tok",
                        "TELEGRAM_CHANNEL_ID": "premium"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("MARKET_SIGNALS", "false")
    monkeypatch.delenv("DAILY_DIGEST", raising=False)
    monkeypatch.delenv("TELEGRAM_FREE_CHANNEL_ID", raising=False)
    monkeypatch.setattr(scraper, "read_channels", lambda p: [
        {"channel_id": "c1", "digest": False, "max_per_run": 10, "only": []},
    ])
    monkeypatch.setattr(scraper, "load_state", lambda p: {
        "channels": {"c1": {"last_video_id": "seed",
                            "last_published": "2026-01-01T00:00:00+00:00"}},
        "pending": {},
    })
    monkeypatch.setattr(scraper, "save_state", lambda p, s: None)
    monkeypatch.setattr(scraper, "save_to_json", lambda r, f: None)
    monkeypatch.setattr(scraper, "get_recent_videos", lambda k, c: feed)
    monkeypatch.setattr(scraper, "send_telegram_message", lambda *a: True)
    monkeypatch.setattr(scraper, "_summarize_video", summarize)
    warnings = []
    monkeypatch.setattr(scraper, "log_warn", lambda msg: warnings.append(msg))
    return warnings


def test_a_gemini_success_is_not_reported_as_a_transcript_failure(monkeypatch):
    # The run summary used to read "Transcript failures by reason: gemini_ok=4,
    # gemini_http_400=1" — four successful transcripts counted as failures,
    # because scraper kept its own copy of the success reasons and it drifted
    # when the Gemini source was added.
    def summarize(d, a, compact=False, want_signals=False, hours_since_first=None):
        d["transcript_reason"] = "gemini_ok"
        return ("S", "sent", True, None)

    warnings = _main_harness(monkeypatch, [_vid("v1", "2026-07-03T00:00:00+00:00")], summarize)
    scraper.main()
    assert not [w for w in warnings if "Transcript failures" in w]


def test_a_real_transcript_failure_is_still_reported(monkeypatch):
    def summarize(d, a, compact=False, want_signals=False, hours_since_first=None):
        d["transcript_reason"] = "gemini_too_large"
        return ("S", "no_transcript", True, None)

    warnings = _main_harness(monkeypatch, [_vid("v1", "2026-07-03T00:00:00+00:00")], summarize)
    scraper.main()
    assert [w for w in warnings if "Transcript failures" in w and "gemini_too_large" in w]
