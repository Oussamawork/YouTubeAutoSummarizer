"""Delivery is part of "decided": a video's watermark advances only once
Telegram has accepted its message, and finished text that could not be
delivered is held in the pending record and re-sent on a later run without
paying for the transcript or the LLM call again."""
import scraper


def _vid(video_id, published, title="T"):
    return {"video_id": video_id, "channel_name": "Chan", "video_title": title,
            "video_url": f"http://u/{video_id}", "published_at": published}


def _harness(monkeypatch, feed, *, state=None, digest=False, daily_digest=False,
             send=None, send_digest=None, summarize=None, deadline=0):
    """Drive main() over one channel with everything mocked. `feed` is newest
    first, like a real RSS feed. Returns a dict of the recorded calls plus the
    live state object."""
    for name, value in {"YOUTUBE_API_KEY": "yt", "TELEGRAM_TOKEN": "tok",
                        "TELEGRAM_CHANNEL_ID": "premium"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("MARKET_SIGNALS", "false")
    monkeypatch.delenv("TELEGRAM_FREE_CHANNEL_ID", raising=False)
    if daily_digest:
        monkeypatch.setenv("DAILY_DIGEST", "true")
    else:
        monkeypatch.delenv("DAILY_DIGEST", raising=False)
    monkeypatch.setattr(scraper, "RUN_DEADLINE_MINUTES", deadline)

    state = state or {
        "channels": {"c1": {"last_video_id": "seed",
                            "last_published": "2026-01-01T00:00:00+00:00"}},
        "pending": {},
    }
    calls = {"summarized": [], "sent": [], "digests": [], "saves": 0, "state": state}
    monkeypatch.setattr(scraper, "read_channels", lambda p: [
        {"channel_id": "c1", "digest": digest, "max_per_run": 10, "only": []},
    ])
    monkeypatch.setattr(scraper, "load_state", lambda p: state)

    def save(path, s):
        calls["saves"] += 1
    monkeypatch.setattr(scraper, "save_state", save)
    monkeypatch.setattr(scraper, "get_recent_videos", lambda k, c: list(feed))

    def default_summarize(d, a, compact=False, want_signals=False, hours_since_first=None):
        calls["summarized"].append(d["video_id"])
        return (f"summary of {d['video_id']}", "sent", True, None)
    monkeypatch.setattr(scraper, "_summarize_video", summarize or default_summarize)

    def default_send(token, chat, channel, title, url, published, body):
        calls["sent"].append(body)
        return True
    monkeypatch.setattr(scraper, "send_telegram_message", send or default_send)

    def default_digest(token, chat, entries, title="Daily digest", footer=None):
        calls["digests"].append([e["body"] for e in entries])
        return True
    monkeypatch.setattr(scraper, "send_telegram_digest", send_digest or default_digest)
    scraper.main()
    return calls


def test_failed_send_keeps_the_video_pending_with_its_text(monkeypatch):
    calls = _harness(monkeypatch, [_vid("v1", "2026-07-03T00:00:00+00:00")],
                     send=lambda *a: False)
    state = calls["state"]
    # The watermark must not move: the reader never got the summary.
    assert state["channels"]["c1"]["last_video_id"] == "seed"
    held = state["pending"]["v1"]["undelivered"]
    assert held["body"] == "summary of v1"
    assert held["outcome"] == "sent"
    assert held["video_title"] == "T"


def test_held_text_is_redelivered_without_summarizing_again(monkeypatch):
    state = {
        "channels": {"c1": {"last_video_id": "seed",
                            "last_published": "2026-01-01T00:00:00+00:00"}},
        "pending": {"v1": {"channel_id": "c1", "attempts": 0, "undelivered": {
            "video_id": "v1", "body": "held summary", "outcome": "sent",
            "channel_name": "Chan", "video_title": "T", "video_url": "http://u/v1",
            "published_at": "2026-07-03T00:00:00+00:00"}}},
    }
    calls = _harness(monkeypatch, [_vid("v1", "2026-07-03T00:00:00+00:00")], state=state)
    assert calls["summarized"] == []          # no transcript credit, no LLM call
    assert calls["sent"] == ["held summary"]
    assert state["pending"] == {}
    assert state["channels"]["c1"]["last_video_id"] == "v1"


def test_held_video_gone_from_the_feed_is_still_delivered(monkeypatch):
    held = {"video_id": "old", "body": "held summary", "outcome": "sent",
            "channel_name": "Chan", "video_title": "Old", "video_url": "http://u/old",
            "published_at": "2026-06-01T00:00:00+00:00"}
    state = {
        "channels": {"c1": {"last_video_id": "seed",
                            "last_published": "2026-01-01T00:00:00+00:00"}},
        "pending": {"old": {"channel_id": "c1", "attempts": 0, "undelivered": held}},
    }
    # The feed has moved on; "old" is not in it any more.
    calls = _harness(monkeypatch, [_vid("v9", "2026-07-03T00:00:00+00:00")], state=state)
    assert "held summary" in calls["sent"]
    assert "old" not in state["pending"]      # delivered, not evicted


def test_eviction_never_drops_held_text():
    pending = {"a": {"channel_id": "c1", "attempts": 2},
               "b": {"channel_id": "c1", "attempts": 0,
                     "undelivered": {"body": "x", "outcome": "sent"}}}
    assert scraper._evict_orphaned_pending(pending, "c1", set()) == 1
    assert set(pending) == {"b"}


def test_digest_videos_are_finalized_only_after_the_digest_lands(monkeypatch):
    feed = [_vid("v2", "2026-07-04T00:00:00+00:00"), _vid("v1", "2026-07-03T00:00:00+00:00")]
    calls = _harness(monkeypatch, feed, digest=True)
    state = calls["state"]
    assert calls["digests"] == [["summary of v1", "summary of v2"]]
    assert state["pending"] == {}
    assert state["channels"]["c1"]["last_video_id"] == "v2"


def test_failed_digest_holds_every_entry_for_the_next_run(monkeypatch):
    feed = [_vid("v2", "2026-07-04T00:00:00+00:00"), _vid("v1", "2026-07-03T00:00:00+00:00")]
    calls = _harness(monkeypatch, feed, digest=True, send_digest=lambda *a, **k: False)
    state = calls["state"]
    assert state["channels"]["c1"]["last_video_id"] == "seed"
    assert {k for k in state["pending"]} == {"v1", "v2"}
    assert state["pending"]["v2"]["undelivered"]["body"] == "summary of v2"


def test_daily_digest_failure_holds_entries_too(monkeypatch):
    feed = [_vid("v2", "2026-07-04T00:00:00+00:00"), _vid("v1", "2026-07-03T00:00:00+00:00")]
    calls = _harness(monkeypatch, feed, daily_digest=True, send_digest=lambda *a, **k: False)
    assert set(calls["state"]["pending"]) == {"v1", "v2"}
    assert calls["state"]["channels"]["c1"]["last_video_id"] == "seed"


def test_deferral_notices_do_not_gate_the_watermark(monkeypatch):
    # A deferral notice that fails to send is not a lost summary: the video is
    # retried anyway, and must not gain an "undelivered" block.
    def summarize(d, a, compact=False, want_signals=False, hours_since_first=None):
        return ("⏳ deferred", "quota_deferred", False, None)
    calls = _harness(monkeypatch, [_vid("v1", "2026-07-03T00:00:00+00:00")],
                     send=lambda *a: False, summarize=summarize)
    record = calls["state"]["pending"]["v1"]
    assert "undelivered" not in record


def test_run_deadline_defers_videos_not_yet_started(monkeypatch):
    feed = [_vid("v2", "2026-07-04T00:00:00+00:00"), _vid("v1", "2026-07-03T00:00:00+00:00")]
    # run_started, then one reading before each video: the second reading is
    # far past a one-minute deadline.
    clock = iter([0.0, 0.0, 10_000.0, 10_000.0, 10_000.0])
    monkeypatch.setattr(scraper.time, "monotonic", lambda: next(clock))
    calls = _harness(monkeypatch, feed, deadline=1)
    # The oldest video started before the deadline; the newer one waits, and
    # the watermark stops at the one that went out.
    assert calls["summarized"] == ["v1"]
    assert calls["state"]["channels"]["c1"]["last_video_id"] == "v1"
