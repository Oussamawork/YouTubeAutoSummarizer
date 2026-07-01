"""Tests for Telegram message building (HTML escaping vs plain text)."""
import sendToTelegram as tg


def test_html_message_escapes_entities():
    msg = tg._build_html_message(
        "A&B", "T<i>", "http://u?a=1&b=2", "2024", "sum & <stuff>"
    )
    assert "&amp;" in msg
    assert "&lt;" in msg
    assert "<b>Channel</b>" in msg  # our own markup is preserved


def test_plain_message_does_not_escape():
    msg = tg._build_plain_message("A&B", "T<i>", "http://u", "2024", "s")
    assert "A&B" in msg
    assert "&amp;" not in msg


def test_max_len_constant():
    assert tg.TELEGRAM_MAX_LEN == 4096


def _no_dangling_entity(chunk):
    amp = chunk.rfind("&")
    return amp == -1 or ";" in chunk[amp:]


def test_split_short_text_single_chunk():
    assert tg._split_message("hello", 4096) == ["hello"]


def test_split_respects_limit_and_newlines():
    text = "\n".join(f"line{i}" for i in range(1000))
    chunks = tg._split_message(text, 100)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    assert "line0" in chunks[0]
    assert "line999" in chunks[-1]


def test_split_never_cuts_inside_entity():
    text = "a" * 98 + "&amp;" + "b" * 98
    chunks = tg._split_message(text, 100)
    assert all(len(c) <= 100 for c in chunks)
    assert all(_no_dangling_entity(c) for c in chunks)


DIGEST_ENTRIES = [
    {
        "channel_name": "A&B",
        "video_title": "Title <one>",
        "video_url": "http://u?a=1&b=2",
        "published_at": "2026-06-01",
        "body": "sum & <stuff>",
    },
    {
        "channel_name": "Chan2",
        "video_title": "Title two",
        "video_url": "http://u2",
        "published_at": "2026-06-02",
        "body": "second summary",
    },
]


def test_html_digest_escapes_and_sections():
    msg = tg._build_html_digest(DIGEST_ENTRIES)
    assert "2 new videos" in msg
    assert msg.count(tg.DIGEST_DIVIDER) == 2  # header + 2 sections
    assert "&amp;" in msg and "&lt;one&gt;" in msg
    assert "<b>A&amp;B</b>" in msg  # our own markup preserved, fields escaped


def test_plain_digest_does_not_escape():
    msg = tg._build_plain_digest(DIGEST_ENTRIES)
    assert "A&B" in msg
    assert "&amp;" not in msg


def test_send_digest_falls_back_to_plain(monkeypatch):
    sent = []

    def fake_post(url, data=None, timeout=None):
        sent.append(data)

        class R:
            # Reject the HTML attempt, accept the plain one.
            status_code = 400 if data.get("parse_mode") else 200
            text = "bad entities"
        return R()

    monkeypatch.setattr(tg.requests, "post", fake_post)
    assert tg.send_telegram_digest("tok", "chat", DIGEST_ENTRIES) is True
    assert sent[0].get("parse_mode") == "HTML"
    assert "parse_mode" not in sent[-1]


def test_send_digest_empty_is_noop(monkeypatch):
    monkeypatch.setattr(
        tg.requests, "post",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not post")),
    )
    assert tg.send_telegram_digest("tok", "chat", []) is True


def test_post_splits_long_message(monkeypatch):
    sent = []

    class R:
        status_code = 200
        text = ""

    def fake_post(url, data=None, timeout=None):
        sent.append(data["text"])
        return R()

    monkeypatch.setattr(tg.requests, "post", fake_post)
    long_text = "x\n" * 5000  # ~10k chars, well over the 4096 limit
    assert tg._post("tok", "chat", long_text, parse_mode="HTML") is True
    assert len(sent) >= 2
    assert all(len(t) <= tg.TELEGRAM_MAX_LEN for t in sent)
