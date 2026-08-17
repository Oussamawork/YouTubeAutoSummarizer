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


def test_digest_custom_title_escaped():
    # Per-channel digests use "New from <channel>" as the header.
    msg = tg._build_html_digest(DIGEST_ENTRIES, title="New from A&B")
    assert "<b>New from A&amp;B</b>" in msg
    assert "Daily digest" not in msg


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


# --- Free/premium teaser split ---


def test_build_teaser_takes_first_nonempty_line():
    assert tg.build_teaser("TL;DR line\n\n• bullet 1\n• bullet 2") == "TL;DR line"
    assert tg.build_teaser("\n\n  spaced first line  \nrest") == "spaced first line"


def test_build_teaser_empty_input_never_raises():
    assert tg.build_teaser("") == ""
    assert tg.build_teaser(None) == ""
    assert tg.build_teaser("\n \n") == ""


def test_html_teaser_escapes_and_includes_cta():
    msg = tg._build_html_teaser("A&B", "T<i>", "http://u?a=1&b=2", "tl;dr & more", "https://t.me/+inv")
    assert "&amp;" in msg and "&lt;i&gt;" in msg
    assert "<b>A&amp;B</b>" in msg  # our markup preserved, fields escaped
    assert "🔓 Full summary: https://t.me/+inv" in msg


def test_html_teaser_omits_cta_when_no_premium_url():
    msg = tg._build_html_teaser("C", "T", "http://u", "tl;dr", None)
    assert "🔓" not in msg


def test_plain_teaser_does_not_escape():
    msg = tg._build_plain_teaser("A&B", "T", "http://u", "tl;dr", "https://t.me/+inv")
    assert "A&B" in msg
    assert "&amp;" not in msg
    assert "https://t.me/+inv" in msg


def test_send_teaser_falls_back_to_plain(monkeypatch):
    sent = []

    def fake_post(url, data=None, timeout=None):
        sent.append(data)

        class R:
            status_code = 400 if data.get("parse_mode") else 200
            text = "bad entities"
        return R()

    monkeypatch.setattr(tg.requests, "post", fake_post)
    assert tg.send_telegram_teaser("tok", "free", "C", "T", "http://u", "tl;dr") is True
    assert sent[0].get("parse_mode") == "HTML"
    assert "parse_mode" not in sent[-1]
    assert all(d["chat_id"] == "free" for d in sent)


def test_send_teaser_empty_skips_without_posting(monkeypatch):
    monkeypatch.setattr(
        tg.requests, "post",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not post")),
    )
    assert tg.send_telegram_teaser("tok", "free", "C", "T", "http://u", "") is False


def test_digest_footer_rendered_and_escaped():
    html_msg = tg._build_html_digest(DIGEST_ENTRIES, footer="🔓 Full & more: https://t.me/+inv")
    assert html_msg.rstrip().endswith("🔓 Full &amp; more: https://t.me/+inv")
    plain_msg = tg._build_plain_digest(DIGEST_ENTRIES, footer="🔓 Full & more: https://t.me/+inv")
    assert "&amp;" not in plain_msg
    assert plain_msg.rstrip().endswith("🔓 Full & more: https://t.me/+inv")


def test_digest_no_footer_by_default():
    assert "🔓" not in tg._build_html_digest(DIGEST_ENTRIES)


# --- Plain-text sender (weekly pulse) ---


def test_send_text_plain_no_parse_mode(monkeypatch):
    sent = []

    def fake_post(url, data=None, timeout=None):
        sent.append(data)

        class R:
            status_code = 200
            text = ""
        return R()

    monkeypatch.setattr(tg.requests, "post", fake_post)
    assert tg.send_telegram_text("tok", "chat", "pulse <text> & stuff") is True
    assert len(sent) == 1
    assert "parse_mode" not in sent[0]
    assert sent[0]["text"] == "pulse <text> & stuff"


def test_send_text_empty_skips(monkeypatch):
    monkeypatch.setattr(
        tg.requests, "post",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not post")),
    )
    assert tg.send_telegram_text("tok", "chat", "  ") is False


def _photo(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"\x89PNG fake image bytes")
    return str(path)


def test_photo_album_uses_media_group_with_single_caption(tmp_path, monkeypatch):
    import json as jsonlib
    calls = []

    def fake_post(url, data=None, files=None, timeout=None):
        calls.append((url, data, dict(files)))

        class R:
            status_code = 200
            text = ""
        return R()

    monkeypatch.setattr(tg.requests, "post", fake_post)
    paths = [_photo(tmp_path, f"{i}.png") for i in range(3)]
    assert tg.send_telegram_photo_album("tok", "chat", paths, caption="Weekly charts") is True
    assert len(calls) == 1
    url, data, files = calls[0]
    assert "sendMediaGroup" in url
    media = jsonlib.loads(data["media"])
    assert len(media) == 3 and len(files) == 3
    assert media[0]["caption"] == "Weekly charts"
    assert all("caption" not in item for item in media[1:])
    assert all(item["media"].startswith("attach://") for item in media)


def test_photo_album_single_photo_uses_send_photo(tmp_path, monkeypatch):
    calls = []

    def fake_post(url, data=None, files=None, timeout=None):
        calls.append((url, data))

        class R:
            status_code = 200
            text = ""
        return R()

    monkeypatch.setattr(tg.requests, "post", fake_post)
    assert tg.send_telegram_photo_album("tok", "chat", [_photo(tmp_path, "a.png")],
                                        caption="One chart") is True
    url, data = calls[0]
    assert "sendPhoto" in url
    assert data["caption"] == "One chart"


def test_photo_album_skips_missing_files_and_fails_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(
        tg.requests, "post",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not post")),
    )
    assert tg.send_telegram_photo_album("tok", "chat", [str(tmp_path / "gone.png")]) is False
    assert tg.send_telegram_photo_album("tok", "chat", []) is False


def test_photo_album_reports_rejection(tmp_path, monkeypatch):
    def fake_post(url, data=None, files=None, timeout=None):
        class R:
            status_code = 400
            text = "bad request"
        return R()

    monkeypatch.setattr(tg.requests, "post", fake_post)
    paths = [_photo(tmp_path, f"{i}.png") for i in range(2)]
    assert tg.send_telegram_photo_album("tok", "chat", paths) is False
