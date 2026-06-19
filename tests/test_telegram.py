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
