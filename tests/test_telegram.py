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
