import html

import requests
from log import log_info, log_warn, log_error

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
TELEGRAM_TIMEOUT = 15
TELEGRAM_MAX_LEN = 4096  # Telegram's hard limit on message text length


def _build_html_message(channel_name, video_title, video_url, published_at, summary):
    """Build an HTML-formatted message with all dynamic fields escaped.

    HTML parse mode only needs &, <, > (and quotes) escaped, which html.escape
    handles — far safer than Telegram's legacy Markdown, which breaks on any
    stray * _ ` [ in LLM-generated summaries.
    """
    e = html.escape
    return (
        f"🎥 <b>Channel</b>: {e(channel_name)}\n"
        f"📅 <b>Published At</b>: {e(published_at)}\n"
        f"📌 <b>Video Title</b>: {e(video_title)}\n"
        f'🔗 <a href="{e(video_url)}">Watch on YouTube</a>\n\n'
        f"📜 <b>Summary</b>:\n{e(summary)}"
    )


def _build_plain_message(channel_name, video_title, video_url, published_at, summary):
    """Plain-text fallback — no parse mode, so it can never fail to parse."""
    return (
        f"🎥 Channel: {channel_name}\n"
        f"📅 Published At: {published_at}\n"
        f"📌 Video Title: {video_title}\n"
        f"🔗 {video_url}\n\n"
        f"📜 Summary:\n{summary}"
    )


def _post(bot_token, chat_id, text, parse_mode=None):
    """POST one message to Telegram. Returns True on HTTP 200."""
    url = TELEGRAM_API.format(token=bot_token)
    data = {"chat_id": chat_id, "text": text[:TELEGRAM_MAX_LEN]}
    if parse_mode:
        data["parse_mode"] = parse_mode
    try:
        resp = requests.post(url, data=data, timeout=TELEGRAM_TIMEOUT)
    except requests.RequestException as e:
        log_error(f"Telegram request failed: {e}")
        return False

    if resp.status_code == 200:
        return True
    log_warn(f"Telegram send failed ({resp.status_code}): {resp.text[:200]}")
    return False


def send_telegram_message(bot_token, chat_id, channel_name, video_title, video_url, published_at, summary):
    """
    Send a formatted message to a Telegram channel.

    Tries HTML formatting first; if Telegram rejects it (e.g. an entity-parse
    error), retries once as plain text so a summary is never lost to a
    formatting issue. Returns True if either attempt succeeds.
    """
    html_message = _build_html_message(channel_name, video_title, video_url, published_at, summary)
    if _post(bot_token, chat_id, html_message, parse_mode="HTML"):
        log_info("Message sent successfully to Telegram channel.")
        return True

    log_warn("HTML send failed; retrying as plain text.")
    plain_message = _build_plain_message(channel_name, video_title, video_url, published_at, summary)
    if _post(bot_token, chat_id, plain_message):
        log_info("Message sent to Telegram as plain text (fallback).")
        return True

    log_error("Failed to send Telegram message (HTML and plain text both failed).")
    return False
