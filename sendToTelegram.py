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


def _split_message(text, limit=TELEGRAM_MAX_LEN):
    """
    Split text into chunks no longer than `limit`, so long summaries aren't
    silently truncated at Telegram's 4096-char cap.

    Splits preferentially on a newline boundary (then a space), which keeps our
    messages safe under HTML parse mode: tags and entities never contain
    newlines, so a newline split never lands inside `<b>...</b>` or `&amp;`. For
    a pathological single line longer than the limit, it hard-splits but backs
    off so the cut never falls inside an HTML entity. Always returns >=1 chunk.
    """
    chunks = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind("\n", 0, limit)
        if cut <= 0:
            cut = remaining.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        # Don't cut inside an HTML entity like &amp; — back up to before the '&'.
        amp = remaining.rfind("&", max(0, cut - 10), cut)
        if amp != -1 and ";" not in remaining[amp:cut]:
            cut = amp
        if cut <= 0:
            cut = limit
        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")
    chunks.append(remaining)
    return chunks


def _post(bot_token, chat_id, text, parse_mode=None):
    """
    POST a message to Telegram, splitting it across multiple sends if it exceeds
    the 4096-char limit (rather than silently truncating). Returns True only if
    every chunk was accepted (HTTP 200).
    """
    url = TELEGRAM_API.format(token=bot_token)
    for chunk in _split_message(text):
        data = {"chat_id": chat_id, "text": chunk}
        if parse_mode:
            data["parse_mode"] = parse_mode
        try:
            resp = requests.post(url, data=data, timeout=TELEGRAM_TIMEOUT)
        except requests.RequestException as e:
            log_error(f"Telegram request failed: {e}")
            return False
        if resp.status_code != 200:
            log_warn(f"Telegram send failed ({resp.status_code}): {resp.text[:200]}")
            return False
    return True


# Separator between per-video sections in a digest. Newline-heavy on purpose:
# the splitter cuts on newlines, so a digest too long for one message always
# splits cleanly between sections or lines, never inside an HTML tag.
DIGEST_DIVIDER = "\n\n— — — — —\n\n"


def _build_html_digest(entries):
    """One HTML message covering every new video from a run (digest mode)."""
    e = html.escape
    parts = [f"🗞️ <b>Daily digest</b> — {len(entries)} new videos"]
    for entry in entries:
        parts.append(
            f"🎥 <b>{e(entry['channel_name'])}</b>\n"
            f"📌 {e(entry['video_title'])}\n"
            f'🔗 <a href="{e(entry["video_url"])}">Watch on YouTube</a> · 📅 {e(entry["published_at"])}\n\n'
            f"{e(entry['body'])}"
        )
    return DIGEST_DIVIDER.join(parts)


def _build_plain_digest(entries):
    """Plain-text digest fallback — no parse mode, so it can never fail to parse."""
    parts = [f"🗞️ Daily digest — {len(entries)} new videos"]
    for entry in entries:
        parts.append(
            f"🎥 {entry['channel_name']}\n"
            f"📌 {entry['video_title']}\n"
            f"🔗 {entry['video_url']} · 📅 {entry['published_at']}\n\n"
            f"{entry['body']}"
        )
    return DIGEST_DIVIDER.join(parts)


def send_telegram_digest(bot_token, chat_id, entries):
    """
    Send one combined message for several videos (digest mode). Each entry is a
    dict with channel_name, video_title, video_url, published_at and body keys.
    Tries HTML first, then plain text, like send_telegram_message; anything over
    the 4096-char limit is split across messages by _post.
    """
    if not entries:
        return True
    if _post(bot_token, chat_id, _build_html_digest(entries), parse_mode="HTML"):
        log_info(f"Digest with {len(entries)} entries sent to Telegram.")
        return True

    log_warn("HTML digest send failed; retrying as plain text.")
    if _post(bot_token, chat_id, _build_plain_digest(entries)):
        log_info("Digest sent to Telegram as plain text (fallback).")
        return True

    log_error("Failed to send Telegram digest (HTML and plain text both failed).")
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
