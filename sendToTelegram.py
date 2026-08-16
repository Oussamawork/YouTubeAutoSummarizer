import html
import json
import os

import requests
from log import log_info, log_warn, log_error

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
TELEGRAM_PHOTO_API = "https://api.telegram.org/bot{token}/sendPhoto"
TELEGRAM_MEDIA_GROUP_API = "https://api.telegram.org/bot{token}/sendMediaGroup"
TELEGRAM_TIMEOUT = 15
# Photo uploads carry megabytes, not kilobytes; give them a wider window.
TELEGRAM_UPLOAD_TIMEOUT = 60
TELEGRAM_MAX_LEN = 4096  # Telegram's hard limit on message text length
TELEGRAM_ALBUM_MAX = 10  # Telegram's hard limit on media items per album
TELEGRAM_CAPTION_MAX = 1024  # Telegram's hard limit on media caption length


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


def _build_html_digest(entries, title="Daily digest", footer=None):
    """One HTML message covering every new video from a run (digest mode)."""
    e = html.escape
    parts = [f"🗞️ <b>{e(title)}</b> — {len(entries)} new videos"]
    for entry in entries:
        parts.append(
            f"🎥 <b>{e(entry['channel_name'])}</b>\n"
            f"📌 {e(entry['video_title'])}\n"
            f'🔗 <a href="{e(entry["video_url"])}">Watch on YouTube</a> · 📅 {e(entry["published_at"])}\n\n'
            f"{e(entry['body'])}"
        )
    if footer:
        parts.append(e(footer))
    return DIGEST_DIVIDER.join(parts)


def _build_plain_digest(entries, title="Daily digest", footer=None):
    """Plain-text digest fallback — no parse mode, so it can never fail to parse."""
    parts = [f"🗞️ {title} — {len(entries)} new videos"]
    for entry in entries:
        parts.append(
            f"🎥 {entry['channel_name']}\n"
            f"📌 {entry['video_title']}\n"
            f"🔗 {entry['video_url']} · 📅 {entry['published_at']}\n\n"
            f"{entry['body']}"
        )
    if footer:
        parts.append(footer)
    return DIGEST_DIVIDER.join(parts)


def send_telegram_digest(bot_token, chat_id, entries, title="Daily digest", footer=None):
    """
    Send one combined message for several videos (digest mode). Each entry is a
    dict with channel_name, video_title, video_url, published_at and body keys;
    `title` heads the message (e.g. "Daily digest", "New from <channel>") and
    `footer`, when given, closes it (e.g. a link to the premium channel).
    Tries HTML first, then plain text, like send_telegram_message; anything over
    the 4096-char limit is split across messages by _post.
    """
    if not entries:
        return True
    if _post(bot_token, chat_id, _build_html_digest(entries, title, footer), parse_mode="HTML"):
        log_info(f"Digest with {len(entries)} entries sent to Telegram.")
        return True

    log_warn("HTML digest send failed; retrying as plain text.")
    if _post(bot_token, chat_id, _build_plain_digest(entries, title, footer)):
        log_info("Digest sent to Telegram as plain text (fallback).")
        return True

    log_error("Failed to send Telegram digest (HTML and plain text both failed).")
    return False


def send_telegram_text(bot_token, chat_id, text):
    """
    Send a plain-text message (no parse mode, so it can never fail to parse and
    needs no escaping). Long texts are split by _post. Used for non-video
    messages like the weekly market pulse. Returns True on success.
    """
    if not (text or "").strip():
        log_warn("Empty text; nothing to send to Telegram.")
        return False
    if _post(bot_token, chat_id, text):
        log_info("Text message sent to Telegram.")
        return True
    log_error("Failed to send Telegram text message.")
    return False


def send_telegram_photo_album(bot_token, chat_id, photo_paths, caption=None):
    """
    Send local image files as one photo album (sendMediaGroup), with `caption`
    shown under the album (Telegram displays the first item's caption). A
    single photo goes through sendPhoto instead — sendMediaGroup requires at
    least two items. More than TELEGRAM_ALBUM_MAX photos are split across
    albums. Returns True only if every send was accepted; missing files are
    skipped with a warning, and no failure ever raises.
    """
    paths = []
    for path in photo_paths or []:
        if os.path.isfile(path):
            paths.append(path)
        else:
            log_warn(f"Album photo missing, skipping: {path}")
    if not paths:
        log_warn("No photos to send to Telegram.")
        return False

    caption = (caption or "")[:TELEGRAM_CAPTION_MAX] or None
    for start in range(0, len(paths), TELEGRAM_ALBUM_MAX):
        batch = paths[start:start + TELEGRAM_ALBUM_MAX]
        first_batch = start == 0
        try:
            if len(batch) == 1:
                with open(batch[0], "rb") as f:
                    data = {"chat_id": chat_id}
                    if caption and first_batch:
                        data["caption"] = caption
                    resp = requests.post(
                        TELEGRAM_PHOTO_API.format(token=bot_token), data=data,
                        files={"photo": f}, timeout=TELEGRAM_UPLOAD_TIMEOUT,
                    )
            else:
                media, files, handles = [], {}, []
                try:
                    for i, path in enumerate(batch):
                        key = f"photo{i}"
                        handle = open(path, "rb")
                        handles.append(handle)
                        files[key] = (os.path.basename(path), handle, "image/png")
                        item = {"type": "photo", "media": f"attach://{key}"}
                        if caption and first_batch and i == 0:
                            item["caption"] = caption
                        media.append(item)
                    resp = requests.post(
                        TELEGRAM_MEDIA_GROUP_API.format(token=bot_token),
                        data={"chat_id": chat_id, "media": json.dumps(media)},
                        files=files, timeout=TELEGRAM_UPLOAD_TIMEOUT,
                    )
                finally:
                    for handle in handles:
                        handle.close()
        except (OSError, requests.RequestException) as e:
            log_error(f"Telegram photo album send failed: {e}")
            return False
        if resp.status_code != 200:
            log_warn(f"Telegram album send failed ({resp.status_code}): {resp.text[:200]}")
            return False
    log_info(f"Photo album with {len(paths)} image(s) sent to Telegram.")
    return True


def build_teaser(summary):
    """
    First non-empty line of a summary — the TL;DR sentence the prompt format
    guarantees — for the free public channel. Returns "" for empty input
    (never raises), which callers treat as "nothing to tease".
    """
    if not summary:
        return ""
    for line in summary.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _build_html_teaser(channel_name, video_title, video_url, teaser, premium_url=None):
    """Short HTML teaser card: TL;DR + video link, plus an optional premium CTA.

    The CTA URL is inserted as escaped text (not an anchor) — Telegram clients
    auto-link bare URLs, and plain text survives the plain-text fallback too.
    """
    e = html.escape
    lines = [
        f"🎥 <b>{e(channel_name)}</b>",
        f"📌 {e(video_title)}",
        f"💡 {e(teaser)}",
        f'🔗 <a href="{e(video_url)}">Watch on YouTube</a>',
    ]
    if premium_url:
        lines.append(f"🔓 Full summary: {e(premium_url)}")
    return "\n".join(lines)


def _build_plain_teaser(channel_name, video_title, video_url, teaser, premium_url=None):
    """Plain-text teaser fallback — no parse mode, so it can never fail to parse."""
    lines = [
        f"🎥 {channel_name}",
        f"📌 {video_title}",
        f"💡 {teaser}",
        f"🔗 {video_url}",
    ]
    if premium_url:
        lines.append(f"🔓 Full summary: {premium_url}")
    return "\n".join(lines)


def send_telegram_teaser(bot_token, chat_id, channel_name, video_title, video_url, teaser, premium_url=None):
    """
    Send a short teaser (TL;DR + link) to the free public channel, with an
    optional "full summary" CTA pointing at the premium channel. Same
    HTML-then-plain fallback as send_telegram_message. Returns True on success;
    an empty teaser is skipped (False) without contacting Telegram.
    """
    if not teaser:
        log_warn("Empty teaser; skipping free-channel send.")
        return False

    html_message = _build_html_teaser(channel_name, video_title, video_url, teaser, premium_url)
    if _post(bot_token, chat_id, html_message, parse_mode="HTML"):
        log_info("Teaser sent to free Telegram channel.")
        return True

    log_warn("HTML teaser send failed; retrying as plain text.")
    plain_message = _build_plain_teaser(channel_name, video_title, video_url, teaser, premium_url)
    if _post(bot_token, chat_id, plain_message):
        log_info("Teaser sent to free Telegram channel as plain text (fallback).")
        return True

    log_error("Failed to send Telegram teaser (HTML and plain text both failed).")
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
