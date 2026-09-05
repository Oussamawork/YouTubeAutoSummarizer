import argparse
import re
import requests
import sys
import time
from dotenv import load_dotenv
from defusedxml import ElementTree as SafeET
from transcript import (
    TRANSCRIPT_SUCCESS_REASONS,
    get_transcript_from_video,
    budget_status,
)
from helpers import (
    read_channels, clean_summary, load_state, save_state, env_int, env_float,
    env_flag, append_jsonl, title_matches,
)
from signals import extract_signals, summarize_with_signals
from summarizer import (
    summarize_transcript,
    INSUFFICIENT_TRANSCRIPT_SENTINEL,
    QUOTA_EXHAUSTED_SENTINEL,
    TRUNCATED_SENTINEL,
)
from log import log_info, log_error, log_warn, log_debug, redact
import gemini_quota
from sendToTelegram import (
    send_telegram_message, send_telegram_digest, send_telegram_teaser,
    send_telegram_text, build_teaser,
)
import os
from datetime import datetime, timezone

# Load environment variables from .env file
load_dotenv('.env')

# YouTube API request tuning
YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3/search"
# Channels endpoint, used to resolve an @handle to its channel id (1 quota unit
# per call, vs 100 for a search) so channel_ids.txt can list handles directly.
YOUTUBE_CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"
# videos.list: duration + caption flag for up to 50 ids per request, 1 quota
# unit each (of 10,000/day) — cheap enough to screen every candidate before
# spending a transcript credit on it.
YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
# Channel RSS feed: free, keyless, no API quota, and lists the last ~15 uploads
# (so channels that upload more than once between runs aren't missed). Primary
# video source; the Data API is only the fallback.
RSS_FEED_URL = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
# Keyless oEmbed endpoint, used to fetch title/channel for on-demand summaries.
OEMBED_URL = "https://www.youtube.com/oembed"
REQUEST_TIMEOUT = 15          # seconds before a hung request is abandoned
MAX_RETRIES = 3               # attempts for transient failures
RETRY_BACKOFF = 2            # base seconds, multiplied by the attempt number
TRANSIENT_STATUS = {429, 500, 502, 503, 504}

# Atom XML namespaces used by YouTube channel feeds.
ATOM_NS = "{http://www.w3.org/2005/Atom}"
YT_NS = "{http://www.youtube.com/xml/schemas/2015}"

# File that remembers what was already processed per channel (dedup state).
STATE_FILE = "seen_videos.json"

# Append-only store of per-video summaries + extracted market signals
# (committed back by the daily workflow when MARKET_SIGNALS is enabled).
SIGNALS_FILE = "data/signals.jsonl"

# Cap on videos processed per channel per run; the rest wait for the next run.
# 0 (the default) means no cap, so everything a channel published is processed
# in the run that finds it rather than trickling out over later runs. Set a
# positive value to bound how much a prolific channel can post in one run.
MAX_VIDEOS_PER_RUN = env_int("MAX_VIDEOS_PER_RUN", 0)
# Captions (especially auto-generated ones) often appear hours after upload, so
# a video with no transcript is retried this many runs before giving up.
NO_TRANSCRIPT_MAX_ATTEMPTS = env_int("NO_TRANSCRIPT_MAX_ATTEMPTS", 8)
# Giving up is time-gated as well as count-gated: a video is never written off
# until it has been chased for this long, however many runs that took. A count
# alone is the wrong unit once the polling rate can change — at one run every
# two hours, three attempts is six hours, and auto-captions routinely take
# longer than that to appear. The transcript is the whole product, so the
# default errs heavily toward keeping the video.
NO_TRANSCRIPT_MIN_HOURS = env_float("NO_TRANSCRIPT_MIN_HOURS", 36.0)
# Minimum gap between retries of one deferred video. Supadata answering "no
# captions" costs a credit, so without this, frequent polling would spend one
# per run on every pending video. 0 disables the throttle.
PENDING_RETRY_MIN_HOURS = env_float("PENDING_RETRY_MIN_HOURS", 3.0)
# Videos shorter than this are skipped before any transcript is fetched:
# Shorts and clips rarely carry usable captions and aren't worth summarizing.
# 0 disables the check.
MIN_VIDEO_SECONDS = env_int("MIN_VIDEO_SECONDS", 90)
# Wall-clock budget for one run, in minutes. A single video can legitimately
# take a long time (a Gemini transcript is ~2 minutes per model tried, a
# summary up to several LLM calls), and the workflow kills the job at 45. Videos
# not started by the deadline defer to the next run instead of being cut off
# mid-flight, and the run still gets to flush its digests and save its state.
# 0 disables the deadline.
RUN_DEADLINE_MINUTES = env_float("RUN_DEADLINE_MINUTES", 35.0)


def _hours_since(timestamp, now=None):
    """Hours elapsed since an ISO timestamp, or None if missing/unparsable."""
    moment = _parse_timestamp(timestamp)
    if moment is None:
        return None
    now = now or datetime.now(timezone.utc)
    return (now - moment).total_seconds() / 3600.0


def _retry_wait_remaining(record, now=None):
    """
    Hours still to wait before re-fetching a deferred video, so a fast polling
    schedule doesn't spend a transcript credit per run on the same video (a
    "no captions" answer costs one). Records written before retry timestamps
    existed, and videos deferred for reasons that cost nothing (budget), carry
    no `last_attempt` and are retried immediately.
    """
    if not record or PENDING_RETRY_MIN_HOURS <= 0:
        return 0.0
    elapsed = _hours_since(record.get("last_attempt"), now)
    if elapsed is None:
        return 0.0
    return max(0.0, PENDING_RETRY_MIN_HOURS - elapsed)


def _parse_timestamp(value):
    """Parse an ISO-8601 timestamp into an aware datetime, or None if invalid."""
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def _parse_rss_feed(xml_text):
    """
    Parse a YouTube channel RSS feed into video dicts, newest first (feed order).
    Returns None when the XML is unparseable (caller falls back to the Data API);
    an empty list is a valid "channel has no videos" answer.
    """
    try:
        root = SafeET.fromstring(xml_text)
    except Exception as e:
        log_warn(f"Could not parse RSS feed XML: {e}")
        return None

    videos = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        video_id_el = entry.find(f"{YT_NS}videoId")
        video_id = (video_id_el.text or "").strip() if video_id_el is not None else ""
        if not video_id:
            continue
        title_el = entry.find(f"{ATOM_NS}title")
        author_el = entry.find(f"{ATOM_NS}author/{ATOM_NS}name")
        published_el = entry.find(f"{ATOM_NS}published")
        videos.append({
            "video_id": video_id,
            "channel_name": (author_el.text or "").strip() if author_el is not None else "Unknown channel",
            "video_title": (title_el.text or "").strip() if title_el is not None else "Untitled",
            "video_url": f"https://www.youtube.com/watch?v={video_id}",
            "published_at": (published_el.text or "").strip() if published_el is not None else "",
        })
    return videos


def get_recent_videos(youtube_api_key, channel_id):
    """
    List a channel's recent videos, newest first.

    Primary source is the channel's public RSS feed (no key, no API quota,
    last ~15 uploads). Falls back to the YouTube Data API (latest video only,
    100 quota units per call) when the feed can't be fetched or parsed.
    """
    url = RSS_FEED_URL.format(channel_id=channel_id)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            log_warn(f"RSS feed request error (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
                continue
            break

        if response.status_code == 200:
            videos = _parse_rss_feed(response.text)
            if videos is None:
                break  # unparseable — fall back to the Data API
            log_info(f"RSS feed listed {len(videos)} videos for channel {channel_id}.")
            return videos

        if response.status_code in TRANSIENT_STATUS and attempt < MAX_RETRIES:
            log_warn(
                f"Transient RSS feed status {response.status_code} "
                f"(attempt {attempt}/{MAX_RETRIES}); retrying."
            )
            time.sleep(RETRY_BACKOFF * attempt)
            continue

        log_warn(f"RSS feed returned {response.status_code} for channel {channel_id}.")
        break

    log_warn("RSS feed unavailable; falling back to the YouTube Data API (latest video only).")
    video = get_latest_video(youtube_api_key, channel_id)
    return [video] if video else []


_ISO_DURATION_RE = re.compile(
    r"^P(?:(?P<days>\d+)D)?T?(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?$"
)


def parse_iso_duration(value):
    """
    Seconds from an ISO-8601 duration as returned by the YouTube API
    ("PT4M13S"), or None when it can't be parsed. Never raises.
    """
    match = _ISO_DURATION_RE.match((value or "").strip())
    if not match or not any(match.groupdict().values()):
        return None
    parts = {k: int(v) if v else 0 for k, v in match.groupdict().items()}
    return parts["days"] * 86400 + parts["hours"] * 3600 + parts["minutes"] * 60 + parts["seconds"]


def fetch_video_details(youtube_api_key, video_ids):
    """
    Duration and caption flag for each video id, as
    {video_id: {"duration_seconds": int|None, "has_captions": bool|None}}.

    videos.list accepts 50 ids per request and costs 1 quota unit per request
    (of 10,000/day), so this is effectively free compared with the transcript
    credit it can save. Ids that can't be looked up are simply absent from the
    result and the caller keeps the video — failing open, never dropping a
    video because a metadata lookup broke.
    """
    details = {}
    ids = [vid for vid in video_ids if vid]
    for start in range(0, len(ids), 50):
        batch = ids[start:start + 50]
        params = {"part": "contentDetails", "id": ",".join(batch), "key": youtube_api_key}
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.get(YOUTUBE_VIDEOS_URL, params=params, timeout=REQUEST_TIMEOUT)
            except requests.RequestException as e:
                log_warn(f"videos.list request error (attempt {attempt}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF * attempt)
                    continue
                break

            if response.status_code == 200:
                try:
                    items = (response.json() or {}).get("items") or []
                except ValueError as e:
                    log_warn(f"videos.list returned invalid JSON: {e}")
                    break
                for item in items:
                    content = (item or {}).get("contentDetails") or {}
                    video_id = (item or {}).get("id")
                    if not video_id:
                        continue
                    details[video_id] = {
                        "duration_seconds": parse_iso_duration(content.get("duration")),
                        # Note: this flag tracks *uploaded* captions and is
                        # commonly "false" for videos that only have
                        # auto-generated ones, so it is recorded for diagnostics
                        # but not used to skip videos unless explicitly enabled.
                        "has_captions": {"true": True, "false": False}.get(
                            str(content.get("caption")).lower()
                        ),
                    }
                break

            if response.status_code in TRANSIENT_STATUS and attempt < MAX_RETRIES:
                log_warn(
                    f"Transient videos.list status {response.status_code} "
                    f"(attempt {attempt}/{MAX_RETRIES}); retrying."
                )
                time.sleep(RETRY_BACKOFF * attempt)
                continue

            log_warn(f"videos.list returned {response.status_code}: {response.text[:200]}")
            break
    return details


def filter_by_duration(videos, details, min_seconds, skip_uncaptioned=False):
    """
    Drop videos shorter than `min_seconds` (Shorts and clips, which rarely have
    captions and aren't worth a transcript credit) and, when explicitly enabled,
    those the API reports as having no captions.

    Videos with no metadata are KEPT: a failed lookup must never silently drop
    content. Returns (kept, skipped_short, skipped_uncaptioned).
    """
    kept, short, uncaptioned = [], 0, 0
    for video in videos:
        info = details.get(video.get("video_id")) or {}
        seconds = info.get("duration_seconds")
        if seconds is not None:
            video["duration_seconds"] = seconds
            if min_seconds > 0 and seconds < min_seconds:
                short += 1
                continue
        if skip_uncaptioned and info.get("has_captions") is False:
            uncaptioned += 1
            continue
        kept.append(video)
    return kept, short, uncaptioned


def resolve_channel_handle(youtube_api_key, handle):
    """
    Resolve an @handle (e.g. "@hkcm") to its UC… channel id via the YouTube
    Data API. Returns the channel id, or None when it can't be resolved
    (logged, never raises) so the caller can skip that channel for this run.
    """
    params = {"part": "id", "forHandle": handle, "key": youtube_api_key}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(YOUTUBE_CHANNELS_URL, params=params, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            log_warn(f"Handle lookup error for {handle} (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
                continue
            log_error(f"Could not resolve {handle}: YouTube API unreachable.")
            return None

        if response.status_code == 200:
            try:
                items = (response.json() or {}).get("items") or []
            except ValueError as e:
                log_error(f"Unexpected handle-lookup response for {handle}: {e}")
                return None
            channel_id = (items[0] or {}).get("id") if items else None
            if not channel_id:
                log_error(f"No channel found for handle {handle}; check the spelling.")
                return None
            log_info(f"Resolved {handle} to channel id {channel_id}.")
            return channel_id

        if response.status_code in TRANSIENT_STATUS and attempt < MAX_RETRIES:
            log_warn(
                f"Transient handle-lookup status {response.status_code} for {handle} "
                f"(attempt {attempt}/{MAX_RETRIES}); retrying."
            )
            time.sleep(RETRY_BACKOFF * attempt)
            continue

        log_error(f"Handle lookup for {handle} returned {response.status_code}: {response.text[:200]}")
        return None


def _select_candidates(videos, channel_state, pending, limit=None):
    """
    Pick which of a channel's videos (newest first) to process this run.

    - New channel (no state): only the latest video, so adding a channel never
      floods Telegram with its back catalog.
    - Stored watermark: every video published after it.
    - Legacy v1 state (id only, no timestamp): every video newer than the
      stored id's position in the feed.
    Deferred videos (in `pending`) are re-included while still in the feed.
    Returns candidates oldest first, capped at `limit`; the newest ones beyond
    the cap wait for the next run so delivery stays chronological. A `limit` of
    0 (or less) means no cap — every due video is processed this run.
    """
    if limit is None:
        limit = MAX_VIDEOS_PER_RUN
    if not videos:
        return []

    if not channel_state:
        candidates = [videos[0]]
    else:
        watermark = _parse_timestamp(channel_state.get("last_published"))
        last_id = channel_state.get("last_video_id")
        if watermark is not None:
            candidates = []
            for video in videos:
                ts = _parse_timestamp(video.get("published_at"))
                if ts is not None and ts > watermark:
                    candidates.append(video)
        elif last_id:
            ids = [video["video_id"] for video in videos]
            if last_id in ids:
                candidates = videos[:ids.index(last_id)]
            else:
                log_warn("Last seen video no longer in the feed; resuming from the latest video only.")
                candidates = [videos[0]]
        else:
            candidates = [videos[0]]

    chosen = {video["video_id"] for video in candidates}
    for video in videos:
        if video["video_id"] in pending:
            chosen.add(video["video_id"])

    ordered = [video for video in reversed(videos) if video["video_id"] in chosen]
    if limit > 0 and len(ordered) > limit:
        log_warn(
            f"{len(ordered)} videos due for this channel; processing the oldest "
            f"{limit} this run, the rest on the next run."
        )
        ordered = ordered[:limit]
    return ordered


def _evict_orphaned_pending(pending, channel_id, feed_video_ids):
    """
    Drop this channel's retry records for videos that are no longer in its feed.

    `_select_candidates` can only re-include a pending video while it is still
    in the ~15-entry RSS window, so once a deferred video scrolls out it can
    never be retried or cleared — it just accumulates in the committed state
    file forever. Only called with a feed we actually fetched, so a failed
    fetch never evicts anything. Returns the number of records dropped.
    """
    orphaned = [
        video_id for video_id, record in pending.items()
        if isinstance(record, dict)
        and record.get("channel_id") == channel_id
        and video_id not in feed_video_ids
        # A finished summary that Telegram has not accepted yet carries
        # everything needed to deliver it and needs no feed entry; dropping
        # it here would lose paid-for work.
        and not _undelivered(record)
    ]
    for video_id in orphaned:
        pending.pop(video_id, None)
    return len(orphaned)


def _advance_channel_state(channels_state, channel_id, video_details):
    """
    Record a decided video as the channel's watermark. Deciding an older
    (previously deferred) video must not move the watermark backwards.
    """
    entry = channels_state.setdefault(channel_id, {})
    new_ts = _parse_timestamp(video_details.get("published_at"))
    old_ts = _parse_timestamp(entry.get("last_published"))
    if old_ts is not None and new_ts is not None and new_ts <= old_ts:
        return
    entry["last_video_id"] = video_details["video_id"]
    if new_ts is not None:
        entry["last_published"] = video_details["published_at"]


def _parse_latest_video(data):
    """Turn a YouTube search response into our video dict, or None if empty/malformed."""
    items = data.get("items") if isinstance(data, dict) else None
    if not items:
        log_warn("No videos found for this channel.")
        return None

    video = items[0] or {}
    video_id = (video.get("id") or {}).get("videoId")
    snippet = video.get("snippet") or {}
    if not video_id or not snippet:
        log_warn("YouTube response item missing videoId/snippet; skipping.")
        return None

    log_info(f"Found video: {snippet.get('title')} | Channel: {snippet.get('channelTitle')}")
    return {
        "video_id": video_id,
        "channel_name": snippet.get("channelTitle", "Unknown channel"),
        "video_title": snippet.get("title", "Untitled"),
        "video_url": f"https://www.youtube.com/watch?v={video_id}",
        "published_at": snippet.get("publishedAt", ""),
    }


def get_latest_video(YOUTUBE_api_key, channel_id):
    log_info(f"get_latest_video called for channel_id={channel_id}")

    params = {
        "part": "snippet",
        "channelId": channel_id,
        "order": "date",
        "type": "video",
        "maxResults": 1,
        "key": YOUTUBE_api_key,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(YOUTUBE_API_URL, params=params, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            log_warn(f"YouTube API request error (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
                continue
            log_error("YouTube API unreachable after retries.")
            return None

        log_debug(f"GET request to: {redact(response.url)}")

        if response.status_code == 200:
            log_debug("Received 200 OK from YouTube API")
            try:
                return _parse_latest_video(response.json())
            except (ValueError, KeyError) as e:
                log_error(f"Unexpected YouTube API response shape: {e}")
                return None

        if response.status_code in TRANSIENT_STATUS and attempt < MAX_RETRIES:
            log_warn(
                f"Transient YouTube API status {response.status_code} "
                f"(attempt {attempt}/{MAX_RETRIES}); retrying."
            )
            time.sleep(RETRY_BACKOFF * attempt)
            continue

        log_error(f"YouTube API returned status code {response.status_code} | {response.text[:300]}")
        return None


def _summarize_video(video_details, no_transcript_attempts=0, compact=False, want_signals=False,
                     hours_since_first=None):
    """
    Fetch and summarize one video's transcript. `compact` requests a short
    TL;DR-style summary (for digest-mode channels) instead of a full one.
    `hours_since_first` is how long this video has already been chased; a
    missing transcript is only written off once both that and the attempt count
    are past their limits. None means count-only (the on-demand path, which has
    no next run).
    `want_signals` asks for the market signals in the same LLM call (one
    request instead of two), falling back to a plain summary call if the
    combined response isn't usable.

    Returns (telegram_body, outcome, decided, signals):
      telegram_body — text to deliver, or None when nothing should be sent yet
        (silent deferral while waiting for captions to appear);
      outcome — key for the run-summary tally;
      decided — True when the video is final (advance dedup state), False when
        it must be retried on a later run;
      signals — extracted signals dict when the combined call produced them,
        else None (the caller can extract separately).
    """
    log_info(f"Fetching transcript for {video_details['video_url']} ...")
    transcript = get_transcript_from_video(video_details['video_url'])

    # Check the actual transcript TEXT, not the dict (a dict is always truthy).
    transcript_text = transcript.get('transcript', '') if isinstance(transcript, dict) else ''
    transcript_text = transcript_text.strip() if transcript_text else ''

    # Why the fetch went the way it did, surfaced so a run can report the mix
    # of failure causes instead of just a count.
    video_details['transcript_reason'] = (
        transcript.get("reason") if isinstance(transcript, dict) else None
    )

    if not transcript_text:
        video_details['transcript'] = "Transcript not found."
        video_details['summary'] = "Summary not available."
        if isinstance(transcript, dict) and transcript.get("budget_exhausted"):
            # We chose not to spend a transcript credit, so this is not the
            # video's fault: defer silently without consuming a retry attempt.
            log_info("Transcript budget spent; deferring this video to a later run.")
            return None, "budget_deferred", False, None
        attempt = no_transcript_attempts + 1
        # Both gates must be satisfied to write a video off: enough attempts AND
        # enough elapsed time. Either one alone gives up too early under some
        # polling schedule, and a lost transcript is a lost product.
        too_soon = (
            hours_since_first is not None
            and hours_since_first < NO_TRANSCRIPT_MIN_HOURS
        )
        if attempt < NO_TRANSCRIPT_MAX_ATTEMPTS or too_soon:
            waited = "" if hours_since_first is None else f", waited {hours_since_first:.1f}h"
            log_warn(
                f"No transcript yet (attempt {attempt}/{NO_TRANSCRIPT_MAX_ATTEMPTS}{waited}); "
                "captions may still be processing — deferring to the next run."
            )
            return None, "no_transcript_deferred", False, None
        log_warn("Transcript still unavailable after retries. Notifying Telegram.")
        return (
            f"⚠️ No transcript available for this video (checked {NO_TRANSCRIPT_MAX_ATTEMPTS} runs), "
            "so no summary could be generated. Manual review needed.",
            "no_transcript",
            True,
            None,
        )

    video_details['transcript'] = transcript
    log_info("Transcript fetched successfully. Summarizing...")
    raw_summary, signals = None, None
    if want_signals:
        # One call for both; None means "not usable" — fall back to the plain
        # summary call so a combined-format hiccup can never cost a summary.
        combined = summarize_with_signals(
            transcript_text, video_details['video_title'], compact=compact,
            channel_name=video_details.get('channel_name'),
        )
        if combined is not None:
            raw_summary, signals = combined
    if raw_summary is None:
        raw_summary = summarize_transcript(transcript_text, video_details['video_title'], compact=compact)

    if raw_summary == INSUFFICIENT_TRANSCRIPT_SENTINEL:
        # A transcript existed but was too garbled/incomplete for the model to
        # summarize meaningfully. That verdict is final — don't retry.
        video_details['summary'] = "Summary not available."
        log_warn("Transcript judged insufficient to summarize. Notifying Telegram.")
        return (
            "⚠️ The transcript was too garbled or incomplete to summarize. Manual review needed.",
            "insufficient",
            True,
            None,
        )

    if raw_summary == TRUNCATED_SENTINEL:
        # The model ran out of room even after escalating. Retryable — never
        # deliver the half-written text — but bounded: an identical transcript
        # produces an identical escalation ladder, so a video that overflows
        # once tends to overflow every run. Without a cap this would re-fetch
        # the transcript (a credit) and re-post the notice on every run until
        # the video aged out of the feed, and never be delivered.
        video_details['summary'] = "Summary not available."
        attempt = no_transcript_attempts + 1
        if attempt < NO_TRANSCRIPT_MAX_ATTEMPTS:
            log_warn(
                f"Summary came back truncated (attempt {attempt}/"
                f"{NO_TRANSCRIPT_MAX_ATTEMPTS}); deferring for retry."
            )
            return (
                "⏳ Summary deferred — the model's response was cut short. "
                "This video will be retried on the next run.",
                "truncated_deferred",
                False,
                None,
            )
        log_warn("Summary still truncated after retries. Notifying Telegram.")
        return (
            f"⚠️ The summary kept coming back cut short (checked "
            f"{NO_TRANSCRIPT_MAX_ATTEMPTS} runs), so no complete summary could "
            "be produced. Manual review needed.",
            "truncated",
            True,
            None,
        )

    if raw_summary == QUOTA_EXHAUSTED_SENTINEL:
        # All LLM providers are rate-limited/out of quota — retryable.
        video_details['summary'] = "Summary not available."
        log_warn("LLM quota exhausted; deferring this video for retry next run.")
        return (
            "⏳ Summary deferred — the LLM provider quota/rate limit was reached. "
            "This video will be retried on the next run.",
            "quota_deferred",
            False,
            None,
        )

    summary = clean_summary(raw_summary)
    if summary:
        video_details['summary'] = summary
        log_info("Summary generated.")
        return summary, "sent", True, signals

    # Transcript existed but the summarizer produced nothing.
    video_details['summary'] = "Summary not available."
    log_warn("Empty summary despite a transcript. Notifying Telegram.")
    return (
        "⚠️ A transcript was found, but summarization produced no output. Manual review needed.",
        "summary_failed",
        True,
        None,
    )


def _alert_delivery_stalled(token, chat_id, outcomes, transcript_reasons):
    """
    Tell Telegram when a run delivered nothing because every transcript source
    was exhausted. Returns True when an alert was sent.

    Worth its own message because this failure is invisible otherwise: when
    Supadata's credits ran out in August 2026 the pipeline went quiet for two
    days while every scheduled run still exited green, since deferring a video
    is a normal, successful outcome. A run that defers everything and delivers
    nothing is not normal.
    """
    if outcomes.get("sent") or not outcomes.get("budget_deferred"):
        return False
    if not token or not chat_id:
        log_warn("Delivery stalled but Telegram is not configured; cannot alert.")
        return False
    reasons = ", ".join(
        f"{k}={v}" for k, v in sorted(transcript_reasons.items(), key=lambda kv: -kv[1])
    ) or "unknown"
    send_telegram_text(token, chat_id, (
        f"⚠️ No summaries this run: {outcomes['budget_deferred']} video(s) deferred "
        f"because no transcript source had budget left ({reasons}). "
        "Nothing is lost — they stay queued and go out automatically once quota returns."
    ))
    return True


def _record_market_signals(channel_id, video_details, summary, signals=None):
    """
    Best-effort: append the delivered summary plus its market signals to
    SIGNALS_FILE. `signals` comes free from the combined summarize+extract
    call; when it's None (combined path unavailable or unusable) a separate
    extraction call is made. Any failure is logged and swallowed — signal
    recording must never affect delivery, outcomes, or dedup state.
    """
    try:
        if signals is None:
            signals = extract_signals(
                summary, video_details.get("video_title"), video_details.get("channel_name")
            )
        record = {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "video_id": video_details.get("video_id"),
            "channel_id": channel_id,
            "channel_name": video_details.get("channel_name"),
            "video_title": video_details.get("video_title"),
            "video_url": video_details.get("video_url"),
            "published_at": video_details.get("published_at"),
            "summary": summary,
            "signals": signals,
        }
        append_jsonl(SIGNALS_FILE, record)
    except Exception as e:
        log_error(f"Market-signal recording failed for {video_details.get('video_url')}: {e}")


# --- Delivery -----------------------------------------------------------------
#
# A video is "decided" once its outcome is final AND Telegram has accepted the
# message. Advancing the watermark before the send was confirmed lost the
# summary for good on any Telegram failure: the transcript and LLM call had been
# paid for, the send failed, and the video was never looked at again. So the
# finished text is held in the pending record (`undelivered`) until a send
# succeeds — a failed or interrupted delivery is retried on the next run from
# that text, with no transcript or LLM cost.

_ENTRY_FIELDS = ("channel_name", "video_title", "video_url", "published_at")


def _undelivered(record):
    """The held-for-delivery block of a pending record, or None."""
    block = (record or {}).get("undelivered") if isinstance(record, dict) else None
    return block if isinstance(block, dict) and block.get("body") else None


def _hold_for_delivery(pending, channel_id, video_details, attempts, body, outcome):
    """Remember a finished message so it can be delivered later without being
    regenerated. Keeps any retry bookkeeping already on the record."""
    record = pending.setdefault(video_id := video_details["video_id"],
                                {"channel_id": channel_id, "attempts": attempts})
    record["undelivered"] = {
        "video_id": video_id,
        "body": body,
        "outcome": outcome,
        **{field: video_details.get(field, "") for field in _ENTRY_FIELDS},
    }


def _finalize_video(channels_state, pending, channel_id, video_details):
    """The video is done and delivered: advance the watermark, drop its record."""
    _advance_channel_state(channels_state, channel_id, video_details)
    pending.pop(video_details["video_id"], None)


def _undelivered_candidates(pending, channel_id, feed_video_ids):
    """
    Videos of this channel that were summarized on an earlier run but never
    accepted by Telegram and have since left the feed. Rebuilt from the held
    record so they can still be delivered; ones still in the feed come through
    _select_candidates like any other pending video. Oldest first.
    """
    found = []
    for video_id, record in pending.items():
        block = _undelivered(record)
        if (block and isinstance(record, dict) and record.get("channel_id") == channel_id
                and video_id not in feed_video_ids):
            found.append({"video_id": video_id,
                          **{field: block.get(field, "") for field in _ENTRY_FIELDS}})
    return sorted(found, key=lambda v: v.get("published_at") or "")


class _Outbox:
    """
    Sends video messages, immediately or batched into digests, and reports
    what Telegram accepted so the caller can decide which videos are done.

    `send` returns True/False for an immediate send and None when the entry
    was buffered; `flush_channel` / `flush_run` send a buffer as one digest
    (or a single message when it holds one entry) and return whether the
    premium send succeeded. Teasers for the free channel are best-effort and
    only go out after the premium copy landed: they never affect the result.
    """

    def __init__(self, token, chat_id, free_chat_id=None, premium_url=None):
        self.token = token
        self.chat_id = chat_id
        self.free_chat_id = free_chat_id
        self.premium_url = premium_url
        self.premium_cta = f"🔓 Full summaries: {premium_url}" if premium_url else None
        self.channel_batch = []
        self.run_batch = []

    @staticmethod
    def _entry(video_details, body):
        return {**{field: video_details.get(field, "") for field in _ENTRY_FIELDS},
                "body": body}

    def send(self, video_details, body, teaser="", buffer=None):
        """Deliver now (buffer=None) or hold for a "channel" or "run" digest."""
        entry = self._entry(video_details, body)
        if buffer == "run":
            self.run_batch.append((entry, teaser))
            return None
        if buffer == "channel":
            self.channel_batch.append((entry, teaser))
            return None
        ok = send_telegram_message(
            self.token, self.chat_id, entry["channel_name"], entry["video_title"],
            entry["video_url"], entry["published_at"], body,
        )
        if ok and teaser and self.free_chat_id:
            send_telegram_teaser(
                self.token, self.free_chat_id, entry["channel_name"],
                entry["video_title"], entry["video_url"], teaser, self.premium_url,
            )
        return ok

    def flush_channel(self):
        batch, self.channel_batch = self.channel_batch, []
        title = f"New from {batch[0][0]['channel_name']}" if batch else None
        return self._flush(batch, title)

    def flush_run(self):
        batch, self.run_batch = self.run_batch, []
        return self._flush(batch, None)

    def discard_channel(self):
        """Drop the channel buffer (the channel failed mid-way). The held
        pending records still carry the text, so nothing is lost."""
        self.channel_batch = []

    def _flush(self, batch, title):
        if not batch:
            return True
        entries = [entry for entry, _ in batch]
        kwargs = {"title": title} if title else {}
        if len(entries) == 1:
            entry = entries[0]
            ok = send_telegram_message(
                self.token, self.chat_id, entry["channel_name"], entry["video_title"],
                entry["video_url"], entry["published_at"], entry["body"],
            )
        else:
            ok = send_telegram_digest(self.token, self.chat_id, entries, **kwargs)
        teasers = [{**entry, "body": teaser} for entry, teaser in batch if teaser]
        if ok and teasers and self.free_chat_id:
            if len(teasers) == 1:
                entry = teasers[0]
                send_telegram_teaser(
                    self.token, self.free_chat_id, entry["channel_name"],
                    entry["video_title"], entry["video_url"], entry["body"], self.premium_url,
                )
            else:
                send_telegram_digest(self.token, self.free_chat_id, teasers,
                                     footer=self.premium_cta, **kwargs)
        return ok


def main():
    log_info("Starting main script.")

    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")
    # Optional free/premium split: when TELEGRAM_FREE_CHANNEL_ID is set, real
    # summaries also produce a TL;DR teaser in that (public) channel, with an
    # optional CTA link to the premium channel. Unset = exactly the old behavior.
    TELEGRAM_FREE_CHANNEL_ID = os.getenv("TELEGRAM_FREE_CHANNEL_ID")
    PREMIUM_INVITE_URL = os.getenv("PREMIUM_INVITE_URL")

    # Fail fast if any required credential is missing, rather than discovering it
    # mid-run when every YouTube lookup or Telegram send fails.
    required = {
        "YOUTUBE_API_KEY": YOUTUBE_API_KEY,
        "TELEGRAM_TOKEN": TELEGRAM_TOKEN,
        "TELEGRAM_CHANNEL_ID": TELEGRAM_CHANNEL_ID,
    }
    missing = [name for name, value in required.items() if not value]

    if missing:
        log_error(
            "Missing required environment variables: "
            f"{', '.join(missing)}. Set them as repository secrets (CI) or in .env (local)."
        )
    else:
        log_info("All required environment variables loaded.")
        if not os.getenv("SUPADATA_API_KEY"):
            log_warn(
                "SUPADATA_API_KEY not set; transcript fetching may fail on CI "
                "where YouTube blocks the runner IP."
            )
        else:
            allowed, used, remaining = budget_status()
            log_info(
                f"Transcript budget: {used}/{allowed} used today, "
                f"{remaining} credit(s) left this month."
            )

        # Read channel entries (id + per-channel options) from the file
        channels = read_channels("channel_ids.txt")
        if not channels:
            log_warn("No channel IDs found. Check your channel_ids.txt file.")
        else:
            log_info(f"Beginning process to fetch video details for each channel.")
            run_started = time.monotonic()
            deadline_hit = False
            state = load_state(STATE_FILE)
            channels_state = state["channels"]
            pending = state["pending"]

            # Digest mode: buffer one entry per video and send a single combined
            # message at the end of the run instead of one message per video.
            digest_mode = env_flag("DAILY_DIGEST")
            # Market-signal recording (data/signals.jsonl) — on by default,
            # disable with MARKET_SIGNALS=false.
            market_signals = env_flag("MARKET_SIGNALS", default=True)
            outbox = _Outbox(TELEGRAM_TOKEN, TELEGRAM_CHANNEL_ID,
                             TELEGRAM_FREE_CHANNEL_ID, PREMIUM_INVITE_URL)
            # Videos whose final message sits in a digest buffer, as
            # (channel_id, video_details): they are finalized only once the
            # digest they belong to has been accepted by Telegram.
            awaiting_run_digest = []

            def settle(awaiting, delivered):
                """Finalize buffered videos after their digest went out, or
                count them as failed deliveries (their text stays held in
                `pending` for the next run)."""
                for cid, details in awaiting:
                    if delivered:
                        _finalize_video(channels_state, pending, cid, details)
                    else:
                        outcomes["delivery_failed"] += 1
                if awaiting:
                    if not delivered:
                        log_warn(
                            f"Digest with {len(awaiting)} video(s) was not accepted by "
                            "Telegram; they stay queued and go out on the next run."
                        )
                    save_state(STATE_FILE, state)
                awaiting.clear()

            # Per-channel outcome tally for the end-of-run summary report.
            outcomes = {
                "sent": 0, "redelivered": 0, "unchanged": 0, "no_transcript": 0,
                "no_transcript_deferred": 0, "insufficient": 0, "summary_failed": 0,
                "quota_deferred": 0, "budget_deferred": 0, "retry_backoff": 0,
                "truncated_deferred": 0, "truncated": 0, "delivery_failed": 0,
                "deadline_deferred": 0, "no_video": 0, "error": 0,
            }

            # Counts reported in the run summary: retry records dropped because
            # their video left the feed, and videos skipped by a title filter.
            evicted_pending = 0
            filtered_out = 0
            skipped_short = 0
            skipped_uncaptioned = 0
            # Opt-in: the API's caption flag tracks uploaded captions and is
            # commonly false for auto-captioned videos, so skipping on it is
            # off until a run's diagnostics show it is safe here.
            skip_uncaptioned = env_flag("SKIP_UNCAPTIONED")
            # Why transcript fetches failed this run, for the run summary.
            transcript_reasons = {}

            # Handles (@name) are resolved to channel ids once per run; the
            # dedup state is always keyed by the resolved id, so switching a
            # line between a handle and its id doesn't re-send old videos.
            resolved_handles = {}

            for channel in channels:
                channel_id = channel["channel_id"]
                # Isolate each channel: one malformed response or unexpected error
                # must not abort the whole run and skip every remaining channel.
                try:
                    if channel_id.startswith("@"):
                        handle = channel_id
                        channel_id = resolved_handles.get(handle)
                        if channel_id is None:
                            channel_id = resolve_channel_handle(YOUTUBE_API_KEY, handle)
                            if not channel_id:
                                outcomes["error"] += 1
                                continue
                            resolved_handles[handle] = channel_id
                    log_info(f"Processing channel ID: {channel_id}")
                    videos = get_recent_videos(YOUTUBE_API_KEY, channel_id)
                    if not videos:
                        log_warn(f"No videos found for channel ID: {channel_id}.")
                        outcomes["no_video"] += 1
                        continue

                    # Title filter (only=…): drop non-matching videos before any
                    # transcript is fetched, so they cost nothing. Applied to
                    # the feed itself, so the watermark simply moves past them
                    # and they are never reconsidered.
                    if channel.get("only"):
                        kept = [v for v in videos if title_matches(v["video_title"], channel.get("only"))]
                        if len(kept) != len(videos):
                            filtered_out += len(videos) - len(kept)
                            log_info(
                                f"Title filter kept {len(kept)}/{len(videos)} videos for "
                                f"channel {channel_id} (only={','.join(channel.get('only') or [])})."
                            )
                        videos = kept
                        if not videos:
                            log_info(f"No videos matching the title filter for {channel_id}. Skipping.")
                            outcomes["unchanged"] += 1
                            continue

                    # Duration gate: one batched videos.list call (1 quota unit
                    # per 50 ids) screens out Shorts before any transcript
                    # credit is spent on them. Videos whose metadata can't be
                    # read are kept, so a lookup failure never drops content.
                    if MIN_VIDEO_SECONDS > 0 or skip_uncaptioned:
                        details = fetch_video_details(
                            YOUTUBE_API_KEY, [v["video_id"] for v in videos]
                        )
                        videos, n_short, n_uncaptioned = filter_by_duration(
                            videos, details, MIN_VIDEO_SECONDS, skip_uncaptioned
                        )
                        skipped_short += n_short
                        skipped_uncaptioned += n_uncaptioned
                        if n_short or n_uncaptioned:
                            log_info(
                                f"Skipped {n_short} short and {n_uncaptioned} caption-less "
                                f"video(s) for channel {channel_id} before fetching transcripts."
                            )
                        if not videos:
                            log_info(f"No videos left after the duration gate for {channel_id}.")
                            outcomes["unchanged"] += 1
                            continue

                    # The feed fetch succeeded, so anything still pending for
                    # this channel that isn't in the feed can never be retried.
                    # Filtered-out videos count as gone, clearing any retry
                    # records the filter now excludes.
                    evicted_pending += _evict_orphaned_pending(
                        pending, channel_id, {v["video_id"] for v in videos}
                    )

                    feed_ids = {v["video_id"] for v in videos}
                    candidates = _select_candidates(
                        videos, channels_state.get(channel_id), pending,
                        limit=channel["max_per_run"],
                    )
                    # Finished-but-undelivered videos that have since left the
                    # feed are still owed to the reader; they cost nothing.
                    candidates = _undelivered_candidates(pending, channel_id, feed_ids) + candidates
                    if not candidates:
                        log_info(f"No new videos for channel {channel_id}. Skipping.")
                        outcomes["unchanged"] += 1
                        continue

                    # Digest-flagged channels (prolific posters) get one bundled
                    # message per run with compact TL;DR entries, instead of one
                    # full-summary message per video.
                    buffer = "run" if digest_mode else ("channel" if channel["digest"] else None)
                    awaiting_channel_digest = []

                    for video_details in candidates:
                        video_id = video_details["video_id"]
                        record = pending.get(video_id) or {}
                        attempts = record.get("attempts", 0)
                        held = _undelivered(record)
                        redelivery = held is not None

                        if redelivery:
                            # Summarized on an earlier run; Telegram never
                            # accepted it. Deliver the held text as-is: no
                            # transcript credit, no LLM request, no new signal
                            # row (that was recorded when it was generated).
                            log_info(f"Re-delivering held summary for {video_id}.")
                            telegram_body, outcome, decided, signals = (
                                held["body"], held.get("outcome") or "sent", True, None
                            )
                            outcomes["redelivered"] += 1
                        else:
                            if (not deadline_hit and RUN_DEADLINE_MINUTES > 0
                                    and (time.monotonic() - run_started) / 60.0 >= RUN_DEADLINE_MINUTES):
                                deadline_hit = True
                                log_warn(
                                    f"Run deadline of {RUN_DEADLINE_MINUTES:g} min reached; "
                                    "remaining videos defer to the next run."
                                )
                            if deadline_hit:
                                outcomes["deadline_deferred"] += 1
                                continue
                            wait = _retry_wait_remaining(record)
                            if wait > 0:
                                log_info(
                                    f"Retried {video_id} recently; waiting {wait:.1f}h more "
                                    "before spending another transcript credit on it."
                                )
                                outcomes["retry_backoff"] += 1
                                continue
                            log_info(
                                f"Processing video: {video_details['video_title']} "
                                f"(published: {video_details['published_at']})"
                            )

                            telegram_body, outcome, decided, signals = _summarize_video(
                                video_details, attempts, compact=channel["digest"],
                                want_signals=market_signals,
                                hours_since_first=_hours_since(record.get("first_attempt")),
                            )
                            outcomes[outcome] += 1
                            reason = video_details.get("transcript_reason")
                            if reason and reason not in TRANSCRIPT_SUCCESS_REASONS:
                                transcript_reasons[reason] = transcript_reasons.get(reason, 0) + 1

                            # One notice per run per deferral kind is enough;
                            # later ones are logged only. Without this a run
                            # that defers ten videos posts ten identical messages.
                            if outcome in ("quota_deferred", "truncated_deferred") and outcomes[outcome] > 1:
                                telegram_body = None

                        # True/False for an immediate send, None when buffered
                        # into a digest, True when there is nothing to send.
                        delivered = True
                        if telegram_body is not None:
                            # Free/premium split: only real summaries get a
                            # teaser — warning/deferral notices stay premium-only.
                            # A failed teaser send is logged inside the sender
                            # and never affects the video's outcome/watermark.
                            teaser = ""
                            if TELEGRAM_FREE_CHANNEL_ID and outcome == "sent":
                                teaser = build_teaser(telegram_body)
                            delivered = outbox.send(video_details, telegram_body, teaser, buffer)

                        if market_signals and outcome == "sent" and not redelivery:
                            _record_market_signals(channel_id, video_details, telegram_body, signals)

                        if decided and delivered is True:
                            # Final outcome, accepted by Telegram: advance the
                            # watermark and drop any pending record.
                            _finalize_video(channels_state, pending, channel_id, video_details)
                        elif decided:
                            # Final outcome, but not delivered yet: keep the
                            # finished text so it is never regenerated. Either it
                            # is waiting in a digest buffer (finalized once that
                            # digest goes out) or the send failed (retried next
                            # run from the held text).
                            _hold_for_delivery(pending, channel_id, video_details,
                                               attempts, telegram_body, outcome)
                            if delivered is None:
                                awaiting = awaiting_run_digest if buffer == "run" else awaiting_channel_digest
                                awaiting.append((channel_id, video_details))
                            else:
                                outcomes["delivery_failed"] += 1
                                log_warn(
                                    f"Telegram did not accept the message for {video_id}; "
                                    "keeping it queued for the next run."
                                )
                        else:
                            # Retryable outcome: leave the watermark alone and
                            # remember the video so the next run picks it up.
                            entry = pending.setdefault(video_id, {"channel_id": channel_id, "attempts": 0})
                            if outcome in ("no_transcript_deferred", "truncated_deferred"):
                                # Stamped only for deferrals that cost a credit,
                                # so a budget-deferred video retries as soon as
                                # credits return rather than serving out a wait.
                                # Truncation costs one too — the transcript was
                                # fetched and summarized — so it backs off and
                                # counts toward the give-up cap the same way.
                                now = datetime.now(timezone.utc).isoformat()
                                entry["attempts"] = attempts + 1
                                entry.setdefault("first_attempt", now)
                                entry["last_attempt"] = now

                        # Persist immediately, so a later crash doesn't cause
                        # already-sent videos to be re-sent.
                        save_state(STATE_FILE, state)

                    settle(awaiting_channel_digest, outbox.flush_channel())
                except Exception as e:
                    # Don't let one channel's failure sink the rest of the batch.
                    # Anything buffered for this channel's digest stays held in
                    # `pending` and is delivered on the next run.
                    log_error(f"Unexpected error processing channel {channel_id}: {e}", exc_info=True)
                    outbox.discard_channel()
                    outcomes["error"] += 1
                    continue

            settle(awaiting_run_digest, outbox.flush_run())

            # End-of-run report: one line summarizing what happened this run.
            summary_line = ", ".join(f"{k}={v}" for k, v in outcomes.items() if v)
            if evicted_pending:
                summary_line += f", pending_evicted={evicted_pending}"
            if filtered_out:
                summary_line += f", title_filtered={filtered_out}"
            if skipped_short:
                summary_line += f", too_short={skipped_short}"
            if skipped_uncaptioned:
                summary_line += f", uncaptioned={skipped_uncaptioned}"
            log_info(f"Run summary: {len(channels)} channels | {summary_line or 'nothing to do'}")
            if transcript_reasons:
                # The point of this line: turn "65% of credits produce nothing"
                # into a breakdown that says which fix would actually help.
                breakdown = ", ".join(
                    f"{k}={v}" for k, v in sorted(transcript_reasons.items(), key=lambda kv: -kv[1])
                )
                log_warn(f"Transcript failures by reason: {breakdown}")

            # What is left to spend today, per model. Without this the first
            # sign of an exhausted budget is summaries quietly not arriving.
            gemini_budget = gemini_quota.report()
            if gemini_budget:
                log_info(f"Gemini daily budget: {gemini_budget}")

            _alert_delivery_stalled(
                TELEGRAM_TOKEN, TELEGRAM_CHANNEL_ID, outcomes, transcript_reasons
            )

    log_info("Main script finished.")


def _fetch_video_metadata(video_url):
    """Best-effort (title, channel_name) via YouTube's keyless oEmbed endpoint."""
    try:
        response = requests.get(
            OEMBED_URL, params={"url": video_url, "format": "json"}, timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("title") or "Unknown title", data.get("author_name") or "Unknown channel"
        log_warn(f"oEmbed lookup returned {response.status_code}; using placeholder metadata.")
    except (requests.RequestException, ValueError) as e:
        log_warn(f"oEmbed lookup failed: {e}; using placeholder metadata.")
    return "Unknown title", "Unknown channel"


def summarize_on_demand(video_url):
    """
    Summarize a single video URL and send it to Telegram, bypassing the channel
    scan and dedup state entirely (used by manual workflow_dispatch runs).
    Returns True when the Telegram send succeeded.
    """
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")
    if not TELEGRAM_TOKEN or not TELEGRAM_CHANNEL_ID:
        log_error("TELEGRAM_TOKEN and TELEGRAM_CHANNEL_ID must be set for on-demand summarization.")
        return False

    log_info(f"On-demand summarization requested for {video_url}")
    video_title, channel_name = _fetch_video_metadata(video_url)
    video_details = {
        "video_id": "",
        "channel_name": channel_name,
        "video_title": video_title,
        "video_url": video_url,
        "published_at": "n/a",
    }

    # Prime the attempt counter so a missing transcript reports immediately
    # instead of deferring — there is no "next run" for an on-demand request.
    telegram_body, outcome, _, _ = _summarize_video(
        video_details, no_transcript_attempts=NO_TRANSCRIPT_MAX_ATTEMPTS - 1
    )
    # There is no next run for an on-demand request, so a "deferred" outcome is
    # simply a failure: say so rather than promising a retry that never comes,
    # and report it as a failure so the manual run doesn't look green.
    deferred = outcome in ("quota_deferred", "truncated_deferred", "budget_deferred")
    if deferred:
        telegram_body = (
            "⚠️ No summary could be produced for this video right now "
            f"({outcome.replace('_', ' ')}). Try again later."
        )
    sent = send_telegram_message(
        TELEGRAM_TOKEN, TELEGRAM_CHANNEL_ID, channel_name, video_title,
        video_url, video_details["published_at"], telegram_body,
    )
    log_info(f"On-demand summary finished (outcome: {outcome}).")
    return sent and not deferred


def cli():
    parser = argparse.ArgumentParser(description="YouTube auto-summarizer")
    parser.add_argument(
        "--video-url",
        help="Summarize this single video and send it to Telegram "
             "(skips the channel scan and dedup state).",
    )
    args = parser.parse_args()
    if args.video_url:
        # Exit non-zero on failure so a broken on-demand run trips the
        # workflow's failure alert instead of silently looking green.
        sys.exit(0 if summarize_on_demand(args.video_url) else 1)
    else:
        main()


if __name__ == "__main__":
    cli()
