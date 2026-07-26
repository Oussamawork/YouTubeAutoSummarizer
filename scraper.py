import argparse
import requests
import sys
import time
from dotenv import load_dotenv
from defusedxml import ElementTree as SafeET
from transcript import get_transcript_from_video
from helpers import read_channels, save_to_json, clean_summary, load_state, save_state, env_int, append_jsonl
from signals import extract_signals, summarize_with_signals
from summarizer import (
    summarize_transcript,
    INSUFFICIENT_TRANSCRIPT_SENTINEL,
    QUOTA_EXHAUSTED_SENTINEL,
)
from log import log_info, log_error, log_warn, log_debug
from sendToTelegram import send_telegram_message, send_telegram_digest, send_telegram_teaser, build_teaser
import os
from datetime import datetime, timezone

# Load environment variables from .env file
load_dotenv('.env')

# YouTube API request tuning
YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3/search"
# Channels endpoint, used to resolve an @handle to its channel id (1 quota unit
# per call, vs 100 for a search) so channel_ids.txt can list handles directly.
YOUTUBE_CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"
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

# Cap on videos processed per channel per run, so a backlog (or a channel that
# uploads a lot) can't flood Telegram in one run. The rest wait for the next run.
MAX_VIDEOS_PER_RUN = env_int("MAX_VIDEOS_PER_RUN", 3)
# Captions (especially auto-generated ones) often appear hours after upload, so
# a video with no transcript is retried this many runs before giving up.
NO_TRANSCRIPT_MAX_ATTEMPTS = env_int("NO_TRANSCRIPT_MAX_ATTEMPTS", 3)


def _env_flag(name, default=False):
    """Boolean env flag. Unset or empty (an unconfigured GitHub Actions repo
    variable arrives as "") falls back to `default`."""
    value = (os.getenv(name) or "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


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
    the cap wait for the next run so delivery stays chronological.
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
    if len(ordered) > limit:
        log_warn(
            f"{len(ordered)} videos due for this channel; processing the oldest "
            f"{limit} this run, the rest on the next run."
        )
        ordered = ordered[:limit]
    return ordered


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

        log_debug(f"GET request to: {response.url}")

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

        log_error(f"YouTube API returned status code {response.status_code} | {response.text}")
        return None


def _summarize_video(video_details, no_transcript_attempts=0, compact=False, want_signals=False):
    """
    Fetch and summarize one video's transcript. `compact` requests a short
    TL;DR-style summary (for digest-mode channels) instead of a full one.
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

    if not transcript_text:
        video_details['transcript'] = "Transcript not found."
        video_details['summary'] = "Summary not available."
        attempt = no_transcript_attempts + 1
        if attempt < NO_TRANSCRIPT_MAX_ATTEMPTS:
            log_warn(
                f"No transcript yet (attempt {attempt}/{NO_TRANSCRIPT_MAX_ATTEMPTS}); "
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

        # Read channel entries (id + per-channel options) from the file
        channels = read_channels("channel_ids.txt")
        if not channels:
            log_warn("No channel IDs found. Check your channel_ids.txt file.")
        else:
            log_info(f"Beginning process to fetch video details for each channel.")
            results = []
            state = load_state(STATE_FILE)
            channels_state = state["channels"]
            pending = state["pending"]

            # Digest mode: buffer one entry per video and send a single combined
            # message at the end of the run instead of one message per video.
            digest_mode = _env_flag("DAILY_DIGEST")
            digest_entries = []
            # Market-signal recording (data/signals.jsonl) — on by default,
            # disable with MARKET_SIGNALS=false.
            market_signals = _env_flag("MARKET_SIGNALS", default=True)
            # Teaser copies of the digest entries for the free channel (only
            # populated when the free/premium split is enabled).
            free_digest_entries = []
            premium_cta = f"🔓 Full summaries: {PREMIUM_INVITE_URL}" if PREMIUM_INVITE_URL else None

            # Per-channel outcome tally for the end-of-run summary report.
            outcomes = {
                "sent": 0, "unchanged": 0, "no_transcript": 0, "no_transcript_deferred": 0,
                "insufficient": 0, "summary_failed": 0, "quota_deferred": 0,
                "no_video": 0, "error": 0,
            }

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

                    candidates = _select_candidates(
                        videos, channels_state.get(channel_id), pending,
                        limit=channel["max_per_run"],
                    )
                    if not candidates:
                        log_info(f"No new videos for channel {channel_id}. Skipping.")
                        outcomes["unchanged"] += 1
                        continue

                    # Digest-flagged channels (prolific posters) get one bundled
                    # message per run with compact TL;DR entries, instead of one
                    # full-summary message per video.
                    channel_entries = []
                    free_channel_entries = []

                    for video_details in candidates:
                        video_id = video_details["video_id"]
                        attempts = pending.get(video_id, {}).get("attempts", 0)
                        log_info(
                            f"Processing video: {video_details['video_title']} "
                            f"(published: {video_details['published_at']})"
                        )

                        telegram_body, outcome, decided, signals = _summarize_video(
                            video_details, attempts, compact=channel["digest"],
                            want_signals=market_signals,
                        )
                        outcomes[outcome] += 1

                        # One quota notice per run is enough; later deferrals are logged only.
                        if outcome == "quota_deferred" and outcomes["quota_deferred"] > 1:
                            telegram_body = None

                        if telegram_body is not None:
                            entry = {
                                "channel_name": video_details['channel_name'],
                                "video_title": video_details['video_title'],
                                "video_url": video_details['video_url'],
                                "published_at": video_details['published_at'],
                                "body": telegram_body,
                            }
                            # Free/premium split: only real summaries get a
                            # teaser — warning/deferral notices stay premium-only.
                            # A failed teaser send is logged inside the sender
                            # and never affects the video's outcome/watermark.
                            teaser = ""
                            if TELEGRAM_FREE_CHANNEL_ID and outcome == "sent":
                                teaser = build_teaser(telegram_body)
                            if digest_mode:
                                digest_entries.append(entry)
                                if teaser:
                                    free_digest_entries.append({**entry, "body": teaser})
                            elif channel["digest"]:
                                channel_entries.append(entry)
                                if teaser:
                                    free_channel_entries.append({**entry, "body": teaser})
                            else:
                                send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_CHANNEL_ID, video_details['channel_name'], video_details['video_title'], video_details['video_url'], video_details['published_at'], telegram_body)
                                if teaser:
                                    send_telegram_teaser(
                                        TELEGRAM_TOKEN, TELEGRAM_FREE_CHANNEL_ID,
                                        video_details['channel_name'], video_details['video_title'],
                                        video_details['video_url'], teaser, PREMIUM_INVITE_URL,
                                    )

                        if market_signals and outcome == "sent":
                            _record_market_signals(channel_id, video_details, telegram_body, signals)

                        if decided:
                            # Final outcome: advance the watermark and drop any
                            # pending-retry record for this video.
                            _advance_channel_state(channels_state, channel_id, video_details)
                            pending.pop(video_id, None)
                            results.append(video_details)
                        else:
                            # Retryable outcome: leave the watermark alone and
                            # remember the video so the next run picks it up.
                            entry = pending.setdefault(video_id, {"channel_id": channel_id, "attempts": 0})
                            if outcome == "no_transcript_deferred":
                                entry["attempts"] = attempts + 1

                        # Persist immediately, so a later crash doesn't cause
                        # already-sent videos to be re-sent.
                        save_state(STATE_FILE, state)

                    if channel_entries:
                        if len(channel_entries) == 1:
                            entry = channel_entries[0]
                            send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_CHANNEL_ID, entry['channel_name'], entry['video_title'], entry['video_url'], entry['published_at'], entry['body'])
                        else:
                            send_telegram_digest(
                                TELEGRAM_TOKEN, TELEGRAM_CHANNEL_ID, channel_entries,
                                title=f"New from {channel_entries[0]['channel_name']}",
                            )
                    if free_channel_entries:
                        if len(free_channel_entries) == 1:
                            entry = free_channel_entries[0]
                            send_telegram_teaser(
                                TELEGRAM_TOKEN, TELEGRAM_FREE_CHANNEL_ID,
                                entry['channel_name'], entry['video_title'],
                                entry['video_url'], entry['body'], PREMIUM_INVITE_URL,
                            )
                        else:
                            send_telegram_digest(
                                TELEGRAM_TOKEN, TELEGRAM_FREE_CHANNEL_ID, free_channel_entries,
                                title=f"New from {free_channel_entries[0]['channel_name']}",
                                footer=premium_cta,
                            )
                except Exception as e:
                    # Don't let one channel's failure sink the rest of the batch.
                    log_error(f"Unexpected error processing channel {channel_id}: {e}")
                    outcomes["error"] += 1
                    continue

            if digest_entries:
                if len(digest_entries) == 1:
                    entry = digest_entries[0]
                    send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_CHANNEL_ID, entry['channel_name'], entry['video_title'], entry['video_url'], entry['published_at'], entry['body'])
                else:
                    send_telegram_digest(TELEGRAM_TOKEN, TELEGRAM_CHANNEL_ID, digest_entries)
            if free_digest_entries:
                if len(free_digest_entries) == 1:
                    entry = free_digest_entries[0]
                    send_telegram_teaser(
                        TELEGRAM_TOKEN, TELEGRAM_FREE_CHANNEL_ID,
                        entry['channel_name'], entry['video_title'],
                        entry['video_url'], entry['body'], PREMIUM_INVITE_URL,
                    )
                else:
                    send_telegram_digest(
                        TELEGRAM_TOKEN, TELEGRAM_FREE_CHANNEL_ID, free_digest_entries,
                        footer=premium_cta,
                    )

            # End-of-run report: one line summarizing what happened this run.
            summary_line = ", ".join(f"{k}={v}" for k, v in outcomes.items() if v)
            log_info(f"Run summary: {len(channels)} channels | {summary_line or 'nothing to do'}")

            # Save the results to a JSON file
            if results:
                filename = 'video_details_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.json'
                log_info(f"Saving results to {filename} ...")
                save_to_json(results, filename)
                log_info(f"Process completed successfully.")
            else:
                log_warn("No results to save.")

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
    sent = send_telegram_message(
        TELEGRAM_TOKEN, TELEGRAM_CHANNEL_ID, channel_name, video_title,
        video_url, video_details["published_at"], telegram_body,
    )
    log_info(f"On-demand summary finished (outcome: {outcome}).")
    return sent


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
