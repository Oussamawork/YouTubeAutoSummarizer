import argparse
import requests
import sys
import time
from dotenv import load_dotenv
from defusedxml import ElementTree as SafeET
from transcript import get_transcript_from_video
from helpers import read_channels, save_to_json, clean_summary, load_state, save_state
from summarizer import (
    summarize_transcript,
    INSUFFICIENT_TRANSCRIPT_SENTINEL,
    QUOTA_EXHAUSTED_SENTINEL,
)
from log import log_info, log_error, log_warn, log_debug
from sendToTelegram import send_telegram_message, send_telegram_digest
import os
from datetime import datetime, timezone

# Load environment variables from .env file
load_dotenv('.env')

# YouTube API request tuning
YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3/search"
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

# Cap on videos processed per channel per run, so a backlog (or a channel that
# uploads a lot) can't flood Telegram in one run. The rest wait for the next run.
MAX_VIDEOS_PER_RUN = int(os.getenv("MAX_VIDEOS_PER_RUN", "3"))
# Captions (especially auto-generated ones) often appear hours after upload, so
# a video with no transcript is retried this many runs before giving up.
NO_TRANSCRIPT_MAX_ATTEMPTS = int(os.getenv("NO_TRANSCRIPT_MAX_ATTEMPTS", "3"))


def _env_flag(name):
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


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


def _summarize_video(video_details, no_transcript_attempts=0, compact=False):
    """
    Fetch and summarize one video's transcript. `compact` requests a short
    TL;DR-style summary (for digest-mode channels) instead of a full one.

    Returns (telegram_body, outcome, decided):
      telegram_body — text to deliver, or None when nothing should be sent yet
        (silent deferral while waiting for captions to appear);
      outcome — key for the run-summary tally;
      decided — True when the video is final (advance dedup state), False when
        it must be retried on a later run.
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
            return None, "no_transcript_deferred", False
        log_warn("Transcript still unavailable after retries. Notifying Telegram.")
        return (
            f"⚠️ No transcript available for this video (checked {NO_TRANSCRIPT_MAX_ATTEMPTS} runs), "
            "so no summary could be generated. Manual review needed.",
            "no_transcript",
            True,
        )

    video_details['transcript'] = transcript
    log_info("Transcript fetched successfully. Summarizing...")
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
        )

    summary = clean_summary(raw_summary)
    if summary:
        video_details['summary'] = summary
        log_info("Summary generated.")
        return summary, "sent", True

    # Transcript existed but the summarizer produced nothing.
    video_details['summary'] = "Summary not available."
    log_warn("Empty summary despite a transcript. Notifying Telegram.")
    return (
        "⚠️ A transcript was found, but summarization produced no output. Manual review needed.",
        "summary_failed",
        True,
    )


def main():
    log_info("Starting main script.")

    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")

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

            # Per-channel outcome tally for the end-of-run summary report.
            outcomes = {
                "sent": 0, "unchanged": 0, "no_transcript": 0, "no_transcript_deferred": 0,
                "insufficient": 0, "summary_failed": 0, "quota_deferred": 0,
                "no_video": 0, "error": 0,
            }

            for channel in channels:
                channel_id = channel["channel_id"]
                # Isolate each channel: one malformed response or unexpected error
                # must not abort the whole run and skip every remaining channel.
                try:
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

                    for video_details in candidates:
                        video_id = video_details["video_id"]
                        attempts = pending.get(video_id, {}).get("attempts", 0)
                        log_info(
                            f"Processing video: {video_details['video_title']} "
                            f"(published: {video_details['published_at']})"
                        )

                        telegram_body, outcome, decided = _summarize_video(
                            video_details, attempts, compact=channel["digest"]
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
                            if digest_mode:
                                digest_entries.append(entry)
                            elif channel["digest"]:
                                channel_entries.append(entry)
                            else:
                                send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_CHANNEL_ID, video_details['channel_name'], video_details['video_title'], video_details['video_url'], video_details['published_at'], telegram_body)

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
    telegram_body, outcome, _ = _summarize_video(
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
