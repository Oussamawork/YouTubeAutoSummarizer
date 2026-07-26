import calendar
import json
import os
import re
import time
from datetime import datetime, timezone
from urllib.parse import urlparse, parse_qs

import requests
from helpers import env_int
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    CouldNotRetrieveTranscript,
)
from log import log_info, log_warn, log_error

# Supadata is the primary transcript source: a hosted API (free tier) that
# fetches captions server-side, so it works even from CI runners whose
# datacenter IPs YouTube blocks. It is only used when SUPADATA_API_KEY is set;
# otherwise (e.g. local dev) the code falls back to youtube-transcript-api.
SUPADATA_API_KEY = os.getenv("SUPADATA_API_KEY")
SUPADATA_URL = "https://api.supadata.ai/v1/transcript"
SUPADATA_TIMEOUT = 30
SUPADATA_POLL_ATTEMPTS = 6
SUPADATA_POLL_DELAY = 5  # seconds between polls for async (202) jobs
SUPADATA_MAX_RETRIES = 3
SUPADATA_RETRY_BACKOFF = 2  # base seconds, multiplied by the attempt number
SUPADATA_TRANSIENT_STATUS = {429, 500, 502, 503, 504}
# Supadata answers "you're out of credits" with these; the request itself did
# not deliver a transcript, so we rotate to the next key rather than retrying.
SUPADATA_CREDIT_STATUS = {402, 403}

# --- Free-tier budget -------------------------------------------------------
#
# Supadata's free tier is a small monthly credit pool per key, and a spent pool
# means no transcripts at all (the youtube-transcript-api fallback is blocked
# from CI IPs). So usage is metered here: each key contributes
# SUPADATA_CREDITS_PER_KEY to a monthly budget, and the budget is spread evenly
# across the days left in the month. Hitting the daily allowance defers videos
# to a later run instead of burning the month's credits in the first week.
SUPADATA_CREDITS_PER_KEY = env_int("SUPADATA_CREDITS_PER_KEY", 100)
SUPADATA_USAGE_FILE = os.getenv("SUPADATA_USAGE_FILE") or "data/supadata_usage.json"


def _supadata_keys():
    """
    Configured Supadata keys, in order. Reads SUPADATA_API_KEYS (comma
    separated) plus the numbered SUPADATA_API_KEY / _2 / _3 forms, de-duped
    and read at call time so tests and reloads see the current environment.
    """
    raw = [os.getenv("SUPADATA_API_KEYS") or ""]
    raw += [os.getenv(name) or "" for name in
            ("SUPADATA_API_KEY", "SUPADATA_API_KEY_2", "SUPADATA_API_KEY_3")]
    keys, seen = [], set()
    for value in raw:
        for key in value.split(","):
            key = key.strip()
            if key and key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


def monthly_budget():
    """Total transcript fetches allowed this month across all configured keys."""
    explicit = env_int("SUPADATA_MONTHLY_BUDGET", 0)
    if explicit > 0:
        return explicit
    return len(_supadata_keys()) * SUPADATA_CREDITS_PER_KEY


def _load_usage():
    """Usage counters for the current month/day; resets when either rolls over."""
    today = datetime.now(timezone.utc).date()
    month, day = today.strftime("%Y-%m"), today.isoformat()
    try:
        with open(SUPADATA_USAGE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("usage file is not an object")
    except (OSError, ValueError):
        data = {}
    if data.get("month") != month:
        data = {"month": month, "count": 0}
    if data.get("day") != day:
        data["day"], data["day_count"] = day, 0
    data.setdefault("count", 0)
    data.setdefault("day_count", 0)
    return data


def _save_usage(usage):
    try:
        directory = os.path.dirname(SUPADATA_USAGE_FILE)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(SUPADATA_USAGE_FILE, "w", encoding="utf-8") as f:
            json.dump(usage, f, indent=2)
    except OSError as e:
        # Losing a counter update is better than losing the run; the worst case
        # is over-counting next run, which errs toward saving credits.
        log_warn(f"Could not persist Supadata usage: {e}")


def _record_call(usage):
    """Count one Supadata request against the month's and today's budget."""
    usage["count"] = usage.get("count", 0) + 1
    usage["day_count"] = usage.get("day_count", 0) + 1
    _save_usage(usage)


def daily_allowance(usage=None, today=None):
    """
    How many fetches today may use: the remaining monthly budget spread over
    the days left in the month (today included), so credits last all month.
    Returns 0 when the monthly budget is spent.
    """
    usage = usage if usage is not None else _load_usage()
    today = today or datetime.now(timezone.utc).date()
    remaining = monthly_budget() - usage.get("count", 0)
    if remaining <= 0:
        return 0
    days_left = calendar.monthrange(today.year, today.month)[1] - today.day + 1
    return max(1, remaining // max(1, days_left))


def budget_status():
    """(allowed_today, used_today, remaining_this_month) — for logging."""
    usage = _load_usage()
    return (
        daily_allowance(usage),
        usage.get("day_count", 0),
        max(0, monthly_budget() - usage.get("count", 0)),
    )


def _extract_video_id(video_url_or_id):
    """
    Accepts a full YouTube URL or a bare video ID and returns the 11-char video ID.
    Returns "" if no ID can be parsed.
    """
    if not video_url_or_id:
        return ""

    candidate = video_url_or_id.strip()

    # Already a bare ID (YouTube IDs are 11 chars: letters, digits, - and _)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
        return candidate

    parsed = urlparse(candidate)

    # youtu.be/<id>
    if parsed.netloc.endswith("youtu.be"):
        vid = parsed.path.lstrip("/").split("/")[0]
        return vid if re.fullmatch(r"[A-Za-z0-9_-]{11}", vid) else ""

    # youtube.com/watch?v=<id>
    if "youtube.com" in parsed.netloc:
        query = parse_qs(parsed.query)
        if "v" in query and query["v"]:
            vid = query["v"][0]
            return vid if re.fullmatch(r"[A-Za-z0-9_-]{11}", vid) else ""
        # /embed/<id> or /shorts/<id>
        m = re.search(r"/(?:embed|shorts|v)/([A-Za-z0-9_-]{11})", parsed.path)
        if m:
            return m.group(1)

    log_warn(f"Could not extract a video ID from: {video_url_or_id}")
    return ""


def _fetch_youtube_transcript_api(vid):
    """Fallback source: scrape captions directly. Returns "" on any failure."""
    try:
        log_info(f"Fetching transcript via youtube-transcript-api for video ID: {vid}")
        fetched = YouTubeTranscriptApi().fetch(vid)
        text = " ".join(snippet.text for snippet in fetched).strip()
        log_info(f"youtube-transcript-api returned {len(text)} chars")
        return text
    except TranscriptsDisabled:
        log_warn(f"Transcripts are disabled for video {vid}.")
    except NoTranscriptFound:
        log_warn(f"No transcript found for video {vid}.")
    except VideoUnavailable:
        log_warn(f"Video {vid} is unavailable.")
    except CouldNotRetrieveTranscript as e:
        # Base class for library-level failures, e.g. RequestBlocked / IpBlocked.
        log_warn(f"youtube-transcript-api could not retrieve {vid} (likely IP block): {e}")
    except Exception as e:
        log_error(f"Unexpected youtube-transcript-api error for {vid}: {e}")
    return ""


def _supadata_text_from_payload(data):
    """Extract transcript text from a Supadata response body. Returns "" if none."""
    if not isinstance(data, dict):
        return ""
    content = data.get("content")
    # text=true mode returns a plain string.
    if isinstance(content, str):
        return content.strip()
    # Segmented mode returns a list of {text, offset, duration, lang}.
    if isinstance(content, list):
        return " ".join(seg.get("text", "") for seg in content if isinstance(seg, dict)).strip()
    return ""


def _fetch_supadata(vid):
    """
    Primary source: Supadata hosted API (free tier). Server-side fetch, so it
    works from blocked CI IPs. Returns (text, budget_exhausted): text is "" if
    no key, on error, or if empty; budget_exhausted is True when the call was
    skipped because this month's/today's credits are spent (the caller defers
    the video instead of reporting a missing transcript).
    Uses mode=native so only existing captions are returned (no paid AI generation).
    """
    keys = _supadata_keys()
    if not keys:
        return "", False

    usage = _load_usage()
    allowance = daily_allowance(usage)
    if usage.get("day_count", 0) >= allowance:
        remaining = max(0, monthly_budget() - usage.get("count", 0))
        log_warn(
            f"Supadata budget reached for today ({usage['day_count']}/{allowance}; "
            f"{remaining} left this month) — deferring {vid} to a later run."
        )
        return "", True

    for index, key in enumerate(keys):
        text = _fetch_supadata_with_key(vid, key, usage)
        if text is not None:
            return text, False
        if index + 1 < len(keys):
            log_warn(f"Supadata key {index + 1} is out of credits; trying the next key.")
    return "", False


def _fetch_supadata_with_key(vid, api_key, usage):
    """
    One key's attempt at a transcript. Returns the text ("" when the key worked
    but there is no transcript), or None when the key is out of credits so the
    caller should rotate to the next one. Every request sent is metered.
    """
    headers = {"x-api-key": api_key}
    params = {
        "url": f"https://www.youtube.com/watch?v={vid}",
        "text": "true",
        "mode": "native",
    }
    log_info(f"Fetching transcript via Supadata for video ID: {vid}")

    for attempt in range(1, SUPADATA_MAX_RETRIES + 1):
        try:
            resp = requests.get(SUPADATA_URL, headers=headers, params=params, timeout=SUPADATA_TIMEOUT)
        except requests.RequestException as e:
            log_warn(f"Supadata request error (attempt {attempt}/{SUPADATA_MAX_RETRIES}): {e}")
            if attempt < SUPADATA_MAX_RETRIES:
                time.sleep(SUPADATA_RETRY_BACKOFF * attempt)
                continue
            log_error("Supadata unreachable after retries.")
            return ""

        # The request reached Supadata, so count it against the budget whatever
        # it answers — an unanswered credit is still a spent one.
        _record_call(usage)

        # Out of credits on this key: rotate rather than retry.
        if resp.status_code in SUPADATA_CREDIT_STATUS:
            log_warn(f"Supadata key rejected ({resp.status_code}): {resp.text[:120]}")
            return None

        # Large videos are processed asynchronously: 202 + a job id to poll.
        if resp.status_code == 202:
            try:
                job_id = resp.json().get("jobId")
            except ValueError:
                job_id = None
            if not job_id:
                log_warn("Supadata returned 202 without a jobId.")
                return ""
            return _poll_supadata_job(job_id, headers)

        if resp.status_code == 200:
            try:
                text = _supadata_text_from_payload(resp.json())
            except ValueError as e:
                log_error(f"Supadata returned invalid JSON: {e}")
                return ""
            log_info(f"Supadata returned {len(text)} chars")
            return text

        # Retry transient server/rate-limit errors; give up on anything else.
        if resp.status_code in SUPADATA_TRANSIENT_STATUS and attempt < SUPADATA_MAX_RETRIES:
            log_warn(
                f"Transient Supadata status {resp.status_code} "
                f"(attempt {attempt}/{SUPADATA_MAX_RETRIES}); retrying."
            )
            time.sleep(SUPADATA_RETRY_BACKOFF * attempt)
            continue

        log_warn(f"Supadata returned {resp.status_code}: {resp.text[:200]}")
        return ""

    return ""


def _poll_supadata_job(job_id, headers):
    """Poll an async Supadata job until it completes, fails, or attempts run out."""
    job_url = f"{SUPADATA_URL}/{job_id}"
    for attempt in range(SUPADATA_POLL_ATTEMPTS):
        time.sleep(SUPADATA_POLL_DELAY)
        try:
            resp = requests.get(job_url, headers=headers, timeout=SUPADATA_TIMEOUT)
            if resp.status_code != 200:
                log_warn(f"Supadata job poll {resp.status_code}: {resp.text[:200]}")
                continue
            data = resp.json()
            status = data.get("status")
            if status == "failed":
                log_warn(f"Supadata job {job_id} failed.")
                return ""
            text = _supadata_text_from_payload(data)
            if text:
                log_info(f"Supadata job {job_id} completed with {len(text)} chars")
                return text
            log_info(f"Supadata job {job_id} not ready (attempt {attempt + 1}).")
        except Exception as e:
            log_warn(f"Supadata job poll error: {e}")
    log_warn(f"Supadata job {job_id} did not complete in time.")
    return ""


def get_transcript_from_video(video_id):
    """
    Fetch the transcript for a YouTube video.

    Tries Supadata first (free tier, works from blocked CI IPs) when a key is
    set, then falls back to youtube-transcript-api (free, no key, works locally).

    `video_id` may be a full URL or a bare ID. Always returns a dict shaped
    {"transcript": <str>, "budget_exhausted": <bool>} so callers never have to
    handle exceptions or None; an empty transcript means none was available,
    and budget_exhausted marks the "we chose not to spend a credit" case, which
    the caller should retry on a later run rather than report as missing.
    """
    vid = _extract_video_id(video_id)
    if not vid:
        log_warn("No valid video ID; cannot fetch transcript.")
        return {"transcript": "", "budget_exhausted": False}

    text, budget_exhausted = _fetch_supadata(vid)
    if not text:
        text = _fetch_youtube_transcript_api(vid)

    if text:
        budget_exhausted = False
    elif not budget_exhausted:
        log_warn(f"No transcript available for video {vid} from any source.")

    return {"transcript": text, "budget_exhausted": budget_exhausted}
