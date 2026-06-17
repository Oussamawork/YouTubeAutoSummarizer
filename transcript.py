import os
import re
import time
from urllib.parse import urlparse, parse_qs

import requests
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
    works from blocked CI IPs. Returns "" if no key, on error, or if empty.
    Uses mode=native so only existing captions are returned (no paid AI generation).
    """
    if not SUPADATA_API_KEY:
        return ""

    headers = {"x-api-key": SUPADATA_API_KEY}
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
    {"transcript": <str>} so callers never have to handle exceptions or None;
    an empty string means no usable transcript was available.
    """
    vid = _extract_video_id(video_id)
    if not vid:
        log_warn("No valid video ID; cannot fetch transcript.")
        return {"transcript": ""}

    text = _fetch_supadata(vid)
    if not text:
        text = _fetch_youtube_transcript_api(vid)

    if not text:
        log_warn(f"No transcript available for video {vid} from any source.")

    return {"transcript": text}
