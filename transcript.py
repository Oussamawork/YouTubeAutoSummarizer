import re
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    CouldNotRetrieveTranscript,
)
from log import log_info, log_warn, log_error


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


def get_transcript_from_video(video_id):
    """
    Fetch the transcript for a YouTube video using youtube-transcript-api.

    `video_id` may be a full URL or a bare ID. Always returns a dict shaped
    {"transcript": <str>} so callers never have to handle exceptions or None;
    an empty string means no usable transcript was available.
    """
    vid = _extract_video_id(video_id)
    if not vid:
        log_warn("No valid video ID; cannot fetch transcript.")
        return {"transcript": ""}

    try:
        log_info(f"Fetching transcript via youtube-transcript-api for video ID: {vid}")
        fetched = YouTubeTranscriptApi().fetch(vid)
        transcript_text = " ".join(snippet.text for snippet in fetched).strip()

        log_info(f"Transcript length: {len(transcript_text)} chars")
        if not transcript_text:
            log_warn("Transcript fetched but contained no text.")
            return {"transcript": ""}

        return {"transcript": transcript_text}

    except TranscriptsDisabled:
        log_warn(f"Transcripts are disabled for video {vid}.")
    except NoTranscriptFound:
        log_warn(f"No transcript found for video {vid}.")
    except VideoUnavailable:
        log_warn(f"Video {vid} is unavailable.")
    except CouldNotRetrieveTranscript as e:
        # Base class for library-level failures, e.g. RequestBlocked / IpBlocked.
        log_error(f"Could not retrieve transcript for {vid}: {e}")
    except Exception as e:
        log_error(f"Unexpected error fetching transcript for {vid}: {e}")

    return {"transcript": ""}
