"""
Raw transcript persistence: the source text, saved BEFORE any cleaning, so
research can be reprocessed later without spending another transcript credit
and every claim can be traced to the words that were actually captured.

Layout (lightweight, in the repo's data/ style, committed by the daily job):
  data/transcripts/<video_id>.json.gz    the raw text plus metadata (gzip)
  data/research/transcript_records.jsonl one index row per stored transcript
The gzip file is written atomically and never overwritten with cleaned text;
a second capture of the same video with a different hash is stored beside it
as <video_id>.<hash12>.json.gz and the index gains a row. A persistence
failure is logged and reported back so the research state can say so — it is
never allowed to look like a successful capture.
"""
import gzip
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone

from helpers import append_jsonl
from log import log_error, log_warn

TRANSCRIPTS_DIR = os.getenv("TRANSCRIPTS_DIR") or "data/transcripts"
TRANSCRIPT_INDEX = os.getenv("TRANSCRIPT_INDEX") or "data/research/transcript_records.jsonl"


def transcript_hash(raw_text):
    return hashlib.sha256((raw_text or "").encode("utf-8")).hexdigest()


def transcript_path(video_id, digest=None, directory=None):
    directory = directory or TRANSCRIPTS_DIR
    name = f"{video_id}.json.gz" if not digest else f"{video_id}.{digest[:12]}.json.gz"
    return os.path.join(directory, name)


def store_transcript(video_details, raw_text, source, reason, language=None,
                     transcript_type=None, directory=None, index_path=None):
    """
    Persist the raw transcript. Returns the record dict (with "stored": True)
    or, on failure, a record with "stored": False and "error" set. Idempotent:
    an identical transcript for the same video is not rewritten.
    """
    directory = directory or TRANSCRIPTS_DIR
    index_path = index_path or TRANSCRIPT_INDEX
    digest = transcript_hash(raw_text)
    video_id = video_details.get("video_id") or ""
    record = {
        "video_id": video_id,
        "video_url": video_details.get("video_url"),
        "channel_id": video_details.get("channel_id"),
        "channel_name": video_details.get("channel_name"),
        "video_title": video_details.get("video_title"),
        "published_at": video_details.get("published_at"),
        "duration_seconds": video_details.get("duration_seconds"),
        "transcript_source": source,
        "transcript_retrieval_reason": reason,
        "transcript_language": language,
        "transcript_type": transcript_type,
        "timestamps_available": _looks_timestamped(raw_text),
        "raw_char_count": len(raw_text or ""),
        "transcript_hash": digest,
        "acquired_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "stored": False,
        "path": None,
    }
    if not video_id or not raw_text:
        record["error"] = "missing video_id or empty transcript"
        log_warn(f"Transcript not persisted: {record['error']}.")
        return record
    path = transcript_path(video_id, directory=directory)
    existing = load_transcript(video_id, directory=directory)
    if existing and existing.get("transcript_hash") == digest:
        record.update(stored=True, path=path, acquired_at=existing.get("acquired_at") or record["acquired_at"])
        return record
    if existing:
        path = transcript_path(video_id, digest, directory=directory)
    tmp = None
    try:
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp_", suffix=".json.gz")
        os.close(fd)
        payload = {k: v for k, v in record.items() if k not in ("stored", "path")}
        payload["raw_transcript"] = raw_text
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, path)
        tmp = None
        record.update(stored=True, path=path)
        index_row = {k: v for k, v in record.items() if k != "stored"}
        append_jsonl(index_path, index_row)
    except (OSError, TypeError, ValueError) as e:
        record["error"] = str(e)
        log_error(f"Could not persist transcript for {video_id}: {e}")
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return record


def load_transcript(video_id, digest=None, directory=None):
    """The stored payload (metadata + raw_transcript) or None."""
    path = transcript_path(video_id, digest, directory=directory)
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError, EOFError):
        return None


def _looks_timestamped(text):
    head = (text or "")[:4000]
    return bool(__import__("re").search(r"(?m)^\s*\[?\d{1,2}:\d{2}(:\d{2})?", head))
