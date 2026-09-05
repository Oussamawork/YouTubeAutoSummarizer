import json
import os
import re
import tempfile

from log import log_error

def env_int(name, default):
    """
    Read an integer env var, falling back to `default` when it is unset, empty,
    or malformed. An unconfigured GitHub Actions repo variable arrives as ""
    (not absent), so plain int(os.getenv(name, default)) would crash.
    """
    value = (os.getenv(name) or "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        log_error(f"Ignoring invalid {name}={value!r}; using default {default}.")
        return default


def env_flag(name, default=False):
    """
    Boolean twin of env_int. Unset or empty (an unconfigured GitHub Actions repo
    variable arrives as "", not absent) falls back to `default`.
    """
    value = (os.getenv(name) or "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def env_float(name, default):
    """Float twin of env_int: unset/empty/malformed values fall back to default."""
    value = (os.getenv(name) or "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        log_error(f"Ignoring invalid {name}={value!r}; using default {default}.")
        return default


# Function to read channel entries from a file
def read_channels(file_path):
    """
    Read channel entries, one per line: a channel ID (UC…) or an @handle,
    optionally followed by whitespace-separated options. Handles are resolved
    to channel IDs at run time (see scraper.resolve_channel_handle). Blank
    lines and comments ('#' to end of line, whether the line starts with it or
    it follows an entry) are ignored. Returns [] if the file is missing.

    Supported options:
      digest — bundle this channel's new videos into one compact TL;DR digest
               message per run instead of one full summary per video (for
               prolific channels that would otherwise flood the chat).
      max=N  — per-run video cap for this channel (overrides the global default).
               0 means no cap: process everything this channel has due.
      only=a,b,c — process a video only when its title mentions one of these
               keywords (whole words, case-insensitive). Filtering happens
               before the transcript fetch, so skipped videos cost nothing.

    Unknown or malformed options are logged and ignored, so a typo can't make
    the whole channel list unreadable.

    Returns a list of {"channel_id", "digest", "max_per_run", "only"} dicts.
    """
    try:
        with open(file_path, "r") as file:
            channels = []
            for line in file:
                # Strip inline comments too, so an entry can be annotated with
                # the channel's name/handle without it parsing as an option.
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                tokens = line.split()
                entry = {"channel_id": tokens[0], "digest": False, "max_per_run": None,
                         "only": []}
                for token in tokens[1:]:
                    option = token.lower()
                    if option == "digest":
                        entry["digest"] = True
                    elif option.startswith("max="):
                        try:
                            entry["max_per_run"] = max(0, int(option[4:]))
                        except ValueError:
                            log_error(f"Ignoring malformed channel option '{token}' for {tokens[0]}")
                    elif option.startswith("only="):
                        keywords = [k.strip() for k in option[5:].split(",") if k.strip()]
                        if keywords:
                            entry["only"].extend(keywords)
                        else:
                            log_error(f"Ignoring empty channel option '{token}' for {tokens[0]}")
                    else:
                        log_error(f"Ignoring unknown channel option '{token}' for {tokens[0]}")
                channels.append(entry)
            return channels
    except FileNotFoundError:
        log_error(f"Channel IDs file not found: {file_path}")
        return []
    
# Dedup state. Current (v2) schema:
#   {
#     "channels": {channel_id: {"last_video_id": str, "last_published": iso-str}},
#     "pending":  {video_id: {"channel_id": str, "attempts": int}}
#   }
# "channels" holds the per-channel watermark (newest decided video); "pending"
# holds videos deferred for retry (captions not up yet, LLM quota exhausted).
# Legacy (v1) files were a flat {channel_id: last_video_id} map; load_state
# migrates them transparently.

def _empty_state():
    return {"channels": {}, "pending": {}}


def load_state(file_path):
    """
    Load the dedup state, migrating legacy formats in memory. Returns a fresh
    empty state if the file is missing or unreadable; never raises.
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        return _empty_state()
    except (ValueError, OSError) as e:
        log_error(f"Could not read state file {file_path}: {e}. Starting fresh.")
        return _empty_state()

    if not isinstance(data, dict):
        return _empty_state()

    # v1: flat {channel_id: last_video_id}
    if "channels" not in data and all(isinstance(v, str) for v in data.values()):
        return {
            "channels": {cid: {"last_video_id": vid} for cid, vid in data.items()},
            "pending": {},
        }

    channels = data.get("channels")
    pending = data.get("pending")
    return {
        "channels": channels if isinstance(channels, dict) else {},
        "pending": pending if isinstance(pending, dict) else {},
    }


def save_state(file_path, seen):
    """
    Persist the dedup state atomically.

    Writes to a temp file in the same directory and os.replace()s it into place,
    so a crash mid-write can't leave a truncated/corrupt dedup file (which would
    reset state and cause already-sent summaries to be re-sent).
    """
    tmp = None
    try:
        directory = os.path.dirname(os.path.abspath(file_path)) or "."
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".seen_", suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(seen, f, indent=2, sort_keys=True)
        os.replace(tmp, file_path)
        tmp = None  # replaced successfully; nothing to clean up
    except OSError as e:
        log_error(f"Could not write seen-videos file {file_path}: {e}")
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def title_matches(title, keywords):
    """
    True when `title` mentions any keyword as a whole word (case-insensitive),
    or when `keywords` is empty (no filter configured).

    Whole-word matching is the point: a substring test for "eth" would match
    "whether" and "together", and "sol" would match "solve" — so tickers must
    stand on their own. Punctuation is not a word character, so "$BTC",
    "BTC/USD" and "Bitcoin's" all match; note that "sol" does NOT match
    "solana", so list every alias you want (e.g. only=sol,solana).
    """
    if not keywords:
        return True
    if not title:
        return False
    for keyword in keywords:
        if re.search(rf"\b{re.escape(keyword)}\b", title, re.IGNORECASE):
            return True
    return False


def append_jsonl(path, record):
    """
    Append one record as a JSON line to `path`, creating parent directories as
    needed. Returns True on success, False on failure (logged, never raises) —
    callers treat persistence as best-effort.
    """
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
    except (OSError, TypeError, ValueError) as e:
        log_error(f"Could not append record to {path}: {e}")
        return False


def clean_summary(summary: str) -> str:
    """
    Tidy a summary while preserving its line structure (e.g. bullet points),
    so formatted LLM output stays readable in the Telegram message.
    """
    if not summary:
        return ""

    # 1. Replace non-breaking spaces with a regular space
    summary_clean = summary.replace("\u00a0", " ")

    # 2. Collapse runs of spaces/tabs, but keep newlines intact
    summary_clean = re.sub(r"[ \t]+", " ", summary_clean)

    # 3. Collapse 3+ blank lines into a single blank line (one paragraph break)
    summary_clean = re.sub(r"\n{3,}", "\n\n", summary_clean)

    # 4. Trim trailing spaces per line, then strip the whole thing
    summary_clean = "\n".join(line.rstrip() for line in summary_clean.splitlines())

    return summary_clean.strip()