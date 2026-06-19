import json
import os
import re
import tempfile

from log import log_error

# Function to read channel IDs from a file
def read_channel_ids(file_path):
    """
    Read channel IDs, one per line. Blank lines and lines starting with '#'
    (comments) are ignored. Returns [] if the file is missing.
    """
    try:
        with open(file_path, "r") as file:
            ids = []
            for line in file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                ids.append(line)
            return ids
    except FileNotFoundError:
        log_error(f"Channel IDs file not found: {file_path}")
        return []
    
# Dedup state: map of {channel_id: last_processed_video_id}
def load_seen_videos(file_path):
    """Load the per-channel last-seen video IDs. Returns {} if missing/invalid."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (ValueError, OSError) as e:
        log_error(f"Could not read seen-videos file {file_path}: {e}. Starting fresh.")
        return {}


def save_seen_videos(file_path, seen):
    """
    Persist the per-channel last-seen video IDs atomically.

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


# Function to save results to a JSON file
def save_to_json(data, filename):
    try:
        with open(filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving data to JSON file: {e}")

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