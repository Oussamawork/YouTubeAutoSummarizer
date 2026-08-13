"""
Per-model daily request accounting for the Gemini free tier.

Free-tier quota is granted *per model* — 20 requests a day each for this
project — and it resets at midnight Pacific, not UTC. Each run is a fresh
process on a fresh runner, so without a persisted count every run rediscovers a
spent model by spending a request on it. Counters live in a small JSON file the
daily workflow commits back, exactly like `data/supadata_usage.json`.

Nothing here raises: a missing, corrupt or unwritable counter file must never
cost a summary, so every failure degrades to "assume there is budget left" and
lets the API be the authority.
"""
import json
import os
from datetime import datetime, timedelta, timezone

from helpers import env_int
from log import log_info, log_warn

GEMINI_USAGE_FILE = os.getenv("GEMINI_USAGE_FILE") or "data/gemini_usage.json"
# Requests per model per day on the free tier. Measured from AI Studio for this
# project on 2026-08-13; published third-party figures (1,500/day) are wrong.
GEMINI_REQUESTS_PER_DAY = env_int("GEMINI_REQUESTS_PER_DAY", 20)

try:  # tzdata is present on the CI runners and on most dev machines
    from zoneinfo import ZoneInfo

    _PACIFIC = ZoneInfo("America/Los_Angeles")
except Exception:  # pragma: no cover - depends on the host's tz database
    _PACIFIC = None


def quota_day(now=None):
    """
    The date whose budget is currently in force. Google rolls RPD at 00:00
    Pacific, so a UTC day boundary would reset the counter up to 8 hours early
    and let a run overspend. Without a tz database, PST is assumed: the reset
    then lands an hour off during daylight saving, which shifts the boundary
    but never loses the count.
    """
    now = now or datetime.now(timezone.utc)
    if _PACIFIC is not None:
        return now.astimezone(_PACIFIC).date()
    return (now - timedelta(hours=8)).date()


def load_usage(now=None):
    """Today's per-model counts, resetting when the Pacific day has rolled."""
    day = quota_day(now).isoformat()
    try:
        with open(GEMINI_USAGE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("usage file is not an object")
    except (OSError, ValueError):
        data = {}
    if data.get("day") != day:
        data = {"day": day, "models": {}}
    models = data.get("models")
    data["models"] = models if isinstance(models, dict) else {}
    return data


def save_usage(usage):
    try:
        directory = os.path.dirname(GEMINI_USAGE_FILE)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(GEMINI_USAGE_FILE, "w", encoding="utf-8") as f:
            json.dump(usage, f, indent=2, sort_keys=True)
    except OSError as e:
        # Losing an increment only risks over-attempting later, which the API
        # rejects harmlessly. Losing the run would be worse.
        log_warn(f"Could not persist Gemini usage: {e}")


def used(model, usage=None):
    """Requests already spent on `model` today."""
    usage = usage if usage is not None else load_usage()
    count = usage["models"].get(model)
    return count if isinstance(count, int) else 0


def is_exhausted(model, usage=None):
    """True when `model` has no free requests left today."""
    return used(model, usage) >= GEMINI_REQUESTS_PER_DAY


def record(model, usage=None):
    """Count one served request against `model`'s daily budget."""
    usage = usage if usage is not None else load_usage()
    usage["models"][model] = used(model, usage) + 1
    save_usage(usage)
    return usage


def mark_exhausted(model):
    """
    Record that the API itself said `model` is out of requests for the day.

    Set to the cap rather than incremented: a 429 means the real count is at
    least the limit, whatever our local tally says — the API is the authority
    and our count can only ever be an undercount (failed requests aren't
    metered).
    """
    usage = load_usage()
    usage["models"][model] = max(used(model, usage), GEMINI_REQUESTS_PER_DAY)
    save_usage(usage)
    log_info(f"Gemini {model} marked spent for today ({quota_day().isoformat()}).")
    return usage


def report():
    """One line of remaining budget per model, for the end-of-run log."""
    usage = load_usage()
    if not usage["models"]:
        return ""
    return ", ".join(
        f"{model}={max(0, GEMINI_REQUESTS_PER_DAY - count)}/{GEMINI_REQUESTS_PER_DAY} left"
        for model, count in sorted(usage["models"].items())
    )
