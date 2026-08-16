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
import re
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


# A free-tier 429 comes in two very different flavors, and the response body is
# the only thing that tells them apart: a per-day quota is gone until the
# Pacific reset, while a per-minute one (5 requests/minute, or the token
# equivalent) clears in about a minute. Google names the offending quota in the
# error payload — "GenerateRequestsPerDayPerProjectPerModel-FreeTier" versus
# "...PerMinute...". Matching the quota id as text rather than walking the JSON
# keeps this working whether the id arrives under `quotaId`, `quotaMetric`, or
# only in the human-readable message.
_PER_DAY_MARKER = re.compile(r"per[_\s-]?day", re.IGNORECASE)
_PER_MINUTE_MARKER = re.compile(r"per[_\s-]?minute", re.IGNORECASE)
# `"retryDelay": "27s"` inside google.rpc.RetryInfo. The REST API sends this in
# the body, not in a Retry-After header, so a header-only reader never sees it.
_RETRY_DELAY = re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"')


def classify_429(body):
    """
    Which quota a 429 refers to: "day", "minute", or "unknown".

    Per-day wins when a payload names both, because that is the binding limit —
    waiting out the minute would just hit the day again.
    """
    body = body or ""
    if _PER_DAY_MARKER.search(body):
        return "day"
    if _PER_MINUTE_MARKER.search(body):
        return "minute"
    return "unknown"


def retry_delay_seconds(body):
    """The RetryInfo delay from a 429 body in seconds, or None if absent."""
    match = _RETRY_DELAY.search(body or "")
    if not match:
        return None
    try:
        return max(0, int(float(match.group(1))))
    except (TypeError, ValueError):  # pragma: no cover - regex already bounds it
        return None


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
    spent = data.get("spent")
    data["spent"] = sorted(m for m in spent if isinstance(m, str)) if isinstance(spent, list) else []
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
    usage = usage if usage is not None else load_usage()
    return model in usage["spent"] or used(model, usage) >= GEMINI_REQUESTS_PER_DAY


def record(model, usage=None):
    """Count one served request against `model`'s daily budget."""
    usage = usage if usage is not None else load_usage()
    usage["models"][model] = used(model, usage) + 1
    save_usage(usage)
    return usage


def mark_exhausted(model):
    """
    Record that the API itself said `model` is out of requests for the day.

    Flagged rather than counted up to the cap. Writing the cap into the count
    made the number mean two different things — "20 requests served" and "the
    API said stop after 3" — so the file could not answer how much budget a run
    actually used, which is the one question it exists to answer. The flag
    carries the "stop asking" decision; the count stays a true tally.
    """
    usage = load_usage()
    if model not in usage["spent"]:
        usage["spent"] = sorted(usage["spent"] + [model])
    save_usage(usage)
    log_info(
        f"Gemini {model} marked spent for today ({quota_day().isoformat()}) "
        f"after {used(model, usage)} recorded request(s)."
    )
    return usage


def report():
    """One line of remaining budget per model, for the end-of-run log."""
    usage = load_usage()
    models = dict(usage["models"])
    # A model can be flagged spent without ever having served a request today,
    # and that is exactly the case worth seeing in the log.
    for model in usage["spent"]:
        models.setdefault(model, 0)
    if not models:
        return ""
    return ", ".join(
        f"{model}={max(0, GEMINI_REQUESTS_PER_DAY - count)}/{GEMINI_REQUESTS_PER_DAY} left"
        + (" (capped by API)" if model in usage["spent"] else "")
        for model, count in sorted(models.items())
    )
