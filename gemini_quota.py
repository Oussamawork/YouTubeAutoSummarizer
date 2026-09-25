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

from helpers import env_int, write_json_atomic
from log import log_info

GEMINI_USAGE_FILE = os.getenv("GEMINI_USAGE_FILE") or "data/gemini_usage.json"
# Requests per model per day on the free tier. Measured from AI Studio for this
# project on 2026-08-13; published third-party figures (1,500/day) are wrong.
GEMINI_REQUESTS_PER_DAY = env_int("GEMINI_REQUESTS_PER_DAY", 20)
# How long the API's "your daily quota is gone" verdict is believed when our own
# tally says the model still has requests left. The two disagreed badly in
# production: on 2026-08-18 gemini-3.7-flash was written off at 00:41 Pacific
# after 3 recorded requests and then sat idle all day with 17 of its 20 free
# requests unspent, which is what pushed that evening's videos onto the deferral
# path. A rejected request costs no quota, so re-probing a flagged model once a
# run is nearly free; believing a premature verdict costs a whole day of the
# best summary model.
GEMINI_SPENT_RECHECK_MINUTES = env_int("GEMINI_SPENT_RECHECK_MINUTES", 45)
# Every repeat of the verdict doubles the wait, so a model that really is out
# for the day is left alone instead of being probed by every two-hourly run.
GEMINI_SPENT_RECHECK_MAX_MINUTES = env_int("GEMINI_SPENT_RECHECK_MAX_MINUTES", 360)

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
# The quota Google actually names inside a 429. Its `message` is generic ("You
# exceeded your current quota"), so the id is the only thing that says whether
# the limit counts requests or tokens — and a fixed-length body preview in the
# log always cut off before the violation list, which is why a model written off
# after 3 requests could not be explained from the logs alone.
_QUOTA_ID = re.compile(r'"quota(?:Id|_id|Metric|_metric)"\s*:\s*"([^"]+)"')
_QUOTA_VALUE = re.compile(r'"quotaValue"\s*:\s*"?(\d+)"?')


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


def violation_summary(body):
    """
    One line naming the quota(s) a 429 blames, for the log. "" when the body
    carries no violation details.

    Worth extracting rather than leaning on a body preview: the ids sit at the
    end of the payload, past the generic message, so the preview never reached
    them — and the id is what distinguishes "20 requests a day" from a
    token-per-day limit that a handful of long transcripts can exhaust.
    """
    body = body or ""
    parts = []
    ids = list(dict.fromkeys(_QUOTA_ID.findall(body)))
    if ids:
        parts.append("quota=" + ",".join(ids))
    values = list(dict.fromkeys(_QUOTA_VALUE.findall(body)))
    if values:
        parts.append("limit=" + ",".join(values))
    return "; ".join(parts)


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
    data["spent"] = _normalize_spent(data.get("spent"))
    return data


def _normalize_spent(spent):
    """
    Today's API write-offs as `{model: {"at", "used", "confirmations"}}`.

    The v1 shape — a plain list of model names — is still accepted, because the
    counter file is committed by one run and read by the next, so a deploy
    always meets a file written by the previous version. A migrated entry has no
    timestamp, which reads as "due for a recheck": the safe direction, since the
    worst case is one rejected request.
    """
    if isinstance(spent, dict):
        items = list(spent.items())
    elif isinstance(spent, list):
        items = [(model, {}) for model in spent]
    else:
        return {}
    normalized = {}
    for model, entry in items:
        if not isinstance(model, str):
            continue
        entry = entry if isinstance(entry, dict) else {}
        at = entry.get("at")
        count = entry.get("used")
        confirmations = entry.get("confirmations")
        normalized[model] = {
            "at": at if isinstance(at, str) else None,
            "used": count if isinstance(count, int) else 0,
            "confirmations": confirmations if isinstance(confirmations, int) and confirmations > 0 else 1,
        }
    return normalized


def _parse_iso(value):
    """A stored timestamp as an aware datetime, or None when unusable."""
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def recheck_delay_minutes(entry):
    """How long a write-off is honored before the model is tried again."""
    confirmations = max(1, (entry or {}).get("confirmations") or 1)
    # Doubling per repeat, so the first (and usually wrong) verdict costs the
    # model one run rather than a day, while a model that keeps refusing is
    # asked less and less often.
    delay = GEMINI_SPENT_RECHECK_MINUTES * (2 ** (confirmations - 1))
    return min(delay, GEMINI_SPENT_RECHECK_MAX_MINUTES)


def _recheck_due(entry, now=None):
    """True once a written-off model has rested long enough to be re-probed."""
    at = _parse_iso((entry or {}).get("at"))
    if at is None:  # migrated or corrupt flag: nothing says how long it has sat
        return True
    return (now or datetime.now(timezone.utc)) >= at + timedelta(
        minutes=recheck_delay_minutes(entry)
    )


def save_usage(usage):
    # Atomic, like the dedup state: a truncated counter file reads as "nothing
    # spent today". Losing an increment only risks over-attempting later, which
    # the API rejects harmlessly, so a failure is logged, never raised.
    write_json_atomic(GEMINI_USAGE_FILE, usage)


def used(model, usage=None):
    """Requests already spent on `model` today."""
    usage = usage if usage is not None else load_usage()
    count = usage["models"].get(model)
    return count if isinstance(count, int) else 0


def is_exhausted(model, usage=None, now=None):
    """
    True when `model` cannot serve another request right now.

    Two sources say so, and they are not equally final. Our own tally reaching
    the daily cap is: those requests were served and cannot be unserved. The
    API's 429 verdict is not — it has written a model off after three requests,
    a limit whose shape we cannot see from here — so it is honored for a
    cooling-off period and then re-tested, instead of costing the model the rest
    of its day. Skipping it for the remainder of the *current* run is the
    caller's job (`summarizer._EXHAUSTED_PROVIDERS`); this decides whether a
    later run may try again.
    """
    usage = usage if usage is not None else load_usage()
    if used(model, usage) >= GEMINI_REQUESTS_PER_DAY:
        return True
    entry = usage["spent"].get(model)
    return bool(entry) and not _recheck_due(entry, now)


def written_off(model, usage=None):
    """
    True when the API's "out of requests" verdict for `model` is on file today,
    whether or not it is still being honored.

    `is_exhausted` answers "may I call this model right now?", which goes False
    the moment the cooling-off period lapses. Callers that need to know the
    verdict *exists* — to keep treating a re-probe that fails as a budget
    problem rather than a broken video — have to ask separately.
    """
    usage = usage if usage is not None else load_usage()
    return model in usage["spent"]


def record(model, usage=None):
    """Count one served request against `model`'s daily budget."""
    usage = usage if usage is not None else load_usage()
    usage["models"][model] = used(model, usage) + 1
    if usage["spent"].pop(model, None) is not None:
        # The API just served a request it had refused earlier today, so that
        # verdict is stale. Clearing the flag puts the model back at its proper
        # place in the chain for the rest of the day instead of leaving the
        # preferred model skipped on the strength of one old 429.
        log_info(f"Gemini {model} answered again; clearing today's spent flag.")
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

    The decision is provisional, not final: the flag records when it was set and
    how many verdicts the API has now given, so `is_exhausted` can re-probe the
    model later instead of surrendering the rest of its day (see
    `GEMINI_SPENT_RECHECK_MINUTES`).
    """
    usage = load_usage()
    confirmations = ((usage["spent"].get(model) or {}).get("confirmations") or 0) + 1
    entry = {
        "at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "used": used(model, usage),
        "confirmations": confirmations,
    }
    usage["spent"][model] = entry
    save_usage(usage)
    log_info(
        f"Gemini {model} marked spent for today ({quota_day().isoformat()}) "
        f"after {entry['used']} recorded request(s); skipped for the next "
        f"{recheck_delay_minutes(entry)} min (verdict #{confirmations})."
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
        + _capped_note(usage["spent"].get(model))
        for model, count in sorted(models.items())
    )


def _capped_note(entry):
    """
    How a written-off model is annotated in the end-of-run budget line.

    The remaining-requests figure is what exposed the bug this guards against —
    "gemini-3.7-flash=17/20 left (capped by API)" was the whole day's loss in
    one line — so the note now also says when the model comes back, which is the
    next question that line raises.
    """
    if not entry:
        return ""
    if _recheck_due(entry):
        return " (capped by API, due for a retry)"
    return f" (capped by API, retried after {recheck_delay_minutes(entry)} min)"
