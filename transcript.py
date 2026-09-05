import json
import os
import re
import time
from datetime import datetime, timezone
from urllib.parse import urlparse, parse_qs

import requests
import gemini_quota
from helpers import env_flag, env_int, write_json_atomic
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
# It also answers with 429 — the same status as ordinary rate limiting — and
# only the body's error code tells the two apart. Retrying a spent key just
# burns the run's time and, worse, never rotates to a key that still has
# credits, so the code has to be matched explicitly.
SUPADATA_CREDIT_ERRORS = {"limit-exceeded"}

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
# Supadata resets credits on the plan's anniversary day, which is usually NOT
# the 1st (a dashboard may read e.g. "Credits reset on 08/17"). Pacing has to
# follow that cycle: assuming calendar months would compute a huge allowance in
# the last days of a month and drain the real pool in a day or two. Clamped to
# 1..28 so every month has the day.
SUPADATA_RESET_DAY = min(max(env_int("SUPADATA_RESET_DAY", 1), 1), 28)
# Safety net against a misconfigured reset day: never spend more than this
# fraction of the budget in a single day, whatever the cycle math says.
SUPADATA_MIN_CYCLE_DAYS = 28


def _daily_pacing_enabled():
    """
    Whether to ration the cycle's credits across its remaining days.

    Off by default: a run spends whatever the cycle has left, so every video
    published today is summarised today instead of trickling out over later
    runs. The cost is that a busy stretch can exhaust the pool before the reset
    date — when it does, videos defer via `budget_exhausted` rather than being
    written off, so nothing is lost, but there will be a quiet gap until credits
    return. Set SUPADATA_DAILY_PACING=true to restore rationing. Read at call
    time so tests and reloads see the current environment.
    """
    return env_flag("SUPADATA_DAILY_PACING", default=False)


def _shift_month(day, months):
    """Same day-of-month, `months` later/earlier (day is already <= 28)."""
    index = (day.year * 12 + day.month - 1) + months
    return day.replace(year=index // 12, month=index % 12 + 1)


def cycle_bounds(today=None):
    """(start, end) of the billing cycle containing `today`, per the reset day."""
    today = today or datetime.now(timezone.utc).date()
    if today.day >= SUPADATA_RESET_DAY:
        start = today.replace(day=SUPADATA_RESET_DAY)
        return start, _shift_month(start, 1)
    end = today.replace(day=SUPADATA_RESET_DAY)
    return _shift_month(end, -1), end


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


def _load_usage(today=None):
    """Usage counters for the current billing cycle and day; each resets when
    its period rolls over. Counters carry a `cycle` (the cycle's start date)
    rather than a calendar month, so a mid-month reset day is honored."""
    today = today or datetime.now(timezone.utc).date()
    cycle_start = cycle_bounds(today)[0].isoformat()
    day = today.isoformat()
    try:
        with open(SUPADATA_USAGE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("usage file is not an object")
    except (OSError, ValueError):
        data = {}
    if data.get("cycle") != cycle_start:
        data = {"cycle": cycle_start, "count": 0}
    if data.get("day") != day:
        data["day"], data["day_count"] = day, 0
    data.setdefault("count", 0)
    data.setdefault("day_count", 0)
    return data


def _save_usage(usage):
    # Atomic, like the dedup state: a counter file truncated by a crash reads
    # as "nothing spent" and authorises draining the pool. Losing one update
    # is better than losing the run, so a failure is logged, never raised.
    write_json_atomic(SUPADATA_USAGE_FILE, usage, sort_keys=False)


def _record_call(usage):
    """Count one Supadata request against the month's and today's budget."""
    usage["count"] = usage.get("count", 0) + 1
    usage["day_count"] = usage.get("day_count", 0) + 1
    _save_usage(usage)


def daily_allowance(usage=None, today=None):
    """
    How many fetches today may use. Returns 0 when the cycle's budget is spent.

    Unpaced (the default), that is everything the cycle has left, so a day's
    videos are all processed on the day they are published. With
    SUPADATA_DAILY_PACING enabled it is instead the credits left in this billing
    cycle spread over the days remaining in it (today included), so they last
    until the reset date — and capped at budget/SUPADATA_MIN_CYCLE_DAYS per day,
    so a wrong reset day (which would make "days remaining" tiny) can't
    authorise draining the pool in a single run.
    """
    today = today or datetime.now(timezone.utc).date()
    usage = usage if usage is not None else _load_usage(today)
    budget = monthly_budget()
    remaining = budget - usage.get("count", 0)
    if remaining <= 0:
        return 0
    if not _daily_pacing_enabled():
        return remaining
    days_left = max(1, (cycle_bounds(today)[1] - today).days)
    paced = remaining // days_left
    ceiling = max(1, budget // SUPADATA_MIN_CYCLE_DAYS)
    return max(1, min(paced, ceiling))


def budget_status(today=None):
    """(allowed_today, used_today, remaining_this_cycle) — for logging."""
    today = today or datetime.now(timezone.utc).date()
    usage = _load_usage(today)
    return (
        daily_allowance(usage, today),
        usage.get("day_count", 0),
        max(0, monthly_budget() - usage.get("count", 0)),
    )


# --- Gemini video transcripts -----------------------------------------------
#
# Supadata's free tier runs dry (it did on 2026-08-11, and the pipeline went
# quiet for two days), and youtube-transcript-api is IP-blocked from CI runners,
# so a spent credit pool used to mean no summaries at all. Gemini accepts a
# YouTube URL directly and fetches the video server-side: no credit, and no
# runner IP involved. Measured on a 20-minute video: ~123k input tokens, ~6k
# output, ~125 seconds.
#
# Free-tier quota is per model (5 RPM / 250k TPM / 20 RPD), so a list of models
# is tried in turn — each has its own daily bucket. Summarization deliberately
# runs on a different model (GEMINI_MODEL, gemini-3.7-flash), so transcription
# can never eat the summary budget.
GEMINI_TRANSCRIPT_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
# Kept disjoint from summarizer's model chain on purpose — 3.7 and 3.6 are
# reserved for summaries, where model quality is visible in the output. A
# transcript is a transcription of speech that already exists, so the older
# generations do it just as well, and three of them give 60 requests/day: more
# than the summary side can consume, which is where the quality matters.
GEMINI_TRANSCRIPT_MODELS_DEFAULT = (
    "gemini-3.5-flash,gemini-3-flash-preview,gemini-2.5-flash"
)
# Measured at ~125s for a 20-minute video; 2.5-flash timed out at 60s, which is
# what a too-tight timeout looks like from the outside.
GEMINI_TRANSCRIPT_TIMEOUT = env_int("GEMINI_TRANSCRIPT_TIMEOUT", 300)
GEMINI_TRANSCRIPT_MAX_OUTPUT_TOKENS = env_int("GEMINI_TRANSCRIPT_MAX_OUTPUT_TOKENS", 32768)
# Reasons that mean a transcript actually arrived. Defined beside the code that
# produces them because the scraper's end-of-run failure breakdown filters on
# this set: it used to keep its own hand-written copy, and when `gemini_ok` was
# added here the copy was not updated, so every Gemini success was counted and
# reported as a transcript failure.
TRANSCRIPT_SUCCESS_REASONS = frozenset({"ok", "gemini_ok", "fallback_ok"})
# Google rejects an over-long video with a 400 naming the context window. Every
# Gemini model here shares that 1,048,576-token window, so the rotation cannot
# rescue it — trying the rest only spends requests to be told the same thing.
GEMINI_TOO_LARGE_MARKER = "input token count exceeds"
GEMINI_TRANSCRIPT_RETRY_STATUS = {500, 502, 503, 504}
GEMINI_TRANSCRIPT_MAX_RETRIES = 2
GEMINI_TRANSCRIPT_RETRY_BACKOFF = 5  # base seconds, multiplied by the attempt
# Models that hit a rate limit this run without the API naming a per-day quota.
# Such a limit may clear within the hour, so it must not be written into the
# day's counter — but asking again for every video in the same run only buys
# the same refusal, so it is remembered for the length of the process.
_RATE_LIMITED_THIS_RUN = set()
# A model that answers with a sentence *about* the video instead of its words
# has failed at transcription. Videos this short are already filtered out by the
# duration gate, so anything below this is a refusal or a summary, not speech.
GEMINI_MIN_TRANSCRIPT_CHARS = env_int("GEMINI_MIN_TRANSCRIPT_CHARS", 500)
GEMINI_TRANSCRIPT_PROMPT = (
    "Transcribe the spoken audio of this video verbatim, in full, as plain "
    "text. Do not summarize, do not paraphrase, do not add commentary, "
    "speaker labels or timestamps. Output only the transcript text."
)


def _gemini_transcript_models():
    """
    Transcription models, in order. Read at call time so tests and reloads see
    the current environment; an unset GitHub Actions variable arrives as "",
    which must fall back to the default rather than becoming an empty list.
    """
    raw = os.getenv("GEMINI_TRANSCRIPT_MODELS") or GEMINI_TRANSCRIPT_MODELS_DEFAULT
    return [model.strip() for model in raw.split(",") if model.strip()]


def _gemini_text_from_payload(data):
    """Concatenated text parts of a generateContent response ("" if none)."""
    if not isinstance(data, dict):
        return ""
    try:
        parts = data["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError):
        return ""
    if not isinstance(parts, list):
        return ""
    return "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()


def _fetch_gemini_with_model(vid, model, api_key):
    """
    One model's attempt at transcribing a video. Returns (text, reason); an
    empty text with reason "quota" means this model's daily requests are spent,
    which tells the caller to rotate rather than retry.
    """
    payload = {
        "contents": [{"parts": [
            {"text": GEMINI_TRANSCRIPT_PROMPT},
            {"file_data": {"file_uri": f"https://www.youtube.com/watch?v={vid}"}},
        ]}],
        "generationConfig": {
            "maxOutputTokens": GEMINI_TRANSCRIPT_MAX_OUTPUT_TOKENS,
            "temperature": 0.0,
        },
    }
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    url = GEMINI_TRANSCRIPT_URL.format(model=model)
    log_info(f"Fetching transcript via Gemini ({model}) for video ID: {vid}")

    for attempt in range(1, GEMINI_TRANSCRIPT_MAX_RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload,
                                 timeout=GEMINI_TRANSCRIPT_TIMEOUT)
        except requests.RequestException as e:
            log_warn(f"Gemini request error (attempt {attempt}/{GEMINI_TRANSCRIPT_MAX_RETRIES}): {e}")
            if attempt < GEMINI_TRANSCRIPT_MAX_RETRIES:
                time.sleep(GEMINI_TRANSCRIPT_RETRY_BACKOFF * attempt)
                continue
            return "", "unreachable"

        # 429 is the free tier's daily/per-minute cap. Retrying the same model
        # cannot help — the next model has its own bucket.
        if resp.status_code == 429:
            body = resp.text or ""
            kind = gemini_quota.classify_429(body)
            # Same reason as in summarizer.py: the quota id is the only part of
            # the body that says which limit was hit, and it sits past the
            # preview.
            detail = gemini_quota.violation_summary(body)
            log_warn(
                f"Gemini {model} rate-limited (429, {kind} quota"
                + (f", {detail}" if detail else "")
                + f"): {body[:200]}"
            )
            return "", f"quota_{kind}"

        if resp.status_code in GEMINI_TRANSCRIPT_RETRY_STATUS and attempt < GEMINI_TRANSCRIPT_MAX_RETRIES:
            log_warn(f"Transient Gemini status {resp.status_code}; retrying.")
            time.sleep(GEMINI_TRANSCRIPT_RETRY_BACKOFF * attempt)
            continue

        if resp.status_code != 200:
            log_warn(f"Gemini {model} returned {resp.status_code}: {resp.text[:200]}")
            if GEMINI_TOO_LARGE_MARKER in (resp.text or "").lower():
                return "", "too_large"
            return "", f"http_{resp.status_code}"

        try:
            text = _gemini_text_from_payload(resp.json())
        except ValueError as e:
            log_error(f"Gemini returned invalid JSON: {e}")
            return "", "invalid_json"

        if len(text) < GEMINI_MIN_TRANSCRIPT_CHARS:
            # Either a refusal or a summary of the video; both are useless as a
            # transcript, and another model may well answer properly.
            log_warn(
                f"Gemini {model} returned {len(text)} chars, under the "
                f"{GEMINI_MIN_TRANSCRIPT_CHARS}-char floor — not a transcript."
            )
            return "", "too_short"

        log_info(f"Gemini ({model}) returned {len(text)} chars")
        return text, "ok"

    return "", "retries_exhausted"


def _fetch_gemini_transcript(vid):
    """
    Transcribe a video with the first model that can. Returns
    (text, quota_exhausted, reason): quota_exhausted is True when every model
    was out of daily requests, which is a "come back later", not a video
    without captions — the caller defers instead of writing the video off.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "", False, "no_gemini_key"

    models = _gemini_transcript_models()
    quota_hit = 0
    reason = "no_gemini_key"
    for index, model in enumerate(models):
        # A model already at its daily cap is still capped: asking again costs a
        # round trip per video per model to be told the same thing, and the cap
        # outlives the run, so the count has to as well.
        if gemini_quota.is_exhausted(model) or model in _RATE_LIMITED_THIS_RUN:
            log_info(f"Skipping Gemini {model} (rate-limited or out of daily quota).")
            quota_hit, reason = quota_hit + 1, "quota"
            continue
        # A model carrying the API's write-off is being re-probed on the chance
        # the verdict has lapsed. Remember that, because how the probe turns out
        # decides whether this is a budget problem or a broken video.
        reprobe = gemini_quota.written_off(model)
        text, reason = _fetch_gemini_with_model(vid, model, api_key)
        if text:
            gemini_quota.record(model)
            return text, False, "gemini_ok"
        if reason == "too_large":
            # The video does not fit any model here, so the rotation has
            # nothing left to try. Not a budget problem: the video stays queued
            # and a Supadata credit can still transcribe it later.
            log_warn(
                f"Video {vid} exceeds the Gemini context window; skipping the "
                f"remaining {len(models) - index - 1} model(s)."
            )
            return "", False, "gemini_too_large"
        if reason.startswith("quota"):
            # Only a per-day 429 means this model is finished until the Pacific
            # reset. A per-minute one clears on its own, and rotating to the
            # next model is a cheaper answer than writing off the day — but
            # re-asking it for every video in the same run just buys the same
            # refusal, so it is skipped for the rest of the run.
            if reason == "quota_day":
                gemini_quota.mark_exhausted(model)
            else:
                _RATE_LIMITED_THIS_RUN.add(model)
            quota_hit += 1
        elif reprobe:
            # The re-probe failed for some other reason. It is still a model the
            # API has written off today, so it must keep counting as "out of
            # budget": otherwise `quota_hit` falls short of the model count, the
            # caller gets budget_exhausted=False, and scraper.py spends one of
            # the video's give-up attempts on what is really a quota outage —
            # eight of those and a perfectly good video is written off as having
            # no transcript. Skipping it for the rest of the run also stops one
            # lapsed verdict from being re-probed once per video.
            _RATE_LIMITED_THIS_RUN.add(model)
            quota_hit += 1
        if index + 1 < len(models):
            log_warn(f"Gemini {model} failed ({reason}); trying {models[index + 1]}.")

    if quota_hit == len(models):
        log_warn(f"All {len(models)} Gemini transcript model(s) are out of daily quota.")
        return "", True, "gemini_quota"
    log_warn(f"All {len(models)} Gemini transcript model(s) failed (last: {reason}).")
    return "", False, f"gemini_{reason}"


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


def _is_credit_exhausted(resp):
    """
    True when a response means "this key's credit pool is spent" (rotate to the
    next key) rather than "slow down" (retry the same key). 402/403 say it by
    status alone; a 429 only counts when the body names a credit error, since
    Supadata reuses 429 for genuine rate limiting. A body we can't parse is
    treated as a rate limit, which errs toward retrying rather than writing a
    working key off.
    """
    if resp.status_code in SUPADATA_CREDIT_STATUS:
        return True
    if resp.status_code != 429:
        return False
    try:
        error = (resp.json() or {}).get("error")
    except (ValueError, AttributeError):
        return False
    return isinstance(error, str) and error.strip().lower() in SUPADATA_CREDIT_ERRORS


def _fetch_supadata(vid):
    """
    Primary source: Supadata hosted API (free tier). Server-side fetch, so it
    works from blocked CI IPs. Returns (text, budget_exhausted, reason): text is
    "" if no key, on error, or if empty; budget_exhausted is True when the call
    was skipped because this cycle's/today's credits are spent (the caller
    defers the video instead of reporting a missing transcript); reason is a
    short tag naming what happened, so a run can report *why* fetches failed
    rather than only how often.
    Uses mode=native so only existing captions are returned (no paid AI generation).
    """
    keys = _supadata_keys()
    if not keys:
        return "", False, "no_key"

    usage = _load_usage()
    allowance = daily_allowance(usage)
    if usage.get("day_count", 0) >= allowance:
        remaining = max(0, monthly_budget() - usage.get("count", 0))
        log_warn(
            f"Supadata budget reached for today ({usage['day_count']}/{allowance}; "
            f"{remaining} left this cycle) — deferring {vid} to a later run."
        )
        return "", True, "budget_paced"

    # Any failure rotates to the next key, not just an out-of-credits one: the
    # failure modes are not reliably distinguishable by status code (a spent
    # pool and a rate limit share 429), and a key that cannot deliver is worth
    # no more than a key that has no credits. The one case that does not rotate
    # is a key answering "this video has no captions" — it worked, its answer is
    # authoritative, and asking the other keys would just spend their credits to
    # be told the same thing.
    saw_credit_failure = False
    reason = "no_credits"
    for index, key in enumerate(keys):
        served, text, reason = _fetch_supadata_with_key(vid, key, usage)
        if served:
            return text, False, reason
        saw_credit_failure = saw_credit_failure or reason == "no_credits"
        if index + 1 < len(keys):
            log_warn(f"Supadata key {index + 1} failed ({reason}); trying key {index + 2}.")

    # Out of credits is exhaustion, not a video without captions: report it as
    # such so the caller defers instead of eventually writing the video off as
    # untranscribable.
    if saw_credit_failure:
        log_warn(
            f"All {len(keys)} Supadata key(s) failed and at least one is out of "
            f"credits; deferring this video."
        )
        return "", True, "no_credits"
    log_warn(f"All {len(keys)} Supadata key(s) failed (last: {reason}).")
    return "", False, reason


def _fetch_supadata_with_key(vid, api_key, usage):
    """
    One key's attempt at a transcript. Returns (served, text, reason): `served`
    is True when this key gave an answer worth accepting — a transcript, or an
    authoritative "this video has no captions" — and False for every failure,
    which tells the caller to rotate to the next key. `reason` names the outcome
    for run diagnostics. Only requests that consume a credit are metered.
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
            return False, "", "unreachable"

        # Out of credits on this key: rotate rather than retry. Nothing was
        # delivered, so this must not be metered.
        if _is_credit_exhausted(resp):
            log_warn(f"Supadata key rejected ({resp.status_code}): {resp.text[:120]}")
            return False, "", "no_credits"

        # Meter only answers that consume a credit — a served transcript (200)
        # or an accepted async job (202). Counting rejections and transient
        # retries would defer videos while credits are still available.
        if resp.status_code in (200, 202):
            _record_call(usage)

        # Large videos are processed asynchronously: 202 + a job id to poll.
        if resp.status_code == 202:
            try:
                job_id = resp.json().get("jobId")
            except ValueError:
                job_id = None
            if not job_id:
                log_warn("Supadata returned 202 without a jobId.")
                return False, "", "job_no_id"
            job_text = _poll_supadata_job(job_id, headers)
            if job_text:
                return True, job_text, "ok"
            return False, "", "job_incomplete"

        if resp.status_code == 200:
            try:
                text = _supadata_text_from_payload(resp.json())
            except ValueError as e:
                log_error(f"Supadata returned invalid JSON: {e}")
                return False, "", "invalid_json"
            if not text:
                # 200 with no content is Supadata saying "this video has no
                # captions I can serve" — the single most useful thing to
                # distinguish, since it is a wasted credit by definition.
                log_warn(f"Supadata returned 200 but no transcript content for {vid}.")
                return True, "", "empty_content"
            log_info(f"Supadata returned {len(text)} chars")
            return True, text, "ok"

        # Retry transient server/rate-limit errors; give up on anything else.
        if resp.status_code in SUPADATA_TRANSIENT_STATUS and attempt < SUPADATA_MAX_RETRIES:
            log_warn(
                f"Transient Supadata status {resp.status_code} "
                f"(attempt {attempt}/{SUPADATA_MAX_RETRIES}); retrying."
            )
            time.sleep(SUPADATA_RETRY_BACKOFF * attempt)
            continue

        log_warn(f"Supadata returned {resp.status_code}: {resp.text[:200]}")
        return False, "", f"http_{resp.status_code}"

    return False, "", "retries_exhausted"


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

    Sources are tried cheapest-first: Supadata (one credit, ~9k tokens of
    transcript), then Gemini from the video itself (no credit, but ~123k tokens
    and one request from a per-model daily quota), then youtube-transcript-api
    (free, no key, works locally — but IP-blocked from CI runners).

    `video_id` may be a full URL or a bare ID. Always returns a dict shaped
    {"transcript": <str>, "budget_exhausted": <bool>, "reason": <str>} so
    callers never have to handle exceptions or None; an empty transcript means
    none was available, budget_exhausted marks the "every source we metered is
    spent" case (retry later rather than report as missing), and reason names
    the outcome so a run can report why fetches failed.
    """
    vid = _extract_video_id(video_id)
    if not vid:
        log_warn("No valid video ID; cannot fetch transcript.")
        return {"transcript": "", "budget_exhausted": False, "reason": "bad_video_id"}

    text, budget_exhausted, reason = _fetch_supadata(vid)
    if not text:
        # Not named `gemini_quota`: that is the imported module, and a local
        # of the same name would shadow it for the rest of this function.
        gemini_text, gemini_spent, gemini_reason = _fetch_gemini_transcript(vid)
        if gemini_text:
            text, reason = gemini_text, gemini_reason
        elif gemini_reason != "no_gemini_key":
            # Gemini actually tried, so its outcome is the more informative one.
            # Its quota being spent defers the video just like Supadata's is:
            # the transcript exists, we simply have nothing left to spend today.
            reason = gemini_reason
            budget_exhausted = budget_exhausted or gemini_spent
    if not text:
        fallback = _fetch_youtube_transcript_api(vid)
        if fallback:
            text, reason = fallback, "fallback_ok"

    if text:
        budget_exhausted = False
    elif not budget_exhausted:
        log_warn(f"No transcript available for video {vid} from any source (reason: {reason}).")

    return {"transcript": text, "budget_exhausted": budget_exhausted, "reason": reason}
