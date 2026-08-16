import os
import time

import requests
import gemini_quota
from helpers import env_int, env_float
from log import log_info, log_warn, log_error

# Provider-agnostic summarization.
#
# This module speaks the OpenAI-compatible Chat Completions API, which almost
# every LLM provider exposes (Gemini, Groq, OpenRouter, OpenAI, Together, and
# local servers like Ollama / llama.cpp). Providers are configured purely through
# environment variables and tried in order until one returns a summary, so you
# can swap or chain providers without touching code.
#
# Recognised env vars (each provider is skipped when its API key is unset):
#   Generic (highest priority): LLM_API_KEY + LLM_BASE_URL [+ LLM_MODEL, LLM_NAME]
#   Gemini (free tier):         GEMINI_API_KEY            [+ GEMINI_MODEL]
#   Groq   (free tier):         GROQ_API_KEY              [+ GROQ_MODEL]

# Generous because the prompt can now carry a very long transcript and the
# response budget can escalate; 60s was sized for a 48k-char input.
LLM_TIMEOUT = env_int("LLM_TIMEOUT", 120)
# 4000, not 2000: Gemini 3 counts thinking tokens against this cap, and an
# escalation re-sends the entire transcript as a second request — which now
# costs one of a model's 20 free requests for the day. Paying a few hundred
# tokens up front is cheaper than paying a whole request to retry.
LLM_MAX_TOKENS = env_int("LLM_MAX_TOKENS", 4000)
# A response cut off at max_tokens is retried with a bigger budget, doubling up
# to this ceiling. Shipping a half-written summary is worse than spending a
# second request: the reader can't tell it was cut, and the video is marked
# decided so it never comes back.
LLM_MAX_TOKENS_CEILING = env_int("LLM_MAX_TOKENS_CEILING", 8000)
# How many times one call may double its budget. Separate from LLM_MAX_RETRIES
# so a transient 5xx can't consume the escalation allowance.
LLM_MAX_ESCALATIONS = env_int("LLM_MAX_ESCALATIONS", 2)
# Gemini 3 is a thinking model and counts thinking tokens against max_tokens
# (unlike OpenAI's reasoning models, which meter them separately). A budget
# mostly consumed by thinking leaves a few hundred tokens of visible text and
# finish_reason=length — exactly the shape of the truncated summaries we
# shipped: 241, 332 and 860 chars against a 3000-token budget. Capping the
# thinking budget addresses the cause; escalating max_tokens only the symptom.
# Empty disables the parameter.
LLM_REASONING_EFFORT = (os.getenv("LLM_REASONING_EFFORT") or "low").strip()
LLM_TEMPERATURE = env_float("LLM_TEMPERATURE", 0.3)
# ~4 chars/token, so 120k chars is ~30k tokens — over 6x the longest video
# these channels publish, i.e. in practice nothing gets cut. The context window
# is not the binding constraint: the free tier meters tokens PER MINUTE, and
# because an escalation re-sends the whole input, a near-window request would
# blow the minute's allowance and be rejected rather than buy a better summary.
LLM_MAX_TRANSCRIPT_CHARS = env_int("LLM_MAX_TRANSCRIPT_CHARS", 120000)
# Per-provider input ceilings. Sending more than a provider accepts fails
# outright instead of degrading, and on free tiers the per-minute token budget
# bites long before the context window does — so each provider declares what it
# can actually take and the prompt is trimmed to fit at call time.
GEMINI_MAX_INPUT_CHARS = env_int("GEMINI_MAX_INPUT_CHARS", 120000)  # 1M window, TPM-bound
# Summaries are the product, so they get the strongest Flash model; the
# fallbacks exist because each model carries its own 20-requests-per-day free
# tier, so a busy day can draw on four budgets instead of one. Transcription
# runs on GEMINI_TRANSCRIPT_MODELS (transcript.py) and is deliberately kept off
# this list: a 123k-token video call must never spend the summary budget.
GEMINI_DEFAULT_MODEL = "gemini-3.7-flash"
# Only 3.6 backs up 3.7. The summary *is* the product, and the older Flash
# generations are a visible drop in quality, so the chain buys a second daily
# quota of comparable output rather than a longer tail of weaker ones. When both
# are spent the video defers and goes out on the next run, which is the right
# trade: late and good beats prompt and worse.
GEMINI_DEFAULT_FALLBACKS = "gemini-3.6-flash"
# Models permitted to write a summary, whatever else is configured. Model
# choice is visible in the output, so this is a hard guarantee rather than a
# default: a stale GEMINI_MODEL repo variable, a GROQ_API_KEY, or a custom
# LLM_* endpoint would otherwise quietly hand summaries to a weaker model. A
# request that no allowed model can serve defers to the next run instead.
# Same source of truth as the chain above, so the two can't drift. Set
# SUMMARY_MODELS to a comma-separated roster to change it, or to "*" to allow
# any configured provider.
SUMMARY_MODELS_DEFAULT = f"{GEMINI_DEFAULT_MODEL},{GEMINI_DEFAULT_FALLBACKS}"
# Groq reserves max_tokens against TPM at admission, so input AND requested
# output must both fit the per-minute budget or the call is rejected before
# inference. ~3k input tokens leaves room for the reserved output.
GROQ_MAX_INPUT_CHARS = env_int("GROQ_MAX_INPUT_CHARS", 12000)
LLM_MAX_RETRIES = 3
# Rate limits get their own budget rather than sharing the failure retries
# above. They shared it once, and an unrelated pair of 503s could then leave a
# 429 with no attempts left — which retired a model for the whole Pacific day
# on its first rate limit, with most of its daily requests still unspent.
LLM_MAX_RATE_LIMIT_RETRIES = env_int("LLM_MAX_RATE_LIMIT_RETRIES", 3)
LLM_RETRY_BACKOFF = 2  # base seconds, multiplied by the attempt number
# Honor a server's Retry-After on 429, but cap it so a huge value can't stall
# the daily run past its workflow timeout.
# Must be able to outlast a per-minute token window, or a TPM 429 is retried
# while still inside the same window and the provider gets written off.
LLM_RETRY_AFTER_CAP = env_int("LLM_RETRY_AFTER_CAP", 75)
TRANSIENT_STATUS = {500, 502, 503, 504}  # 429 is handled separately (quota/rate limit)

# Providers found to be quota/rate-limited during this run. Once a provider 429s
# persistently, later channels skip it instead of re-hitting the dead endpoint.
_EXHAUSTED_PROVIDERS = set()

# Marker appended when a transcript is truncated, so the model (and a curious
# human reading the raw input) knows the text was cut short.
TRANSCRIPT_TRUNCATION_MARKER = "\n\n...[transcript truncated]"

# Sentinel the model is told to emit when the transcript is genuinely
# unsummarizable. We detect it in code and treat it as "no summary" (so the
# caller sends its dedicated no-output notice) rather than forwarding a refusal
# sentence to Telegram as if it were a real summary.
INSUFFICIENT_TRANSCRIPT_SENTINEL = "INSUFFICIENT_TRANSCRIPT"

# Sentinel returned when every available provider is quota/rate-limited. The
# caller treats this as a retryable "deferred" outcome (don't mark the video as
# seen) rather than a permanent failure, so it's retried on the next run.
QUOTA_EXHAUSTED_SENTINEL = "QUOTA_EXHAUSTED"

# Sentinel returned when a response was cut off at the token cap and could not
# be recovered by escalating. Retryable, not permanent: whether a response fits
# varies with the video, so the caller defers instead of writing the video off
# (and never delivers the half-written text).
TRUNCATED_SENTINEL = "SUMMARY_TRUNCATED"

# The summary is read by someone who has NOT watched the video and is delivered
# as a plain-text Telegram message (the sender HTML-escapes it, so markdown like
# **bold** or "#" headers will not render). Hence: plain text, "• " bullets only.
SUMMARY_SYSTEM_PROMPT = (
    "You are an expert analyst summarizing market and investing videos for a "
    "reader who has NOT watched them and may act on what you write. Specificity "
    "is the entire value: a summary that omits the numbers is worse than none.\n"
    "\n"
    "Output format (plain text only — no markdown, no headers, no bold):\n"
    "1. First line: a single-sentence TL;DR giving the speaker's actual "
    "conclusion or call, not the topic. Shape it like \"Speaker is buying "
    "<company> below <the level they name>, expecting <their stated thesis> "
    "through <their timeframe>\" — not \"The video discusses <company>'s "
    "outlook\". Fill the placeholders only from the transcript.\n"
    "2. A blank line, then 3-5 bullets starting with \"• \" covering the "
    "speaker's reasoning, thesis and market view. One or two sentences each.\n"
    "3. A blank line, then ONE LINE PER ASSET the speaker discussed:\n"
    "     TICKER (Name) — stance, conviction | levels/targets | timeframe\n"
    "   Write a line for every asset, including ones mentioned only in "
    "passing, and omit any field the speaker didn't give. This roster is where "
    "completeness lives — a bullet is not the only place an asset can appear, "
    "so running short on bullets must never cost you an asset.\n"
    "\n"
    "Roster rules:\n"
    "- First column: the ticker when the speaker says one, otherwise the "
    "company or asset name in normal case. Never uppercase a name into a "
    "ticker-looking string for something that has no ticker — private "
    "companies get their name, not a fake symbol.\n"
    "- The level / target / invalidation fields are for NUMBERS the speaker "
    "gave: prices, percentages, valuations. If they gave none, leave the field "
    "out entirely. Never paraphrase a view into a numeric field — "
    "\"target: outperform\" and \"level: a 2-year cycle\" are wrong; omitting "
    "them is right.\n"
    "- Don't repeat a label the format already supplies (write \"bearish\", "
    "not \"stance: sell\").\n"
    "\n"
    "Capture these whenever the speaker states them — they are the point:\n"
    "- EVERY asset the speaker discusses, with their stance (bullish / bearish "
    "/ neutral) and how strongly they hold it. Do not drop an asset for "
    "brevity: if the speaker covers eight assets, all eight must appear. Give "
    "the ticker only when the speaker says it or it is on screen in the title; "
    "otherwise use the company name alone. Never supply a ticker you happen to "
    "know but did not hear.\n"
    "- Concrete numbers: price levels, targets, support and resistance, stop or "
    "invalidation levels, valuations, growth and margin figures.\n"
    "- The timeframe over which the speaker expects it to play out.\n"
    "- The reasoning behind the call, and any condition that would invalidate it.\n"
    "- Positions the speaker discloses or changes (bought, sold, trimmed, added).\n"
    "\n"
    "Content rules:\n"
    "- Always write in English, even if the transcript is in another language.\n"
    "- Be strictly faithful to the transcript. Never invent or guess facts, "
    "names, numbers, dates, or conclusions that are not present. If the speaker "
    "gives no numbers, give none — do not fill the gap with plausible ones.\n"
    "- Attribute views to the speaker rather than stating them as fact.\n"
    "- Leave out sponsor reads, subscription pitches, and channel housekeeping.\n"
    "- Auto-generated captions are often messy, informal, or missing "
    "punctuation; that is normal — do your best to summarize them anyway.\n"
    "- Only if the transcript is so garbled, fragmentary, or empty that NO "
    "meaningful summary is possible, output exactly the single token "
    "INSUFFICIENT_TRANSCRIPT and nothing else.\n"
    "- Aim for under 3000 characters so it fits in one Telegram message "
    "alongside the video's title and link. If the asset roster alone needs more "
    "room, keep the roster and cut the bullets to two.\n"
    "- If you are running long, shorten the bullets, then drop bullets "
    "entirely. Never drop an asset line, and never leave a sentence "
    "unfinished — the roster is the last thing to go, not the first.\n"
    "\n"
    "Output only the summary itself — no preamble, no sign-off, and no phrases "
    "like \"Here is the summary\"."
)

# Compact variant used for digest-mode channels: prolific channels get one
# bundled message per run, so each entry must be skimmable — a short TL;DR and
# at most a few bullets — instead of a full summary.
COMPACT_SUMMARY_SYSTEM_PROMPT = (
    "You are an expert summarizer of YouTube video transcripts, producing "
    "COMPACT digest entries. The reader has NOT watched the video and skims "
    "several of these entries in one message, so be brief.\n"
    "\n"
    "Output format (plain text only — no markdown, no headers, no bold):\n"
    "1. First line: a one-to-two-sentence TL;DR giving the speaker's actual "
    "call or conclusion, not the topic.\n"
    "2. A blank line, then 1-2 short bullets starting with \"• \" giving the "
    "speaker's reasoning — why they hold this view. A roster of tickers "
    "without the thinking behind them is not worth reading.\n"
    "3. A blank line, then one line per asset the speaker covered:\n"
    "     TICKER — stance | level/target | invalidation\n"
    "\n"
    "Cut the narration, never the specifics — an asset the speaker covered "
    "must not be missing from the entry.\n"
    "\n"
    "Roster rules:\n"
    "- First column: the ticker when the speaker says one, otherwise the "
    "company or asset name in normal case. Never uppercase a name into a "
    "ticker-looking string for something that has no ticker.\n"
    "- The level / target / invalidation fields are for NUMBERS the speaker "
    "gave. If they gave none, leave the field out entirely — never paraphrase "
    "a view into a numeric field.\n"
    "- Don't repeat a label the format already supplies (write \"bearish\", "
    "not \"stance: sell\").\n"
    "\n"
    "Content rules:\n"
    "- Always write in English, even if the transcript is in another language.\n"
    "- Be strictly faithful to the transcript. Never invent or guess facts, "
    "names, numbers, dates, or conclusions that are not present.\n"
    "- Attribute views to the speaker rather than stating them as fact.\n"
    "- Leave out sponsor reads, subscription pitches, and channel housekeeping.\n"
    "- Auto-generated captions are often messy, informal, or missing "
    "punctuation; that is normal — do your best to summarize them anyway.\n"
    "- Only if the transcript is so garbled, fragmentary, or empty that NO "
    "meaningful summary is possible, output exactly the single token "
    "INSUFFICIENT_TRANSCRIPT and nothing else.\n"
    "- Budget roughly 300 characters for the TL;DR, 200 for the bullets, and "
    "70 per asset line, up to 1500 total. Never omit an asset to stay short — "
    "shorten its line instead.\n"
    "- Finish every sentence; never leave one unfinished.\n"
    "\n"
    "Output only the summary itself — no preamble, no sign-off, and no phrases "
    "like \"Here is the summary\"."
)

# User-message template. A title (when known) grounds the model on the video's
# topic; the transcript follows. {title_line} is either an empty string or a
# "Video title: ...\n\n" line.
SUMMARY_USER_TEMPLATE = (
    "{title_line}Summarize the following transcript:\n\n{transcript}"
)


def _gemini_models():
    """
    Gemini models to try, in order: the preferred one then the fallbacks, with
    duplicates dropped so pinning GEMINI_MODEL to a fallback doesn't call it
    twice. Read at call time so tests and reloads see the current environment;
    `or` (not a getenv default) because an unconfigured GitHub Actions variable
    arrives as "" and must fall back to the default rather than becoming a
    literal empty model name.
    """
    preferred = os.getenv("GEMINI_MODEL") or GEMINI_DEFAULT_MODEL
    raw = os.getenv("GEMINI_FALLBACK_MODELS") or GEMINI_DEFAULT_FALLBACKS
    models, seen = [], set()
    for model in [preferred] + [m.strip() for m in raw.split(",")]:
        if model and model not in seen:
            seen.add(model)
            models.append(model)
    return models


def _summary_allowlist():
    """
    Models allowed to produce a summary, or None when unrestricted.

    An unset GitHub Actions variable arrives as "", which must mean "use the
    default roster" — reading it as "no restriction" would silently undo the
    guarantee exactly when nobody configured anything. Disabling is therefore
    explicit: SUMMARY_MODELS="*".
    """
    raw = (os.getenv("SUMMARY_MODELS") or SUMMARY_MODELS_DEFAULT).strip()
    if raw == "*":
        return None
    return {model.strip() for model in raw.split(",") if model.strip()}


def _provider_configs():
    """
    Build the ordered list of configured LLM providers from the environment,
    keeping only models allowed to write a summary (see _summary_allowlist).
    """
    providers = []

    # `or` (not a getenv default): an unconfigured GitHub Actions repo variable
    # arrives as "" — which must fall back to the code default, not become the
    # literal model name.
    if os.getenv("LLM_API_KEY") and os.getenv("LLM_BASE_URL"):
        providers.append({
            "name": os.getenv("LLM_NAME") or "custom",
            "base_url": os.getenv("LLM_BASE_URL").rstrip("/"),
            "api_key": os.getenv("LLM_API_KEY"),
            "model": os.getenv("LLM_MODEL") or "gpt-4o-mini",
            "max_input_chars": env_int("LLM_MAX_INPUT_CHARS", LLM_MAX_TRANSCRIPT_CHARS),
        })

    if os.getenv("GEMINI_API_KEY"):
        gemini_base = "https://generativelanguage.googleapis.com/v1beta/openai"
        # Free-tier quota is per model — 20 requests per day each — so the
        # fallbacks are not just insurance against a bad model ID, they are
        # extra daily capacity. Measured 2026-08-13: the pipeline had already
        # peaked at 26/20 requests on its preferred model while three other
        # Flash models sat nearly unused, i.e. summaries were being lost to
        # quota with capacity to spare. Each entry is named for its model so
        # exhaustion is tracked per model rather than per provider.
        for model in _gemini_models():
            providers.append({
                "name": model,
                "base_url": gemini_base,
                "api_key": os.getenv("GEMINI_API_KEY"),
                "model": model,
                "max_input_chars": GEMINI_MAX_INPUT_CHARS,
            })

    if os.getenv("GROQ_API_KEY"):
        providers.append({
            "name": "groq",
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": os.getenv("GROQ_API_KEY"),
            "model": os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile",
            "max_input_chars": GROQ_MAX_INPUT_CHARS,
        })

    allowed = _summary_allowlist()
    if allowed is None:
        return providers
    kept = [p for p in providers if p["model"] in allowed]
    blocked = [p["model"] for p in providers if p["model"] not in allowed]
    if blocked:
        # Loud, because this is configuration being overruled: someone set a
        # provider up and it is not being used.
        log_warn(
            f"Ignoring provider model(s) not allowed to summarize: {', '.join(blocked)}. "
            f"Allowed: {', '.join(sorted(allowed))} (set SUMMARY_MODELS to change)."
        )
    return kept


def _metered_model(provider):
    """
    The Gemini model whose daily quota this provider spends, or None when the
    provider isn't Gemini. Groq and custom endpoints have their own limits and
    must not be counted against — or blocked by — the Gemini budget.
    """
    if "generativelanguage.googleapis.com" in (provider.get("base_url") or ""):
        return provider.get("model")
    return None


def _provider_is_spent(provider):
    """True when this provider can't serve another request right now: either it
    failed with a quota error earlier in this run, or its model's daily free
    requests are already used up (which outlives the run)."""
    if provider["name"] in _EXHAUSTED_PROVIDERS:
        return True
    model = _metered_model(provider)
    return bool(model) and gemini_quota.is_exhausted(model)


def _extract_summary(data):
    """Pull the assistant message text out of an OpenAI-style response."""
    try:
        return (data["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError):
        return ""


def _fit_to_provider(message, provider):
    """
    Trim a prompt to what this provider can actually accept. The global
    transcript cap is sized for the widest context in the chain, so a fallback
    with a smaller window would otherwise get a prompt it must reject outright.
    """
    limit = provider.get("max_input_chars") or 0
    if limit <= 0 or len(message) <= limit:
        return message
    log_warn(
        f"Prompt is {len(message)} chars, over {provider['name']}'s "
        f"{limit}-char input budget; trimming to fit."
    )
    return _head_and_tail(message, limit)


def _was_truncated(data):
    """
    True when the model stopped because it hit the token cap rather than
    because it finished. The content that comes back is a real string ending
    mid-sentence, so without this check it reads as a complete summary and is
    delivered as one.
    """
    try:
        reason = data["choices"][0].get("finish_reason") or ""
    except (AttributeError, KeyError, IndexError, TypeError):
        return False
    return str(reason).lower() in {"length", "max_tokens"}


def _truncate_transcript(transcript):
    """
    Cap the transcript at LLM_MAX_TRANSCRIPT_CHARS, appending a clear marker and
    logging a warning when content is dropped. Returns the (possibly shorter)
    transcript.
    """
    if len(transcript) <= LLM_MAX_TRANSCRIPT_CHARS:
        return transcript

    log_warn(
        f"Transcript is {len(transcript)} chars, exceeding the "
        f"{LLM_MAX_TRANSCRIPT_CHARS}-char cap; truncating before summarization."
    )
    return _head_and_tail(transcript, LLM_MAX_TRANSCRIPT_CHARS)


def _head_and_tail(text, limit):
    """
    Cut the MIDDLE out of an over-long text, keeping both ends.

    Keeping only the head is wrong for this content: a market video opens with
    the setup and closes with the price targets, invalidation levels and "what
    I am doing" — so a head-only cut discards exactly what the summary exists
    to capture. Roughly 60% head / 40% tail keeps thesis and conclusion both.
    """
    marker = TRANSCRIPT_TRUNCATION_MARKER
    budget = max(0, limit - len(marker))
    head = int(budget * 0.6)
    tail = budget - head
    if tail <= 0:
        return text[:budget] + marker
    return text[:head] + marker + text[-tail:]


def _build_user_message(transcript, title=None):
    """Render the user-message template, grounding on the video title if given."""
    title = (title or "").strip()
    title_line = f"Video title: {title}\n\n" if title else ""
    return SUMMARY_USER_TEMPLATE.format(title_line=title_line, transcript=transcript)


def _parse_retry_after(resp):
    """
    Return the Retry-After header value in seconds (capped), or None if absent
    or unparseable. Only the delta-seconds form is honored; HTTP-date values
    fall back to None (and thus to our normal backoff).
    """
    value = resp.headers.get("Retry-After") if hasattr(resp, "headers") else None
    if not value:
        return None
    try:
        secs = int(float(value))
    except (TypeError, ValueError):
        return None
    return max(0, min(secs, LLM_RETRY_AFTER_CAP))


def _call_provider(provider, transcript, title=None, system_prompt=None, user_message=None,
                   json_mode=False, max_tokens=None):
    """
    Call one provider's chat-completions endpoint with retry on transient errors.
    Returns the summary text, "" on failure, or QUOTA_EXHAUSTED_SENTINEL when the
    provider is persistently rate-limited / out of quota (HTTP 429).

    `user_message`, when given, is sent verbatim instead of the transcript
    template — used by complete() for non-summarization calls. `json_mode`
    requests forced-JSON output (response_format json_object — supported by
    Gemini's OpenAI-compat endpoint and Groq), so structured-output callers
    don't depend on the model resisting the urge to add prose or fences.
    """
    url = f"{provider['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {provider['api_key']}",
        "Content-Type": "application/json",
    }
    content = user_message if user_message is not None else _build_user_message(transcript, title)
    payload = {
        "model": provider["model"],
        "messages": [
            {"role": "system", "content": system_prompt or SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": _fit_to_provider(content, provider)},
        ],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": max_tokens or LLM_MAX_TOKENS,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if LLM_REASONING_EFFORT:
        payload["reasoning_effort"] = LLM_REASONING_EFFORT

    # Budget escalation is not a failure retry: a response cut off at the cap
    # means the request worked and the room was too small. Counting it against
    # LLM_MAX_RETRIES let one unrelated 5xx eat the escalation budget, so the
    # ceiling was never actually reached. The two are tracked separately.
    # The ceiling must always leave room to escalate from wherever this call
    # starts. Clamping it to the configured budget would mean that raising the
    # budget past LLM_MAX_TOKENS_CEILING silently disables escalation and turns
    # every truncation into a discard — worse behavior for a bigger setting.
    start_cap = payload["max_tokens"]
    ceiling = max(LLM_MAX_TOKENS_CEILING, start_cap * (2 ** max(1, LLM_MAX_ESCALATIONS)))
    attempt, escalations, rate_limits = 0, 0, 0
    while attempt < LLM_MAX_RETRIES:
        attempt += 1
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=LLM_TIMEOUT)
        except requests.RequestException as e:
            log_warn(f"{provider['name']} request error (attempt {attempt}/{LLM_MAX_RETRIES}): {e}")
            if attempt < LLM_MAX_RETRIES:
                time.sleep(LLM_RETRY_BACKOFF * attempt)
                continue
            log_error(f"{provider['name']} unreachable after retries.")
            return ""

        if resp.status_code == 200:
            # One request served, so one request spent from this model's daily
            # free-tier budget. Counted here rather than at the call site so an
            # escalation retry — which is a second request — is counted too.
            metered = _metered_model(provider)
            if metered:
                gemini_quota.record(metered)
            try:
                data = resp.json()
            except ValueError as e:
                log_error(f"{provider['name']} returned invalid JSON: {e}")
                return ""
            if _was_truncated(data):
                # Cut off at the token cap: the text ends mid-sentence and, in
                # JSON mode, isn't even parseable. Retry with room rather than
                # deliver half a summary.
                cap = payload["max_tokens"]
                log_warn(f"{provider['name']} response hit the {cap}-token cap and was truncated.")
                if escalations < LLM_MAX_ESCALATIONS and cap < ceiling:
                    payload["max_tokens"] = min(cap * 2, ceiling)
                    escalations += 1
                    attempt -= 1  # an escalation is not one of the failure retries
                    log_warn(f"Retrying with max_tokens={payload['max_tokens']}.")
                    continue
                log_error(
                    f"{provider['name']} still truncated at {cap} tokens; discarding "
                    "the partial response rather than delivering it."
                )
                # Not "": an empty summary is a permanent failure that advances
                # the watermark, and truncation varies run to run. Say what
                # happened so the caller can retry the video instead.
                return TRUNCATED_SENTINEL
            summary = _extract_summary(data)
            if not summary:
                log_warn(f"{provider['name']} returned an empty summary.")
            return summary

        # 429 = rate limit / quota — two different things behind one status
        # code. A per-day quota is gone until the Pacific reset; a per-minute
        # one clears in about a minute. Only the body tells them apart, so it
        # is logged (previously it was discarded, leaving the logs unable to
        # say which limit had been hit).
        if resp.status_code == 429:
            body = resp.text or ""
            kind = gemini_quota.classify_429(body)
            metered = _metered_model(provider)
            log_warn(
                f"{provider['name']} rate-limited (429, {kind} quota): {body[:200]}"
            )
            if kind == "day":
                # No amount of waiting brings the day's budget back.
                if metered:
                    gemini_quota.mark_exhausted(metered)
                return QUOTA_EXHAUSTED_SENTINEL

            rate_limits += 1
            if rate_limits < LLM_MAX_RATE_LIMIT_RETRIES:
                wait = gemini_quota.retry_delay_seconds(body)
                if wait is None:
                    wait = _parse_retry_after(resp)
                if wait is None:
                    wait = LLM_RETRY_BACKOFF * rate_limits
                wait = min(wait, LLM_RETRY_AFTER_CAP)
                log_warn(
                    f"Retrying in {wait}s (rate-limit attempt "
                    f"{rate_limits}/{LLM_MAX_RATE_LIMIT_RETRIES})."
                )
                time.sleep(wait)
                # Waiting out a rate limit is not a failed attempt: it must not
                # consume the budget reserved for genuine errors.
                attempt -= 1
                continue

            # Still limited, and the API never said the day is gone. Skipping
            # this provider for the rest of the run is enough — writing the day
            # off would cost every later run its preferred model on the word of
            # a limit that may well clear in a minute. The worst case is one
            # wasted request next run; the old behavior cost seven.
            log_warn(
                f"{provider['name']} still rate-limited after "
                f"{rate_limits} attempt(s); skipping it for this run only."
            )
            return QUOTA_EXHAUSTED_SENTINEL

        if resp.status_code in TRANSIENT_STATUS and attempt < LLM_MAX_RETRIES:
            log_warn(
                f"Transient {provider['name']} status {resp.status_code} "
                f"(attempt {attempt}/{LLM_MAX_RETRIES}); retrying."
            )
            time.sleep(LLM_RETRY_BACKOFF * attempt)
            continue

        # Some providers reject reasoning_effort outright. Drop it and retry
        # rather than lose the summary over an optional parameter.
        if (resp.status_code == 400 and "reasoning_effort" in payload
                and "reasoning_effort" in (resp.text or "").lower()):
            log_warn(f"{provider['name']} rejected reasoning_effort; retrying without it.")
            payload.pop("reasoning_effort")
            attempt -= 1
            continue

        log_warn(f"{provider['name']} returned {resp.status_code}: {resp.text[:200]}")
        return ""

    return ""


def complete(system_prompt, user_message, json_mode=False, max_tokens=None):
    """
    Generic completion over the same provider chain as summarize_transcript:
    first configured provider that succeeds wins, quota-exhausted providers are
    skipped for the rest of the run. Returns the response text, "" when no
    provider is configured or all fail, or QUOTA_EXHAUSTED_SENTINEL when every
    available provider is rate-limited. Never raises. `json_mode` forces JSON
    output at the API level for structured-output callers.
    """
    if not (user_message or "").strip():
        log_warn("Empty prompt for completion; nothing to do.")
        return ""

    providers = _provider_configs()
    if not providers:
        log_warn(
            "No usable LLM provider. Set GEMINI_API_KEY, GROQ_API_KEY, or "
            "LLM_API_KEY + LLM_BASE_URL — and check SUMMARY_MODELS if a "
            "provider is configured but disallowed."
        )
        return ""

    quota_hit = False
    for provider in providers:
        if _provider_is_spent(provider):
            log_info(f"Skipping {provider['name']} (quota exhausted earlier this run).")
            quota_hit = True
            continue

        text = _call_provider(
            provider, "", system_prompt=system_prompt, user_message=user_message,
            json_mode=json_mode, max_tokens=max_tokens,
        )
        if text == TRUNCATED_SENTINEL:
            log_warn(f"{provider['name']} response was truncated; trying next provider.")
            continue
        if text == QUOTA_EXHAUSTED_SENTINEL:
            log_warn(f"{provider['name']} quota/rate limit hit; skipping it for the rest of the run.")
            _EXHAUSTED_PROVIDERS.add(provider["name"])
            quota_hit = True
            continue
        if text:
            return text
        log_warn(f"{provider['name']} did not produce a completion; trying next provider.")

    if quota_hit:
        return QUOTA_EXHAUSTED_SENTINEL
    return ""


def summarize_transcript(transcript, title=None, compact=False):
    """
    Summarize a transcript using the first configured LLM provider that succeeds.

    Args:
        transcript: The transcript text to summarize.
        title: Optional video title used to ground the model on the topic.
        compact: Produce a short digest entry (TL;DR + up to 3 bullets) instead
            of a full summary — used for digest-mode channels.

    Returns the summary text, or "" if the transcript is empty, no provider is
    configured, or every provider fails. Returns INSUFFICIENT_TRANSCRIPT_SENTINEL
    if a provider judged the transcript too garbled/empty to summarize. The
    caller treats both as "no summary" and notifies accordingly; never raises.
    """
    transcript = (transcript or "").strip()
    if not transcript:
        log_warn("Transcript is empty. Nothing to summarize.")
        return ""

    transcript = _truncate_transcript(transcript)

    providers = _provider_configs()
    if not providers:
        log_warn(
            "No usable LLM provider. Set GEMINI_API_KEY, GROQ_API_KEY, or "
            "LLM_API_KEY + LLM_BASE_URL — and check SUMMARY_MODELS if a "
            "provider is configured but disallowed."
        )
        # Deferred, not empty: "" is a permanent failure that advances the
        # watermark and loses the video. Misconfiguration must cost a delay,
        # never a summary — and the run's stalled-delivery alert then fires.
        return QUOTA_EXHAUSTED_SENTINEL

    quota_hit = False
    truncated_hit = False
    for provider in providers:
        # Skip providers already known to be quota-exhausted earlier this run.
        if _provider_is_spent(provider):
            log_info(f"Skipping {provider['name']} (quota exhausted earlier this run).")
            quota_hit = True
            continue

        log_info(f"Summarizing via {provider['name']} ({provider['model']})...")
        system_prompt = COMPACT_SUMMARY_SYSTEM_PROMPT if compact else SUMMARY_SYSTEM_PROMPT
        summary = _call_provider(provider, transcript, title, system_prompt=system_prompt)

        if summary == QUOTA_EXHAUSTED_SENTINEL:
            log_warn(f"{provider['name']} quota/rate limit hit; skipping it for the rest of the run.")
            _EXHAUSTED_PROVIDERS.add(provider["name"])
            quota_hit = True
            continue

        if summary == TRUNCATED_SENTINEL:
            # This provider couldn't fit the answer even after escalating.
            # Another may have a different budget, so try it before giving up.
            log_warn(f"{provider['name']} could not produce a complete summary; trying the next provider.")
            truncated_hit = True
            continue

        if summary:
            # The model signalled the transcript was unsummarizable. Return the
            # sentinel (not the bare refusal text) so the caller can send an
            # accurate "transcript too garbled" notice instead of forwarding the
            # token to Telegram. The verdict is about the transcript, not the
            # provider, so don't retry the others.
            if summary.strip().upper().startswith(INSUFFICIENT_TRANSCRIPT_SENTINEL):
                log_warn(
                    f"{provider['name']} judged the transcript insufficient to "
                    "summarize."
                )
                return INSUFFICIENT_TRANSCRIPT_SENTINEL
            log_info(f"Summary generated via {provider['name']}.")
            return summary
        log_warn(f"{provider['name']} did not produce a summary; trying next provider.")

    # No provider produced a summary. Distinguish "everything is rate-limited"
    # (retryable — defer) from genuine failure, so the caller can notify accurately.
    if quota_hit:
        log_warn("All available LLM providers are quota/rate-limited; deferring summary.")
        return QUOTA_EXHAUSTED_SENTINEL

    if truncated_hit:
        # Every provider ran out of room. Retryable, not permanent — returning
        # "" here would mark the video decided and lose it for good.
        log_warn("No provider produced a complete summary; deferring for retry.")
        return TRUNCATED_SENTINEL

    log_warn("All configured LLM providers failed to produce a summary.")
    return ""
