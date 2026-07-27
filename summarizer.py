import os
import time

import requests
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

LLM_TIMEOUT = env_int("LLM_TIMEOUT", 60)
LLM_MAX_TOKENS = env_int("LLM_MAX_TOKENS", 2000)
# A response cut off at max_tokens is retried with a bigger budget, doubling up
# to this ceiling. Shipping a half-written summary is worse than spending a
# second request: the reader can't tell it was cut, and the video is marked
# decided so it never comes back.
LLM_MAX_TOKENS_CEILING = env_int("LLM_MAX_TOKENS_CEILING", 8000)
LLM_TEMPERATURE = env_float("LLM_TEMPERATURE", 0.3)
# Cap the transcript length sent to the model, to avoid blowing past context
# windows and to keep token cost predictable. Overridable via env. Generous
# because the models in use have very large context windows and the cut falls
# at the END of the video — exactly where a market video puts its price targets
# and conclusions. 200k chars is roughly a 4-hour video.
LLM_MAX_TRANSCRIPT_CHARS = env_int("LLM_MAX_TRANSCRIPT_CHARS", 200000)
LLM_MAX_RETRIES = 3
LLM_RETRY_BACKOFF = 2  # base seconds, multiplied by the attempt number
# Honor a server's Retry-After on 429, but cap it so a huge value can't stall
# the daily run past its workflow timeout.
LLM_RETRY_AFTER_CAP = 30
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
    "conclusion or call, not the topic. Write \"Speaker is buying Nvidia below "
    "$130, expecting the AI capex cycle to run through 2027\" — not \"The video "
    "discusses Nvidia's outlook\".\n"
    "2. A blank line, then key takeaways as bullets, each starting with \"• \". "
    "Use as many bullets as the content warrants (typically 3-7) — more for "
    "dense, information-rich videos, fewer for simple ones. Keep each bullet to "
    "one or two sentences.\n"
    "\n"
    "Capture these whenever the speaker states them — they are the point:\n"
    "- Each asset discussed by name and ticker, with the speaker's stance "
    "(bullish / bearish / neutral) and how strongly they hold it.\n"
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
    "- Keep the entire summary under roughly 3000 characters so it fits in one "
    "Telegram message alongside the video's title and link.\n"
    "- Finish every sentence. If you are running long, write fewer bullets — "
    "never an unfinished one.\n"
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
    "2. Optionally, up to 3 short bullets starting with \"• \" for genuinely "
    "important specifics. Skip the bullets entirely for thin content.\n"
    "\n"
    "Even when brief, keep the numbers: tickers, price levels, targets, "
    "support/resistance, invalidation levels, and the speaker's stance on each "
    "asset. Cut the narration, not the specifics.\n"
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
    "- Keep the whole entry under roughly 800 characters.\n"
    "- Finish every sentence. If you are running long, write fewer bullets — "
    "never an unfinished one.\n"
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


def _provider_configs():
    """Build the ordered list of configured LLM providers from the environment."""
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
        })

    if os.getenv("GEMINI_API_KEY"):
        gemini_base = "https://generativelanguage.googleapis.com/v1beta/openai"
        # Preferred model first (Gemini 3 Flash: same free RPM as 2.5-flash but
        # ~6x the daily request quota). If its ID is rejected or the model is
        # unavailable, the chain falls through to the proven 2.5-flash entry in
        # the same run — a bad preferred ID costs one failed call, never a
        # missed summary.
        preferred = os.getenv("GEMINI_MODEL") or "gemini-3-flash-preview"
        providers.append({
            "name": "gemini",
            "base_url": gemini_base,
            "api_key": os.getenv("GEMINI_API_KEY"),
            "model": preferred,
        })
        if preferred != "gemini-2.5-flash":
            providers.append({
                "name": "gemini-2.5-flash",
                "base_url": gemini_base,
                "api_key": os.getenv("GEMINI_API_KEY"),
                "model": "gemini-2.5-flash",
            })

    if os.getenv("GROQ_API_KEY"):
        providers.append({
            "name": "groq",
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": os.getenv("GROQ_API_KEY"),
            "model": os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile",
        })

    return providers


def _extract_summary(data):
    """Pull the assistant message text out of an OpenAI-style response."""
    try:
        return (data["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError):
        return ""


def _was_truncated(data):
    """
    True when the model stopped because it hit the token cap rather than
    because it finished. The content that comes back is a real string ending
    mid-sentence, so without this check it reads as a complete summary and is
    delivered as one.
    """
    try:
        reason = data["choices"][0].get("finish_reason") or ""
    except (KeyError, IndexError, TypeError):
        return False
    return reason.lower() in {"length", "max_tokens"}


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
    return transcript[:LLM_MAX_TRANSCRIPT_CHARS] + TRANSCRIPT_TRUNCATION_MARKER


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
    payload = {
        "model": provider["model"],
        "messages": [
            {"role": "system", "content": system_prompt or SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_message if user_message is not None else _build_user_message(transcript, title)},
        ],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": max_tokens or LLM_MAX_TOKENS,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    for attempt in range(1, LLM_MAX_RETRIES + 1):
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
                if attempt < LLM_MAX_RETRIES and cap < LLM_MAX_TOKENS_CEILING:
                    payload["max_tokens"] = min(cap * 2, LLM_MAX_TOKENS_CEILING)
                    log_warn(f"Retrying with max_tokens={payload['max_tokens']}.")
                    continue
                log_error(
                    f"{provider['name']} still truncated at {cap} tokens; discarding "
                    "the partial response rather than delivering it."
                )
                return ""
            summary = _extract_summary(data)
            if not summary:
                log_warn(f"{provider['name']} returned an empty summary.")
            return summary

        # 429 = rate limit / quota. Honor Retry-After while retrying; if it
        # persists, signal quota exhaustion so the caller can defer (and so we
        # stop hammering this provider for the rest of the run).
        if resp.status_code == 429:
            if attempt < LLM_MAX_RETRIES:
                retry_after = _parse_retry_after(resp)
                wait = retry_after if retry_after is not None else LLM_RETRY_BACKOFF * attempt
                log_warn(
                    f"{provider['name']} rate-limited (429); retrying in {wait}s "
                    f"(attempt {attempt}/{LLM_MAX_RETRIES})."
                )
                time.sleep(wait)
                continue
            log_warn(f"{provider['name']} rate-limited (429) after retries; treating as quota exhausted.")
            return QUOTA_EXHAUSTED_SENTINEL

        if resp.status_code in TRANSIENT_STATUS and attempt < LLM_MAX_RETRIES:
            log_warn(
                f"Transient {provider['name']} status {resp.status_code} "
                f"(attempt {attempt}/{LLM_MAX_RETRIES}); retrying."
            )
            time.sleep(LLM_RETRY_BACKOFF * attempt)
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
            "No LLM provider configured. Set GEMINI_API_KEY, GROQ_API_KEY, or "
            "LLM_API_KEY + LLM_BASE_URL to enable completions."
        )
        return ""

    quota_hit = False
    for provider in providers:
        if provider["name"] in _EXHAUSTED_PROVIDERS:
            log_info(f"Skipping {provider['name']} (quota exhausted earlier this run).")
            quota_hit = True
            continue

        text = _call_provider(
            provider, "", system_prompt=system_prompt, user_message=user_message,
            json_mode=json_mode, max_tokens=max_tokens,
        )
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
            "No LLM provider configured. Set GEMINI_API_KEY, GROQ_API_KEY, or "
            "LLM_API_KEY + LLM_BASE_URL to enable summarization."
        )
        return ""

    quota_hit = False
    for provider in providers:
        # Skip providers already known to be quota-exhausted earlier this run.
        if provider["name"] in _EXHAUSTED_PROVIDERS:
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

    log_warn("All configured LLM providers failed to produce a summary.")
    return ""
