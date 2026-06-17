import os
import time

import requests
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

LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1500"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
# Cap the transcript length sent to the model, to avoid blowing past context
# windows and to keep token cost predictable. Overridable via env.
LLM_MAX_TRANSCRIPT_CHARS = int(os.getenv("LLM_MAX_TRANSCRIPT_CHARS", "48000"))
LLM_MAX_RETRIES = 3
LLM_RETRY_BACKOFF = 2  # base seconds, multiplied by the attempt number
TRANSIENT_STATUS = {429, 500, 502, 503, 504}

# Marker appended when a transcript is truncated, so the model (and a curious
# human reading the raw input) knows the text was cut short.
TRANSCRIPT_TRUNCATION_MARKER = "\n\n...[transcript truncated]"

# Sentinel the model is told to emit when the transcript is genuinely
# unsummarizable. We detect it in code and treat it as "no summary" (so the
# caller sends its dedicated no-output notice) rather than forwarding a refusal
# sentence to Telegram as if it were a real summary.
INSUFFICIENT_TRANSCRIPT_SENTINEL = "INSUFFICIENT_TRANSCRIPT"

# The summary is read by someone who has NOT watched the video and is delivered
# as a plain-text Telegram message (the sender HTML-escapes it, so markdown like
# **bold** or "#" headers will not render). Hence: plain text, "• " bullets only.
SUMMARY_SYSTEM_PROMPT = (
    "You are an expert summarizer of YouTube video transcripts. Your summary is "
    "read by someone who has NOT watched the video, so it must stand on its own.\n"
    "\n"
    "Output format (plain text only — no markdown, no headers, no bold):\n"
    "1. First line: a single-sentence TL;DR that captures what the video is "
    "about and its main point.\n"
    "2. A blank line, then key takeaways as bullets, each starting with \"• \". "
    "Use as many bullets as the content warrants (typically 3-7) — more for "
    "dense, information-rich videos, fewer for simple ones. Keep each bullet to "
    "one or two sentences.\n"
    "\n"
    "Content rules:\n"
    "- Always write in English, even if the transcript is in another language.\n"
    "- Be strictly faithful to the transcript. Never invent or guess facts, "
    "names, numbers, dates, or conclusions that are not present.\n"
    "- Prefer concrete specifics — key arguments, conclusions, steps, named "
    "people/products/places, and notable data — over vague generalities.\n"
    "- Auto-generated captions are often messy, informal, or missing "
    "punctuation; that is normal — do your best to summarize them anyway.\n"
    "- Only if the transcript is so garbled, fragmentary, or empty that NO "
    "meaningful summary is possible, output exactly the single token "
    "INSUFFICIENT_TRANSCRIPT and nothing else.\n"
    "- Keep the entire summary under roughly 3500 characters so it fits in one "
    "Telegram message alongside the video's title and link.\n"
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

    if os.getenv("LLM_API_KEY") and os.getenv("LLM_BASE_URL"):
        providers.append({
            "name": os.getenv("LLM_NAME", "custom"),
            "base_url": os.getenv("LLM_BASE_URL").rstrip("/"),
            "api_key": os.getenv("LLM_API_KEY"),
            "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
        })

    if os.getenv("GEMINI_API_KEY"):
        providers.append({
            "name": "gemini",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "api_key": os.getenv("GEMINI_API_KEY"),
            "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        })

    if os.getenv("GROQ_API_KEY"):
        providers.append({
            "name": "groq",
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": os.getenv("GROQ_API_KEY"),
            "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        })

    return providers


def _extract_summary(data):
    """Pull the assistant message text out of an OpenAI-style response."""
    try:
        return (data["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError):
        return ""


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


def _call_provider(provider, transcript, title=None):
    """
    Call one provider's chat-completions endpoint with retry on transient errors.
    Returns the summary text, or "" on failure.
    """
    url = f"{provider['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {provider['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": provider["model"],
        "messages": [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(transcript, title)},
        ],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
    }

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
                summary = _extract_summary(resp.json())
            except ValueError as e:
                log_error(f"{provider['name']} returned invalid JSON: {e}")
                return ""
            if not summary:
                log_warn(f"{provider['name']} returned an empty summary.")
            return summary

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


def summarize_transcript(transcript, title=None):
    """
    Summarize a transcript using the first configured LLM provider that succeeds.

    Args:
        transcript: The transcript text to summarize.
        title: Optional video title used to ground the model on the topic.

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

    for provider in providers:
        log_info(f"Summarizing via {provider['name']} ({provider['model']})...")
        summary = _call_provider(provider, transcript, title)
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

    log_warn("All configured LLM providers failed to produce a summary.")
    return ""
