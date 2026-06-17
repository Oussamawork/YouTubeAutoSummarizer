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
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_RETRIES = 3
LLM_RETRY_BACKOFF = 2  # base seconds, multiplied by the attempt number
TRANSIENT_STATUS = {429, 500, 502, 503, 504}

SUMMARY_SYSTEM_PROMPT = (
    "You summarize YouTube video transcripts for a reader who has not watched "
    "the video. Always respond in English, even if the transcript is in another "
    "language. Start with a 1-2 sentence overview, then list 3-6 concise bullet "
    "points capturing the key takeaways. Be faithful to the transcript and do "
    "not invent details. Output only the summary, with no preamble or sign-off."
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


def _call_provider(provider, transcript):
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
            {"role": "user", "content": f"Summarize this transcript:\n\n{transcript}"},
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


def summarize_transcript(transcript):
    """
    Summarize a transcript using the first configured LLM provider that succeeds.

    Returns the summary text, or "" if the transcript is empty, no provider is
    configured, or every provider fails. The caller treats "" as "no summary"
    and notifies accordingly, so this never raises.
    """
    transcript = (transcript or "").strip()
    if not transcript:
        log_warn("Transcript is empty. Nothing to summarize.")
        return ""

    providers = _provider_configs()
    if not providers:
        log_warn(
            "No LLM provider configured. Set GEMINI_API_KEY, GROQ_API_KEY, or "
            "LLM_API_KEY + LLM_BASE_URL to enable summarization."
        )
        return ""

    for provider in providers:
        log_info(f"Summarizing via {provider['name']} ({provider['model']})...")
        summary = _call_provider(provider, transcript)
        if summary:
            log_info(f"Summary generated via {provider['name']}.")
            return summary
        log_warn(f"{provider['name']} did not produce a summary; trying next provider.")

    log_warn("All configured LLM providers failed to produce a summary.")
    return ""
