"""
Token-aware request sizing.

The question this module answers is "does THIS request fit THIS model?", and
it answers it in tokens, for the complete request — system prompt, user
instructions, title, transcript, JSON-envelope instructions — not in
characters of transcript alone. Character counts are kept as telemetry only.

Counting, cheapest-exact-first:
  1. Gemini's native `countTokens` endpoint, when the provider is Gemini and
     LLM_LIVE_COUNT_TOKENS is on. It is a separate endpoint from
     generateContent: it does not consume one of a model's 20 daily
     generation requests and is free, but it has its own per-minute rate
     limit, so a failure here (429, timeout, anything) falls through to the
     estimate rather than blocking the request.
  2. A deliberately conservative local estimate: 1 token per
     ESTIMATE_CHARS_PER_TOKEN characters (3.0 by default, where English prose
     runs at roughly 4), plus a small per-message overhead. Overestimating
     only costs an unnecessary chunked pass; underestimating gets a request
     rejected.
No generative call is ever made to count tokens.

Budget: input and output are separate limits and are kept separate.
  available_input = min(input_token_limit,
                        context_window_limit - reserved_output   [if shared window],
                        tokens_per_minute_limit - reserved_output [if metered])
                    - CONTEXT_SAFETY_MARGIN_TOKENS
"""
import hashlib
import math
from dataclasses import dataclass, field

import requests

from helpers import env_int, env_float, env_flag
from log import log_warn
from model_capabilities import capabilities_for, GEMINI_HOST

# Conservative characters-per-token for the estimate path (see docstring).
ESTIMATE_CHARS_PER_TOKEN = env_float("LLM_ESTIMATE_CHARS_PER_TOKEN", 3.0)
# Tokens the provider adds around each message (role markers, separators).
MESSAGE_OVERHEAD_TOKENS = 8
# Headroom left unused under the computed input capacity: covers tokenizer
# drift between the estimate and the model, response_format scaffolding, and
# whatever wrapper text the compatibility layer adds.
CONTEXT_SAFETY_MARGIN_TOKENS = env_int("CONTEXT_SAFETY_MARGIN_TOKENS", 2048)
LIVE_COUNT_TOKENS = env_flag("LLM_LIVE_COUNT_TOKENS", default=True)
COUNT_TOKENS_TIMEOUT = env_int("LLM_COUNT_TOKENS_TIMEOUT", 20)
GEMINI_COUNT_TOKENS_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:countTokens"

_COUNT_MEMO = {}


def reset_count_cache():
    """Test seam."""
    _COUNT_MEMO.clear()


@dataclass
class RequestSize:
    """Token size of one complete candidate request."""
    input_tokens: int
    method: str                   # "count_tokens" (exact) | "estimate"
    parts: dict = field(default_factory=dict)   # per-part token estimates
    transcript_chars: int = 0

    @property
    def estimated(self):
        return self.method != "count_tokens"


@dataclass
class ContextBudget:
    model_id: str
    provider: str
    input_token_limit: int
    available_input_tokens: int
    reserved_output_tokens: int
    safety_margin_tokens: int
    capability_source: str

    def fits(self, size):
        return size.input_tokens <= self.available_input_tokens


def estimate_tokens(text):
    """Conservative token estimate for a string (never fewer than 1 for
    non-empty text)."""
    if not text:
        return 0
    return max(1, int(math.ceil(len(text) / ESTIMATE_CHARS_PER_TOKEN)))


def estimate_request(system_prompt, user_message, extra_parts=None):
    """Estimate a whole request: every message plus per-message overhead."""
    parts = {
        "system_prompt": estimate_tokens(system_prompt),
        "user_message": estimate_tokens(user_message),
    }
    for name, text in (extra_parts or {}).items():
        parts[name] = estimate_tokens(text)
    total = sum(parts.values()) + MESSAGE_OVERHEAD_TOKENS * (2 + len(extra_parts or {}))
    return total, parts


def _gemini_count_tokens(provider, system_prompt, user_message):
    """Exact count from countTokens, or None on any failure."""
    payload = {
        "contents": [{"role": "user", "parts": [{"text": user_message or ""}]}],
    }
    if system_prompt:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
    try:
        resp = requests.post(
            GEMINI_COUNT_TOKENS_URL.format(model=provider["model"]),
            headers={"x-goog-api-key": provider["api_key"], "Content-Type": "application/json"},
            json=payload, timeout=COUNT_TOKENS_TIMEOUT,
        )
    except requests.RequestException as e:
        log_warn(f"countTokens failed for {provider['model']} ({e}); using the estimate.")
        return None
    if resp.status_code != 200:
        log_warn(f"countTokens returned {resp.status_code} for {provider['model']}; using the estimate.")
        return None
    try:
        return int(resp.json()["totalTokens"])
    except (ValueError, KeyError, TypeError):
        log_warn(f"countTokens gave no totalTokens for {provider['model']}; using the estimate.")
        return None


def measure_request(provider, system_prompt, user_message, transcript_chars=0, use_live=None):
    """
    Size the exact candidate request for `provider`. Exact where the provider
    offers a counting endpoint, a conservative estimate otherwise; the method
    is recorded on the result so telemetry can say which it was.
    """
    est_total, parts = estimate_request(system_prompt, user_message)
    use_live = LIVE_COUNT_TOKENS if use_live is None else use_live
    if use_live and GEMINI_HOST in (provider.get("base_url") or "") and provider.get("api_key"):
        digest = hashlib.sha1(
            (provider["model"] + "\x00" + (system_prompt or "") + "\x00" + (user_message or "")).encode("utf-8")
        ).hexdigest()
        if digest not in _COUNT_MEMO:
            _COUNT_MEMO[digest] = _gemini_count_tokens(provider, system_prompt, user_message)
        exact = _COUNT_MEMO[digest]
        if exact is not None:
            return RequestSize(input_tokens=exact, method="count_tokens", parts=parts,
                               transcript_chars=transcript_chars)
    return RequestSize(input_tokens=est_total, method="estimate", parts=parts,
                       transcript_chars=transcript_chars)


def context_budget(provider, reserved_output_tokens, margin=None, caps=None):
    """
    How many input tokens one request to `provider` may carry, given the
    output allowance it reserves. Each model is evaluated on its own limits —
    a fallback never inherits the primary model's window.
    """
    caps = caps or capabilities_for(provider)
    margin = CONTEXT_SAFETY_MARGIN_TOKENS if margin is None else margin
    reserved = min(int(reserved_output_tokens or 0), caps.output_token_limit)
    candidates = [caps.input_token_limit]
    if caps.context_window_limit:
        candidates.append(caps.context_window_limit - reserved)
    if caps.tokens_per_minute_limit:
        candidates.append(caps.tokens_per_minute_limit - reserved)
    available = max(0, min(candidates) - margin)
    return ContextBudget(
        model_id=caps.model_id, provider=caps.provider,
        input_token_limit=caps.input_token_limit,
        available_input_tokens=available, reserved_output_tokens=reserved,
        safety_margin_tokens=margin, capability_source=caps.capability_source,
    )


def usage_from_response(data):
    """(prompt_tokens, completion_tokens) reported by an OpenAI-style
    response, or (None, None) when absent."""
    try:
        usage = data.get("usage") or {}
        p = usage.get("prompt_tokens")
        c = usage.get("completion_tokens")
        return (int(p) if p is not None else None, int(c) if c is not None else None)
    except (AttributeError, TypeError, ValueError):
        return None, None
