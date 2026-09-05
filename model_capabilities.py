"""
Per-model capability registry: how much a model can read, how much it can
write, and where that knowledge came from.

Why this exists: the summarizer used to cap every transcript at 120,000
characters and cut the middle out of anything longer. That number was an
application decision dressed up as a model limit, and it silently dropped the
middle of long videos — the part that carries the second asset, the price
target, the condition on the forecast. The real limits are per model, and a
fallback model may have a different one, so each configured model is evaluated
on its own capabilities.

Sources, in order of trust:
  1. live   — Gemini's `models.get` endpoint reports `inputTokenLimit` and
              `outputTokenLimit`; cached in data/model_capabilities.json so a
              value that changes once a quarter is not fetched eight times a day.
  2. env    — MODEL_CAPABILITIES_JSON, an operator override for a proxy or a
              model the registry does not know.
  3. registry — the versioned table below, dated and conservative.
  4. default  — an unknown model gets a small window on purpose: guessing big
              risks a rejected request; guessing small merely chunks.

Nothing here raises: a failed lookup falls through to the next source.
"""
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import requests

from helpers import env_int, env_flag, write_json_atomic
from log import log_info, log_warn

REGISTRY_VERSION = "2026-09-05"
CAPABILITIES_CACHE_FILE = os.getenv("MODEL_CAPABILITIES_CACHE") or "data/model_capabilities.json"
# Live metadata is re-fetched after this many hours; model limits change on
# the order of releases, not runs.
CAPABILITIES_TTL_HOURS = env_int("MODEL_CAPABILITIES_TTL_HOURS", 24 * 7)
CAPABILITIES_TIMEOUT = env_int("MODEL_CAPABILITIES_TIMEOUT", 15)
# Live lookups are opt-out, not opt-in: without them every model runs on the
# registry's assumptions. Tests and offline runs set this to false.
LIVE_CAPABILITIES = env_flag("LLM_LIVE_CAPABILITIES", default=True)
GEMINI_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}"
GEMINI_HOST = "generativelanguage.googleapis.com"

# The free tier meters tokens per minute per model (250k on this project's
# key, see docs/tdd-gemini-transcripts.md §2). It is a rate limit, not a
# context limit, but a single request larger than the minute's allowance is
# rejected outright, so it caps one request's input just as hard as the
# window does. Configurable because paid tiers lift it; 0 disables the cap.
GEMINI_FREE_TIER_TPM = env_int("GEMINI_FREE_TIER_TPM", 250000)


@dataclass
class ModelCapabilities:
    model_id: str
    provider: str
    input_token_limit: int
    output_token_limit: int
    context_window_limit: int = None  # when the API reports one separately
    tokens_per_minute_limit: int = None  # per-request hard cap on some tiers
    supports_structured_output: bool = True
    supports_count_tokens: bool = False
    supports_chat_completions: bool = True
    capability_source: str = "default"
    capability_checked_at: str = None

    def to_dict(self):
        return asdict(self)


# Versioned table. Values are the vendors' published limits on the registry
# date; the live source overrides them when reachable. Keep entries
# conservative: an understated limit chunks a transcript that would have fit,
# an overstated one gets the request rejected.
REGISTRY = {
    # Gemini 3.x Flash family: 1,048,576-token window, 65,536 output.
    "gemini-3.7-flash": dict(provider="gemini", input_token_limit=1048576,
                             output_token_limit=65536, supports_count_tokens=True),
    "gemini-3.6-flash": dict(provider="gemini", input_token_limit=1048576,
                             output_token_limit=65536, supports_count_tokens=True),
    "gemini-3.5-flash": dict(provider="gemini", input_token_limit=1048576,
                             output_token_limit=65536, supports_count_tokens=True),
    "gemini-3-flash-preview": dict(provider="gemini", input_token_limit=1048576,
                                   output_token_limit=65536, supports_count_tokens=True),
    "gemini-2.5-flash": dict(provider="gemini", input_token_limit=1048576,
                             output_token_limit=65536, supports_count_tokens=True),
    # Groq-hosted Llama: 128k context, 32k output.
    # These share one window between input and output (unlike Gemini, whose
    # input and output limits are reported separately), so the context limit
    # is declared and the budget subtracts the reserved output from it.
    "llama-3.3-70b-versatile": dict(provider="groq", input_token_limit=131072,
                                    context_window_limit=131072, output_token_limit=32768),
    # OpenAI small model: 128k context, 16k output.
    "gpt-4o-mini": dict(provider="openai", input_token_limit=128000,
                        context_window_limit=128000, output_token_limit=16384),
}

# An unknown model: small on purpose (see module docstring).
DEFAULT_INPUT_TOKEN_LIMIT = 32768
DEFAULT_OUTPUT_TOKEN_LIMIT = 4096

_MEMO = {}


def reset_cache(memo=None):
    """Test seam: clear the in-process memo (and optionally seed it)."""
    global _MEMO
    _MEMO = dict(memo or {})
    return _MEMO


def provider_kind(provider):
    """'gemini' | 'groq' | 'openai' | 'custom' from a provider config."""
    base = (provider.get("base_url") or "").lower()
    if GEMINI_HOST in base:
        return "gemini"
    if "api.groq.com" in base:
        return "groq"
    if "api.openai.com" in base:
        return "openai"
    return provider.get("name") or "custom"


def _now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _from_env_override(model):
    raw = (os.getenv("MODEL_CAPABILITIES_JSON") or "").strip()
    if not raw:
        return None
    try:
        table = json.loads(raw)
    except ValueError:
        log_warn("MODEL_CAPABILITIES_JSON is not valid JSON; ignoring it.")
        return None
    entry = table.get(model) if isinstance(table, dict) else None
    if not isinstance(entry, dict):
        return None
    try:
        return {
            "input_token_limit": int(entry["input_token_limit"]),
            "output_token_limit": int(entry.get("output_token_limit") or DEFAULT_OUTPUT_TOKEN_LIMIT),
            "context_window_limit": entry.get("context_window_limit"),
            "tokens_per_minute_limit": entry.get("tokens_per_minute_limit"),
        }
    except (KeyError, TypeError, ValueError):
        log_warn(f"MODEL_CAPABILITIES_JSON entry for {model} is malformed; ignoring it.")
        return None


def _load_disk_cache():
    try:
        with open(CAPABILITIES_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("models"), dict):
            return data
    except (OSError, ValueError):
        pass
    return {"version": 1, "models": {}}


def _cache_fresh(entry):
    checked = (entry or {}).get("capability_checked_at")
    try:
        at = datetime.fromisoformat(checked)
    except (TypeError, ValueError):
        return False
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - at).total_seconds() / 3600.0
    return age_hours < CAPABILITIES_TTL_HOURS


def _fetch_gemini_live(model, api_key):
    """(input_limit, output_limit) from models.get, or None on any failure.
    A metadata GET is not a generateContent request, so it spends none of the
    model's 20 daily requests."""
    try:
        resp = requests.get(
            GEMINI_MODELS_URL.format(model=model),
            headers={"x-goog-api-key": api_key},
            timeout=CAPABILITIES_TIMEOUT,
        )
    except requests.RequestException as e:
        log_warn(f"Model metadata lookup failed for {model}: {e}")
        return None
    if resp.status_code != 200:
        log_warn(f"Model metadata lookup for {model} returned {resp.status_code}.")
        return None
    try:
        data = resp.json()
        return int(data["inputTokenLimit"]), int(data["outputTokenLimit"])
    except (ValueError, KeyError, TypeError):
        log_warn(f"Model metadata for {model} had no token limits.")
        return None


def capabilities_for(provider, use_live=None):
    """
    The ModelCapabilities for one provider config ({"model", "base_url",
    "api_key", ...}). Memoized per process and cached on disk for the live
    source; never raises.
    """
    model = provider.get("model") or ""
    kind = provider_kind(provider)
    key = (kind, model)
    if key in _MEMO:
        return _MEMO[key]

    caps = None
    use_live = LIVE_CAPABILITIES if use_live is None else use_live

    override = _from_env_override(model)
    if override:
        caps = ModelCapabilities(model_id=model, provider=kind, capability_source="env",
                                 capability_checked_at=_now_iso(),
                                 supports_count_tokens=(kind == "gemini"), **override)

    if caps is None and kind == "gemini":
        disk = _load_disk_cache()
        entry = disk["models"].get(model)
        if entry and _cache_fresh(entry):
            caps = ModelCapabilities(**{**entry, "model_id": model, "provider": kind,
                                        "capability_source": "live-cached"})
        elif use_live and provider.get("api_key"):
            live = _fetch_gemini_live(model, provider["api_key"])
            if live:
                caps = ModelCapabilities(
                    model_id=model, provider=kind, input_token_limit=live[0],
                    output_token_limit=live[1], supports_count_tokens=True,
                    capability_source="live", capability_checked_at=_now_iso(),
                )
                stored = caps.to_dict()
                stored.pop("model_id"), stored.pop("provider"), stored.pop("capability_source")
                disk["models"][model] = stored
                write_json_atomic(CAPABILITIES_CACHE_FILE, disk)
                log_info(f"Model {model}: input limit {live[0]:,} / output {live[1]:,} tokens (live).")

    if caps is None:
        entry = REGISTRY.get(model)
        if entry:
            caps = ModelCapabilities(model_id=model, capability_source=f"registry@{REGISTRY_VERSION}",
                                     capability_checked_at=_now_iso(), **entry)
        else:
            log_warn(
                f"No capability entry for model {model!r}; assuming a "
                f"{DEFAULT_INPUT_TOKEN_LIMIT:,}-token input window. Set "
                "MODEL_CAPABILITIES_JSON to declare its real limits."
            )
            caps = ModelCapabilities(model_id=model, provider=kind,
                                     input_token_limit=DEFAULT_INPUT_TOKEN_LIMIT,
                                     output_token_limit=DEFAULT_OUTPUT_TOKEN_LIMIT,
                                     capability_source="default",
                                     capability_checked_at=_now_iso())

    if caps.provider == "gemini" and caps.tokens_per_minute_limit is None and GEMINI_FREE_TIER_TPM > 0:
        caps.tokens_per_minute_limit = GEMINI_FREE_TIER_TPM
    if caps.provider == "gemini":
        caps.supports_count_tokens = True
    _MEMO[key] = caps
    return caps
