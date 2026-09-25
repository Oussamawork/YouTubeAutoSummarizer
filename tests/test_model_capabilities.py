"""Per-model capability discovery: live, env override, registry, default."""
import json

import model_capabilities as mc


GEMINI = {"name": "gemini-3.7-flash", "model": "gemini-3.7-flash",
          "base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "api_key": "k"}


def test_registry_gives_gemini_its_own_million_token_window():
    caps = mc.capabilities_for(GEMINI)
    assert caps.input_token_limit == 1048576
    assert caps.output_token_limit == 65536
    assert caps.provider == "gemini"
    assert caps.capability_source.startswith("registry@")
    assert caps.supports_count_tokens is True
    # Free-tier TPM is carried as a separate per-request cap, not as the window.
    assert caps.tokens_per_minute_limit == mc.GEMINI_FREE_TIER_TPM


def test_unknown_model_gets_a_conservative_default(monkeypatch):
    caps = mc.capabilities_for({"name": "x", "model": "mystery-9", "base_url": "http://x", "api_key": "k"})
    assert caps.input_token_limit == mc.DEFAULT_INPUT_TOKEN_LIMIT
    assert caps.capability_source == "default"


def test_env_override_wins_over_registry(monkeypatch):
    monkeypatch.setenv("MODEL_CAPABILITIES_JSON", json.dumps({
        "gemini-3.7-flash": {"input_token_limit": 12345, "output_token_limit": 999}}))
    mc.reset_cache()
    caps = mc.capabilities_for(GEMINI)
    assert (caps.input_token_limit, caps.output_token_limit, caps.capability_source) == (12345, 999, "env")


def test_live_metadata_is_used_and_cached(monkeypatch, tmp_path):
    calls = []

    class R:
        status_code = 200

        def json(self):
            return {"inputTokenLimit": 2000000, "outputTokenLimit": 70000}

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        return R()

    monkeypatch.setattr(mc.requests, "get", fake_get)
    mc.reset_cache()
    caps = mc.capabilities_for(GEMINI, use_live=True)
    assert caps.input_token_limit == 2000000 and caps.capability_source == "live"
    assert caps.capability_checked_at
    # Second lookup in a fresh process reads the disk cache: no request.
    mc.reset_cache()
    caps2 = mc.capabilities_for(GEMINI, use_live=True)
    assert caps2.capability_source == "live-cached" and caps2.input_token_limit == 2000000
    assert len(calls) == 1


def test_live_failure_falls_back_to_registry(monkeypatch):
    class R:
        status_code = 503

    monkeypatch.setattr(mc.requests, "get", lambda *a, **k: R())
    mc.reset_cache()
    caps = mc.capabilities_for(GEMINI, use_live=True)
    assert caps.capability_source.startswith("registry@")


def test_fallback_models_are_not_assumed_to_share_a_window():
    groq = mc.capabilities_for({"name": "groq", "model": "llama-3.3-70b-versatile",
                                "base_url": "https://api.groq.com/openai/v1", "api_key": "k"})
    gem = mc.capabilities_for(GEMINI)
    assert groq.input_token_limit != gem.input_token_limit
    assert groq.context_window_limit == 131072  # shared input/output window
    assert gem.context_window_limit is None      # Gemini reports input and output separately
