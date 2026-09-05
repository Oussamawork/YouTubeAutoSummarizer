"""Token-aware request sizing: the complete request, per model, never chars."""
import model_capabilities as mc
import token_budget as tb


GEMINI = {"name": "gemini-3.7-flash", "model": "gemini-3.7-flash",
          "base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "api_key": "k"}


def test_estimate_is_conservative():
    text = "word " * 1000  # 5000 chars, ~1250 real tokens
    assert tb.estimate_tokens(text) >= 1250
    assert tb.estimate_tokens("") == 0


def test_measure_counts_every_part_of_the_request():
    size = tb.measure_request(GEMINI, "SYSTEM " * 100, "Video title: T\n\nTRANSCRIPT " * 200,
                              transcript_chars=2200)
    assert size.method == "estimate"
    assert size.parts["system_prompt"] > 0 and size.parts["user_message"] > 0
    assert size.input_tokens > size.parts["system_prompt"] + size.parts["user_message"]  # + overhead
    assert size.transcript_chars == 2200


def test_live_count_tokens_is_exact_and_memoized(monkeypatch):
    calls = []

    class R:
        status_code = 200

        def json(self):
            return {"totalTokens": 4321}

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json)
        return R()

    monkeypatch.setattr(tb.requests, "post", fake_post)
    tb.reset_count_cache()
    size = tb.measure_request(GEMINI, "sys", "user text", use_live=True)
    assert size.input_tokens == 4321 and size.method == "count_tokens"
    # The exact request went to countTokens: system instruction + user content.
    assert calls[0]["systemInstruction"]["parts"][0]["text"] == "sys"
    assert calls[0]["contents"][0]["parts"][0]["text"] == "user text"
    tb.measure_request(GEMINI, "sys", "user text", use_live=True)
    assert len(calls) == 1  # an escalation retry must not count again


def test_live_count_failure_falls_back_to_estimate(monkeypatch):
    class R:
        status_code = 429

    monkeypatch.setattr(tb.requests, "post", lambda *a, **k: R())
    tb.reset_count_cache()
    size = tb.measure_request(GEMINI, "sys", "user", use_live=True)
    assert size.method == "estimate"


def test_budget_keeps_input_and_output_separate():
    budget = tb.context_budget(GEMINI, reserved_output_tokens=8000, margin=1000)
    # Gemini's input limit stands on its own; the free-tier TPM cap is the
    # binding per-request bound here, minus reserved output and margin.
    assert budget.input_token_limit == 1048576
    assert budget.available_input_tokens == mc.GEMINI_FREE_TIER_TPM - 8000 - 1000
    assert budget.reserved_output_tokens == 8000


def test_budget_without_tpm_cap_uses_the_model_window(monkeypatch):
    monkeypatch.setattr(mc, "GEMINI_FREE_TIER_TPM", 0)
    mc.reset_cache()
    budget = tb.context_budget(GEMINI, reserved_output_tokens=8000, margin=1000)
    assert budget.available_input_tokens == 1048576 - 1000


def test_shared_window_models_subtract_reserved_output():
    groq = {"name": "groq", "model": "llama-3.3-70b-versatile",
            "base_url": "https://api.groq.com/openai/v1", "api_key": "k"}
    budget = tb.context_budget(groq, reserved_output_tokens=4000, margin=1000)
    assert budget.available_input_tokens == 131072 - 4000 - 1000


def test_fits_compares_the_measured_request():
    budget = tb.context_budget(GEMINI, reserved_output_tokens=4000)
    small = tb.measure_request(GEMINI, "s", "u")
    huge = tb.RequestSize(input_tokens=budget.available_input_tokens + 1, method="estimate")
    assert budget.fits(small) and not budget.fits(huge)


def test_usage_from_response():
    assert tb.usage_from_response({"usage": {"prompt_tokens": 10, "completion_tokens": 5}}) == (10, 5)
    assert tb.usage_from_response({}) == (None, None)
    assert tb.usage_from_response(None) == (None, None)
