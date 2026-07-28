"""Tests for market-signal extraction and parsing (LLM mocked)."""
import json

import signals


VALID = {
    "assets": [{
        "name": "Tesla", "ticker": "tsla", "type": "stock", "stance": "bullish",
        "conviction": "high", "action": "buy", "catalysts": ["FSD launch"],
        "price_target": 500, "horizon": "long",
    }],
    "market_sentiment": "mixed",
    "topics": ["EV", "tech"],
}


def test_parse_valid_json():
    parsed = signals._parse_signals(json.dumps(VALID))
    asset = parsed["assets"][0]
    assert asset["ticker"] == "TSLA"  # normalized to uppercase
    assert asset["stance"] == "bullish"
    assert asset["price_target"] == 500
    assert parsed["market_sentiment"] == "mixed"
    assert parsed["topics"] == ["EV", "tech"]


def test_parse_fenced_json():
    parsed = signals._parse_signals("```json\n" + json.dumps(VALID) + "\n```")
    assert parsed is not None
    assert parsed["assets"][0]["name"] == "Tesla"


def test_parse_garbage_returns_none():
    assert signals._parse_signals("not json at all") is None
    assert signals._parse_signals("") is None
    assert signals._parse_signals(None) is None
    assert signals._parse_signals("[1, 2]") is None  # JSON but not an object


def test_parse_drops_invalid_assets_and_normalizes():
    data = {
        "assets": [
            {"name": "", "stance": "bullish"},       # no name -> dropped
            {"name": "X", "stance": "sideways"},     # invalid stance -> dropped
            "not a dict",                            # -> dropped
            {"name": "Nvidia", "stance": "bearish", "type": "weird",
             "conviction": "extreme", "action": "yolo", "catalysts": "not a list",
             "price_target": "high", "horizon": "2027"},
        ],
        "market_sentiment": "euphoric",  # invalid -> neutral
        "topics": "AI",                  # not a list -> []
    }
    parsed = signals._parse_signals(json.dumps(data))
    assert len(parsed["assets"]) == 1
    assert parsed["assets"][0] == {
        "name": "Nvidia", "ticker": None, "type": "other", "stance": "bearish",
        "conviction": "unspecified", "action": "none", "catalysts": [],
        "price_target": None, "horizon": "unspecified",
    }
    assert parsed["market_sentiment"] == "neutral"
    assert parsed["topics"] == []


def test_parse_bool_price_target_rejected():
    data = {"assets": [{"name": "Y", "stance": "neutral", "price_target": True}]}
    parsed = signals._parse_signals(json.dumps(data))
    assert parsed["assets"][0]["price_target"] is None


def test_extract_empty_summary_skips_llm(monkeypatch):
    monkeypatch.setattr(
        signals, "complete",
        lambda *a: (_ for _ in ()).throw(AssertionError("LLM should not be called")),
    )
    assert signals.extract_signals("") is None
    assert signals.extract_signals(None) is None
    assert signals.extract_signals("   ") is None


def test_extract_success(monkeypatch):
    monkeypatch.setattr(signals, "complete", lambda sp, um, **kw: json.dumps(VALID))
    out = signals.extract_signals("summary text", "title", "chan")
    assert out["assets"][0]["name"] == "Tesla"


def test_extract_quota_or_failure_returns_none(monkeypatch):
    monkeypatch.setattr(signals, "complete", lambda sp, um, **kw: signals.QUOTA_EXHAUSTED_SENTINEL)
    assert signals.extract_signals("summary") is None
    monkeypatch.setattr(signals, "complete", lambda sp, um, **kw: "")
    assert signals.extract_signals("summary") is None
    monkeypatch.setattr(signals, "complete", lambda sp, um, **kw: "sorry, I cannot")
    assert signals.extract_signals("summary") is None


def test_extract_passes_context_and_faithfulness_rule(monkeypatch):
    captured = {}

    def fake(system_prompt, user_message, **kw):
        captured["sp"], captured["um"] = system_prompt, user_message
        return json.dumps(VALID)

    monkeypatch.setattr(signals, "complete", fake)
    signals.extract_signals("the summary", "My Video", "My Channel")
    assert "My Video" in captured["um"] and "My Channel" in captured["um"]
    assert "the summary" in captured["um"]
    assert "Never infer" in captured["sp"]


def test_parse_tolerates_preamble_and_trailing_prose():
    wrapped = "Here is the JSON you asked for:\n" + json.dumps(VALID) + "\nLet me know if you need more."
    parsed = signals._parse_signals(wrapped)
    assert parsed is not None and parsed["assets"][0]["name"] == "Tesla"
    # Fenced with a preamble (regex alone can't match) also works.
    fenced = "Sure!\n```json\n" + json.dumps(VALID) + "\n```"
    assert signals._parse_signals(fenced) is not None
    # Still None when there is no JSON object at all.
    assert signals._parse_signals("no braces here") is None


def test_extract_requests_json_mode(monkeypatch):
    captured = {}

    def fake(system_prompt, user_message, **kw):
        captured.update(kw)
        return json.dumps(VALID)

    monkeypatch.setattr(signals, "complete", fake)
    signals.extract_signals("summary", "t", "c")
    assert captured.get("json_mode") is True


# --- Combined summarize + extract (one call instead of two) ---


COMBINED_OK = {"summary": "TL;DR line\n\n• bullet one", "signals": VALID}


def test_summarize_with_signals_success(monkeypatch):
    captured = {}

    def fake(system_prompt, user_message, **kw):
        captured.update(kw, sp=system_prompt, um=user_message)
        return json.dumps(COMBINED_OK)

    monkeypatch.setattr(signals, "complete", fake)
    summary, sig = signals.summarize_with_signals("a transcript", "Title")
    assert summary == "TL;DR line\n\n• bullet one"
    assert sig["assets"][0]["ticker"] == "TSLA"
    assert captured["json_mode"] is True
    assert captured["max_tokens"] == signals.COMBINED_MAX_TOKENS
    assert "a transcript" in captured["um"]
    # Prompt carries both the summary rules and the signals schema.
    assert "TL;DR" in captured["sp"] and '"market_sentiment"' in captured["sp"]


def test_summarize_with_signals_compact_prompt(monkeypatch):
    seen = {}
    monkeypatch.setattr(signals, "complete",
                        lambda sp, um, **kw: seen.update(sp=sp) or json.dumps(COMBINED_OK))
    signals.summarize_with_signals("t", "T", compact=True)
    assert "COMPACT digest entries" in seen["sp"]


def test_summarize_with_signals_bad_json_falls_back(monkeypatch):
    # None tells the caller to use the separate summarize/extract calls.
    monkeypatch.setattr(signals, "complete", lambda sp, um, **kw: "not json at all")
    assert signals.summarize_with_signals("t") is None
    monkeypatch.setattr(signals, "complete", lambda sp, um, **kw: json.dumps({"signals": VALID}))
    assert signals.summarize_with_signals("t") is None  # no summary field
    monkeypatch.setattr(signals, "complete", lambda sp, um, **kw: "")
    assert signals.summarize_with_signals("t") is None


def test_summarize_with_signals_quota_propagates(monkeypatch):
    # Quota is the provider chain's verdict — don't burn a second request.
    monkeypatch.setattr(signals, "complete", lambda sp, um, **kw: signals.QUOTA_EXHAUSTED_SENTINEL)
    summary, sig = signals.summarize_with_signals("t")
    assert summary == signals.QUOTA_EXHAUSTED_SENTINEL and sig is None


def test_summarize_with_signals_insufficient_sentinel(monkeypatch):
    monkeypatch.setattr(
        signals, "complete",
        lambda sp, um, **kw: json.dumps({"summary": "INSUFFICIENT_TRANSCRIPT", "signals": {}}),
    )
    summary, sig = signals.summarize_with_signals("t")
    assert summary == signals.INSUFFICIENT_TRANSCRIPT_SENTINEL and sig is None


def test_summarize_with_signals_summary_survives_bad_signals(monkeypatch):
    # A malformed signals object must not cost us the summary.
    monkeypatch.setattr(
        signals, "complete",
        lambda sp, um, **kw: json.dumps({"summary": "Good summary", "signals": "oops"}),
    )
    summary, sig = signals.summarize_with_signals("t")
    assert summary == "Good summary" and sig is None


def test_summarize_with_signals_empty_transcript_skips_call(monkeypatch):
    monkeypatch.setattr(
        signals, "complete",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not call")),
    )
    assert signals.summarize_with_signals("   ") is None


def test_combined_prompt_has_one_output_contract():
    # The base prompts end with "output only the summary itself". Appending a
    # JSON envelope after that leaves the model two contradictory answers to
    # "what is the response?" — and a malformed response costs a whole extra
    # provider chain on the fallback path.
    import summarizer
    combined = signals._build_combined_prompt()
    assert signals.TRAILING_OUTPUT_RULE not in combined
    assert "ONE JSON object" in combined
    # The rule must still exist in the standalone prompt, and must still match
    # it exactly — a reworded prompt would silently stop being stripped.
    assert summarizer.SUMMARY_SYSTEM_PROMPT.endswith(signals.TRAILING_OUTPUT_RULE)
    assert summarizer.COMPACT_SUMMARY_SYSTEM_PROMPT.endswith(signals.TRAILING_OUTPUT_RULE)
    assert signals._build_combined_prompt(compact=True).count("OUTPUT ENVELOPE") == 1


def test_combined_prompt_routes_the_insufficient_sentinel_into_json():
    # In json_mode "output the single token and nothing else" is not valid
    # JSON, so the sentinel could never come back in a parseable form.
    combined = signals._build_combined_prompt()
    assert '"summary": "INSUFFICIENT_TRANSCRIPT"' in combined


def test_signals_are_scoped_to_the_transcript_not_the_summary():
    # Scoping them to the summary made the structured record a strict subset of
    # the prose, so any asset the prose had no room for vanished from the data
    # that market_pulse and channel_scorecard read.
    assert "market content of the TRANSCRIPT" in signals.COMBINED_SUFFIX


def test_ticker_rule_forbids_supplying_one_from_model_knowledge():
    assert "or unambiguous" not in signals.SIGNALS_SCHEMA
    assert "Never supply one from your own knowledge" in signals.SIGNALS_SCHEMA
