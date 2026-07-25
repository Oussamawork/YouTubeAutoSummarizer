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
