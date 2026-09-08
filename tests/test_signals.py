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


# --- Combined summarize + claims (one call, two products) ---


def _nt(text="Nvidia will hit $200 by year end I think. I own Tesla shares."):
    import transcript_normalize as tn
    return tn.normalize_transcript(text, "v1")


CLAIM_OK = {
    "attribution_type": "speaker_personal_view", "claim_type": "price_target",
    "is_forward_looking": True, "subject_mention": "Nvidia", "stance": "bullish",
    "target_kind": "absolute_value", "target_value": 200, "currency": "USD",
    "horizon_original": "by year end", "certainty_original": "I think", "certainty_level": "medium",
    "evidence_text": "Nvidia will hit $200 by year end I think", "extraction_confidence": "high",
}
COMBINED_OK = {"summary": "TL;DR line\n\n• bullet one", "claims": [CLAIM_OK],
               "extraction_metadata": {"warnings": []}}


def _ctx():
    # transcript_language arrives verified from the fetch in production
    # (language_detect.verify_language); a 12-word fixture is too short to detect.
    return {"video_id": "v1", "channel_id": "c1", "channel_name": "Chan", "video_title": "T",
            "published_at": "2026-07-01T00:00:00+00:00", "normalized": _nt(), "transcript_language": "en"}


def test_summarize_with_signals_success(monkeypatch):
    captured = {}

    def fake(system_prompt, user_message, **kw):
        captured.update(kw, sp=system_prompt, um=user_message)
        return json.dumps(COMBINED_OK)

    monkeypatch.setattr(signals, "complete", fake)
    summary, research = signals.summarize_with_signals("a transcript", "Title", context=_ctx())
    assert summary == "TL;DR line\n\n• bullet one"
    assert research["status"] == "complete"
    assert research["claims"][0]["ticker"] == "NVDA"          # curated mapping, not a guess
    assert research["claims"][0]["ticker_source"] == "curated_mapping"
    assert research["signals"]["assets"][0]["ticker"] == "NVDA"  # compatibility view
    assert research["signals"]["assets"][0]["price_target"] == 200
    assert captured["json_mode"] is True
    assert captured["max_tokens"] == signals.COMBINED_MAX_OUTPUT_TOKENS
    assert "a transcript" in captured["um"]
    # Prompt carries both the summary rules and the claims contract.
    assert "TL;DR" in captured["sp"] and '"claims"' in captured["sp"]


def test_summarize_with_signals_compact_prompt(monkeypatch):
    seen = {}
    monkeypatch.setattr(signals, "complete",
                        lambda sp, um, **kw: seen.update(sp=sp) or json.dumps(COMBINED_OK))
    signals.summarize_with_signals("t", "T", compact=True, context=_ctx())
    assert "COMPACT digest entries" in seen["sp"]


def test_summarize_with_signals_bad_json_falls_back(monkeypatch):
    # None tells the caller to use the separate summarize/extract calls.
    monkeypatch.setattr(signals, "complete", lambda sp, um, **kw: "not json at all")
    assert signals.summarize_with_signals("t", context=_ctx()) is None
    monkeypatch.setattr(signals, "complete", lambda sp, um, **kw: json.dumps({"claims": []}))
    assert signals.summarize_with_signals("t", context=_ctx()) is None  # no summary field
    monkeypatch.setattr(signals, "complete", lambda sp, um, **kw: "")
    assert signals.summarize_with_signals("t", context=_ctx()) is None
    monkeypatch.setattr(signals, "complete", lambda sp, um, **kw: signals.INPUT_TOO_LARGE_SENTINEL)
    assert signals.summarize_with_signals("t", context=_ctx()) is None  # chunked paths take over


def test_summarize_with_signals_quota_propagates(monkeypatch):
    # Quota is the provider chain's verdict — don't burn a second request.
    monkeypatch.setattr(signals, "complete", lambda sp, um, **kw: signals.QUOTA_EXHAUSTED_SENTINEL)
    summary, research = signals.summarize_with_signals("t", context=_ctx())
    assert summary == signals.QUOTA_EXHAUSTED_SENTINEL and research is None


def test_summarize_with_signals_insufficient_sentinel(monkeypatch):
    monkeypatch.setattr(
        signals, "complete",
        lambda sp, um, **kw: json.dumps({"summary": "INSUFFICIENT_TRANSCRIPT", "claims": []}),
    )
    summary, research = signals.summarize_with_signals("t", context=_ctx())
    assert summary == signals.INSUFFICIENT_TRANSCRIPT_SENTINEL and research is None


def test_valid_summary_with_malformed_claims_keeps_summary_and_flags_research(monkeypatch):
    # The summary is delivered; the claims failure is a retryable research
    # state — never an empty-signal success.
    monkeypatch.setattr(
        signals, "complete",
        lambda sp, um, **kw: json.dumps({"summary": "Good summary", "claims": "oops"}),
    )
    summary, research = signals.summarize_with_signals("t", context=_ctx())
    assert summary == "Good summary"
    assert research["status"] == "failed_retryable"
    assert research["failure_reason"] == "malformed_claims"
    assert research["signals"] is None                  # not {"assets": []}
    assert research["claims"] == []


def test_genuinely_no_claims_is_distinct_from_failure(monkeypatch):
    monkeypatch.setattr(
        signals, "complete",
        lambda sp, um, **kw: json.dumps({"summary": "S", "claims": [], "extraction_metadata": {"warnings": []}}),
    )
    ctx = _ctx()
    ctx["normalized"] = _nt("Welcome back everyone. Today I walk through how to open a brokerage account "
                            "and what a limit order is. Thanks for watching.")
    summary, research = signals.summarize_with_signals("t", context=ctx)
    assert research["status"] == "no_claims_found"
    assert research["signals"] == {"assets": [], "market_sentiment": "neutral", "topics": [],
                                   "derived_from": "claims"}


def test_empty_claims_on_a_forecast_transcript_is_suspicious_not_no_claims(monkeypatch):
    # The default fixture transcript says "Nvidia will hit $200 by year end":
    # an empty array against it is not believed. From the combined call the
    # video goes to a standalone extraction pass; the compatibility view is
    # null, never an empty asset list.
    monkeypatch.setattr(
        signals, "complete",
        lambda sp, um, **kw: json.dumps({"summary": "S", "claims": [], "extraction_metadata": {"warnings": []}}),
    )
    summary, research = signals.summarize_with_signals("t", context=_ctx())
    assert summary == "S"
    assert research["status"] == "failed_retryable"
    assert research["failure_reason"] == "suspicious_empty_extraction"
    assert research["signals"] is None


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
    assert summarizer.SUMMARY_SYSTEM_PROMPT.endswith(signals.TRAILING_OUTPUT_RULE)
    assert summarizer.COMPACT_SUMMARY_SYSTEM_PROMPT.endswith(signals.TRAILING_OUTPUT_RULE)
    assert signals._build_combined_prompt(compact=True).count("OUTPUT ENVELOPE") == 1


def test_combined_prompt_routes_the_insufficient_sentinel_into_json():
    combined = signals._build_combined_prompt()
    assert '"summary": "INSUFFICIENT_TRANSCRIPT"' in combined


def test_claims_are_scoped_to_the_transcript_not_the_summary():
    assert "built from the whole TRANSCRIPT" in signals.COMBINED_SUFFIX


def test_ticker_rule_forbids_supplying_one_from_model_knowledge():
    assert "or unambiguous" not in signals.SIGNALS_SCHEMA
    assert "Never supply one from your own knowledge" in signals.SIGNALS_SCHEMA
    assert "Never invent a ticker" in signals.COMBINED_SUFFIX


def test_conviction_cannot_be_forced_into_a_guess():
    assert '"unspecified"' in signals.SIGNALS_SCHEMA.split('"conviction"')[1].split("\n")[0]
    assert "never guess one" in signals.SIGNALS_SCHEMA


# --- Standalone research extraction (separate call / retry) ---


def test_extract_research_single_call(monkeypatch):
    monkeypatch.setattr(signals, "complete",
                        lambda sp, um, **kw: json.dumps({"claims": [CLAIM_OK], "extraction_metadata": {}}))
    out = signals.extract_research(_nt(), _ctx())
    assert out["status"] == "complete" and out["coverage_status"] == "full"
    assert len(out["claims"]) == 1


def test_extract_research_quota_is_deferred_not_failed(monkeypatch):
    monkeypatch.setattr(signals, "complete", lambda sp, um, **kw: signals.QUOTA_EXHAUSTED_SENTINEL)
    out = signals.extract_research(_nt(), _ctx())
    assert out["status"] == "quota_deferred" and out["signals"] is None


def test_extract_research_chunks_when_too_large_and_resumes(monkeypatch, tmp_path):
    # Over-limit research runs in complete-coverage chunks; a quota stop
    # midway leaves the finished chunks on disk and reports partial coverage;
    # the next attempt reruns only what is missing.
    import transcript_normalize as tn
    text = " ".join(f"Sentence {i} says Nvidia will hit ${100 + i} by year end I think." for i in range(120))
    nt = tn.normalize_transcript(text, "v1")
    ctx = dict(_ctx(), normalized=nt)
    monkeypatch.setattr(signals, "_research_chunk_tokens", lambda: 600)
    calls = {"n": 0}

    def fake_complete(sp, um, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return signals.INPUT_TOO_LARGE_SENTINEL   # the single call does not fit
        if calls["n"] == 3:
            return signals.QUOTA_EXHAUSTED_SENTINEL   # second chunk hits quota
        claim = dict(CLAIM_OK, evidence_text="Sentence 0 says Nvidia will hit $100 by year end I think",
                     target_value=100)
        return json.dumps({"claims": [claim] if calls["n"] == 2 else []})

    monkeypatch.setattr(signals, "complete", fake_complete)
    out = signals.extract_research(nt, ctx)
    assert out["status"] == "quota_deferred"
    assert out["coverage_status"] == "partial"
    assert len(out["processed_chunk_ids"]) == 1 and out["chunks"] > 1
    assert out["signals"] is None                       # partial claims never feed headline data
    assert all(c["coverage_status"] == "partial" for c in out["claims"])

    # Resume: the first chunk is served from disk, only the rest run.
    before = calls["n"]
    monkeypatch.setattr(signals, "complete",
                        lambda sp, um, **kw: json.dumps({"claims": []}) if "part 1 of" not in um
                        else (_ for _ in ()).throw(AssertionError("chunk 1 must not rerun")))
    monkeypatch.setattr(signals, "EXHAUSTIVE_RESEARCH_MODE", True)  # go straight to chunks
    out2 = signals.extract_research(nt, ctx)
    assert out2["status"] == "complete" and out2["coverage_status"] == "chunked_full"
    assert len(out2["processed_chunk_ids"]) == out2["chunks"]
    assert calls["n"] == before  # the fake above never touched the counter, and chunk 1 was cached
