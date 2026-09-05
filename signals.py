import json
import re

from helpers import env_int
from log import log_info, log_warn
import os

import claims as claims_mod
import research_state
import summarizer
import token_budget
from helpers import env_flag
from summarizer import (
    complete,
    QUOTA_EXHAUSTED_SENTINEL,
    INSUFFICIENT_TRANSCRIPT_SENTINEL,
    INPUT_TOO_LARGE_SENTINEL,
    TRUNCATED_SENTINEL,
    SUMMARY_SYSTEM_PROMPT,
    COMPACT_SUMMARY_SYSTEM_PROMPT,
    _build_user_message,
)

# Structured market-signal extraction from finance-video summaries.
#
# One extra (cheap) LLM call per summarized video turns the free-text summary
# into machine-readable signals — which assets the speaker discussed and what
# stance they took — appended to data/signals.jsonl for later aggregation
# (sentiment time series, consensus flips, per-channel scorecards). Extraction
# is strictly best-effort: any failure returns None and must never affect
# summary delivery or dedup state. The output is research data, not advice.

ASSET_TYPES = {"stock", "crypto", "etf", "index", "commodity", "macro"}
STANCES = {"bullish", "bearish", "neutral"}
CONVICTIONS = {"low", "medium", "high"}
ACTIONS = {"buy", "sell", "hold", "watch", "none"}
HORIZONS = {"short", "medium", "long", "unspecified"}
MARKET_SENTIMENTS = {"bullish", "bearish", "neutral", "mixed"}

# The signals object's shape and rules, shared by the standalone extraction
# prompt and the combined summarize+extract prompt so they can never drift.
# `include_catalysts=False` slims the per-asset cost for the combined call:
# catalysts restate reasoning the summary bullets already carry, they are read
# by no aggregator, and at ~17 assets their token cost is exactly what made the
# model quietly stop listing assets partway through the array.
def _signals_schema(include_catalysts=True):
    catalysts_line = (
        '      "catalysts": ["<short phrase per reason/catalyst the speaker gives>"],\n'
        if include_catalysts else ""
    )
    return (
        "{\n"
        '  "assets": [\n'
        "    {\n"
        '      "name": "<company/asset name as stated>",\n'
        '      "ticker": "<the ticker ONLY if the speaker says it aloud or it appears in the video title; otherwise null. Never supply one from your own knowledge of the company>",\n'
        '      "type": "stock" | "crypto" | "etf" | "index" | "commodity" | "macro",\n'
        '      "stance": "bullish" | "bearish" | "neutral",\n'
        '      "conviction": "low" | "medium" | "high" | "unspecified",\n'
        '      "action": "buy" | "sell" | "hold" | "watch" | "none",\n'
        + catalysts_line +
        '      "price_target": <number or null>,\n'
        '      "horizon": "short" | "medium" | "long" | "unspecified"\n'
        "    }\n"
        "  ],\n"
        '  "market_sentiment": "bullish" | "bearish" | "neutral" | "mixed",\n'
        '  "topics": ["<2-5 short topic tags>"]\n'
        "}\n"
        "\n"
        "Rules:\n"
        "- Include EVERY asset the source discusses — one entry per asset, "
        "including passing mentions. Never stop the list early: an incomplete "
        "assets array corrupts the downstream analytics that read it. If an "
        "asset gets no clear stance, include it with stance \"neutral\" rather "
        "than leaving it out.\n"
        "- Report ONLY what the speaker actually says. Never infer, "
        "extrapolate, or invent stances, tickers, price targets, or reasons.\n"
        '- "conviction" is how strongly the speaker holds the view AS STATED; '
        'use "unspecified" when they give no strength — never guess one.\n'
        '- "action" is the speaker\'s own stated action or recommendation; use '
        '"none" when they state no action.\n'
        '- "market_sentiment" is the speaker\'s overall tone about markets in '
        "this video, not your own view.\n"
        "- If there is no market-relevant content, use "
        '{"assets": [], "market_sentiment": "neutral", "topics": []}.'
    )


SIGNALS_SCHEMA = _signals_schema()

SIGNALS_SYSTEM_PROMPT = (
    "You extract structured market signals from the summary of a finance-related "
    "YouTube video. Output ONLY a JSON object — no markdown fences, no prose — "
    "with exactly this shape:\n" + SIGNALS_SCHEMA
)

SIGNALS_USER_TEMPLATE = (
    "Channel: {channel_name}\n"
    "Video title: {video_title}\n"
    "\n"
    "Video summary:\n"
    "{summary}"
)

# Matches an optional ```json ... ``` fence around the model's output.
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def _strip_code_fences(text):
    match = _FENCE_RE.match(text.strip())
    return match.group(1) if match else text


def _normalize_asset(asset):
    """Validate/normalize one asset entry; None drops entries missing the core
    fields (a signal without a named asset and a stance is unusable)."""
    if not isinstance(asset, dict):
        return None
    name = asset.get("name")
    stance = asset.get("stance")
    if not isinstance(name, str) or not name.strip():
        return None
    if stance not in STANCES:
        return None

    ticker = asset.get("ticker")
    if not isinstance(ticker, str) or not ticker.strip():
        ticker = None
    price_target = asset.get("price_target")
    if not isinstance(price_target, (int, float)) or isinstance(price_target, bool):
        price_target = None
    catalysts = asset.get("catalysts")
    if not isinstance(catalysts, list):
        catalysts = []
    catalysts = [c.strip() for c in catalysts if isinstance(c, str) and c.strip()]

    return {
        "name": name.strip(),
        "ticker": ticker.strip().upper() if ticker else None,
        "type": asset.get("type") if asset.get("type") in ASSET_TYPES else "other",
        "stance": stance,
        "conviction": asset.get("conviction") if asset.get("conviction") in CONVICTIONS else "unspecified",
        "action": asset.get("action") if asset.get("action") in ACTIONS else "none",
        "catalysts": catalysts,
        "price_target": price_target,
        "horizon": asset.get("horizon") if asset.get("horizon") in HORIZONS else "unspecified",
    }


def _parse_signals(text):
    """
    Parse the model's output into a validated signals dict, or None when it
    isn't usable JSON of the expected shape. Tolerates code fences and prose
    around the JSON object (models add preambles despite instructions — seen
    in production with gemini-2.5-flash). Never raises.
    """
    if not text:
        return None
    stripped = _strip_code_fences(text)
    try:
        data = json.loads(stripped)
    except (ValueError, TypeError):
        # Fall back to the outermost {...} span — handles "Here is the JSON:"
        # preambles, trailing commentary, and fences the regex didn't match.
        start, end = stripped.find("{"), stripped.rfind("}")
        if start == -1 or end <= start:
            log_warn("Signal extraction returned no JSON object; discarding.")
            return None
        try:
            data = json.loads(stripped[start:end + 1])
        except (ValueError, TypeError) as e:
            log_warn(f"Signal extraction returned unparseable JSON: {e}")
            return None
    if not isinstance(data, dict):
        log_warn("Signal extraction returned JSON that is not an object; discarding.")
        return None

    raw_assets = data.get("assets")
    assets = []
    if isinstance(raw_assets, list):
        for asset in raw_assets:
            normalized = _normalize_asset(asset)
            if normalized is not None:
                assets.append(normalized)

    sentiment = data.get("market_sentiment")
    if sentiment not in MARKET_SENTIMENTS:
        sentiment = "neutral"

    topics = data.get("topics")
    if not isinstance(topics, list):
        topics = []
    topics = [t.strip() for t in topics if isinstance(t, str) and t.strip()]

    return {"assets": assets, "market_sentiment": sentiment, "topics": topics}


# --- Combined summarize + claims: one call, two products ---------------------
#
# The ordinary fast path: one request returns the Telegram summary AND the
# atomic claims for the research dataset. The two are validated independently
# — a malformed claims array never costs the summary, and a valid summary
# never makes a failed extraction look like "no claims" (see build_research).
# Output caps are separate from input capacity: this cap bounds what the
# model may WRITE; how much transcript it may READ is the model's own input
# limit, checked per request by the summarizer.
COMBINED_MAX_OUTPUT_TOKENS = env_int("COMBINED_MAX_OUTPUT_TOKENS", env_int("LLM_COMBINED_MAX_TOKENS", 8000))
COMBINED_MAX_TOKENS = COMBINED_MAX_OUTPUT_TOKENS  # legacy name
# Standalone claim extraction (research retry, chunked research) writes claims
# only, so it needs less than the combined envelope.
CLAIMS_MAX_OUTPUT_TOKENS = env_int("CLAIMS_MAX_OUTPUT_TOKENS", 6000)
# High-recall research: split even a fitting transcript into chunks of this
# many tokens for claim extraction. Off by default — it multiplies requests —
# and only affects the research branch, never the summary.
EXHAUSTIVE_RESEARCH_MODE = env_flag("EXHAUSTIVE_RESEARCH_MODE", default=False)
RESEARCH_CHUNK_TOKENS = env_int("RESEARCH_CHUNK_TOKENS", 12000)

# The base prompts close by saying "output only the summary itself". Appending a
# JSON envelope after that leaves two contradictory answers to "what is the
# response?", so the trailing rule is removed rather than argued with.
TRAILING_OUTPUT_RULE = (
    "\n\nOutput only the summary itself — no preamble, no sign-off, and no "
    "phrases like \"Here is the summary\"."
)
COMBINED_SUFFIX = claims_mod.COMBINED_ENVELOPE


def _build_combined_prompt(compact=False):
    """Summary rules + the JSON envelope that also carries the claims."""
    base = COMPACT_SUMMARY_SYSTEM_PROMPT if compact else SUMMARY_SYSTEM_PROMPT
    return base.replace(TRAILING_OUTPUT_RULE, "") + COMBINED_SUFFIX


def _research_context(context, coverage_status="full"):
    ctx = dict(context or {})
    nt = ctx.get("normalized")
    if nt is not None:
        ctx.setdefault("run_key", research_state.run_key(
            nt.transcript_hash, nt.normalization_version,
            claims_mod.EXTRACTION_PROMPT_VERSION, claims_mod.SCHEMA_VERSION))
    ctx.setdefault("extraction_model", summarizer.LAST_CALL_TELEMETRY.get("model"))
    return ctx


def _failed(status, reason, context=None, **extra):
    out = {"status": status, "failure_reason": reason, "claims": [], "signals": None,
           "warnings": [], "coverage_status": None, "run_key": (context or {}).get("run_key"),
           "telemetry": dict(summarizer.LAST_CALL_TELEMETRY)}
    out.update(extra)
    return out


def build_research(data, context, coverage_status="full", chunk_id=None, standalone=False):
    """
    Validate the claims half of a model response independently of the
    summary. Returns a research result dict:
      status: complete | no_claims_found | needs_review | failed_retryable
      claims: canonical records (may be empty)
      signals: the legacy compatibility object, or None when extraction failed
    A missing/malformed claims array is failed_retryable — never an empty
    success — and an empty array is no_claims_found only after validation AND
    the deterministic suspicious-empty check (claims.suspicious_empty_check)
    found no claim-bearing language. `standalone` says the response came from
    a dedicated claims call rather than the combined summary call, which
    decides whether a suspicious empty result is retried or reviewed.
    """
    ctx = _research_context(context, coverage_status)
    nt = ctx.get("normalized")
    raw = claims_mod.raw_claims_from(data)
    if raw is None:
        return _failed("failed_retryable", "malformed_claims", ctx)
    if nt is None:
        return _failed("failed_retryable", "no_transcript_for_validation", ctx)
    validated, warnings = claims_mod.validate_claims(raw, nt, ctx, coverage_status, chunk_id)
    meta = data.get("extraction_metadata") if isinstance(data, dict) else None
    if isinstance(meta, dict) and isinstance(meta.get("warnings"), list):
        warnings.extend(str(w) for w in meta["warnings"] if w)
    failure_reason = None
    signals_view = claims_mod.claims_to_legacy_signals(validated)
    if not raw:
        # An empty array proves nothing by itself. When the transcript plainly
        # carries forecast or recommendation language about an identifiable
        # asset, "no claims" is not believed: the combined path hands the
        # video to a standalone extraction pass (failed_retryable), and a
        # standalone pass that comes back empty again goes to a human
        # (needs_review). Neither is ever recorded as no_claims_found, and
        # neither writes an empty asset list.
        check = claims_mod.suspicious_empty_check(nt.text)
        if check["suspicious"]:
            status = "needs_review" if standalone else "failed_retryable"
            failure_reason = "suspicious_empty_extraction"
            signals_view = None
            warnings.append("suspicious_empty_extraction: " + "; ".join(check["strong"] or check["moderate"]))
            log_warn(f"Model returned no claims but the transcript looks claim-bearing "
                     f"(score {check['score']}); research marked {status}.")
        else:
            status = "no_claims_found"
    elif all(c.get("review_required") for c in validated):
        status = "needs_review"
    else:
        status = "complete"
    return {
        "status": status, "failure_reason": failure_reason, "claims": validated,
        "signals": signals_view,
        "warnings": warnings, "coverage_status": coverage_status,
        "run_key": ctx.get("run_key"), "extraction_model": ctx.get("extraction_model"),
        "telemetry": dict(summarizer.LAST_CALL_TELEMETRY),
    }


def summarize_with_signals(transcript, title=None, compact=False, channel_name=None, context=None):
    """
    One LLM call producing both the summary and the research claims.

    Returns (summary, research) on success — `summary` may be the
    INSUFFICIENT_TRANSCRIPT sentinel, `research` is a build_research() result
    (or None when the summary was the sentinel / quota) — or None when the
    combined path didn't work, telling the caller to fall back to the
    separate summarize/extract calls. A truncated combined response returns
    (None, research) with research failed_retryable / combined_output_truncated
    and `retry_separately=True`: the caller makes the summary-only call and
    extracts the claims on their own. The complete transcript is sent; a
    transcript too large for every model returns None so the summary takes
    the chunked path and research runs separately. Never raises.
    """
    transcript = (transcript or "").strip()
    if not transcript:
        return None
    ctx = dict(context or {})
    ctx.setdefault("video_title", title)
    ctx.setdefault("channel_name", channel_name)

    text = complete(
        _build_combined_prompt(compact),
        _build_user_message(transcript, title),
        json_mode=True,
        max_tokens=COMBINED_MAX_OUTPUT_TOKENS,
    )
    if text == QUOTA_EXHAUSTED_SENTINEL:
        # Quota is the provider chain's verdict, not a parsing problem:
        # falling back would just burn another request against the same wall.
        return text, None
    if text == INPUT_TOO_LARGE_SENTINEL:
        log_info("Combined request exceeds every model's input capacity; using the chunked paths.")
        return None
    if text == TRUNCATED_SENTINEL:
        # The claims array outgrew the output budget even after escalation.
        # That is a research-output-size problem, not a summary problem: the
        # caller makes a summary-only call (which fits — it is the claims
        # that did not) and delivers it; the claims are extracted separately,
        # in chunks small enough to fit. Deferring the whole video here would
        # let research block delivery. Never recorded as no_claims_found.
        log_warn("Combined summary+claims response was truncated after escalation; "
                 "falling back to a summary-only call and separate claim extraction.")
        return None, _failed("failed_retryable", "combined_output_truncated", _research_context(ctx),
                             retry_separately=True)
    if not text:
        return None

    data = claims_mod.parse_json_object(text)
    if data is None:
        log_warn("Combined summary+claims call returned no JSON; using the separate calls.")
        return None
    summary = data.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        log_warn("Combined call produced no summary; using the separate calls.")
        return None
    summary = summary.strip()
    if summary.upper().startswith(INSUFFICIENT_TRANSCRIPT_SENTINEL):
        return INSUFFICIENT_TRANSCRIPT_SENTINEL, None

    research = build_research(data, ctx, coverage_status="full")
    log_info(
        f"Combined summary+claims call succeeded: research {research['status']}"
        + (f" ({len(research['claims'])} claim(s))" if research["claims"] else "")
        + (f" — {research['failure_reason']}" if research["failure_reason"] else "") + "."
    )
    return summary, research


# --- Standalone / chunked claim extraction -----------------------------------


def _claims_user_message(ctx, transcript, part=""):
    return claims_mod.CLAIMS_USER_TEMPLATE.format(
        channel_name=(ctx.get("channel_name") or "unknown"),
        video_title=(ctx.get("video_title") or "unknown"),
        published_at=(ctx.get("published_at") or "unknown"),
        part=part, transcript=transcript,
    )


def _claims_partial(nt, chunk):
    """Cached raw claims for this chunk, only under the current transcript,
    normalization, chunking, boundaries, prompt, schema and provider policy
    (partial_cache validates the record). None otherwise."""
    import partial_cache
    payload = partial_cache.load("research_claims", nt, chunk, claims_mod.EXTRACTION_PROMPT_VERSION,
                                 claims_mod.SCHEMA_VERSION, _model_policy())
    return payload if isinstance(payload, dict) and isinstance(payload.get("raw_claims"), list) else None


def _save_claims_partial(nt, chunk, raw):
    import partial_cache
    partial_cache.save("research_claims", nt, chunk, {"raw_claims": raw},
                       claims_mod.EXTRACTION_PROMPT_VERSION, claims_mod.SCHEMA_VERSION, _model_policy(),
                       model=summarizer.LAST_CALL_TELEMETRY.get("model"))


def _model_policy():
    import partial_cache
    return partial_cache.model_policy_string(summarizer._provider_configs())


def _research_chunk_tokens():
    """Input tokens per research chunk: the first usable provider's budget
    (less prompt overhead), or the exhaustive-mode size when smaller."""
    providers = [p for p in summarizer._provider_configs() if not summarizer._provider_is_spent(p)]
    budgets = [token_budget.context_budget(p, CLAIMS_MAX_OUTPUT_TOKENS).available_input_tokens
               for p in providers]
    prompt_tokens = token_budget.estimate_tokens(claims_mod.CLAIMS_SYSTEM_PROMPT) + 128
    per = (max(budgets) if budgets else 0) - prompt_tokens
    if EXHAUSTIVE_RESEARCH_MODE:
        per = min(per, RESEARCH_CHUNK_TOKENS) if per > 0 else RESEARCH_CHUNK_TOKENS
    return per


def extract_research(nt, context, prefer_chunked=False):
    """
    Claim extraction from the (complete) normalized transcript, independent
    of the summary: used when the combined path did not yield usable claims
    and by the research retry job. One request when the transcript fits;
    otherwise complete-coverage chunks with per-chunk resume. A single
    request whose OUTPUT is truncated (too many claims for the output cap)
    is retried as smaller chunks, each of which writes fewer claims;
    `prefer_chunked` starts there directly (the combined call already proved
    the whole-transcript output does not fit). Returns a
    build_research()-shaped result whose status is quota_deferred /
    failed_retryable / partial when it could not finish. Never raises.
    """
    ctx = _research_context(context)
    ctx["normalized"] = nt
    if not nt or not nt.text.strip():
        return _failed("failed_final", "empty_transcript", ctx)

    if not EXHAUSTIVE_RESEARCH_MODE and not prefer_chunked:
        text = complete(claims_mod.CLAIMS_SYSTEM_PROMPT, _claims_user_message(ctx, nt.text),
                        json_mode=True, max_tokens=CLAIMS_MAX_OUTPUT_TOKENS)
        if text == QUOTA_EXHAUSTED_SENTINEL:
            return _failed("quota_deferred", "llm_quota", ctx)
        if text == INPUT_TOO_LARGE_SENTINEL:
            return _extract_research_chunked(nt, ctx)
        if text == TRUNCATED_SENTINEL:
            log_warn("Standalone claim extraction was truncated; re-running in smaller chunks.")
            return _extract_research_chunked(nt, ctx, max_chunk_tokens=RESEARCH_CHUNK_TOKENS)
        if not text:
            return _failed("failed_retryable", "no_output", ctx)
        data = claims_mod.parse_json_object(text)
        if data is None:
            return _failed("failed_retryable", "unparseable_json", ctx)
        ctx["extraction_model"] = summarizer.LAST_CALL_TELEMETRY.get("model")
        return build_research(data, ctx, coverage_status="full", standalone=True)
    return _extract_research_chunked(nt, ctx, max_chunk_tokens=RESEARCH_CHUNK_TOKENS if prefer_chunked else None)


def _extract_research_chunked(nt, ctx, max_chunk_tokens=None):
    import transcript_normalize as tn

    per_chunk = _research_chunk_tokens()
    if max_chunk_tokens and per_chunk > 0:
        per_chunk = min(per_chunk, max_chunk_tokens)
    if per_chunk < 500:
        return _failed("quota_deferred", "no_provider_with_input_capacity", ctx)
    chunks = tn.chunk_transcript(nt, per_chunk, token_budget.estimate_tokens)
    ok, problems = tn.validate_coverage(chunks, len(nt.text))
    if not ok:
        return _failed("failed_retryable", f"chunk_coverage:{problems}", ctx)
    all_raw, processed, failed, stop_reason = [], [], [], None
    for chunk in chunks:
        cached = _claims_partial(nt, chunk)
        if cached:
            all_raw.append((chunk, cached["raw_claims"]))
            processed.append(chunk.chunk_id)
            continue
        part = f" (part {chunk.sequence_number} of {len(chunks)}; other parts are handled separately)"
        text = complete(claims_mod.CLAIMS_SYSTEM_PROMPT, _claims_user_message(ctx, chunk.text, part),
                        json_mode=True, max_tokens=CLAIMS_MAX_OUTPUT_TOKENS)
        if text in (QUOTA_EXHAUSTED_SENTINEL, INPUT_TOO_LARGE_SENTINEL, TRUNCATED_SENTINEL) or not text:
            failed.append(chunk.chunk_id)
            stop_reason = text or "no_output"
            break  # the remaining chunks would hit the same wall this run
        data = claims_mod.parse_json_object(text)
        raw = claims_mod.raw_claims_from(data)
        if raw is None:
            failed.append(chunk.chunk_id)
            stop_reason = "malformed_claims"
            continue
        _save_claims_partial(nt, chunk, raw)
        all_raw.append((chunk, raw))
        processed.append(chunk.chunk_id)

    ctx["extraction_model"] = summarizer.LAST_CALL_TELEMETRY.get("model")
    complete_coverage = len(processed) == len(chunks) and not failed
    coverage = "chunked_full" if complete_coverage else "partial"
    validated, warnings = [], []
    for chunk, raw in all_raw:
        v, w = claims_mod.validate_claims(raw, nt, ctx, coverage, chunk.chunk_id)
        validated.extend(v)
        warnings.extend(w)
    claims_mod.dedupe_across_chunks(validated)
    if not complete_coverage:
        # A quota stop is a deferral (nothing is wrong with the video); any
        # other stop leaves the run partial and retryable.
        status = "quota_deferred" if stop_reason == QUOTA_EXHAUSTED_SENTINEL else "partial"
        return {
            "status": status, "failure_reason": stop_reason, "claims": validated,
            "signals": None, "warnings": warnings, "coverage_status": "partial",
            "run_key": ctx.get("run_key"), "extraction_model": ctx.get("extraction_model"),
            "processed_chunk_ids": processed, "failed_chunk_ids": failed,
            "chunks": len(chunks), "telemetry": dict(summarizer.LAST_CALL_TELEMETRY),
        }
    total_raw = sum(len(raw) for _, raw in all_raw)
    failure_reason, signals_view = None, claims_mod.claims_to_legacy_signals(validated)
    if not total_raw:
        check = claims_mod.suspicious_empty_check(nt.text)
        if check["suspicious"]:
            status, failure_reason, signals_view = "needs_review", "suspicious_empty_extraction", None
            warnings.append("suspicious_empty_extraction: " + "; ".join(check["strong"] or check["moderate"]))
        else:
            status = "no_claims_found"
    else:
        status = "needs_review" if all(c.get("review_required") for c in validated) else "complete"
    return {
        "status": status, "failure_reason": failure_reason, "claims": validated,
        "signals": signals_view, "warnings": warnings,
        "coverage_status": "chunked_full", "run_key": ctx.get("run_key"),
        "extraction_model": ctx.get("extraction_model"), "processed_chunk_ids": processed,
        "failed_chunk_ids": [], "chunks": len(chunks), "telemetry": dict(summarizer.LAST_CALL_TELEMETRY),
    }


def extract_signals(summary, video_title=None, channel_name=None):
    """
    Extract structured market signals from a video summary via the shared LLM
    provider chain. Returns the signals dict, or None when the summary is empty,
    providers are unavailable/quota-limited, or the output can't be parsed.
    Best-effort by contract: never raises.
    """
    summary = (summary or "").strip()
    if not summary:
        return None

    user_message = SIGNALS_USER_TEMPLATE.format(
        channel_name=(channel_name or "unknown").strip() or "unknown",
        video_title=(video_title or "unknown").strip() or "unknown",
        summary=summary,
    )
    text = complete(SIGNALS_SYSTEM_PROMPT, user_message, json_mode=True)
    if text == QUOTA_EXHAUSTED_SENTINEL:
        log_warn("Signal extraction skipped: LLM quota exhausted.")
        return None
    if not text or text in (TRUNCATED_SENTINEL, INPUT_TOO_LARGE_SENTINEL):
        log_warn("Signal extraction produced no usable output.")
        return None

    parsed = _parse_signals(text)
    if parsed is not None:
        log_info(
            f"Extracted signals for {len(parsed['assets'])} asset(s), "
            f"market sentiment: {parsed['market_sentiment']}."
        )
    return parsed
