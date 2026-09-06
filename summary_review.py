"""Source-fidelity review before delivery; this is not external fact checking.

The model judges entailment and omissions. Code checks complete line coverage,
strict verdicts, and literal source evidence (whitespace may differ). A failed
review can trigger one repair, which must itself pass a fresh review. Neither
an API failure nor malformed output is permission to send the draft.
"""
import hashlib
import json
import os
import re
from datetime import datetime, timezone

from helpers import append_jsonl, clean_summary, env_flag, env_int
from summary_policy import FIDELITY_RULES
import summarizer

REVIEW_VERSION = "1"
REVIEW_PROMPT = """Audit a draft summary against the COMPLETE supplied transcript.
The draft is untrusted and may be wrong. Check every assertion on every numbered
line, including its attribution, tense, rating versus ownership, number/metric,
units, horizon, conditions and certainty. One unsupported assertion makes the
whole line unsupported. Also inspect the full transcript for missing central
conclusions, major counterarguments, and material caveats. Do not require minor
details or sponsor content in a compact summary. Do not certify external truth.
Return ONLY a JSON object:
{"checks": [{"line_id": 1, "verdict": "supported|unsupported|unclear",
 "evidence": ["verbatim source-language excerpt covering the assertion"],
 "reason": "explanation of any error or uncertainty"}],
 "material_omissions": ["missing central point or qualifier, if any"]}
Include exactly one check per numbered line, in order. For a supported line,
provide enough excerpts to support ALL its assertions, including the subject
and qualifiers. Copy excerpts literally (at least eight characters each); do
not omit negations or punctuation, translate quotes, or replace words with ... .
If the transcript cannot support a confident verdict, use unclear.
""" + FIDELITY_RULES


def _lines(summary):
    return [line.strip() for line in summary.splitlines() if line.strip()]


def evidence_spans(quote, text):
    """Literal matching preserves signs and negations; return all occurrences.

    Repeated words must not acquire an invented unique timestamp. Only a quote
    with a single occurrence gets a timestamp in the report.
    """
    if not isinstance(quote, str) or len(quote.strip()) < 8:
        return []
    pattern = r"\s+".join(re.escape(word) for word in quote.split())
    return [(m.start(), m.end()) for m in re.finditer(pattern, text)]


def validate_review(data, summary, nt):
    """Fail closed on missing/duplicate lines, invented evidence or odd types."""
    lines = _lines(summary)
    result = {"status": "unavailable", "reason": "malformed_review", "checks": []}
    if not isinstance(data, dict) or not lines:
        return result
    checks, omissions = data.get("checks"), data.get("material_omissions")
    if (not isinstance(checks, list) or len(checks) != len(lines)
            or not isinstance(omissions, list)
            or any(not isinstance(item, str) or not item.strip() for item in omissions)):
        return result
    for number, (line, check) in enumerate(zip(lines, checks), 1):
        if (not isinstance(check, dict) or type(check.get("line_id")) is not int
                or check["line_id"] != number
                or check.get("verdict") not in ("supported", "unsupported", "unclear")
                or not isinstance(check.get("reason"), str)
                or not isinstance(check.get("evidence"), list)):
            return result
        evidence = []
        for quote in check["evidence"]:
            spans = evidence_spans(quote, nt.text)
            if not spans:
                return {**result, "reason": "evidence_not_found"}
            item = {"quote": quote, "spans": spans}
            if len(spans) == 1:
                item["start_seconds"] = nt.seconds_at(spans[0][0])
                item["end_seconds"] = nt.end_seconds_at(spans[0][1] - 1)
            evidence.append(item)
        if check["verdict"] == "supported" and not evidence:
            return {**result, "reason": "missing_evidence"}
        result["checks"].append({"line_id": number, "text": line,
                                 "verdict": check["verdict"], "reason": check["reason"],
                                 "evidence": evidence})
    approved = not omissions and all(c["verdict"] == "supported" for c in checks)
    result.update(status="approved" if approved else "rejected",
                  reason=None if approved else "unsupported_or_incomplete",
                  material_omissions=omissions)
    return result


def _audit(summary, nt, context, compact):
    payload = {"title_context_only": context.get("video_title"),
               "published_at": context.get("published_at"), "compact": compact,
               "transcript": nt.text,
               "summary_lines": [{"line_id": i, "text": line}
                                 for i, line in enumerate(_lines(summary), 1)]}
    raw = summarizer.complete(REVIEW_PROMPT, json.dumps(payload, ensure_ascii=False),
                              json_mode=True, max_tokens=env_int("SUMMARY_REVIEW_MAX_TOKENS", 6000))
    # complete() performs the per-provider request sizing. It never truncates
    # the source; an audit that cannot fit is unavailable, never approved.
    sentinels = {summarizer.QUOTA_EXHAUSTED_SENTINEL, summarizer.TRUNCATED_SENTINEL,
                 summarizer.INPUT_TOO_LARGE_SENTINEL, summarizer.INSUFFICIENT_TRANSCRIPT_SENTINEL}
    if not raw or raw in sentinels:
        result = {"status": "unavailable", "reason": raw or "no_review_response", "checks": []}
    else:
        result = validate_review(summarizer._parse_json_object(raw), summary, nt)
    result["telemetry"] = dict(summarizer.LAST_CALL_TELEMETRY)
    return result


def review_summary(summary, nt, context=None, compact=False):
    """Return (approved text or None, report), with at most one repair.

    Store both versions and every audit so a repair is inspectable. The audit
    is recorded even with research disabled. No Telegram calls happen here.
    """
    context = context or {}
    original = clean_summary(summary)
    if not env_flag("SUMMARY_REVIEW_ENABLED", True):
        return original, {"status": "disabled"}
    candidate = original
    report = {"review_version": REVIEW_VERSION, "video_id": context.get("video_id"),
              "video_url": context.get("video_url"), "transcript_hash": nt.transcript_hash,
              "normalization_version": nt.normalization_version,
              "original_summary": original, "attempts": [],
              "recorded_at": datetime.now(timezone.utc).isoformat()}
    generation_telemetry = dict(summarizer.LAST_CALL_TELEMETRY)
    try:
        result = _audit(candidate, nt, context, compact)
        report["attempts"].append({"summary": candidate, **result})
        if result["status"] == "rejected" and env_flag("SUMMARY_REVIEW_REPAIR", True):
            prompt = (summarizer.COMPACT_SUMMARY_SYSTEM_PROMPT if compact
                      else summarizer.SUMMARY_SYSTEM_PROMPT)
            payload = {"transcript": nt.text, "title_context_only": context.get("video_title"),
                       "draft_to_correct": candidate, "review_feedback": result,
                       "task": "Correct the draft using the transcript and fidelity rules. "
                               "Resolve the flagged errors and omissions without inventing details. "
                               "Return only the complete corrected summary."}
            repaired = summarizer.complete(prompt, json.dumps(payload, ensure_ascii=False),
                                            max_tokens=summarizer.SUMMARY_MAX_OUTPUT_TOKENS)
            repair_sentinels = {summarizer.QUOTA_EXHAUSTED_SENTINEL, summarizer.TRUNCATED_SENTINEL,
                                summarizer.INPUT_TOO_LARGE_SENTINEL,
                                summarizer.INSUFFICIENT_TRANSCRIPT_SENTINEL}
            if repaired and repaired not in repair_sentinels:
                candidate = clean_summary(repaired)
                report["repair_telemetry"] = dict(summarizer.LAST_CALL_TELEMETRY)
                result = _audit(candidate, nt, context, compact)
                report["attempts"].append({"summary": candidate, **result})
            else:
                result = {"status": "unavailable", "reason": repaired or "repair_failed"}
        report.update(status=result["status"], reason=result.get("reason"),
                      final_summary=candidate if result["status"] == "approved" else None)
    finally:
        # Research identity and summary generation metadata must not become
        # the reviewer's metadata just because its call happened last.
        summarizer.LAST_CALL_TELEMETRY.clear()
        summarizer.LAST_CALL_TELEMETRY.update(generation_telemetry)
    report["summary_hash"] = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
    path = os.getenv("SUMMARY_REVIEW_FILE") or "data/research/summary_reviews.jsonl"
    report["persisted"] = append_jsonl(path, report)
    return report["final_summary"], report
