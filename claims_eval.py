"""
Extraction-quality evaluation harness for the atomic-claims extractor.

    python claims_eval.py --fixtures evals/claims            # offline: stored model outputs
    python claims_eval.py --fixtures evals/claims --live     # live: current prompt, real model
    python claims_eval.py --fixtures evals/claims --report out.json

This is NOT part of the ordinary test run. Offline mode replays the model
output stored in each fixture through the real deterministic pipeline
(normalization -> validate_claims) and scores the result against the
hand-labelled expected claims; it measures the software, not the model.
Live mode calls the configured provider chain with the current extraction
prompt and consumes real quota: it runs only when BOTH `--live` is passed
AND the CLAIMS_EVAL_LIVE=1 environment variable is set, and it says so. A
report from offline mode never claims live model quality was measured.

FIXTURE FORMAT (one JSON file per fixture under the fixtures directory):
{
  "fixture_id": "nvda-two-horizons",
  "channel_name": "...", "video_title": "...", "published_at": "2026-07-01T14:00:00+00:00",
  "transcript": "<raw transcript text, caption timestamps allowed>",
  "expected_claims": [
    {"evidence_text": "<verbatim excerpt>", "subject": "NVDA" | "<name>",
     "claim_type": "forecast", "attribution_type": "speaker_personal_view",
     "stance": "bearish", "forecast_direction": "decrease", "horizon_bucket": "short",
     "target_value": 200, "recommendation_action": "none", "ticker": "NVDA",
     "portfolio_disclosure": "not_stated", "host_position": "not_applicable",
     "entity_resolution_method": "explicit_mention"}
  ],
  "model_output": {"claims": [<raw claim records exactly as the model returns them>]}
}
Only `evidence_text` and `subject` are required on an expected claim; every
other field present is scored. `model_output` is what offline mode replays;
`--live` ignores it. To expand the set: copy a real transcript segment from
data/transcripts/ (see transcript_store), label every atomic claim by hand,
and store one real model response as `model_output` so the offline replay
stays representative.

MATCHING (never exact-JSON equality): a predicted claim matches an expected
one when their evidence spans overlap in the transcript (or sit in the same
segment), the resolved subject is the same asset, the claim type is in the
same family (forecast ~ price_target), and the direction, target and horizon
bucket agree where the expected claim states them. Each expected claim is
matched at most once; extra matches to the same expected claim count as
duplicates.
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import claims as cm
import transcript_normalize as tn
from signals_data import ASSET_ALIASES, TICKER_ALIASES

TYPE_FAMILY = {"forecast": "forecast", "price_target": "forecast", "stance": "view", "valuation_view": "view",
               "opinion": "view", "recommendation": "recommendation", "third_party_view": "third_party",
               "portfolio_disclosure": "disclosure", "question": "question", "historical_claim": "history",
               "news_report": "third_party", "fact": "fact"}
SCORED_FIELDS = ("attribution_type", "stance", "horizon_bucket", "recommendation_action", "ticker",
                 "portfolio_disclosure", "host_position", "entity_resolution_method", "claim_type",
                 "forecast_direction", "testability_type", "condition_status")
MAX_EXAMPLES = 5


def load_fixtures(directory):
    fixtures = []
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get("transcript") is not None:
            data.setdefault("fixture_id", os.path.splitext(os.path.basename(path))[0])
            fixtures.append(data)
    return fixtures


def _asset_id(value):
    key = " ".join((value or "").split()).upper()
    ticker = ASSET_ALIASES.get(key) or key
    return TICKER_ALIASES.get(ticker, ticker)


def _direction(c):
    if c.get("stance") in ("bullish", "bearish"):
        return c["stance"]
    d = c.get("forecast_direction")
    if d in ("increase", "recover", "outperform"):
        return "bullish"
    if d in ("decrease", "decline", "underperform"):
        return "bearish"
    return None


def _span(claim, nt):
    if claim.get("evidence_start_character") is not None:
        return claim["evidence_start_character"], claim["evidence_end_character"]
    return cm.locate_evidence(claim.get("evidence_text") or "", nt.text)


def _overlaps(a, b):
    return a is not None and b is not None and a[0] < b[1] and b[0] < a[1]


def _same_segment(a, b, nt):
    if a is None or b is None:
        return False
    sa, sb = nt.segment_at(a[0]), nt.segment_at(b[0])
    return sa is not None and sb is not None and sa.segment_id == sb.segment_id


def _numbers_agree(expected, predicted):
    for key in ("target_value", "target_low", "target_high"):
        if key in expected and expected[key] is not None:
            p = predicted.get(key)
            if p is None or abs(float(p) - float(expected[key])) > 1e-6:
                return False
    return True


def match_claims(expected_claims, predicted_claims, nt):
    """(matches [(expected, predicted)], unmatched_expected, unmatched_predicted,
    duplicates [predicted])."""
    remaining = list(range(len(expected_claims)))
    matches, duplicates, unmatched_pred = [], [], []
    matched_expected = {}
    for p in predicted_claims:
        p_span, p_subject = _span(p, nt), _asset_id(p.get("ticker") or p.get("canonical_entity_name")
                                                     or p.get("subject_mention"))
        best = None
        for i, e in enumerate(expected_claims):
            e_span = cm.locate_evidence(e.get("evidence_text") or "", nt.text)
            if not (_overlaps(e_span, p_span) or _same_segment(e_span, p_span, nt)):
                continue
            if _asset_id(e.get("subject")) != p_subject:
                continue
            if e.get("claim_type") and TYPE_FAMILY.get(e["claim_type"]) != TYPE_FAMILY.get(p.get("claim_type")):
                continue
            if (e.get("stance") in ("bullish", "bearish") or e.get("forecast_direction")) and \
                    _direction(e) is not None and _direction(e) != _direction(p):
                continue
            if not _numbers_agree(e, p):
                continue
            if e.get("horizon_bucket") and e["horizon_bucket"] != p.get("horizon_bucket"):
                continue
            best = i
            break
        if best is None:
            unmatched_pred.append(p)
        elif best in matched_expected:
            duplicates.append(p)
        else:
            matched_expected[best] = p
            matches.append((expected_claims[best], p))
            remaining.remove(best)
    return matches, [expected_claims[i] for i in remaining], unmatched_pred, duplicates


def _field_accuracy(matches):
    correct, total = Counter(), Counter()
    errors = defaultdict(list)
    for e, p in matches:
        for field in SCORED_FIELDS:
            if field not in e:
                continue
            total[field] += 1
            if e[field] == p.get(field):
                correct[field] += 1
            else:
                errors[field].append({"expected": e[field], "predicted": p.get(field),
                                      "evidence": e.get("evidence_text")})
        for key in ("target_value", "target_low", "target_high"):
            if key in e:
                total["numerical_value"] += 1
                if p.get(key) is not None and abs(float(p[key]) - float(e[key])) <= 1e-6:
                    correct["numerical_value"] += 1
                else:
                    errors["numerical_value"].append({"expected": e[key], "predicted": p.get(key),
                                                      "evidence": e.get("evidence_text")})
    return correct, total, errors


def evaluate_fixture(fixture, predicted, nt):
    expected = fixture.get("expected_claims") or []
    primary = [p for p in predicted if not p.get("repeat_of_claim_id")]
    matches, fn, fp, dups = match_claims(expected, primary, nt)
    correct, total, errors = _field_accuracy(matches)
    grounded = sum(1 for _, p in matches if p.get("evidence_start_character") is not None
                   and _overlaps(_span(p, nt), cm.locate_evidence(_["evidence_text"], nt.text)))
    located = sum(1 for p in primary if p.get("evidence_start_character") is not None)
    return {
        "fixture_id": fixture["fixture_id"], "expected": len(expected), "predicted": len(primary),
        "repeats": len(predicted) - len(primary), "matched": len(matches),
        "false_positives": fp, "false_negatives": fn, "duplicates": dups,
        "field_correct": correct, "field_total": total, "field_errors": errors,
        "evidence_grounded": grounded, "evidence_located": located,
        "false_no_claims": bool(expected) and not primary,
    }


def aggregate(results):
    tp = sum(r["matched"] for r in results)
    predicted = sum(r["predicted"] for r in results)
    expected = sum(r["expected"] for r in results)
    dups = sum(len(r["duplicates"]) + r["repeats"] for r in results)
    correct, total = Counter(), Counter()
    for r in results:
        correct.update(r["field_correct"])
        total.update(r["field_total"])

    def rate(n, d):
        return None if not d else n / d

    metrics = {
        "fixtures": len(results),
        "atomic_claim_precision": rate(tp, predicted),
        "atomic_claim_recall": rate(tp, expected),
        "numerical_value_accuracy": rate(correct["numerical_value"], total["numerical_value"]),
        "evidence_grounding_accuracy": rate(sum(r["evidence_grounded"] for r in results), tp),
        "attribution_accuracy": rate(correct["attribution_type"], total["attribution_type"]),
        "entity_resolution_accuracy": rate(correct["ticker"], total["ticker"]),
        "stance_accuracy": rate(correct["stance"], total["stance"]),
        "horizon_accuracy": rate(correct["horizon_bucket"], total["horizon_bucket"]),
        "recommendation_accuracy": rate(correct["recommendation_action"], total["recommendation_action"]),
        "false_no_claims_rate": rate(sum(1 for r in results if r["false_no_claims"]),
                                     sum(1 for r in results if r["expected"])),
        "duplicate_rate": rate(dups, predicted + dups),
        "counts": {"expected": expected, "predicted": predicted, "matched": tp, "duplicates": dups},
    }
    return metrics


def error_groups(results):
    groups = defaultdict(list)
    for r in results:
        for p in r["false_positives"]:
            groups["false_positive"].append({"fixture": r["fixture_id"], "subject": p.get("subject_mention"),
                                             "claim_type": p.get("claim_type"), "evidence": p.get("evidence_text")})
        for e in r["false_negatives"]:
            groups["false_negative"].append({"fixture": r["fixture_id"], "subject": e.get("subject"),
                                             "claim_type": e.get("claim_type"), "evidence": e.get("evidence_text")})
        for p in r["duplicates"]:
            groups["duplicate"].append({"fixture": r["fixture_id"], "subject": p.get("subject_mention"),
                                        "evidence": p.get("evidence_text")})
        if r["false_no_claims"]:
            groups["false_no_claims"].append({"fixture": r["fixture_id"], "expected": r["expected"]})
        for field, errs in r["field_errors"].items():
            for err in errs:
                groups[f"field:{field}"].append(dict(err, fixture=r["fixture_id"]))
    return dict(groups)


def _fmt(value):
    return "n/a" if value is None else f"{100 * value:.0f}%"


def format_report(metrics, groups, mode, model=None):
    lines = [f"Claims extraction evaluation — mode: {mode}"
             + (f" (model {model})" if model else "")
             + ("" if mode == "live" else "; offline replay of STORED model outputs — live model quality NOT measured"),
             f"Fixtures {metrics['fixtures']} · expected {metrics['counts']['expected']} · predicted "
             f"{metrics['counts']['predicted']} · matched {metrics['counts']['matched']}"]
    for key in ("atomic_claim_precision", "atomic_claim_recall", "numerical_value_accuracy",
                "evidence_grounding_accuracy", "attribution_accuracy", "entity_resolution_accuracy",
                "stance_accuracy", "horizon_accuracy", "recommendation_accuracy", "false_no_claims_rate",
                "duplicate_rate"):
        lines.append(f"  {key:<32} {_fmt(metrics[key])}")
    lines.append("")
    lines.append("Errors by category:")
    if not groups:
        lines.append("  none")
    for category, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"  {category}: {len(items)}")
        for item in items[:MAX_EXAMPLES]:
            lines.append("    - " + json.dumps(item, ensure_ascii=False)[:220])
    return "\n".join(lines)


def run(fixtures, live=False, max_fixtures=None):
    results, model = [], None
    for fixture in fixtures[:max_fixtures] if max_fixtures else fixtures:
        nt = tn.normalize_transcript(fixture["transcript"], fixture["fixture_id"])
        ctx = {"video_id": fixture["fixture_id"], "channel_id": "eval", "channel_name": fixture.get("channel_name"),
               "video_title": fixture.get("video_title"), "published_at": fixture.get("published_at"),
               "transcript_source": "fixture", "run_key": "eval", "extraction_model": "stored"}
        if live:
            import signals
            ctx["normalized"] = nt
            research = signals.extract_research(nt, ctx)
            predicted = research.get("claims") or []
            model = research.get("extraction_model") or model
            if research.get("status") in ("quota_deferred", "failed_retryable", "failed_final"):
                print(f"[{fixture['fixture_id']}] extraction {research['status']}: {research.get('failure_reason')}",
                      file=sys.stderr)
        else:
            raw = ((fixture.get("model_output") or {}).get("claims")) or []
            predicted, _ = cm.validate_claims(raw, nt, ctx)
        results.append(evaluate_fixture(fixture, predicted, nt))
    return results, model


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate claim extraction against labelled fixtures")
    parser.add_argument("--fixtures", default="evals/claims")
    parser.add_argument("--live", action="store_true",
                        help="call the real model with the current prompt (also needs CLAIMS_EVAL_LIVE=1)")
    parser.add_argument("--report", help="write the metrics and error groups as JSON here")
    parser.add_argument("--max-fixtures", type=int, default=None)
    args = parser.parse_args(argv)

    fixtures = load_fixtures(args.fixtures)
    if not fixtures:
        print(f"No fixtures under {args.fixtures}.", file=sys.stderr)
        return 2
    live = False
    if args.live:
        has_creds = any(os.getenv(k) for k in ("GEMINI_API_KEY", "GROQ_API_KEY", "LLM_API_KEY"))
        if os.getenv("CLAIMS_EVAL_LIVE") == "1" and has_creds:
            live = True
            print("LIVE evaluation: this consumes real model quota.", file=sys.stderr)
        else:
            print("Live evaluation requested but CLAIMS_EVAL_LIVE=1 and provider credentials are both "
                  "required; falling back to the offline replay.", file=sys.stderr)
    results, model = run(fixtures, live=live, max_fixtures=args.max_fixtures)
    metrics, groups = aggregate(results), error_groups(results)
    print(format_report(metrics, groups, "live" if live else "offline", model))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({"mode": "live" if live else "offline", "model": model, "metrics": metrics,
                       "errors": groups}, f, indent=2, ensure_ascii=False, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
