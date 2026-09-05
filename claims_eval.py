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

BREAKDOWNS: every metric is also reported by transcript length bucket
(short < 10k normalized chars, medium < 30k, long), by transcript source
(the fixture's `transcript_source`: supadata / gemini_video /
youtube_transcript_api / fixture), by language (the fixture's `language`,
else detected) and by transcript quality (noisy = caption overlap, duplicate
cues or unintelligible markers were found; clean otherwise). `--compare-
chunked` scores the chunked extraction beside the full-context one (live:
a second `extract_research(..., prefer_chunked=True)` call per fixture;
offline: the fixture's stored `model_output_chunked`, when present).

BENCHMARK SPEC: the report header states whether the fixture set meets the
real-benchmark specification (>= 20 real videos from several channels,
>= 200 labelled atomic claims, questions / third-party views /
retrospectives / recommendations / conditional forecasts / several horizons
/ no-claim videos, short and long, noisy and clean transcripts) and, until
it does AND live mode has run, says that production extraction quality is
NOT established. `--export-transcripts DIR` writes labelling skeletons from
the stored real transcripts (data/transcripts) — skeletons carry
`expected_claims: null` and are ignored by the loader until labelled.

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


LENGTH_BUCKETS = (("short", 10_000), ("medium", 30_000), ("long", None))
BENCHMARK_SPEC = {"real_videos": 20, "channels": 2, "labelled_claims": 200, "no_claim_videos": 1}
BENCHMARK_CATEGORIES = ("question", "third_party_view", "historical_claim", "recommendation",
                        "conditional_forecast", "multiple_horizons")


def load_fixtures(directory):
    """Labelled fixtures only: a skeleton whose `expected_claims` is null
    (written by --export-transcripts, not yet labelled) is skipped, so an
    unlabelled real transcript never counts as a no-claim video."""
    fixtures, skipped = [], 0
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or data.get("transcript") is None:
            continue
        if "expected_claims" in data and data["expected_claims"] is None:
            skipped += 1
            continue
        data.setdefault("fixture_id", os.path.splitext(os.path.basename(path))[0])
        fixtures.append(data)
    if skipped:
        print(f"{skipped} unlabelled skeleton(s) under {directory} ignored (expected_claims is null).",
              file=sys.stderr)
    return fixtures


def length_bucket(char_count):
    for name, limit in LENGTH_BUCKETS:
        if limit is None or char_count < limit:
            return name
    return "long"


def fixture_facets(fixture, nt):
    """The facets a fixture's results are broken down by."""
    from language_detect import language_of
    flags = nt.quality_flags or {}
    noisy = any(flags.get(k) for k in ("caption_overlap_removed", "duplicate_cues_removed", "unintelligible_markers"))
    return {
        "length_bucket": length_bucket(nt.normalized_char_count),
        "transcript_source": fixture.get("transcript_source") or "fixture",
        "language": language_of(nt.text, fixture.get("language")),
        "quality": "noisy" if noisy else "clean",
        "real_video": bool(fixture.get("source_video_id")) and (fixture.get("transcript_source") or "fixture") != "fixture",
    }


def _asset_id(value):
    key = " ".join((value or "").split()).upper()
    ticker = ASSET_ALIASES.get(key) or key
    return TICKER_ALIASES.get(ticker, ticker)


def _value(c, key):
    """A predicted claim's forecast slot wherever non-view normalization put
    it (reported_* / hypothetical_* / displaced_fields); expected claims are
    labelled on the plain names."""
    return cm._slot(c, key) if key in cm.FORECAST_FIELDS else c.get(key)


def _direction(c):
    if _value(c, "stance") in ("bullish", "bearish"):
        return _value(c, "stance")
    d = _value(c, "forecast_direction")
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
            p = _value(predicted, key)
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
            if e.get("horizon_bucket") and e["horizon_bucket"] != _value(p, "horizon_bucket"):
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
            predicted = _value(p, field)
            if e[field] in ("none", "not_applicable"):
                predicted = p.get(field)  # the normalized slot is what a consumer sees
            if e[field] == predicted:
                correct[field] += 1
            else:
                errors[field].append({"expected": e[field], "predicted": predicted,
                                      "evidence": e.get("evidence_text")})
        for key in ("target_value", "target_low", "target_high"):
            if key in e:
                total["numerical_value"] += 1
                value = _value(p, key)
                if value is not None and abs(float(value) - float(e[key])) <= 1e-6:
                    correct["numerical_value"] += 1
                else:
                    errors["numerical_value"].append({"expected": e[key], "predicted": value,
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
        "fixture_id": fixture["fixture_id"], **fixture_facets(fixture, nt),
        "expected": len(expected), "predicted": len(primary),
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


def breakdown(results, facet):
    """{facet value: aggregate(results with that value)}."""
    groups = defaultdict(list)
    for r in results:
        groups[r.get(facet) or "unknown"].append(r)
    return {value: aggregate(rows) for value, rows in sorted(groups.items())}


def benchmark_spec_report(fixtures):
    """
    Whether the fixture set is the real benchmark the extraction quality
    claim needs. Each requirement is (met, actual, required).
    """
    real = [f for f in fixtures if f.get("source_video_id") and (f.get("transcript_source") or "fixture") != "fixture"]
    channels = {f.get("channel_name") for f in real}
    labelled = [c for f in fixtures for c in (f.get("expected_claims") or [])]
    no_claim = [f for f in fixtures if not f.get("expected_claims")]
    types = Counter(c.get("claim_type") for c in labelled)
    categories = {
        "question": types.get("question", 0), "third_party_view": types.get("third_party_view", 0),
        "historical_claim": types.get("historical_claim", 0), "recommendation": types.get("recommendation", 0),
        "conditional_forecast": sum(1 for c in labelled if c.get("condition") or c.get("testability_type") == "conditional_testable"),
        "multiple_horizons": len({c.get("horizon_bucket") for c in labelled if c.get("horizon_bucket")}),
    }
    lengths, sources, languages, qualities = Counter(), Counter(), Counter(), Counter()
    for f in fixtures:
        facets = fixture_facets(f, tn.normalize_transcript(f["transcript"], f.get("fixture_id", "f")))
        lengths[facets["length_bucket"]] += 1
        sources[facets["transcript_source"]] += 1
        languages[facets["language"]] += 1
        qualities[facets["quality"]] += 1
    rows = {
        "real_videos": (len(real) >= BENCHMARK_SPEC["real_videos"], len(real), BENCHMARK_SPEC["real_videos"]),
        "channels": (len(channels) >= BENCHMARK_SPEC["channels"], len(channels), BENCHMARK_SPEC["channels"]),
        "labelled_claims": (len(labelled) >= BENCHMARK_SPEC["labelled_claims"], len(labelled),
                            BENCHMARK_SPEC["labelled_claims"]),
        "no_claim_videos": (len(no_claim) >= 1, len(no_claim), 1),
        "short_and_long": (bool(lengths.get("short")) and bool(lengths.get("long")), dict(lengths), "both"),
        "noisy_and_clean": (bool(qualities.get("noisy")) and bool(qualities.get("clean")), dict(qualities), "both"),
        "sources": (len(sources) >= 1 and "fixture" not in sources, dict(sources), "real sources only"),
        "languages": (True, dict(languages), "as present in the channel list"),
    }
    for name in BENCHMARK_CATEGORIES:
        need = 2 if name == "multiple_horizons" else 1
        rows[f"category:{name}"] = (categories[name] >= need, categories[name], need)
    return {"met": all(v[0] for v in rows.values()), "requirements": rows}


def format_benchmark_spec(spec, mode):
    lines = ["Benchmark specification: " + ("MET" if spec["met"] else "NOT MET")]
    for name, (ok, actual, required) in spec["requirements"].items():
        lines.append(f"  [{'x' if ok else ' '}] {name:<28} {actual} (required: {required})")
    if spec["met"] and mode == "live":
        lines.append("Production extraction quality: measured on the real benchmark above (review the errors before relying on it).")
    else:
        lines.append("Production extraction quality: NOT ESTABLISHED — "
                     + ("the benchmark specification is not met" if not spec["met"] else "live mode has not run on it")
                     + "; SCORECARD_RANKINGS must stay false.")
    return "\n".join(lines)


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


METRIC_KEYS = ("atomic_claim_precision", "atomic_claim_recall", "numerical_value_accuracy",
               "evidence_grounding_accuracy", "attribution_accuracy", "entity_resolution_accuracy",
               "stance_accuracy", "horizon_accuracy", "recommendation_accuracy", "false_no_claims_rate",
               "duplicate_rate")


def _metric_lines(metrics, indent="  "):
    return [f"{indent}{key:<32} {_fmt(metrics[key])}" for key in METRIC_KEYS]


def format_breakdowns(results):
    lines = []
    for facet, label in (("length_bucket", "by transcript length"), ("transcript_source", "by transcript source"),
                         ("language", "by language"), ("quality", "by transcript quality")):
        lines.append(f"Results {label}:")
        for value, metrics in breakdown(results, facet).items():
            lines.append(f"  {value}: fixtures {metrics['fixtures']}, precision {_fmt(metrics['atomic_claim_precision'])}, "
                         f"recall {_fmt(metrics['atomic_claim_recall'])}, false no-claims "
                         f"{_fmt(metrics['false_no_claims_rate'])}, duplicates {_fmt(metrics['duplicate_rate'])}")
    return "\n".join(lines)


def format_comparison(full_results, chunked_results):
    """Full-context versus chunked extraction, side by side."""
    if not chunked_results:
        return "Full-context vs chunked: no chunked results (run with --compare-chunked; offline needs model_output_chunked)."
    full, chunked = aggregate(full_results), aggregate(chunked_results)
    lines = [f"Full-context vs chunked extraction ({full['fixtures']} vs {chunked['fixtures']} fixtures):",
             f"  {'metric':<32} {'full':>8} {'chunked':>8}"]
    for key in METRIC_KEYS:
        lines.append(f"  {key:<32} {_fmt(full[key]):>8} {_fmt(chunked[key]):>8}")
    return "\n".join(lines)


def format_report(metrics, groups, mode, model=None, results=None, chunked_results=None, spec=None):
    lines = [f"Claims extraction evaluation — mode: {mode}"
             + (f" (model {model})" if model else "")
             + ("" if mode == "live" else "; offline replay of STORED model outputs — live model quality NOT measured"),
             f"Fixtures {metrics['fixtures']} · expected {metrics['counts']['expected']} · predicted "
             f"{metrics['counts']['predicted']} · matched {metrics['counts']['matched']}"]
    lines.extend(_metric_lines(metrics))
    if spec is not None:
        lines.append("")
        lines.append(format_benchmark_spec(spec, mode))
    if results:
        lines.append("")
        lines.append(format_breakdowns(results))
        lines.append("")
        lines.append(format_comparison(results, chunked_results or []))
    lines.append("")
    lines.append("Errors by category:")
    if not groups:
        lines.append("  none")
    for category, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"  {category}: {len(items)}")
        for item in items[:MAX_EXAMPLES]:
            lines.append("    - " + json.dumps(item, ensure_ascii=False)[:220])
    return "\n".join(lines)


def run(fixtures, live=False, max_fixtures=None, compare_chunked=False):
    """(results, model) — and, when `compare_chunked`, (results, model,
    chunked_results) with the chunked extraction scored beside the
    full-context one."""
    results, chunked_results, model = [], [], None
    for fixture in fixtures[:max_fixtures] if max_fixtures else fixtures:
        nt = tn.normalize_transcript(fixture["transcript"], fixture["fixture_id"])
        ctx = {"video_id": fixture["fixture_id"], "channel_id": "eval", "channel_name": fixture.get("channel_name"),
               "video_title": fixture.get("video_title"), "published_at": fixture.get("published_at"),
               "transcript_source": fixture.get("transcript_source") or "fixture",
               "transcript_language": fixture.get("language"), "run_key": "eval", "extraction_model": "stored"}
        if live:
            import signals
            ctx["normalized"] = nt
            research = signals.extract_research(nt, ctx)
            predicted = research.get("claims") or []
            model = research.get("extraction_model") or model
            if research.get("status") in ("quota_deferred", "failed_retryable", "failed_final"):
                print(f"[{fixture['fixture_id']}] extraction {research['status']}: {research.get('failure_reason')}",
                      file=sys.stderr)
            if compare_chunked:
                chunked = signals.extract_research(nt, ctx, prefer_chunked=True)
                chunked_results.append(evaluate_fixture(fixture, chunked.get("claims") or [], nt))
        else:
            raw = ((fixture.get("model_output") or {}).get("claims")) or []
            predicted, _ = cm.validate_claims(raw, nt, ctx)
            if compare_chunked and isinstance(fixture.get("model_output_chunked"), dict):
                raw_chunked = fixture["model_output_chunked"].get("claims") or []
                predicted_chunked, _ = cm.validate_claims(raw_chunked, nt, ctx, "chunked_full")
                chunked_results.append(evaluate_fixture(fixture, predicted_chunked, nt))
        results.append(evaluate_fixture(fixture, predicted, nt))
    if compare_chunked:
        return results, model, chunked_results
    return results, model


def export_transcript_skeletons(directory, transcripts_dir=None, index_path=None, max_videos=None):
    """
    Write one labelling skeleton per stored real transcript into
    `directory`: the raw transcript, its metadata (channel, title, published
    time, source, language, duration, video id) and `expected_claims: null`.
    Skeletons are ignored by load_fixtures until a person replaces null with
    the labelled claims. Returns the number written; never overwrites a file.
    """
    import transcript_store
    written = 0
    os.makedirs(directory, exist_ok=True)
    index = index_path or transcript_store.TRANSCRIPT_INDEX
    rows = []
    if os.path.exists(index):
        with open(index, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    for row in rows[:max_videos] if max_videos else rows:
        vid = row.get("video_id")
        stored = transcript_store.load_transcript(vid, directory=transcripts_dir) if vid else None
        if not stored or not stored.get("raw_transcript"):
            continue
        path = os.path.join(directory, f"{vid}.json")
        if os.path.exists(path):
            continue
        skeleton = {
            "fixture_id": vid, "source_video_id": vid, "channel_name": stored.get("channel_name"),
            "video_title": stored.get("video_title"), "published_at": stored.get("published_at"),
            "transcript_source": stored.get("transcript_source"), "language": stored.get("transcript_language"),
            "duration_seconds": stored.get("duration_seconds"), "transcript": stored["raw_transcript"],
            "expected_claims": None, "model_output": None,
            "notes": "Label every atomic claim by hand, then replace expected_claims with the list "
                     "(see evals/claims/README.md). model_output may hold one raw model response.",
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(skeleton, f, indent=1, ensure_ascii=False)
        written += 1
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate claim extraction against labelled fixtures")
    parser.add_argument("--fixtures", default="evals/claims")
    parser.add_argument("--live", action="store_true",
                        help="call the real model with the current prompt (also needs CLAIMS_EVAL_LIVE=1)")
    parser.add_argument("--report", help="write the metrics and error groups as JSON here")
    parser.add_argument("--max-fixtures", type=int, default=None)
    parser.add_argument("--compare-chunked", action="store_true",
                        help="also score the chunked extraction (live: a second call per fixture; "
                             "offline: the fixture's model_output_chunked)")
    parser.add_argument("--export-transcripts", metavar="DIR",
                        help="write labelling skeletons for the stored real transcripts into DIR and exit")
    args = parser.parse_args(argv)

    if args.export_transcripts:
        n = export_transcript_skeletons(args.export_transcripts)
        print(f"{n} skeleton(s) written to {args.export_transcripts}; label expected_claims before evaluating.")
        return 0

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
    results, model, chunked_results = run(fixtures, live=live, max_fixtures=args.max_fixtures, compare_chunked=True) \
        if args.compare_chunked else (*run(fixtures, live=live, max_fixtures=args.max_fixtures), [])
    metrics, groups = aggregate(results), error_groups(results)
    spec = benchmark_spec_report(fixtures)
    print(format_report(metrics, groups, "live" if live else "offline", model, results, chunked_results, spec))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({"mode": "live" if live else "offline", "model": model, "metrics": metrics,
                       "benchmark_spec": spec, "quality_established": spec["met"] and live,
                       "by_length": breakdown(results, "length_bucket"),
                       "by_source": breakdown(results, "transcript_source"),
                       "by_language": breakdown(results, "language"),
                       "by_quality": breakdown(results, "quality"),
                       "chunked_metrics": aggregate(chunked_results) if chunked_results else None,
                       "errors": groups}, f, indent=2, ensure_ascii=False, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
