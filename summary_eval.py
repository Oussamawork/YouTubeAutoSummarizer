"""Small real-source regression benchmark for the summary review gate.

Default mode validates fixture integrity only. --live requires a configured
provider and SUMMARY_EVAL_LIVE=1; it audits drafts without repairing them or
sending anything. Agent-reviewed labels are not an independent human benchmark.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from helpers import env_flag, write_json_atomic
import summarizer
import summary_review
from transcript_normalize import normalize_transcript
from transcript_store import load_transcript, transcript_hash


def load_cases(directory):
    cases = []
    for path in sorted(Path(directory).glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(case, dict) or case.get("expected_status") not in ("approved", "rejected")
                or not isinstance(case.get("draft"), str) or not case["draft"].strip()
                or not case.get("label_origin") or not case.get("category")):
            raise ValueError(f"Invalid fixture: {path.name}")
        stored = load_transcript(case.get("source_video_id") or "")
        if (not stored or not isinstance(stored.get("raw_transcript"), str)
                or case.get("source_transcript_hash") != stored.get("transcript_hash")
                or transcript_hash(stored["raw_transcript"]) != case["source_transcript_hash"]):
            raise ValueError(f"Missing or changed transcript: {path.name}")
        nt = normalize_transcript(stored["raw_transcript"], case["source_video_id"])
        excerpts = case.get("supporting_excerpts")
        if (not isinstance(excerpts, list) or not excerpts
                or any(not summary_review.evidence_spans(quote, nt.text) for quote in excerpts)):
            raise ValueError(f"Reference evidence missing from transcript: {path.name}")
        cases.append((path.stem, case, stored, nt))
    if not cases:
        raise ValueError("No labelled fixtures found")
    return cases


def evaluate(cases, live=False):
    results = []
    for name, case, stored, nt in cases:
        result = summary_review._audit(case["draft"], nt, stored, True) if live else None
        results.append({"case": name, "category": case["category"],
                        "source_video_id": case["source_video_id"],
                        "source_transcript_hash": case["source_transcript_hash"],
                        "label_origin": case["label_origin"],
                        "expected_status": case["expected_status"], "review": result,
                        "matches_label": result["status"] == case["expected_status"] if live else None})
    report = {"mode": "live_review" if live else "fixture_validation_only",
              "review_version": summary_review.REVIEW_VERSION,
              "production_accuracy": "NOT ESTABLISHED: small agent-labelled, single-channel sample",
              "cases": len(cases), "source_videos": len({c[1]["source_video_id"] for c in cases}),
              "channels": len({c[2].get("channel_name") for c in cases}),
              "categories": dict(Counter(c[1]["category"] for c in cases)), "results": results}
    if live:
        report.update(
            matched=sum(r["matches_label"] for r in results),
            false_approvals=sum(r["expected_status"] == "rejected" and r["review"]["status"] == "approved"
                                for r in results),
            false_rejections=sum(r["expected_status"] == "approved" and r["review"]["status"] == "rejected"
                                 for r in results),
            unavailable=sum(r["review"]["status"] == "unavailable" for r in results))
    return report


def main(argv=None):
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", default="evals/summaries")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args(argv)
    if args.live and (not env_flag("SUMMARY_EVAL_LIVE") or not summarizer._provider_configs()):
        parser.error("--live requires SUMMARY_EVAL_LIVE=1 and an allowed configured model provider")
    try:
        report = evaluate(load_cases(args.fixtures), args.live)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    if args.output:
        write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if args.live and report["matched"] != report["cases"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
