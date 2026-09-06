import json
from pathlib import Path

import pytest

import summary_eval
import summary_review


FIXTURES = Path(__file__).resolve().parents[1] / "evals" / "summaries"


def real_cases(monkeypatch):
    # The general fixture isolates production transcripts; this read-only
    # benchmark test explicitly loads the four checked-in source artifacts.
    from transcript_store import load_transcript
    root = Path(__file__).resolve().parents[1] / "data" / "transcripts"
    monkeypatch.setattr(summary_eval, "load_transcript", lambda video: load_transcript(video, directory=str(root)))
    return summary_eval.load_cases(FIXTURES)


def test_real_fixtures_have_matching_hashes_and_literal_evidence(monkeypatch):
    cases = real_cases(monkeypatch)
    assert len(cases) == 9
    report = summary_eval.evaluate(cases)
    assert report["source_videos"] == 4 and report["channels"] == 1
    assert report["mode"] == "fixture_validation_only"
    assert "NOT ESTABLISHED" in report["production_accuracy"]
    assert "matched" not in report
    assert all(r["matches_label"] is None for r in report["results"])


def test_changed_source_hash_rejects_fixture(monkeypatch, tmp_path):
    case = json.loads(next(FIXTURES.glob("*.json")).read_text())
    case["source_transcript_hash"] = "changed"
    (tmp_path / "case.json").write_text(json.dumps(case))
    real_cases(monkeypatch)
    with pytest.raises(ValueError, match="changed transcript"):
        summary_eval.load_cases(tmp_path)


def test_live_metrics_distinguish_false_approval_from_unavailable(monkeypatch):
    cases = real_cases(monkeypatch)
    good = next(c for c in cases if c[1]["expected_status"] == "approved")
    bad = next(c for c in cases if c[1]["expected_status"] == "rejected")
    replies = iter([{"status": "unavailable"}, {"status": "approved"}])
    monkeypatch.setattr(summary_review, "_audit", lambda *a: next(replies))
    report = summary_eval.evaluate([good, bad], live=True)
    assert report["matched"] == 0
    assert report["unavailable"] == 1
    assert report["false_approvals"] == 1
    assert report["false_rejections"] == 0


def test_live_without_opt_in_fails_instead_of_silently_replaying(monkeypatch):
    monkeypatch.setenv("SUMMARY_EVAL_LIVE", "false")
    with pytest.raises(SystemExit) as exc:
        summary_eval.main(["--live"])
    assert exc.value.code == 2
