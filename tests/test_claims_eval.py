"""Item 8: the evaluation harness (offline replay only — no model is called here)."""
import json
import os

import claims_eval
import transcript_normalize as tn

FIXTURES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evals", "claims")


def test_fixture_set_loads_and_offline_replay_produces_every_metric():
    fixtures = claims_eval.load_fixtures(FIXTURES)
    assert len(fixtures) >= 5
    results, model = claims_eval.run(fixtures, live=False)
    metrics = claims_eval.aggregate(results)
    for key in ("atomic_claim_precision", "atomic_claim_recall", "numerical_value_accuracy",
                "evidence_grounding_accuracy", "attribution_accuracy", "entity_resolution_accuracy",
                "stance_accuracy", "horizon_accuracy", "recommendation_accuracy", "false_no_claims_rate",
                "duplicate_rate"):
        assert metrics[key] is not None, key
    groups = claims_eval.error_groups(results)
    # The stored outputs deliberately include a missed news target and a
    # duplicate, so the report has representative false negatives/positives.
    assert groups["false_no_claims"] and groups["false_negative"]
    report = claims_eval.format_report(metrics, groups, "offline")
    assert "live model quality NOT measured" in report and "Errors by category" in report


def test_matching_is_stable_not_exact_json_equality():
    text = "I expect Nvidia to fall over the next three months. Goldman expects the stock to reach $200."
    nt = tn.normalize_transcript(text, "f")
    expected = [{"evidence_text": "I expect Nvidia to fall over the next three months", "subject": "NVDA",
                 "claim_type": "forecast", "forecast_direction": "decrease", "horizon_bucket": "short"}]
    # Different evidence boundaries, a name instead of a ticker, and extra
    # fields still match; a wrong direction or asset does not.
    predicted = [{"evidence_text": "Nvidia to fall over the next three months", "canonical_entity_name": "Nvidia",
                  "claim_type": "price_target", "stance": "bearish", "horizon_bucket": "short",
                  "evidence_start_character": 9, "evidence_end_character": 51, "extra": 1}]
    matches, fn, fp, dups = claims_eval.match_claims(expected, predicted, nt)
    assert len(matches) == 1 and not fn and not fp
    wrong = [dict(predicted[0], stance="bullish"), dict(predicted[0], canonical_entity_name="AMD")]
    matches, fn, fp, dups = claims_eval.match_claims(expected, wrong, nt)
    assert not matches and len(fn) == 1 and len(fp) == 2
    matches, fn, fp, dups = claims_eval.match_claims(expected, [predicted[0], dict(predicted[0])], nt)
    assert len(matches) == 1 and len(dups) == 1


def test_live_mode_requires_the_explicit_flag_and_credentials(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("CLAIMS_EVAL_LIVE", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    called = []
    import signals
    monkeypatch.setattr(signals, "extract_research", lambda *a, **k: called.append(1))
    report = tmp_path / "r.json"
    assert claims_eval.main(["--fixtures", FIXTURES, "--live", "--report", str(report)]) == 0
    assert not called
    assert "falling back to the offline replay" in capsys.readouterr().err
    assert json.load(open(report))["mode"] == "offline"


# --- Hardening item 8: breakdowns, the benchmark specification and skeletons ---

def test_report_breaks_results_down_and_states_the_benchmark_is_not_met():
    fixtures = claims_eval.load_fixtures(FIXTURES)
    results, model, chunked = claims_eval.run(fixtures, live=False, compare_chunked=True)
    assert chunked == []  # no fixture stores a chunked output; nothing is invented
    by_length = claims_eval.breakdown(results, "length_bucket")
    assert set(by_length) <= {"short", "medium", "long"} and "short" in by_length
    assert claims_eval.breakdown(results, "transcript_source") == {"fixture": claims_eval.aggregate(results)}
    assert set(claims_eval.breakdown(results, "language")) == {"en"}
    spec = claims_eval.benchmark_spec_report(fixtures)
    assert spec["met"] is False
    reqs = spec["requirements"]
    assert reqs["real_videos"] == (False, 0, 20) and reqs["labelled_claims"][0] is False
    assert reqs["sources"][0] is False  # hand-authored fixtures are not real sources
    report = claims_eval.format_report(claims_eval.aggregate(results), claims_eval.error_groups(results),
                                       "offline", None, results, chunked, spec)
    assert "Benchmark specification: NOT MET" in report
    assert "Production extraction quality: NOT ESTABLISHED" in report and "SCORECARD_RANKINGS must stay false" in report
    assert "Results by transcript length:" in report and "Results by language:" in report
    assert "Full-context vs chunked: no chunked results" in report


def test_chunked_comparison_replays_a_stored_chunked_output(tmp_path):
    import shutil
    shutil.copy(os.path.join(FIXTURES, "nvda-two-horizons.json"), tmp_path / "a.json")
    data = json.load(open(tmp_path / "a.json"))
    data["model_output_chunked"] = {"claims": data["model_output"]["claims"][:1]}
    json.dump(data, open(tmp_path / "a.json", "w"))
    fixtures = claims_eval.load_fixtures(str(tmp_path))
    results, _, chunked = claims_eval.run(fixtures, live=False, compare_chunked=True)
    assert len(chunked) == 1 and chunked[0]["predicted"] == 1 and results[0]["predicted"] > 1
    text = claims_eval.format_comparison(results, chunked)
    assert "Full-context vs chunked extraction" in text and "atomic_claim_recall" in text


def test_skeletons_from_real_transcripts_are_not_fixtures_until_labelled(tmp_path, monkeypatch):
    import transcript_store
    monkeypatch.setattr(transcript_store, "TRANSCRIPTS_DIR", str(tmp_path / "transcripts"))
    monkeypatch.setattr(transcript_store, "TRANSCRIPT_INDEX", str(tmp_path / "index.jsonl"))
    transcript_store.store_transcript({"video_id": "real1", "channel_name": "HKCM", "video_title": "T",
                                       "published_at": "2026-07-01T00:00:00+00:00", "duration_seconds": 900},
                                      "Ich erwarte, dass Nvidia 200 Dollar erreicht.", "supadata", "ok", language="de")
    out = tmp_path / "bench"
    assert claims_eval.export_transcript_skeletons(str(out)) == 1
    skeleton = json.load(open(out / "real1.json"))
    assert skeleton["expected_claims"] is None and skeleton["source_video_id"] == "real1"
    assert skeleton["language"] == "de" and skeleton["transcript_source"] == "supadata"
    assert claims_eval.load_fixtures(str(out)) == []          # unlabelled: never scored
    assert claims_eval.export_transcript_skeletons(str(out)) == 0  # never overwritten
    skeleton["expected_claims"] = []
    json.dump(skeleton, open(out / "real1.json", "w"))
    fixtures = claims_eval.load_fixtures(str(out))
    assert len(fixtures) == 1
    spec = claims_eval.benchmark_spec_report(fixtures)
    assert spec["requirements"]["real_videos"][1] == 1 and spec["requirements"]["languages"][1] == {"de": 1}
