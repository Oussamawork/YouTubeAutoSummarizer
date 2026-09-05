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
