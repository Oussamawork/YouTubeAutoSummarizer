"""Item 5: conditional forecasts are typed, evaluated and reported separately."""
from datetime import date

import claims as cm
import canonical_claims as cc
import research_analytics as ra
import research_state
import transcript_normalize as tn


def _ctx():
    return {"video_id": "v1", "channel_id": "c1", "channel_name": "Chan", "video_title": "T",
            "published_at": "2026-07-01T00:00:00+00:00", "transcript_source": "supadata", "run_key": "rk"}


def _forecast(text, condition, subject="Nvidia", **over):
    raw = {"attribution_type": "speaker_personal_view", "claim_type": "forecast", "is_forward_looking": True,
           "subject_mention": subject, "asset_type": "stock", "stance": "bullish", "forecast_metric": "price",
           "forecast_direction": "increase", "horizon_original": "by year end", "condition": condition,
           "certainty_level": "medium", "evidence_text": text, "extraction_confidence": "high"}
    raw.update(over)
    return cm.validate_claims([raw], tn.normalize_transcript(text, "v1"), _ctx())[0][0]


def test_objectively_measurable_condition_is_conditional_testable():
    c = _forecast("If Nvidia holds above $150 it goes to 200 by year end", "if Nvidia holds above $150")
    assert c["testability_type"] == "conditional_testable" and c["testable"] is True
    assert c["condition_text"] == "if Nvidia holds above $150"
    assert c["condition_observable"] is True and c["condition_kind"] == "price_level"
    assert c["condition_data_source"] == "daily_close"
    assert c["condition_status"] == "not_evaluated" and c["condition_evaluation_date"] is None
    assert c["testability_issues"] == []


def test_subjective_condition_is_not_testable_for_that_specific_reason():
    c = _forecast("If management executes well Nvidia rallies by year end", "if management executes well")
    assert c["testability_type"] == "not_testable" and c["testable"] is False
    assert c["testability_issues"] == ["condition_not_objectively_observable"]
    assert c["condition_observable"] is False and c["condition_kind"] == "subjective"
    assert c["condition_data_source"] is None
    # The generic "it has a condition" verdict is gone.
    assert "conditional_outcome_not_observable" not in c["testability_issues"]


def test_condition_met_makes_the_forecast_scorable():
    c = _forecast("If the Fed cuts in September Nvidia rallies by year end", "if the Fed cuts in September",
                  horizon_original="by end of the year")
    cm.record_condition_outcome(c, "met", "2026-09-17", "FOMC cut 25bp on 2026-09-17", "fomc_decisions")
    assert c["condition_status"] == "met" and c["condition_data_source"] == "fomc_decisions"
    rows = ra.conditional_forecast_report([c])
    assert rows[0]["status"] == "met" and rows[0]["observable"] is True
    text = ra.format_conditional_report(rows)
    assert "met 1" in text and "fomc_decisions" in text


def test_condition_not_met_keeps_the_forecast_out_of_scoring():
    c = _forecast("If the Fed cuts in September Nvidia rallies by year end", "if the Fed cuts in September")
    cm.record_condition_outcome(c, "not_met", "2026-09-17", "FOMC held rates", "fomc_decisions")
    assert c["condition_status"] == "not_met"
    assert ra.conditional_forecast_report([c])[0]["status"] == "not_met"
    c.update(forecast_end_date="2026-07-15", ticker="NVDA")
    sc = ra.scorecard([c], date(2026, 8, 1), lambda *a: {}, min_sample=1, include_conditional=True)
    assert sc["_excluded"] == {"condition_not_met": 1}


def test_condition_unknown_is_recorded_and_reported_as_unknown():
    c = _forecast("If Nvidia holds above $150 it goes to 200 by year end", "if Nvidia holds above $150")
    cm.record_condition_outcome(c, "bogus-status")
    assert c["condition_status"] == "unknown"
    c2 = _forecast("If Nvidia holds above $150 it goes to 200 by year end", "if Nvidia holds above $150")
    cm.record_condition_outcome(c2, "unknown", evidence="no price data for the window")
    assert [r["status"] for r in ra.conditional_forecast_report([c, c2])] == ["unknown", "unknown"]


def test_condition_evaluations_are_persisted_and_overlaid_by_the_loader(tmp_path, monkeypatch):
    monkeypatch.setattr(research_state, "RESEARCH_DIR", str(tmp_path))
    c = _forecast("If the Fed cuts in September Nvidia rallies by year end", "if the Fed cuts in September")
    state = research_state.load_state()
    research_state.store_claims([c], state, "v1", "rk")
    assert cc.load_canonical_claims(state)[0]["condition_status"] == "not_evaluated"
    research_state.record_condition_evaluation(c["claim_id"], "met", "2026-09-17", "cut 25bp", "fomc_decisions")
    loaded = cc.load_canonical_claims(state)[0]
    assert loaded["condition_status"] == "met" and loaded["condition_evaluation_date"] == "2026-09-17"
    assert loaded["condition_evidence"] == "cut 25bp"
    # The latest evaluation wins.
    research_state.record_condition_evaluation(c["claim_id"], "partially_met")
    assert cc.load_canonical_claims(state)[0]["condition_status"] == "partially_met"


def test_observability_rules_cover_the_documented_kinds():
    assert cm.condition_observability("if CPI comes in above 3 percent")[1] == "macro_release"
    assert cm.condition_observability("if earnings beat")[1] == "earnings_release"
    assert cm.condition_observability("if the FDA approves the drug")[1] == "scheduled_event"
    assert cm.condition_observability("if sentiment improves") == (False, "subjective", None)
    assert cm.condition_observability("") == (False, None, None)
