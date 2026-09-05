"""Item 3: production analytics read canonical claims through one loader,
never the reduced legacy signal shape."""
import inspect
import json
from datetime import date

import canonical_claims as cc
import channel_scorecard
import claims as cm
import market_pulse as mp
import research_analytics as ra
import research_state
import warm_prices


def _claim(source, asset, stance, bucket, published="2026-07-10", video=None, **over):
    base = {
        "claim_id": f"{source}-{asset}-{stance}-{bucket}-{published}-{over.get('n', 0)}",
        "video_id": video or f"vid-{source}-{published}", "channel_name": source,
        "published_at": f"{published}T12:00:00+00:00", "extracted_at": f"{published}T13:00:00+00:00",
        "ticker": asset, "subject_mention": asset, "canonical_entity_name": asset, "stance": stance,
        "horizon_bucket": bucket, "review_required": False, "coverage_status": "full",
        "evidence_start_character": 10, "attribution_type": "speaker_personal_view",
        "repeat_of_claim_id": None, "is_forward_looking": True, "testable": False, "claim_type": "stance",
        "certainty_level": "medium", "recommendation_action": "none", "catalysts": [], "risks": [],
        "assumptions": [], "asset_type": "stock", "schema_version": "2", "run_key": "rk1",
        "forecast_direction": "increase" if stance == "bullish" else "decrease" if stance == "bearish" else None,
        "evidence_text": "e",
    }
    base.update(over)
    return base


def test_short_bearish_and_long_bullish_stay_in_separate_buckets_in_the_weekly_consensus():
    claims = [_claim("A", "NVDA", "bearish", "short"), _claim("A", "NVDA", "bullish", "long", n=1)]
    cons = ra.consensus(claims)
    assert cons[("NVDA", "short")]["bearish"] == 1 and cons[("NVDA", "short")]["net_stance"] == -1.0
    assert cons[("NVDA", "long")]["bullish"] == 1 and cons[("NVDA", "long")]["net_stance"] == 1.0
    views = cc.aggregate_views(claims)
    assert set(views) == {("NVDA", "short"), ("NVDA", "long")}
    assert views[("NVDA", "short")]["bear"] == 1 and views[("NVDA", "long")]["bull"] == 1
    # The weekly pulse renders both horizon-specific views, and never a
    # single averaged stance.
    inputs = mp._canonical_pulse_inputs(7, date(2026, 7, 12), lambda *a: {}, claims=claims)
    text = mp.build_canonical_pulse(inputs, date(2026, 7, 12))
    assert "NVDA [short] — net bearish" in text and "NVDA [long] — net bullish" in text
    assert "net mixed" not in text


def test_legacy_compatibility_rows_never_reach_canonical_analytics(tmp_path, monkeypatch):
    monkeypatch.setattr(research_state, "RESEARCH_DIR", str(tmp_path))
    state = research_state.load_state()
    canonical = _claim("A", "NVDA", "bearish", "short")
    research_state.update(state, canonical["video_id"], active_run_key="rk1")
    research_state.store_claims([canonical], state, canonical["video_id"], "rk1")
    legacy = cm.legacy_record_to_claims({"video_id": "old", "channel_name": "A", "published_at": "2026-07-10T00:00:00+00:00",
                                         "signals": {"assets": [{"name": "Nvidia", "ticker": "NVDA", "stance": "bullish"}]}})
    research_state.update(state, "old", active_run_key=None)
    research_state.store_claims(legacy, state, "old", "legacy", superseding=False)
    research_state.save_state(state)
    loaded = cc.load_canonical_claims(state)
    assert [c["claim_id"] for c in loaded] == [canonical["claim_id"]]
    # The legacy bullish row would have cancelled the canonical bearish view.
    assert ra.consensus(loaded)[("NVDA", "short")]["net_stance"] == -1.0
    assert legacy[0]["schema_version"] == "legacy" and not cm.is_headline_claim(legacy[0])


def test_repeated_claims_and_superseded_runs_are_not_counted(tmp_path, monkeypatch):
    monkeypatch.setattr(research_state, "RESEARCH_DIR", str(tmp_path))
    state = research_state.load_state()
    first = _claim("A", "NVDA", "bearish", "short", run_key="rk1")
    repeat = _claim("A", "NVDA", "bearish", "short", n=1, run_key="rk1", repeat_of_claim_id=first["claim_id"])
    research_state.store_claims([first, repeat], state, first["video_id"], "rk1")
    assert len(cc.load_canonical_claims(state)) == 1
    # A newer run supersedes the old: only its claims count, once.
    newer = _claim("A", "NVDA", "bullish", "short", n=2, run_key="rk2")
    research_state.store_claims([newer], state, first["video_id"], "rk2")
    loaded = cc.load_canonical_claims(state)
    assert [c["claim_id"] for c in loaded] == [newer["claim_id"]]
    assert ra.consensus(loaded)[("NVDA", "short")]["bullish"] == 1
    assert cc.aggregate_views(loaded)[("NVDA", "short")]["mentions"] == 1


def test_one_source_is_one_vote_per_asset_and_bucket_in_the_pulse():
    claims = [_claim("A", "NVDA", "bullish", "short"), _claim("A", "NVDA", "bullish", "short", n=1),
              _claim("B", "NVDA", "bearish", "short", video="vb")]
    views = cc.aggregate_views(claims)[("NVDA", "short")]
    assert views["bull"] == 1 and views["bear"] == 1 and views["mentions"] == 3
    assert views["channels"] == {"A", "B"}


def test_canonical_pulse_tone_comes_from_each_videos_own_views():
    claims = [_claim("A", "NVDA", "bullish", "short"), _claim("A", "AMD", "bearish", "short", n=1),
              _claim("B", "NVDA", "bullish", "long", video="vb"),
              _claim("C", "TSLA", "not_applicable", "unspecified", video="vc", claim_type="portfolio_disclosure")]
    assert cc.video_tone(claims) == {"vid-A-2026-07-10": "mixed", "vb": "bullish"}
    weeks = mp.canonical_tone_weeks(claims, date(2026, 7, 12))
    assert weeks[-1]["n"] == 2 and weeks[-1]["mixed"] == 1 and weeks[-1]["bullish"] == 1


def test_scheduled_jobs_read_canonical_claims_not_signals_jsonl():
    # The production entry points are wired to the shared loader; the legacy
    # loader survives only in the explicitly opted-in compatibility paths.
    assert "load_canonical_claims" in inspect.getsource(channel_scorecard.generate_canonical_scorecard)
    assert "load_canonical_claims" in inspect.getsource(mp._canonical_pulse_inputs)
    assert "load_canonical_claims" in inspect.getsource(ra.main)
    assert "load_signals" not in inspect.getsource(cc)
    assert mp.PULSE_DATA_SOURCE == "canonical"
    assert "load_signals" not in inspect.getsource(mp._canonical_pulse_inputs)
    assert "load_signals" not in inspect.getsource(channel_scorecard.generate_canonical_scorecard)


def test_warm_prices_covers_canonical_forecasts_and_their_benchmarks():
    claim = _claim("A", "NVDA", "bullish", "short", published="2026-07-01", testable=True,
                   forecast_end_date="2026-07-20", target_kind="absolute_value", target_value=150.0)
    ranges = warm_prices.canonical_ranges([claim], date(2026, 8, 1))
    symbols = {s for s, _, _ in ranges}
    assert {"nvda.us", "spy.us"} <= symbols


def test_canonical_scorecard_job_is_experimental_and_unranked_by_default():
    claim = _claim("A", "NVDA", "bullish", "short", published="2026-07-01", testable=True,
                   testability_type="unconditional_testable", forecast_end_date="2026-07-15")
    prices = {"nvda.us": {date(2026, 7, 1) + __import__("datetime").timedelta(days=i): 100.0 + i for i in range(40)}}

    def fetch(symbol, start, end):
        return prices.get(symbol, {})
    fetch.price_provenance = {"provider": "test", "adjustment": "split_adjusted", "corporate_action_status": "split_adjusted"}
    text = channel_scorecard.generate_canonical_scorecard(date(2026, 8, 1), fetch, claims=[claim])
    assert "EXPERIMENTAL" in text and "(unranked)" in text and "canonical claims" in text
