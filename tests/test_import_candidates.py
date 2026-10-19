"""research_backfill --import-candidates: committed offline extractions."""
import json

import claims as cm
import research_backfill as rb
import research_state as rs
import transcript_store as ts

TEXT = ("Welcome back. I think Nvidia will reach $250 by the end of next year, because data center demand "
        "keeps growing. That is my view on the stock. " * 6)
LANGS = {"c-en": ["en"]}


def _store(vid="v1", text=TEXT):
    ts.store_transcript({"video_id": vid, "video_url": f"https://www.youtube.com/watch?v={vid}",
                         "channel_id": "c-en", "channel_name": "Chan", "video_title": "Nvidia outlook",
                         "published_at": "2026-09-14T11:45:00+00:00", "duration_seconds": 600},
                        text, "supadata", "ok", language="en")
    return ts.load_transcript(vid)


def _write(tmp_path, vid, claims, transcript_hash, version=cm.EXTRACTION_PROMPT_VERSION):
    folder = tmp_path / "candidates"
    folder.mkdir(exist_ok=True)
    (folder / f"{vid}.json").write_text(json.dumps({
        "video_id": vid, "transcript_hash": transcript_hash, "extraction_prompt_version": version,
        "source": "assistant_backfill", "response": {"claims": claims, "extraction_metadata": {"warnings": []}},
    }), encoding="utf-8")
    return str(folder)


_CLAIM = {"claim_type": "price_target", "attribution_type": "speaker_personal_view", "speaker": "host",
          "subject_mention": "Nvidia", "stance": "bullish", "is_forward_looking": True,
          "forecast_metric": "price", "forecast_direction": "increase", "target_kind": "absolute_value",
          "target_value": 250, "currency": "USD", "horizon_original": "by the end of next year",
          "evidence_text": "I think Nvidia will reach $250 by the end of next year, because data center "
                           "demand keeps growing."}


def test_candidates_are_validated_and_stored_like_a_model_run(tmp_path, monkeypatch):
    stored = _store()
    folder = _write(tmp_path, "v1", [_CLAIM], stored["transcript_hash"])
    monkeypatch.setattr(rb, "SIGNALS_FILE", str(tmp_path / "signals.jsonl"))
    # No model request is made, so the summary reserve is never consulted.
    monkeypatch.setattr(rb.research_budget, "check", lambda *a, **k: (_ for _ in ()).throw(AssertionError))

    assert rb.run_import_candidates(directory=folder, channel_languages=LANGS) == {"complete": 1}
    entry = rs.load_state()["videos"]["v1"]
    assert entry["research_status"] == "complete" and entry["adopted_from"] == "stored_transcript"
    assert entry["extraction_model"] == rb.OFFLINE_EXTRACTION_MODEL
    assert entry["run_identity"]["model_policy"] == "offline:assistant_backfill"
    claims = rs.load_active_claims(rs.load_state())
    assert len(claims) == 1 and claims[0]["ticker"] == "NVDA" and claims[0]["target_value"] == 250
    assert claims[0]["extraction_model"] == rb.OFFLINE_EXTRACTION_MODEL
    assert claims[0]["run_key"] == entry["active_run_key"]
    runs = rs.load_runs()
    assert runs[-1]["job"] == "import_candidates" and runs[-1]["status"] == "complete"
    # Idempotent, and a finished video is never re-imported over.
    assert rb.run_import_candidates(directory=folder, channel_languages=LANGS) == {"already_done": 1}
    assert len(rs.load_claims("v1")) == 1


def test_evidence_that_is_not_in_the_transcript_is_flagged_not_trusted(tmp_path):
    stored = _store()
    made_up = dict(_CLAIM, evidence_text="Nvidia will triple next quarter, guaranteed.", target_value=900)
    folder = _write(tmp_path, "v1", [made_up], stored["transcript_hash"])
    assert rb.run_import_candidates(directory=folder, channel_languages=LANGS) == {"needs_review": 1}
    claim = rs.load_active_claims(rs.load_state())[0]
    assert claim["review_required"] and "evidence_not_found" in claim["review_reasons"]


def test_a_changed_transcript_or_prompt_version_is_skipped(tmp_path):
    _store()
    folder = _write(tmp_path, "v1", [_CLAIM], "0" * 64)
    assert rb.run_import_candidates(directory=folder, channel_languages=LANGS) == {"transcript_changed": 1}
    stored = ts.load_transcript("v1")
    folder = _write(tmp_path, "v1", [_CLAIM], stored["transcript_hash"], version="0")
    assert rb.run_import_candidates(directory=folder, channel_languages=LANGS) == {"other_prompt_version": 1}
    assert "v1" not in rs.load_state()["videos"]


def test_a_model_run_is_never_overridden(tmp_path):
    stored = _store()
    state = rs.load_state()
    rs.update(state, "v1", delivery_status="sent", research_status="needs_review", active_run_key="rk-model")
    rs.save_state(state)
    folder = _write(tmp_path, "v1", [_CLAIM], stored["transcript_hash"])
    assert rb.run_import_candidates(directory=folder, channel_languages=LANGS) == {"already_done": 1}
    assert rs.load_state()["videos"]["v1"]["active_run_key"] == "rk-model"


def test_an_empty_response_for_an_asset_free_video_is_no_claims_found(tmp_path):
    quiz = "Wer ist der beste Fussballer aller Zeiten? Maldini oder Nesta? Ich gehe auf Maldini. " * 8
    stored = _store("q1", quiz)
    folder = _write(tmp_path, "q1", [], stored["transcript_hash"])
    langs = {"c-en": ["en", "de"]}
    assert rb.run_import_candidates(directory=folder, channel_languages=langs) == {"no_claims_found": 1}
