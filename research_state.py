"""
Independent research state and the canonical research data products.

Telegram delivery and research extraction are two products of one video, and
they succeed or fail independently: a delivered summary says nothing about
whether claims were captured. This module keeps that second ledger.

  data/research/research_state.json   per-video research status (atomic)
  data/research/claims.jsonl          every validated claim, append-only
  data/research/extraction_runs.jsonl one row per extraction attempt
  data/research/review_queue.jsonl    claims a human should look at
  data/research/segments.jsonl        transcript segments per (video, hash)
  data/research/gate_outcomes.jsonl   why each discovered video was/wasn't analyzed
  data/research/video_records.jsonl   one row per video seen by the research side
  data/research/condition_evaluations.jsonl  observed outcomes of forecast conditions

Idempotency: an extraction run is keyed by its complete identity — the
transcript hash, normalization version, extraction prompt version, schema
version, chunking version, extraction mode (combined / standalone / chunked /
exhaustive), chunk policy, provider-model policy and model configuration
(`run_key`). Rerunning the same key appends nothing. A run with a new key —
a new prompt, but equally a different extraction model or chunking policy —
is a NEW run that supersedes the older one: the state records the active
run, and `load_active_claims` returns only that run's claims, so two
versions are never counted together.
"""
import hashlib
import json
import os
from datetime import datetime, timezone

from helpers import append_jsonl, write_json_atomic
from log import log_error, log_warn

RESEARCH_DIR = os.getenv("RESEARCH_DIR") or "data/research"
STATE_FILE = os.path.join(RESEARCH_DIR, "research_state.json")
CLAIMS_FILE = os.path.join(RESEARCH_DIR, "claims.jsonl")
RUNS_FILE = os.path.join(RESEARCH_DIR, "extraction_runs.jsonl")
REVIEW_FILE = os.path.join(RESEARCH_DIR, "review_queue.jsonl")
SEGMENTS_FILE = os.path.join(RESEARCH_DIR, "segments.jsonl")
GATE_FILE = os.path.join(RESEARCH_DIR, "gate_outcomes.jsonl")
VIDEOS_FILE = os.path.join(RESEARCH_DIR, "video_records.jsonl")
CONDITIONS_FILE = os.path.join(RESEARCH_DIR, "condition_evaluations.jsonl")

STATUSES = {"pending", "extracting", "complete", "no_claims_found", "partial", "needs_review",
            "quota_deferred", "failed_retryable", "failed_final", "superseded", "transcript_rejected"}
# transcript_rejected: the stored transcript was in the wrong language and
# has been retired (transcript_store.reject_transcript); the video's runs are
# superseded and it waits for a re-capture (research_backfill --refetch),
# never for a retry from the retired text.
RETRYABLE = {"pending", "partial", "quota_deferred", "failed_retryable"}
GATE_OUTCOMES = {"included", "already_decided", "title_filtered", "duration_filtered",
                 "retry_backoff", "metadata_unavailable", "transcript_unavailable",
                 "budget_deferred", "model_quota_deferred", "manual_review", "deadline_deferred",
                 "other"}
FILTER_CONFIG_VERSION = "1"
# Research retries stop after this many attempts (failed_final); a quota
# deferral never counts as an attempt.
MAX_RESEARCH_ATTEMPTS = 6

_files = {}


def _paths():
    """Resolved file paths (RESEARCH_DIR can be repointed by tests)."""
    d = RESEARCH_DIR
    return {
        "state": os.path.join(d, "research_state.json"), "claims": os.path.join(d, "claims.jsonl"),
        "runs": os.path.join(d, "extraction_runs.jsonl"), "review": os.path.join(d, "review_queue.jsonl"),
        "segments": os.path.join(d, "segments.jsonl"), "gate": os.path.join(d, "gate_outcomes.jsonl"),
        "videos": os.path.join(d, "video_records.jsonl"),
        "conditions": os.path.join(d, "condition_evaluations.jsonl"),
    }


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


RUN_KEY_FORMAT = "<hash16>:n<normalization>:p<prompt>:s<schema>:c<chunking>:m<mode>:k<policy10>"
# The identity fields hashed into the trailing `k` segment, in order. They are
# recorded in full on every extraction run (`identity` on the runs file and
# `run_identity` on the state entry) so a key can always be validated.
POLICY_FIELDS = ("chunk_policy", "model_policy", "model_config")


def policy_digest(identity):
    policy = "|".join(str((identity or {}).get(k) or "") for k in POLICY_FIELDS)
    return hashlib.sha1(policy.encode("utf-8")).hexdigest()[:10]


def run_key(transcript_hash, normalization_version, prompt_version, schema_version, identity=None):
    """
    The run identity string (see RUN_KEY_FORMAT). `identity` carries the
    processing inputs beyond the four versions: `chunking_version`,
    `extraction_mode` (combined | standalone | chunked | exhaustive),
    `chunk_policy` ("full" or the chunk token size), `model_policy` (the
    provider/model chain) and `model_config` (temperature, reasoning effort,
    output caps). Two runs that differ in any of them get different keys, so
    changing the extraction model or the chunking policy permits a new
    active run instead of reading as the same extraction. Without an
    identity the legacy four-part key is returned (old rows stay readable).
    """
    base = f"{transcript_hash[:16]}:n{normalization_version}:p{prompt_version}:s{schema_version}"
    if not identity:
        return base
    return (f"{base}:c{identity.get('chunking_version')}:m{identity.get('extraction_mode')}"
            f":k{policy_digest(identity)}")


def parse_run_key(key):
    """The segments of a run key as a dict (legacy keys have no mode)."""
    out = {"transcript_hash_prefix": None, "normalization_version": None, "prompt_version": None,
           "schema_version": None, "chunking_version": None, "extraction_mode": None, "policy_digest": None}
    parts = (key or "").split(":")
    if not parts or not parts[0]:
        return out
    out["transcript_hash_prefix"] = parts[0]
    for part in parts[1:]:
        tag, value = part[:1], part[1:]
        name = {"n": "normalization_version", "p": "prompt_version", "s": "schema_version",
                "c": "chunking_version", "m": "extraction_mode", "k": "policy_digest"}.get(tag)
        if name:
            out[name] = value
    return out


# --- State ------------------------------------------------------------------


def load_state():
    path = _paths()["state"]
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("videos"), dict):
            return data
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as e:
        log_error(f"Could not read research state {path}: {e}. Starting fresh.")
    return {"version": 1, "videos": {}}


def save_state(state):
    return write_json_atomic(_paths()["state"], state)


def get(state, video_id):
    return state["videos"].get(video_id)


def update(state, video_id, **fields):
    """Merge fields into a video's research record and stamp updated_at."""
    entry = state["videos"].setdefault(video_id, {
        "video_id": video_id, "delivery_status": None, "research_status": "pending",
        "transcript_hash": None, "schema_version": None, "normalization_version": None,
        "extraction_prompt_version": None, "extraction_model": None, "attempt_count": 0,
        "last_attempt_at": None, "next_eligible_at": None, "processed_chunk_ids": [],
        "failed_chunk_ids": [], "coverage_status": None, "failure_reason": None,
        "active_run_key": None, "superseded_run_keys": [], "transcript_stored": None,
        "created_at": now_iso(), "updated_at": None,
    })
    status = fields.get("research_status")
    if status and status not in STATUSES:
        log_warn(f"Unknown research status {status!r} for {video_id}; recording as failed_retryable.")
        fields["research_status"] = "failed_retryable"
    entry.update(fields)
    entry["updated_at"] = now_iso()
    return entry


def note_attempt(state, video_id, counts=True):
    entry = update(state, video_id, last_attempt_at=now_iso())
    if counts:
        entry["attempt_count"] = int(entry.get("attempt_count") or 0) + 1
    return entry


def retry_candidates(state, now=None):
    """Video ids whose research is due for another try."""
    now = now or datetime.now(timezone.utc)
    due = []
    for video_id, entry in state["videos"].items():
        if entry.get("research_status") not in RETRYABLE:
            continue
        nxt = entry.get("next_eligible_at")
        if nxt:
            try:
                if datetime.fromisoformat(nxt) > now:
                    continue
            except ValueError:
                pass
        due.append(video_id)
    return due


# --- Append-only products -----------------------------------------------------


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except OSError as e:
        log_error(f"Could not read {path}: {e}")
    return rows


def existing_run_keys(video_id):
    """Run keys already recorded for a video (from the runs file)."""
    return {r.get("run_key") for r in _read_jsonl(_paths()["runs"]) if r.get("video_id") == video_id}


def record_run(run):
    return append_jsonl(_paths()["runs"], run)


def record_video(video_details, channel_id, extra=None):
    row = {
        "video_id": video_details.get("video_id"), "channel_id": channel_id,
        "channel_name": video_details.get("channel_name"),
        "video_title": video_details.get("video_title"),
        "video_url": video_details.get("video_url"),
        "published_at": video_details.get("published_at"),
        "duration_seconds": video_details.get("duration_seconds"),
        "recorded_at": now_iso(),
    }
    row.update(extra or {})
    return append_jsonl(_paths()["videos"], row)


def record_gate_outcome(channel_id, video, outcome, channel_config=None, discovery_source=None,
                        min_duration=None, latest_only=False, reason=None):
    if outcome not in GATE_OUTCOMES:
        outcome = "other"
    return append_jsonl(_paths()["gate"], {
        "video_id": video.get("video_id"), "channel_id": channel_id,
        "channel_name": video.get("channel_name"), "video_title": video.get("video_title"),
        "published_at": video.get("published_at"), "outcome": outcome, "reason": reason,
        "filter_config_version": FILTER_CONFIG_VERSION,
        "channel_config": channel_config, "min_duration_seconds": min_duration,
        "title_filters": (channel_config or {}).get("only") if channel_config else None,
        "discovery_source": discovery_source, "latest_only": bool(latest_only),
        "decided_at": now_iso(),
    })


def store_segments(video_id, nt):
    """Persist a transcript's segments once per (video, hash)."""
    path = _paths()["segments"]
    marker = f"{video_id}:{nt.transcript_hash}"
    if any(r.get("transcript_key") == marker for r in _read_jsonl(path)):
        return True
    ok = True
    for seg in nt.segments:
        row = seg.to_dict()
        row.update({"video_id": video_id, "transcript_key": marker,
                    "normalization_version": nt.normalization_version})
        ok = append_jsonl(path, row) and ok
    return ok


def store_claims(claims, state, video_id, run_key_value, superseding=True):
    """
    Append a run's claims and make that run the video's active one. Returns
    the number appended (0 when the run key was already stored — idempotent).
    """
    if not claims:
        return 0
    entry = state["videos"].get(video_id) or update(state, video_id)
    if run_key_value in existing_run_keys(video_id) and entry.get("active_run_key") == run_key_value:
        return 0
    already = {c.get("claim_id") for c in load_claims(video_id=video_id)}
    appended = 0
    for claim in claims:
        if claim.get("claim_id") in already:
            continue
        row = dict(claim)
        row["run_key"] = run_key_value
        if append_jsonl(_paths()["claims"], row):
            appended += 1
            if claim.get("review_required"):
                append_jsonl(_paths()["review"], {
                    "claim_id": claim["claim_id"], "video_id": video_id,
                    "reasons": claim.get("review_reasons"), "queued_at": now_iso(),
                    "evidence_text": claim.get("evidence_text"),
                    "subject_mention": claim.get("subject_mention"),
                })
    if superseding:
        old = entry.get("active_run_key")
        if old and old != run_key_value:
            superseded = list(entry.get("superseded_run_keys") or [])
            superseded.append(old)
            update(state, video_id, superseded_run_keys=superseded)
        update(state, video_id, active_run_key=run_key_value)
    return appended


def load_claims(video_id=None, path=None):
    rows = _read_jsonl(path or _paths()["claims"])
    if video_id is not None:
        rows = [r for r in rows if r.get("video_id") == video_id]
    return rows


def load_active_claims(state=None, path=None, include_legacy=True):
    """
    Claims of each video's ACTIVE run only. Legacy-imported claims (no run
    key) are included when asked and marked as such; superseded runs are
    filtered out and flagged on the returned copies for traceability.
    """
    state = state or load_state()
    active = {vid: e.get("active_run_key") for vid, e in state["videos"].items()}
    superseded = {vid: set(e.get("superseded_run_keys") or []) for vid, e in state["videos"].items()}
    out = []
    for row in load_claims(path=path):
        key = row.get("run_key")
        vid = row.get("video_id")
        if row.get("schema_version") == "legacy":
            if include_legacy:
                out.append(row)
            continue
        if key and (key in superseded.get(vid, ()) or active.get(vid) not in (None, key)):
            continue  # superseded (a rejected transcript's runs have no active successor yet)
        out.append(row)
    return out


def load_gate_outcomes(path=None):
    return _read_jsonl(path or _paths()["gate"])


def load_runs(path=None):
    return _read_jsonl(path or _paths()["runs"])


# --- Condition evaluations ----------------------------------------------------
#
# Claims are append-only, so the observed outcome of a conditional forecast's
# condition ("if the Fed cuts in September") is recorded here and overlaid on
# the claim by canonical_claims.load_canonical_claims. The latest evaluation
# per claim wins; each row says what was observed, when, from which source.


def record_condition_evaluation(claim_id, status, evaluation_date=None, evidence=None, data_source=None):
    import claims as claims_mod
    if status not in claims_mod.CONDITION_STATUSES:
        log_warn(f"Unknown condition status {status!r} for {claim_id}; recording as unknown.")
        status = "unknown"
    return append_jsonl(_paths()["conditions"], {
        "claim_id": claim_id, "condition_status": status,
        "condition_evaluation_date": evaluation_date, "condition_evidence": evidence,
        "condition_data_source": data_source, "recorded_at": now_iso(),
    })


def load_condition_evaluations(path=None):
    """{claim_id: latest evaluation row}."""
    latest = {}
    for row in _read_jsonl(path or _paths()["conditions"]):
        if row.get("claim_id"):
            latest[row["claim_id"]] = row
    return latest
