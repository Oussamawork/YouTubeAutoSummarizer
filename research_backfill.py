"""
Research retry / backfill job: resumable, idempotent, quota-aware.

  python research_backfill.py --retry           retry videos whose research is
                                                pending / failed_retryable /
                                                quota_deferred / partial, from
                                                their stored raw transcripts
  python research_backfill.py --reprocess       re-extract videos whose active
                                                run predates the current
                                                normalization / prompt / schema
                                                versions (stored transcripts only;
                                                nothing is re-fetched)
  python research_backfill.py --import-legacy   import data/signals.jsonl rows
                                                as schema_version="legacy" claims
                                                (no evidence, review_required)

Every mode is safe to stop and restart: state is saved per video, an
extraction run is keyed by (transcript hash, normalization, prompt, schema)
versions so a repeat appends nothing, and a quota deferral ends the run
without counting an attempt. Transcripts are never re-fetched here: a video
without a stored transcript stays failed_retryable with reason
transcript_not_stored, to be re-captured only under the daily job's existing
budget policy.
"""
import argparse
import sys
from datetime import datetime, timedelta, timezone

import claims as claims_mod
import research_state
import transcript_store
from helpers import env_int
from log import log_info, log_warn
from signals import extract_research
from signals_data import load_signals, SIGNALS_FILE
from transcript_normalize import normalize_transcript

# Bound one run: each video is at least one model request from the shared
# summary chain, so a large backlog is drained over several runs.
MAX_VIDEOS_PER_RUN = env_int("RESEARCH_BACKFILL_MAX_VIDEOS", 5)
# Failed (non-quota) retries back off this long before the next attempt.
RETRY_BACKOFF_HOURS = 6


def current_run_key(nt):
    return research_state.run_key(nt.transcript_hash, nt.normalization_version,
                                  claims_mod.EXTRACTION_PROMPT_VERSION, claims_mod.SCHEMA_VERSION)


def _context(entry, stored):
    return {
        "video_id": entry.get("video_id"), "channel_id": entry.get("channel_id") or stored.get("channel_id"),
        "channel_name": entry.get("channel_name") or stored.get("channel_name"),
        "video_title": entry.get("video_title") or stored.get("video_title"),
        "published_at": entry.get("published_at") or stored.get("published_at"),
        "transcript_source": entry.get("transcript_source") or stored.get("transcript_source"),
    }


def process_video(state, video_id, extractor=extract_research):
    """
    One research attempt for a stored transcript. Returns the research status
    written to the state. Idempotent: a run key already active for the video
    is a no-op ("complete").
    """
    entry = research_state.get(state, video_id) or research_state.update(state, video_id)
    stored = transcript_store.load_transcript(video_id)
    if not stored or not stored.get("raw_transcript"):
        research_state.update(state, video_id, research_status="failed_retryable",
                              failure_reason="transcript_not_stored")
        return "failed_retryable"
    nt = normalize_transcript(stored["raw_transcript"], video_id)
    key = current_run_key(nt)
    if entry.get("active_run_key") == key and entry.get("research_status") in ("complete", "no_claims_found", "needs_review"):
        return entry["research_status"]
    ctx = _context(entry, stored)
    ctx["normalized"] = nt
    ctx["run_key"] = key
    research_state.update(state, video_id, research_status="extracting")
    result = extractor(nt, ctx)
    status = result["status"]
    fields = dict(
        research_status=status, transcript_hash=nt.transcript_hash,
        normalization_version=nt.normalization_version,
        extraction_prompt_version=claims_mod.EXTRACTION_PROMPT_VERSION,
        schema_version=claims_mod.SCHEMA_VERSION,
        extraction_model=result.get("extraction_model"), coverage_status=result.get("coverage_status"),
        failure_reason=result.get("failure_reason"),
        processed_chunk_ids=result.get("processed_chunk_ids") or [],
        failed_chunk_ids=result.get("failed_chunk_ids") or [], transcript_stored=True,
        next_eligible_at=None,
    )
    if status == "quota_deferred":
        research_state.update(state, video_id, **fields)
    else:
        research_state.update(state, video_id, **fields)
        counts = status not in ("complete", "no_claims_found", "needs_review")
        e = research_state.note_attempt(state, video_id, counts=counts)
        if counts:
            if e["attempt_count"] >= research_state.MAX_RESEARCH_ATTEMPTS:
                research_state.update(state, video_id, research_status="failed_final")
                status = "failed_final"
            else:
                research_state.update(state, video_id, next_eligible_at=(
                    datetime.now(timezone.utc) + timedelta(hours=RETRY_BACKOFF_HOURS)
                ).replace(microsecond=0).isoformat())
    research_state.record_run({
        "video_id": video_id, "run_key": key, "status": status, "job": "backfill",
        "coverage_status": result.get("coverage_status"), "claims": len(result.get("claims") or []),
        "warnings": result.get("warnings") or [], "failure_reason": result.get("failure_reason"),
        "telemetry": result.get("telemetry") or {}, "recorded_at": research_state.now_iso(),
    })
    research_state.store_segments(video_id, nt)
    if result.get("claims") and status in ("complete", "needs_review", "partial"):
        research_state.store_claims(result["claims"], state, video_id, key)
    if result.get("signals") is not None:
        _refresh_signal_row(video_id, result, ctx)
    research_state.save_state(state)
    return status


def _refresh_signal_row(video_id, result, ctx):
    """Append the compatibility row for a video whose research finished later
    than its delivery. Consumers key on video_id; the row carries the newer
    research_status so a reader can prefer it over an earlier null row."""
    from helpers import append_jsonl
    append_jsonl(SIGNALS_FILE, {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "video_id": video_id,
        "channel_id": ctx.get("channel_id"), "channel_name": ctx.get("channel_name"),
        "video_title": ctx.get("video_title"), "published_at": ctx.get("published_at"),
        "summary": None, "signals": result["signals"], "research_status": result["status"],
        "coverage_status": result.get("coverage_status"), "backfilled": True,
    })


def run_retry(max_videos=None, extractor=extract_research, state=None):
    state = state if state is not None else research_state.load_state()
    max_videos = MAX_VIDEOS_PER_RUN if max_videos is None else max_videos
    due = research_state.retry_candidates(state)
    log_info(f"Research retry: {len(due)} video(s) due, processing up to {max_videos}.")
    done = {}
    for video_id in due[:max_videos] if max_videos > 0 else due:
        status = process_video(state, video_id, extractor)
        done[video_id] = status
        if status == "quota_deferred":
            log_warn("LLM quota exhausted; stopping the research retry here (state saved).")
            break
    research_state.save_state(state)
    return done


def run_reprocess(max_videos=None, extractor=extract_research, state=None):
    """Re-extract stored transcripts whose active run predates the current
    versions. Older runs are superseded, never double-counted."""
    state = state if state is not None else research_state.load_state()
    max_videos = MAX_VIDEOS_PER_RUN if max_videos is None else max_videos
    stale = []
    for video_id, entry in state["videos"].items():
        if (entry.get("extraction_prompt_version") != claims_mod.EXTRACTION_PROMPT_VERSION
                or entry.get("schema_version") != claims_mod.SCHEMA_VERSION
                or entry.get("normalization_version") != normalize_transcript("").normalization_version):
            stale.append(video_id)
    done = {}
    for video_id in stale[:max_videos] if max_videos > 0 else stale:
        status = process_video(state, video_id, extractor)
        done[video_id] = status
        if status == "quota_deferred":
            break
    research_state.save_state(state)
    return done


def import_legacy(path=SIGNALS_FILE, state=None):
    """Import legacy signals rows as review-required legacy claims. Idempotent
    by claim id; videos without research state get a record so the ledger
    covers the whole history."""
    state = state if state is not None else research_state.load_state()
    existing = {c.get("claim_id") for c in research_state.load_claims()}
    imported, videos = 0, 0
    for record in load_signals(path):
        legacy = [c for c in claims_mod.legacy_record_to_claims(record) if c["claim_id"] not in existing]
        vid = record.get("video_id")
        if vid and not research_state.get(state, vid):
            research_state.update(state, vid, delivery_status="sent", research_status="superseded"
                                  if record.get("research_status") else "complete",
                                  schema_version="legacy", channel_id=record.get("channel_id"),
                                  channel_name=record.get("channel_name"),
                                  video_title=record.get("video_title"),
                                  published_at=record.get("published_at"), transcript_stored=False)
            videos += 1
        if legacy:
            imported += research_state.store_claims(legacy, state, vid, "legacy", superseding=False)
            existing.update(c["claim_id"] for c in legacy)
    research_state.save_state(state)
    log_info(f"Legacy import: {imported} claim(s) from {videos} newly tracked video(s).")
    return imported


def main():
    parser = argparse.ArgumentParser(description="Research retry / backfill")
    parser.add_argument("--retry", action="store_true")
    parser.add_argument("--reprocess", action="store_true")
    parser.add_argument("--import-legacy", action="store_true")
    parser.add_argument("--max-videos", type=int, default=None)
    args = parser.parse_args()
    if not (args.retry or args.reprocess or args.import_legacy):
        parser.print_help()
        return 1
    if args.import_legacy:
        import_legacy()
    if args.retry:
        run_retry(args.max_videos)
    if args.reprocess:
        run_reprocess(args.max_videos)
    return 0


if __name__ == "__main__":
    sys.exit(main())
