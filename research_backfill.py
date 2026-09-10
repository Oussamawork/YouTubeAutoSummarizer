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
  python research_backfill.py --reject-foreign-transcripts
                                                retire every stored transcript
                                                that is not in its channel's
                                                language (a translated caption
                                                track); its runs are superseded
                                                and the video waits for --refetch
  python research_backfill.py --refetch         re-capture rejected videos'
                                                transcripts in the channel's
                                                language, under the normal
                                                transcript budget, then extract

Every mode is safe to stop and restart: state is saved per video, an
extraction run is keyed by its complete identity (transcript hash,
normalization / prompt / schema / chunking versions, extraction mode, chunk
policy, model policy and configuration — research_state.run_key) so a
repeat appends nothing while a changed model or chunk policy is a new run,
and a quota deferral ends the run without counting an attempt. `--retry` and
`--reprocess` never fetch a transcript: a video without a stored one stays
failed_retryable with reason transcript_not_stored, to be re-captured under
the daily job's existing budget policy. Only `--refetch` spends transcript
credits, on the bounded list of videos whose capture was rejected, and it
uses the same sources, budget and language verification as the daily job.
"""
import argparse
import sys
from datetime import datetime, timedelta, timezone

import claims as claims_mod
import language_detect
import research_budget
import research_state
import transcript_store
from helpers import env_int, read_channels
from log import log_info, log_warn
import signals
from signals import extract_research
from signals_data import load_signals, SIGNALS_FILE
import transcript_normalize as tn
from transcript_normalize import normalize_transcript

# Bound one run: each video is at least one model request from the shared
# summary chain, so a large backlog is drained over several runs.
MAX_VIDEOS_PER_RUN = env_int("RESEARCH_BACKFILL_MAX_VIDEOS", 5)
# Failed (non-quota) retries back off this long before the next attempt.
RETRY_BACKOFF_HOURS = 6
# --refetch spends transcript credits; keep one run small.
MAX_REFETCH_PER_RUN = env_int("RESEARCH_REFETCH_MAX_VIDEOS", 3)
CHANNELS_FILE = "channel_ids.txt"


def current_run_key(nt):
    """The run key a retry of `nt` would get NOW: the complete identity
    (versions, chunking, mode, model policy and configuration), so a video
    extracted under another model or chunk policy is not "already done"."""
    return signals.planned_run_key(nt)


def _context(entry, stored):
    return {
        "video_id": entry.get("video_id"), "channel_id": entry.get("channel_id") or stored.get("channel_id"),
        "channel_name": entry.get("channel_name") or stored.get("channel_name"),
        "video_title": entry.get("video_title") or stored.get("video_title"),
        "published_at": entry.get("published_at") or stored.get("published_at"),
        "transcript_source": entry.get("transcript_source") or stored.get("transcript_source"),
        "transcript_language": stored.get("transcript_language") or entry.get("transcript_language"),
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
    ctx["run_key"], ctx["run_identity"] = signals.planned_run(nt)
    # The same reserve the daily run applies: a backlog never eats the
    # requests today's summaries still need.
    verdict = research_budget.check(nt)
    if not verdict["allowed"]:
        research_state.update(state, video_id, research_status="quota_deferred",
                              failure_reason="summary_reserve_protected", budget_check=verdict)
        return "quota_deferred"
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
        next_eligible_at=None, run_identity=result.get("run_identity"),
    )
    # The extraction may have changed mode on the way (a standalone call
    # that overflowed and re-ran chunked): the key it actually ran under is
    # the one the claims carry.
    key = result.get("run_key") or key
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
        "identity": result.get("run_identity"), "chunk_boundaries": result.get("chunk_boundaries"),
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


def _channel_languages(channels_file=CHANNELS_FILE):
    """{channel_id: [languages]} from the channel list; a channel without a
    `lang=` option gets the pipeline default. Handles (@name) cannot be
    resolved offline and fall to the default too."""
    out = {}
    for entry in read_channels(channels_file):
        code = entry.get("language")
        out[entry["channel_id"]] = [code] if code else list(language_detect.DEFAULT_LANGUAGES)
    return out


def _channel_id_of(entry, stored):
    """The channel id for a research entry: its own, the stored transcript's,
    or the one the gate log recorded for that channel name (RSS videos
    carried no id before 2026-09)."""
    cid = entry.get("channel_id") or (stored or {}).get("channel_id")
    if cid:
        return cid
    name = entry.get("channel_name") or (stored or {}).get("channel_name")
    if not name:
        return None
    for row in research_state.load_gate_outcomes():
        if row.get("channel_name") == name and row.get("channel_id"):
            return row["channel_id"]
    return None


def languages_for(entry, stored=None, channel_languages=None):
    """The languages a video's transcript must be in."""
    channel_languages = _channel_languages() if channel_languages is None else channel_languages
    cid = _channel_id_of(entry, stored)
    return list(channel_languages.get(cid) or language_detect.DEFAULT_LANGUAGES)


def run_reject_foreign(state=None, channel_languages=None):
    """
    Retire every stored transcript that is not in its channel's language.
    For each video with a stored transcript: verify the raw text against
    the channel's languages; on a mismatch retire the file
    (transcript_store.reject_transcript), supersede the video's active run
    (its claims leave the canonical set), record the verdict and mark the
    video transcript_rejected for --refetch. Returns {"checked", "rejected",
    "videos": [ids]}. Offline, idempotent: a retired video is skipped.
    """
    state = state if state is not None else research_state.load_state()
    channel_languages = _channel_languages() if channel_languages is None else channel_languages
    tally = {"checked": 0, "rejected": 0, "videos": []}
    for video_id, entry in list(state["videos"].items()):
        if entry.get("research_status") == "transcript_rejected":
            continue
        stored = transcript_store.load_transcript(video_id)
        if not stored or not stored.get("raw_transcript"):
            continue
        tally["checked"] += 1
        accepted = languages_for(entry, stored, channel_languages)
        verdict = language_detect.verify_language(stored["raw_transcript"], accepted)
        if verdict["ok"]:
            continue
        reason = f"transcript_language_mismatch:{verdict['reason']}"
        meta = transcript_store.reject_transcript(video_id, reason)
        if meta is None:
            continue
        superseded = list(entry.get("superseded_run_keys") or [])
        if entry.get("active_run_key") and entry["active_run_key"] not in superseded:
            superseded.append(entry["active_run_key"])
        research_state.update(
            state, video_id, research_status="transcript_rejected", failure_reason=reason,
            transcript_stored=False, transcript_language=verdict["language"],
            transcript_check=verdict, refetch_languages=accepted,
            active_run_key=None, superseded_run_keys=superseded, next_eligible_at=None,
            video_meta={k: meta.get(k) for k in ("video_url", "video_title", "published_at",
                                                 "duration_seconds", "channel_id", "channel_name")},
        )
        research_state.record_run({
            "video_id": video_id, "run_key": None, "status": "transcript_rejected", "job": "reject_foreign",
            "failure_reason": reason, "superseded_run_keys": superseded,
            "transcript_language": verdict["language"], "accepted_languages": accepted,
            "recorded_at": research_state.now_iso(),
        })
        tally["rejected"] += 1
        tally["videos"].append(video_id)
        log_warn(f"{video_id}: stored transcript is {verdict['language']} ({verdict['reason']}); "
                 f"channel speaks {', '.join(accepted)}. Retired; claims of run "
                 f"{superseded[-1] if superseded else '-'} superseded.")
    research_state.save_state(state)
    log_info(f"Foreign-transcript check: {tally['checked']} checked, {tally['rejected']} rejected.")
    return tally


def run_refetch(max_videos=None, fetcher=None, extractor=extract_research, state=None):
    """
    Re-capture the transcripts of transcript_rejected videos, in the
    channel's language, and extract their claims. Uses the daily job's
    source chain and budget (transcript.get_transcript_from_video): a spent
    transcript budget ends the run without counting an attempt; a capture
    that fails otherwise counts one, and MAX_RESEARCH_ATTEMPTS of those make
    the video failed_final. Returns {status: count}.
    """
    import transcript as transcript_mod
    fetcher = fetcher or transcript_mod.get_transcript_from_video
    state = state if state is not None else research_state.load_state()
    max_videos = MAX_REFETCH_PER_RUN if max_videos is None else max_videos
    due = [vid for vid, e in state["videos"].items() if e.get("research_status") == "transcript_rejected"]
    log_info(f"Transcript re-fetch: {len(due)} video(s) rejected, re-capturing up to {max_videos}.")
    tally = {}
    for video_id in due[:max_videos]:
        entry = research_state.get(state, video_id)
        meta = entry.get("video_meta") or {}
        url = meta.get("video_url") or f"https://www.youtube.com/watch?v={video_id}"
        languages = entry.get("refetch_languages") or languages_for(entry)
        result = fetcher(url, languages=languages)
        text = (result or {}).get("transcript") or ""
        if not text:
            if (result or {}).get("budget_exhausted"):
                log_warn(f"{video_id}: transcript budget spent; re-fetch stops here for today.")
                tally["budget_deferred"] = tally.get("budget_deferred", 0) + 1
                break
            e = research_state.note_attempt(state, video_id)
            reason = f"refetch_failed:{(result or {}).get('reason')}"
            if e["attempt_count"] >= research_state.MAX_RESEARCH_ATTEMPTS:
                research_state.update(state, video_id, research_status="failed_final", failure_reason=reason)
                tally["failed_final"] = tally.get("failed_final", 0) + 1
            else:
                research_state.update(state, video_id, failure_reason=reason)
                tally["refetch_failed"] = tally.get("refetch_failed", 0) + 1
            research_state.save_state(state)
            continue
        details = {"video_id": video_id, "video_url": url, "channel_id": meta.get("channel_id") or entry.get("channel_id"),
                   "channel_name": meta.get("channel_name") or entry.get("channel_name"),
                   "video_title": meta.get("video_title") or entry.get("video_title"),
                   "published_at": meta.get("published_at") or entry.get("published_at"),
                   "duration_seconds": meta.get("duration_seconds")}
        source = {"ok": "supadata", "gemini_ok": "gemini_video", "fallback_ok": "youtube_transcript_api"}.get(
            result.get("reason"))
        record = transcript_store.store_transcript(details, text, source, result.get("reason"),
                                                   language=result.get("language"))
        if not record.get("stored"):
            research_state.update(state, video_id, failure_reason=f"refetch_not_stored:{record.get('error')}")
            research_state.save_state(state)
            tally["refetch_failed"] = tally.get("refetch_failed", 0) + 1
            continue
        research_state.update(state, video_id, research_status="pending", failure_reason=None,
                              transcript_stored=True, transcript_source=source,
                              transcript_language=result.get("language"), transcript_hash=record["transcript_hash"])
        research_state.record_video(details, details["channel_id"], {
            "transcript_source": source, "transcript_language": result.get("language"),
            "transcript_hash": record["transcript_hash"], "raw_char_count": len(text), "refetched": True,
        })
        status = process_video(state, video_id, extractor=extractor)
        tally[status] = tally.get(status, 0) + 1
        if status == "quota_deferred":
            break
    research_state.save_state(state)
    log_info(f"Transcript re-fetch done: {tally}")
    return tally


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
        identity = entry.get("run_identity") or {}
        if (entry.get("extraction_prompt_version") != claims_mod.EXTRACTION_PROMPT_VERSION
                or entry.get("schema_version") != claims_mod.SCHEMA_VERSION
                or entry.get("normalization_version") != normalize_transcript("").normalization_version
                or (identity and identity.get("chunking_version") != tn.CHUNKING_VERSION)):
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
    parser.add_argument("--reject-foreign-transcripts", action="store_true")
    parser.add_argument("--refetch", action="store_true")
    parser.add_argument("--max-videos", type=int, default=None)
    args = parser.parse_args()
    if not (args.retry or args.reprocess or args.import_legacy or args.reject_foreign_transcripts
            or args.refetch):
        parser.print_help()
        return 1
    if args.import_legacy:
        import_legacy()
    if args.reject_foreign_transcripts:
        run_reject_foreign()
    if args.refetch:
        run_refetch(args.max_videos)
    if args.retry:
        run_retry(args.max_videos)
    if args.reprocess:
        run_reprocess(args.max_videos)
    return 0


if __name__ == "__main__":
    sys.exit(main())
