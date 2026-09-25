"""
Versioned per-chunk partial results for the resumable chunked paths.

A chunked summary (`summarizer._summarize_chunked`) and a chunked claim
extraction (`signals._extract_research_chunked`) each persist one record per
chunk so a run that stops halfway resumes without repeating the chunks that
already succeeded. A cached chunk result is only reusable when EVERY input
that shaped it is unchanged, so each record carries the full key and is
validated against the current inputs before it is reused:

    task_type            summary_notes | research_claims (different prompts,
                         different products — never interchangeable)
    transcript_hash      the raw transcript the chunk was cut from
    normalization_version, chunking_version
    chunk boundaries     start/end character offsets and the chunk id
    prompt_version       the prompt the model answered
    schema_version       the record shape expected of the answer
    model_policy         the provider/model chain the run was configured with

Anything missing or different means the record is stale: it is ignored (and
overwritten by the new result), never merged into a product built under
other rules.
"""
import json
import os

from helpers import write_json_atomic
from log import log_info

CACHE_RECORD_VERSION = 1


def partials_dir():
    import summarizer
    return summarizer.PARTIALS_DIR


def path_for(transcript_hash, task_type, chunk_id):
    return os.path.join(partials_dir(), transcript_hash, f"{task_type}-{chunk_id}.json")


def cache_key(task_type, nt, chunk, prompt_version, schema_version, model_policy):
    import transcript_normalize as tn
    return {
        "record_version": CACHE_RECORD_VERSION,
        "task_type": task_type,
        "transcript_hash": nt.transcript_hash,
        "normalization_version": nt.normalization_version,
        "chunking_version": tn.CHUNKING_VERSION,
        "chunk_id": chunk.chunk_id,
        "start_character": chunk.start_character,
        "end_character": chunk.end_character,
        "prompt_version": str(prompt_version),
        "schema_version": str(schema_version),
        "model_policy": model_policy or "",
    }


def matches(record, key):
    """True when a stored record was produced under exactly `key`."""
    if not isinstance(record, dict) or not isinstance(record.get("key"), dict):
        return False
    stored = record["key"]
    return all(stored.get(k) == v for k, v in key.items())


def load(task_type, nt, chunk, prompt_version, schema_version, model_policy):
    """The cached payload for this chunk under the current inputs, or None."""
    key = cache_key(task_type, nt, chunk, prompt_version, schema_version, model_policy)
    try:
        with open(path_for(nt.transcript_hash, task_type, chunk.chunk_id), "r", encoding="utf-8") as f:
            record = json.load(f)
    except (OSError, ValueError):
        return None
    if not matches(record, key):
        log_info(f"Ignoring stale partial for {chunk.chunk_id} ({task_type}): inputs changed.")
        return None
    return record.get("payload")


def save(task_type, nt, chunk, payload, prompt_version, schema_version, model_policy, model=None):
    key = cache_key(task_type, nt, chunk, prompt_version, schema_version, model_policy)
    return write_json_atomic(path_for(nt.transcript_hash, task_type, chunk.chunk_id), {
        "key": key, "chunk": chunk.to_dict(), "model": model, "payload": payload,
    })


def model_policy_string(providers):
    """A stable description of the provider chain a run was configured with."""
    return ",".join(f"{p.get('name')}:{p.get('model')}" for p in (providers or []))
