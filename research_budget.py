"""
Keeps optional research extraction from spending the requests that later
summaries in the same day need.

Summaries and claim extraction share one provider chain and, on the Gemini
free tier, one per-model daily budget of 20 requests. A single video's
standalone or chunked claim extraction can cost several of those; done in
the middle of the delivery loop it would starve a later video's summary —
the product readers actually wait for. Two rules protect the summaries:

1. DEFERRAL: `scraper.main` never extracts claims separately while videos
   are still being delivered. A combined summary+claims call is the summary
   call, so it runs in place; a failed combined call gets its summary-only
   fallback immediately; but the separate research extraction is queued and
   drained only after every eligible video has been processed.

2. RESERVE: before any separate extraction starts, the requests it is
   estimated to need are checked against what is left today:

       remaining_requests - estimated_research_requests >= reserved_summary_requests
                                                           (+ the run's remaining delivery workload)

   `SUMMARY_REQUEST_RESERVE` requests are always kept back for summaries
   (later runs the same day, the retry of a deferred video). A research
   extraction that would cut into the reserve is recorded as
   `quota_deferred` / `summary_reserve_protected` for the retry job and costs
   nothing now. The retry job applies the same check.

The estimate counts model requests, not tokens: one for a transcript that
fits a single claims request, else one per complete-coverage chunk. When an
unmetered provider (Groq, a custom endpoint) or no provider at all is
configured, no daily cap binds and the check passes — the API stays the
authority, as everywhere else in this pipeline.
"""
from helpers import env_int
from log import log_info

# Requests held back for summaries whenever a separate research extraction
# is considered. Four covers the two Gemini models' worth of one deferred
# video's escalation ladder on the default chain.
SUMMARY_REQUEST_RESERVE = env_int("SUMMARY_REQUEST_RESERVE", 4)


def remaining_requests(providers=None):
    """
    Requests the metered providers may still make today, summed over the
    models not yet exhausted — or None when no daily cap binds (an unmetered
    provider is configured, or none at all).
    """
    import gemini_quota
    import summarizer
    providers = summarizer._provider_configs() if providers is None else providers
    if not providers:
        return None
    total = 0
    for provider in providers:
        model = summarizer._metered_model(provider)
        if model is None:
            return None
        if provider["name"] in summarizer._EXHAUSTED_PROVIDERS or gemini_quota.is_exhausted(model):
            continue
        total += max(0, gemini_quota.GEMINI_REQUESTS_PER_DAY - gemini_quota.used(model))
    return total


def estimate_research_requests(nt, prefer_chunked=False):
    """
    Model requests a separate claim extraction of `nt` is expected to make:
    one when the transcript fits one claims request, else the number of
    complete-coverage chunks it splits into (at the research chunk size when
    chunking is forced or exhaustive mode is on). Never below one.
    """
    import signals
    import token_budget
    import transcript_normalize as tn
    if nt is None or not nt.text.strip():
        return 0
    per_chunk = signals._research_chunk_tokens()
    if signals.EXHAUSTIVE_RESEARCH_MODE or prefer_chunked:
        per_chunk = min(per_chunk, signals.RESEARCH_CHUNK_TOKENS) if per_chunk > 0 else signals.RESEARCH_CHUNK_TOKENS
    if per_chunk <= 0:
        return 1
    if not (signals.EXHAUSTIVE_RESEARCH_MODE or prefer_chunked) and token_budget.estimate_tokens(nt.text) <= per_chunk:
        return 1
    return max(1, len(tn.chunk_transcript(nt, per_chunk, token_budget.estimate_tokens)))


def check(nt, prefer_chunked=False, reserve=None, workload=0, providers=None):
    """
    Whether a separate research extraction may start now. Returns a dict:
      allowed, remaining (None = no cap binds), estimated, reserve,
      workload, reason.
    `workload` is how many summary requests this run still expects to make.
    """
    reserve = SUMMARY_REQUEST_RESERVE if reserve is None else reserve
    remaining = remaining_requests(providers)
    estimated = estimate_research_requests(nt, prefer_chunked)
    verdict = {"allowed": True, "remaining": remaining, "estimated": estimated,
               "reserve": reserve, "workload": workload, "reason": None}
    if remaining is None:
        verdict["reason"] = "no_daily_cap_binds"
        return verdict
    if remaining - estimated >= reserve + workload:
        verdict["reason"] = "within_budget"
        return verdict
    verdict["allowed"] = False
    verdict["reason"] = "summary_reserve_protected"
    log_info(
        f"Research extraction deferred: {remaining} request(s) left today, extraction needs about "
        f"{estimated}, {reserve + workload} reserved for summaries (SUMMARY_REQUEST_RESERVE={reserve}"
        + (f" + workload {workload}" if workload else "") + ")."
    )
    return verdict
