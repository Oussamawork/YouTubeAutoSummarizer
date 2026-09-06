"""Shared test setup.

Network calls are mocked throughout this suite (CI and the dev sandbox cannot
reach YouTube or the LLM providers). The video-metadata lookup runs inside
main() for every channel, so it is stubbed out by default here: without this,
main() tests reach for googleapis.com and burn retry backoff on each one.
Tests that exercise the duration gate override it explicitly.
"""
import pytest

import market_pulse
import price_cache
import scraper

# Captured before the autouse fixture can replace it, so tests that exercise
# the lookup itself still reach the real implementation.
_REAL_FETCH_VIDEO_DETAILS = scraper.fetch_video_details


@pytest.fixture(autouse=True)
def _no_video_metadata_lookup(monkeypatch):
    # {} means "no metadata", and filter_by_duration fails open on that, so
    # this default leaves behavior unchanged.
    monkeypatch.setattr(scraper, "fetch_video_details", lambda key, ids: {})


@pytest.fixture
def real_fetch_video_details():
    """The real scraper.fetch_video_details, for tests that target it."""
    return _REAL_FETCH_VIDEO_DETAILS


@pytest.fixture(autouse=True)
def _isolated_gemini_quota(tmp_path, monkeypatch):
    """Per-model daily counters go to a scratch file for every test.

    The real one is committed state: a test must neither be influenced by
    yesterday's counts nor write today's.
    """
    import gemini_quota
    import transcript

    monkeypatch.setattr(
        gemini_quota, "GEMINI_USAGE_FILE", str(tmp_path / "gemini_usage.json")
    )
    # Run-scoped rate-limit memory is process state, so it leaks between tests
    # unless it is reset with the counter it complements.
    monkeypatch.setattr(transcript, "_RATE_LIMITED_THIS_RUN", set())


@pytest.fixture(autouse=True)
def _isolated_price_cache():
    """Isolate the process-wide daily-close cache.

    Sharing it across a run is deliberate in production (a run must never
    refetch a symbol it already has), but that same sharing would let one test
    answer another's price lookup from cache instead of the mock it installed.
    """
    price_cache.reset({})
    yield
    price_cache.reset(None)


@pytest.fixture(autouse=True)
def _isolated_ticker_map():
    """Keep the learned ticker map out of tests: it is committed data, and a
    real entry would silently change what canonical_ticker returns."""
    market_pulse.reset_learned_tickers({})
    yield
    market_pulse.reset_learned_tickers(None)


@pytest.fixture(autouse=True)
def _offline_model_metadata(tmp_path, monkeypatch):
    """No live capability or token-count lookups in tests, and every research
    data product goes to a scratch directory.

    The live paths hit generativelanguage.googleapis.com; the registry and the
    conservative estimate are what the suite exercises. Research state,
    transcripts, partials and the capability cache are committed data in
    production, so a test must never read or write the real files.
    """
    import model_capabilities
    import research_state
    import signals
    import summarizer
    import token_budget
    import transcript_store

    monkeypatch.setattr(model_capabilities, "LIVE_CAPABILITIES", False)
    monkeypatch.setattr(model_capabilities, "CAPABILITIES_CACHE_FILE",
                        str(tmp_path / "model_capabilities.json"))
    model_capabilities.reset_cache()
    monkeypatch.setattr(token_budget, "LIVE_COUNT_TOKENS", False)
    token_budget.reset_count_cache()
    monkeypatch.setattr(research_state, "RESEARCH_DIR", str(tmp_path / "research"))
    monkeypatch.setattr(transcript_store, "TRANSCRIPTS_DIR", str(tmp_path / "transcripts"))
    monkeypatch.setattr(transcript_store, "TRANSCRIPT_INDEX",
                        str(tmp_path / "research" / "transcript_records.jsonl"))
    monkeypatch.setattr(summarizer, "PARTIALS_DIR", str(tmp_path / "partials"))
    # Existing orchestration tests mock generation only. Review tests opt in
    # explicitly and mock the additional calls; no test spends live quota.
    monkeypatch.setenv("SUMMARY_REVIEW_ENABLED", "false")
    monkeypatch.setenv("SUMMARY_REVIEW_FILE", str(tmp_path / "summary_reviews.jsonl"))
    monkeypatch.setattr(scraper, "SIGNALS_FILE", str(tmp_path / "signals.jsonl"))
    monkeypatch.setattr(signals, "EXHAUSTIVE_RESEARCH_MODE", False)
    yield
    model_capabilities.reset_cache()
    token_budget.reset_count_cache()
