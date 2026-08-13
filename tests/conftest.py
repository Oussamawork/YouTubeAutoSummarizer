"""Shared test setup.

Network calls are mocked throughout this suite (CI and the dev sandbox cannot
reach YouTube or the LLM providers). The video-metadata lookup runs inside
main() for every channel, so it is stubbed out by default here: without this,
main() tests reach for googleapis.com and burn retry backoff on each one.
Tests that exercise the duration gate override it explicitly.
"""
import pytest

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

    monkeypatch.setattr(
        gemini_quota, "GEMINI_USAGE_FILE", str(tmp_path / "gemini_usage.json")
    )
