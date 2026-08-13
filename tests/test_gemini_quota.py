"""Tests for per-model daily Gemini quota accounting (no network).

The counter's job is to stop a run from spending requests it doesn't have, and
— just as important — to never be the reason a summary doesn't happen. So the
failure modes matter as much as the happy path: a corrupt or unwritable counter
file must fail open, not block.
"""
from datetime import datetime, timezone

import gemini_quota


class TestDayBoundary:
    def test_pacific_not_utc(self):
        # 05:00 UTC is still the previous evening in California. Rolling the
        # counter on the UTC boundary would hand out a fresh 20 requests up to
        # eight hours before Google does.
        utc = datetime(2026, 8, 14, 5, 0, tzinfo=timezone.utc)
        assert gemini_quota.quota_day(utc).isoformat() == "2026-08-13"

    def test_after_pacific_midnight_it_is_the_new_day(self):
        utc = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)
        assert gemini_quota.quota_day(utc).isoformat() == "2026-08-14"


class TestCounting:
    def test_starts_empty(self):
        assert gemini_quota.used("gemini-3.6-flash") == 0
        assert gemini_quota.is_exhausted("gemini-3.6-flash") is False

    def test_record_increments_that_model_only(self):
        gemini_quota.record("gemini-3.6-flash")
        gemini_quota.record("gemini-3.6-flash")
        assert gemini_quota.used("gemini-3.6-flash") == 2
        # Each model has its own bucket; spending one must not spend another.
        assert gemini_quota.used("gemini-3.5-flash") == 0

    def test_exhausted_at_the_cap(self):
        for _ in range(gemini_quota.GEMINI_REQUESTS_PER_DAY):
            gemini_quota.record("gemini-3.7-flash")
        assert gemini_quota.is_exhausted("gemini-3.7-flash") is True

    def test_mark_exhausted_trusts_the_api_over_the_local_count(self):
        # A 429 means the real count is at least the cap, whatever we counted:
        # failed requests are never metered locally, so we can only undercount.
        gemini_quota.record("gemini-3.5-flash")
        gemini_quota.mark_exhausted("gemini-3.5-flash")
        assert gemini_quota.is_exhausted("gemini-3.5-flash") is True

    def test_counts_survive_a_new_process(self):
        # Each scheduled run is a fresh process on a fresh runner; the whole
        # point of the file is that the count outlives it.
        gemini_quota.record("gemini-3.6-flash")
        reloaded = gemini_quota.load_usage()
        assert reloaded["models"]["gemini-3.6-flash"] == 1

    def test_a_new_day_starts_fresh(self, monkeypatch):
        gemini_quota.mark_exhausted("gemini-3.6-flash")
        assert gemini_quota.is_exhausted("gemini-3.6-flash") is True
        real_day = gemini_quota.quota_day()
        monkeypatch.setattr(
            gemini_quota, "quota_day", lambda now=None: real_day.replace(day=real_day.day % 28 + 1)
        )
        assert gemini_quota.used("gemini-3.6-flash") == 0


class TestFailsOpen:
    def test_corrupt_file_is_treated_as_no_usage(self, tmp_path, monkeypatch):
        bad = tmp_path / "corrupt.json"
        bad.write_text("{not json")
        monkeypatch.setattr(gemini_quota, "GEMINI_USAGE_FILE", str(bad))
        # Refusing to call Gemini because a counter file is unreadable would
        # turn a cosmetic problem into an outage.
        assert gemini_quota.is_exhausted("gemini-3.6-flash") is False

    def test_non_object_file_is_treated_as_no_usage(self, tmp_path, monkeypatch):
        odd = tmp_path / "list.json"
        odd.write_text("[1, 2, 3]")
        monkeypatch.setattr(gemini_quota, "GEMINI_USAGE_FILE", str(odd))
        assert gemini_quota.used("gemini-3.6-flash") == 0

    def test_unwritable_path_does_not_raise(self, monkeypatch):
        monkeypatch.setattr(gemini_quota, "GEMINI_USAGE_FILE", "/proc/nope/usage.json")
        gemini_quota.record("gemini-3.6-flash")  # must not raise


class TestReport:
    def test_reports_remaining_per_model(self):
        gemini_quota.record("gemini-3.6-flash")
        line = gemini_quota.report()
        remaining = gemini_quota.GEMINI_REQUESTS_PER_DAY - 1
        assert f"gemini-3.6-flash={remaining}/{gemini_quota.GEMINI_REQUESTS_PER_DAY} left" in line

    def test_empty_when_nothing_spent(self):
        assert gemini_quota.report() == ""
