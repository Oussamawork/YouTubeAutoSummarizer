"""Tests for per-model daily Gemini quota accounting (no network).

The counter's job is to stop a run from spending requests it doesn't have, and
— just as important — to never be the reason a summary doesn't happen. So the
failure modes matter as much as the happy path: a corrupt or unwritable counter
file must fail open, not block.
"""
import json
from datetime import datetime, timedelta, timezone

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

    def test_being_capped_does_not_forge_a_request_count(self):
        # The flag and the tally answer different questions. Writing the cap
        # into the count made "20" mean either "20 requests served" or "the API
        # refused after 1", which left the file unable to say how much budget a
        # run actually used — the one thing it is committed for.
        gemini_quota.record("gemini-3.5-flash")
        gemini_quota.mark_exhausted("gemini-3.5-flash")
        assert gemini_quota.used("gemini-3.5-flash") == 1

    def test_report_shows_a_capped_model_that_never_served_a_request(self):
        gemini_quota.mark_exhausted("gemini-3.7-flash")
        line = gemini_quota.report()
        assert "gemini-3.7-flash" in line and "capped by API" in line

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


# Real payload shapes, trimmed. Google names the offending quota in the error
# body and nowhere else — the status code alone cannot tell a limit that clears
# in a minute from one that lasts until the Pacific reset.
PER_DAY_BODY = (
    '{"error":{"code":429,"status":"RESOURCE_EXHAUSTED","details":[{"@type":'
    '"type.googleapis.com/google.rpc.QuotaFailure","violations":[{"quotaId":'
    '"GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}}'
)
PER_MINUTE_BODY = (
    '{"error":{"code":429,"status":"RESOURCE_EXHAUSTED","details":[{"@type":'
    '"type.googleapis.com/google.rpc.QuotaFailure","violations":[{"quotaId":'
    '"GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]},{"@type":'
    '"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"27s"}]}}'
)


class TestClassify429:
    def test_per_day(self):
        assert gemini_quota.classify_429(PER_DAY_BODY) == "day"

    def test_per_minute(self):
        assert gemini_quota.classify_429(PER_MINUTE_BODY) == "minute"

    def test_token_per_minute_quota_is_still_a_minute(self):
        body = '{"violations":[{"quotaId":"GenerateContentInputTokensPerModelPerMinute"}]}'
        assert gemini_quota.classify_429(body) == "minute"

    def test_per_day_wins_when_both_are_named(self):
        # Waiting out the minute would only run into the day again, so the
        # longer limit is the binding one.
        assert gemini_quota.classify_429(PER_MINUTE_BODY + PER_DAY_BODY) == "day"

    def test_unnamed_quota_is_unknown(self):
        # Not "day": guessing the worst case here is what costs a model its
        # remaining requests for the rest of the day.
        assert gemini_quota.classify_429('{"error":"Too Many Requests"}') == "unknown"

    def test_empty_body(self):
        assert gemini_quota.classify_429("") == "unknown"
        assert gemini_quota.classify_429(None) == "unknown"


class TestRetryDelay:
    def test_reads_googles_retry_info(self):
        # Sent in the body, not the Retry-After header, so a header-only
        # reader falls back to a blind guess.
        assert gemini_quota.retry_delay_seconds(PER_MINUTE_BODY) == 27

    def test_absent(self):
        assert gemini_quota.retry_delay_seconds(PER_DAY_BODY) is None
        assert gemini_quota.retry_delay_seconds("") is None


class TestProvisionalWriteOff:
    """
    A 429 that names a per-day quota is the API's verdict, not a fact we can
    verify. On 2026-08-18 it retired gemini-3.7-flash at 00:41 Pacific after 3
    recorded requests; the model then sat idle for the rest of the day with 17
    of its 20 free requests unspent, and the evening's videos were deferred for
    "LLM quota reached" while the best model had budget left. So the verdict
    buys a cooling-off period, not the day.
    """

    def _later(self, minutes):
        return datetime.now(timezone.utc) + timedelta(minutes=minutes)

    def test_honored_immediately_after_the_verdict(self):
        gemini_quota.mark_exhausted("gemini-3.7-flash")
        assert gemini_quota.is_exhausted("gemini-3.7-flash") is True

    def test_retried_once_the_cooling_off_period_passes(self):
        gemini_quota.mark_exhausted("gemini-3.7-flash")
        later = self._later(gemini_quota.GEMINI_SPENT_RECHECK_MINUTES + 1)
        assert gemini_quota.is_exhausted("gemini-3.7-flash", now=later) is False

    def test_a_repeated_verdict_doubles_the_wait(self):
        # A model that really is out for the day should be left alone, not
        # probed by every two-hourly run for the next twenty hours.
        gemini_quota.mark_exhausted("gemini-3.6-flash")
        gemini_quota.mark_exhausted("gemini-3.6-flash")
        base = gemini_quota.GEMINI_SPENT_RECHECK_MINUTES
        assert gemini_quota.is_exhausted("gemini-3.6-flash", now=self._later(base + 1)) is True
        assert gemini_quota.is_exhausted("gemini-3.6-flash", now=self._later(base * 2 + 1)) is False

    def test_the_wait_is_capped(self):
        for _ in range(20):
            gemini_quota.mark_exhausted("gemini-3.6-flash")
        entry = gemini_quota.load_usage()["spent"]["gemini-3.6-flash"]
        assert gemini_quota.recheck_delay_minutes(entry) == (
            gemini_quota.GEMINI_SPENT_RECHECK_MAX_MINUTES
        )

    def test_a_served_request_clears_the_verdict(self):
        # The API answering is proof the write-off no longer holds, so the
        # model goes back into the chain instead of staying skipped.
        gemini_quota.mark_exhausted("gemini-3.7-flash")
        gemini_quota.record("gemini-3.7-flash")
        assert gemini_quota.is_exhausted("gemini-3.7-flash") is False
        assert "gemini-3.7-flash" not in gemini_quota.load_usage()["spent"]

    def test_the_counted_cap_is_never_provisional(self):
        # Requests we watched being served cannot be un-served: no amount of
        # waiting brings them back before the Pacific reset.
        for _ in range(gemini_quota.GEMINI_REQUESTS_PER_DAY):
            gemini_quota.record("gemini-3.5-flash")
        gemini_quota.mark_exhausted("gemini-3.5-flash")
        assert gemini_quota.is_exhausted("gemini-3.5-flash", now=self._later(10000)) is True

    def test_the_verdict_records_what_it_overruled(self):
        # "Written off after 3 of 20" is the whole diagnosis; the tally alone
        # cannot say it, because later models keep incrementing the file.
        gemini_quota.record("gemini-3.7-flash")
        gemini_quota.record("gemini-3.7-flash")
        gemini_quota.mark_exhausted("gemini-3.7-flash")
        assert gemini_quota.load_usage()["spent"]["gemini-3.7-flash"]["used"] == 2

    def test_report_says_when_a_capped_model_comes_back(self):
        gemini_quota.mark_exhausted("gemini-3.7-flash")
        line = gemini_quota.report()
        assert "capped by API" in line and "retried after" in line


class TestSpentSchemaMigration:
    """
    One run writes the counter file and the next run reads it, so a deploy
    always meets a file written by the previous version.
    """

    def test_v1_list_is_read_as_a_write_off_due_for_a_recheck(self, tmp_path, monkeypatch):
        path = tmp_path / "usage.json"
        path.write_text(json.dumps({
            "day": gemini_quota.quota_day().isoformat(),
            "models": {"gemini-3.7-flash": 3},
            "spent": ["gemini-3.7-flash"],
        }))
        monkeypatch.setattr(gemini_quota, "GEMINI_USAGE_FILE", str(path))
        # No timestamp survives the migration, and a flag of unknown age is
        # worth one rejected request to re-test — not a day of the best model.
        assert gemini_quota.is_exhausted("gemini-3.7-flash") is False
        assert gemini_quota.used("gemini-3.7-flash") == 3

    def test_a_junk_spent_value_does_not_block_the_model(self, tmp_path, monkeypatch):
        path = tmp_path / "usage.json"
        path.write_text(json.dumps({
            "day": gemini_quota.quota_day().isoformat(),
            "models": {},
            "spent": "gemini-3.7-flash",
        }))
        monkeypatch.setattr(gemini_quota, "GEMINI_USAGE_FILE", str(path))
        assert gemini_quota.is_exhausted("gemini-3.7-flash") is False


class TestViolationSummary:
    """
    The logged 429 preview stops at 200 characters, which lands inside Google's
    generic "You exceeded your current quota" sentence — so the logs could not
    say whether a model was written off for requests-per-day or something else
    entirely. The named quota is pulled out and logged separately.
    """

    def test_names_the_quota(self):
        assert gemini_quota.violation_summary(PER_DAY_BODY) == (
            "quota=GenerateRequestsPerDayPerProjectPerModel-FreeTier"
        )

    def test_includes_the_limit_when_given(self):
        body = (
            '{"violations":[{"quotaMetric":"generativelanguage.googleapis.com/'
            'generate_content_free_tier_requests","quotaId":'
            '"GenerateRequestsPerDayPerProjectPerModel-FreeTier","quotaValue":"20"}]}'
        )
        summary = gemini_quota.violation_summary(body)
        assert "GenerateRequestsPerDayPerProjectPerModel-FreeTier" in summary
        assert "limit=20" in summary

    def test_empty_when_the_body_names_nothing(self):
        assert gemini_quota.violation_summary('{"error":"Too Many Requests"}') == ""
        assert gemini_quota.violation_summary(None) == ""


def test_report_marks_a_verdict_that_is_already_due_for_a_retry(tmp_path, monkeypatch):
    # A migrated flag has no timestamp, so it is retried on sight. Saying
    # "retried after 45 min" there would describe a wait that is not happening.
    path = tmp_path / "usage.json"
    path.write_text(json.dumps({
        "day": gemini_quota.quota_day().isoformat(),
        "models": {"gemini-3.7-flash": 3},
        "spent": ["gemini-3.7-flash"],
    }))
    monkeypatch.setattr(gemini_quota, "GEMINI_USAGE_FILE", str(path))
    assert "due for a retry" in gemini_quota.report()
