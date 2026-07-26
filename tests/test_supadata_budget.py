import json, os
from datetime import date
import pytest
import transcript as tr


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "SUPADATA_USAGE_FILE", str(tmp_path / "usage.json"))
    for name in ("SUPADATA_API_KEYS", "SUPADATA_API_KEY", "SUPADATA_API_KEY_2",
                 "SUPADATA_API_KEY_3", "SUPADATA_MONTHLY_BUDGET", "SUPADATA_CREDITS_PER_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_keys_collected_and_deduped(monkeypatch):
    monkeypatch.setenv("SUPADATA_API_KEY", "a")
    monkeypatch.setenv("SUPADATA_API_KEY_2", "b")
    assert tr._supadata_keys() == ["a", "b"]
    monkeypatch.setenv("SUPADATA_API_KEYS", "b, c")
    assert tr._supadata_keys() == ["b", "c", "a"]  # csv first, then numbered, deduped
    monkeypatch.delenv("SUPADATA_API_KEYS")
    monkeypatch.setenv("SUPADATA_API_KEY_2", "a")  # duplicate of key 1
    assert tr._supadata_keys() == ["a"]


def test_monthly_budget_scales_with_keys(monkeypatch):
    monkeypatch.setenv("SUPADATA_API_KEY", "a")
    assert tr.monthly_budget() == 100
    monkeypatch.setenv("SUPADATA_API_KEY_2", "b")
    assert tr.monthly_budget() == 200          # two free tiers
    monkeypatch.setenv("SUPADATA_MONTHLY_BUDGET", "50")
    assert tr.monthly_budget() == 50           # explicit override wins


def test_daily_allowance_paces_over_the_month(monkeypatch):
    monkeypatch.setenv("SUPADATA_API_KEY", "a")
    monkeypatch.setenv("SUPADATA_API_KEY_2", "b")   # 200/month
    # 1st of a 31-day month, nothing used: 200 // 31 = 6 per day.
    assert tr.daily_allowance({"count": 0}, today=date(2026, 7, 1)) == 6
    # 100 spent with 6 days left would pace to 16/day, but the per-day ceiling
    # (budget/28) keeps a late-cycle spike from draining the pool in two runs.
    assert tr.daily_allowance({"count": 100}, today=date(2026, 7, 26)) == 200 // 28
    # Budget spent -> nothing allowed.
    assert tr.daily_allowance({"count": 200}, today=date(2026, 7, 26)) == 0
    # Near the end of the cycle the remainder is still ceiling-capped.
    assert tr.daily_allowance({"count": 190}, today=date(2026, 7, 31)) == 7


def test_usage_resets_on_new_month_and_day(monkeypatch, tmp_path):
    path = tmp_path / "usage.json"
    path.write_text(json.dumps({"cycle": "2026-06-01", "count": 99, "day": "2026-06-30", "day_count": 5}))
    usage = tr._load_usage()          # today is not in that cycle
    assert usage["count"] == 0 and usage["day_count"] == 0


def test_budget_blocks_call_and_signals_exhaustion(monkeypatch):
    monkeypatch.setenv("SUPADATA_API_KEY", "a")
    monkeypatch.setenv("SUPADATA_MONTHLY_BUDGET", "1")
    current = tr._load_usage()
    tr._save_usage({"cycle": current["cycle"], "count": 1,
                    "day": current["day"], "day_count": 1})
    monkeypatch.setattr(
        tr.requests, "get",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not spend a credit")),
    )
    text, exhausted, reason = tr._fetch_supadata("vid00000001")
    assert text == "" and exhausted is True and reason == "budget_paced"


def test_successful_fetch_meters_one_credit(monkeypatch):
    monkeypatch.setenv("SUPADATA_API_KEY", "a")

    class R:
        status_code = 200
        text = ""

        def json(self):
            return {"content": "hello transcript"}

    monkeypatch.setattr(tr.requests, "get", lambda *a, **k: R())
    text, exhausted, reason = tr._fetch_supadata("vid00000001")
    assert text == "hello transcript" and exhausted is False and reason == "ok"
    assert tr._load_usage()["count"] == 1


def test_rotates_to_second_key_when_first_is_out_of_credits(monkeypatch):
    monkeypatch.setenv("SUPADATA_API_KEY", "spent")
    monkeypatch.setenv("SUPADATA_API_KEY_2", "fresh")
    seen = []

    def fake_get(url, headers=None, params=None, timeout=None):
        seen.append(headers["x-api-key"])

        class R:
            status_code = 402 if headers["x-api-key"] == "spent" else 200
            text = "no credits"

            def json(self):
                return {"content": "second key transcript"}
        return R()

    monkeypatch.setattr(tr.requests, "get", fake_get)
    text, exhausted, reason = tr._fetch_supadata("vid00000001")
    assert text == "second key transcript" and exhausted is False and reason == "ok"
    assert seen == ["spent", "fresh"]
    # Only the delivering request costs a credit; the rejection does not.
    assert tr._load_usage()["count"] == 1


def test_get_transcript_reports_budget_exhaustion(monkeypatch):
    monkeypatch.setattr(tr, "_fetch_supadata", lambda vid: ("", True, "no_credits"))
    monkeypatch.setattr(tr, "_fetch_youtube_transcript_api", lambda vid: "")
    out = tr.get_transcript_from_video("https://www.youtube.com/watch?v=vid00000001")
    assert out == {"transcript": "", "budget_exhausted": True, "reason": "no_credits"}


def test_fallback_success_clears_budget_flag(monkeypatch):
    monkeypatch.setattr(tr, "_fetch_supadata", lambda vid: ("", True, "no_credits"))
    monkeypatch.setattr(tr, "_fetch_youtube_transcript_api", lambda vid: "from fallback")
    out = tr.get_transcript_from_video("https://www.youtube.com/watch?v=vid00000001")
    assert out == {"transcript": "from fallback", "budget_exhausted": False,
                   "reason": "fallback_ok"}


def test_all_keys_out_of_credits_defers_instead_of_writing_off(monkeypatch):
    # Exhaustion must not look like "this video has no captions", or the video
    # is eventually written off permanently.
    monkeypatch.setenv("SUPADATA_API_KEY", "spent1")
    monkeypatch.setenv("SUPADATA_API_KEY_2", "spent2")

    class R:
        status_code = 402
        text = "no credits"

    monkeypatch.setattr(tr.requests, "get", lambda *a, **k: R())
    assert tr._fetch_supadata("vid00000001") == ("", True, "no_credits")


def test_credit_rejections_and_retries_are_not_metered(monkeypatch):
    # Only answers that actually consume a credit may count, otherwise we
    # defer videos while credits remain.
    monkeypatch.setenv("SUPADATA_API_KEY", "spent")

    class R:
        status_code = 402
        text = "no credits"

    monkeypatch.setattr(tr.requests, "get", lambda *a, **k: R())
    tr._fetch_supadata("vid00000001")
    assert tr._load_usage()["count"] == 0

    calls = []

    def flaky(*a, **k):
        calls.append(1)

        class T:
            status_code = 503 if len(calls) < 3 else 200
            text = "busy"

            def json(self):
                return {"content": "ok"}
        return T()

    monkeypatch.setattr(tr.requests, "get", flaky)
    monkeypatch.setattr(tr.time, "sleep", lambda s: None)
    text, _, _ = tr._fetch_supadata("vid00000002")
    assert text == "ok"
    assert tr._load_usage()["count"] == 1  # 3 requests, one credit


# --- Billing cycle (credits reset on the plan's anniversary day, not the 1st) ---


def test_cycle_bounds_follow_the_reset_day(monkeypatch):
    monkeypatch.setattr(tr, "SUPADATA_RESET_DAY", 17)
    # On/after the reset day: cycle started this month.
    assert tr.cycle_bounds(date(2026, 7, 26)) == (date(2026, 7, 17), date(2026, 8, 17))
    assert tr.cycle_bounds(date(2026, 7, 17)) == (date(2026, 7, 17), date(2026, 8, 17))
    # Before it: cycle started last month.
    assert tr.cycle_bounds(date(2026, 7, 16)) == (date(2026, 6, 17), date(2026, 7, 17))
    # Year boundary.
    assert tr.cycle_bounds(date(2026, 12, 20)) == (date(2026, 12, 17), date(2027, 1, 17))


def test_cycle_bounds_default_is_calendar_month(monkeypatch):
    monkeypatch.setattr(tr, "SUPADATA_RESET_DAY", 1)
    assert tr.cycle_bounds(date(2026, 7, 26)) == (date(2026, 7, 1), date(2026, 8, 1))


def test_allowance_paces_over_the_billing_cycle_not_the_month(monkeypatch):
    monkeypatch.setattr(tr, "SUPADATA_RESET_DAY", 17)
    monkeypatch.setenv("SUPADATA_MONTHLY_BUDGET", "300")
    # Jul 26 with 86 spent: 214 left over 22 days to the Aug 17 reset -> 9/day.
    # A calendar-month assumption would have said 214/6 = 35/day and drained it.
    assert tr.daily_allowance({"count": 86}, today=date(2026, 7, 26)) == 9


def test_allowance_ceiling_survives_a_wrong_reset_day(monkeypatch):
    # Even with the default reset day (wrong for this plan), the per-day
    # ceiling of budget/28 stops the pool being drained in one run.
    monkeypatch.setattr(tr, "SUPADATA_RESET_DAY", 1)
    monkeypatch.setenv("SUPADATA_MONTHLY_BUDGET", "300")
    assert tr.daily_allowance({"count": 86}, today=date(2026, 7, 31)) == 300 // 28


def test_usage_counters_reset_on_cycle_rollover(monkeypatch, tmp_path):
    monkeypatch.setattr(tr, "SUPADATA_RESET_DAY", 17)
    path = tmp_path / "usage.json"
    monkeypatch.setattr(tr, "SUPADATA_USAGE_FILE", str(path))
    path.write_text(json.dumps({"cycle": "2026-06-17", "count": 250,
                                "day": "2026-07-16", "day_count": 9}))
    # Still inside the old cycle: counters kept.
    assert tr._load_usage(date(2026, 7, 16))["count"] == 250
    # Past the reset day: fresh cycle, counters cleared.
    assert tr._load_usage(date(2026, 7, 17))["count"] == 0


def test_empty_200_is_reported_as_empty_content(monkeypatch):
    """The most valuable diagnostic: Supadata served a response but had no
    captions for the video — a credit spent for nothing."""
    monkeypatch.setenv("SUPADATA_API_KEY", "a")

    class R:
        status_code = 200
        text = ""

        def json(self):
            return {"content": ""}

    monkeypatch.setattr(tr.requests, "get", lambda *a, **k: R())
    text, exhausted, reason = tr._fetch_supadata("vid00000001")
    assert (text, exhausted, reason) == ("", False, "empty_content")
    assert tr._load_usage()["count"] == 1  # still charged


def test_http_error_reason_carries_status(monkeypatch):
    monkeypatch.setenv("SUPADATA_API_KEY", "a")

    class R:
        status_code = 404
        text = "not found"

    monkeypatch.setattr(tr.requests, "get", lambda *a, **k: R())
    assert tr._fetch_supadata("vid00000001")[2] == "http_404"
