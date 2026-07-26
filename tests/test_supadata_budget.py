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
    # Same month, 100 already spent with 6 days left: 100 // 6 = 16.
    assert tr.daily_allowance({"count": 100}, today=date(2026, 7, 26)) == 16
    # Budget spent -> nothing allowed.
    assert tr.daily_allowance({"count": 200}, today=date(2026, 7, 26)) == 0
    # Last day of the month gets whatever is left.
    assert tr.daily_allowance({"count": 190}, today=date(2026, 7, 31)) == 10


def test_usage_resets_on_new_month_and_day(monkeypatch, tmp_path):
    path = tmp_path / "usage.json"
    path.write_text(json.dumps({"month": "2026-06", "count": 99, "day": "2026-06-30", "day_count": 5}))
    usage = tr._load_usage()          # today is not June
    assert usage["count"] == 0 and usage["day_count"] == 0


def test_budget_blocks_call_and_signals_exhaustion(monkeypatch):
    monkeypatch.setenv("SUPADATA_API_KEY", "a")
    monkeypatch.setenv("SUPADATA_MONTHLY_BUDGET", "1")
    tr._save_usage({"month": tr._load_usage()["month"], "count": 1,
                    "day": tr._load_usage()["day"], "day_count": 1})
    monkeypatch.setattr(
        tr.requests, "get",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not spend a credit")),
    )
    text, exhausted = tr._fetch_supadata("vid00000001")
    assert text == "" and exhausted is True


def test_successful_fetch_meters_one_credit(monkeypatch):
    monkeypatch.setenv("SUPADATA_API_KEY", "a")

    class R:
        status_code = 200
        text = ""

        def json(self):
            return {"content": "hello transcript"}

    monkeypatch.setattr(tr.requests, "get", lambda *a, **k: R())
    text, exhausted = tr._fetch_supadata("vid00000001")
    assert text == "hello transcript" and exhausted is False
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
    text, exhausted = tr._fetch_supadata("vid00000001")
    assert text == "second key transcript" and exhausted is False
    assert seen == ["spent", "fresh"]
    # Only the delivering request costs a credit; the rejection does not.
    assert tr._load_usage()["count"] == 1


def test_get_transcript_reports_budget_exhaustion(monkeypatch):
    monkeypatch.setattr(tr, "_fetch_supadata", lambda vid: ("", True))
    monkeypatch.setattr(tr, "_fetch_youtube_transcript_api", lambda vid: "")
    out = tr.get_transcript_from_video("https://www.youtube.com/watch?v=vid00000001")
    assert out == {"transcript": "", "budget_exhausted": True}


def test_fallback_success_clears_budget_flag(monkeypatch):
    monkeypatch.setattr(tr, "_fetch_supadata", lambda vid: ("", True))
    monkeypatch.setattr(tr, "_fetch_youtube_transcript_api", lambda vid: "from fallback")
    out = tr.get_transcript_from_video("https://www.youtube.com/watch?v=vid00000001")
    assert out == {"transcript": "from fallback", "budget_exhausted": False}


def test_all_keys_out_of_credits_defers_instead_of_writing_off(monkeypatch):
    # Exhaustion must not look like "this video has no captions", or the video
    # is eventually written off permanently.
    monkeypatch.setenv("SUPADATA_API_KEY", "spent1")
    monkeypatch.setenv("SUPADATA_API_KEY_2", "spent2")

    class R:
        status_code = 402
        text = "no credits"

    monkeypatch.setattr(tr.requests, "get", lambda *a, **k: R())
    assert tr._fetch_supadata("vid00000001") == ("", True)


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
    text, _ = tr._fetch_supadata("vid00000002")
    assert text == "ok"
    assert tr._load_usage()["count"] == 1  # 3 requests, one credit
