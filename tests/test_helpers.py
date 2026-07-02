"""Tests for summary cleaning and dedup-state persistence."""
import os

import helpers


def test_empty_returns_empty():
    assert helpers.clean_summary("") == ""


def test_collapses_spaces_keeps_newlines():
    assert helpers.clean_summary("a    b\nc") == "a b\nc"


def test_collapses_excess_blank_lines():
    assert helpers.clean_summary("a\n\n\n\nb") == "a\n\nb"


def test_strips_trailing_whitespace_per_line():
    assert helpers.clean_summary("a   \nb  ") == "a\nb"


def test_preserves_bullet_structure():
    s = "Overview\n\n• one\n• two"
    assert helpers.clean_summary(s) == "Overview\n\n• one\n• two"


def test_nbsp_replaced():
    # Non-breaking space (U+00A0) should become a regular space.
    assert helpers.clean_summary("a" + chr(0x00A0) + "b") == "a b"


def test_state_roundtrip_atomic(tmp_path):
    p = tmp_path / "seen.json"
    state = {
        "channels": {"UCabc": {"last_video_id": "vid1", "last_published": "2026-01-01T00:00:00+00:00"}},
        "pending": {"vid2": {"channel_id": "UCdef", "attempts": 1}},
    }
    helpers.save_state(str(p), state)
    assert helpers.load_state(str(p)) == state
    # The atomic write must not leave a temp file behind.
    leftovers = [f for f in os.listdir(tmp_path) if f.startswith(".seen_")]
    assert leftovers == []


def test_load_state_missing_file(tmp_path):
    assert helpers.load_state(str(tmp_path / "nope.json")) == {"channels": {}, "pending": {}}


def test_load_state_corrupt_file(tmp_path):
    p = tmp_path / "seen.json"
    p.write_text("{not json")
    assert helpers.load_state(str(p)) == {"channels": {}, "pending": {}}


def test_load_state_migrates_v1(tmp_path):
    # v1 files were a flat {channel_id: last_video_id} map.
    p = tmp_path / "seen.json"
    p.write_text('{"UCabc": "vid1", "UCdef": "vid2"}')
    state = helpers.load_state(str(p))
    assert state == {
        "channels": {
            "UCabc": {"last_video_id": "vid1"},
            "UCdef": {"last_video_id": "vid2"},
        },
        "pending": {},
    }


def test_load_state_tolerates_wrong_shapes(tmp_path):
    p = tmp_path / "seen.json"
    p.write_text('{"channels": [1, 2], "pending": "nope"}')
    assert helpers.load_state(str(p)) == {"channels": {}, "pending": {}}


def test_read_channels_plain_and_options(tmp_path):
    p = tmp_path / "channels.txt"
    p.write_text(
        "# comment\n"
        "\n"
        "UCplain\n"
        "UCdigest digest\n"
        "UCcapped max=5\n"
        "UCboth digest max=2\n"
    )
    assert helpers.read_channels(str(p)) == [
        {"channel_id": "UCplain", "digest": False, "max_per_run": None},
        {"channel_id": "UCdigest", "digest": True, "max_per_run": None},
        {"channel_id": "UCcapped", "digest": False, "max_per_run": 5},
        {"channel_id": "UCboth", "digest": True, "max_per_run": 2},
    ]


def test_read_channels_ignores_bad_options(tmp_path):
    # A typo in an option must not lose the channel itself.
    p = tmp_path / "channels.txt"
    p.write_text("UCx digset max=oops\n")
    assert helpers.read_channels(str(p)) == [
        {"channel_id": "UCx", "digest": False, "max_per_run": None}
    ]


def test_read_channels_missing_file():
    assert helpers.read_channels("does-not-exist.txt") == []


def test_env_int_empty_string_falls_back(monkeypatch):
    # An unconfigured GitHub Actions repo variable arrives as "" (not absent);
    # this crashed the first live run of the new pipeline.
    monkeypatch.setenv("X_TEST_INT", "")
    assert helpers.env_int("X_TEST_INT", 3) == 3
    monkeypatch.setenv("X_TEST_INT", "  ")
    assert helpers.env_int("X_TEST_INT", 3) == 3
    monkeypatch.setenv("X_TEST_INT", "garbage")
    assert helpers.env_int("X_TEST_INT", 3) == 3
    monkeypatch.setenv("X_TEST_INT", "7")
    assert helpers.env_int("X_TEST_INT", 3) == 7
    monkeypatch.delenv("X_TEST_INT")
    assert helpers.env_int("X_TEST_INT", 3) == 3


def test_env_float_empty_string_falls_back(monkeypatch):
    monkeypatch.setenv("X_TEST_FLOAT", "")
    assert helpers.env_float("X_TEST_FLOAT", 0.3) == 0.3
    monkeypatch.setenv("X_TEST_FLOAT", "0.7")
    assert helpers.env_float("X_TEST_FLOAT", 0.3) == 0.7
