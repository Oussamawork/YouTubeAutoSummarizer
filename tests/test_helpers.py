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
        {"channel_id": "UCplain", "digest": False, "max_per_run": None, "only": []},
        {"channel_id": "UCdigest", "digest": True, "max_per_run": None, "only": []},
        {"channel_id": "UCcapped", "digest": False, "max_per_run": 5, "only": []},
        {"channel_id": "UCboth", "digest": True, "max_per_run": 2, "only": []},
    ]


def test_read_channels_ignores_bad_options(tmp_path):
    # A typo in an option must not lose the channel itself.
    p = tmp_path / "channels.txt"
    p.write_text("UCx digset max=oops\n")
    assert helpers.read_channels(str(p)) == [
        {"channel_id": "UCx", "digest": False, "max_per_run": None, "only": []}
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


# --- append_jsonl (market-signals persistence) ---


def test_append_jsonl_creates_dirs_and_appends(tmp_path):
    import json
    path = tmp_path / "data" / "signals.jsonl"
    assert helpers.append_jsonl(str(path), {"a": 1}) is True
    assert helpers.append_jsonl(str(path), {"b": "é"}) is True
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0]) == {"a": 1}
    assert json.loads(lines[1]) == {"b": "é"}


def test_append_jsonl_failure_returns_false(tmp_path):
    # Target path is a directory -> OSError, swallowed.
    assert helpers.append_jsonl(str(tmp_path), {"a": 1}) is False
    # Unserializable record -> TypeError, swallowed.
    assert helpers.append_jsonl(str(tmp_path / "f.jsonl"), {"x": {1, 2}}) is False


def test_read_channels_strips_inline_comments(tmp_path):
    path = tmp_path / "channels.txt"
    path.write_text(
        "# full-line comment\n"
        "UC111 digest max=5   # @somehandle\n"
        "UC222   # another name\n"
        "\n"
        "@handle3 digest\n",
        encoding="utf-8",
    )
    channels = helpers.read_channels(str(path))
    assert [c["channel_id"] for c in channels] == ["UC111", "UC222", "@handle3"]
    assert channels[0]["digest"] is True and channels[0]["max_per_run"] == 5
    assert channels[1]["digest"] is False and channels[1]["max_per_run"] is None
    assert channels[2]["digest"] is True


# --- Title filtering (only=) ---


def test_title_matches_whole_words_only():
    kw = ["btc", "bitcoin", "eth", "ethereum", "sol", "solana"]
    assert helpers.title_matches("Has Bitcoin Started the Next Sell-off?", kw)
    assert helpers.title_matches("ETH | Market Resistance Testing", kw)
    assert helpers.title_matches("$BTC/USD breakout", kw)          # punctuation is a boundary
    assert helpers.title_matches("Bitcoin's Secret 260-Day Cycle", kw)
    assert helpers.title_matches("Solana flips higher", kw)
    # The reason this uses word boundaries rather than substrings:
    assert not helpers.title_matches("Whether markets rally together", kw)  # 'eth' inside words
    assert not helpers.title_matches("Solve the console problem", kw)       # 'sol' inside words
    assert not helpers.title_matches("XRP Price Analysis", kw)
    assert not helpers.title_matches("HBAR Elliott Wave Analysis", kw)


def test_title_matches_no_keywords_keeps_everything():
    assert helpers.title_matches("anything at all", []) is True
    assert helpers.title_matches("anything at all", None) is True


def test_title_matches_empty_title():
    assert helpers.title_matches("", ["btc"]) is False
    assert helpers.title_matches(None, ["btc"]) is False


def test_title_matches_is_case_insensitive():
    assert helpers.title_matches("bitcoin rallies", ["BITCOIN"])
    assert helpers.title_matches("BITCOIN rallies", ["bitcoin"])


def test_read_channels_parses_only_option(tmp_path):
    path = tmp_path / "channels.txt"
    path.write_text(
        "UC1 digest only=btc,bitcoin\n"
        "UC2 only=ETH,Solana max=2\n"
        "UC3\n"
        "UC4 only=\n",  # malformed: ignored, no filter
        encoding="utf-8",
    )
    channels = helpers.read_channels(str(path))
    assert channels[0]["only"] == ["btc", "bitcoin"] and channels[0]["digest"] is True
    assert channels[1]["only"] == ["eth", "solana"] and channels[1]["max_per_run"] == 2
    assert channels[2]["only"] == []
    assert channels[3]["only"] == []
