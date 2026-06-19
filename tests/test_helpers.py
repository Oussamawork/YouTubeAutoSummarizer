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


def test_seen_videos_roundtrip_atomic(tmp_path):
    p = tmp_path / "seen.json"
    data = {"UCabc": "vid1", "UCdef": "vid2"}
    helpers.save_seen_videos(str(p), data)
    assert helpers.load_seen_videos(str(p)) == data
    # The atomic write must not leave a temp file behind.
    leftovers = [f for f in os.listdir(tmp_path) if f.startswith(".seen_")]
    assert leftovers == []
