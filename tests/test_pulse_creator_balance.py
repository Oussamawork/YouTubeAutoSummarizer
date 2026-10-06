"""The Monday pulse counts creators, not videos or claims (Oct 2026 audit: one
channel made 57-69% of a week's videos and views and set the mood, the
attention movers and the asset-class lean on its own)."""
from datetime import date

import canonical_claims as cc
import helpers
import market_pulse as mp
import pulse_charts
from signals_data import canonical_ticker

from tests.test_canonical_analytics import _claim

TODAY = date(2026, 7, 12)


def _inputs(claims, **kw):
    kw.setdefault("groups", {})
    return mp._canonical_pulse_inputs(7, TODAY, lambda *a: {}, claims=claims, **kw)


def test_channels_of_one_house_cast_one_vote():
    claims = [_claim("HKCM", "BTC", "bullish", "unspecified", video="v1", channel_id="UCde"),
              _claim("HKCM GLOBAL", "BTC", "bullish", "unspecified", video="v2", channel_id="UCen"),
              _claim("HKCM GLOBAL", "ETH", "bullish", "unspecified", video="v3", channel_id=None),
              _claim("Other", "BTC", "bearish", "unspecified", video="v4")]
    voter_of = cc.voter_resolver({"UCde": "HKCM", "UCen": "HKCM"}, claims)
    assert voter_of(claims[2]) == "HKCM"          # no channel id: matched by channel name
    entry = cc.aggregate_views_by_asset(claims, voter_of=voter_of)["BTC"]
    assert entry["votes"] == {"HKCM": "bullish", "Other": "bearish"}
    assert entry["vote_dates"]["HKCM"] == date(2026, 7, 10)
    assert cc.aggregate_views_by_asset(claims)["BTC"]["bull"] == 2   # without groups: two "creators"


def test_creator_mood_is_one_voice_per_creator_and_asset():
    prolific = [_claim("Busy", "NVDA", "bullish", "unspecified", video=f"b{i}") for i in range(5)]
    others = [_claim("X", "NVDA", "bearish", "unspecified", video="x1"),
              _claim("X", "AMD", "bearish", "unspecified", video="x1", n=1),
              _claim("Y", "TSLA", "bearish", "unspecified", video="y1"),
              _claim("Z", "AAPL", "bullish", "unspecified", video="z1"),
              _claim("Z", "MSFT", "bearish", "unspecified", video="z1", n=1)]
    moods = cc.creator_moods(prolific + others)
    assert moods == {"Busy": "bullish", "X": "bearish", "Y": "bearish", "Z": "mixed"}
    text = mp.build_canonical_pulse(_inputs(prolific + others), TODAY)
    # Five bullish videos from one channel do not make an upbeat week.
    assert text.splitlines()[1].startswith("Cautious week: 1 of 4 creators leaned bullish (8 videos).")
    weeks = mp.canonical_tone_weeks(prolific + others, TODAY)
    assert weeks[-1]["unit"] == "creators" and weeks[-1]["n"] == 4 and weeks[-1]["bearish"] == 2
    assert pulse_charts.tone_title(weeks) == "1 of 4 creators lean bullish this week"


def test_board_keeps_standing_views_and_says_how_many_are_fresh():
    claims = [_claim("A", "NVDA", "bullish", "unspecified", published="2026-07-10", video="a1"),
              _claim("B", "NVDA", "bullish", "unspecified", published="2026-06-22", video="b1"),
              # Only discussed weeks ago: not on this week's board.
              _claim("A", "PLTR", "bullish", "unspecified", published="2026-06-20", video="a0"),
              _claim("B", "PLTR", "bullish", "unspecified", published="2026-06-22", video="b1", n=1),
              # Older than the board's four weeks.
              _claim("C", "NVDA", "bearish", "unspecified", published="2026-06-01", video="c0")]
    inputs = _inputs(claims)
    assert set(inputs["board"]) == {"NVDA"} and inputs["board"]["NVDA"]["votes"] == {"A": "bullish", "B": "bullish"}
    agree = mp.build_canonical_pulse(inputs, TODAY).split("Where creators agree")[1].split("\n\n")[0]
    assert "NVDA — 2 of 2 creators bullish (1 this week)" in agree and "PLTR" not in agree


def test_a_creator_reversing_their_own_view_is_reported():
    claims = [_claim("A", "NVDA", "bullish", "unspecified", published="2026-06-25", video="a1"),
              _claim("A", "NVDA", "bearish", "unspecified", published="2026-07-10", video="a2"),
              # Different horizons are not a change of mind.
              _claim("B", "TSLA", "bullish", "long", published="2026-06-25", video="b1"),
              _claim("B", "TSLA", "bearish", "short", published="2026-07-10", video="b2"),
              # A split video is not a side.
              _claim("C", "AMD", "bullish", "unspecified", published="2026-06-25", video="c1"),
              _claim("C", "AMD", "bullish", "unspecified", published="2026-07-10", video="c2"),
              _claim("C", "AMD", "bearish", "unspecified", published="2026-07-10", video="c2", n=1)]
    changes = cc.view_changes(claims, date(2026, 7, 5))
    assert [(c["voter"], c["label"], c["before"], c["after"]) for c in changes] == [("A", "NVDA", "bullish", "bearish")]
    text = mp.build_canonical_pulse(_inputs(claims), TODAY)
    assert "Changed their mind\n🔄 A: NVDA bullish → bearish" in text


def test_attention_counts_creators_present_in_both_weeks():
    claims = [_claim(s, "MU", "bullish", "unspecified", published="2026-07-10", video=f"{s}1") for s in "ABC"]
    claims += [_claim("A", "MU", "bullish", "unspecified", published="2026-07-03", video="A0")]
    claims += [_claim(s, "META", "bullish", "unspecified", published="2026-07-03", video=f"{s}0", n=1)
               for s in "ABD"]
    claims += [_claim("B", "AMD", "bullish", "unspecified", published="2026-07-03", video="B0")]
    claims += [_claim("C", "AMD", "bullish", "unspecified", published="2026-07-03", video="C0")]
    inputs = _inputs(claims)
    up, down = mp._attention_movers(inputs["current_views"], inputs["previous_views"], inputs["active_both_weeks"])
    assert [(l, a, b) for _, l, a, b in up] == [("MU", 1, 3)]
    # META's drop counts A and B (here both weeks), not D, who posted nothing this week.
    assert [(l, a, b) for _, l, a, b in down] == [("AMD", 2, 0), ("META", 2, 0)]


def test_coverage_footer_names_the_channels_still_waiting():
    claims = [_claim("A", "NVDA", "bullish", "unspecified", video="v1"),
              _claim("B", "NVDA", "bullish", "unspecified", video="v2")]

    def gate(vid, outcome, name):
        return {"video_id": vid, "outcome": outcome, "channel_name": name,
                "published_at": "2026-07-10T09:00:00+00:00"}
    gates = [gate("v1", "model_quota_deferred", "A"), gate("v1", "included", "A"), gate("v2", "included", "B"),
             gate("v3", "model_quota_deferred", "TheStreet"), gate("v4", "model_quota_deferred", "TheStreet"),
             gate("v5", "included", "C"), gate("v6", "title_filtered", "C"), gate("v7", "included", "C")]
    state = {"videos": {"v5": {"research_status": "quota_deferred"}, "v7": {"research_status": "no_claims_found"}}}
    inputs = _inputs(claims, gates=gates, state=state)
    assert inputs["coverage"] == {"in_scope": 6, "analyzed": 3, "waiting": [("TheStreet", 2), ("C", 1)]}
    text = mp.build_canonical_pulse(inputs, TODAY)
    assert "⚠️ Partial week: 3 of 6 videos analyzed so far" in text
    assert "(3 of 6 videos analyzed)" in text and "Not analyzed yet: TheStreet 2, C 1" in text


def test_a_creator_family_is_named_in_the_footer():
    claims = [_claim("HKCM", "BTC", "bullish", "unspecified", video="v1", channel_id="UCde"),
              _claim("Phantom", "BTC", "bullish", "unspecified", video="v2", channel_id="UCph"),
              _claim("B", "BTC", "bullish", "unspecified", video="v3")]
    inputs = _inputs(claims, groups={"UCde": "HKCM", "UCph": "HKCM"})
    text = mp.build_canonical_pulse(inputs, TODAY)
    assert "BTC — 2 of 2 creators bullish" in text and "HKCM: one creator across its channels." in text
    assert inputs["channels"] == 2


def test_group_channel_option_is_parsed(tmp_path):
    p = tmp_path / "channels.txt"
    p.write_text("UCde digest lang=de group=HKCM # German\nUCen group=\n")
    entries = helpers.read_channels(str(p))
    assert entries[0]["group"] == "HKCM" and entries[0]["digest"] and entries[1]["group"] is None


def test_names_unresolved_in_the_audit_fold_to_their_tickers():
    assert canonical_ticker({"name": "Fortinet"}) == "FTNT"
    assert canonical_ticker({"name": "McDonald's stock"}) == "MCD"
    assert canonical_ticker({"name": "General Mills Incorporated"}) == "GIS"
    nasdack = _claim("A", None, "bullish", "unspecified", subject_mention="Nasdack", canonical_entity_name="Nasdack",
                     asset_type="index")
    assert cc.asset_key(nasdack) == "NASDAQ"
    assert cc.is_placeholder_asset("SUPPORT ZONE") and cc.is_placeholder_asset("STOCK")
