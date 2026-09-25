"""Regressions from the 2026-09-21 weekly pulse: bogus price targets,
"[unspecified]" labels, unresolved subjects counted as assets, names in the
ticker field, and an unreadable portfolio-disclosure list."""
from datetime import date

import canonical_claims as cc
import market_pulse as mp
import pulse_charts
from signals_data import canonical_ticker

from tests.test_canonical_analytics import _claim


def test_revenue_percent_and_entry_levels_are_not_price_targets():
    revenue = _claim("A", "NVDA", "bullish", "unspecified", claim_type="forecast", forecast_metric="revenue",
                     target_kind="absolute_value", target_value=50e9, currency="USD")
    percent = _claim("B", "XRP", "bullish", "unspecified", video="vb", claim_type="forecast", forecast_metric="price",
                     target_kind="range", target_low=90, target_high=148, target_unit="percent")
    entry = _claim("C", "NVDA", "bullish", "unspecified", video="vc", claim_type="recommendation",
                   target_kind="absolute_value", target_value=200.0)
    real = _claim("D", "META", "bullish", "unspecified", video="vd", claim_type="price_target",
                  forecast_metric="price", target_kind="absolute_value", target_value=800.0)
    assert [cc.price_target_of(c) for c in (revenue, percent, entry, real)] == [None, None, None, 800.0]
    views = cc.aggregate_views([revenue, percent, entry, real])
    assert views[("NVDA", "unspecified")]["targets"] == []
    assert views[("META", "unspecified")]["targets"] == [800.0]


def test_implausible_target_is_dropped_against_the_latest_price():
    assert pulse_charts.plausible_targets([50e9, 250.0], 222.0) == [250.0]
    assert pulse_charts.plausible_targets([50e9], None) == [50e9]
    entry = {"label": "NVDA", "targets": [50e9, 250.0], "bull": 1, "bear": 0, "neutral": 0,
             "bull_w": 1.0, "bear_w": 0.0, "neutral_w": 0.0,
             "mentions": 1, "channels": {"A"}, "actions": {}}
    line = mp._format_asset_line(entry, 222.0)
    assert "avg target 250" in line and "50,000" not in line
    rows = pulse_charts.upside_rows({"NVDA": entry}, {"NVDA": 222.0})
    assert rows[0]["target"] == 250.0 and rows[0]["n_targets"] == 1


def test_labels_carry_no_unspecified_tag():
    views = cc.aggregate_views([_claim("A", "NVDA", "bullish", "unspecified"),
                                _claim("B", "NVDA", "bullish", "long", video="vb")])
    assert views[("NVDA", "unspecified")]["label"] == "NVDA"
    assert views[("NVDA", "long")]["label"] == "NVDA (long-term)"


def test_unresolved_reference_phrases_are_not_assets():
    placeholder = _claim("A", None, "bullish", "unspecified", canonical_entity_name="this business",
                         subject_mention="this business")
    cash = _claim("B", None, "bullish", "unspecified", video="vb", canonical_entity_name="Cash",
                  subject_mention="Cash")
    trade_desk = _claim("C", None, "bullish", "unspecified", video="vc", canonical_entity_name="The Trade Desk",
                        subject_mention="The Trade Desk")
    assert cc.asset_key(placeholder) is None and cc.asset_key(cash) is None
    assert cc.asset_key(trade_desk) == "TTD"
    assert set(cc.aggregate_views([placeholder, cash, trade_desk])) == {("TTD", "unspecified")}


def test_names_recorded_as_tickers_and_name_suffixes_fold_to_the_ticker():
    assert canonical_ticker({"ticker": "SOLANA", "name": "Solana"}) == "SOL"
    assert canonical_ticker({"ticker": "META", "name": "Meta"}) == "META"
    assert canonical_ticker({"name": "Amazon stock"}) == "AMZN"
    assert canonical_ticker({"name": "Chainlink"}) == "LINK"
    assert canonical_ticker({"name": "Dutch Bros"}) == "BROS"


def test_new_on_radar_is_per_asset_not_per_horizon():
    old = _claim("A", "NVDA", "bullish", "unspecified", published="2026-06-20")
    new_horizon = [_claim("A", "NVDA", "bullish", "long", published="2026-07-10", video="v2"),
                   _claim("B", "NVDA", "bearish", "short", published="2026-07-10", video="v3")]
    fresh = [_claim("C", "PLTR", "bullish", "unspecified", published="2026-07-10", video="v4"),
             _claim("D", "PLTR", "bearish", "unspecified", published="2026-07-10", video="v5")]
    inputs = mp._canonical_pulse_inputs(7, date(2026, 7, 12), lambda *a: {}, claims=[old] + new_horizon + fresh)
    assert "NVDA" in inputs["older_keys"]
    text = mp.build_canonical_pulse(inputs, date(2026, 7, 12))
    radar = text.split("New on the radar")[1].split("\n\n")[0]
    assert "PLTR" in radar and "NVDA" not in radar


def test_portfolio_disclosures_group_per_source_and_drop_non_assets():
    rows = [
        {"source": "P", "asset": "MSFT", "ticker": "MSFT", "position": "owns_unspecified", "review_required": False},
        {"source": "P", "asset": "ADBE", "ticker": "ADBE", "position": "owns_unspecified", "review_required": False},
        {"source": "P", "asset": "TSLA", "ticker": "TSLA", "position": "owns_unspecified", "review_required": True},
        {"source": "Q", "asset": None, "ticker": None, "position": "owns_unspecified", "review_required": False},
        {"source": "Q", "asset": "UNITED HEALTH AKTIE", "ticker": None, "position": "owns_unspecified",
         "review_required": False},
    ]
    text = cc.format_portfolio_disclosures(rows)
    assert "• P — owns ADBE, MSFT" in text
    assert "TSLA" not in text and "owns_unspecified" not in text and "Q —" not in text
    assert cc.format_portfolio_disclosures(rows[3:]) == ""


def test_negated_or_ownerless_disclosures_are_dropped():
    base = {"claim_type": "portfolio_disclosure", "portfolio_disclosure": "owns_unspecified", "channel_name": "C",
            "published_at": "2026-09-15T00:00:00+00:00", "schema_version": "2", "subject_mention": "SpaceX"}
    for evidence in ("I'm not invested in SpaceX because it's a so-called hype.",
                     "It's a name that I do not currently own in my own portfolio.",
                     "It's a company that I used to own called Booking Holdings.",
                     "we know Macy's and names like that that are controversial"):
        assert cc.portfolio_disclosures([dict(base, evidence_text=evidence)]) == []
    kept = cc.portfolio_disclosures([dict(base, evidence_text="I own Adobe stock.", subject_mention="Adobe")])
    assert kept[0]["ticker"] == "ADBE"
    assert cc.portfolio_disclosures([dict(base, portfolio_disclosure="no_position", evidence_text="I own it")]) == []


def test_one_vote_per_creator_from_their_latest_video():
    claims = [_claim("A", "ETH", "bearish", "unspecified", published="2026-07-08", video="v1"),
              _claim("A", "ETH", "bullish", "unspecified", published="2026-07-10", video="v2", n=1),
              _claim("A", "ETH", "bullish", "unspecified", published="2026-07-10", video="v2", n=2),
              _claim("A", "ETH", "bearish", "unspecified", published="2026-07-10", video="v2", n=3)]
    entry = cc.aggregate_views_by_asset(claims)["ETH"]
    assert entry["votes"] == {"A": "bullish"} and entry["mentions"] == 4 and entry["videos"] == 2


def test_lean_needs_two_thirds_so_one_dissent_does_not_make_a_video_mixed():
    assert cc.lean(8, 3) == "bullish" and cc.lean(2, 1) == "bullish"
    assert cc.lean(3, 2) == "mixed" and cc.lean(1, 1) == "mixed"
    assert cc.lean(0, 2) == "bearish" and cc.lean(0, 0) == "neutral"


def test_single_creator_assets_are_not_ranked_and_html_is_escaped():
    claims = [_claim("A", "NVDA", "bullish", "unspecified", video="v1"),
              _claim("B & Co", "NVDA", "bullish", "long", video="v2"),
              _claim("A", "PLTR", "bullish", "unspecified", video="v1", n=1)]
    inputs = mp._canonical_pulse_inputs(7, date(2026, 7, 12), lambda *a: {}, claims=claims)
    text = mp.build_canonical_pulse(inputs, date(2026, 7, 12))
    agree = text.split("Where creators agree")[1].split("\n\n")[0]
    assert "NVDA — 2 of 2 creators bullish · horizon: long-term" in agree and "PLTR" not in agree
    html = mp.build_canonical_pulse(inputs, date(2026, 7, 12), html=True)
    assert "<b>Where creators agree</b>" in html and "<b>NVDA</b>" in html
    assert mp._Fmt(True).b("B & Co <x>") == "<b>B &amp; Co &lt;x&gt;</b>"
    assert mp._Fmt(False).b("B & Co") == "B & Co"
