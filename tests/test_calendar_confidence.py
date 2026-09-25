"""Hardening item 4: weekday-only exchange calendars are labelled, scored,
and kept out of source rankings by default."""
from datetime import date, timedelta

import instruments
import research_analytics as ra
import scorecard_pricing as sp

from tests.test_scorecard_pricing import _claim, _fetch

XETRA, HKEX, US = sp.EXCHANGES["xetra"], sp.EXCHANGES["hkex"], sp.EXCHANGES["us"]


def _closes(start, end, closed=()):
    out, d, i = {}, start, 0
    while d <= end:
        if d.weekday() < 5 and d not in closed:
            out[d] = 100.0 + i
            i += 1
        d += timedelta(days=1)
    return out


def test_consecutive_local_holidays_are_invisible_to_a_weekday_calendar():
    # Xetra is closed on 24 and 25 December 2026 (Thu, Fri); the weekday
    # calendar does not know that, so the "next trading day" after a
    # Wednesday-evening publication is Christmas Eve. No close exists, and
    # the entry resolves to Monday 28th only through the price lag search.
    closed = {date(2026, 12, 24), date(2026, 12, 25)}
    closes = _closes(date(2026, 12, 14), date(2027, 1, 15), closed)
    assert sp.is_trading_day(XETRA, date(2026, 12, 24)) is True       # the calendar's (wrong) belief
    e = sp.entry_point(XETRA, "2026-12-23T18:00:00+01:00", closes)     # after the 17:30 close
    assert e["session_relation"] == "after_close"
    assert e["resolved_trading_date"] == "2026-12-28" and e["requested_date"] == "2026-12-23"
    assert XETRA.calendar_confidence == "weekdays_only"
    # Hong Kong, Lunar New Year 2026: three consecutive closures (17-19 Feb).
    closed = {date(2026, 2, 17), date(2026, 2, 18), date(2026, 2, 19)}
    closes = _closes(date(2026, 2, 2), date(2026, 3, 6), closed)
    e = sp.entry_point(HKEX, "2026-02-16T10:00:00+00:00", closes)      # 18:00 HK, after the close
    assert e["resolved_trading_date"] == "2026-02-20"
    # A closure longer than the lag window is simply no entry — never a
    # guessed one.
    closed = {date(2026, 2, 17) + timedelta(days=i) for i in range(8)}
    e = sp.entry_point(HKEX, "2026-02-16T10:00:00+00:00", _closes(date(2026, 2, 2), date(2026, 3, 6), closed))
    assert e["price"] is None and e["reason"] == "no_close_within_lag"


def test_weekday_calendar_claims_are_scored_labelled_and_not_ranked(monkeypatch):
    de = instruments.Instrument("XETR:SAP", "SAP", "sap.xetra", "xetra", "XETR", "stock", "DE", "EUR", None, "auto")
    us = instruments.resolve_instrument("NVDA")
    monkeypatch.setattr(instruments, "resolve_instrument",
                        lambda ticker=None, name=None, asset_type=None: de if ticker == "SAP" else us)
    closed = {date(2026, 12, 24), date(2026, 12, 25)}
    series = _closes(date(2026, 12, 14), date(2027, 1, 20), closed)
    claims = [_claim("2026-12-23T18:00:00+01:00", end="2027-01-10", ticker="SAP", subject_mention="SAP"),
              _claim("2026-12-23T18:00:00+01:00", end="2027-01-10", ticker="NVDA", claim_id="c2")]
    fetch = _fetch({"sap.xetra": series, "nvda.us": series, "xlk.us": series})
    sc = ra.scorecard(claims, date(2027, 2, 1), fetch, min_sample=1)
    a = sc["A"]
    assert a["n"] == 2 and a["ranked_n"] == 1                       # both scored, one rankable
    assert a["calendar_confidence"] == {"weekdays_only": 1, "full": 1}
    sap = next(s for s in a["scored"] if s["symbol"] == "sap.xetra")
    assert sap["calendar_confidence"] == "weekdays_only" and sap["rankable"] is False
    assert sap["calendar_note"] == sp.WEEKDAY_CALENDAR_NOTE
    assert sap["entry_resolved_trading_date"] == "2026-12-28"
    nvda = next(s for s in a["scored"] if s["symbol"] == "nvda.us")
    assert nvda["calendar_confidence"] == "full" and nvda["rankable"] is True and nvda["calendar_note"] is None
    # Rankings (when enabled) count only the full-confidence forecasts.
    monkeypatch.setattr(sp, "SCORECARD_RANKINGS_ENABLED", True)
    assert ra.rankable_sources(dict(sc), min_sample=2) == []          # one rankable forecast is below 2
    assert ra.rankable_sources(dict(sc), min_sample=1) == ["A"]
    text = ra.format_scorecard_lines({k: v for k, v in sc.items() if not k.startswith("_")}, {}, 1, True, ["A"])
    assert "1 on weekday-only calendars (not ranked)" in text
    # The opt-in flag includes them.
    monkeypatch.setattr(sp, "RANK_WEEKDAY_CALENDARS", True)
    sc = ra.scorecard(claims, date(2027, 2, 1), fetch, min_sample=1)
    assert sc["A"]["ranked_n"] == 2 and ra.rankable_sources(dict(sc), min_sample=2) == ["A"]


def test_the_rules_do_not_claim_a_one_day_bound():
    rules = ra.SCORECARD_RULES["trading_calendar"].lower()
    assert "not modelled" in rules and "no bound" in rules
    assert "at most" not in rules and "one day" not in rules
