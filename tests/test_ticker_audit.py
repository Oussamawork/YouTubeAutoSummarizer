"""The ticker audit (ticker_audit.py): a deterministic, offline, read-only
check that every alias, claim ticker and cached price series is the
instrument it claims to be. Synthetic claims, legacy rows and caches only;
nothing here reads the committed data."""
import json
import os
import subprocess
import sys
from datetime import date, timedelta

import research_state
import signals_data
import ticker_audit as ta
from tests.test_canonical_analytics import _claim

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TODAY = date(2026, 10, 16)


def _entry(closes, start=None, end=None):
    """A price-cache entry in price_cache.load's shape."""
    days = sorted(closes)
    return {"from": start or days[0], "to": end or days[-1], "closes": dict(closes), "adjustment": "split_adjusted"}


def _bars(days, last_range, end=date(2026, 10, 11)):
    """A crypto entry with daily bars ending on its requested 'to' day: a 3%
    high-low range every day, `last_range` on the last."""
    entry = _entry({end - timedelta(days=i): 100.0 + i / 100 for i in range(days)})
    entry["highs"] = {d: c * 1.015 for d, c in entry["closes"].items()}
    entry["lows"] = {d: c * 0.985 for d, c in entry["closes"].items()}
    entry["highs"][end], entry["lows"][end] = 100 + 50 * last_range, 100 - 50 * last_range
    return entry


def _legacy(*assets, day="2026-08-03", channel="C", video="old"):
    return {"date": day, "channel_name": channel, "video_id": video,
            "signals": {"assets": [dict(a) for a in assets]}}


def _unticked(source, name, stance="bullish", asset_type="stock", n=0, **over):
    """A canonical claim whose subject resolves to no ticker."""
    base = _claim(source, name, stance, "unspecified", n=n, ticker=None, asset_type=asset_type,
                  video=f"v-{source}-{name}-{n}", **over)
    base["claim_id"] = f"{source}-{name}-{n}"
    return base


def test_alias_registry_gaps_flags_an_unregistered_alias_output(monkeypatch):
    monkeypatch.setattr(signals_data, "ASSET_ALIASES", {"ZORBLAX": "ZRBX", "NVIDIA": "NVDA"})
    # A registered output (GOOGL) and an unpriceable decision (OPENAI) are not gaps.
    monkeypatch.setattr(signals_data, "TICKER_ALIASES", {"GOOG": "GOOGL", "OPEN AI": "OPENAI"})
    gaps = ta.alias_registry_gaps()
    assert [g["ticker"] for g in gaps] == ["ZRBX"]
    assert gaps[0]["aliases"] == ["ASSET_ALIASES['ZORBLAX']"] and gaps[0]["severity"] == "error"


def test_symbol_trace_finds_producers_and_orphans():
    claims = [_claim("A", "NVDA", "bullish", "short"),
              _claim("B", "ZZZQ", "bearish", "short", video="vb", canonical_entity_name="Zorblax Quantum",
                     subject_mention="Zorblax Quantum")]
    records = [_legacy({"name": "Nvidia", "ticker": "NVDA", "type": "stock", "stance": "bullish"},
                       {"name": "AMD", "ticker": "AMD", "type": "stock", "stance": "bearish"})]
    cache = {"nvda.us": _entry({date(2026, 7, 10): 180.0, date(2026, 7, 13): 182.5}),
             "orph.us": _entry({date(2026, 7, 10): 5.0}),
             "xlk.us": _entry({date(2026, 7, 10): 250.0})}
    trace = ta.symbol_trace(claims, records, cache, reference={"xlk.us": {"benchmark": 1}})

    nvda = trace["nvda.us"]
    assert (nvda["canonical"], nvda["legacy"], nvda["directional"]) == (1, 1, 1)
    assert nvda["producers"] == {"NVDA": 1, "NVIDIA": 1}
    assert nvda["cached"] and nvda["last_date"] == date(2026, 7, 13) and nvda["last_close"] == 182.5
    assert {e["kind"] for e in nvda["examples"]} == {"canonical", "legacy"}
    # Produced but never cached; produced by a legacy row only.
    assert trace["zzzq.us"]["canonical"] == 1 and not trace["zzzq.us"]["cached"]
    assert trace["zzzq.us"]["producers"] == {"ZORBLAX QUANTUM": 1}
    assert trace["amd.us"]["legacy"] == 1 and trace["amd.us"]["canonical"] == 0
    # A cached series nothing produces is an orphan; a benchmark is a producer.
    assert [o["symbol"] for o in ta.orphan_symbols(trace)] == ["orph.us"]


def test_known_wrong_symbol_still_produced_is_an_error_and_cached_only_is_info(monkeypatch):
    trace = {
        "mogu.us": {"canonical": 1, "legacy": 1, "producers": {"MOGU": 2}, "cached": True, "examples": [],
                    "from": date(2026, 9, 9), "to": date(2026, 10, 11), "last_close": 2.24},
        "sy.us": {"canonical": 0, "legacy": 0, "producers": {}, "cached": True, "examples": [],
                  "from": date(2026, 8, 20), "to": date(2026, 10, 1), "last_close": 2.67},
    }
    findings = {f["symbol"]: f for f in ta.known_wrong_findings(trace)}
    assert set(findings) == set(ta.KNOWN_WRONG_SYMBOLS)
    assert (findings["mogu.us"]["severity"], findings["mogu.us"]["status"]) == ("error", "produced")
    assert findings["mogu.us"]["producers"] == {"MOGU": 2} and "Moog" in findings["mogu.us"]["why"]
    assert (findings["sy.us"]["severity"], findings["sy.us"]["status"]) == ("info", "cached_only")
    assert (findings["crwe.us"]["severity"], findings["crwe.us"]["status"]) == ("ok", "clear")

    # End to end: a legacy row that still produces a known-wrong symbol is an
    # error in the report; the same symbol only cached is a prune candidate.
    # (The legacy symbol rule is channel_scorecard's; pinned here.)
    monkeypatch.setattr(ta, "legacy_symbol", lambda asset, channel=None: f"{asset['ticker'].lower()}.us")
    records = [_legacy({"name": "mogu", "ticker": "MOGU", "type": "stock", "stance": "bullish"})]
    cache = {"mogu.us": _entry({date(2026, 9, 9): 1.85}), "sy.us": _entry({date(2026, 8, 27): 2.32})}
    report = ta.run_audit([], records, cache, {}, TODAY, reference={})
    by_symbol = {k["symbol"]: k for k in report["known_wrong"]}
    assert by_symbol["mogu.us"]["severity"] == "error" and by_symbol["mogu.us"]["legacy"] == 1
    assert by_symbol["sy.us"]["severity"] == "info"
    assert "sy.us" in [o["symbol"] for o in report["orphans"]]
    assert report["counts"]["known_wrong_produced"] == 1 and report["errors"] >= 1


def test_price_level_check_flags_a_tenfold_mismatch_and_skips_multiples():
    published = "2026-09-01"

    def zzzq(n, **over):
        return _claim("A", "ZZZQ", "bullish", "short", published=published, n=n,
                      canonical_entity_name="Zorblax Quantum", subject_mention="Zorblax Quantum", **over)

    wrong = zzzq(1, evidence_text="The current market price is $2.60, and I think it can double from here.")
    multiple = zzzq(2, evidence_text="The stock is trading at 14 times earnings, which is cheap.")
    pe_context = zzzq(3, evidence_text="On a forward price to earnings basis, the stock is trading at 16.4.")
    history = zzzq(4, evidence_text="When it was trading at 175 I sold every share.")
    baseline = zzzq(5, forecast_metric="price", baseline_value=25.0, currency="USD")
    coin = _claim("B", "BTC", "bullish", "short", published=published, n=6, asset_type="crypto",
                  evidence_text="Bitcoin is now at 83k and I expect a push higher.")
    cache = {"zzzq.us": _entry({date(2026, 9, 1): 24.0, date(2026, 9, 2): 24.5}),
             "btcusd": _entry({date(2026, 9, 1): 83_500.0})}
    res = ta.price_level_checks([wrong, multiple, pe_context, history, baseline, coin], cache)

    # Three statements are prices: the $2.60, the baseline and the 83k.
    assert res["checked"] == 3 and res["in_band"] == 2
    [out] = res["outliers"]
    assert out["claim_id"] == wrong["claim_id"] and out["symbol"] == "zzzq.us"
    assert (out["stated"], out["close"], out["ratio"]) == (2.6, 24.0, 0.108)
    assert out["close_date"] == date(2026, 9, 1) and out["severity"] == "warning"
    assert res["corroborated"] == ["btcusd"] and res["contradicted"] == []
    # A wider band accepts what the default flags.
    assert ta.price_level_checks([wrong], cache, band=(0.1, 10))["outliers"] == []


def test_registry_gaps_are_ranked_by_view_claims():
    alpha = {"canonical_entity_name": "Alpha Widgets", "subject_mention": "Alpha Widgets"}
    beta = {"canonical_entity_name": "Beta Widgets", "subject_mention": "Beta Widgets"}
    claims = [
        # three claims, no view (a fact carries no stance analytics may count)
        *[_claim("A", "ZZZA", "not_applicable", "unspecified", n=i, claim_type="fact", **alpha) for i in range(3)],
        # two directional views
        _claim("B", "ZZZB", "bullish", "short", video="vb1", **beta),
        _claim("C", "ZZZB", "bearish", "short", video="vc1", **beta),
        _claim("A", "NVDA", "bullish", "short", n=9, canonical_entity_name="Nvidia"),    # registered: not a gap
        _claim("A", "OPENAI", "bullish", "short", n=8, canonical_entity_name="OpenAI"),  # unpriceable: not a gap
    ]
    gaps = ta.registry_gaps(claims, cache={"zzzb.us": _entry({date(2026, 7, 10): 3.0})})
    assert [g["ticker"] for g in gaps] == ["ZZZB", "ZZZA"]
    zzzb, zzza = gaps
    assert (zzzb["views"], zzzb["directional"], zzzb["claims"]) == (2, 2, 2)
    assert zzzb["channels"] == {"B": 1, "C": 1} and zzzb["symbol"] == "zzzb.us" and zzzb["cached"] is True
    assert (zzza["views"], zzza["claims"], zzza["cached"]) == (0, 3, False)
    assert zzza["severity"] == "info" and zzza["us_ticker_shape"]


def test_unresolved_backlog_ranks_by_directional_views_and_gives_legacy_hints(monkeypatch):
    # The legacy symbol rule is channel_scorecard's; pinned here.
    monkeypatch.setattr(ta, "legacy_symbol", lambda asset, channel=None: f"{asset['ticker'].lower()}.us")
    claims = [
        _unticked("A", "Fortex Labs", "bullish", n=1, ticker_spoken="FTXL", evidence_text="Fortex Labs is a buy."),
        _unticked("B", "Fortex Labs", "bearish", n=2, evidence_text="Fortex Labs will fall."),
        # more claims, but no directional view: ranked after Fortex Labs
        _unticked("A", "Quorbit", "neutral", n=3, evidence_text="Quorbit is fairly valued."),
        _unticked("A", "Quorbit", "not_applicable", n=4, claim_type="fact", evidence_text="Quorbit sells widgets."),
        _unticked("A", "Quorbit stock", "not_applicable", n=5, claim_type="fact", evidence_text="Quorbit grew."),
        # not a ticker problem: a theme, a market subject, a placeholder, a private company
        _unticked("A", "Widget Cycle", "bullish", n=6, asset_type="sector"),
        _unticked("A", "S&P 500", "bearish", n=7, asset_type="index"),
        _unticked("A", "the company", "bullish", n=8),
        _unticked("A", "Anthropic", "bullish", n=9, asset_type="private_company"),
    ]
    records = [_legacy({"name": "Fortex Labs", "ticker": "FTXL", "type": "stock", "stance": "bullish"},
                       {"name": "Fortex Labs", "ticker": "FTXL", "type": "stock", "stance": "bullish"},
                       {"name": "Fortex Labs", "ticker": "FTX", "type": "stock", "stance": "bullish"})]
    cache = {"ftxl.us": _entry({date(2026, 8, 3): 12.0})}
    backlog = ta.unresolved_backlog(claims, records, cache)

    assert [e["name"] for e in backlog["entries"]] == ["FORTEX LABS", "QUORBIT"]
    fortex, quorbit = backlog["entries"]
    assert (fortex["directional"], fortex["views"], fortex["claims"]) == (2, 2, 2)
    assert fortex["channels"] == {"A": 1, "B": 1} and fortex["spoken_hint"] == {"ticker": "FTXL", "claims": 1}
    hint = fortex["legacy_hint"]
    assert (hint["ticker"], hint["rows"], hint["symbol"], hint["cached"]) == ("FTXL", 2, "ftxl.us", True)
    assert hint["others"] == {"FTX": 1}
    assert (quorbit["claims"], quorbit["views"], quorbit["directional"]) == (3, 1, 0)   # "Quorbit stock" folded in
    assert quorbit["legacy_hint"] is None and len(quorbit["examples"]) == 2
    assert quorbit["examples"][0]["text"] == "Quorbit is fairly valued."            # the view comes first
    assert (backlog["names"], backlog["claims"], backlog["directional"]) == (2, 5, 2)
    assert backlog["themes"]["names"] == 1 and backlog["themes"]["top"] == {"WIDGET CYCLE": 1}
    assert backlog["market_subjects"] == {"S&P 500": 1}
    assert backlog["decided_unpriceable"] == {"ANTHROPIC": 1}
    assert backlog["placeholders"]["claims"] == 1
    # The hint is shown, never applied: the claims still have no ticker.
    assert all(c["ticker"] is None for c in claims)
    assert ta.unresolved_backlog(claims, records, cache, top=1)["entries"][0]["name"] == "FORTEX LABS"


def test_learned_map_mismatch_and_unknown_exchange_are_errors():
    entries = {
        "ROCKET LAB": {"ticker": "RKLB", "exchange": "NASDAQ", "company": "Rocket Lab Corporation", "via": "search"},
        "RUBRIK": {"ticker": "RUBY", "exchange": "NYSE", "company": "Ruby Tuesday, Inc.", "via": "llm"},
        "SONY": {"ticker": "SONY", "exchange": "TSE", "company": "Sony Group Corporation", "via": "search"},
        "OPENAI": {"ticker": None, "via": "none"},              # a "no US listing" verdict
    }
    issues = {i["key"]: i for i in ta.learned_map_issues(entries)}
    assert set(issues) == {"RUBRIK", "SONY"}
    assert issues["RUBRIK"]["problems"] == ["company_mismatch"] and issues["RUBRIK"]["severity"] == "error"
    assert issues["SONY"]["problems"] == ["unknown_exchange"]
    assert ta.learned_map_issues({}) == [] and ta.learned_map_issues(None) == []


def test_series_anomalies():
    weekdays = [date(2026, 9, 8) + timedelta(days=i) for i in range(8)]
    weekdays = [d for d in weekdays if d.weekday() < 5]                 # Tue 09-08 .. Tue 09-15
    cache = {
        "jump.us": _entry({date(2026, 9, 1): 10.0, date(2026, 9, 2): 15.0, date(2026, 9, 3): 15.2}),
        "flat.us": _entry({d: 7.5 for d in weekdays}),
        # a Saturday and Labor Day (2026-09-07) are not NYSE sessions
        "wknd.us": _entry({date(2026, 9, 4): 20.0, date(2026, 9, 5): 20.1, date(2026, 9, 7): 20.2,
                           date(2026, 9, 8): 20.3}),
        # crypto trades every day; the bar dated on the requested 'to' day was not final yet
        "coinusd": _entry({date(2026, 10, 9): 1.0, date(2026, 10, 10): 1.02, date(2026, 10, 11): 1.01},
                          end=date(2026, 10, 11)),
        # with daily bars the range decides: a compressed last bar is partial, a full one complete
        "thinusd": _bars(10, last_range=0.005),
        "fullusd": _bars(10, last_range=0.03),
        "fine.us": _entry({date(2026, 10, 12): 50.0, date(2026, 10, 13): 51.0, date(2026, 10, 14): 50.5}),
        "old.us": _entry({date(2026, 9, 1): 30.0, date(2026, 9, 2): 31.0}),
    }
    trace = {
        "old.us": {"cached": True, "canonical": 2, "directional": 1, "legacy": 0, "reference": {},
                   "last_date": date(2026, 9, 2), "producers": {"OLD": 2}},
        "fine.us": {"cached": True, "canonical": 3, "directional": 3, "legacy": 0, "reference": {},
                    "last_date": date(2026, 10, 14), "producers": {"FINE": 3}},
        "miss.us": {"cached": False, "canonical": 1, "directional": 1, "legacy": 0, "reference": {},
                    "last_date": None, "producers": {"MISS": 1}},
    }
    found = {}
    for a in ta.series_anomalies(cache, TODAY, trace):
        found.setdefault(a["symbol"], []).append(a)
    assert [a["kind"] for a in found["jump.us"]] == ["big_move"] and found["jump.us"][0]["change"] == 0.5
    assert [(a["kind"], a["days"]) for a in found["flat.us"]] == [("flat_run", 6)]
    assert [(a["kind"], a["dates"]) for a in found["wknd.us"]] == [
        ("non_trading_day", [date(2026, 9, 5), date(2026, 9, 7)])]
    assert [(a["kind"], a["date"], a["severity"]) for a in found["coinusd"]] == [
        ("partial_candle", date(2026, 10, 11), "warning")]                     # no bars to judge by
    assert [(a["kind"], a["range_ratio"], a["severity"]) for a in found["thinusd"]] == [
        ("partial_candle", 0.167, "warning")]
    assert [(a["kind"], a["range_ratio"], a["severity"]) for a in found["fullusd"]] == [
        ("partial_candle", 1.0, "info")]
    assert [(a["kind"], a["days"], a["severity"]) for a in found["old.us"]] == [("stale", 44, "warning")]
    assert [(a["kind"], a["severity"]) for a in found["miss.us"]] == [("no_series", "info")]
    assert "fine.us" not in found
    # Without a trace there is nothing to call stale or missing.
    assert {a["kind"] for a in ta.series_anomalies(cache, TODAY)} == {
        "big_move", "flat_run", "non_trading_day", "partial_candle"}


def test_needs_verification_entries_have_evidence_and_a_verify_step():
    keys = [e["key"] for e in ta.NEEDS_VERIFICATION]
    assert len(keys) == len(set(keys))
    for entry in ta.NEEDS_VERIFICATION:
        assert set(entry) == {"key", "proposed", "confidence", "claims", "evidence", "verify"}, entry["key"]
        assert entry["confidence"] in ("medium", "low"), entry["key"]
        assert len(entry["evidence"]) > 40 and entry["verify"].strip(), entry["key"]
        assert isinstance(entry["claims"], int) and entry["key"] in ta._VERIFY_MATCH
    for needed in ("IRON", "MOGU", "SK HYNIX", "XP (Phantom", "CRONOS", "TARGET"):
        assert any(k.startswith(needed) for k in keys), needed
    assert any("warm_prices.py --find" in e["verify"] for e in ta.NEEDS_VERIFICATION)

    # The report adds today's state of each entry, matched by name and channel.
    iron = _unticked("FinVid", "Iron", "bullish", n=1, evidence_text="I continue to be bullish on Iron.")
    iron_elsewhere = _unticked("HKCM", "Iron", "bullish", n=2)       # IRON is FinVid-scoped
    status = {s["key"]: s for s in ta.needs_verification_status([iron, iron_elsewhere])}
    assert status["IRON (FinVid)"]["current"]["claims"] == 1
    assert status["IRON (FinVid)"]["current"]["examples"][0]["text"] == "I continue to be bullish on Iron."
    assert status["IRON (FinVid)"]["evidence"] == next(
        e for e in ta.NEEDS_VERIFICATION if e["key"] == "IRON (FinVid)")["evidence"]


def _store(claims):
    state = research_state.load_state()
    for c in claims:
        research_state.update(state, c["video_id"], active_run_key="rk1")
        research_state.store_claims([c], state, c["video_id"], "rk1")
    research_state.save_state(state)


def test_cli_json_and_strict_exit_codes(tmp_path, monkeypatch, capsys):
    # Small, consistent alias tables, so only the gap planted below is an error.
    monkeypatch.setattr(signals_data, "ASSET_ALIASES", {"NVIDIA": "NVDA"})
    monkeypatch.setattr(signals_data, "TICKER_ALIASES", {"GOOG": "GOOGL"})
    # Stored claims are re-resolved on load, so the subject names the company.
    _store([_claim("A", "NVDA", "bullish", "short", published="2026-10-01", canonical_entity_name="Nvidia",
                   subject_mention="Nvidia", evidence_text="Nvidia is trading at $180 and I am buying."),
            _unticked("B", "Fortex Labs", "bullish", n=1, published="2026-10-02")])
    cache_path = tmp_path / "prices.json"
    cache_path.write_text(json.dumps({"version": 1, "symbols": {"nvda.us": {
        "from": "2026-09-30", "to": "2026-10-09", "adjustment": "split_adjusted",
        "closes": {"2026-10-01": 181.0, "2026-10-02": 183.0, "2026-10-09": 190.0}}}}))
    out = tmp_path / "audit.json"
    args = ["--cache", str(cache_path), "--signals", str(tmp_path / "missing.jsonl"),
            "--ticker-map", str(tmp_path / "missing_map.json"), "--today", "2026-10-16"]

    assert ta.main(args + ["--json", str(out), "--strict"]) == 0
    report = json.loads(out.read_text())
    assert list(report) == list(ta.REPORT_KEYS)
    assert report["today"] == "2026-10-16" and report["errors"] == 0
    assert report["counts"]["claims"] == 2 and report["price_level"]["in_band"] == 1
    assert report["unresolved_backlog"]["entries"][0]["name"] == "FORTEX LABS"
    text = capsys.readouterr().out
    headers = ["ALIAS OUTPUTS WITHOUT AN INSTRUMENT (0)", "KNOWN-WRONG SYMBOLS (0)", "ORPHANED CACHE SERIES (0)",
               "STATED PRICES VS THE CACHED CLOSE (0)", "RESOLVED TICKERS WITHOUT AN INSTRUMENT (0)",
               "UNRESOLVED NAMES (1)", "LEARNED TICKER MAP (0)", "PRICE-SERIES ANOMALIES",
               f"NEEDS VERIFICATION ({len(ta.NEEDS_VERIFICATION)})"]
    positions = [text.index(h) for h in headers]
    assert positions == sorted(positions)

    # --json - prints only valid JSON on stdout, with exactly the documented keys.
    assert ta.main(args + ["--json", "-"]) == 0
    assert set(json.loads(capsys.readouterr().out)) == set(ta.REPORT_KEYS)

    # --strict exits 1 exactly when the report has an error.
    monkeypatch.setattr(signals_data, "ASSET_ALIASES", {"NVIDIA": "NVDA", "ZORBLAX": "ZRBX"})
    assert ta.main(args + ["--json", str(out), "--strict"]) == 1
    report = json.loads(out.read_text())
    assert report["errors"] == 1 and report["alias_registry_gaps"][0]["ticker"] == "ZRBX"
    assert ta.main(args) == 0
    capsys.readouterr()


def test_bad_data_is_skipped_never_raised():
    claims = [None, "not a claim", {"claim_id": "x", "canonical_entity_name": 42, "published_at": "garbage",
                                    "evidence_text": "trading at $5", "ticker": ["NVDA"]}]
    records = [None, {"signals": {"assets": [None, {"name": None, "ticker": 7}]}},
               {"signals": "not a dict"}]
    cache = {"bad.us": {"from": None, "to": None, "closes": {date(2026, 9, 1): 0.0, date(2026, 9, 2): 5.0}},
             "junk.us": "not an entry", 7: {"closes": {}}}
    report = ta.run_audit(claims, records, cache, {"KEY": "not a dict"}, TODAY, reference={})
    assert list(report) == list(ta.REPORT_KEYS)
    assert report["counts"]["failed_sections"] == [] and report["counts"]["cached_symbols"] == 1
    assert [i["problems"] for i in report["learned_map"]] == [["malformed_entry"]]
    assert ta.format_text(report)
    json.dumps(ta._jsonable(report))


def test_importing_ticker_audit_does_not_import_requests():
    code = "import sys, ticker_audit; sys.exit(1 if 'requests' in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", code], cwd=REPO).returncode == 0
