"""Tests for learning a real ticker from a badly-named asset.

The safety property under test throughout: a model suggestion never reaches
the data on its own. It must be confirmed by the provider's catalogue, and
the confirmed listing must actually be the company we asked about.
"""
import json
from datetime import date

import market_pulse as mp
import ticker_resolver as tr


def _listing(symbol, name, country="United States", currency="USD"):
    return {"symbol": symbol, "instrument_name": name, "exchange": "NASDAQ",
            "country": country, "currency": currency}


def test_names_match_tolerates_transcript_spelling():
    assert tr.names_match("Rubric", "Rubrik Inc")        # mis-heard
    assert tr.names_match("Nebus", "Nebius Group N.V.")  # mis-spelled
    assert tr.names_match("Apple", "Apple Inc.")


def test_names_match_rejects_a_different_company():
    """The guard that matters: a real ticker for the wrong company."""
    assert not tr.names_match("Rubrik", "Ruby Tuesday Inc")
    assert not tr.names_match("Nebius", "Nike Inc")
    # Corporate filler alone must never constitute a match.
    assert not tr.names_match("Acme Holdings Inc", "Zeta Holdings Inc")


def test_search_hit_resolves_without_involving_the_model():
    def searcher(query, api_key):
        return [_listing("RBRK", "Rubrik Inc")] if query else []

    def completer(*a, **k):
        raise AssertionError("the model must not be consulted when search answers")

    entry = tr.resolve("Rubrik", None, "key", searcher=searcher, completer=completer)
    assert entry["ticker"] == "RBRK" and entry["via"] == "search"


def test_model_suggestion_is_accepted_only_after_verification():
    calls = {"n": 0}

    def searcher(query, api_key):
        calls["n"] += 1
        # The catalogue cannot match the mis-heard name, but knows the ticker.
        return [_listing("RBRK", "Rubrik Inc")] if query == "RBRK" else []

    def completer(system, user, json_mode=False):
        return json.dumps({"candidates": ["RBRK"], "company": "Rubrik"})

    entry = tr.resolve("Rubric", "RUBY", "key", searcher=searcher, completer=completer)
    assert entry["ticker"] == "RBRK"
    assert entry["via"] == "llm+verified"   # provenance recorded for audit
    assert entry["company"] == "Rubrik Inc"


def test_model_suggestion_for_the_wrong_company_is_rejected():
    """A hallucinated ticker that happens to exist must not be accepted."""
    def searcher(query, api_key):
        return [_listing("RUBY", "Ruby Tuesday Inc")] if query == "RUBY" else []

    def completer(system, user, json_mode=False):
        return json.dumps({"candidates": ["RUBY"], "company": "Rubrik"})

    entry = tr.resolve("Rubrik", None, "key", searcher=searcher, completer=completer)
    assert entry["ticker"] is None and entry["via"] == "unresolved"


def test_non_us_and_non_usd_listings_are_not_usable():
    def searcher(query, api_key):
        return [_listing("000660", "SK hynix Inc.", country="South Korea",
                         currency="KRW")]

    entry = tr.resolve("SK Hynix", None, "key", searcher=searcher,
                       completer=lambda *a, **k: "")
    assert entry["ticker"] is None  # the free plan prices US/USD only


def test_llm_failures_and_quota_degrade_to_unresolved():
    def searcher(query, api_key):
        return []

    for bad in ("", "QUOTA_EXHAUSTED", "not json", json.dumps({"candidates": "nope"})):
        entry = tr.resolve("Whatever Corp", None, "key", searcher=searcher,
                           completer=lambda *a, **k: bad)
        assert entry["ticker"] is None and entry["via"] == "unresolved"


def test_map_roundtrip(tmp_path):
    path = str(tmp_path / "ticker_map.json")
    entries = {"RUBY": {"ticker": "RBRK", "company": "Rubrik Inc",
                        "via": "llm+verified", "checked": "2026-08-17"}}
    assert tr.save(entries, path)
    assert tr.load(path)["RUBY"]["ticker"] == "RBRK"
    assert tr.load(str(tmp_path / "absent.json")) == {}


def test_curated_table_outranks_the_learned_map():
    """A learned entry must never quietly override a hand-checked decision."""
    mp.reset_learned_tickers({"RUBY": "WRONG", "SK HYNIX": "HXSCF"})
    try:
        assert mp.canonical_ticker({"name": "Rubric", "ticker": "RUBY"}) == "RBRK"
        assert mp.canonical_ticker({"name": "SK Hynix", "ticker": "SK HYNIX"}) == "HXSCF"
    finally:
        mp.reset_learned_tickers(None)


def test_resolve_cli_runs_end_to_end(tmp_path, monkeypatch, capsys):
    """A smoke test over main(): the first live --resolve run crashed on an
    unassigned `records`, which no unit test covered because they all called
    resolve() directly."""
    import json as jsonlib
    import warm_prices

    signals = tmp_path / "signals.jsonl"
    signals.write_text(jsonlib.dumps({
        "date": "2026-08-10", "channel_name": "A",
        "signals": {"assets": [{"name": "Rubric", "ticker": "RUBY",
                                "stance": "bullish", "type": "stock",
                                "conviction": "medium", "action": "none",
                                "price_target": None, "catalysts": [],
                                "horizon": "unspecified"}],
                    "market_sentiment": "bullish", "topics": []},
    }) + "\n", encoding="utf-8")
    map_path = tmp_path / "ticker_map.json"

    monkeypatch.setenv("TWELVEDATA_API", "tok")
    # A non-empty cache that lacks ruby.us: the ticker builds a symbol but
    # never priced, which is exactly what --resolve is for.
    cache_path = tmp_path / "prices.json"
    import price_cache
    seeded = {}
    price_cache.remember(seeded, "nvda.us", date(2026, 8, 10), date(2026, 8, 16),
                         {date(2026, 8, 14): 1.0})
    price_cache.save(seeded, str(cache_path), today=date(2026, 8, 17))
    monkeypatch.setattr(tr, "search_candidates",
                        lambda *a, **k: [_listing("RBRK", "Rubrik Inc")])
    monkeypatch.setattr(warm_prices.sys, "argv",
                        ["warm_prices.py", "--resolve", "--signals", str(signals),
                         "--cache", str(cache_path),
                         "--ticker-map", str(map_path)])

    assert warm_prices.main() == 0
    saved = jsonlib.loads(map_path.read_text(encoding="utf-8"))
    assert saved["tickers"]["RUBY"]["ticker"] == "RBRK"


def test_unreachable_catalogue_is_unchecked_not_unresolved():
    """A lookup that never happened must not be recorded as a verdict."""
    import channel_scorecard as cs

    def searcher(query, api_key):
        raise cs.SearchUnavailable("rate limited")

    entry = tr.resolve("Rubrik", None, "key", searcher=searcher,
                       completer=lambda *a, **k: "")
    assert entry["ticker"] is None and entry["via"] == "unchecked"


def test_backlog_is_scoped_to_scoreable_assets_and_ranked(tmp_path):
    """Only a directional call or a price target makes an asset worth a
    lookup; a neutral name-drop buys nothing."""
    import warm_prices

    def rec(name, ticker, stance="neutral", target=None, atype="stock"):
        return {"date": "2026-08-10", "channel_name": "A", "signals": {
            "assets": [{"name": name, "ticker": ticker, "stance": stance,
                        "action": "none", "price_target": target,
                        "catalysts": [], "type": atype,
                        "conviction": "medium", "horizon": "unspecified"}],
            "market_sentiment": "neutral", "topics": []}}

    records = [
        rec("Loud Co", "LOUD", stance="bullish"),   # directional, twice
        rec("Loud Co", "LOUD", stance="bullish"),
        rec("Quiet Co", "QUIET"),                   # neutral only -> skipped
        rec("Target Co", "TGTC", target=42.0),      # target -> kept
        rec("WTI Crude Oil", None, stance="bullish", atype="commodity"),  # skipped
    ]
    backlog = warm_prices.resolvable_assets(records, cache={}, learned={})
    names = [name for _, name, _ in backlog]
    assert names == ["Loud Co", "Target Co"]        # ranked by mentions


def _row(symbol, name, exchange="NASDAQ", itype="Common Stock"):
    return {"symbol": symbol, "instrument_name": name, "exchange": exchange,
            "country": "United States", "currency": "USD",
            "instrument_type": itype}


def test_sector_labels_are_never_companies():
    """'Aerospace' matched Honeywell Aerospace and 'Software' matched Unity
    Software in the first live run — coincidences, not resolutions."""
    for label in ("Aerospace", "Software", "Semiconductors", "Healthcare", "China"):
        entry = tr.resolve(label, None, "key",
                           searcher=lambda *a, **k: pytest.fail("must not look up a sector"),
                           completer=lambda *a, **k: pytest.fail("must not ask the model"))
        assert entry["ticker"] is None and entry["via"] == "not-a-company"


def test_funds_are_rejected_for_a_company_asset():
    """A private company must not resolve to an ETF that merely tracks it."""
    def searcher(query, api_key):
        return [_row("ANTW", "Harbor Anthropic AI Lab Ecosystem ETF")]

    entry = tr.resolve("Anthropic", None, "key", searcher=searcher,
                       completer=lambda *a, **k: "")
    assert entry["ticker"] is None


def test_a_fund_is_accepted_when_the_speaker_named_its_ticker():
    """Saying 'GDX' means the ETF; saying 'gold miners' does not."""
    def searcher(query, api_key):
        return [_row("GDX", "VanEck Gold Miners ETF", exchange="NYSE ARCA")]

    entry = tr.resolve("GDX", "GDX", "key", searcher=searcher,
                       completer=lambda *a, **k: "")
    assert entry["ticker"] == "GDX"


def test_primary_listing_beats_a_foreign_franchise_entity():
    """McDonald's resolved to the Japan holding company on OTC in the first run."""
    def searcher(query, api_key):
        return [_row("MDNDF", "McDonald's Holdings Company (Japan), Ltd.", exchange="OTC"),
                _row("MCD", "McDonald's Corporation", exchange="NYSE")]

    entry = tr.resolve("McDonald's", None, "key", searcher=searcher,
                       completer=lambda *a, **k: "")
    assert entry["ticker"] == "MCD"


def test_warrants_are_never_usable():
    assert not tr._usable_listing(_row("57MS28", "Korea Warrant 2026 on SK hynix",
                                       itype="Warrant"))


def test_franchise_entity_in_another_country_is_not_the_company():
    """Measured 2026-08-17: a search for "McDonald's" returns the Japan
    holding company on OTC and never MCD, so rejecting the geography is the
    only thing standing between a call on McDonald's and the wrong security."""
    assert not tr.names_match("McDonald's", "McDonald's Holdings Company (Japan), Ltd.")
    assert not tr.names_match("BASF", "BASF India Ltd.")
    assert tr.names_match("McDonald's", "McDonald's Corporation")
    # A geography the speaker did say is not a mismatch.
    assert tr.names_match("Pan American Silver", "Pan American Silver Corp.")


def test_assets_marked_unpriceable_are_never_resolved():
    """A curated 'private company' decision must outrank the resolver."""
    import warm_prices

    records = [{"date": "2026-08-10", "channel_name": "A", "signals": {
        "assets": [{"name": "OpenAI", "ticker": "OPENAI", "stance": "bullish",
                    "action": "none", "price_target": None, "catalysts": [],
                    "type": "stock", "conviction": "high",
                    "horizon": "unspecified"}],
        "market_sentiment": "bullish", "topics": []}}]
    assert warm_prices.resolvable_assets(records, cache={}, learned={}) == []


def test_probe_windows_parse_and_dedupe():
    """--probe-days exists to tell 'this symbol never prices' apart from
    'this symbol has no close in that particular week'."""
    import warm_prices

    assert warm_prices._probe_windows("3,10,60") == [3, 10, 60]
    assert warm_prices._probe_windows("10, 10 ,3") == [10, 3]      # deduped
    assert warm_prices._probe_windows("") == [warm_prices.PROBE_DAYS]
    assert warm_prices._probe_windows(None) == [warm_prices.PROBE_DAYS]
    assert warm_prices._probe_windows("0,-4,abc") == [warm_prices.PROBE_DAYS]
