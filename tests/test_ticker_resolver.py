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
                        lambda q, k, searcher=None: [_listing("RBRK", "Rubrik Inc")])
    monkeypatch.setattr(warm_prices.sys, "argv",
                        ["warm_prices.py", "--resolve", "--signals", str(signals),
                         "--cache", str(cache_path),
                         "--ticker-map", str(map_path)])

    assert warm_prices.main() == 0
    saved = jsonlib.loads(map_path.read_text(encoding="utf-8"))
    assert saved["tickers"]["RUBY"]["ticker"] == "RBRK"
