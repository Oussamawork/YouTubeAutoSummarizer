"""Claim-level entity resolution: the curated name decides, the model's ticker
only confirms or contradicts it, markets / baskets / metrics are told apart
from single instruments, and stored claims are re-resolved at load time."""
import inspect

import canonical_claims as cc
import claims as cm
import research_state
import transcript_normalize as tn
from tests.test_canonical_analytics import _claim as _row


def _validate(text, raw, title="T", channel="Chan"):
    ctx = {"video_id": "v1", "channel_id": "c1", "channel_name": channel, "video_title": title,
           "published_at": "2026-10-01T00:00:00+00:00", "transcript_source": "supadata", "run_key": "rk",
           "extraction_model": "m"}
    return cm.validate_claims(raw, tn.normalize_transcript(text, "v1"), ctx)[0][0]


def _raw(**over):
    base = {"attribution_type": "speaker_personal_view", "claim_type": "opinion", "stance": "bullish",
            "extraction_confidence": "high", "is_forward_looking": False}
    base.update(over)
    return base


def _named(name, **over):
    """A canonical row about an un-tickered name (no resolution_version: as stored before RESOLUTION_VERSION 2)."""
    return _row("A", None, "bullish", "unspecified", canonical_entity_name=name, subject_mention=name, **over)


# --- the model's ticker field -------------------------------------------------------------------


def test_name_in_ticker_field_is_folded_and_never_a_review_reason():
    # clm_9be98175ab36d030f8cd: "BITCOIN" (7 letters) never matched the old 1-6 letter word regex.
    ev = "Bitcoin remains below the 50 week simple moving average for now"
    c = _validate(ev + ".", [_raw(subject_mention="Bitcoin", ticker_spoken="BITCOIN", asset_type="crypto",
                                  evidence_text=ev)])
    assert c["ticker"] == "BTC" and c["ticker_source"] == "curated_mapping"
    assert c["review_reasons"] == [] and c["review_required"] is False
    # clm_f072527db0c2ad9e8179: "SOLANA" used to be stored AS the ticker.
    ev = "Solana is consolidating between support and resistance at the moment"
    c = _validate(ev + ".", [_raw(subject_mention="Solana", ticker_spoken="SOLANA", asset_type="crypto",
                                  evidence_text=ev)], title="Solana Is Getting Quiet, and That Is Usually When It Moves")
    assert c["ticker"] == "SOL"
    assert cm.fold_spoken_ticker("XPENG") == ("XPEV", True)
    assert cm.fold_spoken_ticker("GOOG") == ("GOOGL", False)
    assert cm.fold_spoken_ticker("NASDAQ") == (None, True)
    assert cm.fold_spoken_ticker("000660.KS") == (None, True)


def test_model_ticker_agreeing_with_the_curated_name_needs_no_review():
    ev = "I like Chevron a lot here"
    c = _validate(ev + ".", [_raw(subject_mention="Chevron", ticker_spoken="CVX", evidence_text=ev)])
    assert c["ticker"] == "CVX" and c["review_reasons"] == [] and c["resolution_notes"] == []


def test_model_ticker_contradicting_the_curated_name_is_reviewed():
    ev = "I like Chevron a lot here"
    c = _validate(ev + ".", [_raw(subject_mention="Chevron", ticker_spoken="XOM", evidence_text=ev)])
    assert c["ticker"] == "CVX" and "ticker_subject_mismatch" in c["review_reasons"] and c["review_required"]
    # A ticker that is no real listing is replaced without a review.
    c = _validate("Canopy Growth could double.", [_raw(subject_mention="Canopy Growth", ticker_spoken="WEED",
                                                       evidence_text="Canopy Growth could double")])
    assert c["ticker"] == "CGC" and c["review_reasons"] == [] and c["resolution_notes"] == ["ticker_spoken_replaced"]


def test_index_name_is_never_a_security_ticker():
    # "NASDAQ kämpft mit dem Allzeithoch!" -> ticker "NASDAQ" from the title.
    ev = "Der Nasdaq kämpft mit dem Allzeithoch und sollte weiter steigen"
    c = _validate(ev + ".", [_raw(subject_mention="Nasdaq", ticker_spoken="NASDAQ", asset_type="index",
                                  evidence_text=ev)], title="NASDAQ kämpft mit dem Allzeithoch!")
    assert c["ticker"] is None and c["subject_kind"] == "market" and c["canonical_entity_name"] == "NASDAQ"
    assert c["entity_resolution_status"] == "confirmed"


def test_ticker_copied_from_an_unknown_name_is_not_a_ticker():
    # clm_78e1ebe8fa482f5c86f6: "ticker symbol Mogu" is Moog (MOG.A), not MOGU Inc.
    ev = "A company like move, ticker symbol Mogu. They really diversified between space, space propulsion"
    c = _validate(ev + ".", [_raw(subject_mention="mogu", ticker_spoken="MOGU", asset_type="stock",
                                  evidence_text=ev)])
    assert c["ticker"] is None and "ticker_equals_name_unverified" in c["resolution_notes"]
    assert c["review_reasons"] == []
    # clm_c1d07c71ddacc98c1a97: "New" (Nu Holdings) became ticker NEW.
    ev = "New is also a tremendously well-run company, a very, very good business"
    c = _validate(ev + ".", [_raw(subject_mention="New", ticker_spoken="NEW", asset_type="stock",
                                  evidence_text=ev)], channel="Some Channel")
    assert c["ticker"] is None
    c = _validate(ev + ".", [_raw(subject_mention="New", ticker_spoken="NEW", asset_type="stock",
                                  evidence_text=ev)], channel="Couch Investor")
    assert c["ticker"] == "NU"


def test_channel_spelling_resolves_only_for_that_channel():
    ev = "XP ist eben nicht nur ein Elektromobilitätshersteller"
    raw = [_raw(subject_mention="XP", ticker_spoken="XP", asset_type="stock", evidence_text=ev)]
    assert _validate(ev + ".", raw, channel="HKCM")["ticker"] == "XPEV"
    assert _validate(ev + ".", raw, channel="Phantom by HKCM")["ticker"] is None   # medium confidence only
    assert _validate(ev + ".", raw, channel="Other")["ticker"] is None
    ev = "I'm bullish on Iron if the world is going to be compute constrained through 2028"
    c = _validate(ev + ".", [_raw(subject_mention="Iron", ticker_spoken="IRON", asset_type="stock",
                                  evidence_text=ev)], channel="FinVid")
    assert c["ticker"] is None                                                       # not IREN on a guess
    # The loader applies the same channel-scoped spelling to stored rows.
    assert cc.resolved_ticker(_named("new", channel_name="Couch Investor")) == "NU"
    assert cc.resolved_ticker(_named("new", channel_name="TheStreet")) is None
    # TheStreet's US fund managers mean Merck & Co.; a German channel's Merck is Merck KGaA.
    assert cc.resolved_ticker(_named("Merck", channel_name="TheStreet")) == "MRK"
    assert cc.resolved_ticker(_named("Merck", channel_name="HKCM")) is None


def test_ticker_after_a_ticker_cue_is_spoken():
    ev = "eingestiegen bei Deckers mit dem Ticker Deck. Was ist Deckers? Ist eine Outdoor Bekleidungsfirma"
    c = _validate(ev + ".", [_raw(subject_mention="Deckers", ticker_spoken="DECK", asset_type="stock",
                                  evidence_text=ev)])
    assert c["ticker"] == "DECK" and c["ticker_source"] == "spoken"


def test_a_ticker_word_in_lower_case_is_not_a_spoken_ticker():
    assert cm._spoken_in_evidence("NVDA", "I am buying more NVDA today")
    assert cm._spoken_in_evidence("MU", "loading up on $mu here")
    assert not cm._spoken_in_evidence("NEW", "New is a tremendously well-run company")
    assert not cm._spoken_in_evidence("T", "T is cheap")                  # one letter needs "$" or a cue
    assert not cm._spoken_in_evidence("NEW", "NEW IS A TREMENDOUSLY WELL-RUN COMPANY")   # all-caps captions


# --- markets, baskets, metrics, references ------------------------------------------------------


def test_market_subject_spellings_fold_to_the_canonical_keys():
    expected = {"THE S&P 500": "S&P 500", "S&P": "S&P 500", "Silber": "SILVER", "Nickey": "NIKKEI 225",
                "U.S. stocks": "US STOCK MARKET", "chip stocks": "SEMICONDUCTORS", "Brent": "OIL"}
    for name, key in expected.items():
        assert cm.market_key(name) == key, name
        assert cc.asset_key(_named(name)) == key, name
    assert cc.asset_key(_named("the market")) is None and cm.market_key("the market") is None
    # "<market> stocks" is the market only for equity indices and sectors.
    assert cm.market_key("DAX Aktien") == "DAX" and cm.market_key("oil stocks") == "ENERGY"
    assert cm.market_key("gold stocks") is None and cm.market_key("Dow Inc") is None
    # Every spelling lands on a key, and every key is its own spelling.
    assert set(cm.MARKET_SUBJECTS.values()) == set(cm._MARKET_SPELLINGS)
    assert all(cm.MARKET_SUBJECTS[key] == key for key in cm._MARKET_SPELLINGS)


def test_brand_is_brent_only_when_typed_as_a_commodity():
    ev = "Brand is above $15 this morning after more attacks on ships"
    c = _validate(ev + ".", [_raw(subject_mention="Brand", asset_type="commodity", evidence_text=ev)])
    assert c["subject_kind"] == "market" and c["canonical_entity_name"] == "OIL"
    assert cm.resolve_entity("Brand", None, ev, "T", "stock")["canonical_entity_name"] == "Brand"


def test_basket_subject_takes_no_single_ticker():
    # clm_f54d884e681735495469 -> MSFT from the title.
    ev = "I'm estimating their spending will remain elevated to 2028 and then start to moderate"
    c = _validate(ev + ".", [_raw(subject_mention="Amazon, Alphabet, Microsoft, and Meta Platforms",
                                  ticker_spoken="MSFT", asset_type="stock", evidence_text=ev)],
                  title="Massive Update for Microsoft Stock Investors! | MSFT Stock Analysis")
    assert c["ticker"] is None and c["subject_kind"] == "basket"
    for basket in ("Google and Meta", "SOL/BTC", "AT&T and Verizon", "Google and Constellation"):
        assert cm.resolve_entity(basket, None, "e", "T", "stock")["subject_kind"] == "basket", basket
    for whole in ("Hims & Hers", "AT&T", "Procter & Gamble", "Fair Isaac Corporation or FICO stock"):
        out = cm.resolve_entity(whole, None, "e", "T", "stock")
        assert out["subject_kind"] == "instrument" and out["ticker"], whole


def test_metric_subject_takes_the_single_title_asset_as_a_metric():
    # clm_dc46464326d604e04fa6 "The earnings per share" -> T (AT&T title).
    ev = "The earnings per share are likely to grow at a rate faster than revenue growth"
    c = _validate(ev + ".", [_raw(subject_mention="The earnings per share", ticker_spoken="T",
                                  asset_type="stock", evidence_text=ev)],
                  title="Is AT&T an Undervalued Dividend Stock to Buy for Passive Income Investors? | T Stock Analysis")
    assert c["ticker"] == "T" and c["subject_kind"] == "metric"
    assert c["entity_resolution_method"] == "title"
    # A company named inside the metric is its company; a segment typed "sector" is still a metric.
    assert cm.resolve_entity("Alphabet's revenue growth", None, "e", "T", "stock")["ticker"] == "GOOGL"
    aws = cm.resolve_entity("AWS segment", "AMZN", "e", "Is the Upside Worth the Risk in Amazon Stock?", "sector")
    assert aws["subject_kind"] == "metric" and aws["ticker"] == "AMZN"
    defense = cm.resolve_entity("defense and intelligence", "PL", "e",
                                "What's Going on With Planet Labs Stock? | PL Stock Deep Dive Part 1", "sector")
    assert defense["subject_kind"] == "metric" and defense["ticker"] == "PL"
    assert cm.resolve_entity("AI infrastructure", None, "e", "T", "sector")["subject_kind"] == "market"


def test_generic_subject_in_a_two_company_title_is_only_probable():
    ev = "the company is growing faster than PANW right now"
    title = "Better Buy: Fortinet or Palo Alto Networks? | FTNT Stock vs. PANW Stock"
    c = _validate(ev + ".", [_raw(subject_mention="the company", ticker_spoken="FTNT", asset_type="stock",
                                  evidence_text=ev)], title=title)
    assert c["ticker"] == "FTNT" and c["entity_resolution_status"] == "probable"
    assert c["entity_resolution_method"] == "title" and c["entity_resolution_confidence"] == 0.5
    c = _validate(ev + ".", [_raw(subject_mention="the company", ticker_spoken="ORCL", asset_type="stock",
                                  evidence_text=ev)], title=title)
    assert c["ticker"] is None


def test_generic_subject_method_is_title_not_explicit_mention():
    c = _validate("the company looks strong now", [_raw(subject_mention="the company", ticker_spoken="MU",
                                                        evidence_text="the company looks strong now")],
                  title="Why $MU is my top pick")
    assert c["ticker"] == "MU" and c["entity_resolution_method"] == "title"
    assert c["entity_resolution_confidence"] == 0.6
    # Without a model ticker, a one-company title still names the reference.
    c = _validate("the business keeps compounding", [_raw(subject_mention="the business",
                                                          evidence_text="the business keeps compounding")],
                  title="Why Is CAVA Stock Crashing, and is it a Buying Opportunity?")
    assert c["ticker"] == "CAVA" and c["entity_resolution_status"] == "confirmed"
    # ...but a stray capitalised word is not an asset to inherit, and the
    # default test title "T" names nothing.
    assert cm.resolve_entity("the stock", None, "e", "Why Kevin WARSH matters for rates", "stock")["ticker"] is None
    assert cm.resolve_entity("the stock", None, "e", "T", "stock")["ticker"] is None


def test_crypto_market_reference_takes_the_single_title_coin():
    ev = "honestly, if the low is already in, price could even push above 200K"
    c = _validate(ev + ".", [_raw(subject_mention="price", asset_type="crypto", evidence_text=ev)],
                  title="Bitcoin: What Has To Happen For 160,000")
    assert c["ticker"] == "BTC" and c["entity_resolution_status"] == "probable"
    assert c["resolution_notes"] == ["crypto_title_coreference"]
    # Never for a stock video's "the market", nor with two coins in the title.
    assert cm.resolve_entity("the market", None, ev, "Nvidia earnings", "stock")["ticker"] is None
    assert cm.resolve_entity("price", None, ev, "Bitcoin vs Ethereum", "crypto")["ticker"] is None


def test_title_tickers_are_typed_tickers_not_capitalised_words():
    assert cm.title_tickers("Better Buy: MSFT Stock vs. GOOG Stock") == {"MSFT", "GOOGL"}
    assert cm.title_tickers("Is AT&T a Buy? | T Stock Analysis") == {"T"}
    assert cm.title_tickers("Should Investors Buy Visa Instead of Mastercard? | V Stock vs. MA Stock") == {"V", "MA"}
    assert cm.title_tickers("The U.S. Central Bank, E-Commerce and a Plan B for AI") == set()
    assert cm.title_tickers("MARKET PULLS BACK A BIT, META MUSE TAKES OVER | MARKET CLOSE") == set()
    assert cm.title_assets("Down 40%, Is Grab Stock a Buy in September?") == {"GRAB"}
    # A curated name that is also an ordinary word needs capitals or "stock".
    assert cm.title_assets("Does Nike Have the Right Strategy Moving Forward? | NKE Stock Deep Dive") == {"NKE"}


# --- stored claims ------------------------------------------------------------------------------


def _stored_tsm(**over):
    row = _row("Parkev", "TSM", "bullish", "long", ticker_spoken="TSM", ticker_source="curated_mapping",
               canonical_entity_name="Taiwan Semiconductor Manufacturing Company",
               subject_mention="Taiwan Semiconductor Manufacturing Company", entity_resolution_status="confirmed",
               entity_resolution_method="explicit_mention", entity_resolution_confidence=1.0,
               evidence_text="I calculated a fair value for the stock at $609 per share",
               review_required=True, review_reasons=["ticker_not_in_evidence"], extraction_confidence="high")
    row.update(over)
    return row


def test_reresolve_corrects_stored_claims_without_reextraction(tmp_path, monkeypatch):
    stored = _stored_tsm()
    fixed = cc.reresolve(stored)
    assert fixed is not stored and stored["review_required"] is True            # pure
    assert fixed["review_reasons"] == [] and fixed["review_required"] is False
    assert fixed["ticker"] == "TSM" and fixed["resolution_version"] == cm.RESOLUTION_VERSION
    assert fixed["subject_kind"] == "instrument" and fixed["resolution_notes"] == []
    assert cc.reresolve(fixed) is fixed                                          # current version: unchanged
    # Other review reasons survive; a wrong stored ticker is dropped.
    still = cc.reresolve(_stored_tsm(review_reasons=["ticker_not_in_evidence", "evidence_not_found"]))
    assert still["review_reasons"] == ["evidence_not_found"] and still["review_required"] is True
    nasdaq = cc.reresolve(_named("Nasdaq", ticker="NASDAQ", ticker_spoken="NASDAQ", asset_type="index"))
    assert nasdaq["ticker"] is None and nasdaq["subject_kind"] == "market" and cc.asset_key(nasdaq) == "NASDAQ"
    # The loader applies it to every stored row.
    monkeypatch.setattr(research_state, "RESEARCH_DIR", str(tmp_path))
    state = research_state.load_state()
    research_state.store_claims([stored], state, stored["video_id"], "rk1")
    loaded = cc.load_canonical_claims(state)
    assert loaded[0]["review_required"] is False and len(cc.view_claims(loaded)) == 1


def test_reresolve_keeps_a_stored_or_newly_ambiguous_verdict():
    target = _named("Target", ticker=None, entity_resolution_status="ambiguous",
                    review_reasons=["ambiguous_entity"], review_required=True)
    out = cc.reresolve(target)
    assert out["entity_resolution_status"] == "ambiguous" and out["review_required"] is True
    coref = _named("the stock", entity_resolution_status="ambiguous", review_reasons=["ambiguous_coreference"],
                   video_title="Why Is CAVA Stock Crashing?")
    assert cc.reresolve(coref)["ticker"] is None                     # the transcript said ambiguous


def test_resolution_version_stays_out_of_the_run_key_and_extraction_versions():
    assert cm.SCHEMA_VERSION == "2" and cm.EXTRACTION_PROMPT_VERSION == "2" and cm.RESOLUTION_VERSION == "3"
    assert "RESOLUTION_VERSION" not in inspect.getsource(research_state)
    ev = "I like Chevron a lot here"
    c = _validate(ev + ".", [_raw(subject_mention="Chevron", evidence_text=ev)])
    assert c["resolution_version"] == cm.RESOLUTION_VERSION and c["subject_kind"] == "instrument"
    assert c["resolution_notes"] == []
    # Every verdict speaks the documented vocabulary.
    for subject, spoken, asset_type in (("Nasdaq", "NASDAQ", "index"), ("Google and Meta", None, "stock"),
                                        ("gross margin", None, "stock"), ("mogu", "MOGU", "stock"),
                                        ("price", None, "crypto")):
        out = cm.resolve_entity(subject, spoken, "e", "Bitcoin: what next", asset_type)
        assert out["subject_kind"] in cm.SUBJECT_KINDS and set(out["resolution_notes"]) <= cm.RESOLUTION_NOTES


# --- identity in analytics ----------------------------------------------------------------------


def test_placeholders_and_chart_vocabulary_are_not_assets():
    for name in ("SUPPORT ZONE", "SOL/BTC", "THEY", "DIE AKTIE", "50-WEEK SMA", "Markets", "Ethereum against Bitcoin"):
        assert cc.asset_key(_named(name)) is None, name
    assert cc.asset_key(_named("THE TRADEDESK")) == "TTD"
    assert cc.asset_key(_named("The Fed")) == "FED"
    assert cc.asset_key(_named("IonQ stock")) == "IONQ"
    assert cc.asset_key(_named("growth stocks")) == "GROWTH STOCKS"             # a category keeps its plural
    assert cc.asset_key(_named("Google and Meta", subject_kind="basket")) is None


def test_display_name_keeps_apostrophes():
    assert cc._display_name("MCDONALD'S") == "McDonald's"
    assert cc._display_name("DICK'S SPORTING GOODS") == "Dick's Sporting Goods"
    assert cc._display_name("O'REILLY") == "O'Reilly"
    assert cc._display_name("T-MOBILE") == "T-Mobile"
    assert cc._display_name("OIL") == "Oil" and cc._display_name("DXY") == "DXY"
    assert cc._display_name("S&P 500") == "S&P 500"


def test_priceable_symbol_is_registry_first():
    spacex = _row("A", "SPCX", "bullish", "unspecified", canonical_entity_name="SpaceX", subject_mention="SpaceX",
                  asset_type="private_company")
    bitcoin = _row("A", "BTC", "bullish", "unspecified", canonical_entity_name="Bitcoin", subject_mention="Bitcoin",
                   asset_type="stock")
    assert cc.priceable_symbol(spacex) == "spcx.us"
    assert cc.priceable_symbol(bitcoin) == "btcusd"                              # never the btc.us Grayscale trust
    assert cc.priceable_symbol({"ticker": "STX", "asset_type": "crypto"}) is None   # Seagate, not Stacks
    assert cc.priceable_symbol({"ticker": "STX", "type": "stock"}) == "stx.us"
    assert cc.priceable_symbol({"ticker": "ZZZQ", "type": "stock"}) == "zzzq.us"
    assert cc.priceable_symbol({"ticker": "ZZZQ", "type": "index"}) is None
    assert cc.priceable_symbol({"ticker": "ANTHROPIC", "type": "other"}) is None
    assert cc.priceable_symbol({"ticker": "SOLANA", "type": "crypto"}) == "solusd"   # a name in the ticker field
    assert cc.priceable_symbol({"ticker": "CRO", "type": None}) == "crousd"


def test_aggregate_entry_type_comes_from_the_instrument():
    claims = [_row("A", "SPCX", "bearish", "unspecified", canonical_entity_name="SpaceX", subject_mention="SpaceX",
                   asset_type="private_company"),
              _row("B", "SPCX", "bearish", "unspecified", video="vb", canonical_entity_name="SpaceX",
                   subject_mention="SpaceX", asset_type="private_company")]
    entry = cc.aggregate_views_by_asset(claims)["SPCX"]
    assert entry["type"] == "stock" and cc.priceable_symbol(entry) == "spcx.us"
    assert cc.aggregate_views(claims)[("SPCX", "unspecified")]["type"] == "stock"


def test_metric_and_basket_subjects_leave_view_analytics():
    claims = [_row("A", "NVDA", "bullish", "unspecified"),
              _row("B", "T", "bullish", "unspecified", video="vb", canonical_entity_name="The earnings per share",
                   subject_mention="The earnings per share", subject_kind="metric"),
              _row("C", None, "bullish", "unspecified", video="vc", canonical_entity_name="Google and Meta",
                   subject_mention="Google and Meta", subject_kind="basket")]
    assert set(cc.aggregate_views_by_asset(claims)) == {"NVDA"}
    assert set(cc.aggregate_views(claims)) == {("NVDA", "unspecified")}
    # The metric keeps its company ticker for traceability.
    assert cc.resolved_ticker(claims[1]) == "T" and cc.asset_key(claims[1]) is None


def _disclosure(evidence, ticker, published="2026-10-01", source="P", position="owns_unspecified"):
    return _row(source, ticker, "not_applicable", "unspecified", published=published, claim_type="portfolio_disclosure",
                portfolio_disclosure=position, evidence_text=evidence, video=f"v-{ticker}-{published}")


def test_disclosure_actions_come_from_the_evidence():
    assert cc.disclosed_action("I sold my PayPal stock earlier this year") == "sold"
    assert cc.disclosed_action("I no longer own Tesla") == "sold"
    assert cc.disclosed_action("I sold 1/3 of my Trade Desk stock position") == "trimmed"
    assert cc.disclosed_action("I've cut my position in Netflix already by half") == "trimmed"
    assert cc.disclosed_action("Ich habe teilweise verkauft") == "trimmed"
    assert cc.disclosed_action("I bought more Micron after earnings") == "added"
    assert cc.disclosed_action("I haven't sold any of it") is None
    assert cc.disclosed_action("I still own every share that I've bought.") is None
    assert cc.disclosed_action("I sold Tesla and bought Nvidia", "Nvidia") == "added"
    rows = cc.portfolio_disclosures([
        _disclosure("Now, I sold my PayPal stock earlier this year.", "PYPL"),
        _disclosure("I sold 1/3 of my Trade Desk stock position", "TTD"),
        _disclosure("I bought more Nvidia this week", "NVDA"),
        _disclosure("I own Tesla.", "TSLA", published="2026-09-01"),
        _disclosure("I no longer own Tesla shares.", "TSLA", published="2026-10-05", position="no_position"),
    ])
    text = cc.format_portfolio_disclosures(rows)
    assert "sold PYPL, TSLA" in text and "trimmed TTD" in text and "added to NVDA" in text
    assert "owns" not in text                                       # the latest action per creator and ticker wins
    # On one date a trade outranks a holding statement made in the same video.
    same_day = cc.portfolio_disclosures([_disclosure("I bought this stock earlier this year.", "LLY"),
                                         _disclosure("And in fact, I own Eli Lilly stock.", "LLY")])
    assert cc.disclosure_groups(same_day) == [("P", [("added to", "LLY")])]
    assert cc.disclosed_action("where I announced that I added to my Netflix position") == "added"


# --- review fixes: capital tickers, coin pairs, articles, ordinary words, tables ------------------------


def test_a_subject_written_as_a_registered_ticker_is_that_ticker():
    # FinVid 2026-10-14: subject "STX", spoken STX, the evidence never repeats it.
    ev = "We think it's a $1000 stock. It's suffering from the same things as Micron"
    assert cm.resolve_entity("STX", "STX", ev, "T", "stock")["ticker"] == "STX"
    assert cm.resolve_entity("Stx", "STX", ev, "T", "stock")["ticker"] is None        # not written as a ticker
    assert cm.resolve_entity("ZORB", "ZORB", ev, "T", "stock")["ticker"] is None      # not a registry row
    assert cm.resolve_entity("Lulu", None, "Lulu could turn around", "T", "stock")["ticker"] == "LULU"
    # An ETF the model typed "stock" is still that listing; a coin never is.
    import instruments
    assert cm.resolve_entity("RSP", None, "the broader market, RSP as an ETF", "T", "stock")["ticker"] == "RSP"
    assert instruments.resolve_instrument("RSP", None, "stock").symbol == "rsp.us"
    assert instruments.resolve_instrument("BTC", None, "stock") is None


def test_two_coins_without_a_slash_are_a_pair_not_one_coin():
    ev = "sole BTC could be moving into the resistance area between 0.0015 and 0.0019"
    title = "SOLANA: $133 Is the Next Major Target Zone"
    for pair in ("sole BTC", "ETH BTC", "Ethereum Bitcoin"):
        out = cm.resolve_entity(pair, "SOL", ev, title, "crypto")
        assert out["subject_kind"] == "basket" and out["ticker"] is None, pair
    assert cm.resolve_entity("Wrapped Bitcoin", None, "e", "T", "crypto")["subject_kind"] != "basket"
    assert cm.resolve_entity("ETH BTC", None, "e", "T", "stock")["subject_kind"] != "basket"   # crypto only


def test_a_german_article_does_not_hide_a_name():
    ev = "Der Bitcoin hat in den letzten zwei Tagen seine Korrektur begonnen"
    assert cm.resolve_entity("Der Bitcoin", None, ev, "T", "crypto")["ticker"] == "BTC"
    row = cc.reresolve(_named("Der Bitcoin", asset_type="crypto", evidence_text=ev))
    assert cc.asset_key(row) == "BTC"
    assert cm.market_key("der DAX") == "DAX" and not cc.is_placeholder_asset("DER BITCOIN")
    assert cc.is_placeholder_asset("DIE FETT")                                   # still nothing the tables know


def test_ordinary_words_are_no_asset_mentions():
    ev = "it could double from here"
    assert cm.resolve_entity("it", None, ev, "This Upstart Could Be the Next Big Winner", "stock")["ticker"] is None
    assert cm.resolve_entity("it", None, ev, "Why UPSTART Stock Could Double", "stock")["ticker"] == "UPST"
    assert tn.alias_mentions("you have to outrun the avalanche") == []
    assert [m.group(1) for m in tn.alias_mentions("AVALANCHE and Upstart stock")] == ["AVALANCHE", "Upstart"]
    assert tn._assets_in("loans to upstart cloud computing companies") == set()


def test_bonds_name_no_issuer_and_sector_funds_have_keys():
    assert cm.market_key("Staatsanleihen") == "BONDS" and cm.market_key("bonds") == "BONDS"
    assert cm.market_key("Treasuries") == "US TREASURIES" and cm.market_key("10-year Treasury") == "US TREASURIES"
    expected = {"health care": "HEALTHCARE", "industrials": "INDUSTRIALS", "staples": "CONSUMER STAPLES",
                "utility sector": "UTILITIES", "materials": "MATERIALS", "banks": "BANKS",
                "healthcare stocks": "HEALTHCARE"}
    for name, key in expected.items():
        assert cm.market_key(name) == key, name
    import creator_tracker
    keys = set(cm._MARKET_SPELLINGS)
    assert set(creator_tracker.PROXIES) <= keys                 # every priced key is a market key


def test_listed_tickers_and_coin_collisions_follow_the_explicit_type():
    from signals_data import canonical_ticker
    assert canonical_ticker({"ticker": "FORD", "name": "Forward Industries"}) == "FORD"   # Nasdaq: FORD
    assert canonical_ticker({"ticker": None, "name": "Ford"}) == "F"
    assert cm.fold_spoken_ticker("FORD") == ("FORD", False)
    for ticker in ("SUI", "VET", "AXS", "FET", "TRX"):
        assert cc.priceable_symbol({"ticker": ticker, "asset_type": "stock"}) == f"{ticker.lower()}.us", ticker
    assert cc.priceable_symbol({"ticker": "SUI", "asset_type": "crypto"}) == "suiusd"
    assert cc.priceable_symbol({"ticker": "SUI"}) == "suiusd"                            # untyped: the registry
    import channel_scorecard as cs
    for ticker in ("MOGU", "RCI"):                       # known-wrong symbols are never built
        assert cs.symbol_for({"ticker": ticker, "name": ticker, "type": "stock"}) is None, ticker
    assert canonical_ticker({"name": "Novo Nordless"}) == "NVO" and canonical_ticker({"name": "nCino"}) == "NCNO"


def test_backlog_channel_spellings_and_market_spellings():
    # Phantom by HKCM's transcripts spell VeChain's gas token several ways in
    # the same passages; only that channel's reading is folded.
    assert cc.resolved_ticker(_named("WTHO", channel_name="Phantom by HKCM")) == "VTHO"
    assert cc.resolved_ticker(_named("BTHO", channel_name="Phantom by HKCM")) == "VTHO"
    assert cc.resolved_ticker(_named("WTHO", channel_name="TheStreet")) is None
    # HKCM's live streams say "Nastech": the Nasdaq market key, never a ticker.
    assert cm.market_key("Nastech") == "NASDAQ" and cm.market_key("Nesteck") == "NASDAQ"
