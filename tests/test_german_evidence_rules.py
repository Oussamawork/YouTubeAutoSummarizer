"""German evidence rules: recommendation actions, positions and adoption.

The validator decides recommendation_action, portfolio_disclosure and
host_position from the evidence's own wording, and that wording used to be
read with English patterns only. In fKyDalNgdxs (an HKCM livestream,
2026-10-05) it reset 11 German actions to "none" and stored "sold", "short"
and "not invested" as owns_unspecified. The rule set now follows the claim's
transcript_language (claims.EVIDENCE_RULES). Unless marked, every German
phrase below is from the stored HKCM / Phantom by HKCM transcripts.
"""
import pytest

import canonical_claims as cc
import claims as cm
import transcript_normalize as tn
from tests.test_canonical_analytics import _claim as _row


def _ctx(language):
    return {"video_id": "fKyDalNgdxs", "channel_id": "c1", "channel_name": "HKCM",
            "video_title": "Die Zeit läuft ab! Alle Märkte LIVE", "published_at": "2026-10-05T18:55:23+00:00",
            "transcript_source": "supadata", "run_key": "rk", "extraction_model": "m",
            "transcript_language": language}


def _validate(evidence, language="de", text=None, **over):
    raw = {"attribution_type": "speaker_personal_view", "claim_type": "recommendation", "stance": "bullish",
           "subject_mention": "Tesla", "evidence_text": evidence, "extraction_confidence": "high",
           "is_forward_looking": False}
    raw.update(over)
    claims, warnings = cm.validate_claims([raw], tn.normalize_transcript(text or evidence, "v1"), _ctx(language))
    return claims[0], warnings


# --- recommendation actions -------------------------------------------------------------------------------

# The 11 actions the English-only check reset in fKyDalNgdxs, as the extractor set them.
FKYDALNGDXS_ACTIONS = [
    ("accumulate", "Altcoins", "aber es ist eine schöne Zeit zu akkumulieren und äh ja, hab da dementsprechend auch "
                               "diese diese knallharte Meinung wohlgemerkt in Bezug auf Altcoins"),
    ("hold", "Solana", "Ich denke äh Solana ist definitiv ein Wert, wenn wir uns die Altcoins anschauen, den ich von "
                       "den alten Dinos 100% halten würde."),
    ("buy", "Solana", "194 US-Dollar sehen und äh da würde ich eventuell im Rahmen des Altcoins noch mal eine "
                      "Kaufzielzone setzen."),
    ("buy", "Alibaba", "Genau bei 107$, da sind wir ganz leicht runtergegangen und äh sind jetzt wieder leicht drüber "
                       "bei 110. Also ist für uns auf diesem Level ein klarer Kauf."),
    ("sell", "Alibaba", "Ansonsten müssen wir uns wirklich überlegen, die Position auch komplett zu schließen."),
    ("buy", "Tencent", "Jetzt sieht's auch immer mehr danach aus, dass wir auch wirklich in dieser C-Welle drin sind. "
                       "Hier unten sehr attraktiver Kauf meines Erachtens"),
    ("buy", "WTI", "Das könnte ich mir aktuell auch vorstellen. Also den Long würde ich auf jeden Fall noch mal nehmen."),
    ("buy", "SpaceX", "Aber ich sage auch, also wenn das Ding da unten diesen Bereich zwischen 86 und 48 hittet, dann "
                      "werde ich mich da auch eindecken mit dem Titel."),
    ("buy", "Gold", "also schauen hier auf einen Long Trade aktuell ähm gibt's auch eine Zielzone dazu"),
    ("short", "Gold", "und würden dann eben entsprechend noch mal von der Gegenbewegung ausgehen, die wir dann "
                      "wegschorten und dann den in den weiteren Abverkauf hinein übergehen werden."),
    ("accumulate", "XP", "Ich würde die Preise wirklich begrüßen, ja, wenn wir dann noch mal runterkommen, einfach weil "
                         "ich dann noch mal günstig aufladen kann, weil ich von der Story überzeugt bin"),
]

# More explicit instructions from the other German runs.
OTHER_GERMAN_ACTIONS = [
    ("buy", "Lululemon", "Wir stehen kurz davor, wir wollen einkaufen bei Lulu."),
    ("buy", "Solana", "Es gingen gestern insgesamt zwei Pushnachrichten zu Solana raus. Wir steigen auf den aktuellen "
                      "Niveaus ein."),
    ("buy", "Meta", "Sobald diese erreicht wird, werden wir einen Einstieg tätigen bei Meta"),
    ("buy", "Lululemon", "Wir gehen Positionen ein bei Aktien, die diese Korrektur schon vorweggenommen haben"),
    ("buy", "Nike", "Das ist dann aber definitiv eine historische Kaufgelegenheit, die ich hier bei der Nike Aktie sehe"),
    ("buy", "Bitcoin", "und finaler Abverkauf in diesem Bärenmarkt unterhalb von 84,93 ein Must-Buy"),
    ("add", "Gold", "Dort haben wir eine Kaufzone hinterlegt. Dort wollen wir weitere Positionen aufstocken beim Gold."),
    ("add", "The Trade Desk", "Und im Endeffekt haben wir bei dem keine Wahl als nachzukaufen. Wir müssen nachkaufen."),
    ("accumulate", "China", "dass ich persönlich für mein Depot in den kommenden Wochen und Monaten meine Bestände "
                            "in diesem Bereich deutlich deutlich ausbauen werde"),
    ("hold", "Tesla", "Das bleiben wir sowieso. Wir halten hier garantiert unsere Position bei Tesla."),
    ("hold", "Mara Holdings", "Mara Holdings hat sich sehr schön entwickelt. Den wollen wir weiter halten"),
    ("reduce", "Bitcoin", "Deswegen nur Teilprofite realisieren, wer hier drin ist"),
    ("reduce", "Snowflake", "dass wir auch da die Hälfte unserer Position getrimmt haben nach fast einer Verdopplung"),
    ("sell", "Bitcoin", "wenn du erst deine Haltefristen ausläuft, verkauf lieber jetzt, nehm lieber dann die "
                        "Spekulationssteuer in Kauf."),
    ("avoid", "Day Trading", "Day Trading. Finger weg davon."),
    ("avoid", "Schweiz", "deswegen rate ich eher von der Schweiz ab, gerade weil es in Europa etliche andere Länder gibt"),
    ("avoid", "ETFs", "dass ich auf 10 Jahre Sicht überhaupt nicht mehr in ETFs investieren würde."),
    ("avoid", "SpaceX", "SpaceX halte ich für sehr hoch bewertet. Ich würde es nicht machen, aber das ist einfach meine "
                        "grundsätzliche Einstellung."),
    ("avoid", "Gold", "Mit vorsichtig meine ich jetzt nicht verkaufen, sondern ich meine damit aktuell jetzt nicht mehr "
                      "auf diesen Levels hier oben im Bereich 2700, 2600 nochmal nachzukaufen."),
    ("watch", "Staatsanleihen", "warum ich für meinen Teil aktuell überlege, in zehnjährige Staatsanleihen zu investieren"),
]


@pytest.mark.parametrize("action,subject,evidence", FKYDALNGDXS_ACTIONS + OTHER_GERMAN_ACTIONS)
def test_an_explicit_german_instruction_keeps_its_action(action, subject, evidence):
    c, warnings = _validate(evidence, subject_mention=subject, recommendation_action=action)
    assert c["recommendation_action"] == action, warnings
    assert not any("recommendation" in w for w in warnings)


@pytest.mark.parametrize("action,subject,evidence", FKYDALNGDXS_ACTIONS)
def test_the_english_rule_set_reads_none_of_them(action, subject, evidence):
    # The reported behaviour, now confined to English transcripts: the rule
    # set follows the claim's language, and English wording is not German.
    c, warnings = _validate(evidence, language="en", subject_mention=subject, recommendation_action=action)
    assert c["recommendation_action"] == "none"
    assert any("not stated in evidence" in w for w in warnings)


NOT_AN_INSTRUCTION = [
    # Praise, mood and opinion (the "praise alone is none" rule).
    ("buy", "Rheinmetall", "Rheinmetall ist ein tolles Unternehmen mit einer super Story."),          # constructed
    ("watch", "Chevron", "genauso Chevron, genauso ExxonMobil, das könnten sehr interessante Werte sein."),
    ("accumulate", "SpaceX", "ist für mich einfach eine Once in a Lifetime Opportunity über einen Zeithorizont von 10 "
                             "plus Jahre."),
    ("accumulate", "Altcoins", "Auch ich bin da inzwischen tatsächlich sehr sehr bullisch drauf."),
    ("avoid", "Barilla", "Barilla blau. Ganz klar Garbage."),
    ("hold", "SpaceX", "SpaceX halte ich für sehr hoch bewertet."),
    ("buy", "Nvidia", "Nvidia ist für die Zukunft der KI gut positioniert."),                          # constructed
    # Chart talk: support that "holds", a sell-off, a selling reaction, a target zone.
    ("hold", "Bitcoin", "Die Unterstützung bei 100.000 muss jetzt halten, sonst geht es tiefer."),     # constructed
    ("hold", "Bitcoin", "solange wir die 214 weiter halten können, bleibt das Szenario intakt"),       # constructed
    ("short", "Gold", "in diesem Areal sollte es so oder so zu einer Verkaufsreaktion kommen"),
    ("sell", "Gold", "Wir gehen davon aus, dass dieser Abverkauf noch nicht vorbei ist"),
    ("buy", "Gold", "Wir gehen aber davon aus, dass wir noch mal einmal tiefer kommen sollten in unsere Zielzone."),
    ("buy", "Bitcoin", "nimm lieber die Spekulationssteuer in Kauf"),                                   # an idiom
    ("buy", "Vechain", "Mit einem Einstieg von etwa 32 US-Dollar würde das einer prozentualen Rendite von 900% "
                       "entsprechen."),
    ("hold", "Tesla", "Ich halte euch auf dem Laufenden."),                                            # constructed
]


@pytest.mark.parametrize("action,subject,evidence", NOT_AN_INSTRUCTION)
def test_praise_opinion_and_chart_talk_are_no_instruction(action, subject, evidence):
    c, warnings = _validate(evidence, subject_mention=subject, recommendation_action=action)
    assert c["recommendation_action"] == "none"
    assert any(f"recommendation '{action}'" in w for w in warnings)


def test_a_refused_purchase_states_avoid_and_never_buy():
    for evidence in ("Ich würde Tesla hier nicht kaufen.", "Kaufen würde ich Tesla hier nicht.",
                     "Ich würde Tesla auf diesen Levels hier oben auf keinen Fall nachkaufen."):    # constructed
        buy, warnings = _validate(evidence, recommendation_action="buy")
        avoid, _ = _validate(evidence, recommendation_action="avoid")
        assert buy["recommendation_action"] == "none" and avoid["recommendation_action"] == "avoid", evidence
        assert "contradicts the evidence (avoid)" in warnings[0]
    # A refused sale is a hold, never a sale.
    sell, _ = _validate("Ich würde Tesla jetzt nicht verkaufen.", recommendation_action="sell")
    hold, _ = _validate("Ich würde Tesla jetzt nicht verkaufen.", recommendation_action="hold")
    assert sell["recommendation_action"] == "none" and hold["recommendation_action"] == "hold"
    # "nicht verkaufen, sondern halten": the negation ends with its clause.
    c, _ = _validate("Ich würde Tesla jetzt nicht verkaufen, sondern weiter halten.", recommendation_action="hold")
    assert c["recommendation_action"] == "hold"


def test_words_that_look_like_a_negation_negate_nothing():
    for evidence in ("Ich kaufe nicht nur Tesla, sondern auch Nvidia.",                              # constructed
                     "Ich würde nicht lange zögern und hier kaufen.",                                # constructed
                     "Und im Endeffekt haben wir bei dem keine Wahl als nachzukaufen."):
        c, _ = _validate(evidence, recommendation_action="buy")
        assert c["recommendation_action"] == "buy", evidence
    # "Ich glaube nicht, dass ..." negates the belief, not the purchase.
    c, _ = _validate("Ich glaube nicht, dass es noch tiefer geht, deshalb kaufen wir hier.", recommendation_action="buy")
    assert c["recommendation_action"] == "buy"


def test_an_action_of_another_kind_than_the_evidence_states_is_dropped():
    # German instructions say WHICH action: a model "sell" on a purchase, or a
    # "watch" on two trades, is not what the speaker said.
    c, warnings = _validate("Wir steigen auf den aktuellen Niveaus ein.", recommendation_action="sell")
    assert c["recommendation_action"] == "none" and "contradicts the evidence (buy)" in warnings[0]
    c, _ = _validate("am nächsten Montag werden wir genauso bei Ethereum auch einen Long und ein Short eingehen.",
                     subject_mention="Ethereum", recommendation_action="watch")
    assert c["recommendation_action"] == "none"
    # A short is also a sale.
    c, _ = _validate("Wir sind short im Hang Seng und würden die Gegenbewegung wegshorten.", subject_mention="Hang Seng",
                     recommendation_action="sell")
    assert c["recommendation_action"] == "sell"


# --- positions --------------------------------------------------------------------------------------------

# (expected position, subject, evidence, what the extractor said)
GERMAN_POSITIONS = [
    # The four fKyDalNgdxs disclosures stored as owns_unspecified, with the extractor's own value.
    ("no_position", "Reheinmetall", "weil ich in einem Moment von von naiver moralischer ähm ja äh gemeint habe, ich "
                                    "muss ja den moralischen machen. Ähm ich die scheiß Aktien verkauft habe und ähm ja, "
                                    "mich heute da schwarzärger drüber", "no_position"),
    ("short", "Hansen", "wir sind ja auch schon seit seit vielen Monaten wirklich in einem Monstershort drin im Hansen.",
     "short"),
    ("no_position", "WTI", "Also WTI sind wir ja raus aus dem Trade.", "no_position"),
    ("no_position", "SpaceX", "Okay, du bist du bist also noch nicht drin. Ja. Neö, aktuell nicht", "no_position"),
    # The report's phrasings (constructed); the evidence decides, whatever the model said.
    ("no_position", "Rheinmetall", "Wir haben Rheinmetall verkauft.", "owns_unspecified"),
    ("short", "Hang Seng", "Wir sind short im Hang Seng.", "owns_unspecified"),
    ("no_position", "WTI", "Wir sind aus dem WTI-Trade raus.", "owns_unspecified"),
    ("no_position", "Nvidia", "Bei Nvidia sind wir nicht investiert.", "owns_unspecified"),
    # More from the stored transcripts.
    ("no_position", "Ethereum", "Ethereum, weil ich stand heute noch immer nicht in Ethereum investiert bin",
     "owns_unspecified"),
    ("no_position", "Adidas", "Und ihr seht es, wir steigen bei Adidas aus, Profit 3,62 %.", "owns_unspecified"),
    ("short", "Service Now", "weil ich da short bin auf den Titel, also ich generell halt einfach eine bärische Prognose "
                             "habe", "owns_unspecified"),
    ("short", "Shopify", "haben die Position dann veräußert und sind short reingegangen, wo wir jetzt auch schon wieder "
                         "Teilprofite heute realisiert haben und die restliche Position eben in Höhe von 50% halten.",
     "owns_unspecified"),
    ("long", "Nasdaq", "Aktuell befinden wir uns long im Markt. Wir halten eine Long Position.", "long"),
    # A portion sold leaves the rest held.
    ("owns_unspecified", "Snowflake", "Wir steigen teilweise bei Snowflake aus, das heißt, wir haben die Hälfte unserer "
                                      "Position verkauft, einfach 50 % unserer Aktien.", "no_position"),
    ("owns_unspecified", "Atlassian", "Wir haben eine Position von dreien, die wir halten, verkauft.", "no_position"),
    # Holdings.
    ("owns_unspecified", "SpaceX", "So, ist übrigens die einzige Aktie, die ich bespare.", "long"),
    ("owns_unspecified", "Chevron", "Wir sind ja zum einen in Chevron und zum anderen in Occidental Petroleum "
                                    "investiert im Rahmen unseres US Titan Paketes", "owns_unspecified"),
    ("owns_unspecified", "Solana", "ist meine größte ähm Altcoin Position. Solana und halte ich auch weiterhin.", "long"),
    ("owns_unspecified", "Nokia", "Also, ich habe mir den aktuell auch noch mal gekauft.", "long"),
]


@pytest.mark.parametrize("expected,subject,evidence,model_value", GERMAN_POSITIONS)
def test_a_german_position_statement_keeps_its_position(expected, subject, evidence, model_value):
    c, _ = _validate(evidence, claim_type="portfolio_disclosure", subject_mention=subject, stance="not_applicable",
                     recommendation_action="none", portfolio_disclosure=model_value)
    assert c["portfolio_disclosure"] == expected
    assert c["claim_type"] == "portfolio_disclosure" and c["stance"] == "not_applicable"
    assert c["recommendation_action"] == "none" and cm.carries_view(c) is False


@pytest.mark.parametrize("evidence", [
    "Ich würde Tesla hier nicht kaufen.",
    "Ich würde nicht kaufen.",
    "SpaceX halte ich für sehr hoch bewertet.",
    # HKCM's "wir" is the price, not the portfolio.
    "Jetzt sieht's auch immer mehr danach aus, dass wir auch wirklich in dieser C-Welle drin sind.",
    "Genau bei 107$, da sind wir ganz leicht runtergegangen und sind jetzt wieder leicht drüber bei 110.",
    # A chart zone is not a position.
    "wir stehen in der Zielzone für den S&P 500 und wir haben die Short Zielzone für den Dow Jones",
    "Ich habe Tesla nicht gekauft.",                                                                   # constructed
    "Die Unterstützung muss halten.",                                                                  # constructed
])
def test_refusals_chart_talk_and_the_price_are_no_position(evidence):
    assert cm.portfolio_position(evidence, "de") is None


def test_a_german_holding_without_a_view_is_a_disclosure_and_a_chart_zone_leaves_the_view_alone():
    # As in English ("I own Tesla"): ownership alone is no view.
    c, _ = _validate("Ich bin in Tesla investiert.", claim_type="stance", stance="neutral", recommendation_action="none")
    assert c["claim_type"] == "portfolio_disclosure" and c["portfolio_disclosure"] == "owns_unspecified"
    assert c["stance"] == "not_applicable"
    # A short target zone on a chart is no short position: the bearish view stays a view.
    evidence = "wir stehen in der Zielzone für den S&P 500 und wir haben die Short Zielzone für den Dow Jones erreicht"
    c, _ = _validate(evidence, claim_type="stance", stance="bearish", subject_mention="Dow Jones", asset_type="index",
                     recommendation_action="none")
    assert c["claim_type"] == "stance" and c["stance"] == "bearish" and c["portfolio_disclosure"] == "not_stated"


# --- host position ----------------------------------------------------------------------------------------

JUSOV = ("Emil Jusov hat geschrieben: 35% Kursverlust an einem Tag machen eine Aktie nicht automatisch billig. "
         "Das stimme ich ihm auch zu. FMC erwartet im dritten Quartal nur noch 65 bis 125 Millionen Umsatz.")


def _third_party(evidence, language="de", **over):
    raw = {"attribution_type": "speaker_quoting_third_party", "attributed_person_or_organization": "Goldman Sachs",
           "claim_type": "third_party_view", "subject_mention": "Gold", "asset_type": "commodity", "stance": "bullish",
           "is_forward_looking": True, "forecast_metric": "price", "forecast_direction": "increase"}
    raw.update(over)
    return _validate(evidence, language=language, **raw)[0]


def test_a_german_adoption_makes_the_view_the_speakers_own():
    c = _third_party(JUSOV, attributed_person_or_organization="Emil Jusov", subject_mention="FMC", asset_type="stock",
                     stance="bearish", forecast_direction="decrease")
    assert c["host_position"] == "adopted" and c["attribution_type"] == "speaker_personal_view"
    assert c["third_party_origin"] == "Emil Jusov" and c["claim_type"] == "forecast" and c["stance"] == "bearish"
    # Read with the English set it stays the reported view it was stored as.
    c = _third_party(JUSOV, language="en", attributed_person_or_organization="Emil Jusov", subject_mention="FMC",
                     asset_type="stock", stance="bearish", forecast_direction="decrease")
    assert c["host_position"] == "neutral" and c["claim_type"] == "third_party_view"


@pytest.mark.parametrize("reply,expected", [                                                         # constructed
    ("Das sehe ich genauso.", "adopted"),
    ("Da bin ich ganz bei ihnen.", "adopted"),
    ("Das ist auch mein Kursziel.", "adopted"),
    ("Da stimme ich nicht zu.", "rejected"),
    ("Das sehe ich ganz anders.", "rejected"),
    ("Das halte ich für übertrieben.", "rejected"),
    ("Die sind mir viel zu bullisch.", "rejected"),
    ("Mal schauen, was passiert.", "neutral"),
])
def test_where_a_german_speaker_stands_on_a_bank_s_target(reply, expected):
    c = _third_party("Goldman Sachs sieht Gold bei 5000 Dollar. " + reply)
    assert c["host_position"] == expected
    assert (c["claim_type"] == "third_party_view") is (expected != "adopted")
    if expected != "adopted":
        assert c["stance"] == "not_applicable"


def test_an_own_view_that_agrees_with_a_bank_needs_no_review():
    evidence = "Goldman Sachs sieht Gold bei 5000 Dollar, und das sehe ich genauso."                    # constructed
    c, _ = _validate(evidence, claim_type="forecast", subject_mention="Gold", asset_type="commodity",
                     forecast_metric="price", forecast_direction="increase", recommendation_action="none")
    assert "possible_third_party_view" not in c["review_reasons"]
    c, _ = _validate(evidence, language="en", claim_type="forecast", subject_mention="Gold", asset_type="commodity",
                     forecast_metric="price", forecast_direction="increase", recommendation_action="none")
    assert "possible_third_party_view" in c["review_reasons"]


# --- the disclosure report (canonical_claims) -------------------------------------------------------------

def test_german_trades_are_read_from_the_evidence():
    da = cc.disclosed_action
    assert da("Wir haben Rheinmetall verkauft.", "Rheinmetall", "de") == "sold"                        # constructed
    assert da("Ähm ich die scheiß Aktien verkauft habe und ähm ja", None, "de") == "sold"
    assert da("Also WTI sind wir ja raus aus dem Trade.", "WTI", "de") == "sold"
    assert da("Und ihr seht es, wir steigen bei Adidas aus, Profit 3,62 %.", "Adidas", "de") == "sold"
    assert da("Wir steigen teilweise bei Snowflake aus, das heißt, wir haben die Hälfte unserer Position verkauft, "
              "einfach 50 % unserer Aktien.", "Snowflake", "de") == "trimmed"
    assert da("Wir haben eine Position von dreien, die wir halten, verkauft.", "Atlassian", "de") == "trimmed"
    assert da("dass wir auch da die Hälfte unserer Position getrimmt haben", None, "de") == "trimmed"
    assert da("haben heute im Tagesverlauf auch direkt einmal Teilprofite realisiert", None, "de") == "trimmed"
    assert da("Am 24.02. diesen Jahres haben wir unsere erste Tranche gekauft", None, "de") == "added"
    assert da("Ganz im Gegenteil, ich habe hier auch weitere Zukäufe getätigt.", None, "de") == "added"
    assert da("Mara Holdings sind wir am 2. September eingestiegen", "Mara Holdings", "de") == "added"
    assert da("Wir sind hier vor nicht allzu langer Zeit am 2.9. mit einer ersten Position eingestiegen",
              None, "de") == "added"
    # A refused or negated trade, or a relative clause describing a holding, is no trade.
    assert da("Ich kaufe da aber auch nicht mehr nach.", None, "de") is None
    assert da("Wir haben Tesla nicht verkauft.", "Tesla", "de") is None                               # constructed
    assert da("eine Occidental Petroleum, die wir ja eingekauft haben", None, "de") is None
    assert da("Ich würde nicht kaufen.", None, "de") is None
    # The table follows the claim's language; an unknown language reads every table.
    assert da("Wir haben Rheinmetall verkauft.", "Rheinmetall", "en") is None
    assert da("Wir haben Rheinmetall verkauft.", "Rheinmetall") == "sold"
    assert da("I sold my PayPal stock earlier this year", "PayPal", "en") == "sold"


def _stored(evidence, ticker, subject, published, position="owns_unspecified", language="de", **over):
    """A German disclosure as the old rules stored it: owns_unspecified."""
    return _row("HKCM", ticker, "not_applicable", "unspecified", published=published, video=f"v-{ticker}-{published}",
                claim_type="portfolio_disclosure", portfolio_disclosure=position, evidence_text=evidence,
                subject_mention=subject, canonical_entity_name=subject, transcript_language=language, **over)


def test_the_disclosure_report_reads_stored_german_claims_under_the_current_rules():
    rows = cc.portfolio_disclosures([
        _stored("weil ich da short bin auf den Titel", "NOW", "Service Now", "2026-10-01"),
        _stored("Wir steigen teilweise bei Snowflake aus, das heißt, wir haben die Hälfte unserer Position verkauft",
                "SNOW", "Snowflake", "2026-10-02"),
        _stored("Okay, du bist du bist also noch nicht drin. Ja. Neö, aktuell nicht", "SPCX", "SpaceX", "2026-10-03"),
        _stored("So, ist übrigens die einzige Aktie, die ich bespare.", "TSLA", "Tesla", "2026-10-04"),
        _stored("Wir haben Rheinmetall verkauft.", "XOM", "Exxon", "2026-10-05"),                     # constructed
        # A specific stored position is kept.
        _stored("Wir sind short im Hang Seng.", "AAPL", "Apple", "2026-10-06", position="no_position"),
    ])
    assert {r["ticker"]: r["position"] for r in rows} == {
        "NOW": "short", "SNOW": "trimmed", "TSLA": "owns_unspecified", "XOM": "sold"}
    text = cc.format_portfolio_disclosures(rows)
    assert "short NOW" in text and "trimmed SNOW" in text and "owns TSLA" in text and "sold XOM" in text
    assert "SPCX" not in text and "AAPL" not in text
    # Read with the English rules (an English claim), the German wording states nothing.
    english = cc.portfolio_disclosures([_stored("weil ich da short bin auf den Titel", "NOW", "Service Now",
                                                "2026-10-01", language="en")])
    assert english == []


# --- language selection -----------------------------------------------------------------------------------

GERMAN_TRANSCRIPT = ("Ich sage es ganz klar, und das ist auch für uns wichtig. Wir sind hier noch nicht am Ende, "
                     "und ich denke, dass die Aktie noch sehr viel Potenzial hat. Also ist für uns auf diesem Level "
                     "ein klarer Kauf.")


def test_the_rule_set_follows_the_transcript_language():
    assert cm.rule_languages("de") == ("de",) and cm.rule_languages("en-US") == ("en",)
    assert cm.rule_languages("unknown") == cm.rule_languages(None) == cm.rule_languages("fr") == ("en", "de")
    assert cm.evidence_language("de", "The stock is cheap.") == "de"
    assert cm.evidence_language(None, GERMAN_TRANSCRIPT) == "de"
    assert cm.evidence_language("unknown", GERMAN_TRANSCRIPT) == "de"
    assert cm.evidence_language(None, "Kurz.") is None
    # Without a declared language the transcript's own language is detected.
    c, _ = _validate("Also ist für uns auf diesem Level ein klarer Kauf.", language=None, text=GERMAN_TRANSCRIPT,
                     recommendation_action="buy")
    assert c["recommendation_action"] == "buy" and c["transcript_language"] is None


def test_english_validation_is_unchanged():
    # Any English recommendation word still supports any action, and praise none.
    c, _ = _validate("I would buy Palantir under $20, that is my plan.", language="en", subject_mention="Palantir",
                     recommendation_action="buy", condition="under $20")
    assert c["recommendation_action"] == "buy"
    c, _ = _validate("Costco is a wonderful business.", language="en", subject_mention="Costco", claim_type="opinion",
                     recommendation_action="buy")
    assert c["recommendation_action"] == "none"
    c, _ = _validate("And yes I'm short Tesla.", language="en", claim_type="portfolio_disclosure",
                     stance="not_applicable", recommendation_action="none")
    assert c["portfolio_disclosure"] == "short"
    assert cm.portfolio_position("I sold my PayPal stock", "en") == "owns_unspecified"
