"""
A transcript must be in the language the channel speaks.

September 2026: Supadata served Arabic translations of English videos (More
Crypto Online, Parkev Tatevosian) and English translations of German ones
(HKCM), and the pipeline summarized, stored and mined them. These tests pin
the fix at every layer: verification rules, the Supadata request and its
track-aware retry, the source chain, the scraper's per-channel languages,
and the quarantine / re-fetch of captures that were already stored.
"""
import json

import pytest

import language_detect as ld
import research_backfill as rb
import research_state as rs
import scraper
import transcript as tr
import transcript_store as ts

ARABIC = ("لا يزال مخطط البيتكوين يتداول دون المتوسط المتحرك البسيط لمدة 50 أسبوعاً. "
          "لم يحدث الكثير خلال عطلة نهاية الأسبوع، لكن حقيقة بقائنا تحت هذا المستوى تعني " * 12)
ENGLISH = ("Germany's economy continues to head full throttle on a crash course and we think that "
           "the stock is going to move higher because of it. " * 20)
GERMAN = ("Kennt ihr diesen Chart hier? Er zeigt den langfristigen Aufwärtstrend der deutschen "
          "Industrieproduktion und das ist nicht gut für die Aktie, aber wir werden sehen. " * 20)
FRENCH = ("le marché va monter et nous pensons que cette action est une bonne affaire pour les "
          "investisseurs qui cherchent des rendements sur le long terme " * 25)


# --- verification rules ---------------------------------------------------

def test_script_census_and_detection():
    assert ld.script_of(ARABIC)["script"] == "arabic"
    assert ld.script_of(ENGLISH)["script"] == "latin"
    assert ld.script_of("12345 $%")["script"] == "unknown"
    assert ld.detect_language(ARABIC)["language"] == "unknown"
    assert ld.detect_language(ENGLISH)["language"] == "en"
    assert ld.detect_language(GERMAN)["language"] == "de"


def test_arabic_track_is_refused_for_an_english_channel():
    verdict = ld.verify_language(ARABIC, ["en"])
    assert verdict["ok"] is False and verdict["reason"] == "script_mismatch:arabic"


def test_english_track_is_refused_for_a_german_channel_and_vice_versa():
    assert ld.verify_language(ENGLISH, ["de"])["reason"] == "language_mismatch:en"
    assert ld.verify_language(GERMAN, ["en"])["reason"] == "language_mismatch:de"
    assert ld.verify_language(GERMAN, ["de"])["ok"] and ld.verify_language(GERMAN, ["de"])["language"] == "de"


def test_default_accepts_every_supported_language_only():
    assert ld.verify_language(ENGLISH)["ok"] and ld.verify_language(GERMAN)["ok"]
    # A long Latin-script transcript in neither language is another
    # translation, not an undecidable clip.
    assert ld.verify_language(FRENCH)["reason"] == "language_unrecognized"


def test_provider_tag_is_decisive_and_normalized():
    assert ld.verify_language(ENGLISH, ["en"], reported="ar")["reason"] == "language_mismatch:ar"
    short = ld.verify_language("short clip", ["en"], reported="en-US")
    assert short["ok"] and short["language"] == "en"
    # No tag and too short to judge: accepted on trust, language unknown.
    assert ld.verify_language("short clip", ["en"]) == {
        "ok": True, "language": "unknown", "reason": None, "script": "latin", "reported": None,
        "detected": "unknown", "accepted": ["en"]}


def test_language_lists_normalize_and_default(monkeypatch):
    assert ld.normalize_languages(["EN-us", "de", "en", ""]) == ("en", "de")
    assert ld.normalize_languages(None) == ld.DEFAULT_LANGUAGES == ("en", "de")
    assert ld.language_names(["de", "en"]) == "German, English"


# --- Supadata request and track-aware retry -------------------------------

@pytest.fixture
def supadata(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "SUPADATA_USAGE_FILE", str(tmp_path / "usage.json"))
    for name in ("SUPADATA_API_KEYS", "SUPADATA_API_KEY_2", "SUPADATA_API_KEY_3",
                 "SUPADATA_MONTHLY_BUDGET", "SUPADATA_DAILY_PACING"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SUPADATA_API_KEY", "k")
    monkeypatch.setattr(tr.time, "sleep", lambda *_: None)
    requests_made = []

    def serve(tracks):
        """A fake API: `tracks` maps language code → transcript text; the
        served track is the requested one, else the first."""
        class R:
            status_code = 200
            text = ""

            def __init__(self, lang):
                self.lang = lang

            def json(self):
                return {"content": tracks[self.lang], "lang": self.lang, "availableLangs": list(tracks)}

        def fake_get(url, headers=None, params=None, timeout=None):
            requests_made.append(dict(params))
            lang = params.get("lang") if params.get("lang") in tracks else next(iter(tracks))
            return R(lang)
        monkeypatch.setattr(tr.requests, "get", fake_get)
        return requests_made
    return serve


def test_channel_language_is_requested_and_default_is_not(supadata):
    made = supadata({"en": ENGLISH})
    assert tr._fetch_supadata("vid00000001", ["de"])[2] == "language_mismatch"   # only English exists
    assert made[0]["lang"] == "de"
    made.clear()
    text, _, reason = tr._fetch_supadata("vid00000001", ["en", "de"])
    assert reason == "ok" and text == ENGLISH.strip()
    assert "lang" not in made[0]       # two acceptable languages: let the provider pick, then verify


def test_wrong_track_is_swapped_for_an_acceptable_one_at_one_extra_credit(supadata):
    made = supadata({"ar": ARABIC, "en": ENGLISH})     # Arabic first, as served in September 2026
    text, exhausted, reason = tr._fetch_supadata("vid00000001", ["en", "de"])
    assert reason == "ok" and text == ENGLISH.strip() and exhausted is False
    assert [p.get("lang") for p in made] == [None, "en"]
    assert tr._load_usage()["count"] == 2              # both requests consumed a credit


def test_no_acceptable_track_reports_language_mismatch_without_key_rotation(supadata, monkeypatch):
    monkeypatch.setenv("SUPADATA_API_KEY_2", "k2")
    made = supadata({"ar": ARABIC})
    text, exhausted, reason = tr._fetch_supadata("vid00000001", ["en"])
    assert text == "" and reason == "language_mismatch" and exhausted is False
    # The key answered; asking the second key would only buy the same track.
    assert len(made) == 1 and tr._load_usage()["count"] == 1


def test_english_translation_of_a_german_video_is_refused(supadata):
    made = supadata({"en": ENGLISH, "de": GERMAN})
    text, _, reason = tr._fetch_supadata("vid00000001", ["de"])
    assert reason == "ok" and text == GERMAN.strip() and made[0]["lang"] == "de"


# --- the source chain ------------------------------------------------------

def test_chain_falls_through_to_gemini_in_the_channel_language(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setattr(tr, "_fetch_supadata", lambda vid, languages=None: (ARABIC, False, "ok"))
    seen = {}

    def gemini(vid, languages=None):
        seen["languages"] = languages
        return GERMAN, False, "gemini_ok"
    monkeypatch.setattr(tr, "_fetch_gemini_transcript", gemini)
    out = tr.get_transcript_from_video("https://youtu.be/dQw4w9WgXcQ", languages=["de"])
    assert out["transcript"] == GERMAN and out["reason"] == "gemini_ok" and out["language"] == "de"
    assert seen["languages"] == ("de",)
    assert out["language_check"]["source"] == "gemini" and out["language_check"]["ok"]


def test_every_source_wrong_means_no_transcript_not_a_wrong_one(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setattr(tr, "_fetch_supadata", lambda vid, languages=None: (ARABIC, False, "ok"))
    monkeypatch.setattr(tr, "_fetch_gemini_transcript", lambda vid, languages=None: (FRENCH, False, "gemini_ok"))
    monkeypatch.setattr(tr, "_fetch_youtube_transcript_api", lambda vid, languages=None: ARABIC)
    out = tr.get_transcript_from_video("https://youtu.be/dQw4w9WgXcQ", languages=["en"])
    assert out["transcript"] == "" and out["reason"] == "language_mismatch"
    assert out["budget_exhausted"] is False and out["language"] is None
    assert out["language_check"]["reason"] == "script_mismatch:arabic"


def test_gemini_prompt_names_the_language():
    assert "German" in tr._gemini_transcript_prompt(["de"]) and "Do not translate" in tr._gemini_transcript_prompt(["de"])
    assert tr._gemini_transcript_prompt(None) == tr.GEMINI_TRANSCRIPT_PROMPT


def test_youtube_transcript_api_asks_for_the_channel_language(monkeypatch):
    calls = {}

    class Snippet:
        text = "hello"

    class Api:
        def fetch(self, vid, languages=("en",)):
            calls["languages"] = list(languages)
            return [Snippet()]
    monkeypatch.setattr(tr, "YouTubeTranscriptApi", Api)
    assert tr._fetch_youtube_transcript_api("v", ["de"]) == "hello"
    assert calls["languages"] == ["de"]


# --- scraper: per-channel languages ---------------------------------------

def test_channel_option_reaches_the_transcript_fetch(monkeypatch):
    seen = []
    monkeypatch.setattr(scraper, "get_transcript_from_video",
                        lambda url, languages=None: seen.append(languages) or {"transcript": ""})
    scraper._summarize_video({"video_id": "v1", "video_url": "u", "video_title": "T", "channel_name": "C",
                              "published_at": "2026-07-01T00:00:00+00:00"},
                             languages=scraper._channel_languages({"language": "de"}))
    scraper._summarize_video({"video_id": "v1", "video_url": "u", "video_title": "T", "channel_name": "C",
                              "published_at": "2026-07-01T00:00:00+00:00"},
                             languages=scraper._channel_languages({"language": None}))
    assert seen == [["de"], list(ld.DEFAULT_LANGUAGES)]


def test_language_mismatch_defers_instead_of_summarizing(monkeypatch):
    monkeypatch.setattr(scraper, "get_transcript_from_video",
                        lambda url, languages=None: {"transcript": "", "reason": "language_mismatch",
                                                     "budget_exhausted": False})
    monkeypatch.setattr(scraper, "summarize_transcript", lambda *a, **k: pytest.fail("no summary from no text"))
    details = {"video_id": "v1", "video_url": "u", "video_title": "T", "channel_name": "C",
               "published_at": "2026-07-01T00:00:00+00:00"}
    body, outcome, decided, _ = scraper._summarize_video(details, languages=["en"])
    assert body is None and outcome == "no_transcript_deferred" and decided is False
    assert details["transcript_reason"] == "language_mismatch"


def test_verified_language_is_the_videos_language(monkeypatch):
    monkeypatch.setattr(scraper, "get_transcript_from_video",
                        lambda url, languages=None: {"transcript": GERMAN, "reason": "ok", "language": "de"})
    monkeypatch.setattr(scraper, "summarize_transcript", lambda t, title, **kw: "TLDR\n\n• x")
    details = {"video_id": "v1", "video_url": "u", "video_title": "T", "channel_name": "C",
               "published_at": "2026-07-01T00:00:00+00:00"}
    scraper._summarize_video(details, languages=["de"])
    assert details["transcript_language"] == "de"


# --- quarantine and re-fetch of stored captures ---------------------------

def _details(vid, channel_id="c-en"):
    return {"video_id": vid, "video_url": f"https://www.youtube.com/watch?v={vid}", "channel_id": channel_id,
            "channel_name": "Chan", "video_title": "T", "published_at": "2026-09-05T00:00:00+00:00",
            "duration_seconds": 900}


def _seed(vid, text, channel_id="c-en", run_key="rk1"):
    ts.store_transcript(_details(vid, channel_id), text, "supadata", "ok")
    state = rs.load_state()
    rs.update(state, vid, delivery_status="sent", research_status="complete", channel_id=channel_id,
              channel_name="Chan", video_title="T", transcript_stored=True)
    rs.store_claims([{"claim_id": f"clm-{vid}", "video_id": vid, "schema_version": "2", "claim_type": "forecast",
                      "subject_mention": "x"}], state, vid, run_key)
    rs.save_state(state)


LANGS = {"c-en": ["en"], "c-de": ["de"]}


def test_reject_foreign_retires_wrong_language_captures_and_their_claims():
    _seed("ar1", ARABIC, "c-en")            # Arabic translation of an English channel's video
    _seed("en1", ENGLISH, "c-de")           # English translation of a German channel's video
    _seed("ok1", ENGLISH, "c-en")           # correct
    assert len(rs.load_active_claims()) == 3
    tally = rb.run_reject_foreign(channel_languages=LANGS)
    assert tally == {"checked": 3, "rejected": 2, "videos": ["ar1", "en1"]}
    state = rs.load_state()
    assert state["videos"]["ar1"]["research_status"] == "transcript_rejected"
    assert state["videos"]["ar1"]["failure_reason"] == "transcript_language_mismatch:script_mismatch:arabic"
    assert state["videos"]["en1"]["failure_reason"] == "transcript_language_mismatch:language_mismatch:en"
    assert state["videos"]["ar1"]["superseded_run_keys"] == ["rk1"] and state["videos"]["ar1"]["active_run_key"] is None
    assert state["videos"]["ar1"]["refetch_languages"] == ["en"] and state["videos"]["en1"]["refetch_languages"] == ["de"]
    assert state["videos"]["ok1"]["research_status"] == "complete"
    # The claims leave the canonical set; the correct video's stay.
    assert [c["video_id"] for c in rs.load_active_claims(state)] == ["ok1"]
    # The capture is retired, not deleted, and the index says why.
    assert ts.load_transcript("ar1") is None and ts.load_transcript("ok1")
    index = [json.loads(l) for l in open(ts.TRANSCRIPT_INDEX, encoding="utf-8")]
    rejected = [r for r in index if r.get("rejected")]
    assert {r["video_id"] for r in rejected} == {"ar1", "en1"}
    assert rejected[0]["path"].endswith(".json.gz") and "rejected-" in rejected[0]["path"]
    # Idempotent.
    assert rb.run_reject_foreign(channel_languages=LANGS) == {"checked": 1, "rejected": 0, "videos": []}
    # --retry never touches a rejected video (no stored text to retry from).
    assert "ar1" not in rs.retry_candidates(rs.load_state())


def test_channel_id_is_recovered_from_the_gate_log_when_the_entry_has_none():
    ts.store_transcript({**_details("x1"), "channel_id": None}, ENGLISH, "supadata", "ok")
    state = rs.load_state()
    rs.update(state, "x1", research_status="complete", channel_id=None, channel_name="Chan")
    rs.record_gate_outcome("c-de", {"video_id": "other", "channel_name": "Chan"}, "included")
    assert rb.languages_for(state["videos"]["x1"], None, LANGS) == ["de"]
    assert rb.run_reject_foreign(state, channel_languages=LANGS)["rejected"] == 1


def test_refetch_recaptures_in_the_channel_language_and_re_extracts(monkeypatch, tmp_path):
    monkeypatch.setattr(rb, "SIGNALS_FILE", str(tmp_path / "signals.jsonl"))
    monkeypatch.setattr(rb.research_budget, "remaining_requests", lambda providers=None: None)
    _seed("ar1", ARABIC, "c-en")
    rb.run_reject_foreign(channel_languages=LANGS)
    fetched = []

    def fetcher(url, languages=None):
        fetched.append((url, languages))
        return {"transcript": "Nvidia will hit $200 this year. " + ENGLISH, "reason": "gemini_ok",
                "budget_exhausted": False, "language": "en"}

    def extractor(nt, ctx):
        assert "Nvidia" in nt.text and ctx["transcript_language"] == "en"
        claims = [{"claim_id": "clm-new", "video_id": "ar1", "schema_version": "2", "claim_type": "forecast",
                   "subject_mention": "Nvidia", "run_key": ctx["run_key"]}]
        return {"status": "complete", "claims": claims, "signals": None, "warnings": [], "coverage_status": "full",
                "run_key": ctx["run_key"], "extraction_model": "m", "telemetry": {}}

    assert rb.run_refetch(fetcher=fetcher, extractor=extractor) == {"complete": 1}
    assert fetched == [("https://www.youtube.com/watch?v=ar1", ["en"])]
    state = rs.load_state()
    entry = state["videos"]["ar1"]
    assert entry["research_status"] == "complete" and entry["transcript_source"] == "gemini_video"
    assert entry["transcript_language"] == "en" and entry["active_run_key"] and entry["superseded_run_keys"] == ["rk1"]
    assert ts.load_transcript("ar1")["raw_transcript"].startswith("Nvidia")
    active = rs.load_active_claims(state)
    assert [c["claim_id"] for c in active] == ["clm-new"]
    # Nothing left to re-fetch.
    assert rb.run_refetch(fetcher=fetcher, extractor=extractor) == {}


def test_refetch_stops_on_a_spent_budget_and_counts_real_failures(monkeypatch):
    _seed("ar1", ARABIC, "c-en")
    _seed("ar2", ARABIC, "c-en")
    rb.run_reject_foreign(channel_languages=LANGS)
    calls = []
    monkeypatch.setattr(rb.research_budget, "remaining_requests", lambda providers=None: None)
    spent = rb.run_refetch(fetcher=lambda url, languages=None: calls.append(url) or {
        "transcript": "", "reason": "no_credits", "budget_exhausted": True})
    assert spent == {"budget_deferred": 1} and len(calls) == 1
    state = rs.load_state()
    assert state["videos"]["ar1"]["research_status"] == "transcript_rejected"
    assert state["videos"]["ar1"]["attempt_count"] == 0          # a spent budget is not the video's fault
    failed = rb.run_refetch(fetcher=lambda url, languages=None: {
        "transcript": "", "reason": "language_mismatch", "budget_exhausted": False})
    assert failed == {"refetch_failed": 2}
    state = rs.load_state()
    assert state["videos"]["ar1"]["attempt_count"] == 1
    assert state["videos"]["ar1"]["failure_reason"] == "refetch_failed:language_mismatch"
    assert state["videos"]["ar1"]["research_status"] == "transcript_rejected"   # still owed a capture
