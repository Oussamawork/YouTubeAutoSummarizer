"""
Deterministic transcript language detection and verification for the
languages the channel list actually carries: English and German (HKCM,
Phantom by HKCM). No model, no network — a script census plus a stop-word
ratio over the tokens, so the result is reproducible and cheap enough to
run on every transcript.

Two consumers:

* the suspicious-empty guard (claims.suspicious_empty_check), whose
  vocabulary is language-specific: a transcript in a language the guard has
  no rule set for must not have an empty extraction certified as "no
  claims". Anything that is not clearly English or German is therefore
  reported as "unknown", never guessed;
* transcript acceptance (`verify_language`): a caption track is only used
  when it is in a language the channel speaks. YouTube carries translated
  and auto-dubbed tracks beside the original, and a provider asked for "the
  transcript" may hand back any of them — in September 2026 Supadata served
  Arabic translations of English videos and English translations of German
  ones, and the pipeline summarized, stored and mined them. The channel's
  language (channel_ids.txt `lang=`) or the pipeline-wide default
  (`TRANSCRIPT_LANGUAGES`) is what a transcript must be in; anything else
  is rejected before it costs a model request or enters the research data.
"""
import re

from helpers import env_str_list

SUPPORTED_LANGUAGES = ("en", "de")
# Languages a transcript may be in when the channel line says nothing:
# every language the guard has a rule set for. `TRANSCRIPT_LANGUAGES`
# narrows or widens it pipeline-wide; a per-channel `lang=` narrows it to one.
DEFAULT_LANGUAGES = tuple(env_str_list("TRANSCRIPT_LANGUAGES", SUPPORTED_LANGUAGES))
LANGUAGE_NAMES = {"en": "English", "de": "German"}
# Writing systems this detector can tell apart. Every supported language is
# written in Latin script, so a transcript in another script cannot be one
# of them whatever its stop-word ratio says.
SCRIPT_RANGES = (
    ("latin", (0x0041, 0x024F)), ("latin", (0x1E00, 0x1EFF)),
    ("greek", (0x0370, 0x03FF)), ("cyrillic", (0x0400, 0x04FF)),
    ("hebrew", (0x0590, 0x05FF)), ("arabic", (0x0600, 0x06FF)), ("arabic", (0x0750, 0x077F)),
    ("devanagari", (0x0900, 0x097F)), ("thai", (0x0E00, 0x0E7F)),
    ("cjk", (0x3040, 0x30FF)), ("cjk", (0x3400, 0x4DBF)), ("cjk", (0x4E00, 0x9FFF)),
    ("hangul", (0xAC00, 0xD7AF)),
)
LANGUAGE_SCRIPTS = {"en": "latin", "de": "latin"}
# A Latin-script transcript this long that is neither clearly English nor
# clearly German is some other language (a French or Spanish track, say),
# not a short clip the ratio cannot judge.
MIN_TOKENS_TO_REJECT_UNKNOWN = 200

# Function words that occur in one language and (as spelled) not in the
# other. Tokens shared by both — "in", "so", "was", "man", "will", "die"
# (rare in English but real), "also" — are deliberately left out of both sets.
_STOPWORDS = {
    "en": {"the", "and", "is", "are", "to", "of", "that", "this", "it", "you", "for", "with", "have",
           "not", "be", "on", "we", "they", "going", "what", "but", "just", "about", "there", "here",
           "because", "think", "would", "could", "should", "from", "your", "these", "those", "which",
           "when", "than", "them", "been", "into", "more", "very", "right", "now", "really", "some"},
    "de": {"und", "der", "das", "ist", "nicht", "ich", "wir", "auch", "mit", "sich", "auf", "ein", "eine",
           "dass", "für", "wird", "werden", "aber", "noch", "hier", "jetzt", "sind", "haben", "oder",
           "wenn", "kann", "dann", "schon", "mal", "sehr", "dem", "den", "des", "zu", "bei", "nach",
           "über", "wie", "uns", "euch", "ihr", "natürlich", "aktie", "aktien", "muss", "können",
           "wieder", "ganz", "also", "diese", "dieser", "dieses", "einfach", "jahr", "prozent"},
}
# "also" is English too, but as a German discourse marker it is far more
# frequent per token; it stays in the German set only.
_TOKEN_RE = re.compile(r"[a-zäöüß]+", re.IGNORECASE)
MIN_TOKENS = 12
# Real transcripts score 0.30–0.37 of their tokens in their language's set
# with 34–54 distinct stop words (29 stored English and German transcripts,
# 2026-09); a French track hit the old 0.04 floor on the single shared word
# "des". The floor and the distinct-word minimum (scaled down for short
# clips) keep one coincidental word from naming a language.
MIN_RATIO = 0.10
MIN_DISTINCT = 8
MIN_MARGIN = 1.5


def script_of(text):
    """
    {"script": dominant writing system of the letters in `text` ("latin",
    "arabic", "cyrillic", "cjk", ... or "unknown" when there are no letters),
    "share": its fraction of all letters, "letters": n}. Deterministic.
    """
    counts = {}
    letters = 0
    for ch in text or "":
        if not ch.isalpha():
            continue
        letters += 1
        code = ord(ch)
        name = "other"
        for script, (lo, hi) in SCRIPT_RANGES:
            if lo <= code <= hi:
                name = script
                break
        counts[name] = counts.get(name, 0) + 1
    if not letters:
        return {"script": "unknown", "share": 0.0, "letters": 0}
    script, n = max(counts.items(), key=lambda kv: kv[1])
    return {"script": script, "share": round(n / letters, 3), "letters": letters}


def detect_language(text):
    """
    {"language": "en" | "de" | "unknown", "confidence": 0..1, "method":
    "stopword_ratio", "tokens": n, "script": dominant script}. Deterministic;
    never raises. A transcript whose letters are not mostly Latin script is
    "unknown" outright: the stop-word sets cannot describe it.
    """
    tokens = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    script = script_of(text)
    out = {"language": "unknown", "confidence": 0.0, "method": "stopword_ratio", "tokens": len(tokens),
           "script": script["script"]}
    if len(tokens) < MIN_TOKENS or script["script"] != "latin" or script["share"] < 0.5:
        return out
    ratios = {lang: sum(1 for t in tokens if t in words) / len(tokens) for lang, words in _STOPWORDS.items()}
    best, runner = sorted(ratios.items(), key=lambda kv: -kv[1])[:2]
    distinct = len({t for t in tokens if t in _STOPWORDS[best[0]]})
    if best[1] < MIN_RATIO or distinct < min(MIN_DISTINCT, len(tokens) // 8) \
            or (runner[1] > 0 and best[1] < MIN_MARGIN * runner[1]):
        return out
    out["language"] = best[0]
    out["confidence"] = round(min(1.0, best[1] / 0.15), 2)
    return out


def language_of(text, declared=None):
    """The language to run language-specific rules under: the declared
    metadata when it names a supported language, else detection."""
    declared = normalize_code(declared)
    if declared in SUPPORTED_LANGUAGES:
        return declared
    return detect_language(text)["language"]


def normalize_code(code):
    """ISO 639-1 code from a provider's language tag ("en-US", "de_DE",
    "EN") or "" when there is none."""
    code = (code or "").strip().lower().replace("_", "-")
    return code.split("-", 1)[0] if code else ""


def normalize_languages(languages):
    """A clean, de-duplicated tuple of language codes; DEFAULT_LANGUAGES
    when nothing usable is given."""
    out = []
    for code in (languages or ()):
        code = normalize_code(code)
        if code and code not in out:
            out.append(code)
    return tuple(out) if out else DEFAULT_LANGUAGES


def language_names(languages):
    return ", ".join(LANGUAGE_NAMES.get(code, code) for code in normalize_languages(languages))


def verify_language(text, accepted=None, reported=None):
    """
    Whether a transcript is in one of the `accepted` languages. Returns
      {"ok": bool, "language": the language it is in (or "unknown"),
       "reason": why it was rejected (None when ok), "script": ...,
       "reported": the provider's normalized tag, "detected": detection}.

    Rules, in order — every one is a hard reason to refuse a track:
      1. `reported` (the provider's own language tag) names a language
         outside `accepted`  → language_mismatch:<reported>
      2. the letters are not in the script an accepted language is written
         in (Arabic script for an English channel) → script_mismatch:<script>
      3. detection says a supported language outside `accepted` (an English
         track for a German channel) → language_mismatch:<detected>
      4. a long Latin-script transcript detection cannot place, with no
         acceptable provider tag to vouch for it → language_unrecognized
    A short transcript detection cannot judge is accepted on the provider's
    tag, or on trust when there is none: rejecting it would lose real
    clips, and the claims guard still treats its language as unknown.
    Never raises.
    """
    accepted = normalize_languages(accepted)
    reported_code = normalize_code(reported)
    detected = detect_language(text)
    out = {"ok": True, "language": None, "reason": None, "script": detected["script"],
           "reported": reported_code or None, "detected": detected["language"], "accepted": list(accepted)}
    if reported_code and reported_code not in accepted:
        out.update(ok=False, language=reported_code, reason=f"language_mismatch:{reported_code}")
        return out
    scripts = {LANGUAGE_SCRIPTS.get(code, "latin") for code in accepted}
    if detected["script"] not in scripts and detected["script"] != "unknown":
        out.update(ok=False, language="unknown", reason=f"script_mismatch:{detected['script']}")
        return out
    if detected["language"] != "unknown":
        if detected["language"] not in accepted:
            out.update(ok=False, language=detected["language"],
                       reason=f"language_mismatch:{detected['language']}")
            return out
        out["language"] = detected["language"]
        return out
    if reported_code:
        out["language"] = reported_code
        return out
    if detected["tokens"] >= MIN_TOKENS_TO_REJECT_UNKNOWN:
        out.update(ok=False, language="unknown", reason="language_unrecognized")
        return out
    out["language"] = "unknown"
    return out
