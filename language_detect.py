"""
Deterministic transcript language detection for the languages the channel
list actually carries: English and German (HKCM, Phantom by HKCM). No model,
no network — a stop-word ratio over the tokens, so the result is
reproducible and cheap enough to run on every transcript.

The answer feeds the suspicious-empty guard (claims.suspicious_empty_check),
whose vocabulary is language-specific: a transcript in a language the guard
has no rule set for must not have an empty extraction certified as
"no claims". Anything that is not clearly English or German is therefore
reported as "unknown", never guessed.
"""
import re

SUPPORTED_LANGUAGES = ("en", "de")

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
MIN_RATIO = 0.04
MIN_MARGIN = 1.5


def detect_language(text):
    """
    {"language": "en" | "de" | "unknown", "confidence": 0..1, "method":
    "stopword_ratio", "tokens": n}. Deterministic; never raises.
    """
    tokens = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    out = {"language": "unknown", "confidence": 0.0, "method": "stopword_ratio", "tokens": len(tokens)}
    if len(tokens) < MIN_TOKENS:
        return out
    ratios = {lang: sum(1 for t in tokens if t in words) / len(tokens) for lang, words in _STOPWORDS.items()}
    best, runner = sorted(ratios.items(), key=lambda kv: -kv[1])[:2]
    if best[1] < MIN_RATIO or (runner[1] > 0 and best[1] < MIN_MARGIN * runner[1]):
        return out
    out["language"] = best[0]
    out["confidence"] = round(min(1.0, best[1] / 0.15), 2)
    return out


def language_of(text, declared=None):
    """The language to run language-specific rules under: the declared
    metadata when it names a supported language, else detection."""
    declared = (declared or "").strip().lower()[:2]
    if declared in SUPPORTED_LANGUAGES:
        return declared
    return detect_language(text)["language"]
