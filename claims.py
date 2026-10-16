"""
Atomic, evidence-backed claims: the canonical research unit.

One claim is one independent statement — one asset, one metric, one
direction, one target, one horizon, one condition. "Bearish on Nvidia for
three months but bullish over five years" is two claims, never stance=mixed.
The model proposes candidates; THIS module decides what enters the dataset:
it locates each claim's verbatim evidence in the normalized transcript,
checks that every number the claim carries appears in that evidence, lets
the curated tables decide the entity (a model ticker counts only when the
evidence or the title carries it as a ticker), tells markets, baskets and
metrics apart from single instruments, turns relative horizons into dates
only under documented rules, and computes testability deterministically.
Anything it cannot verify stays in the record with review_required=True and
out of headline analytics.

The compatibility view (`claims_to_legacy_signals`) reduces validated claims
back into the shape data/signals.jsonl consumers read, and marks every place
information was reduced.
"""
import hashlib
import json
import re
from datetime import date, datetime, timedelta, timezone

from log import log_info, log_warn
from signals_data import (ASSET_ALIASES, COMMON_WORD_NAMES, FORMER_TICKERS, LISTED_TICKER_KEYS,
                          NAME_SUFFIXES as _NAME_SUFFIXES, NOT_A_TICKER, TICKER_ALIASES, learned_tickers)

SCHEMA_VERSION = "2"
EXTRACTION_PROMPT_VERSION = "2"
# The entity-resolution rules' own version, stored on every claim. Bumped
# whenever a resolution rule or table below changes: canonical_claims.reresolve
# re-runs resolution at load time on rows resolved under an older version, so
# stored claims follow the current rules without a re-extraction. It is
# deliberately NOT part of the run key (research_state.run_key): a resolution
# change must never mark stored runs stale.
RESOLUTION_VERSION = "2"

# --- Enumerations -----------------------------------------------------------

ATTRIBUTION_TYPES = {
    "speaker_personal_view", "guest_personal_view", "speaker_reporting_facts",
    "speaker_reporting_news", "speaker_quoting_third_party",
    "speaker_describing_market_consensus", "interviewer_question",
    "hypothetical_example", "retrospective_claim", "sarcasm_or_humor", "unclear",
}
# Attributions that count as the source's own current view in analytics.
OWN_VIEW_ATTRIBUTIONS = {"speaker_personal_view", "guest_personal_view"}
THIRD_PARTY_ATTRIBUTIONS = {"speaker_quoting_third_party", "speaker_describing_market_consensus",
                            "speaker_reporting_news"}
CLAIM_TYPES = {
    "forecast", "price_target", "recommendation", "stance", "valuation_view",
    "fact", "news_report", "third_party_view", "historical_claim",
    "portfolio_disclosure", "question", "hypothetical", "opinion", "risk",
    "catalyst", "other",
}
# Claim types that can carry an investment VIEW (a stance a reader may act
# on). Everything else — a portfolio disclosure, a question, a fact, a news
# report, a third party's view, a retrospective, a hypothetical — is recorded
# but never enters stance counts, net stance, consensus, sentiment, flips or
# forecast scorecards, whatever stance value it carries.
VIEW_CLAIM_TYPES = {"forecast", "price_target", "recommendation", "stance", "valuation_view",
                    "opinion", "risk", "catalyst"}
NON_VIEW_CLAIM_TYPES = CLAIM_TYPES - VIEW_CLAIM_TYPES
# Testability is three-valued: an unconditional forecast is scorable as soon
# as it matures; a conditional one is scorable only once its condition has
# been observed to hold; and a claim missing an instrument, metric, horizon,
# evidence or an objectively observable condition is not testable at all.
TESTABILITY_TYPES = {"unconditional_testable", "conditional_testable", "not_testable"}
CONDITION_STATUSES = {"met", "not_met", "partially_met", "unknown", "not_evaluated"}
# How the claim's entity was pinned down, and where a third-party view's
# speaker stands on it.
ENTITY_RESOLUTION_METHODS = {"explicit_mention", "local_coreference", "title", "curated_mapping",
                             "unresolved"}
HOST_POSITIONS = {"adopted", "rejected", "neutral", "not_applicable"}
SPEAKER_CONFIDENCE = {"high", "medium", "low", "unknown"}
STANCES = {"bullish", "bearish", "neutral", "mixed", "not_applicable", "unclear"}
STANCE_BASIS = {"explicit", "inferred_from_context", "not_applicable"}
ACTIONS = {"buy", "add", "accumulate", "hold", "reduce", "sell", "short", "avoid",
           "watch", "none", "unclear"}
DIRECTIONS = {"increase", "decrease", "remain_flat", "outperform", "underperform",
              "recover", "decline", "volatile", "unspecified"}
TARGET_KINDS = {"absolute_value", "percentage_change", "multiple", "range",
                "direction_only", "qualitative", "none"}
CERTAINTY = {"high", "medium", "low", "not_stated", "unclear"}
PORTFOLIO = {"long", "short", "owns_unspecified", "no_position", "not_stated", "unclear"}
EXTRACTION_CONFIDENCE = {"high", "medium", "low"}
ENTITY_STATUS = {"confirmed", "probable", "ambiguous", "unresolved"}
TICKER_SOURCES = {"spoken", "title", "curated_mapping", "unresolved"}
# What a claim's subject IS: one security or coin; a market (index, sector,
# commodity, macro theme: never a ticker); a basket of several assets; or a
# metric / segment of a company ("gross margin", "the AWS segment"), whose
# company ticker is kept for traceability only.
SUBJECT_KINDS = {"instrument", "market", "basket", "metric"}
# Non-blocking notes on how the model's ticker was treated. None of them is a
# review reason: the curated tables decided, and the note records it.
RESOLUTION_NOTES = {"ticker_spoken_replaced", "ticker_spoken_unverified", "ticker_equals_name_unverified",
                    "crypto_title_coreference", "multi_asset_subject"}
COVERAGE = {"full", "chunked_full", "partial"}
ASSET_TYPES = {"stock", "crypto", "etf", "index", "commodity", "macro", "currency",
               "bond", "private_company", "sector", "other"}

# Spoken names that map to more than one listed security. Never guessed.
AMBIGUOUS_MENTIONS = {"TARGET", "CROWN", "GENERAL", "MATCH", "BLOCK INC CLASS", "ALLY"}

MIN_EVIDENCE_CHARS = 12
# Horizon buckets by resolved length: "three months" (up to 92 days) is short,
# anything within about a year is medium, beyond is long.
SHORT_MAX_DAYS = 95
MEDIUM_MAX_DAYS = 366

# --- Prompt -------------------------------------------------------------------

# Compact field list the model returns per claim. Application code adds ids,
# versions, coverage, timestamps, entity resolution and dates; the model must
# OMIT any field it would set to null or [] (that halves output tokens).
CLAIM_FIELDS_SPEC = (
    '{"speaker": "host|guest|<name if given>", '
    '"speaker_confidence": "high|medium|low|unknown", '
    '"attribution_type": "speaker_personal_view|guest_personal_view|speaker_reporting_facts|'
    'speaker_reporting_news|speaker_quoting_third_party|speaker_describing_market_consensus|'
    'interviewer_question|hypothetical_example|retrospective_claim|sarcasm_or_humor|unclear", '
    '"attributed_person_or_organization": "<who, for third-party views>", '
    '"host_position": "adopted|rejected|neutral <for third-party views: does the speaker adopt it, reject it, or just report it>", '
    '"claim_type": "forecast|price_target|recommendation|stance|valuation_view|fact|news_report|'
    'third_party_view|historical_claim|portfolio_disclosure|question|hypothetical|opinion|risk|catalyst|other", '
    '"is_forward_looking": true|false, '
    '"subject_mention": "<asset/company EXACTLY as spoken>", '
    '"ticker_spoken": "<ONLY if the speaker says the ticker or it is in the title>", '
    '"asset_type": "stock|crypto|etf|index|commodity|macro|currency|bond|private_company|sector|other", '
    '"benchmark_name": "<e.g. S&P 500 when the claim is relative>", '
    '"stance": "bullish|bearish|neutral|mixed|not_applicable|unclear", '
    '"stance_basis": "explicit|inferred_from_context|not_applicable", '
    '"recommendation_action": "buy|add|accumulate|hold|reduce|sell|short|avoid|watch|none|unclear", '
    '"forecast_metric": "price|revenue|margin|earnings|market_cap|rate|other <as stated>", '
    '"forecast_direction": "increase|decrease|remain_flat|outperform|underperform|recover|decline|volatile|unspecified", '
    '"target_kind": "absolute_value|percentage_change|multiple|range|direction_only|qualitative|none", '
    '"target_value": <number>, "target_low": <number>, "target_high": <number>, '
    '"target_unit": "USD|percent|x|...", "currency": "USD|EUR|...", '
    '"baseline_value": <number>, "expected_change_value": <number>, "expected_change_unit": "percent|USD|...", '
    '"horizon_original": "<time wording EXACTLY as spoken>", '
    '"condition": "<the if/unless clause, verbatim-ish>", "trigger": "<event that would trigger it>", '
    '"certainty_original": "<the hedge word(s) used: will, expect, could, ...>", '
    '"certainty_level": "high|medium|low|not_stated|unclear", '
    '"reasoning_summary": "<why, in one short sentence>", '
    '"catalysts": ["..."], "risks": ["..."], "assumptions": ["..."], "counterarguments": ["..."], '
    '"portfolio_disclosure": "long|short|owns_unspecified|no_position|not_stated|unclear", '
    '"evidence_text": "<SHORT VERBATIM excerpt from the transcript that contains the claim, its numbers and its condition>", '
    '"extraction_confidence": "high|medium|low", '
    '"review_reasons": ["<why a human should check this, if anything>"]}'
)

CLAIM_RULES = (
    "You are performing EXTRACTION for a research dataset, not giving investment "
    "advice. Use ONLY the transcript and the metadata supplied; never use outside "
    "knowledge to fill a field. Return ONE record per ATOMIC claim: split a "
    "statement whenever asset, metric, direction, target, recommendation, "
    "horizon, condition, catalyst or risk differs. Rules:\n"
    "- Preserve conditions, negations, hedges (certainty_original), targets and "
    "the speaker's exact time wording (horizon_original). Never convert a "
    "possibility into a certainty.\n"
    "- A question is not a forecast (attribution_type=interviewer_question, "
    "claim_type=question, is_forward_looking=false).\n"
    "- A reported analyst/bank/consensus target is a third-party view "
    "(speaker_quoting_third_party / speaker_describing_market_consensus) with "
    "host_position=neutral, unless the speaker explicitly adopts it "
    "(host_position=adopted) or rejects it (host_position=rejected).\n"
    "- \"Last year I said X\" is retrospective_claim / historical_claim, not a "
    "new forecast.\n"
    "- Owning a stock is portfolio_disclosure with stance=not_applicable and "
    "recommendation_action=none — it is not a view; praise without an explicit "
    "buy/sell/hold instruction is opinion with recommendation_action=none. Only "
    "an explicit instruction gets buy/sell/hold/etc.\n"
    "- When the speaker says \"the stock\", \"it\" or \"the company\", keep "
    "subject_mention as spoken; the surrounding transcript decides which "
    "company it is, never outside knowledge.\n"
    "- Keep short-term and long-term views as separate claims. Do not assign a "
    "market-wide statement to every company mentioned elsewhere.\n"
    "- ticker_spoken only when the speaker SAYS the ticker or it appears in the "
    "video title; otherwise omit it. Never invent a ticker, a number, a date, a "
    "company name, a speaker identity or a timestamp.\n"
    "- evidence_text must be copied verbatim from the transcript (one or two "
    "sentences) and must contain the numbers and condition you extracted.\n"
    "- Never silently drop an uncertain claim: include it with "
    "extraction_confidence=low and a review_reasons entry.\n"
    "- Omit any field you would set to null or []. Valid JSON only."
)

CLAIM_EXAMPLES = (
    "Examples (transcript fragment -> claims, other fields omitted for brevity):\n"
    "1. \"I expect Nvidia to fall over the next three months, but I remain bullish over five years.\" -> "
    "two claims: {claim_type:forecast, subject_mention:Nvidia, stance:bearish, forecast_metric:price, "
    "forecast_direction:decrease, horizon_original:\"over the next three months\", certainty_original:expect, "
    "certainty_level:medium} and {claim_type:stance, subject_mention:Nvidia, stance:bullish, "
    "horizon_original:\"over five years\", certainty_level:medium}.\n"
    "2. \"Could Apple fall 30 percent from here?\" -> {claim_type:question, attribution_type:interviewer_question, "
    "subject_mention:Apple, is_forward_looking:false, stance:not_applicable}.\n"
    "3. \"Goldman expects the stock to reach $200.\" -> {claim_type:third_party_view, "
    "attribution_type:speaker_quoting_third_party, attributed_person_or_organization:Goldman, "
    "host_position:neutral, subject_mention:\"the stock\", target_kind:absolute_value, target_value:200, "
    "currency:USD, stance:not_applicable}.\n"
    "3b. \"I own Tesla.\" -> {claim_type:portfolio_disclosure, subject_mention:Tesla, "
    "portfolio_disclosure:owns_unspecified, stance:not_applicable, recommendation_action:none, "
    "is_forward_looking:false}.\n"
    "4. \"Last year I said Bitcoin would double, and it did.\" -> {claim_type:historical_claim, "
    "attribution_type:retrospective_claim, subject_mention:Bitcoin, is_forward_looking:false}.\n"
    "5. \"Micron is the cheapest memory name right now.\" -> {claim_type:valuation_view, subject_mention:Micron, "
    "stance:bullish, stance_basis:inferred_from_context, recommendation_action:none} (no ticker: none was spoken).\n"
    "6. \"I'd buy Palantir under $20.\" -> {claim_type:recommendation, subject_mention:Palantir, "
    "recommendation_action:buy, stance:bullish, condition:\"under $20\", target_kind:none}.\n"
    "7. \"Costco is a wonderful business.\" -> {claim_type:opinion, subject_mention:Costco, stance:bullish, "
    "stance_basis:inferred_from_context, recommendation_action:none}.\n"
    "8. \"If the Fed cuts in September, small caps should rally into year end.\" -> {claim_type:forecast, "
    "subject_mention:small caps, asset_type:sector, forecast_direction:increase, condition:\"if the Fed cuts in "
    "September\", horizon_original:\"into year end\", certainty_original:should, certainty_level:medium}.\n"
    "9. \"I see Bitcoin between 150 and 180 thousand next year.\" -> {claim_type:price_target, "
    "subject_mention:Bitcoin, target_kind:range, target_low:150000, target_high:180000, currency:USD, "
    "horizon_original:\"next year\"}.\n"
    "10. \"Revenue should rise but margins may fall.\" -> two claims: {forecast_metric:revenue, "
    "forecast_direction:increase, certainty_original:should, certainty_level:medium} and "
    "{forecast_metric:margin, forecast_direction:decrease, certainty_original:may, certainty_level:low}."
)

CLAIMS_SYSTEM_PROMPT = (
    CLAIM_RULES + "\n\n" + CLAIM_EXAMPLES + "\n\nEach claim record has this shape:\n"
    + CLAIM_FIELDS_SPEC + "\n\nRespond with ONE JSON object and nothing else: "
    '{"claims": [<records>], "extraction_metadata": {"warnings": ["<anything you could not resolve>"]}}. '
    'If the transcript has no market-relevant content, return {"claims": [], '
    '"extraction_metadata": {"warnings": []}}.'
)

CLAIMS_USER_TEMPLATE = (
    "Channel: {channel_name}\nVideo title: {video_title}\nPublished: {published_at}\n\n"
    "Transcript{part}:\n\n{transcript}"
)

# The combined envelope: summary rules first (the caller supplies them), then
# the claims contract, so one request yields both products.
COMBINED_ENVELOPE = (
    "\n\n=== OUTPUT ENVELOPE ===\n"
    "Everything above describes the TEXT that belongs in the \"summary\" field: "
    "still plain text with \"• \" bullets and the asset roster, no markdown, no "
    "preamble. The RESPONSE as a whole is ONE JSON object and nothing else:\n"
    '{"summary": "<the summary described above, as one JSON string using \\n for '
    'line breaks>", "claims": [<claim records>], "extraction_metadata": '
    '{"warnings": ["..."]}}\n'
    "If the transcript is unsummarizable, return exactly "
    '{"summary": "INSUFFICIENT_TRANSCRIPT", "claims": [], "extraction_metadata": {"warnings": []}}.\n\n'
    "The \"claims\" array is a SECOND, INDEPENDENT product built from the whole "
    "TRANSCRIPT (not from the summary), under these rules:\n"
    + CLAIM_RULES + "\n\n" + CLAIM_EXAMPLES + "\n\nEach claim record has this shape:\n" + CLAIM_FIELDS_SPEC
)


# --- Parsing ------------------------------------------------------------------


def parse_json_object(text):
    """Outermost JSON object in a model response (fences/preambles tolerated),
    or None."""
    if not text or not isinstance(text, str):
        return None
    stripped = text.strip()
    m = re.match(r"^```[a-zA-Z]*\s*\n?(.*?)\n?```\s*$", stripped, re.DOTALL)
    if m:
        stripped = m.group(1)
    try:
        data = json.loads(stripped)
    except (ValueError, TypeError):
        start, end = stripped.find("{"), stripped.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(stripped[start:end + 1])
        except (ValueError, TypeError):
            return None
    return data if isinstance(data, dict) else None


def raw_claims_from(data):
    """
    The claims array from an envelope, or None when it is malformed. [] is a
    valid answer (the model found nothing); a missing or non-list value is a
    failure and must never be read as "no claims".
    """
    if not isinstance(data, dict):
        return None
    raw = data.get("claims")
    if not isinstance(raw, list):
        return None
    if any(not isinstance(c, dict) for c in raw):
        return None
    return raw


# --- Evidence -----------------------------------------------------------------

_FOLD_MAP = str.maketrans({
    "’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-",
    " ": " ",
})


def _fold(text):
    """Comparison form: quotes/dashes unified, lowercase, whitespace collapsed,
    punctuation (except inside numbers) dropped. Returns (folded, index map)."""
    out, idx = [], []
    prev_space = True
    text = text.translate(_FOLD_MAP)
    for i, ch in enumerate(text):
        lower = ch.lower()
        if lower.isalnum() or lower in "$%€£":
            out.append(lower)
            idx.append(i)
            prev_space = False
        elif lower in ".,:" and 0 < i < len(text) - 1 and text[i - 1].isdigit() and text[i + 1].isdigit():
            out.append(lower)
            idx.append(i)
            prev_space = False
        elif not prev_space:
            out.append(" ")
            idx.append(i)
            prev_space = True
    while out and out[-1] == " ":
        out.pop(), idx.pop()
    return "".join(out), idx


def locate_evidence(evidence, text):
    """
    (start, end) offsets of `evidence` inside `text` under a controlled,
    normalization-aware match (case, quotes, dashes, whitespace and
    punctuation differences are tolerated; words are not). None when the
    excerpt is not there — a paraphrase never passes.
    """
    if not evidence or not text:
        return None
    ev_f, _ = _fold(evidence.strip())
    if len(ev_f) < MIN_EVIDENCE_CHARS:
        return None
    tx_f, idx = _fold(text)
    pos = tx_f.find(ev_f)
    if pos == -1:
        return None
    return idx[pos], idx[pos + len(ev_f) - 1] + 1


_NUMBER_RE = re.compile(
    r"(?<![\w.])(?:\$|€|£)?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s*"
    r"(thousand|million|billion|trillion|percent|k|m|b|t|%)?(?![a-z])", re.IGNORECASE)
_MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "b": 1e9, "billion": 1e9,
         "t": 1e12, "trillion": 1e12}


_RANGE_JOIN = re.compile(r"^\s*(?:and|to|or|-|–|—|through)\s*$", re.IGNORECASE)


def numbers_in(text):
    """Numeric values a human would read in `text`, including k/m/b and
    'thousand'-style multipliers and both the bare and scaled readings. In a
    range such as "150 and 180 thousand" or "150-180k" the multiplier of the
    upper bound is applied to the lower bound too."""
    text = text or ""
    matches = list(_NUMBER_RE.finditer(text))
    found = set()
    for i, m in enumerate(matches):
        whole = m.group(1).replace(",", "")
        frac = m.group(2)
        try:
            value = float(f"{whole}.{frac}" if frac else whole)
        except ValueError:
            continue
        found.add(value)
        suffix = (m.group(3) or "").lower()
        if suffix in _MULT:
            found.add(value * _MULT[suffix])
        elif i + 1 < len(matches):
            nxt = matches[i + 1]
            nxt_suffix = (nxt.group(3) or "").lower()
            if nxt_suffix in _MULT and _RANGE_JOIN.match(text[m.end():nxt.start()] or ""):
                found.add(value * _MULT[nxt_suffix])
    return found


def number_supported(value, evidence):
    if value is None:
        return True
    nums = numbers_in(evidence)
    return any(abs(float(value) - n) <= max(1e-9, abs(n) * 1e-9) for n in nums)


# --- Helpers ------------------------------------------------------------------

_QUESTION_START = re.compile(
    r"^\s*(could|can|will|would|should|is|are|do|does|did|what|why|how|when|where|who|which)\b",
    re.IGNORECASE)
_RETRO_RE = re.compile(
    r"\b(last (year|month|week|time)|back in|i (said|told|predicted|called|warned)|"
    r"as i (said|predicted)|i was (right|wrong)|in 20[0-2]\d i)\b", re.IGNORECASE)
_THIRD_PARTY_RE = re.compile(
    r"\b(analysts?|wall street|consensus|goldman|morgan stanley|jp ?morgan|bank of america|"
    r"citi|ubs|barclays|according to|reports? (say|said)|the street expects?)\b", re.IGNORECASE)
_ADOPT_RE = re.compile(
    r"\b(i agree|i think so too|i share|my target|i also (think|expect|see)|"
    r"that'?s my (target|view|call|number)( too| as well)?|i'?m with (them|him|her)|"
    r"i think (they|he|she)('re| are|'s| is) right|same (view|call) here|"
    r"i('d| would) (go|say) (the )?same|and i believe (them|that|it))\b", re.IGNORECASE)
_REJECT_RE = re.compile(
    r"\b(not my (call|view|target|number)|(their|his|her) call,? not mine|"
    r"that('s| is) (their|his|her) (call|view|number)|i disagree|i don'?t (agree|buy (that|it|this))|"
    r"i do not agree|i'?m not (convinced|buying (that|it|this))|i think (they|he|she)('re| are|'s| is) wrong|"
    r"(way |far )?too (bullish|bearish|optimistic|pessimistic|aggressive)|i wouldn'?t (go|say) that)\b",
    re.IGNORECASE)
_RECOMMEND_RE = re.compile(
    r"\b(buy|buying|bought|sell|selling|sold|accumulat\w*|short\w*|avoid\w*|trim\w*|reduc\w*|"
    r"add(?:ing)? to|hold(?:ing)? (?:on|it|them|this|the)|recommend\w*|you should|i would|i'd)\b",
    re.IGNORECASE)
_OWNERSHIP_RE = re.compile(
    r"\b(i own|we own|i hold|we hold|i'?m holding|i am holding|my position|in my portfolio|"
    r"i have a position|i'?m long|i am long|we'?re long|i'?m short|i am short|we'?re short|"
    r"i bought|i sold|i don'?t own|i do not own|i have no position|no position in)\b", re.IGNORECASE)
_LONG_RE = re.compile(r"\b(i'?m long|i am long|we'?re long|we are long|long (position|the stock|shares))\b",
                      re.IGNORECASE)
_SHORT_RE = re.compile(r"\b(i'?m short|i am short|we'?re short|we are short|short (position|the stock))\b",
                       re.IGNORECASE)
_NO_POSITION_RE = re.compile(r"\b(i don'?t own|i do not own|i have no position|no position in|"
                             r"i sold (all|out|everything|my (whole )?position))\b", re.IGNORECASE)
# Generic references whose referent lives in the surrounding transcript.
_GENERIC_SUBJECT = re.compile(
    r"^(it|its|it'?s|they|them|their|this|that|the (stock|company|name|shares|share price|coin|token|etf|"
    r"business|firm)|this (stock|company|name|one|coin|token|business)|that (stock|company|name|one)|"
    r"(the )?shares|the price)$", re.IGNORECASE)
# More references of the same kind, including the German channels' ("die
# Aktie", "das Unternehmen"). Bare "stocks" is not one: it means the market.
_GENERIC_EXTRA = re.compile(
    r"^(these|those|these shares|those shares|this kind of business|business|stock|"
    r"(die|diese) aktie|(der|den|dem) kurs|(das|dieses) unternehmen|(die|diese) firma|"
    r"(der|dieser|den) (coin|token)|es)$", re.IGNORECASE)
# How far back a generic reference may look for its subject: the sentences
# just before the evidence, inside the same segment.
COREFERENCE_SENTENCES = 2
COREFERENCE_MAX_CHARS = 400
# A condition is objectively observable when it names something a data
# series or a scheduled release settles. Each rule records the source that
# would settle it. Anything else stays a subjective condition — the claim is
# recorded as conditional, not testable, with the specific reason.
# ("6 months", "5 Jahre"; a hyphenated "50-day" / "200-week" average is a level)
_NOT_A_SPAN = r"(?!\s*(?:days?|weeks?|months?|quarters?|years?|hours?|minutes?|jahre?n?|monate?n?|wochen?|tage?n?)\b)"
OBSERVABLE_CONDITIONS = (
    (re.compile(r"\b(fed|fomc|federal reserve|central bank|ecb|boe|boj)\b.*\b(cut|cuts|cutting|hike|hikes|"
                r"hiking|raise|raises|lower|lowers|pause|pauses|hold|holds)\b|\brate (cut|hike)s?\b",
                re.IGNORECASE), "policy_rate_decision", "fomc_decisions"),
    (re.compile(r"\b(cpi|inflation|pce|ppi|unemployment|jobs report|payrolls|gdp)\b.*"
                r"\b(comes? in|prints?|above|below|under|over|hot|cool|rises?|falls?|drops?|\d)",
                re.IGNORECASE), "macro_release", "official_statistics_release"),
    (re.compile(r"\b(earnings|revenue|guidance|eps)\b.*\b(beat|beats|miss|misses|above|below|"
                r"raise|raises|cut|cuts|\d)", re.IGNORECASE), "earnings_release", "company_filings"),
    (re.compile(r"\b(approv(al|ed|es)|fda|launch(es|ed)?|ships?|delivers?|merger|acquisition|"
                r"deal closes|ipo|listing|halving|etf approval)\b", re.IGNORECASE),
     "scheduled_event", "public_announcements"),
    # A price level is a number with a comparator — never a span of time:
    # "over the next 6 months" and "hold it for the next 5 years" are not levels.
    (re.compile(r"(\$\s?\d|\d+(\.\d+)?\s?(k|thousand|million|billion|trillion|%|percent)?\b)" + _NOT_A_SPAN + r".*"
                r"\b(above|below|under|over|beyond|past|through|breaks?|holds?|reclaims?|closes?|"
                r"drops? to|falls? to|hits?|reach(es)?)\b|\b(above|below|under|over|beyond|past|"
                r"breaks?|holds?|reclaims?|closes? (above|below)|drops? to|falls? to|hits?|reach(es)?)\b.*"
                r"(\$\s?\d|\d+(\.\d+)?\s?(k|thousand|million|billion|trillion|%|percent)?\b)" + _NOT_A_SPAN,
                re.IGNORECASE), "price_level", "daily_close"),
)


def _norm_name(name):
    return " ".join((name or "").split()).upper()


def _is_generic(subject):
    """True for a reference whose referent lives elsewhere ("the stock",
    "it", "die Aktie"), never for a name."""
    return bool(subject and (_GENERIC_SUBJECT.match(subject) or _GENERIC_EXTRA.match(subject)))


def _s(value):
    return value.strip() if isinstance(value, str) and value.strip() else None


def _enum(value, allowed, default):
    return value if isinstance(value, str) and value in allowed else default


def _num(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _strlist(value):
    if not isinstance(value, list):
        return []
    return [v.strip() for v in value if isinstance(v, str) and v.strip()]


def _parse_date(value):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.date()
    except ValueError:
        return None


def _quarter_end(d):
    q_end_month = ((d.month - 1) // 3 + 1) * 3
    return _month_end(d.year, q_end_month)


def _month_end(year, month):
    if month == 12:
        return date(year, 12, 31)
    return date(year, month + 1, 1) - timedelta(days=1)


def _add_months(d, months):
    index = d.year * 12 + d.month - 1 + months
    year, month = index // 12, index % 12 + 1
    day = min(d.day, _month_end(year, month).day)
    return date(year, month, day)


_WORD_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12, "eighteen": 18,
             "twenty": 20, "couple of": 2, "few": 3, "several": 3}
VAGUE_HORIZONS = re.compile(
    r"^(soon|eventually|long[- ]?term|longer[- ]?term|short[- ]?term|near[- ]?term|"
    r"medium[- ]?term|in the future|over time|someday|one day|down the road|"
    r"in the coming (weeks|months|years)|going forward|at some point|"
    r"in the long run|in the (short|near) run)$", re.IGNORECASE)


def resolve_horizon(horizon_original, published):
    """
    (start_date, end_date, bucket, issue) from the speaker's wording and the
    publication date. Dates are set ONLY when the wording is precise enough
    under a documented rule; vague wording keeps its bucket (when implied)
    and gets no dates. Rules (relative to the publication date P):
      "this year" / "by (the end of) (this|the) year" / "by year end" -> Dec 31 of P.year
      "next year"                              -> Dec 31 of P.year+1
      "by/in/before <YYYY>"                    -> Dec 31 of YYYY (if >= P.year)
      "end of the decade"                      -> Dec 31 of the decade's last year
      "this quarter" / "next quarter"          -> end of that calendar quarter
      "Q<n> (<YYYY>)"                          -> end of that quarter
      "(over|in|within) the next N days/weeks/months/years", "N-year", "N months"
                                               -> P + N units
      vague words (soon, eventually, long term, over time, in the future, ...)
                                               -> no dates, issue=vague_horizon
    Buckets: short <= 95 days, medium <= 366 days, long beyond; wording alone
    can set a bucket (short/near term -> short; long term -> long).
    """
    text = (horizon_original or "").strip().lower()
    if not text:
        return None, None, "unspecified", "missing_horizon"
    if published is None:
        bucket = _bucket_from_wording(text)
        return None, None, bucket, "missing_publication_date"
    P = published
    end = None
    if VAGUE_HORIZONS.match(text):
        return None, None, _bucket_from_wording(text), "vague_horizon"
    if re.search(r"\b(this year|by (the )?end of (this|the) year|by year[- ]?end|into year[- ]?end|"
                 r"before (the )?end of (this|the) year|by (the )?end of the year|year[- ]end)\b", text):
        end = date(P.year, 12, 31)
    elif re.search(r"\bnext year\b", text):
        end = date(P.year + 1, 12, 31)
    elif (m := re.search(r"\b(?:by|in|before|end of|through|until)\s+(?:the end of\s+)?(20[2-9]\d)\b", text)):
        year = int(m.group(1))
        end = date(year, 12, 31) if year >= P.year else None
    elif re.search(r"\bend of (the|this) decade\b", text):
        end = date(P.year - P.year % 10 + 9, 12, 31)
    elif re.search(r"\bthis quarter\b", text):
        end = _quarter_end(P)
    elif re.search(r"\bnext quarter\b", text):
        end = _quarter_end(_add_months(_quarter_end(P), 1))
    elif (m := re.search(r"\bq([1-4])(?:\s*(?:of\s*)?(20[2-9]\d))?\b", text)):
        q = int(m.group(1))
        year = int(m.group(2)) if m.group(2) else P.year
        end = _month_end(year, q * 3)
        if end < P and not m.group(2):
            end = _month_end(year + 1, q * 3)
    elif (m := re.search(r"\b(?:over|in|within|during|for)?\s*(?:the\s+)?(?:next|coming|following)?\s*"
                         r"(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|twelve|eighteen|twenty|"
                         r"couple of|few|several)[- ]?(day|week|month|year)s?\b", text)):
        n = int(m.group(1)) if m.group(1).isdigit() else _WORD_NUM[m.group(1)]
        unit = m.group(2)
        if unit == "day":
            end = P + timedelta(days=n)
        elif unit == "week":
            end = P + timedelta(weeks=n)
        elif unit == "month":
            end = _add_months(P, n)
        else:
            end = _add_months(P, 12 * n)
    if end is None:
        return None, None, _bucket_from_wording(text), "vague_horizon"
    days = (end - P).days
    bucket = "short" if days <= SHORT_MAX_DAYS else "medium" if days <= MEDIUM_MAX_DAYS else "long"
    return P, end, bucket, None


def _bucket_from_wording(text):
    if re.search(r"short[- ]?term|near[- ]?term|coming (days|weeks)|this (week|month)|next (week|month)", text):
        return "short"
    if re.search(r"long[- ]?term|longer[- ]?term|decade|multi[- ]?year|years", text):
        return "long"
    if re.search(r"medium[- ]?term|months|quarter|this year|next year|year[- ]?end", text):
        return "medium"
    return "unspecified"


# --- Entity resolution --------------------------------------------------------
#
# The NAME decides. A subject the curated tables know resolves to its ticker
# whatever the model wrote in ticker_spoken; the model's ticker only confirms
# it (provenance "spoken" / "title") or contradicts it, and a contradiction is
# reviewed only when both sides are real tickers (ticker_subject_mismatch). On
# its own a model ticker is accepted only when the evidence carries it AS A
# TICKER ("$NVDA", "NVDA" in capitals, the word after "ticker"/"symbol") or
# the creator typed it in the video title; anything else is dropped with a
# NON-blocking resolution note, never a review reason. Markets, baskets and
# metrics are told apart from single instruments (SUBJECT_KINDS), so an index,
# a list of companies or a margin never borrows a security's ticker.

# Market subjects: indices, the broad market, sectors, commodities and macro
# themes. None is a security, so none carries a ticker; every spelling folds to
# ONE canonical key, the contract creator_tracker.PROXIES prices against
# ("S&P 500" -> SPY). Bare "the market", "markets" and "stocks" are not here:
# what they mean depends on the channel (the crypto market, the stock market),
# so they stay placeholders.
_MARKET_SPELLINGS = {
    "S&P 500": ("S&P", "S&P 500", "S&P500", "SP500", "SP 500", "S AND P 500", "S&P 500 INDEX", "SPX",
                "THE S&P 500", "THE S&P"),
    "US STOCK MARKET": ("U.S. STOCKS", "US STOCKS", "U.S. STOCK MARKET", "US STOCK MARKET", "STOCK MARKET",
                        "THE STOCK MARKET", "THE U.S. STOCK MARKET", "U.S. EQUITIES", "US EQUITIES",
                        "EQUITY MARKETS", "AMERICAN STOCKS"),
    "NASDAQ": ("NASDAQ", "NASDAQ 100", "NASDAQ-100", "NASDAC", "NASDACK", "NAST 100", "NASDAQ COMPOSITE", "NDX",
               "THE NASDAQ"),
    "DOW": ("DOW", "DOW JONES", "THE DOW", "DJIA"),
    "RUSSELL 2000": ("RUSSELL 2000", "RUSSELL", "THE RUSSELL 2000"),
    "SMALL CAPS": ("SMALL CAPS", "SMALL-CAPS", "SMALLCAPS", "SMALL CAP STOCKS"),
    "MID CAPS": ("MID CAPS", "MIDCAPS", "MID-CAPS"),
    "DAX": ("DAX", "DER DAX"),
    "NIKKEI 225": ("NIKKEI", "NIKKEI 225", "NICKEY", "NICK 225"),
    "DXY": ("DXY", "DOLLAR INDEX", "US DOLLAR INDEX", "US-DOLLAR INDEX"),
    "BITCOIN DOMINANCE": ("BITCOIN DOMINANCE", "BTC DOMINANCE", "BITCOIN DOMINANZ", "DOMINANZ DES BITCOINS",
                          "BITCOIN DOMINANCE EXCLUDING USDT AND USDC"),
    "USDT DOMINANCE": ("USDT DOMINANCE",),
    "GOLD": ("GOLD", "GOLD PRICE", "GOLDPREIS", "THE GOLD PRICE"),
    "SILVER": ("SILVER", "SILBER", "SILBERPREIS"),
    "OIL": ("OIL", "OIL PRICE", "OIL PRICES", "CRUDE", "CRUDE OIL", "WTI", "BRENT", "ÖL", "ÖLPREIS",
            "LIGHT CRUDE OIL FUTURES"),
    "COPPER": ("COPPER", "KUPFER"),
    "GOLD MINERS": ("GOLD MINERS", "GOLD MINING STOCKS", "MINENAKTIEN", "GOLD AND SILVER MINING COMPANIES"),
    "TECHNOLOGY": ("TECHNOLOGY", "TECH", "TECH SECTOR", "TECH STOCKS", "TECHNOLOGY SECTOR", "TECHNOLOGY STOCKS"),
    "SEMICONDUCTORS": ("SEMICONDUCTORS", "SEMIS", "CHIPS", "CHIP STOCKS", "SEMICONDUCTOR STOCKS",
                       "SEMICONDUCTOR SECTOR", "CHIP SPACE"),
    "SOFTWARE": ("SOFTWARE", "SOFTWARE STOCKS", "SOFTWARE SECTOR", "SOFTWAREBEREICH", "SOFTWARESEKTOR"),
    "FINANCIALS": ("FINANCIALS", "FINANCIAL SECTOR", "FINANCIAL STOCKS"),
    "ENERGY": ("ENERGY", "ENERGY SECTOR", "ENERGY STOCKS", "OIL STOCKS"),
    "CONSUMER DISCRETIONARY": ("CONSUMER DISCRETIONARY",),
    "CYBERSECURITY": ("CYBERSECURITY", "CYBERSECURITY STOCKS", "CYBERSECURITY INDUSTRY", "CYBERSECURITY SECTOR"),
    # The SPDR sector funds the scorecard already benchmarks against, and the
    # bank fund (creator_tracker.PROXIES prices each).
    "HEALTHCARE": ("HEALTHCARE", "HEALTH CARE", "HEALTHCARE SECTOR", "HEALTH CARE SECTOR", "HEALTHCARE STOCKS"),
    "INDUSTRIALS": ("INDUSTRIALS", "INDUSTRIAL SECTOR", "INDUSTRIAL STOCKS", "INDUSTRIALS SECTOR"),
    "CONSUMER STAPLES": ("CONSUMER STAPLES", "STAPLES", "CONSUMER STAPLES SECTOR", "STAPLES SECTOR"),
    "UTILITIES": ("UTILITIES", "UTILITY SECTOR", "UTILITIES SECTOR", "UTILITY STOCKS"),
    "MATERIALS": ("MATERIALS", "MATERIALS SECTOR", "BASIC MATERIALS"),
    "BANKS": ("BANKS", "BANK STOCKS", "BANKING SECTOR", "BANKEN"),
    "FED": ("FED", "THE FED", "FEDERAL RESERVE", "THE FEDERAL RESERVE", "CENTRAL BANK"),
    "INFLATION": ("INFLATION", "US INFLATION", "U.S. INFLATION", "CPI", "CORE CPI", "HEADLINE CPI"),
    # Only what names US government debt is US TREASURIES; "bonds" and German
    # "Staatsanleihen" name no issuer, so they are BONDS (never priced either).
    "US TREASURIES": ("TREASURIES", "US TREASURIES", "U.S. TREASURIES", "TREASURY BONDS", "10-YEAR TREASURY",
                      "10-YEAR TREASURIES", "US-STAATSANLEIHEN"),
    "BONDS": ("BONDS", "STAATSANLEIHEN", "10-YEAR BONDS", "ZEHNJÄHRIGE STAATSANLEIHEN", "ANLEIHEN",
              "FIXED INCOME"),
    "ALTCOINS": ("ALTCOINS", "ALTCOIN", "DEN ALTCOINS", "SELECTED ALTCOINS"),
}
MARKET_SUBJECTS = {spelling: key for key, spellings in _MARKET_SPELLINGS.items() for spelling in (key,) + spellings}
# Spellings that are a market only under one asset type: the transcripts hear
# Brent as "Brand" ("Brand is above $15 ... attacks on ships around the
# straight of Hormuz"), but a "brand" is anything else in a stock video.
_TYPED_MARKET_SPELLINGS = {("commodity", "BRAND"): "OIL"}
# "<market> stocks" still names the market only for equity indices and
# sectors ("DAX Aktien", "S&P 500 stocks"). Gold stocks are miners, not the
# metal, and "Dow stock" is Dow Inc. A company form ("Inc", "Corp") is never
# stripped into a market: "Dow Inc" is a company.
_EQUITY_MARKET_KEYS = frozenset({
    "S&P 500", "US STOCK MARKET", "NASDAQ", "RUSSELL 2000", "SMALL CAPS", "MID CAPS", "DAX", "NIKKEI 225",
    "GOLD MINERS", "TECHNOLOGY", "SEMICONDUCTORS", "SOFTWARE", "FINANCIALS", "ENERGY", "CONSUMER DISCRETIONARY",
    "CYBERSECURITY", "HEALTHCARE", "INDUSTRIALS", "CONSUMER STAPLES", "UTILITIES", "MATERIALS", "BANKS",
})
_STOCK_WORDS = (" STOCK", " STOCKS", " SHARES", " SHARE", " AKTIE", " AKTIEN", "-AKTIE", "-AKTIEN")
# Asset types that are never one security: without a curated ticker the
# subject is a market, and no ticker is the right answer rather than a gap.
NON_SECURITY_TYPES = ("macro", "sector", "index", "commodity", "currency", "bond")

# Creator-specific spellings: a name ONE channel uses for an asset (its
# transcripts' recurring mis-hearing, its shorthand). Matched on the claim's
# channel_name only and never applied globally: "new" and "core" are English
# words everywhere else, and XP Inc. is a real company. Evidence:
#   Couch Investor  "New had the first quarter of a billion dollars profit ...
#                   comments from David, the CEO" and "moving on to new
#                   holdings ... added more" (Nu Holdings); "core January 2028
#                   $100 calls ... core is back under $80" against crwv.us at
#                   ~87 (CoreWeave).
#   HKCM            2026-10-13 "... ist Joo Peng geschrieben. XP ... XP ist eben
#                   nicht nur ein Elektromobilitätshersteller" (XPeng, NYSE: XPEV).
#   TheStreet       2026-09-11/14, a US dividend-growth fund manager: "our large
#                   position in Merck" (Merck & Co., NYSE: MRK). Never global:
#                   in the German channels Merck is Merck KGaA (XETRA: MRK).
# FinVid's "Iron" (IREN, or Iron Mountain?) and Phantom by HKCM's "XP" are
# medium-confidence reads and stay unresolved until the ticker audit verifies them.
CHANNEL_ALIASES = {
    "Couch Investor": {"NEW": "NU", "NEW HOLDINGS": "NU", "CORE": "CRWV", "COREE": "CRWV", "CORRE": "CRWV",
                       "COROREE": "CRWV"},
    "HKCM": {"XP": "XPEV", "JOO PENG": "XPEV"},
    "TheStreet": {"MERCK": "MRK", "MERCK & CO": "MRK"},
}

# COMMON_WORD_NAMES (signals_data): curated names that are also ordinary
# words count in a title or inside a subject only in capitals or before
# "stock" / "Aktie".
# Coins the dataset discusses beyond the instrument registry's crypto rows.
_EXTRA_COINS = frozenset({"AXS", "BNB", "CRO", "FET", "HBAR", "TRX", "XLM", "ZEC"})

_TICKER_SHAPE = re.compile(r"^[A-Z]{1,5}(?:\.[A-Z])?$")
# A ticker the creator typed in a title: 1-5 capitals (optionally "$"-led or
# a share class "BRK.B"), not part of a word, a hyphenation, "S&P" / "AT&T"
# or an abbreviation such as "U.S.". One letter ("T Stock Analysis") counts
# only with a "$" or followed by "stock".
_TITLE_TICKER = re.compile(
    r"(?<![\w$&.'’\-])(\$)?([A-Z]{1,5}(?:\.[A-Z](?![.\w]))?)(?![\w&\-]|\.[A-Za-z]|['’](?![sS]\b))")
_STOCK_WORD_AFTER = re.compile(r"\s+(stock|stocks|shares|aktie|aktien)\b", re.IGNORECASE)
_TITLE_NOISE = frozenset({
    "I", "A", "AI", "AGI", "US", "USA", "UK", "EU", "CEO", "CFO", "CIO", "CTO", "IPO", "ETF", "ETFS", "FED", "THE",
    "EPS", "OK", "TV", "CNBC", "NOT", "THIS", "DAS", "DIE", "DER", "UND", "AND", "ALL", "NOW", "BUY", "SELL",
    "HOLD", "MORE", "NEW", "Q", "GDP", "CPI", "ATH", "WHY", "HOW", "JETZT", "ALLES", "NICHT", "DNA", "MAG", "SK",
    "ATT", "AST", "PHASE", "CLOSE", "SMA", "EMA", "RSI", "IS", "IT", "MY", "TOP", "BIG", "WAS", "ARE", "AN", "OR",
    "IN", "ON", "OF", "TO", "AT", "BY", "FOR", "GO", "NO", "YES", "VS", "PART", "LIVE", "NEWS", "WILL", "CAN",
    "BEST", "HUGE", "WOW", "RED", "BIT", "USD", "EUR", "NFT", "ROI", "PE", "YTD", "YOY", "QOQ", "LLM", "GPU",
    "CPU", "API", "HBM", "DRAM", "NAND", "SAAS", "EV", "EVS", "AR", "VR", "OTC", "SEC", "IRS", "FOMC", "ECB",
})
_TICKER_CUE = re.compile(r"\b(ticker|symbol)", re.IGNORECASE)
# A ticker-cue only counts when it sits this close before the token.
_TICKER_CUE_CHARS = 25
_METRIC_SUBJECT = re.compile(
    r"\b(revenues?|sales|margins?|earnings|eps|income|profits?|profitability|cash flow|free cash|buybacks?|"
    r"guidance|growth rate|segment|annual recurring revenue|arr|debt|shipments|bit supply|ratios?|"
    r"returns? on invested capital|roic|platform|management team|outlook|capex|backlog|bookings|market share|"
    r"dividend (yield|growth|payments?|payout))\b", re.IGNORECASE)
# A crypto channel's "price", "the market", "the bulls": the coin the video is about.
_MARKET_REFERENCE = re.compile(r"^(the )?(market|price|bulls|bears)$", re.IGNORECASE)
# Several assets in one subject: "Google and Meta", "SOL/BTC", "Fortinet vs.
# Palo Alto". "AT&T" and "S&P" carry no spaces around "&", so they never split.
_BASKET_SPLIT = re.compile(r"\s*[,/+]\s*|\s+&\s+|\s+(?:and|or|vs\.?|versus|plus|und|oder)\s+", re.IGNORECASE)


def _name_key(name):
    """Normalized lookup form: single-spaced, upper case, typographic
    apostrophes folded ("McDonald’s" -> "MCDONALD'S")."""
    if not isinstance(name, str):
        return ""
    return _norm_name(name.replace("’", "'").replace("‘", "'"))


def _strip_suffixes(key, suffixes=_NAME_SUFFIXES):
    """`key` without the trailing words speakers attach to a name ("Nvidia
    stock", "Fair Isaac Corporation stock"); "earnings per share" and
    "market share" keep their last word."""
    while key:
        for suffix in suffixes:
            if key.endswith(suffix) and len(key) > len(suffix) and not (
                    suffix == " SHARE" and key.endswith((" PER SHARE", " MARKET SHARE"))):
                key = key[: -len(suffix)].rstrip(" ,")
                break
        else:
            break
    return key


# A grouping key keeps a plural: "growth stocks" and "foreign stocks" are
# categories, while "IonQ stock" is one company.
_GROUPING_SUFFIXES = tuple(s for s in _NAME_SUFFIXES if s not in (" STOCKS", " AKTIEN", "-AKTIEN"))


def strip_name(name):
    """A spoken name as a grouping key: normalized, without trailing words
    such as " stock" / " Aktie" / " Inc" ("IonQ stock" -> "IONQ")."""
    return _strip_suffixes(_name_key(name), _GROUPING_SUFFIXES) or None


# Leading articles a lookup tries without ("the S&P 500", "Der Bitcoin",
# "die Aktie" is still generic: it names nothing once stripped).
_ARTICLES = ("THE ", "DER ", "DIE ", "DAS ", "DEN ", "DEM ")


def _without_article(key):
    for article in _ARTICLES:
        if key.startswith(article):
            return key[len(article):]
    return key


def _name_variants(key):
    """The spellings a name lookup tries, in order: as written, without
    trailing words (" stock", " Inc"), without a possessive "'s", and
    without a leading article ("the", German "der" / "die" / "das" /
    "den" / "dem": "Der Bitcoin" is Bitcoin)."""
    out = [key]
    stripped = _strip_suffixes(key)
    if stripped and stripped not in out:
        out.append(stripped)
    for variant in list(out):
        if variant.endswith("'S") and variant[:-2] not in out:
            out.append(variant[:-2])
    for variant in list(out):
        bare = _without_article(variant)
        if bare != variant and bare not in out:
            out.append(bare)
    return [v for v in out if v]


def market_key(name):
    """The canonical MARKET_SUBJECTS key for an index / sector / commodity /
    macro spelling ("the S&P 500", "Silber", "chip stocks"), else None."""
    key = _name_key(name)
    if not key:
        return None
    for candidate in (key, _without_article(key)):
        if candidate in MARKET_SUBJECTS:
            return MARKET_SUBJECTS[candidate]
    stripped = _strip_suffixes(key, _STOCK_WORDS)
    if stripped and stripped != key:
        for candidate in (stripped, _without_article(stripped)):
            found = MARKET_SUBJECTS.get(candidate)
            if found in _EQUITY_MARKET_KEYS:
                return found
    return None


def _name_shaped_alias(key):
    """A TICKER_ALIASES key read as a NAME ("SPACEX", "AMAZON", "I-REN"). The
    ticker-shaped keys (GOOG, RUBY, NEBL) are only ever read from a ticker
    field: as a spoken name they mean something else."""
    return TICKER_ALIASES.get(key) if not _TICKER_SHAPE.match(key) else None


def _lookup_name(name, channel=None):
    """(ticker, table) for a name: the channel's own spellings, then
    ASSET_ALIASES, then the name-shaped TICKER_ALIASES keys, then the
    catalogue-verified learned map, each trying the _name_variants. The
    ticker is folded through TICKER_ALIASES; table is "channel", "curated" or
    "learned". (None, None) when no table knows the name."""
    key = _name_key(name)
    if not key:
        return None, None
    variants = _name_variants(key)
    scoped = CHANNEL_ALIASES.get(channel or "", {})
    learned = learned_tickers()
    for table, lookup in (("channel", scoped.get), ("curated", ASSET_ALIASES.get),
                          ("curated", _name_shaped_alias), ("learned", learned.get)):
        for variant in variants:
            ticker = lookup(variant)
            if ticker:
                return TICKER_ALIASES.get(ticker, ticker), table
    return None, None


def name_ticker(name, channel=None):
    """The curated ticker for a NAME (see _lookup_name), or None."""
    return _lookup_name(name, channel)[0]


def channel_ticker(name, channel):
    """The ticker `channel`'s own spelling of `name` stands for (CHANNEL_ALIASES), or None."""
    scoped = CHANNEL_ALIASES.get(channel or "")
    key = _name_key(name)
    if not scoped or not key:
        return None
    for variant in _name_variants(key):
        if variant in scoped:
            return TICKER_ALIASES.get(scoped[variant], scoped[variant])
    return None


def known_tickers():
    """Tickers the curated tables vouch for: every alias value, every
    TICKER_ALIASES target, the learned map and the instrument registry. A
    ticker-shaped word that merely repeats the subject ("New", "mogu") is a
    ticker only when it is one of these."""
    import instruments
    return (set(ASSET_ALIASES.values()) | set(TICKER_ALIASES.values()) | set(learned_tickers().values())
            | set(instruments.REGISTRY))


def crypto_tickers():
    """Coins: the registry's crypto instruments plus the few other coins the dataset discusses."""
    import instruments
    return {t for t, inst in instruments.REGISTRY.items() if inst.asset_type == "crypto"} | _EXTRA_COINS


def _spoken_raw(ticker_spoken):
    """The model's ticker field as written: upper case, without a "$"."""
    return _name_key((_s(ticker_spoken) or "").lstrip("$"))


def fold_spoken_ticker(ticker_spoken, channel=None):
    """
    (ticker, was_name) for the model's ticker_spoken field. A NAME written in
    the ticker field folds through the curated tables ("SOLANA" -> SOL,
    "BITCOIN" -> BTC, "XPENG" -> XPEV; was_name=True); a market spelling
    ("NASDAQ", "S&P 500") or anything not shaped like a ticker
    (^[A-Z]{1,5}(.[A-Z])?$) is no ticker at all; a ticker-to-ticker rename
    folds (GOOG -> GOOGL).
    """
    raw = _spoken_raw(ticker_spoken)
    if not raw:
        return None, False
    if raw in NOT_A_TICKER or market_key(raw):
        return None, True
    if _TICKER_SHAPE.match(raw) and raw in TICKER_ALIASES and raw not in ASSET_ALIASES:
        return TICKER_ALIASES[raw], False
    if raw in LISTED_TICKER_KEYS:
        return raw, False          # FORD in a ticker field is Forward Industries, not Ford Motor
    named = name_ticker(raw, channel)
    if named and named != raw:
        return named, True
    if not _TICKER_SHAPE.match(raw):
        return None, True
    return raw, False


def _mostly_upper(text):
    letters = [c for c in text or "" if c.isalpha()]
    return bool(letters) and sum(1 for c in letters if c.isupper()) / len(letters) >= 0.6


def title_tickers(title):
    """{ticker} the creator TYPED in a title ("| FTNT Stock vs. PANW Stock",
    "Why $MU ..."), folded through TICKER_ALIASES (GOOG -> GOOGL). Common
    capitalised words, market spellings and an all-caps title's words are
    not tickers."""
    title = title if isinstance(title, str) else ""
    if _mostly_upper(title):
        return set()
    found = set()
    for m in _TITLE_TICKER.finditer(title):
        dollar, token = m.group(1), m.group(2)
        if token in _TITLE_NOISE or token in NOT_A_TICKER or token in MARKET_SUBJECTS:
            continue
        if len(token) == 1 and not dollar and not _STOCK_WORD_AFTER.match(title, m.end()):
            continue
        found.add(TICKER_ALIASES.get(token, token))
    return found


_LEARNED_RE_CACHE = {}


def _learned_name_re():
    learned = learned_tickers()
    names = tuple(sorted(learned, key=len, reverse=True))
    if names not in _LEARNED_RE_CACHE:
        _LEARNED_RE_CACHE.clear()
        _LEARNED_RE_CACHE[names] = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b",
                                              re.IGNORECASE) if names else None
    return _LEARNED_RE_CACHE[names]


def _names_in(text, channel=None):
    """{ticker} for the curated and learned names `text` mentions. A
    COMMON_WORD_NAMES entry counts only in capitals or before "stock"."""
    import transcript_normalize as tn
    text = text if isinstance(text, str) else ""
    found = set()
    for m in tn._ALIAS_RE.finditer(text):
        name = m.group(1)
        if _name_key(name) in COMMON_WORD_NAMES and not (
                (len(name) > 1 and name.isupper()) or _STOCK_WORD_AFTER.match(text, m.end())):
            continue
        ticker = name_ticker(name, channel)
        if ticker:
            found.add(ticker)
    pattern = _learned_name_re()
    if pattern is not None:
        for m in pattern.finditer(text):
            ticker = name_ticker(m.group(1), channel)
            if ticker:
                found.add(ticker)
    return found


# "Is Grab Stock a Buy?": a word the creator calls a stock in the title.
_NAMED_STOCK = re.compile(r"(?<![\w$&.'’\-])([A-Z][A-Za-z]{1,4})\s+(?i:stock|stocks|shares|aktie|aktien)\b")


def title_assets(title, channel=None):
    """{ticker} a video title names: the curated company names it mentions,
    the tickers the creator typed (title_tickers), and a known ticker the
    title calls a stock by name ("Is Grab Stock a Buy?" -> GRAB). Markets
    are not assets."""
    title = title if isinstance(title, str) else ""
    found = _names_in(title, channel) | title_tickers(title)
    named = {m.group(1).upper() for m in _NAMED_STOCK.finditer(title)} - _TITLE_NOISE
    if named:
        found |= named & known_tickers()
    return found


def _spoken_in_evidence(ticker, evidence):
    """True when the evidence carries `ticker` AS A TICKER: "$NVDA", "NVDA"
    written in capitals (two letters or more, in an evidence that is not
    all capitals), or within a few characters after "ticker" / "symbol"
    ("ticker symbol Mogu"). An ordinary word in lower or title case ("New
    is a well-run company", "iron") is not a spoken ticker."""
    ev = evidence if isinstance(evidence, str) else ""
    caps_count = not _mostly_upper(ev)
    for m in re.finditer(r"(?<![\w$])(\$?)(" + re.escape(ticker) + r")(?!\w)", ev, re.IGNORECASE):
        token = m.group(2)
        if m.group(1) or _TICKER_CUE.search(ev[max(0, m.start() - _TICKER_CUE_CHARS):m.start()]):
            return True
        if caps_count and len(token) > 1 and token == ticker:
            return True
    return False


def _name_in_text(name, text):
    key = _name_key(name)
    return bool(key) and re.search(r"(?<!\w)" + re.escape(key) + r"(?!\w)", _name_key(text)) is not None


_NOT_A_NAME_START = re.compile(r"^(the|this|that|these|those|its|it|their|our|my|his|her|a|an|der|die|das|den)\b",
                               re.IGNORECASE)


def _subject_parts(mention, channel=None):
    """[(part, identity)] for a subject joined by "and" / "or" / "vs" / "," /
    "/" (one entry for a plain subject). identity is ("ticker", T) or
    ("market", KEY) when the tables know the part, else None."""
    text = (mention or "").replace("’", "'")
    parts = [p.strip() for p in _BASKET_SPLIT.split(text) if p and p.strip()]
    out = []
    for part in parts:
        ticker = name_ticker(part, channel)
        market = None if ticker else market_key(part)
        out.append((part, ("ticker", ticker) if ticker else ("market", market) if market else None))
    return out


def _is_basket(mention, channel=None):
    """
    True for a subject that names several assets: a list with two or more
    commas; two or more parts naming distinct assets or markets ("Google and
    Meta", "SOL/BTC", "AT&T and Verizon"); or one named asset beside a name
    the tables do not know ("Google and Constellation"). "Hims & Hers",
    "AT&T", "S&P 500" and "Procter & Gamble" stay whole (their parts name
    nothing), and so does "Fair Isaac or FICO" (one asset, twice).
    """
    if (mention or "").count(",") >= 2:
        return True
    parts = _subject_parts(mention, channel)
    if len(parts) < 2:
        return False
    distinct = {identity for _, identity in parts} - {None}
    if len(distinct) >= 2:
        return True
    return len(distinct) == 1 and any(
        identity is None and part[:1].isupper() and not _NOT_A_NAME_START.match(part) and not _is_generic(part)
        for part, identity in parts)


# A pair's quote side ("ETH BTC", "Ethereum Bitcoin", "sole BTC"): a coin
# priced in another coin is a ratio chart, not one asset.
_QUOTE_COINS = frozenset({"BTC", "ETH", "USDT", "USD", "BITCOIN", "ETHEREUM"})


def _is_coin_pair(mention, asset_type=None, channel=None):
    """True for a two-word crypto subject whose second word is a quote coin
    and whose first is a coin too, or a short word the tables do not know
    (a mis-hearing: "sole BTC" is SOL/BTC): "ETH BTC", "Ethereum Bitcoin"."""
    if asset_type != "crypto":
        return False
    words = _name_key(mention).split()
    if len(words) != 2 or words[1] not in _QUOTE_COINS:
        return False
    first = words[0]
    if name_ticker(first, channel) or first in crypto_tickers() or first in _QUOTE_COINS:
        return True
    return len(first) <= 5 and first.isalpha() and not market_key(first) and first not in known_tickers()


def _ticker_subject(mention, raw):
    """True when the subject is the ticker itself, written as one: two or
    more capitals, nothing else ("STX", not "Stx" or "the STX chart")."""
    text = (mention or "").strip()
    return bool(raw) and text == raw and len(text) >= 2 and text.isupper() and bool(_TICKER_SHAPE.match(text))


def _registered(ticker):
    import instruments
    return ticker in instruments.REGISTRY


def _one_asset_twice(mention, channel=None):
    """The identity every part of a multi-part subject names, when they all
    name the same one ("Fair Isaac Corporation or FICO stock" -> FICO)."""
    parts = _subject_parts(mention, channel)
    identities = {identity for _, identity in parts}
    if len(parts) >= 2 and len(identities) == 1 and None not in identities:
        return identities.pop()
    return None


def resolve_entity(subject_mention, ticker_spoken, evidence, title, asset_type=None, channel=None):
    """
    Deterministic entity resolution (see the section comment above). Returns
    dict(canonical_entity_name, ticker, ticker_source, entity_resolution_status,
    review_reason, subject_kind, resolution_notes, resolution_method_hint).

    In order: a name in AMBIGUOUS_MENTIONS is never resolved; a market subject
    (MARKET_SUBJECTS, or a macro / sector / index / commodity / currency /
    bond type with no curated ticker) takes no ticker and is confirmed under
    its canonical key; a curated name decides; a basket takes no ticker; a
    metric takes the one company named inside it, else the title's; a generic
    subject ("the company") inherits the title's asset (confirmed when the
    title names one, probable for the model's pick among several); a crypto
    channel's "price" / "the market" takes the one coin in the title
    (probable); otherwise the model's ticker stands only when the evidence or
    the title carries it as a ticker. resolution_method_hint="title" marks a
    title coreference. Nothing is resolved on phonetic similarity.
    """
    mention = _s(subject_mention)
    asset_type = asset_type if isinstance(asset_type, str) else None   # stored rows are data, not code
    channel = channel if isinstance(channel, str) else None
    out = {"canonical_entity_name": mention, "ticker": None, "ticker_source": None,
           "entity_resolution_status": "unresolved", "review_reason": None, "subject_kind": "instrument",
           "resolution_notes": [], "resolution_method_hint": None}
    notes = out["resolution_notes"]
    if not mention:
        out["review_reason"] = "missing_asset"
        return out
    key = _name_key(mention)
    if key in AMBIGUOUS_MENTIONS:
        out["entity_resolution_status"] = "ambiguous"
        out["review_reason"] = "ambiguous_entity"
        return out
    spoken, was_name = fold_spoken_ticker(ticker_spoken, channel)
    raw = _spoken_raw(ticker_spoken)   # as written: the evidence carries "GOOG", not its fold GOOGL
    gave_ticker = bool(raw)

    curated, table = _lookup_name(mention, channel)
    twice = None if curated else _one_asset_twice(mention, channel)
    market = market_key(mention) or _TYPED_MARKET_SPELLINGS.get((asset_type, key)) or (
        twice[1] if twice and twice[0] == "market" else None)
    if market:
        out.update(ticker_source="unresolved", entity_resolution_status="confirmed", subject_kind="market",
                   canonical_entity_name=market)
        if spoken and not was_name:
            notes.append("ticker_spoken_replaced")   # "SPY" for the S&P 500: the market, not one fund
        return out

    if twice:
        curated, table = twice[1], "curated"
    if curated:
        source = "curated_mapping"
        if spoken == curated and not was_name:
            if _spoken_in_evidence(raw, evidence):
                source = "spoken"
            elif spoken in title_tickers(title):
                source = "title"
        elif spoken and spoken != curated:
            if spoken in known_tickers() and spoken not in FORMER_TICKERS.get(curated, ()):
                out["review_reason"] = "ticker_subject_mismatch"   # two real tickers disagree
            else:
                notes.append("ticker_spoken_replaced")             # "WEED", "NEW", a former ticker
        out.update(ticker=curated, ticker_source=source,
                   entity_resolution_status="probable" if table == "learned" else "confirmed")
        return out

    if _is_basket(mention, channel) or _is_coin_pair(mention, asset_type, channel):
        out.update(ticker_source="unresolved", subject_kind="basket")
        notes.append("multi_asset_subject")
        return out

    generic = _is_generic(mention)
    metric = not generic and bool(_METRIC_SUBJECT.search(mention))
    titled = title_assets(title, channel)
    # A company's segment the model typed "sector" is still a metric of that
    # company: it says so ("AWS segment") or the model pinned it to the
    # video's company ("defense and intelligence" -> PL in a Planet Labs
    # video). Any other non-security type is a market.
    if asset_type == "sector" and not generic and spoken in titled:
        metric = True
    if asset_type in NON_SECURITY_TYPES and not (metric and asset_type == "sector"):
        out.update(ticker_source="unresolved", entity_resolution_status="confirmed", subject_kind="market")
        return out
    if metric:
        out["subject_kind"] = "metric"
        inside = _names_in(mention, channel) | (title_tickers(mention) & known_tickers())
        if len(inside) == 1:
            out.update(ticker=inside.pop(), ticker_source="curated_mapping", entity_resolution_status="confirmed")
            return out
    if _MARKET_REFERENCE.match(mention) and asset_type == "crypto":
        coins = titled & crypto_tickers()
        if len(coins) == 1:
            out.update(ticker=coins.pop(), ticker_source="title", entity_resolution_status="probable",
                       resolution_method_hint="title")
            notes.append("crypto_title_coreference")
            return out
    if generic or metric:
        # A reference inherits the title's asset: the model's pick when it is
        # one of the title's assets (only probable among several), else the
        # title's one asset when the model gave no ticker at all and the
        # tables know that ticker (a stray capitalised title word is not one).
        pick = spoken if spoken in titled else None
        if pick is None and not gave_ticker and len(titled) == 1:
            only = next(iter(titled))
            pick = only if only in known_tickers() or only in crypto_tickers() else None
        if pick:
            out.update(ticker=pick, ticker_source="title", resolution_method_hint="title",
                       entity_resolution_status="confirmed" if len(titled) == 1 else "probable")
            return out

    # The model's ticker on its own.
    if spoken and not was_name and raw in _name_variants(key) and spoken not in known_tickers():
        notes.append("ticker_equals_name_unverified")   # "New" -> NEW, "mogu" -> MOGU
        spoken = None
    if spoken:
        if was_name:
            # A curated name in the ticker field: the table vouches for its
            # ticker, the evidence or the title for the name.
            if spoken in titled or _name_in_text(raw, evidence):
                out.update(ticker=spoken, ticker_source="curated_mapping", entity_resolution_status="confirmed")
                return out
        elif _spoken_in_evidence(raw, evidence):
            out.update(ticker=spoken, ticker_source="spoken", entity_resolution_status="confirmed")
            return out
        elif spoken in titled:
            out.update(ticker=spoken, ticker_source="title", entity_resolution_status="confirmed")
            return out
        elif _ticker_subject(mention, raw) and spoken == raw and _registered(spoken):
            # The subject IS the ticker, written as one ("STX"), and the
            # registry knows it: the speaker named the ticker.
            out.update(ticker=spoken, ticker_source="spoken", entity_resolution_status="confirmed")
            return out
        notes.append("ticker_spoken_unverified")
    out["ticker_source"] = "unresolved"
    return out


def _is_question(evidence):
    ev = (evidence or "").strip()
    return ev.endswith("?") or (bool(_QUESTION_START.match(ev)) and "?" in ev)


def condition_observability(condition):
    """
    (observable, kind, data_source) for a forecast's condition. Observable
    means a documented data source settles whether it held: a price level, a
    policy-rate decision, a scheduled macro release, an earnings release or a
    dated public event. A condition nobody can measure ("if management
    executes", "if sentiment improves") is (False, "subjective", None). The
    existence of a condition is never by itself what makes a claim
    untestable — only an unobservable one is.
    """
    text = (condition or "").strip()
    if not text:
        return False, None, None
    for pattern, kind, source in OBSERVABLE_CONDITIONS:
        if pattern.search(text):
            return True, kind, source
    return False, "subjective", None


def portfolio_position(evidence):
    """The position the evidence itself supports: long / short / no_position
    / owns_unspecified, or None when it discloses nothing."""
    ev = evidence or ""
    if _NO_POSITION_RE.search(ev):
        return "no_position"
    if _SHORT_RE.search(ev):
        return "short"
    if _LONG_RE.search(ev):
        return "long"
    if _OWNERSHIP_RE.search(ev):
        return "owns_unspecified"
    return None


def host_position_from(evidence, model_value=None):
    """
    Where the speaker stands on a third party's view, from the evidence
    first (explicit rejection beats adoption beats the model's word) and the
    model's `host_position` only when the evidence says nothing.
    """
    ev = evidence or ""
    if _REJECT_RE.search(ev):
        return "rejected"
    if _ADOPT_RE.search(ev):
        return "adopted"
    if model_value in ("adopted", "rejected") :
        # The model may have read a wider context than the excerpt; an
        # adoption it asserts without evidence in the excerpt is not acted
        # on (it would promote a third-party view into the speaker's own).
        return "neutral"
    return "neutral"


def _asset_mentions(text):
    """[(offset, name as written)] for curated asset names and learned
    tickers in `text`, leftmost first. Only names the dataset can already
    identify count — a coreference is never resolved to a name nobody has
    checked."""
    import transcript_normalize as tn
    found = []
    for m in tn.alias_mentions(text):       # an ordinary word ("upstart") is no mention
        found.append((m.start(), m.group(1)))
    learned = learned_tickers()
    if learned:
        for name in learned:
            for m in re.finditer(r"\b" + re.escape(name) + r"\b", text or "", re.IGNORECASE):
                found.append((m.start(), m.group(0)))
    return sorted(found)


def _asset_identity(name):
    key = _norm_name(name)
    return ASSET_ALIASES.get(key) or learned_tickers().get(key) or key


def local_subject(nt, span, evidence):
    """
    Resolve a generic subject ("the stock", "it") from the ACTIVE local
    subject: asset names in the evidence itself, else in the previous
    COREFERENCE_SENTENCES sentences of the same segment (at most
    COREFERENCE_MAX_CHARS back). Returns (name, status) where status is
    "resolved" (exactly one active subject), "ambiguous" (two or more
    distinct assets are active, nothing is guessed) or "unresolved" (none).
    Uses only the transcript, never outside knowledge.
    """
    if span is None:
        return None, "unresolved"
    in_evidence = _asset_mentions(evidence)
    if in_evidence:
        names = {_asset_identity(n) for _, n in in_evidence}
        return (in_evidence[-1][1], "resolved") if len(names) == 1 else (None, "ambiguous")
    seg = nt.segment_at(span[0])
    seg_start = seg.start_character if seg else 0
    window_start = max(seg_start, span[0] - COREFERENCE_MAX_CHARS)
    before = nt.text[window_start:span[0]]
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n", before) if s.strip()]
    context = " ".join(sentences[-COREFERENCE_SENTENCES:])
    mentions = _asset_mentions(context)
    if not mentions:
        return None, "unresolved"
    names = {_asset_identity(n) for _, n in mentions}
    if len(names) > 1:
        return None, "ambiguous"
    return mentions[-1][1], "resolved"


# --- Suspicious-empty check ---------------------------------------------------
#
# Valid JSON with claims: [] proves only that the model returned nothing. A
# deterministic, high-recall scan of the transcript decides whether "nothing"
# is plausible: strong indicators are an identifiable asset in the same
# sentence as forecast or recommendation language (with or without a number);
# moderate ones are an asset with bullish/bearish wording or with a number
# and directional language. Educational, historical and interrogative
# sentences never count, so an explainer full of percentages is not an alert.

_SUSP_PREDICT = re.compile(
    r"\b(will|won'?t|is going to|are going to|gonna|i expect|we expect|expects?|expecting|"
    r"could (hit|reach|go|get|fall|drop|rally|double|triple|see|be)|should (hit|reach|go|rally|fall|be|see)|"
    r"(price )?targets?\b|my target|by (the )?end of|next (year|quarter|month|week)|over the next|"
    r"in the next|within (a|the next|\d)|heading (to|for|towards?)|headed (to|for|towards?)|on its way to|"
    r"(will|could|should|may|might|gonna|going to) (double|triple|halve|crash|rally|soar|drop|fall|rise|"
    r"climb|surge|plunge|tank|moon|explode|collapse|outperform|underperform|recover|bounce|break(out)?)|"
    r"i see (it|them|\w+ (at|going|hitting|reaching)))\b", re.IGNORECASE)
_SUSP_RECO = re.compile(
    r"\b(i('| a)?m (buying|selling|adding|trimming|shorting|accumulating)|i would (buy|sell|short|avoid)|"
    r"i'?d (buy|sell|short|avoid)|buy (it|the dip|this|these|shares|here|now)|sell (it|this|these|"
    r"everything|here|now)|(strong )?(buy|sell|hold) rating|(a|my|our) (buy|sell|hold|avoid)\b|"
    r"i recommend|stay away|avoid (it|this|the stock|them)|accumulate|load up|take profits?|"
    r"(don'?t|do not) (buy|sell|touch|chase)|you should (buy|sell|own|avoid)|(this|it) is a (buy|sell|hold)|"
    r"i'?m (long|short)|(over|under)weight)\b", re.IGNORECASE)
_SUSP_BULLBEAR = re.compile(
    r"\b(bullish|bearish|overvalued|undervalued|overbought|oversold|upside|downside|cheap|expensive|"
    r"(over|under)priced|top pick|conviction)\b", re.IGNORECASE)
_SUSP_NUMBER = re.compile(
    r"(\$\s?\d|\d+(\.\d+)?\s?(%|percent)|\b\d+(\.\d+)?\s?(k|thousand|million|billion|trillion)\b|\b\d+x\b|"
    r"\b(double|triple)\b)", re.IGNORECASE)
_SUSP_EXCLUDE = re.compile(
    r"\b(historically|for example|for instance|let'?s say|imagine|suppose|hypothetically|what is|what'?s|"
    r"how does|how do|means that|is defined|in general|on average|typically|the formula|last (year|month|"
    r"week|time)|back in|in (19|20)\d\d|returned|was|were|had|used to|sponsor|use code|promo)\b",
    re.IGNORECASE)
_SUSP_GENERIC_ASSET = re.compile(r"\b(the stock|this stock|the shares|the coin|this coin|the token|"
                                 r"bitcoin|ethereum|s&p ?500|nasdaq|gold|oil)\b", re.IGNORECASE)
_SUSP_MOVE = re.compile(r"\b(fall|drop|rise|rally|double|triple|crash|soar|climb|surge|plunge|reach|hit)\b",
                        re.IGNORECASE)

# German rule set (HKCM and Phantom by HKCM publish in German). Same shape
# as the English one: prediction wording, recommendation wording, bullish /
# bearish vocabulary, educational / historical exclusions, generic asset
# references, and the price-move verbs that make a prediction "strong".
_SUSP_PREDICT_DE = re.compile(
    r"\b(wird|werden|dürfte|dürften|sollte|sollten|könnte|könnten|ich erwarte|wir erwarten|erwarte ich|"
    r"ich rechne|wir rechnen|rechne ich|ich gehe davon aus|kursziel|kursziele|zielzone|zielbereich|"
    r"mein ziel|unser ziel|bis (ende|jahresende|mitte|zum)|nächste[snm]? (jahr|quartal|monat|woche)|"
    r"in den nächsten|innerhalb (der nächsten|von)|richtung|auf dem weg (zu|nach)|"
    r"ich sehe (sie|ihn|es|die aktie|den kurs|bitcoin) (bei|auf|in richtung))\b", re.IGNORECASE)
_SUSP_RECO_DE = re.compile(
    r"\b(ich kaufe|wir kaufen|kaufe ich|ich verkaufe|wir verkaufen|verkaufe ich|nachkaufen|nachgekauft|"
    r"aufstocken|aufgestockt|einsteigen|eingestiegen|einstieg|ausstieg|aussteigen|ausgestiegen|"
    r"kaufempfehlung|verkaufsempfehlung|ich empfehle|finger weg|nicht (kaufen|anfassen)|gewinne mitnehmen|"
    r"position (aufgebaut|eröffnet|geschlossen)|ich bin (long|short|investiert)|ich halte|wir halten|"
    r"kaufzone|kaufbereich|ein (klarer |ganz klarer )?(kauf|verkauf)|jetzt (kaufen|verkaufen)|"
    r"(über|unter)gewichten|(un)?bedingt (kaufen|verkaufen)|würde ich (kaufen|verkaufen|meiden))\b",
    re.IGNORECASE)
_SUSP_BULLBEAR_DE = re.compile(
    r"\b(bullisch|bärisch|bullish|bearish|unterbewertet|überbewertet|überkauft|überverkauft|"
    r"aufwärtspotenzial|potenzial|upside|downside|abwärtsrisiko|günstig|teuer|top[- ]pick|überzeugung|"
    r"schnäppchen|kursrakete)\b", re.IGNORECASE)
_SUSP_EXCLUDE_DE = re.compile(
    r"\b(historisch|zum beispiel|beispielsweise|stell dir vor|stellen sie sich vor|angenommen|nehmen wir an|"
    r"hypothetisch|was ist|was bedeutet|wie funktioniert|bedeutet,? dass|im allgemeinen|im schnitt|"
    r"im durchschnitt|typischerweise|die formel|letztes? (jahr|quartal|monat|woche)|letzte woche|damals|"
    r"im jahr (19|20)\d\d|war|waren|hatte|hatten|früher|sponsor|werbung|rabattcode|gutscheincode)\b",
    re.IGNORECASE)
_SUSP_GENERIC_ASSET_DE = re.compile(
    r"\b(die aktie|diese aktie|der aktie|die aktien|der kurs|den kurs|der coin|der token|bitcoin|ethereum|"
    r"s&p ?500|nasdaq|dax|gold|öl|der markt)\b", re.IGNORECASE)
_SUSP_MOVE_DE = re.compile(
    r"\b(steigen|steigt|fallen|fällt|sinken|sinkt|rally|rallye|verdoppeln|verdreifachen|crash|abstürzen|"
    r"klettern|explodieren|einbrechen|erreichen|erreicht|erholen|ausbrechen|durchstarten|korrigieren)\b",
    re.IGNORECASE)

# One rule set per language the dataset contains. A transcript in any other
# language has NO guard: an empty extraction there is never certified.
SUSPICIOUS_RULES = {
    "en": {"predict": _SUSP_PREDICT, "reco": _SUSP_RECO, "bullbear": _SUSP_BULLBEAR, "exclude": _SUSP_EXCLUDE,
           "generic_asset": _SUSP_GENERIC_ASSET, "move": _SUSP_MOVE},
    "de": {"predict": _SUSP_PREDICT_DE, "reco": _SUSP_RECO_DE, "bullbear": _SUSP_BULLBEAR_DE,
           "exclude": _SUSP_EXCLUDE_DE, "generic_asset": _SUSP_GENERIC_ASSET_DE, "move": _SUSP_MOVE_DE},
}
_AMOUNT_RE = re.compile(r"([$€£]\s?\d|\d+(\.\d+)?\s?(%|\b(percent|prozent)\b)|\b\d+(\.\d+)?\s?(k|thousand|million|"
                        r"billion|trillion|tausend|millionen|milliarden|dollars?|euros?)\b)", re.IGNORECASE)


# Auto-captions carry little punctuation, so a "sentence" can run for
# minutes; runs longer than this many words are scanned in fixed windows.
_SENTENCE_MAX_WORDS = 40
_WINDOW_WORDS = 25


def _sentences(text):
    out = []
    for piece in re.split(r"(?<=[.!?])\s+|\n", text or ""):
        piece = piece.strip()
        if not piece:
            continue
        words = piece.split()
        if len(words) <= _SENTENCE_MAX_WORDS:
            out.append(piece)
            continue
        for i in range(0, len(words), _WINDOW_WORDS):
            out.append(" ".join(words[i:i + _WINDOW_WORDS]))
    return out


def _allow_ticker_like(text):
    letters = [c for c in text if c.isalpha()]
    # An all-caps transcript would make every word look like a ticker.
    return bool(letters) and sum(1 for c in letters if c.isupper()) / len(letters) < 0.6


def _has_asset(sentence, allow_ticker_like, generic_asset=_SUSP_GENERIC_ASSET):
    if _asset_mentions(sentence) or generic_asset.search(sentence):
        return True
    if allow_ticker_like:
        import transcript_normalize as tn
        return bool(tn._TICKER_LIKE.search(sentence) and tn._assets_in(sentence))
    return False


def asset_free(text):
    """
    True when the transcript names no asset the dataset can identify (no
    curated or learned name, no ticker-like token) and no amount (currency,
    percentage, magnitude) — the one language-independent fact that makes
    an empty extraction believable when no language guard is available. A
    claim needs an identifiable asset; a transcript with none and with no
    number cannot carry a price forecast, target or recommendation this
    pipeline could have captured.
    """
    text = text or ""
    if _asset_mentions(text) or _AMOUNT_RE.search(text):
        return False
    if _allow_ticker_like(text):
        import transcript_normalize as tn
        if tn._TICKER_LIKE.search(text) and tn._assets_in(text):
            return False
    return True


def suspicious_empty_check(text, language=None):
    """
    Whether an empty claims array is believable for this transcript:
    {"suspicious": bool | None, "language", "guard": "available" |
    "unavailable", "strong": [...], "moderate": [...], "score": n,
    "asset_free": bool}.

    The vocabulary is language-specific (SUSPICIOUS_RULES: English and
    German, the languages in the channel list). `language` is the
    transcript's language metadata; without it the language is detected.
    For a language with no rule set the guard is UNAVAILABLE: "suspicious"
    is None (unknown, not False) and the caller must not certify the empty
    result as no_claims_found — unless `asset_free` proves the transcript
    names no asset and no amount at all.

    With a rule set: suspicious when at least one strong sentence exists
    (asset + forecast or recommendation language, no educational /
    historical marker, not a question) or two moderate ones (asset +
    bullish/bearish wording, or asset + number + prediction wording).
    Deterministic; never raises.
    """
    from language_detect import language_of
    text = text or ""
    lang = language_of(text, language)
    rules = SUSPICIOUS_RULES.get(lang)
    if rules is None:
        return {"suspicious": None, "language": lang, "guard": "unavailable", "strong": [], "moderate": [],
                "score": 0, "asset_free": asset_free(text)}
    allow_ticker_like = _allow_ticker_like(text)
    strong, moderate = [], []
    for sentence in _sentences(text):
        if sentence.endswith("?") or rules["exclude"].search(sentence):
            continue
        if not _has_asset(sentence, allow_ticker_like, rules["generic_asset"]):
            continue
        predict, reco = bool(rules["predict"].search(sentence)), bool(rules["reco"].search(sentence))
        number, bullbear = bool(_SUSP_NUMBER.search(sentence)), bool(rules["bullbear"].search(sentence))
        if reco or (predict and (number or bullbear)) or (predict and rules["move"].search(sentence)):
            strong.append(sentence[:200])
        elif bullbear or (predict and number) or predict:
            moderate.append(sentence[:200])
    suspicious = bool(strong) or len(moderate) >= 2
    return {"suspicious": suspicious, "language": lang, "guard": "available", "strong": strong[:5],
            "moderate": moderate[:5], "score": 2 * len(strong) + len(moderate), "asset_free": None}


def empty_extraction_verdict(text, language=None, standalone=False):
    """
    The research status an EMPTY claims array earns for this transcript:
    (status, failure_reason, warning). no_claims_found only when a
    language guard exists and finds no claim-bearing language, or when the
    transcript is provably asset-free; suspicious_empty_extraction is
    retried (combined call) or reviewed (standalone call); a transcript in
    a language without a guard goes to review as
    empty_extraction_language_guard_unavailable.
    """
    check = suspicious_empty_check(text, language)
    if check["guard"] == "unavailable":
        if check["asset_free"]:
            return "no_claims_found", None, (f"language_guard_unavailable:{check['language']}; "
                                             "transcript names no asset and no amount")
        return ("needs_review", "empty_extraction_language_guard_unavailable",
                f"empty_extraction_language_guard_unavailable: language {check['language']}")
    if check["suspicious"]:
        return ("needs_review" if standalone else "failed_retryable", "suspicious_empty_extraction",
                "suspicious_empty_extraction: " + "; ".join(check["strong"] or check["moderate"]))
    return "no_claims_found", None, None


# --- Non-view field normalization --------------------------------------------
#
# A candidate the rules reclassify as a question, a portfolio disclosure, a
# retrospective, a reported third-party view / news item, a hypothetical or
# a fact is NOT the speaker's forecast. Leaving `forecast_direction`,
# targets, a horizon and a condition on such a record would let any consumer
# that forgets to filter read it as a scored prediction. So those fields are
# moved off the forecast slots: a reported view keeps them under `reported_*`
# (it is somebody's forecast, just not the speaker's), a hypothetical under
# `hypothetical_*`, and the rest (question, retrospective, disclosure, fact)
# in `displaced_fields`. The forecast slots themselves are cleared, the
# claim is not forward-looking and not testable, and carries no stance or
# recommendation. `non_view_invariant_violations` states the invariant.

FORECAST_FIELDS = (
    "forecast_metric", "forecast_direction", "target_kind", "target_value", "target_low", "target_high",
    "target_unit", "baseline_value", "expected_change_value", "expected_change_unit", "horizon_original",
    "horizon_bucket", "forecast_start_date", "forecast_end_date", "condition", "condition_text",
    "condition_observable", "condition_kind", "condition_status", "condition_evaluation_date",
    "condition_evidence", "condition_data_source", "trigger", "stance", "stance_basis",
    "recommendation_action",
)
_FORECAST_DEFAULTS = {"target_kind": "none", "stance": "not_applicable", "stance_basis": "not_applicable",
                      "recommendation_action": "none"}
NON_VIEW_PREFIX = {"third_party_view": "reported_", "news_report": "reported_", "hypothetical": "hypothetical_"}
DISPLACED_CLAIM_TYPES = {"question", "historical_claim", "portfolio_disclosure", "fact"}


def _normalize_non_view(claim, raw_stance=None, raw_action=None):
    """Move forecast-implying fields off a non-view claim (see above).
    `raw_stance` / `raw_action` are what the model said before the
    attribution rules cleared them, kept as the REPORTED stance of a
    third-party view."""
    claim_type = claim.get("claim_type")
    prefix = NON_VIEW_PREFIX.get(claim_type)
    if prefix is None and claim_type not in DISPLACED_CLAIM_TYPES:
        return claim
    moved = {}
    for field in FORECAST_FIELDS:
        value = claim.get(field)
        default = _FORECAST_DEFAULTS.get(field)
        if value not in (None, default):
            moved[field] = value
        claim[field] = default
    if prefix:
        if raw_stance in ("bullish", "bearish", "neutral", "mixed"):
            moved.setdefault("stance", raw_stance)
        if raw_action not in (None, "none", "unclear"):
            moved.setdefault("recommendation_action", raw_action)
        for field, value in moved.items():
            claim[prefix + field] = value
        claim["displaced_fields"] = None
    else:
        claim["displaced_fields"] = moved or None
    claim["is_forward_looking"] = False
    claim["testable"] = False
    claim["testability_type"] = "not_testable"
    return claim


def non_view_invariant_violations(claim):
    """
    The schema invariant for non-view claims: a question, disclosure,
    retrospective, reported view, news item, hypothetical or fact carries no
    stance, recommendation, forecast direction, target, horizon, dates or
    condition on the forecast slots, is not forward-looking and not
    testable. Returns the violated field names ([] when the invariant holds
    or the claim is a view-bearing type).
    """
    claim_type = claim.get("claim_type")
    if claim_type not in NON_VIEW_PREFIX and claim_type not in DISPLACED_CLAIM_TYPES:
        return []
    bad = [f for f in FORECAST_FIELDS if claim.get(f) not in (None, _FORECAST_DEFAULTS.get(f))]
    if claim.get("is_forward_looking"):
        bad.append("is_forward_looking")
    if claim.get("testable") or claim.get("testability_type") != "not_testable":
        bad.append("testability_type")
    return bad


def _slot(claim, field):
    """The value of a forecast slot, wherever normalization put it: the
    plain field on a view claim; the reported_* / hypothetical_* field or
    the displaced_fields entry on a non-view claim (else the cleared slot)."""
    claim_type = claim.get("claim_type")
    if claim_type in NON_VIEW_PREFIX or claim_type in DISPLACED_CLAIM_TYPES:
        prefix = NON_VIEW_PREFIX.get(claim_type)
        if prefix and claim.get(prefix + field) is not None:
            return claim[prefix + field]
        displaced = claim.get("displaced_fields") or {}
        if field in displaced:
            return displaced[field]
    return claim.get(field)


# --- Validation ---------------------------------------------------------------


def claim_id_for(video_id, run_key, claim):
    basis = "|".join(str(_slot(claim, k) if k in FORECAST_FIELDS else claim.get(k)) for k in (
        "chunk_id", "evidence_start_character", "subject_mention", "claim_type", "stance",
        "forecast_metric", "forecast_direction", "target_value", "target_low", "target_high",
        "horizon_bucket", "condition", "attributed_person_or_organization",
    ))
    return "clm_" + hashlib.sha1(f"{video_id}|{run_key}|{basis}".encode("utf-8")).hexdigest()[:20]


def _dedup_key(claim):
    return (
        (claim.get("ticker") or _norm_name(claim.get("subject_mention"))),
        claim.get("claim_type"), _slot(claim, "stance"), _slot(claim, "forecast_metric"),
        _slot(claim, "forecast_direction"), _slot(claim, "target_value"), _slot(claim, "target_low"),
        _slot(claim, "target_high"), _slot(claim, "horizon_bucket"),
        _norm_name(_slot(claim, "condition")) or None,
        _norm_name(claim.get("attributed_person_or_organization")) or None,
    )


def validate_claims(raw_claims, nt, context, coverage_status="full", chunk_id=None):
    """
    Turn model candidates into canonical claim records for one extraction run.

    `nt` is the NormalizedTranscript; `context` carries video_id, channel_id,
    channel_name, video_title, published_at, transcript_source,
    extraction_model, run_key. Returns (claims, warnings). Candidates that
    fail a hard check (no locatable evidence, no subject and no market
    statement) are kept as records with review_required=True — nothing is
    silently discarded and nothing unsupported is silently kept.
    """
    claims, warnings, seen = [], [], {}
    published = _parse_date(context.get("published_at"))
    text = nt.text
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    for i, raw in enumerate(raw_claims or []):
        if not isinstance(raw, dict):
            warnings.append(f"claim {i}: not an object")
            continue
        review = list(_strlist(raw.get("review_reasons")))
        issues = []
        evidence = _s(raw.get("evidence_text")) or ""
        span = locate_evidence(evidence, text)
        seg = nt.segment_at(span[0]) if span else None
        if span is None:
            review.append("evidence_not_found")
            issues.append("unsupported_evidence")

        attribution = _enum(raw.get("attribution_type"), ATTRIBUTION_TYPES, "unclear")
        claim_type = _enum(raw.get("claim_type"), CLAIM_TYPES, "other")
        forward = bool(raw.get("is_forward_looking")) if isinstance(raw.get("is_forward_looking"), bool) \
            else claim_type in ("forecast", "price_target")
        stance = raw_stance = _enum(raw.get("stance"), STANCES, "unclear")
        action = raw_action = _enum(raw.get("recommendation_action"), ACTIONS, "none")
        certainty = _enum(raw.get("certainty_level"), CERTAINTY, "not_stated")
        condition = _s(raw.get("condition"))

        # Rule: a question is not a forecast.
        if attribution == "interviewer_question" or _is_question(evidence):
            if claim_type in ("forecast", "price_target", "recommendation"):
                warnings.append(f"claim {i}: question reclassified from {claim_type}")
            claim_type, forward = "question", False
            if attribution != "interviewer_question":
                attribution = "interviewer_question"
            stance = "not_applicable"
            action = "none"
        # Rule: a historical statement is not a new forecast.
        if attribution == "retrospective_claim" or claim_type == "historical_claim" or _RETRO_RE.search(evidence):
            if claim_type in ("forecast", "price_target"):
                claim_type = "historical_claim"
            if attribution not in ("retrospective_claim",):
                attribution = "retrospective_claim"
            forward = False
        # Rule: a reported third-party view is not the speaker's view unless
        # the speaker explicitly adopts it. A reported view needs no review —
        # it is excluded from own-view analytics by its attribution — but an
        # own-view claim whose evidence reads like a quotation does.
        host_position = "not_applicable"
        third_party_origin = None
        if attribution in OWN_VIEW_ATTRIBUTIONS and _THIRD_PARTY_RE.search(evidence) \
                and not _ADOPT_RE.search(evidence):
            review.append("possible_third_party_view")
        if attribution in THIRD_PARTY_ATTRIBUTIONS or claim_type == "third_party_view":
            host_position = host_position_from(evidence, raw.get("host_position"))
            third_party_origin = _s(raw.get("attributed_person_or_organization"))
            if host_position == "adopted":
                # Adoption makes it the speaker's own current view; the
                # origin is kept so the provenance is never lost.
                attribution = "speaker_personal_view"
                if claim_type == "third_party_view":
                    claim_type = "price_target" if any(
                        _num(raw.get(k)) is not None for k in ("target_value", "target_low", "target_high")
                    ) else "forecast"
            else:
                if claim_type in ("forecast", "price_target", "recommendation", "stance",
                                  "valuation_view", "opinion"):
                    claim_type = "third_party_view"
                stance = "not_applicable"
                action = "none"
        # Rules: ownership and praise are not recommendations.
        if action not in ("none", "unclear") and not _RECOMMEND_RE.search(evidence):
            warnings.append(f"claim {i}: recommendation '{action}' not stated in evidence; set to none")
            action = "none"
        if attribution == "hypothetical_example" or claim_type == "hypothetical":
            forward = False

        # Numbers must be in the evidence.
        target_value, target_low, target_high = _num(raw.get("target_value")), _num(raw.get("target_low")), _num(raw.get("target_high"))
        baseline, change = _num(raw.get("baseline_value")), _num(raw.get("expected_change_value"))
        direction = _enum(raw.get("forecast_direction"), DIRECTIONS, None)

        # Rule: a portfolio disclosure is not a view. "I own Tesla" carries
        # no stance, no recommendation and no forecast; only a statement that
        # ALSO forecasts or recommends keeps its view (with the disclosure
        # noted beside it).
        supported_position = portfolio_position(evidence)
        portfolio = _enum(raw.get("portfolio_disclosure"), PORTFOLIO, "not_stated")
        if supported_position:
            portfolio = supported_position
        elif portfolio in ("long", "short", "no_position"):
            portfolio = "owns_unspecified" if _OWNERSHIP_RE.search(evidence) else "not_stated"
        view_content = (
            claim_type in ("forecast", "price_target", "recommendation", "valuation_view")
            or action not in ("none", "unclear") or direction is not None
            or any(v is not None for v in (target_value, target_low, target_high))
            or bool(_RECOMMEND_RE.search(evidence) and action != "none")
        )
        if claim_type == "portfolio_disclosure" or (
                supported_position and claim_type in ("stance", "opinion", "other", "fact") and not view_content):
            if claim_type != "portfolio_disclosure":
                warnings.append(f"claim {i}: ownership-only statement reclassified from {claim_type} "
                                "to portfolio_disclosure")
            claim_type = "portfolio_disclosure"
            stance, action, forward = "not_applicable", "none", False
            if portfolio in ("not_stated", "unclear"):
                portfolio = "owns_unspecified"
        for label, value in (("target_value", target_value), ("target_low", target_low),
                             ("target_high", target_high), ("baseline_value", baseline),
                             ("expected_change_value", change)):
            if value is not None and not number_supported(value, evidence):
                review.append(f"number_not_in_evidence:{label}")
                issues.append("unsupported_evidence")
        target_kind = _enum(raw.get("target_kind"), TARGET_KINDS, None)
        if target_kind == "range" and (target_low is None or target_high is None):
            review.append("range_missing_bound")
        if target_kind is None:
            target_kind = "range" if target_low is not None and target_high is not None \
                else "absolute_value" if target_value is not None else "none"
        target_unit = _s(raw.get("target_unit"))
        currency = _s(raw.get("currency"))
        if target_kind == "percentage_change" and target_unit is None:
            target_unit = "percent"
        if target_kind == "absolute_value" and currency is None and "$" in evidence:
            currency = "USD"

        # Entity resolution. A generic subject ("the stock", "it") resolves
        # through LOCAL coreference only: the one active subject of the
        # surrounding sentences, never a guess between two, never knowledge.
        asset_type = _enum(raw.get("asset_type"), ASSET_TYPES, None)
        subject = _s(raw.get("subject_mention"))
        resolution_method, resolution_confidence = "unresolved", 0.0
        coref_status = None
        if _is_generic(subject):
            resolved_name, coref_status = local_subject(nt, span, evidence)
            if coref_status == "resolved":
                warnings.append(f"claim {i}: '{subject}' resolved to '{resolved_name}' from local context")
                subject = resolved_name
                resolution_method, resolution_confidence = "local_coreference", 0.7
        ent = resolve_entity(subject, raw.get("ticker_spoken"), evidence, context.get("video_title"), asset_type,
                             channel=context.get("channel_name"))
        if coref_status == "ambiguous":
            ent.update(entity_resolution_status="ambiguous", ticker=None, ticker_source="unresolved",
                       review_reason="ambiguous_coreference", resolution_method_hint=None)
        if ent["review_reason"]:
            review.append(ent["review_reason"])
            if ent["review_reason"] == "missing_asset":
                issues.append("missing_asset")
            elif ent["review_reason"] in ("ambiguous_entity", "ambiguous_coreference"):
                issues.append("ambiguous_entity")
        if ent["resolution_method_hint"] == "title" and resolution_method == "unresolved":
            # A generic or metric subject that took the title's asset is a
            # title coreference, never an "explicit mention" of "the company".
            resolution_method = "title"
            resolution_confidence = 0.6 if ent["entity_resolution_status"] == "confirmed" else 0.5
        # The asset mention must be supported: in the evidence itself, or at
        # least in the segment the evidence sits in ("the stock" one sentence
        # after the name is normal speech, a name from nowhere is not). A
        # reference ("the company") names nothing, so it has nothing to find.
        if subject and span is not None and resolution_method == "unresolved" and not _is_generic(subject):
            folded_subject = _fold(subject)[0]
            in_evidence = folded_subject and folded_subject in _fold(evidence)[0]
            in_segment = seg is not None and folded_subject and folded_subject in _fold(seg.normalized_text)[0]
            in_title = folded_subject and folded_subject in _fold(context.get("video_title") or "")[0]
            if in_evidence:
                resolution_method = "explicit_mention"
                resolution_confidence = 1.0 if ent["entity_resolution_status"] == "confirmed" else 0.8
            elif in_segment:
                resolution_method, resolution_confidence = "local_coreference", 0.7
            elif in_title:
                resolution_method, resolution_confidence = "title", 0.6
            elif ent["entity_resolution_status"] in ("confirmed", "probable"):
                resolution_method, resolution_confidence = "curated_mapping", 0.5
                review.append("subject_not_in_evidence")
            else:
                review.append("subject_not_in_evidence")
        if ent["entity_resolution_status"] in ("ambiguous", "unresolved") and coref_status != "resolved":
            resolution_method, resolution_confidence = "unresolved", 0.0
        elif span is None and ent["ticker_source"] == "title" and resolution_method == "unresolved":
            resolution_method, resolution_confidence = "title", 0.6

        # Horizon.
        horizon_original = _s(raw.get("horizon_original"))
        start_d, end_d, bucket, h_issue = resolve_horizon(horizon_original, published)
        if forward and h_issue:
            issues.append(h_issue)

        # Testability (forward-looking claims only). A condition is not by
        # itself disqualifying: an objectively observable one makes the
        # claim conditional_testable; only an unobservable one blocks it.
        metric = _s(raw.get("forecast_metric"))
        observable, condition_kind, condition_source = condition_observability(condition)
        if forward:
            if not metric and target_kind in ("none", "qualitative", None):
                issues.append("missing_metric")
            if direction is None and target_value is None and target_low is None:
                issues.append("missing_direction")
            if target_kind == "qualitative":
                issues.append("purely_qualitative")
            if condition and not observable:
                issues.append("condition_not_objectively_observable")
            if coverage_status == "partial":
                issues.append("incomplete_transcript_coverage")
            if ent["ticker"] is None and asset_type in (None, "stock", "crypto", "etf"):
                issues.append("missing_asset" if not subject else "ambiguous_entity"
                              if ent["entity_resolution_status"] == "ambiguous" else "unresolved_entity")
        issues = list(dict.fromkeys(issues))
        if not forward or issues or span is None:
            testability_type = "not_testable"
        elif condition:
            testability_type = "conditional_testable"
        else:
            testability_type = "unconditional_testable"
        testable = testability_type != "not_testable"

        review_required = bool(review) or span is None or ent["entity_resolution_status"] == "ambiguous" \
            or _enum(raw.get("extraction_confidence"), EXTRACTION_CONFIDENCE, "medium") == "low"
        review = list(dict.fromkeys(review))
        evidence_start_seconds = nt.seconds_at(span[0]) if span and hasattr(nt, "seconds_at") else None
        evidence_end_seconds = nt.end_seconds_at(max(span[0], span[1] - 1)) if span and hasattr(nt, "end_seconds_at") else None
        if evidence_start_seconds is None and seg is not None:
            evidence_start_seconds, evidence_end_seconds = seg.start_seconds, seg.end_seconds

        claim = {
            "schema_version": SCHEMA_VERSION,
            "claim_id": None,
            "video_id": context.get("video_id"),
            "channel_id": context.get("channel_id"),
            "channel_name": context.get("channel_name"),
            "video_title": context.get("video_title"),
            "published_at": context.get("published_at") or None,
            "transcript_source": context.get("transcript_source"),
            "transcript_language": context.get("transcript_language"),
            "transcript_hash": nt.transcript_hash,
            "normalization_version": nt.normalization_version,
            "segment_id": seg.segment_id if seg else None,
            "chunk_id": chunk_id,
            "speaker": _s(raw.get("speaker")) or (seg.speaker if seg and seg.speaker != "unknown" else "unknown"),
            "speaker_confidence": _enum(raw.get("speaker_confidence"), SPEAKER_CONFIDENCE, "unknown"),
            "attribution_type": attribution,
            "attributed_person_or_organization": _s(raw.get("attributed_person_or_organization")),
            "host_position": host_position,
            "third_party_origin": third_party_origin if attribution in OWN_VIEW_ATTRIBUTIONS and host_position == "adopted" else None,
            "claim_type": claim_type,
            "is_forward_looking": forward,
            "subject_mention": _s(raw.get("subject_mention")),
            "canonical_entity_name": ent["canonical_entity_name"],
            "entity_resolution_method": resolution_method,
            "entity_resolution_confidence": resolution_confidence,
            "ticker_spoken": (_s(raw.get("ticker_spoken")) or "").lstrip("$").upper() or None,
            "ticker": ent["ticker"],
            "ticker_source": ent["ticker_source"],
            "exchange": None,
            "asset_type": asset_type,
            "sector": _s(raw.get("sector")),
            "entity_resolution_status": ent["entity_resolution_status"],
            "subject_kind": ent["subject_kind"],
            "resolution_notes": list(ent["resolution_notes"]),
            "resolution_version": RESOLUTION_VERSION,
            "benchmark_name": _s(raw.get("benchmark_name")),
            "benchmark_ticker": None,
            "stance": stance,
            "stance_basis": _enum(raw.get("stance_basis"), STANCE_BASIS,
                                  "not_applicable" if stance in ("not_applicable", "unclear") else "explicit"),
            "recommendation_action": action,
            "forecast_metric": metric,
            "forecast_direction": direction,
            "target_kind": target_kind,
            "target_value": target_value,
            "target_low": target_low,
            "target_high": target_high,
            "target_unit": target_unit,
            "currency": currency,
            "baseline_value": baseline,
            "expected_change_value": change,
            "expected_change_unit": _s(raw.get("expected_change_unit")),
            "horizon_original": horizon_original,
            "horizon_bucket": bucket,
            "forecast_start_date": start_d.isoformat() if start_d else None,
            "forecast_end_date": end_d.isoformat() if end_d else None,
            "condition": condition,
            "condition_text": condition,
            "condition_observable": bool(observable) if condition else None,
            "condition_kind": condition_kind if condition else None,
            "condition_status": "not_evaluated" if condition else None,
            "condition_evaluation_date": None,
            "condition_evidence": None,
            "condition_data_source": condition_source if condition else None,
            "trigger": _s(raw.get("trigger")),
            "certainty_original": _s(raw.get("certainty_original")),
            "certainty_level": certainty,
            "reasoning_summary": _s(raw.get("reasoning_summary")),
            "catalysts": _strlist(raw.get("catalysts")),
            "risks": _strlist(raw.get("risks")),
            "assumptions": _strlist(raw.get("assumptions")),
            "counterarguments": _strlist(raw.get("counterarguments")),
            "portfolio_disclosure": portfolio,
            "evidence_text": evidence,
            "evidence_start_seconds": evidence_start_seconds,
            "evidence_end_seconds": evidence_end_seconds,
            "evidence_start_character": span[0] if span else None,
            "evidence_end_character": span[1] if span else None,
            "testable": testable,
            "testability_type": testability_type,
            "testability_issues": issues,
            "extraction_confidence": _enum(raw.get("extraction_confidence"), EXTRACTION_CONFIDENCE, "medium"),
            "review_required": review_required,
            "review_reasons": review,
            "repeat_of_claim_id": None,
            "coverage_status": coverage_status if coverage_status in COVERAGE else "partial",
            "extraction_model": context.get("extraction_model"),
            "extraction_prompt_version": EXTRACTION_PROMPT_VERSION,
            "extracted_at": now,
        }
        _normalize_non_view(claim, raw_stance, raw_action)
        claim["claim_id"] = claim_id_for(context.get("video_id"), context.get("run_key"), claim)
        key = _dedup_key(claim)
        if key in seen:
            claim["repeat_of_claim_id"] = seen[key]
        else:
            seen[key] = claim["claim_id"]
        claims.append(claim)
    return claims, warnings


def dedupe_across_chunks(claims):
    """Mark claims repeated across overlapping chunks: same content key, the
    first occurrence (by evidence offset) stays primary."""
    seen = {}
    for claim in sorted(claims, key=lambda c: (c.get("evidence_start_character") is None,
                                               c.get("evidence_start_character") or 0)):
        key = _dedup_key(claim)
        if key in seen and seen[key] != claim["claim_id"]:
            claim["repeat_of_claim_id"] = seen[key]
        else:
            seen.setdefault(key, claim["claim_id"])
    return claims


def is_headline_claim(claim):
    """The primary-analytics filter: active, reviewed-clean, fully covered,
    evidence-located, and the source's own view."""
    return (
        not claim.get("review_required")
        and claim.get("coverage_status") in ("full", "chunked_full")
        and claim.get("evidence_start_character") is not None
        and claim.get("attribution_type") in OWN_VIEW_ATTRIBUTIONS
        and not claim.get("repeat_of_claim_id")
        and not claim.get("superseded")
        and claim.get("schema_version") != "legacy"
    )


def carries_view(claim):
    """
    True when a claim is an investment VIEW that stance analytics may count:
    headline-eligible, of a view-bearing claim type, with a bullish / bearish
    / neutral / mixed stance. A portfolio disclosure, a question, a third
    party's view or a retrospective never passes, whatever stance it carries.
    """
    return (
        is_headline_claim(claim)
        and claim.get("claim_type") in VIEW_CLAIM_TYPES
        and claim.get("stance") in ("bullish", "bearish", "neutral", "mixed")
    )


def record_condition_outcome(claim, status, evaluation_date=None, evidence=None, data_source=None):
    """Apply an observed condition outcome to a conditional claim (in
    memory). Returns the claim; an unknown status is recorded as 'unknown'."""
    claim["condition_status"] = status if status in CONDITION_STATUSES else "unknown"
    claim["condition_evaluation_date"] = evaluation_date
    claim["condition_evidence"] = evidence
    if data_source:
        claim["condition_data_source"] = data_source
    return claim


# --- Compatibility view -------------------------------------------------------

LEGACY_ACTION = {"buy": "buy", "add": "buy", "accumulate": "buy", "hold": "hold",
                 "reduce": "sell", "sell": "sell", "short": "sell", "avoid": "none",
                 "watch": "watch", "none": "none", "unclear": "none"}
LEGACY_CONVICTION = {"high": "high", "medium": "medium", "low": "low",
                     "not_stated": "unspecified", "unclear": "unspecified"}
LEGACY_TYPE = {"stock": "stock", "crypto": "crypto", "etf": "etf", "index": "index",
               "commodity": "commodity", "macro": "macro"}
_BUCKET_ORDER = {"short": 0, "medium": 1, "long": 2, "unspecified": 3}


def claims_to_legacy_signals(claims):
    """
    Reduce validated claims to the legacy `signals` object consumed by
    market_pulse / channel_scorecard / pulse_charts. Documented reductions:
      - only the source's own current views (OWN_VIEW_ATTRIBUTIONS) that are
        not repeats and not review-required contribute; third-party views,
        questions, retrospectives and hypotheticals are left out;
      - one entry per asset: stance = the stance of the SHORTEST-horizon
        directional claim (the legacy shape allows one), with
        `reduced: "conflicting_horizons"` and every claim id when horizons
        disagree; neutral when no directional claim exists;
      - conviction from certainty_level; action from recommendation_action
        (add/accumulate -> buy, reduce/short -> sell, avoid -> none);
      - price_target from an absolute target, or the midpoint of a range
        (`reduced: "range_midpoint"`); horizon = bucket.
    Returns None when no claim is usable, so callers never mistake a failed
    or empty extraction for "no assets".
    """
    usable = [c for c in claims if carries_view(c) and c.get("subject_mention")]
    if not claims:
        return {"assets": [], "market_sentiment": "neutral", "topics": [], "derived_from": "claims"}
    groups = {}
    for c in usable:
        key = c.get("ticker") or _norm_name(c.get("subject_mention"))
        groups.setdefault(key, []).append(c)
    assets = []
    for key, group in groups.items():
        group = sorted(group, key=lambda c: (_BUCKET_ORDER.get(c.get("horizon_bucket"), 3),
                                             c.get("evidence_start_character") or 0))
        directional = [c for c in group if c["stance"] in ("bullish", "bearish")]
        stance_set = {c["stance"] for c in directional}
        lead = directional[0] if directional else group[0]
        stance = lead["stance"] if directional else "neutral"
        entry = {
            "name": lead.get("canonical_entity_name") or lead.get("subject_mention"),
            "ticker": lead.get("ticker"),
            "type": LEGACY_TYPE.get(lead.get("asset_type"), "other"),
            "stance": stance,
            "conviction": LEGACY_CONVICTION.get(lead.get("certainty_level"), "unspecified"),
            "action": next((LEGACY_ACTION.get(c["recommendation_action"], "none") for c in group
                            if LEGACY_ACTION.get(c["recommendation_action"], "none") != "none"), "none"),
            "catalysts": list(dict.fromkeys(x for c in group for x in c.get("catalysts", [])))[:6],
            "price_target": None,
            "horizon": lead.get("horizon_bucket") or "unspecified",
            "claim_ids": [c["claim_id"] for c in group],
            "reduced": None,
        }
        if len(stance_set) > 1:
            entry["reduced"] = "conflicting_horizons:" + ",".join(
                f"{c.get('horizon_bucket')}={c['stance']}" for c in directional)
        for c in group:
            if c.get("currency") not in (None, "USD"):
                continue
            if c.get("target_kind") == "absolute_value" and c.get("target_value") is not None:
                entry["price_target"] = c["target_value"]
                break
            if c.get("target_kind") == "range" and c.get("target_low") is not None and c.get("target_high") is not None:
                entry["price_target"] = (c["target_low"] + c["target_high"]) / 2.0
                entry["reduced"] = (entry["reduced"] + ";" if entry["reduced"] else "") + "range_midpoint"
                break
        assets.append(entry)
    bulls = sum(1 for c in usable if c["stance"] == "bullish")
    bears = sum(1 for c in usable if c["stance"] == "bearish")
    sentiment = "mixed" if bulls and bears else "bullish" if bulls else "bearish" if bears else "neutral"
    return {"assets": assets, "market_sentiment": sentiment, "topics": [], "derived_from": "claims"}


def legacy_record_to_claims(record):
    """
    Import one legacy data/signals.jsonl row as schema_version="legacy" claims:
    no evidence (unavailable, not fabricated), review_required=True, excluded
    from evidence-dependent analytics, retained for history.
    """
    signals = record.get("signals")
    if not isinstance(signals, dict):
        return []
    out = []
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    for i, asset in enumerate(signals.get("assets") or []):
        if not isinstance(asset, dict) or not asset.get("name"):
            continue
        stance = asset.get("stance") if asset.get("stance") in ("bullish", "bearish", "neutral") else "unclear"
        basis = f"{record.get('video_id')}|legacy|{i}|{asset.get('name')}|{stance}"
        out.append({
            "schema_version": "legacy",
            "claim_id": "clm_" + hashlib.sha1(basis.encode("utf-8")).hexdigest()[:20],
            "video_id": record.get("video_id"), "channel_id": record.get("channel_id"),
            "channel_name": record.get("channel_name"), "video_title": record.get("video_title"),
            "published_at": record.get("published_at") or None,
            "transcript_source": None, "transcript_hash": None, "normalization_version": None,
            "segment_id": None, "chunk_id": None,
            "speaker": "unknown", "speaker_confidence": "unknown",
            "attribution_type": "unclear", "attributed_person_or_organization": None,
            "claim_type": "stance", "is_forward_looking": stance in ("bullish", "bearish"),
            "subject_mention": asset.get("name"), "canonical_entity_name": asset.get("name"),
            "ticker_spoken": None, "ticker": (asset.get("ticker") or None),
            "ticker_source": "unresolved" if not asset.get("ticker") else "spoken",
            "exchange": None, "asset_type": asset.get("type") if asset.get("type") in ASSET_TYPES else None,
            "sector": None, "entity_resolution_status": "probable" if asset.get("ticker") else "unresolved",
            "benchmark_name": None, "benchmark_ticker": None,
            "stance": stance, "stance_basis": "not_applicable",
            "recommendation_action": asset.get("action") if asset.get("action") in ACTIONS else "none",
            "forecast_metric": "price" if asset.get("price_target") is not None else None,
            "forecast_direction": None,
            "target_kind": "absolute_value" if asset.get("price_target") is not None else "none",
            "target_value": asset.get("price_target"), "target_low": None, "target_high": None,
            "target_unit": None, "currency": None, "baseline_value": None,
            "expected_change_value": None, "expected_change_unit": None,
            "horizon_original": None, "horizon_bucket": asset.get("horizon") or "unspecified",
            "forecast_start_date": None, "forecast_end_date": None,
            "condition": None, "trigger": None, "certainty_original": None,
            "certainty_level": {"high": "high", "medium": "medium", "low": "low"}.get(
                asset.get("conviction"), "not_stated"),
            "reasoning_summary": None, "catalysts": list(asset.get("catalysts") or []),
            "risks": [], "assumptions": [], "counterarguments": [],
            "portfolio_disclosure": "not_stated",
            "evidence_text": None, "evidence_start_seconds": None, "evidence_end_seconds": None,
            "evidence_start_character": None, "evidence_end_character": None,
            "testable": False, "testability_issues": ["unsupported_evidence"],
            "extraction_confidence": "low", "review_required": True,
            "review_reasons": ["legacy_import_no_evidence"], "repeat_of_claim_id": None,
            "coverage_status": "partial", "extraction_model": None,
            "extraction_prompt_version": "legacy", "extracted_at": now,
        })
    return out
