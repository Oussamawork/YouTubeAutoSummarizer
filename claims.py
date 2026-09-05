"""
Atomic, evidence-backed claims: the canonical research unit.

One claim is one independent statement — one asset, one metric, one
direction, one target, one horizon, one condition. "Bearish on Nvidia for
three months but bullish over five years" is two claims, never stance=mixed.
The model proposes candidates; THIS module decides what enters the dataset:
it locates each claim's verbatim evidence in the normalized transcript,
checks that every number the claim carries appears in that evidence, refuses
tickers the speaker did not say, resolves entities through the curated
tables only, turns relative horizons into dates only under documented rules,
and computes testability deterministically. Anything it cannot verify stays
in the record with review_required=True and out of headline analytics.

The compatibility view (`claims_to_legacy_signals`) reduces validated claims
back into the shape data/signals.jsonl consumers read, and marks every place
information was reduced.
"""
import hashlib
import json
import re
from datetime import date, datetime, timedelta, timezone

from log import log_info, log_warn
from signals_data import ASSET_ALIASES, TICKER_ALIASES, learned_tickers

SCHEMA_VERSION = "1"
EXTRACTION_PROMPT_VERSION = "1"

# --- Enumerations -----------------------------------------------------------

ATTRIBUTION_TYPES = {
    "speaker_personal_view", "guest_personal_view", "speaker_reporting_facts",
    "speaker_reporting_news", "speaker_quoting_third_party",
    "speaker_describing_market_consensus", "interviewer_question",
    "hypothetical_example", "retrospective_claim", "sarcasm_or_humor", "unclear",
}
# Attributions that count as the source's own current view in analytics.
OWN_VIEW_ATTRIBUTIONS = {"speaker_personal_view", "guest_personal_view"}
CLAIM_TYPES = {
    "forecast", "price_target", "recommendation", "stance", "valuation_view",
    "fact", "news_report", "third_party_view", "historical_claim",
    "portfolio_disclosure", "question", "hypothetical", "opinion", "risk",
    "catalyst", "other",
}
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
    "(speaker_quoting_third_party / speaker_describing_market_consensus) unless "
    "the speaker explicitly adopts it.\n"
    "- \"Last year I said X\" is retrospective_claim / historical_claim, not a "
    "new forecast.\n"
    "- Owning a stock is portfolio_disclosure, not a recommendation; praise "
    "without an explicit buy/sell/hold instruction is opinion with "
    "recommendation_action=none. Only an explicit instruction gets "
    "buy/sell/hold/etc.\n"
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
    "target_kind:absolute_value, target_value:200, currency:USD, stance:not_applicable}.\n"
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
_ADOPT_RE = re.compile(r"\b(i agree|i think so too|i share|my target|i also (think|expect|see))\b",
                       re.IGNORECASE)
_RECOMMEND_RE = re.compile(
    r"\b(buy|buying|bought|sell|selling|sold|accumulat\w*|short\w*|avoid\w*|trim\w*|reduc\w*|"
    r"add(?:ing)? to|hold(?:ing)? (?:on|it|them|this|the)|recommend\w*|you should|i would|i'd)\b",
    re.IGNORECASE)
_OWNERSHIP_RE = re.compile(r"\b(i own|i hold|i'm holding|my position|in my portfolio|i have a position)\b",
                           re.IGNORECASE)
_TICKER_WORD = re.compile(r"(?<![\w$])\$?([A-Za-z]{1,6})(?![\w])")


def _norm_name(name):
    return " ".join((name or "").split()).upper()


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


def resolve_entity(subject_mention, ticker_spoken, evidence, title, asset_type=None):
    """
    Deterministic entity resolution. Returns dict(canonical_entity_name, ticker,
    ticker_source, entity_resolution_status, review_reason).

    A spoken ticker is kept only when it actually appears in the evidence or
    the title. Missing tickers come from the curated alias tables (and the
    catalogue-verified learned map); a name in AMBIGUOUS_MENTIONS is never
    resolved. Nothing is resolved on phonetic similarity.
    """
    mention = _s(subject_mention)
    name_key = _norm_name(mention)
    out = {"canonical_entity_name": mention, "ticker": None, "ticker_source": None,
           "entity_resolution_status": "unresolved", "review_reason": None}
    if not mention:
        out["review_reason"] = "missing_asset"
        return out
    if name_key in AMBIGUOUS_MENTIONS:
        out["entity_resolution_status"] = "ambiguous"
        out["review_reason"] = "ambiguous_entity"
        return out
    spoken = (_s(ticker_spoken) or "").lstrip("$").upper()
    if spoken:
        ev_words = {m.group(1).upper() for m in _TICKER_WORD.finditer(evidence or "")}
        title_words = {m.group(1).upper() for m in _TICKER_WORD.finditer(title or "")}
        if spoken in ev_words:
            out.update(ticker=TICKER_ALIASES.get(spoken, spoken), ticker_source="spoken",
                       entity_resolution_status="confirmed")
            return out
        if spoken in title_words:
            out.update(ticker=TICKER_ALIASES.get(spoken, spoken), ticker_source="title",
                       entity_resolution_status="confirmed")
            return out
        out["review_reason"] = "ticker_not_in_evidence"
    curated = ASSET_ALIASES.get(name_key) or TICKER_ALIASES.get(name_key)
    if curated:
        out.update(ticker=TICKER_ALIASES.get(curated, curated), ticker_source="curated_mapping",
                   entity_resolution_status="confirmed", canonical_entity_name=mention)
        return out
    learned = learned_tickers().get(name_key)
    if learned:
        out.update(ticker=learned, ticker_source="curated_mapping",
                   entity_resolution_status="probable")
        return out
    if asset_type in ("macro", "sector", "index", "commodity", "currency", "bond"):
        # Not a single security: no ticker is the right answer, not a gap.
        out["entity_resolution_status"] = "confirmed"
        out["ticker_source"] = "unresolved"
        return out
    out["ticker_source"] = "unresolved"
    return out


def _is_question(evidence):
    ev = (evidence or "").strip()
    return ev.endswith("?") or (bool(_QUESTION_START.match(ev)) and "?" in ev)


# --- Validation ---------------------------------------------------------------


def claim_id_for(video_id, run_key, claim):
    basis = "|".join(str(claim.get(k)) for k in (
        "chunk_id", "evidence_start_character", "subject_mention", "claim_type", "stance",
        "forecast_metric", "forecast_direction", "target_value", "target_low", "target_high",
        "horizon_bucket", "condition",
    ))
    return "clm_" + hashlib.sha1(f"{video_id}|{run_key}|{basis}".encode("utf-8")).hexdigest()[:20]


def _dedup_key(claim):
    return (
        (claim.get("ticker") or _norm_name(claim.get("subject_mention"))),
        claim.get("claim_type"), claim.get("stance"), claim.get("forecast_metric"),
        claim.get("forecast_direction"), claim.get("target_value"), claim.get("target_low"),
        claim.get("target_high"), claim.get("horizon_bucket"),
        _norm_name(claim.get("condition")) or None,
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
        stance = _enum(raw.get("stance"), STANCES, "unclear")
        action = _enum(raw.get("recommendation_action"), ACTIONS, "none")
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
        # Rule: a reported third-party target is not the speaker's view.
        if attribution in OWN_VIEW_ATTRIBUTIONS and _THIRD_PARTY_RE.search(evidence) \
                and not _ADOPT_RE.search(evidence):
            review.append("possible_third_party_view")
        if attribution in ("speaker_quoting_third_party", "speaker_describing_market_consensus",
                           "speaker_reporting_news") and claim_type in ("forecast", "price_target", "recommendation"):
            claim_type = "third_party_view"
        # Rules: ownership and praise are not recommendations.
        if action not in ("none", "unclear") and not _RECOMMEND_RE.search(evidence):
            warnings.append(f"claim {i}: recommendation '{action}' not stated in evidence; set to none")
            action = "none"
        portfolio = _enum(raw.get("portfolio_disclosure"), PORTFOLIO, "not_stated")
        if _OWNERSHIP_RE.search(evidence) and portfolio == "not_stated":
            portfolio = "owns_unspecified"
        if attribution == "hypothetical_example" or claim_type == "hypothetical":
            forward = False

        # Numbers must be in the evidence.
        target_value, target_low, target_high = _num(raw.get("target_value")), _num(raw.get("target_low")), _num(raw.get("target_high"))
        baseline, change = _num(raw.get("baseline_value")), _num(raw.get("expected_change_value"))
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

        # Entity resolution.
        asset_type = _enum(raw.get("asset_type"), ASSET_TYPES, None)
        ent = resolve_entity(raw.get("subject_mention"), raw.get("ticker_spoken"), evidence,
                             context.get("video_title"), asset_type)
        if ent["review_reason"]:
            review.append(ent["review_reason"])
            if ent["review_reason"] == "missing_asset":
                issues.append("missing_asset")
            elif ent["review_reason"] == "ambiguous_entity":
                issues.append("ambiguous_entity")
        # The asset mention must be supported: in the evidence itself, or at
        # least in the segment the evidence sits in ("the stock" one sentence
        # after the name is normal speech, a name from nowhere is not).
        subject = _s(raw.get("subject_mention"))
        if subject and span is not None:
            folded_subject = _fold(subject)[0]
            in_evidence = folded_subject and folded_subject in _fold(evidence)[0]
            in_segment = seg is not None and folded_subject and folded_subject in _fold(seg.normalized_text)[0]
            if not (in_evidence or in_segment):
                review.append("subject_not_in_evidence")

        # Horizon.
        horizon_original = _s(raw.get("horizon_original"))
        start_d, end_d, bucket, h_issue = resolve_horizon(horizon_original, published)
        if forward and h_issue:
            issues.append(h_issue)

        # Testability (forward-looking claims only).
        metric = _s(raw.get("forecast_metric"))
        direction = _enum(raw.get("forecast_direction"), DIRECTIONS, None)
        if forward:
            if not metric and target_kind in ("none", "qualitative", None):
                issues.append("missing_metric")
            if direction is None and target_value is None and target_low is None:
                issues.append("missing_direction")
            if target_kind == "qualitative":
                issues.append("purely_qualitative")
            if condition:
                issues.append("conditional_outcome_not_observable")
            if coverage_status == "partial":
                issues.append("incomplete_transcript_coverage")
            if ent["ticker"] is None and asset_type in (None, "stock", "crypto", "etf"):
                issues.append("missing_asset" if not _s(raw.get("subject_mention")) else "ambiguous_entity"
                              if ent["entity_resolution_status"] == "ambiguous" else "unresolved_entity")
        issues = list(dict.fromkeys(issues))
        testable = forward and not issues and span is not None

        review_required = bool(review) or span is None or ent["entity_resolution_status"] == "ambiguous" \
            or _enum(raw.get("extraction_confidence"), EXTRACTION_CONFIDENCE, "medium") == "low"
        review = list(dict.fromkeys(review))

        claim = {
            "schema_version": SCHEMA_VERSION,
            "claim_id": None,
            "video_id": context.get("video_id"),
            "channel_id": context.get("channel_id"),
            "channel_name": context.get("channel_name"),
            "video_title": context.get("video_title"),
            "published_at": context.get("published_at") or None,
            "transcript_source": context.get("transcript_source"),
            "transcript_hash": nt.transcript_hash,
            "normalization_version": nt.normalization_version,
            "segment_id": seg.segment_id if seg else None,
            "chunk_id": chunk_id,
            "speaker": _s(raw.get("speaker")) or (seg.speaker if seg and seg.speaker != "unknown" else "unknown"),
            "speaker_confidence": _enum(raw.get("speaker_confidence"), SPEAKER_CONFIDENCE, "unknown"),
            "attribution_type": attribution,
            "attributed_person_or_organization": _s(raw.get("attributed_person_or_organization")),
            "claim_type": claim_type,
            "is_forward_looking": forward,
            "subject_mention": _s(raw.get("subject_mention")),
            "canonical_entity_name": ent["canonical_entity_name"],
            "ticker_spoken": (_s(raw.get("ticker_spoken")) or "").lstrip("$").upper() or None,
            "ticker": ent["ticker"],
            "ticker_source": ent["ticker_source"],
            "exchange": None,
            "asset_type": asset_type,
            "sector": _s(raw.get("sector")),
            "entity_resolution_status": ent["entity_resolution_status"],
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
            "evidence_start_seconds": seg.start_seconds if seg else None,
            "evidence_end_seconds": seg.end_seconds if seg else None,
            "evidence_start_character": span[0] if span else None,
            "evidence_end_character": span[1] if span else None,
            "testable": testable,
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
    )


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
    usable = [c for c in claims if is_headline_claim(c)
              and c.get("subject_mention") and c.get("stance") in ("bullish", "bearish", "neutral", "mixed")]
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
