"""
Deterministic transcript normalization, segmentation and complete-coverage
chunking. No model is involved anywhere in this module.

Normalization repairs caption FORMATTING, never the speaker's words: line
endings, whitespace, broken caption lines, the rolling overlap that
auto-captions produce between adjacent cues. Every number, percentage,
currency, range, date, negation, hedge and condition is preserved verbatim;
so is legitimate repetition ("sell, sell, sell"). The raw text is kept
alongside, and every normalized character maps back to a raw offset, so a
claim's evidence can always be traced to the source.

Segments are stable, ordered spans with character offsets (and seconds when
the source carried timestamps). They are classified with a keyword heuristic
into the categories analytics need to include or exclude (sponsor reads,
disclaimers, intros); the classification is a hint for filtering, and the
claim extractor is the component that actually reads meaning.

Chunks exist for the case where a complete request genuinely does not fit the
selected model. They cover every character of the normalized transcript,
overlap so that a statement crossing a boundary appears whole in at least one
chunk, and are validated before use — there is no head-and-tail truncation
anywhere in this pipeline any more.
"""
import hashlib
import re
from dataclasses import dataclass, field, asdict

from helpers import env_int
from signals_data import ASSET_ALIASES

NORMALIZATION_VERSION = "1"
# Version of the chunk-boundary algorithm below. Part of every partial-cache
# record: a cached chunk result is only reused when the boundaries it was
# computed for are the boundaries the current algorithm would produce.
CHUNKING_VERSION = "1"

# Segments: a new one starts on a speaker/asset/topic change once the current
# one has this much text; an explicit paragraph break needs less; none grows
# past MAX before a sentence split.
MIN_SEGMENT_CHARS = 200
MIN_PARAGRAPH_SEGMENT_CHARS = 80
MAX_SEGMENT_CHARS = 1200
# Overlap between adjacent chunks so a boundary never cuts a statement in two.
CHUNK_OVERLAP_TOKENS = env_int("CHUNK_OVERLAP_TOKENS", 300)
# Auto-caption overlap: the next cue repeating the previous cue's tail. This
# many shared words (or characters) at the seam is treated as the artifact.
OVERLAP_MIN_WORDS = 3
OVERLAP_MIN_CHARS = 12

_TS_ARROW = re.compile(
    r"^\s*(?P<h>\d{1,2}:)?(?P<m>\d{1,2}):(?P<s>\d{2})(?:[.,]\d{1,3})?\s*-->\s*"
    r"(?:\d{1,2}:)?\d{1,2}:\d{2}(?:[.,]\d{1,3})?.*$"
)
_TS_PREFIX = re.compile(
    r"^\s*[\[(]?(?P<h>\d{1,2}:)?(?P<m>\d{1,2}):(?P<s>\d{2})[\])]?\s*[-–—:]?\s*(?P<rest>.*)$"
)
_SRT_INDEX = re.compile(r"^\s*\d{1,6}\s*$")
_SPEAKER = re.compile(r"^\s*(?:>>\s*)?(?P<label>[A-Z][\w.'\- ]{0,40}?):\s+(?P<rest>\S.*)$")
_UNINTELLIGIBLE = re.compile(
    r"\[(?:inaudible|unintelligible|music|applause|laughter|__+|\?+)\]"
    r"|\((?:inaudible|unintelligible)\)|\?{3,}", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_TICKER_LIKE = re.compile(r"(?<![\w$])\$?[A-Z]{2,5}(?![\w])")
_ALIAS_NAMES = sorted(ASSET_ALIASES, key=len, reverse=True)
_ALIAS_RE = re.compile(r"\b(" + "|".join(re.escape(n) for n in _ALIAS_NAMES) + r")\b", re.IGNORECASE)
# Uppercase words that are not tickers.
_COMMON_UPPER = {"I", "A", "AI", "US", "USA", "UK", "EU", "CEO", "CFO", "IPO", "ETF", "GDP",
                 "FED", "OK", "TV", "PE", "EPS", "YOY", "QOQ", "TLDR", "AM", "PM", "THE"}

CATEGORY_KEYWORDS = {
    "sponsor_or_promotion": ["sponsor", "sponsored", "promo code", "use code", "link in the description",
                             "sign up", "patreon", "affiliate", "discount", "free trial"],
    "disclaimer": ["not financial advice", "financial advisor", "do your own research",
                   "entertainment purposes", "not investment advice", "not a recommendation"],
    "introduction_or_outro": ["welcome back", "in this video", "today we", "thanks for watching",
                              "see you in the next", "like and subscribe", "hit the like"],
    "price_target": ["price target", "target of", "could reach", "could hit", "go to $", "get to $",
                     "reach $", "hit $"],
    "forecast": ["will", "expect", "going to", "by the end of", "next year", "next quarter",
                 "over the next", "i think", "likely", "probably", "could", "should"],
    "recommendation": ["buy", "sell", "hold", "accumulate", "short", "avoid", "add to", "trim"],
    "risk": ["risk", "downside", "danger", "concern", "worried", "bear case", "threat"],
    "catalyst": ["catalyst", "earnings", "launch", "approval", "rate cut", "guidance", "announcement"],
    "portfolio_disclosure": ["i own", "my portfolio", "i bought", "i sold", "my position", "i hold"],
    "valuation": ["valuation", "p/e", "multiple", "price to earnings", "overvalued", "undervalued",
                  "market cap"],
    "macroeconomics": ["inflation", "interest rate", "gdp", "unemployment", "tariff", "the fed",
                       "recession", "cpi"],
    "news_reporting": ["reported", "announced", "according to", "headline", "reuters", "bloomberg"],
    "historical_context": ["last year", "back in", "historically", "in 2020", "in 2021", "in 2022",
                           "in 2023", "in 2024", "in 2025"],
    "market_analysis": ["the market", "s&p", "nasdaq", "index", "dow", "breadth", "sector rotation"],
    "educational_content": ["let me explain", "what is", "how does", "basically", "for those who"],
    "entertainment_or_humor": ["lol", "haha", "just kidding", "joke"],
}
# Categories that headline analytics leave out (the segments are still stored).
EXCLUDED_CATEGORIES = {"sponsor_or_promotion", "disclaimer", "introduction_or_outro",
                       "entertainment_or_humor", "off_topic"}


@dataclass
class Segment:
    segment_id: str
    video_id: str
    sequence_number: int
    start_seconds: float
    end_seconds: float
    start_character: int
    end_character: int
    speaker: str
    original_text: str
    normalized_text: str
    primary_category: str
    secondary_tags: list
    quality_flags: list
    excluded_from_headline: bool = False
    exclusion_reason: str = None

    def to_dict(self):
        return asdict(self)


@dataclass
class Chunk:
    chunk_id: str
    sequence_number: int
    start_character: int
    end_character: int
    start_seconds: float
    end_seconds: float
    input_tokens: int
    overlap_chars: int
    segment_ids: list
    text: str = field(repr=False, default="")

    def to_dict(self):
        d = asdict(self)
        d.pop("text", None)
        return d


@dataclass
class NormalizedTranscript:
    text: str
    raw_text: str
    transcript_hash: str
    normalization_version: str
    timestamps_available: bool
    segments: list
    quality_flags: dict
    offset_map: list = field(repr=False, default_factory=list)
    # Caption cues as (normalized start, normalized end, seconds-or-None):
    # the finest timing the source carried, so a claim's evidence can be
    # stamped with the cue it sits in rather than its segment's first cue.
    cues: list = field(repr=False, default_factory=list)

    @property
    def raw_char_count(self):
        return len(self.raw_text)

    @property
    def normalized_char_count(self):
        return len(self.text)

    def raw_span(self, start, end):
        """Raw-text (start, end) offsets for a normalized [start, end) span."""
        if not self.offset_map or start >= len(self.offset_map):
            return start, end
        end_idx = min(max(end - 1, start), len(self.offset_map) - 1)
        return self.offset_map[start], self.offset_map[end_idx] + 1

    def segment_at(self, char_offset):
        for seg in self.segments:
            if seg.start_character <= char_offset < seg.end_character:
                return seg
        return self.segments[-1] if self.segments and char_offset >= len(self.text) else None

    def _cue_index_at(self, char_offset):
        for i, (s, e, _) in enumerate(self.cues):
            if s <= char_offset < e:
                return i
        if self.cues and char_offset >= self.cues[-1][1]:
            return len(self.cues) - 1
        return None

    def seconds_at(self, char_offset):
        """
        Seconds of the timestamped cue containing `char_offset`. A cue that
        carried no time of its own inherits the nearest earlier timestamped
        cue. None when the source had no timing at all (plain text), never a
        guess.
        """
        if not self.timestamps_available:
            return None
        i = self._cue_index_at(char_offset)
        if i is None:
            return None
        while i >= 0:
            secs = self.cues[i][2]
            if secs is not None:
                return secs
            i -= 1
        return None

    def end_seconds_at(self, char_offset):
        """Seconds at which the cue containing `char_offset` ends: the next
        timestamped cue's start, or None at the end of the source."""
        if not self.timestamps_available:
            return None
        i = self._cue_index_at(char_offset)
        if i is None:
            return None
        for s, e, secs in self.cues[i + 1:]:
            if secs is not None:
                return secs
        return None


def transcript_hash(raw_text):
    return hashlib.sha256((raw_text or "").encode("utf-8")).hexdigest()


def _seconds(match):
    h = int((match.group("h") or "0:").rstrip(":") or 0)
    return h * 3600 + int(match.group("m")) * 60 + int(match.group("s"))


def _split_lines(raw):
    """[(raw_start, raw_end)] for each line, \\r\\n and \\r treated as \\n."""
    lines, start, i, n = [], 0, 0, len(raw)
    while i < n:
        ch = raw[i]
        if ch == "\n" or ch == "\r":
            lines.append((start, i))
            if ch == "\r" and i + 1 < n and raw[i + 1] == "\n":
                i += 1
            start = i + 1
        i += 1
    lines.append((start, n))
    return lines


def _parse_units(raw):
    """
    Turn raw text into ordered cue units: {"start","end" (raw offsets of the
    text part), "seconds", "speaker", "paragraph_break"}. Timestamp lines and
    SRT indices become metadata on the cue that follows them.
    """
    units, pending_seconds, blank_run = [], None, 0
    lines = _split_lines(raw)
    speaker_counts = {}
    for a, b in lines:
        line = raw[a:b]
        if not line.strip():
            blank_run += 1
            continue
        m = _TS_ARROW.match(line)
        if m:
            pending_seconds = _seconds(m)
            continue
        if _SRT_INDEX.match(line):
            continue
        seconds, text_start = pending_seconds, a
        m = _TS_PREFIX.match(line)
        if m and not _SPEAKER.match(line):
            seconds = _seconds(m)
            text_start = a + m.start("rest")
        pending_seconds = None
        speaker, label_start = None, text_start
        sm = _SPEAKER.match(raw[text_start:b])
        if sm and len(sm.group("label").split()) <= 3:
            speaker = sm.group("label").strip()
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
            text_start = text_start + sm.start("rest")
        units.append({"start": text_start, "end": b, "seconds": seconds, "speaker": speaker,
                      "label_start": label_start,
                      "paragraph_break": blank_run > 0 and bool(units)})
        blank_run = 0
    # A label counts as a speaker only when it recurs: a one-off "Note:" or a
    # title-cased clause before a colon is prose, not a turn marker.
    for u in units:
        if u["speaker"] and speaker_counts.get(u["speaker"], 0) < 2:
            u["start"], u["speaker"] = u["label_start"], None
    return units


def _words(text):
    return re.findall(r"\S+", text.lower())


def _strip_caption_overlap(prev_text, next_text):
    """Characters to drop from the start of `next_text` because they repeat
    the end of `prev_text` (auto-caption roll-over). 0 when nothing repeats."""
    prev_w, next_w = _words(prev_text), _words(next_text)
    if not prev_w or not next_w:
        return 0
    best = 0
    for k in range(min(len(prev_w), len(next_w)), 0, -1):
        if prev_w[-k:] == next_w[:k]:
            shared = " ".join(next_w[:k])
            if k >= OVERLAP_MIN_WORDS or len(shared) >= OVERLAP_MIN_CHARS:
                best = k
            break
    if not best:
        return 0
    # Map the k words back onto the raw string.
    count, i, n = 0, 0, len(next_text)
    while i < n and count < best:
        while i < n and next_text[i].isspace():
            i += 1
        while i < n and not next_text[i].isspace():
            i += 1
        count += 1
    while i < n and next_text[i].isspace():
        i += 1
    return i


def _is_line_structured(raw, units):
    """Caption-cue shaped input: several lines, mostly short."""
    if len(units) < 4:
        return False
    lengths = sorted(u["end"] - u["start"] for u in units)
    return lengths[len(lengths) // 2] < 160


def normalize_transcript(raw_text, video_id=""):
    """
    Build a NormalizedTranscript from raw caption text. Deterministic; never
    raises on string input.
    """
    raw = raw_text or ""
    units = _parse_units(raw)
    line_structured = _is_line_structured(raw, units)
    flags = {"caption_overlap_removed": 0, "duplicate_cues_removed": 0,
             "unintelligible_markers": 0, "lines_joined": 0}

    out, omap = [], []
    cue_spans = []  # (norm_start, norm_end, unit)
    prev_text, prev_full = "", ""
    for idx, u in enumerate(units):
        text = raw[u["start"]:u["end"]]
        skip = 0
        if line_structured and prev_full:
            # An identical adjacent cue is the caption renderer repeating
            # itself; the same sentence said twice inside ONE cue is the
            # speaker's emphasis and is never touched.
            if _words(text) == _words(prev_full) and len(_words(text)) <= 15:
                flags["duplicate_cues_removed"] += 1
                continue
            skip = _strip_caption_overlap(prev_text, text)
            if skip:
                flags["caption_overlap_removed"] += 1
        if not text[skip:].strip():
            continue
        # Separator: newline on paragraph/speaker change, else a joining space.
        if out:
            speaker_change = u["speaker"] and u["speaker"] != _last_speaker(cue_spans)
            sep = "\n" if (u["paragraph_break"] or speaker_change) else " "
            if sep == " ":
                flags["lines_joined"] += 1
            out.append(sep)
            omap.append(u["start"] + skip)
        norm_start = len(out)
        if u["speaker"] and u["speaker"] != _last_speaker(cue_spans):
            for ch in u["speaker"] + ": ":
                out.append(ch)
                omap.append(u["start"] + skip)
        # Collapse whitespace runs to one space, char by char, keeping offsets.
        i, n, in_ws, started = u["start"] + skip, u["end"], False, False
        while i < n:
            ch = raw[i]
            if ch == " ":
                ch = " "
            if ch.isspace():
                in_ws = True
            else:
                if in_ws and started:
                    out.append(" ")
                    omap.append(i - 1)
                in_ws = False
                started = True
                out.append(ch)
                omap.append(i)
            i += 1
        cue_spans.append((norm_start, len(out), u))
        prev_text, prev_full = text[skip:], text

    text = "".join(out)
    flags["unintelligible_markers"] = len(_UNINTELLIGIBLE.findall(text))
    nt = NormalizedTranscript(
        text=text, raw_text=raw, transcript_hash=transcript_hash(raw),
        normalization_version=NORMALIZATION_VERSION,
        timestamps_available=any(u["seconds"] is not None for u in units),
        segments=[], quality_flags=flags, offset_map=omap,
        cues=[(s, e, u["seconds"]) for s, e, u in cue_spans],
    )
    nt.segments = _build_segments(nt, cue_spans, video_id)
    return nt


def _last_speaker(cue_spans):
    for _, _, u in reversed(cue_spans):
        if u["speaker"]:
            return u["speaker"]
    return None


def _assets_in(text):
    found = {m.group(1).upper() for m in _ALIAS_RE.finditer(text)}
    for m in _TICKER_LIKE.finditer(text):
        tok = m.group(0).lstrip("$")
        if tok not in _COMMON_UPPER and (m.group(0).startswith("$") or len(tok) >= 3):
            found.add(tok)
    return found


def classify_text(text):
    """(primary_category, secondary_tags) from keyword scores. 'unclear' when
    nothing matches."""
    low = f" {text.lower()} "
    scores = {}
    for cat, words in CATEGORY_KEYWORDS.items():
        s = sum(low.count(f"{w}") for w in words)
        if s:
            scores[cat] = s
    if _assets_in(text) and "company_analysis" not in scores:
        scores["company_analysis"] = 1
    if not scores:
        return "unclear", []
    # Exclusion categories win when present at all: a sponsor read that also
    # says "buy" is still a sponsor read.
    for cat in ("sponsor_or_promotion", "disclaimer"):
        if scores.get(cat):
            primary = cat
            break
    else:
        primary = max(scores.items(), key=lambda kv: (kv[1], kv[0]))[0]
    secondary = sorted(c for c in scores if c != primary)
    return primary, secondary


def _sentence_spans(text, base):
    """[(start, end)] absolute spans of sentences inside `text`."""
    spans, pos = [], 0
    for m in _SENTENCE_END.finditer(text):
        spans.append((base + pos, base + m.start()))
        pos = m.end()
    if pos < len(text):
        spans.append((base + pos, base + len(text)))
    return spans or [(base, base + len(text))]


def _build_segments(nt, cue_spans, video_id):
    text = nt.text
    if not text:
        return []
    # Base units: cues when the input had line structure, sentences otherwise.
    if len(cue_spans) >= 2:
        base = [(s, e, u["seconds"], u["speaker"], u["paragraph_break"]) for s, e, u in cue_spans]
    else:
        base = [(s, e, None, None, False) for s, e in _sentence_spans(text, 0)]

    segments, cur_start, cur_assets, cur_speaker, cur_secs = [], None, set(), None, None
    cur_end = None

    def close(end, seconds_end):
        nonlocal cur_start, cur_assets
        if cur_start is None or end <= cur_start:
            return
        body = text[cur_start:end]
        seq = len(segments) + 1
        primary, secondary = classify_text(body.strip())
        qflags = []
        if _UNINTELLIGIBLE.search(body):
            qflags.append("unintelligible")
        if len(body.strip()) < 40:
            qflags.append("very_short")
        raw_s, raw_e = nt.raw_span(cur_start, end)
        seg = Segment(
            segment_id=f"{nt.transcript_hash[:12]}-s{seq:03d}", video_id=video_id or "",
            sequence_number=seq, start_seconds=cur_secs, end_seconds=seconds_end,
            start_character=cur_start, end_character=end,
            speaker=cur_speaker or "unknown", original_text=nt.raw_text[raw_s:raw_e],
            normalized_text=body, primary_category=primary, secondary_tags=secondary,
            quality_flags=qflags,
        )
        if primary in EXCLUDED_CATEGORIES:
            seg.excluded_from_headline = True
            seg.exclusion_reason = f"category:{primary}"
        segments.append(seg)
        cur_start, cur_assets = None, set()

    for (s, e, secs, speaker, pbreak) in base:
        unit_text = text[s:e]
        assets = _assets_in(unit_text)
        if cur_start is not None:
            length = s - cur_start
            new_topic = bool(assets - cur_assets) and length >= MIN_SEGMENT_CHARS
            speaker_change = speaker is not None and cur_speaker is not None and speaker != cur_speaker
            para = pbreak and length >= MIN_PARAGRAPH_SEGMENT_CHARS
            too_long = length >= MAX_SEGMENT_CHARS
            if new_topic or speaker_change or para or too_long:
                close(s, secs)  # up to the next unit, so segments are contiguous
        if cur_start is None:
            cur_start, cur_secs = s, secs
            cur_speaker = speaker or cur_speaker
        if speaker:
            cur_speaker = speaker
        cur_assets |= assets
        cur_end = e
    close(len(text), None)
    # Fill end_seconds from the next segment's start when the source had times.
    for a, b in zip(segments, segments[1:]):
        if a.end_seconds is None and b.start_seconds is not None:
            a.end_seconds = b.start_seconds
    return segments


# --- Chunking ----------------------------------------------------------------


def chunk_transcript(nt, max_tokens, estimator, overlap_tokens=None, chars_per_token=3.0):
    """
    Split a normalized transcript into ordered, overlapping chunks of at most
    `max_tokens` (per `estimator`). Boundaries prefer segment edges, then
    sentence edges, then a token-safe character cut. Returns [] for empty
    text and exactly one chunk when the whole text fits.
    """
    text = nt.text
    if not text:
        return []
    overlap_tokens = CHUNK_OVERLAP_TOKENS if overlap_tokens is None else overlap_tokens
    if estimator(text) <= max_tokens:
        return [_make_chunk(nt, 1, 0, len(text), 0, estimator)]

    # Candidate cut points: segment ends, then sentence ends.
    cuts = sorted({seg.end_character for seg in nt.segments} | {len(text)})
    sentence_cuts = sorted({e for s, e in _sentence_spans(text, 0)})
    max_chars = max(1, int(max_tokens * chars_per_token))
    overlap_chars = int(overlap_tokens * chars_per_token)

    # Overlap can never be most of a chunk, or consecutive chunks would stop
    # advancing; a quarter of the chunk is plenty to keep a sentence whole.
    overlap_chars = min(int(overlap_tokens * chars_per_token), max_chars // 4)

    def best_end(start, must_exceed):
        for candidates in (cuts, sentence_cuts):
            fitting = [c for c in candidates
                       if c > must_exceed and start < c <= len(text)
                       and estimator(text[start:c]) <= max_tokens]
            if fitting:
                return max(fitting)
        end = min(len(text), start + max_chars)
        while end < len(text) and estimator(text[start:end]) > max_tokens and end > start + 1:
            end -= max(1, (end - start) // 10)
        if end < len(text):  # never cut inside a word when avoidable
            back = text.rfind(" ", start, end)
            if back > max(start, must_exceed):
                end = back
        return max(end, must_exceed + 1) if end <= must_exceed else end

    chunks, start, seq = [], 0, 1
    prev_end = 0
    while start < len(text):
        end = best_end(start, prev_end)
        if end <= prev_end:  # the overlap left no room: drop it for this step
            start = prev_end
            end = best_end(start, prev_end)
        end = min(max(end, start + 1), len(text))
        ov = chunks[-1].end_character - start if chunks else 0
        chunks.append(_make_chunk(nt, seq, start, end, max(0, ov), estimator))
        if end >= len(text):
            break
        seq += 1
        prev_end = end
        start = max(0, end - overlap_chars)
        if start <= chunks[-1].start_character:
            start = chunks[-1].start_character + 1
    return chunks


def _make_chunk(nt, seq, start, end, overlap_chars, estimator):
    body = nt.text[start:end]
    segs = [s for s in nt.segments if s.end_character > start and s.start_character < end]
    start_secs = next((s.start_seconds for s in segs if s.start_seconds is not None), None)
    end_secs = next((s.end_seconds for s in reversed(segs) if s.end_seconds is not None), None)
    return Chunk(
        chunk_id=f"{nt.transcript_hash[:12]}-c{seq:03d}", sequence_number=seq,
        start_character=start, end_character=end, start_seconds=start_secs,
        end_seconds=end_secs, input_tokens=estimator(body), overlap_chars=overlap_chars,
        segment_ids=[s.segment_id for s in segs], text=body,
    )


def validate_coverage(chunks, total_chars):
    """
    (ok, problems): every character position [0, total_chars) must fall in at
    least one chunk, chunks must be ordered, and each must advance the text.
    """
    problems = []
    if total_chars == 0:
        return (not chunks, ["chunks for empty text"] if chunks else [])
    if not chunks:
        return False, ["no chunks"]
    if chunks[0].start_character != 0:
        problems.append(f"first chunk starts at {chunks[0].start_character}, not 0")
    if chunks[-1].end_character != total_chars:
        problems.append(f"last chunk ends at {chunks[-1].end_character}, not {total_chars}")
    for a, b in zip(chunks, chunks[1:]):
        if b.start_character > a.end_character:
            problems.append(f"gap between {a.chunk_id} and {b.chunk_id}: "
                            f"{a.end_character}..{b.start_character}")
        if b.sequence_number != a.sequence_number + 1:
            problems.append(f"sequence break at {b.chunk_id}")
        if b.end_character <= a.end_character:
            problems.append(f"{b.chunk_id} does not advance past {a.chunk_id}")
    for c in chunks:
        if c.end_character <= c.start_character:
            problems.append(f"{c.chunk_id} is empty")
    return not problems, problems
