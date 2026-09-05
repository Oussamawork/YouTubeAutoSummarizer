"""Deterministic normalization, segmentation and complete-coverage chunking."""
import transcript_normalize as tn


def _n(text, vid="v"):
    return tn.normalize_transcript(text, vid)


def test_adjacent_caption_overlap_is_removed():
    raw = ("00:00:01 --> 00:00:03\nwe think the fed will cut rates in\n"
           "00:00:03 --> 00:00:05\ncut rates in september which helps banks\n"
           "00:00:05 --> 00:00:07\nwhich helps banks a lot\n"
           "00:00:07 --> 00:00:09\nand that is it\n")
    nt = _n(raw)
    assert nt.text == "we think the fed will cut rates in september which helps banks a lot and that is it"
    assert nt.quality_flags["caption_overlap_removed"] == 2
    assert nt.timestamps_available is True


def test_identical_adjacent_cues_are_dropped_but_repetition_inside_a_cue_stays():
    raw = "1\n00:00:01,000 --> 00:00:03,000\nsell now. sell now.\n\n2\n00:00:03,000 --> 00:00:05,000\nsell now. sell now.\n\n3\n00:00:05,000 --> 00:00:07,000\nI mean it\n\n4\n00:00:07,000 --> 00:00:09,000\nreally\n"
    nt = _n(raw)
    assert nt.text.count("sell now.") == 2          # in-cue emphasis preserved
    assert nt.quality_flags["duplicate_cues_removed"] == 1


def test_legitimate_repetition_in_plain_text_is_untouched():
    raw = "I am not selling. I am not selling. I am not selling Nvidia here."
    assert _n(raw).text == raw


def test_fragmented_caption_lines_are_joined_and_whitespace_normalized():
    raw = "the   revenue\r\nshould rise\r\nbut margins\tmay fall\r\n\r\nnext topic"
    nt = _n(raw)
    assert nt.text == "the revenue should rise but margins may fall\nnext topic"
    assert nt.quality_flags["lines_joined"] >= 2


def test_numbers_percentages_currencies_ranges_decimals_survive():
    raw = ("target is $150-200 by 2030, margins fall 5.25%, revenue up €1,200.50 million, "
           "range 100 to 150, not 30% but 3.0%")
    assert _n(raw).text == raw


def test_negation_and_conditional_wording_preserved():
    raw = "I would not buy Apple unless it drops below $150, and only if rates fall."
    assert _n(raw).text == raw


def test_multilingual_text_is_not_corrupted():
    raw = "Nvidia wird meiner Meinung nach 200 $ erreichen. 我认为英伟达会涨。 Ça va monter à 15 %."
    assert _n(raw).text == raw


def test_ticker_like_words_and_missing_punctuation_are_kept():
    raw = "so TSLA and NVDA both look weak SOL too no punctuation here at all"
    assert _n(raw).text == raw


def test_speaker_labels_and_timestamps_preserved_at_segment_level():
    raw = ("[00:01] Host: welcome to the show today we discuss apple\n"
           "[00:20] Guest: I think Apple falls 10% this quarter\n"
           "[00:45] Host: interesting and what about the long term\n"
           "[01:00] Guest: long term I am bullish on Apple")
    nt = _n(raw)
    assert "Host: welcome" in nt.text and "Guest: I think" in nt.text
    assert nt.timestamps_available
    speakers = [s.speaker for s in nt.segments]
    assert speakers == ["Host", "Guest", "Host", "Guest"]
    assert nt.segments[1].start_seconds == 20 and nt.segments[1].end_seconds == 45
    assert nt.segments[0].segment_id.endswith("-s001")


def test_one_off_label_is_not_a_speaker():
    nt = _n("Note: this is just prose with a colon\nmore text follows here")
    assert nt.text.startswith("Note: this is")
    assert nt.segments[0].speaker == "unknown"


def test_offset_map_traces_back_to_raw():
    raw = "hello    world\r\nsecond   line"
    nt = _n(raw)
    start, end = nt.raw_span(6, 11)   # "world" in normalized text
    assert raw[start:end] == "world"
    assert nt.transcript_hash == tn.transcript_hash(raw)


def test_unintelligible_markers_flagged_not_deleted():
    nt = _n("the fed [inaudible] will cut ??? rates [Music]")
    assert "[inaudible]" in nt.text and "[Music]" in nt.text
    assert nt.quality_flags["unintelligible_markers"] == 3
    assert "unintelligible" in nt.segments[0].quality_flags


def test_segments_cover_text_and_classify():
    raw = ("Welcome back everyone, in this video we look at markets. Today is a big day for tech. "
           "Let us get right into it after a quick word.\n\n"
           "This video is sponsored by BrokerX, use code SAVE for a discount. They offer zero fees on "
           "stocks and ETFs. Sign up with the link in the description.\n\n"
           "Nvidia should reach $200 by year end given data center demand. Hyperscaler capex keeps "
           "rising quarter after quarter. Margins remain above 70 percent. I expect the stock to grind "
           "higher into earnings.\n\n"
           "This is not financial advice, do your own research. I may hold positions in the names discussed.")
    nt = _n(raw)
    assert nt.segments[0].start_character == 0
    assert nt.segments[-1].end_character == len(nt.text)
    for a, b in zip(nt.segments, nt.segments[1:]):
        assert b.start_character == a.end_character
    cats = [s.primary_category for s in nt.segments]
    assert "sponsor_or_promotion" in cats and "disclaimer" in cats
    sponsor = next(s for s in nt.segments if s.primary_category == "sponsor_or_promotion")
    assert sponsor.excluded_from_headline and sponsor.exclusion_reason.startswith("category:")
    nvda = next(s for s in nt.segments if "Nvidia" in s.normalized_text)
    assert nvda.primary_category in ("price_target", "forecast", "company_analysis")
    assert nvda.original_text  # raw text retained alongside


def test_segment_at_finds_the_owner():
    nt = _n("A sentence about Apple. " * 30 + "\n\n" + "A sentence about Tesla. " * 30)
    assert len(nt.segments) >= 2
    assert nt.segment_at(0) is nt.segments[0]
    assert nt.segment_at(len(nt.text) - 1) is nt.segments[-1]


# --- chunking ---

def _est(t):
    return len(t) // 3


def test_fitting_text_is_one_chunk_with_no_overlap():
    nt = _n("short text that fits")
    chunks = tn.chunk_transcript(nt, 1000, _est)
    assert len(chunks) == 1 and chunks[0].overlap_chars == 0
    assert tn.validate_coverage(chunks, len(nt.text)) == (True, [])


def test_chunks_cover_every_position_with_overlap():
    text = " ".join(f"Sentence {i} about Nvidia hitting ${100 + i} by 2030." for i in range(600))
    nt = _n(text)
    chunks = tn.chunk_transcript(nt, 900, _est, overlap_tokens=100)
    ok, problems = tn.validate_coverage(chunks, len(nt.text))
    assert ok, problems
    assert len(chunks) > 3
    covered = [False] * len(nt.text)
    for c in chunks:
        assert c.input_tokens <= 900
        assert c.text == nt.text[c.start_character:c.end_character]
        for i in range(c.start_character, c.end_character):
            covered[i] = True
    assert all(covered)
    assert all(c.overlap_chars > 0 for c in chunks[1:])
    assert [c.sequence_number for c in chunks] == list(range(1, len(chunks) + 1))
    assert chunks[0].chunk_id.endswith("-c001") and len({c.chunk_id for c in chunks}) == len(chunks)
    # A unique middle marker lands in at least one chunk in full.
    marker = "Sentence 300 about Nvidia hitting $400 by 2030."
    assert any(marker in c.text for c in chunks)


def test_chunk_ids_are_stable_across_runs():
    nt = _n("x y z. " * 2000)
    a = [c.chunk_id for c in tn.chunk_transcript(nt, 500, _est)]
    b = [c.chunk_id for c in tn.chunk_transcript(nt, 500, _est)]
    assert a == b


def test_hard_split_never_loses_text_when_no_boundaries_exist():
    nt = _n("x" * 5000)
    chunks = tn.chunk_transcript(nt, 100, _est)
    ok, problems = tn.validate_coverage(chunks, 5000)
    assert ok, problems


def test_coverage_validator_catches_gaps_and_stalls():
    nt = _n("a b c d e f g h. " * 200)
    chunks = tn.chunk_transcript(nt, 200, _est)
    broken = list(chunks)
    broken[1].start_character = broken[0].end_character + 5   # gap
    ok, problems = tn.validate_coverage(broken, len(nt.text))
    assert not ok and any("gap" in p for p in problems)
    assert tn.validate_coverage([], len(nt.text)) == (False, ["no chunks"])
    assert tn.validate_coverage([], 0) == (True, [])
