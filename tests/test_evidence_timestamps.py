"""Item 9: evidence timestamps come from the cue that contains the evidence."""
import claims as cm
import transcript_normalize as tn

RAW = ("00:00:12 --> 00:00:15\nHost: welcome back everyone today we talk nvidia\n"
       "00:00:15 --> 00:00:19\ntoday we talk nvidia and honestly I expect Nvidia to fall over the next\n"
       "00:00:19 --> 00:00:23\nover the next three months, but I remain bullish over five years\n"
       "00:00:23 --> 00:00:27\nGoldman expects the stock to reach $200 but that is their call not mine\n"
       "00:00:27 --> 00:00:31\nHost: could Apple fall 30 percent from here? I own Tesla by the way\n")


def _ctx():
    return {"video_id": "demo01", "channel_id": "c1", "channel_name": "Chan", "video_title": "T",
            "published_at": "2026-09-01T00:00:00+00:00", "run_key": "rk"}


def test_each_claim_in_one_segment_gets_its_own_cue_timestamp():
    nt = tn.normalize_transcript(RAW, "demo01")
    assert nt.timestamps_available and len(nt.segments) <= 2
    raw = [
        {"attribution_type": "speaker_personal_view", "claim_type": "forecast", "subject_mention": "Nvidia",
         "forecast_direction": "decrease", "horizon_original": "over the next three months",
         "evidence_text": "I expect Nvidia to fall over the next three months"},
        {"attribution_type": "speaker_personal_view", "claim_type": "stance", "subject_mention": "Nvidia",
         "stance": "bullish", "horizon_original": "over five years", "evidence_text": "I remain bullish over five years"},
        {"attribution_type": "speaker_quoting_third_party", "claim_type": "third_party_view", "subject_mention": "the stock",
         "target_value": 200, "evidence_text": "Goldman expects the stock to reach $200"},
    ]
    claims, _ = cm.validate_claims(raw, nt, _ctx())
    first_seg = claims[0]["segment_id"]
    assert all(c["segment_id"] == first_seg for c in claims), "the three claims share one semantic segment"
    starts = [c["evidence_start_seconds"] for c in claims]
    ends = [c["evidence_end_seconds"] for c in claims]
    assert starts == [15, 19, 23]      # the cue each excerpt begins in, not the segment's first cue (12)
    assert ends == [23, 23, 27]        # the cue the excerpt ends in, ended by the next cue's start
    assert nt.segments[0].start_seconds == 12


def test_plain_text_transcripts_keep_null_timestamps():
    nt = tn.normalize_transcript("I expect Nvidia to fall over the next three months. I remain bullish over five years.", "v")
    raw = [{"attribution_type": "speaker_personal_view", "claim_type": "stance", "subject_mention": "Nvidia",
            "stance": "bullish", "evidence_text": "I remain bullish over five years"}]
    c = cm.validate_claims(raw, nt, _ctx())[0][0]
    assert c["evidence_start_seconds"] is None and c["evidence_end_seconds"] is None


def test_untimed_cue_inherits_the_previous_timestamp():
    raw = "00:00:05 --> 00:00:08\nfirst cue words here\nsecond line without a time\n00:00:20 --> 00:00:25\nthird cue"
    nt = tn.normalize_transcript(raw, "v")
    assert nt.seconds_at(nt.text.find("second")) == 5
    assert nt.end_seconds_at(nt.text.find("second")) == 20
    assert nt.seconds_at(nt.text.find("third")) == 20 and nt.end_seconds_at(nt.text.find("third")) is None
