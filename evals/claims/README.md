# Claim-extraction evaluation fixtures

Hand-labelled transcript segments with the atomic claims a correct
extraction must produce, replayed by `python claims_eval.py --fixtures
evals/claims`. Not part of `./scripts/check.sh`; the harness's own code is
unit-tested in `tests/test_claims_eval.py`, which never calls a model.

## Modes

- **offline (default)** — each fixture's stored `model_output.claims` is
  pushed through the real deterministic pipeline (`transcript_normalize` →
  `claims.validate_claims`) and scored against `expected_claims`. This
  measures the software and the labelled outputs, not the live model.
- **live** — `--live` **and** `CLAIMS_EVAL_LIVE=1` **and** a provider key
  (`GEMINI_API_KEY` / `GROQ_API_KEY` / `LLM_API_KEY`). Calls
  `signals.extract_research` with the current prompt and spends real quota;
  the report header says `mode: live`. Without all three it falls back to
  offline and says so.

## Fixture format (`<fixture_id>.json`)

```json
{
  "fixture_id": "nvda-two-horizons",
  "channel_name": "Demo Investor",
  "video_title": "Nvidia: short-term pain, long-term gain?",
  "published_at": "2026-07-01T14:00:00+00:00",
  "transcript": "00:00:12 --> 00:00:15\nHost: ...",
  "expected_claims": [
    {"evidence_text": "I expect Nvidia to fall over the next three months",
     "subject": "NVDA", "claim_type": "forecast", "attribution_type": "speaker_personal_view",
     "stance": "bearish", "forecast_direction": "decrease", "horizon_bucket": "short",
     "recommendation_action": "none", "ticker": "NVDA", "testability_type": "unconditional_testable"}
  ],
  "model_output": {"claims": [ {"...": "raw records exactly as the model returns them"} ]}
}
```

`evidence_text` (verbatim from the transcript) and `subject` (a ticker or
the name as spoken; curated aliases are applied) are required on every
expected claim. Any other field present is scored: `claim_type`,
`attribution_type`, `stance`, `forecast_direction`, `target_value` /
`target_low` / `target_high`, `horizon_bucket`, `recommendation_action`,
`ticker`, `portfolio_disclosure`, `host_position`,
`entity_resolution_method`, `testability_type`, `condition_status`. An
empty `expected_claims` list is a valid fixture (a no-claim transcript).

## Matching rules

A predicted claim matches an expected one when its evidence overlaps the
expected excerpt (or sits in the same segment), the resolved asset is the
same, the claim type is in the same family (forecast ~ price_target), and the
direction, target numbers and horizon bucket agree wherever the expected
claim states them. Each expected claim matches at most once; further
matches count as duplicates. Never exact JSON equality.

## Expanding the set

1. Take a real transcript segment from `data/transcripts/` (see
   `transcript_store.load_transcript`) or a fresh fetch; keep caption
   timestamps if the source had them.
2. Label every atomic claim by hand — one record per asset × metric ×
   direction × target × horizon × condition — including questions,
   retrospectives, third-party views (with `host_position`) and portfolio
   disclosures, so precision is measured on the hard cases.
3. Run the current prompt once (`--live`) and paste the raw model response's
   `claims` array into `model_output` so the offline replay stays
   representative of real output; note the model in a `notes` field.
4. Re-run offline; a labelling mistake usually shows up as a false negative
   with a plausible false positive beside it.

The five shipped fixtures are hand-authored caption-style segments with
hand-written model outputs (two deliberate errors are stored: a missed
news-report target and a duplicate), so the offline numbers describe the
pipeline, not Gemini.
