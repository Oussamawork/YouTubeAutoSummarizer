# Summary source-review regression cases

Nine cases use the four checked-in transcripts in `data/transcripts/`. Each
case pins the raw transcript hash, includes literal reference excerpts, and
labels a candidate summary as approved or rejected. Four faithful candidates
and five deliberately corrupted variants cover rating versus ownership,
quoted price versus entry threshold, past versus current ratings, management
versus host disagreement, and a passing mention mislabelled as neutral.

Labels were prepared by the coding agent after reading the saved transcripts.
They have **not** been independently human-validated. All four videos belong
to one channel. The first live run on 2026-09-06 matched all nine labels
(four approvals and five rejections), with no unavailable reviews. It used
the configured Gemini 3.7/3.6 Flash chain. The complete evidence and model
telemetry are saved in `evals/reports/summary-review-34041316375-1.json`.
This narrow result does not establish production-wide accuracy.

`python summary_eval.py` validates file structure, source hashes and reference
excerpts. Its `fixture_validation_only` report contains no accuracy score.
`SUMMARY_EVAL_LIVE=1 python summary_eval.py --live` calls the actual reviewer,
without repair, Telegram delivery or changes to historical research records.
It uses the configured provider chain and its normal quota accounting. Missing
opt-in or credentials is an error, never a silent fallback to offline mode.

Live reports separate false approvals, false rejections and unavailable
reviews, retain actual review evidence/model telemetry, and return a nonzero
exit code if any verdict fails to match its label. Compare these separately:
a model that rejects every draft is not an accurate reviewer. Expand with
independently labelled videos from the other subscribed channels, languages,
noisy captions and long transcripts before treating the results as representative.
