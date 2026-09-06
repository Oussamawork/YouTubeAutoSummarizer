# Source fidelity improvements

The saved outputs exposed three concrete summary errors: a Rubrik hold rating
became a claim of personal ownership, historical peer ratings appeared as
current views, and Lockheed Martin's quoted market price became a personal
buy-below level. The research validator also stripped numeric minus signs
when matching evidence and did not recognize negative numeric values.

## Behavior

`summary_policy.py` supplies one fidelity policy to full summaries, compact
digests, combined summary/claims calls, chunk notes, merges and review. Chunk
notes retain attribution and whether a view is current, past or a comparison.
Changed prompt/schema versions invalidate older notes; the summary prompt
version is also included in research run identity. The claim extraction prompt
is version 3 and explicitly distinguishes valuations, ratings and transactions.

`summary_review.py` audits the exact cleaned text that would be delivered. It
sends the complete normalized transcript and every numbered nonempty summary
line to the configured model chain. Approval requires an ordered, exhaustive
set of supported verdicts, literal locatable excerpts and no reported material
omissions. Excerpt matching permits whitespace changes but preserves signs,
negations and punctuation. Repeated quotes keep all matching offsets rather
than inventing a unique timestamp. No source timestamp means no timestamp.

A semantic rejection permits one bounded repair, followed by a fresh audit.
API/quota/context failures and malformed reviews are unavailable, not approval.
The scraper applies this gate to both combined and summary-only generation,
including compact and chunked drafts. Direct calls to `summarize_transcript`
remain generation primitives; the production delivery gate is in the scraper.

Rejected or unavailable reviews create a silent retryable outcome. They do
not advance the channel watermark or consume the no-transcript give-up count.
Pending state preserves the latest draft and its transcript identity. Where the
raw source is stored, retries skip transcript fetching and draft generation;
a changed transcript hash invalidates the held draft. Review-pending entries
survive RSS expiry. An on-demand failure reports failure without claiming a
scheduled retry exists. Already-held delivery messages preserve their original
behavior and are not retroactively reviewed.

Reports in `data/research/summary_reviews.jsonl` retain each draft, every
review, source evidence, source hash, repair metadata and final text. Generation
telemetry is restored after reviewing so claims are not attributed to the
reviewer's model. Persistence remains best-effort and reports whether it
succeeded; errors are logged. Historical research/output files are not rewritten.

## Evidence and limits

The baseline had 619 passing tests. New offline tests exercise line coverage,
invented evidence, signs, uncertain/malformed reviews, repair re-auditing,
quota/context failures, production delivery gates, retry persistence, stored
source reuse, RSS expiry and benchmark integrity.

The nine-case real-source benchmark is a regression seed with agent-prepared
labels, not an independent human benchmark. The first live run on 2026-09-06
used the repository's Gemini secret in Actions and matched all nine labels:
four faithful candidates approved and five corrupted candidates rejected,
with no unavailable reviews. Seven cases used Gemini 3.7 Flash and two used
Gemini 3.6 Flash. Full results are in
`evals/reports/summary-review-34041316375-1.json`; Actions run 34041316375 also
passed all 657 tests on Python 3.11. This establishes the targeted regression
result, not general production accuracy or independent human validation.
`summary_eval.py --live` is the explicit, quota-spending measurement path; the
default command only validates fixture integrity. The existing `claims_eval.py`
remains the separate extraction-quality benchmark.

For credentials already stored in GitHub, the manual `summary-quality.yml`
workflow injects `secrets.GEMINI_API_KEY` directly into the live benchmark step.
No secret is retrieved into the local workspace. The job shares the daily summary
concurrency group and uploads reports and its quota counters after evaluation
failure. Code pushes to the isolated `codex/summary-accuracy` branch also trigger
evaluation and commit the report under `evals/reports/` on that branch only, so
the personal SSH account can retrieve results. Report-only commits do not
trigger another run. It does not publish Telegram messages, update main, or
modify the production quota counter. Git operations for this repository use
the verified personal SSH identity; the global GitHub CLI account is unrelated.

Literal evidence is necessary but insufficient for semantic entailment. A
reviewer can still accept a misleading claim or miss a material omission, and
using the same provider chain can correlate generation/review errors. Transcript
errors, unseen charts, missing context and unreliable claims made by a speaker
remain limitations. These summaries describe what the source says; they do not
establish whether the source's financial assertions are true.

Review adds one call for a passing draft or three when repairing, before API
retries/escalations. A very long source that cannot fit any reviewer defers;
this implementation does not claim global semantic review from partial chunks.
The next accuracy milestone is independently labelling a broader channel sample,
running both live benchmarks, inspecting false approvals and omissions, and
comparing model/prompt configurations on the same held-out material.
