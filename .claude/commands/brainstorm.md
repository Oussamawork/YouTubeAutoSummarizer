---
description: Multi-agent feature brainstorming — propose, debate, classify, then choose.
argument-hint: "[optional theme to focus ideas on]"
---

Run a multi-agent feature-ideation pipeline for this repository. If the user
passed a theme in `$ARGUMENTS`, constrain ideas to it; otherwise keep it
open-ended. Pass data between stages via files in `/tmp/brainstorm/`.

First confirm the verification baseline is green: run `./scripts/check.sh`. If it
fails, stop and report — don't brainstorm on a broken tree.

Then run these agents (Task tool). The 3 ideators run in parallel; stages 2A/2B
run in parallel; the rebuttal round reuses the same two evaluators via follow-up
messages.

1. **Ideation — 3 parallel ideators** (general-purpose), each reading the repo
   (`scraper.py`, `summarizer.py`, `transcript.py`, `sendToTelegram.py`,
   `helpers.py`, `README.md`, `.github/workflows/daily-summary.yml`,
   `git log --oneline -20`) but through a distinct lens. Each proposes 4–6
   concrete, actionable ideas; for each: Title, Problem, Proposed change (files
   touched), Effort (S/M/L), Risks/deps.
   - **Ideator A — Product / end-user value**: summary quality, output formats,
     delivery channels, configurability, UX. Writes `/tmp/brainstorm/ideas_A.md`.
   - **Ideator B — Reliability / operations**: error handling, transcript
     sourcing robustness, rate limits, observability/logging, CI. Writes
     `/tmp/brainstorm/ideas_B.md`.
   - **Ideator C — Emerging / state-of-the-art tech**: ideas grounded in the
     newest available solutions — latest LLM models and capabilities (structured
     output, long context, cheaper/faster tiers), modern transcript/ASR options,
     new provider APIs, and recent libraries/techniques. This ideator MUST use
     WebSearch/WebFetch to ground claims in current (today's) offerings and must
     NOT invent model names, prices, or APIs — verify before proposing, and cite
     the source for each tech claim. Writes `/tmp/brainstorm/ideas_C.md`.

   Then **merge & dedupe** the three files into a single numbered list (assign
   IDs I1, I2, …; collapse near-duplicates, keeping the clearest framing) at
   `/tmp/brainstorm/ideas.md`. This merge is done by the orchestrator directly.

2. **Shared rubric**: every idea is scored 1–5 on Impact, Reach, Effort
   (1=large…5=trivial), Risk (1=risky…5=safe), Strategic-fit.

   - **2A — Proponent** (general-purpose): Steelman each idea; argue the best
     realistic case for impact/value; score on the rubric (weight Impact, Reach,
     Strategic-fit). Name the ideas most worth building. Write
     `/tmp/brainstorm/eval_proponent.md`.
   - **2B — Skeptic** (general-purpose): Attack feasibility/risk/cost/maintenance,
     grounded in actual code and constraints (daily GitHub Actions run, secrets,
     YouTube IP-blocking of CI, Telegram 4096/HTML limits, test coverage). Score
     on the rubric (weight Effort, Risk). Flag deceptively expensive/fragile
     ideas. Write `/tmp/brainstorm/eval_skeptic.md`.

3. **Rebuttal round** (same two agents, follow-up message): each reads the
   other's evaluation, concedes where right, pushes back where wrong, per idea.
   Write `/tmp/brainstorm/rebuttal_proponent.md` and `rebuttal_skeptic.md`.

4. **Synthesizer** (general-purpose): Read ideas + both evaluations + both
   rebuttals. Produce a ranked shortlist; bucket each idea (Quick win /
   High-impact bet / Nice-to-have / Skip with reason); explicitly list the ideas
   with the biggest Proponent↔Skeptic disagreement as "decision points". Write
   `/tmp/brainstorm/classified.md`.

5. **Choose**: Present the ranked shortlist via the AskUserQuestion tool so the
   user picks which idea(s) to implement.

6. **Implement (only after the user chooses)**: run `./scripts/check.sh` (before),
   implement the chosen idea + tests, run `./scripts/check.sh` (after), report
   both results, and commit to the working branch. Do not push or merge without
   explicit permission.
