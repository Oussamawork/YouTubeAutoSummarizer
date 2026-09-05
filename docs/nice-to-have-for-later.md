# Nice to have for later

Recommendations from the September 2026 code audit that were **not** applied in
the audit branch, with the reason each was deferred and what applying it would
involve. None of them is a correctness bug; the audit's high-impact findings
(delivery-before-decide, the Telegram sender, the warm-prices persist step, the
analytics import cycle, atomic state writes) are already in.

Ordered by expected payoff.

## 1. Pin GitHub Actions to commit SHAs

**Where:** every `uses:` line in `.github/workflows/*.yml`
(`actions/checkout@v5`, `actions/setup-python@v6`).

**Why:** a major-version tag is mutable. Pinning to a full SHA is the standard
supply-chain hardening for workflows that hold `contents: write` and secrets,
and Dependabot keeps pinned SHAs current with a `github-actions` ecosystem
entry.

**Why deferred:** the audit session could not look up the SHAs, and a wrong SHA
breaks every workflow.

**How:** for each action, copy the commit SHA behind the tag from the action's
releases page and write `uses: actions/checkout@<sha> # v5`. Add
`.github/dependabot.yml` with `package-ecosystem: github-actions`.

## 2. One shared HTTP retry helper

**Where:** `scraper.py` (RSS feed, `videos.list`, handle lookup, search),
`transcript.py` (Supadata, Gemini), `summarizer.py`, `channel_scorecard.py`
(Twelve Data time series and symbol search), `sendToTelegram.py`.

**Why:** eight copies of the attempt / backoff / transient-status loop, each
with slightly different status sets and sleep rules, and each fixed
independently when a bug was found (the comments record several). The next
retry bug gets fixed in one place and not the other seven.

**Why deferred:** high regression surface for a change with no visible
behaviour change; it did not fit alongside the delivery rework.

**How:** add `http_retry.py` with something like

```python
def request_with_retry(method, url, *, transient=TRANSIENT_STATUS, retries=3,
                       backoff=2, retry_after=None, **kwargs):
    """Returns the response, or None after `retries` transient failures.
    Never raises. `retry_after(resp)` may return a wait in seconds."""
```

and migrate one call site per commit, keeping each module's current status
sets. The Telegram sender's `_post_chunk` is the newest loop and a good
template. The Gemini 429 handling in `summarizer._call_provider` is the
hardest case (day vs minute quota, escalation) and should go last or stay
bespoke.

## 3. Replace process-wide mutable state with explicit objects

**Where:**

- `channel_scorecard._twelvedata_pace(_calls=[])` and
  `_spend_request_budget(_state=[0])` (state hidden in default arguments)
- `summarizer._EXHAUSTED_PROVIDERS`, `transcript._RATE_LIMITED_THIS_RUN`
- `price_cache._ACTIVE`, `signals_data._LEARNED`
- `channel_scorecard.fetch_prices_live(_warned=[])` (added by the audit, same
  pattern)

**Why:** `tests/conftest.py` needs four autouse fixtures purely to reset these,
`_spend_request_budget` cannot be reset without reaching into its default, and
any future in-process batching (two runs in one process) would share the
state.

**How:** a `RatePacer` class holding the timestamps and request budget,
instantiated once per run in `warm_prices.main` / `channel_scorecard.main` /
`market_pulse.main` and passed to the fetchers; a `RunContext` for the two
exhausted-provider sets, created in `scraper.main` and threaded through
`_summarize_video`. The autouse fixtures then disappear.

## 4. Rotate `data/signals.jsonl`

**Where:** `signals_data.load_signals`, `scraper.SIGNALS_FILE`,
`.github/workflows/daily-summary.yml` (the persist step's `git add`).

**Why:** the file is 1.5 MB after six weeks, embeds the full summary text of
every video, and is committed up to twelve times a day. Append-only lines
delta well, so the pack is small today, but every workflow clones the whole
history and this becomes a wall-clock cost within a year or two.

**How:** write to `data/signals-YYYY-MM.jsonl` and have `load_signals` glob
`data/signals*.jsonl`. Consider dropping the `summary` field from the signal
record: the pulse and scorecard never read it, and the delivered text already
lives in the Telegram channel.

## 5. Cache handle resolution across runs

**Where:** `scraper.resolve_channel_handle`, `seen_videos.json`.

**Why:** each `@handle` in `channel_ids.txt` costs one YouTube API quota unit
per run to resolve, every run. Trivial today (two handles, twelve runs a day)
but it is the one YouTube call that has no reason to repeat.

**How:** store `{"handles": {"@name": "UC..."}}` in the state file and only
call the API for handles not yet mapped.

## 6. Linting and type hints in the gate

**Where:** `scripts/check.sh`, `.github/workflows/ci.yml`,
`requirements-dev.txt`.

**Why:** there is no linter, formatter or type checker, and no type hints. The
shadowed-module bug the audit fixed in `transcript.py` (a local named
`gemini_quota` hiding the imported module) is exactly what a linter catches
for free.

**How:** add `ruff` to `requirements-dev.txt`, `ruff check .` to `check.sh`,
and a `pyproject.toml` with a small rule set (`E`, `F`, `B`, `A` for
shadowing). Type hints can follow module by module; `signals_data.py` and
`price_cache.py` have the clearest signatures to start with.

## 7. Smaller items

- `signals.summarize_with_signals` serialises the parsed signals dict with
  `json.dumps` only to re-parse it through `_parse_signals`; give
  `_parse_signals` a dict-accepting entry point instead.
- `market_pulse._pulse_inputs`, `generate_pulse` and `generate_charts` each
  call `aggregate_assets` on the same window; compute once and pass through.
- `scraper.fetch_video_details` screens the whole ~15-entry feed before
  candidate selection. Filtering after `_select_candidates` would send fewer
  ids, at the cost of one extra code path; the quota cost today is one unit
  per channel per run, so this is cosmetic.
- `scraper.main` is still a long function with deep nesting. The delivery
  rework extracted `_Outbox`, `_finalize_video`, `_hold_for_delivery` and
  `_undelivered_candidates`; the next step is a `process_channel(channel, ctx)`
  that takes the run context from item 3 above.
