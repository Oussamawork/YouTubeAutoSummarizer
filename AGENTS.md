# Repository identity

- Use the personal GitHub account **Oussamawork** for authenticated operations
  on this repository. Verify the account before any authenticated operation.
- Use the repository's configured personal SSH transport and Git author settings.
  Keep email addresses, unrelated account identifiers and local key paths out of
  repository documentation.
- The global GitHub CLI may use a different account. Its permissions do not
  establish the personal SSH account's access. Do not change global account
  configuration for this repository's work.
- Public, unauthenticated reads of GitHub metadata are fine. Keep secrets
  inside their configured environment; use GitHub Actions for repository secrets.

# Development

Keep all further work local. Do not push branches, commits or reports unless
the user explicitly lifts the no-push instruction given in this task.

Read `CLAUDE.md` for the module map, design documents and verification workflow.
Run `./scripts/check.sh` for code changes. Run live model benchmarks only when
authorized; they consume the same provider quota as the scheduled summaries.
