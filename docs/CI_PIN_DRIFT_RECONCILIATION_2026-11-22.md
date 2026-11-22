# Independent CI cron: pin-drift reconciliation (2026-11-22)

This file records a **source-only candidate fix**, not a successful Railway run.

## Evidence

- Railway staging service `garcar-independent-ci`, project `a8fd2149-530e-4b72-8e4b-131526f9452d`, source `Garrettc123/intelligent-ci-cd-orchestrator`, branch `audit/payment-receiver-independent-ci-20261116`, deployed commit `db89052837b2cab3703f5d1f0367b5890c76e6ef`.
- Last inspected deployment `e17657aa-294b-48c2-9039-97f743187aad` was **CRASHED**; its cron reported a failed run; Railway's diagnosis did **not** specify a root cause and available runtime logs had only mount/start lines.
- Five public smoke target branch/commit pairs were checked against the authenticated GitHub compare API. **Four were identical**.
- `Garrettc123/enterprise-unified-platform` branch `main` was **one commit ahead** of the pinned `27c926ea3fddbf02032f4b3eb31f9e9fb450fe0d`. Current `main` is `40dc11c3922ae36cf1d9b9d687ab7166099eab2b`, commit `Harden first-admin setup and backend database configuration (#89)`. Its GitHub comparison shows 16 affected files, including security and database code, tests and package-lock. This is source identity evidence, not an independent security audit.
- `GARCAR_CI_MODE` and target identity variable names are present in Railway, but **`GARCAR_CI_GITHUB_READ_TOKEN` was not listed**. The entrypoint explicitly rejects private mode without that read-only token; the mode's value was not read, so root cause remains **unconfirmed**.
- The existing CI cron's `Dockerfile` and `deploy/entrypoint.sh` are unchanged by this repair candidate; no privileges, environment variables or provider settings have been changed.

## Proposed patch

Update **only** the public `enterprise-unified-platform` target's exact `expected_commit` from `27c926ea3fddbf02032f4b3eb31f9e9fb450fe0d` to `40dc11c3922ae36cf1d9b9d687ab7166099eab2b`. Preserve `branch=main`, all public targets, their command arrays, and the fail-closed clone + SHA check.

Local static regression: `python -m unittest tests.test_manifest_pin_consistency -v`. Full suite: `python -m unittest discover -s tests -v`.

## Release gates

1. Verify tests on this review branch and inspect changes in upstream commit #89 for dependency/auth/security regression risk before approval.
2. **Do not auto-deploy.** Railway's source is pinned to a previous SHA, so a draft PR does not change the live cron.
3. Verify selected `GARCAR_CI_MODE` using the provider's authorized settings UI or an approved non-sensitive boolean diagnostic; never print or request token values. If private mode is intended, a dedicated scoped read-only GitHub credential must be provisioned through the approved secret manager before retrying; do not fall back to public mode silently.
4. Verify current volume permissions, available space and source-fetch egress, retaining existing evidence. Do not delete audit receipts to make a check pass.
5. After a separately approved staging release, require an actual runner report with the exact SHA, passing source tests, a verifiable evidence digest and a successful Railway cron run. Stop if it still exits nonzero.

No proof of actual payment settlement, customer delivery, source security clearance, or running agents is established by this patch.
