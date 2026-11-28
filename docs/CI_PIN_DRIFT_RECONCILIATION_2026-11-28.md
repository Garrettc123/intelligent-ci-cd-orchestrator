# CI pinned-source reconciliation — 2026-11-28

## Verified evidence

- Railway `garcar-review-staging / garcar-independent-ci` cron deployment `97922d76-334b-4d16-bede-261226237b64` was CRASHED.
- Its November 28 runtime receipt reported `evidence_verified=true`, `deployment_performed=false`: four public targets passed real smoke checks, while `enterprise-unified-platform` was blocked.
- The approved manifest on the deployed branch pinned `Garrettc123/enterprise-unified-platform` main to `40dc11c3922ae36cf1d9b9d687ab7166099eab2b`.
- GitHub main now points to `b93c77cd6a2368fd40b44f9f4231033df695e3df`; comparison shows exactly one additional commit and only three added files: a guarded status workflow, read-only orchestration status endpoint, and its tests.
- This source pin drift explains the blocked status for the fifth target. The SHA mismatch does not imply the new code passed independent tests.

## Scope of this repair candidate

- Change only that target's exact expected SHA to `b93c77cd6a2368fd40b44f9f4231033df695e3df`.
- Update the corresponding static regression assertion.
- Preserve all five targets, commands, branch checks, credentials rules, output hashing, no-deploy guarantee, and fail-closed behavior.
- No changes to Railway, live targets, credentials, money movement, outreach, or production permissions.

## Acceptance gate

1. Run `python -m unittest discover -s tests -v` on this PR branch and inspect the three upstream files before merge.
2. Review and merge into the existing Railway source branch `audit/payment-receiver-independent-ci-20261116` only after CI succeeds; make sure this does not auto-deploy without release review.
3. Redeploy only the approved staging service revision; verify expected image SHA and latest Railway job.
4. Require all five exact-commit smoke checks, `evidence_verified=true`, a successful cron exit, and `deployment_performed=false`.
5. Preserve historical evidence receipts. If further drift, network failures or test failures appear, keep the result blocked and diagnose them individually.

Neither manifest reconciliation nor a successful smoke run establishes production readiness or end-to-end payment settlement.
