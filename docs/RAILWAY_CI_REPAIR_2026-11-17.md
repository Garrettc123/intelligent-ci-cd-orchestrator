# Railway independent CI cron — observed blocker and repair

**Inspected:** November 17, 2026, Railway garcar-review-staging.
**Service:** garcar-independent-ci, scheduled every 15 minutes.
**Observed:** latest run CRASHED; Railway runtime logs showed repeated volume
mount and process-start messages, without the application check report.
The exact runtime exception was not exposed in accessible logs.

## Specific risks identified

1. The existing Dockerfile selected USER runner before launching the
   entrypoint. Railway's volume is mounted separately and may be owned
   by root. The entrypoint's chown-then-gosu bootstrap only ran when
   its UID was 0, so it could not repair root-owned volumes when already
   running as runner. This is a *plausible* filesystem failure, not a
   proven root cause from logs.
2. The Railway service lists approved target metadata variable names,
   but no GARCAR_CI_GITHUB_READ_TOKEN variable name was present.
   If GARCAR_CI_MODE is private, the current private clone is expected
   to fail closed. The actual secret values/mode were not inspected.
3. A recurring failed job is not evidence that the application tests passed;
   never mark the CI target verified based only on successful container build.

## Repairs in this PR

- Allow the entrypoint to begin as root only long enough to create/chown
  the already-mounted evidence directory. Immediately run all cloning,
  installs and tests as the unprivileged runner user via gosu.
- Emit structured blocked status for missing read-only credentials,
  target identity, unsupported mode, or a volume that is not writable
  when already unprivileged. Never print secret values.
- Keep the reviewed repository and commit pinning; do not fall back to
  public mode if private access is missing.
- Run shell syntax checks, the existing CI runner unit tests, and a
  static bootstrap privilege-boundary check in GitHub CI.

## What is NOT done

No Railway configuration/variables were changed, no cron restarts or
deployments were triggered, no private GitHub credentials were generated,
and no jobs were declared green. We cannot legitimately bypass a missing
GitHub read-only credential or a failed pinned-commit/branch check.

## Release verification gates

1. GitHub CI for this branch passes.
2. Operator confirms the intended public/private GARCAR_CI_MODE.
3. In private mode, a dedicated fine-grained read-only GitHub credential
   must be made available through a protected secrets mechanism and
   scoped to the approved repository. Do not paste tokens into chat.
4. Operator authorizes staging service source update and any incremental
   Railway compute/storage usage; the patched image builds.
5. An actual 15-minute scheduled run completes with a source-linked
   report, verified log digests and an accurate exit code; investigate
   an inaccessible pinned branch/commit rather than declaring success.
6. Keep sovereign-autonomy-worker stopped until its separate execution
   grant is renewed through the established approval path.

Do not infer that this repair fixes a GitHub Actions billing problem or
that a successful local CI run proves successful revenue collection.
