# Data Room independent release verification

This is an opt-in **private** worker mode in the existing Garcar independent CI.
It is not active in the normal public five-repository scheduled runner. The
existing Railway service and its configuration are not changed by this branch.

## Target and run gate

Configure these environment **variable names** in a dedicated isolated private
worker. Supply the secret value only in Railway's protected secret manager;
never paste it into GitHub, an issue, a log, or the chat.

- `GARCAR_CI_MODE=private`
- `GARCAR_CI_TARGET_REPOSITORY=Garrettc123/garcar-data-room`
- `GARCAR_CI_TARGET_BRANCH`: exact reviewed PR branch
- `GARCAR_CI_TARGET_SHA`: exact reviewed lowercase 40-character PR head SHA
- `GARCAR_CI_DATA_ROOM_BASE_SHA`: exact historical manifest commit SHA
  (reviewed prior `main` baseline; must be nonzero, distinct, ancestor of head)
- `GARCAR_CI_GITHUB_READ_TOKEN`: least-privilege **read-only** credential scoped
  to the private Data Room repository; set only on the private worker

Do not reuse the production payment receiver's credentials or attach any
customer/billing database secrets. The unprivileged runner clones the approved
branch with complete ancestor history (not shallow), verifies branch and exact
commit, confirms historical base ancestry, then runs an isolated Python
environment. Data Room is standard-library only: its mode does not require
`requirements.txt` or install packages.

The runner-defined checks are (1) actual release gate unit regression tests,
(2) Python source compilation for full mode, and (3) the canonical real
`scripts/check_release.py` test with `DATA_ROOM_BASE_SHA` explicitly passed.
The PR cannot replace the command list by changing a manifest in its own
repository. The source checkout must stay clean throughout the run.

Evidence is SHA-256 hashed in the independent worker output volume. A pass
proves only the commands executed against a specific pinned source revision,
not production safety, founder approval, immutability, SOC 2 certification or
financial transactions. Evidence is neither a cryptographic signature nor
write-once storage. No secrets are forwarded to the command environment.

## Required operational sequence

1. Confirm the expected Data Room PR head SHA and historical base SHA using
   source metadata, without trusting PR description text.
2. Review the exact PR code and GitHub permissions. Configure a **separate**
   staging private worker with isolated read-only GitHub access.
3. Run the pinned full verification. Inspect `report.json` and hashed logs.
   Both the executed check statuses and recorded source SHA must match.
4. Restore GitHub Actions hosted checks separately; an independent runner does
   not override branch protection or verified-commit requirements.
5. Obtain founder authorization and independent release review before merge.
   Never change investor-room verification status on an isolated CI pass alone.

The source system does not activate private mode or provision infrastructure.
Missing credentials, missing baseline, moved PR head, incomplete history, or
failed tests must produce a blocked/failed receipt, never a passing report.
