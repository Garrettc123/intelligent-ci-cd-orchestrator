# Garcar Independent CI — verifiable local runner

This replaces the previous simulated `IntelligentCICD` implementation, which used random outcomes and invented deployment success. This runner **executes actual tests** on an operator-controlled laptop, Ubuntu/Termux host, or private worker. It does not depend on GitHub Actions, paid runner minutes, or GitHub's billing system. GitHub is optional source hosting, not the execution engine.

## What is actually implemented

- **Run:** execute fixed, operator-reviewed commands against *three specified local Git worktrees*, their exact remediation branches, and GitHub origin identities. A nonzero exit, missing executable, wrong branch, incorrect origin, missing checkout, or source modification during checks causes a failed or blocked run. The origin check verifies local Git configuration, **not cryptographic repository provenance**. No fabricated green status.
- **Watch:** monitor local branch, commit SHA, tracked modifications, untracked file signatures and manifest SHA every 120 seconds (configurable, minimum 10 seconds), rerun when the observed state changes; an initial run is performed at startup. It does **not** pull, reset, merge, deploy, or mutate source control.
- **Evidence:** save every command's real combined output in local log files (4 MiB cap per check) and a timestamped JSON report. SHA-256 digests detect accidental local corruption; evidence verification also rejects symlinked log/report files. These are **not** cryptographically signed or immutable external attestations.
- **Verify:** recompute report/log hashes and flag modification.
- **Isolation by default:** child process receives a small allowlisted environment (no Stripe, GitHub, database or other credentials). Commands use argument lists, never `shell=True`, and are limited to Python, Node, npm. The runner *is not a secure sandbox*: test code can read local files and reach network if the host allows it. **Run only trusted checked-out code**; for untrusted contributions use a disposable container/VM with network disabled and read-only mounts.
- **No deployments or CI bypass:** reports explicitly include `deployment_performed: false`. Passing local checks does not override GitHub's blocked status and is not production authorization.

## Prerequisites

Python 3.11+; Git CLI; Node.js for the Unified Platform. For the `full` profile, install each repository's real development dependencies into isolated environments first. The runner itself uses Python's standard library only.

## Prepare your own checkouts

Create a folder on your laptop/Ubuntu installation and check out these three existing draft PR branches. For public repos, the following is sufficient when GitHub read access is available (private repos require your **existing** authenticated Git credentials):

```bash
mkdir -p "$HOME/garcar-workspaces"
cd "$HOME/garcar-workspaces"
git clone --branch fix/remove-synthetic-production-claims-20261106 https://github.com/Garrettc123/garcar-enterprise-production.git
git clone --branch fix/no-synthetic-dashboard-20261106 https://github.com/Garrettc123/enterprise-unified-platform.git
git clone --branch fix/no-fake-orchestration-20261106 https://github.com/Garrettc123/zero-human-enterprise-grid.git
```

To use the runner code, clone **this** repository at branch `fix/real-independent-ci-20261106`, or use the downloadable offline bundle. Run from inside the runner repo.

```bash
python3 -m unittest discover -s tests -v
python3 src/cicd_orchestrator.py run --workspace "$HOME/garcar-workspaces" --profile smoke
python3 src/cicd_orchestrator.py verify ci-results/ACTUAL_RESULT_DIRECTORY
python3 src/cicd_orchestrator.py run --workspace "$HOME/garcar-workspaces" --profile full
python3 src/cicd_orchestrator.py watch --workspace "$HOME/garcar-workspaces" --profile smoke --interval 120
```

The `run` command prints the actual result directory to verify. Exit code `0` is possible **only when every declared check ran, returned zero, the checkout was clean before and after, the local Git origin matched the allowed repository, and the expected branch and commit remained unchanged**. Missing checkouts fail. If `--allow-dirty` is used for development, the resulting report cannot claim a verified pass. The configuration uses manifest schema version 2 and refuses schema-1 inputs until migrated.

## How to operate independent of GitHub Actions

Keep your laptop, Ubuntu host or user-controlled worker running with `watch`. Store `ci-results/` on persistent storage, back it up, and review reports before any merge. Repository source may still be hosted on GitHub; its Actions billing lock does not affect local test execution after code is checked out. If GitHub becomes inaccessible, previously cloned worktrees remain testable.

To validate integration tests and builds, use the `full` profile. Installing dependencies may require network; this runner will **not** pretend a missing dependency is a passing test. It will exit nonzero.

## Limitations and safety

This system does not automatically run on this chat server, on GitHub or on Railway. An always-on host is required for continuous execution. No paid hosting was created. It does not prove deployed services are healthy, receive Stripe payments or satisfy every security criterion. Never run a test manifest from an untrusted PR and never put credentials inside a manifest. `--watch` watches checked-out commits, not remote GitHub branches.

## Optional persistent monitoring on Ubuntu / systemd

The checked-in unit definition is at `deploy/garcar-ci.service`. This is **not** an installed or running service. It assumes the runner itself is at `$HOME/intelligent-ci-cd-orchestrator` and the three checkouts are in `$HOME/garcar-workspaces`. Copy the unit to the user systemd folder only after those directories and Python/Node runtimes exist:

```bash
mkdir -p "$HOME/.config/systemd/user"
cp deploy/garcar-ci.service "$HOME/.config/systemd/user/garcar-ci.service"
systemctl --user daemon-reload
systemctl --user enable --now garcar-ci.service
systemctl --user status garcar-ci.service
journalctl --user -u garcar-ci.service --no-pager -n 50
```

The service runs in your own user account and writes evidence to `$HOME/garcar-ci-evidence`. It has **no auto-pull, auto-merge, deploy, external callback, email, payment, or cloud billing functionality**. `enable` only sets up user-session startup; remaining available after logout may require an authorized host-level linger setting. The unit is specific to Linux systemd, **not Termux**. On Android/Termux, use a foreground `watch` run or a user-managed Termux service only after the environment is provisioned.

### Production qualification gate

This runner's unit tests are **not the application suites** for the three Garcar PRs. Those suites cannot be declared green until their actual source is checked out and corresponding test commands have succeeded with logs. A clean local runner also does not resolve GitHub's billing lock or bypass a required GitHub check. Keep all three remediation PRs in draft, and never use this system to claim deployed services or customer revenue.

### Threat boundaries

The runner does not create a VM/container, remove network access, or guarantee a test process cannot access local files. Even with scrubbed inherited environment variables, a malicious test can read files available to the host OS user. Use a dedicated unprivileged machine account or disposable container for untrusted code. Git origin matching is a local config consistency check, not proof of source trust. The report SHA-256 file is **not signed**; attackers able to rewrite the report and checksum can falsify them both. Keep evidence on protected storage with suitable backups.
## Run the actual GitHub remediation branches without manual checkouts

The new `src/launch.py` clones all three approved public remediation branches into a *fresh disposable workspace* on each run. It verifies repository remote, branch, SHA and test results; only the hash-checked evidence survives. This is independent of GitHub Actions and needs no GitHub token.

Run from the repository's `fix/real-independent-ci-20261106` branch on your Ubuntu/Termux or laptop host:

```bash
python3 -m unittest discover -s tests -v
python3 src/launch.py --profile smoke --output "$HOME/garcar-ci-evidence"
# Continuous polling while this host is powered on:
python3 src/launch.py --profile smoke --output "$HOME/garcar-ci-evidence" --interval 900
```

A zero exit code is possible only when *every* configured check genuinely ran and passed. Clone failures, missing tools and source mismatches produce a nonzero result with a blocked/failed report. The `full` profile needs app dependencies installed separately; the smoke profile uses Python and Node.js.

### Optional Railway worker (additional usage charges may apply)

The `Dockerfile` uses a non-root Python/Node/Git image. A separate staging Railway service could run the default one-shot command on a cron or use:

```bash
python src/launch.py --profile smoke --output /home/runner/ci-results --interval 900
```

Do **not** repoint Garcar production/revenue services. Persistent evidence requires a provider volume; without one, only stdout summaries survive restarts. A CI service and 512 MB evidence volume have been **staged only** inside the existing Railway `garcar-review-staging` project. They are not provisioned, running or billing until the owner explicitly applies the staged changes. No app was deployed.

### Ubuntu / Termux startup

An optional Linux user unit is in `deploy/garcar-ci-remote.service`. It runs every 15 minutes; first create `$HOME/garcar-ci-evidence` and install the unit via your normal systemd user manager. On Termux, install Python, Git and Node.js, then run the continuous command in a foreground session. Android battery optimization can stop background processes.

### Validation boundary

The bootstrap passed isolated tests using a real temporary Git repository and deliberate clone failures. Full app checkouts were unavailable to the model-hosted execution environment (GitHub DNS is inaccessible there), so passing bootstrap tests **do not** mean PR #37, PR #92 or PR #5 application suites passed. No merge or production deployment authorization is implied.

### Prepared Railway staging configuration — awaiting activation

The GitHub source branch is pinned to a review commit in Railway staging. The service is called `garcar-independent-ci`; it is a **staged-create** resource, not live. The 15-minute cron and Dockerfile settings are read from `railway.json` when deployed. A 512 MB evidence volume is staged at `/home/runner/ci-results`.

Railway mounts volumes as root, so only the image's entrypoint performs a brief `chown` and then uses `gosu` to run every clone/test subprocess as unprivileged user `runner`. The staged Railway variable `RAILWAY_RUN_UID=0` is necessary for this permission setup. We deliberately leave out `deploy.startCommand`, because Railway Docker deployments **replace** `ENTRYPOINT` when a custom start command is set, and that would bypass privilege dropping.

Activation would create usage-based compute and volume charges; **do not apply staged changes without explicit budget authorization**. There is no live CI status to report from Railway yet.

## Private-repository CI (GitHub Actions independent)

The reviewed runner can now validate **one explicitly approved private Python PR commit** in a fresh workspace with no repository writes. This is useful when GitHub Actions reports `startup_failure`. The target is configured at runtime rather than publishing the name/branch/SHA of a private repository in this public CI codebase.

Runtime configuration names (set in your host's protected environment or secret manager):

- `GARCAR_CI_TARGET_REPOSITORY`: exact `Garrettc123/<repository>` identity, scoped to an owner-controlled private repo.
- `GARCAR_CI_TARGET_BRANCH`: reviewed PR head branch, not user-supplied from a web request.
- `GARCAR_CI_TARGET_SHA`: exact lowercase 40-hex approved commit SHA. Any branch movement blocks execution before dependency installation.
- `GARCAR_CI_GITHUB_READ_TOKEN`: dedicated, fine-grained, **Contents: read-only** token restricted to the chosen repository, stored only in a host/Railway secret store. This must not be pasted into chat, PRs, workflow logs, or shell history.

The branch is fetched with a temporary Git askpass script and authenticated HTTPS. No token is placed in clone URLs or process argv; the clone helper is removed immediately after Git exits. Dependency installation and tests receive a fresh, explicitly allowlisted environment without GitHub tokens, Stripe secrets, database URLs or deployment credentials. The runner cannot push, merge, update GitHub statuses, deploy, send messages or move money.

Start with a clean, operator-controlled host containing Python 3.11+ and Git. Once the three reviewed target metadata variables and the protected read-only token are configured, execute:

```bash
python3 -m unittest discover -s tests -v
python3 src/private_pr_ci.py --profile smoke --output "$HOME/garcar-ci-evidence"
python3 src/private_pr_ci.py --profile full --output "$HOME/garcar-ci-evidence"
```

`smoke` runs actual Python compilation against the verified checkout. `full` creates an isolated disposable Python virtual environment, installs the target repository's pinned `requirements.txt` under a sanitized environment, then runs real compilation and the full pytest suite, preserving hash-verified reports and bounded per-check logs. A missing credential, mismatched commit, failed pip install, failed check or corrupted report **returns nonzero**. Preflight failures write a separate hash-checked `preflight.json`, never a synthetic green run.

**Security boundary:** test code and `pip` dependencies still execute real code, so this is NOT a sandbox. Run only reviewed commits in a dedicated unprivileged user/container with outbound network policy appropriate to the test. Fine-grained GitHub credentials must be read-only and live only during clone. Evidence hashes detect local tampering but are neither cryptographic signatures nor an immutable audit ledger.

**Activation state:** the existing Railway `garcar-independent-ci` service and evidence volume remain STAGED, with no running deployment and no new billing. Its default cron currently targets public approved remediation branches, not these private PRs. Enabling this private path on Railway requires separately configuring the target identity and read-only repository secret, reviewing its command/profile, and explicit approval for usage-based compute/storage charges. Do not enable a new paid runner or merge production PRs automatically.


For the *already staged* Railway worker, the entrypoint also supports a deliberate mode switch. Default is `GARCAR_CI_MODE=public` (the original approved public smoke suite). If an operator later configures `GARCAR_CI_MODE=private`, along with the protected repository, branch, pinned SHA and read-only credential variables, the same worker runs the full private Python suite. It still runs under the unprivileged `runner` identity, never merges or deploys, and needs explicit hosting-cost approval before activation. Do not select private mode with missing credentials; it will produce only a blocked preflight result.

Passing these private CI checks does not repair a GitHub account-level Actions runner failure or automatically satisfy a repository's required GitHub status checks. Keep branch protection and human approval intact.
