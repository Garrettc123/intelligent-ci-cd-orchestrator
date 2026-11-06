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