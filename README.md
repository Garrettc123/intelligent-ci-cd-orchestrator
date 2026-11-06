# Garcar Independent CI — verifiable local runner

This replaces the previous simulated `IntelligentCICD` implementation, which used random outcomes and invented deployment success. This runner **executes actual tests** on an operator-controlled laptop, Ubuntu/Termux host, or private worker. It does not depend on GitHub Actions, paid runner minutes, or GitHub's billing system. GitHub is optional source hosting, not the execution engine.

## What is actually implemented

- **Run:** execute fixed, operator-reviewed commands against *three specified local Git worktrees* on their exact remediation branches. A nonzero exit, missing executable, wrong branch, missing checkout, or uncommitted change causes a failed or blocked run. No fabricated green status.
- **Watch:** monitor checked-out commit SHA every 120 seconds (configurable, minimum 10 seconds), rerun when it changes; an initial run is performed at startup. It does **not** pull, reset, merge, deploy, or mutate source control.
- **Evidence:** save every command's real combined output in local log files and a timestamped JSON report. SHA-256 digests detect accidental local corruption. These are **not** cryptographically signed or immutable external attestations.
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

The `run` command prints the actual result directory to verify. Exit code `0` is possible **only when every declared check ran, returned zero, the checkout was clean, and the expected branch was checked out**. Missing checkouts fail. If `--allow-dirty` is used for development, the resulting report cannot claim a verified pass.

## How to operate independent of GitHub Actions

Keep your laptop, Ubuntu host or user-controlled worker running with `watch`. Store `ci-results/` on persistent storage, back it up, and review reports before any merge. Repository source may still be hosted on GitHub; its Actions billing lock does not affect local test execution after code is checked out. If GitHub becomes inaccessible, previously cloned worktrees remain testable.

To validate integration tests and builds, use the `full` profile. Installing dependencies may require network; this runner will **not** pretend a missing dependency is a passing test. It will exit nonzero.

## Limitations and safety

This system does not automatically run on this chat server, on GitHub or on Railway. An always-on host is required for continuous execution. No paid hosting was created. It does not prove deployed services are healthy, receive Stripe payments or satisfy every security criterion. Never run a test manifest from an untrusted PR and never put credentials inside a manifest. `--watch` watches checked-out commits, not remote GitHub branches.