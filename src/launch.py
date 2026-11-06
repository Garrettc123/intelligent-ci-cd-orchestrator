"""Garcar independent CI: fetch approved PR branches and execute real tests.

No GitHub Actions, secrets, synthetic test outcomes, merges or deployments.
Disposable source checkout per cycle; only evidence reports survive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from cicd_orchestrator import load_manifest, run, verify


def _clone(repo: dict, workspace: Path, *, timeout: int = 120) -> None:
    owner_repo = repo["github_repository"]
    branch = repo["branch"]
    target = workspace / repo["directory"]
    if target.exists() or target.is_symlink():
        raise RuntimeError(f"Refusing to overwrite existing checkout: {target}")
    if not all(isinstance(x, str) and x for x in (owner_repo, branch)):
        raise ValueError("Invalid GitHub source or branch")
    url = f"https://github.com/{owner_repo}.git"
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", "/tmp"),
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GCM_INTERACTIVE": "Never",
        "LANG": "C.UTF-8",
    }
    result = subprocess.run(
        ["git", "-c", "credential.helper=", "clone", "--depth", "1",
         "--single-branch", "--branch", branch, "--", url, str(target)],
        env=env, cwd=workspace, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, timeout=timeout, check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"Git clone failed for {owner_repo}/{branch}: exit {result.returncode}"
        )


def launch_once(manifest: Path, output: Path, profile: str = "smoke") -> tuple[dict, Path]:
    """Freshly clone approved public branches, then check true command results."""
    repos, _ = load_manifest(manifest, profile)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="garcar-ci-checkouts-") as temporary:
        workspace = Path(temporary)
        clone_errors: dict[str, str] = {}
        for repo in repos:
            try:
                _clone(repo, workspace)
            except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
                clone_errors[repo["directory"]] = f"{type(exc).__name__}: {exc}"
                # The missing checkout will be BLOCKED by the core runner.
        report, location = run(manifest, workspace, output, profile)
    if clone_errors:
        report["source_fetch_errors"] = clone_errors
        data = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
        (location / "report.json").write_bytes(data)
        (location / "report.sha256").write_text(
            hashlib.sha256(data).hexdigest() + "  report.json\n"
        )
    if not verify(location):
        raise RuntimeError("CI evidence integrity verification failed")
    return report, location


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Garcar CI: real smoke checks on approved public GitHub branches"
    )
    parser.add_argument("--manifest", type=Path, default=Path("config/garcar-checks.json"))
    parser.add_argument("--output", type=Path, default=Path("ci-results"))
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--interval", type=int, default=0, help="Seconds between fresh runs; 0 = once"
    )
    args = parser.parse_args(argv)
    if args.interval and args.interval < 300:
        parser.error("Continuous interval must be at least 300 seconds")
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, stop)
    while not stopping:
        try:
            report, location = launch_once(args.manifest, args.output, args.profile)
            event = {
                "runner": "garcar-independent-ci",
                "status": report["status"],
                "report_directory": str(location),
                "evidence_verified": True,
                "deployment_performed": False,
                "repositories": [
                    {
                        "name": record["repository"],
                        "status": record["status"],
                        "commit": record.get("commit"),
                        "checks": [
                            {
                                "name": check["name"], "status": check["status"],
                                "exit_code": check["exit_code"],
                            }
                            for check in record["checks"]
                        ],
                    }
                    for record in report["repositories"]
                ],
            }
            print(json.dumps(event, sort_keys=True), flush=True)
            exit_code = 0 if report["status"] == "passed" else 1
        except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
            print(json.dumps({
                "runner": "garcar-independent-ci",
                "status": "failed_or_blocked",
                "error": f"{type(exc).__name__}: {exc}",
                "deployment_performed": False,
            }), flush=True)
            exit_code = 1
        if not args.interval:
            return exit_code
        next_at = time.monotonic() + args.interval
        while not stopping and time.monotonic() < next_at:
            time.sleep(min(1, max(.05, next_at - time.monotonic())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
