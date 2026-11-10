"""Garcar private-PR CI: immutable checkout, isolated Python test environment.

This tool is independent of GitHub Actions. It has no merge or deploy methods.
It does not grant write-capable credentials to repository tests. Run only vetted
commit SHAs, in a dedicated disposable worker with host/network restrictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

from cicd_orchestrator import git, verify
from launch import _clone

OWNER_REPO = re.compile(r"^Garrettc123/[A-Za-z0-9_.-]+$")
BRANCH = re.compile(r"^[A-Za-z0-9_./-]{1,150}$")
COMMIT = re.compile(r"^[a-f0-9]{40}$")


def validated_target(environment: dict[str, str]) -> dict:
    repository = environment.get("GARCAR_CI_TARGET_REPOSITORY", "")
    branch = environment.get("GARCAR_CI_TARGET_BRANCH", "")
    commit = environment.get("GARCAR_CI_TARGET_SHA", "")
    if not OWNER_REPO.fullmatch(repository) or repository.endswith(".git"):
        raise ValueError("An approved Garrettc123 repository must be configured")
    if not BRANCH.fullmatch(branch) or ".." in branch or branch[0] in "/." or branch.endswith(("/", ".")):
        raise ValueError("An approved Git branch must be configured")
    if not COMMIT.fullmatch(commit):
        raise ValueError("An exact approved lowercase 40-character commit SHA is required")
    return {"directory": repository.split("/", 1)[1], "github_repository": repository,
            "branch": branch, "private": True, "expected_commit": commit}


def make_manifest(target: dict) -> dict:
    """Checks are hardcoded in this reviewed runner; external PRs cannot edit them."""
    return {
        "schema_version": 2,
        "repositories": [{
            **target,
            "checks": {
                "smoke": [{"name": "Python source syntax",
                           "command": ["python", "-m", "compileall", "-q", "main.py", "sync", "tests"],
                           "timeout": 120}],
                "full": [
                    {"name": "Python source syntax",
                     "command": ["python", "-m", "compileall", "-q", "main.py", "sync", "tests"],
                     "timeout": 120},
                    {"name": "Full Python pytest suite",
                     "command": ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                     "timeout": 900},
                ],
            },
        }],
    }


def scrubbed_env(home: Path) -> dict[str, str]:
    """Neither Git read tokens nor production/service secrets reach pip or tests."""
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home),
        "PIP_NO_INPUT": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "LANG": "C.UTF-8",
    }


def record_preflight(output: Path, target: dict, profile: str,
                     stage: str, reason: str, exit_code: int | None) -> Path:
    """Persist a tamper-evident (NOT signed or immutable) blocked run receipt."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with tempfile.TemporaryDirectory(prefix="garcar-ci-receipt-", dir=output) as tmp:
        receipt_dir = output / (stamp + "-preflight-" + Path(tmp).name.rsplit("-", 1)[-1])
        receipt_dir.mkdir(mode=0o700)
    receipt = {
        "kind": "garcar_private_pr_ci_preflight_v1",
        "status": "blocked",
        "profile": profile,
        "stage": stage,
        "reason": reason,
        "exit_code": exit_code,
        "repository": target["github_repository"],
        "expected_branch": target["branch"],
        "expected_commit": target["expected_commit"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "deployment_performed": False,
        "merge_performed": False,
    }
    raw = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    (receipt_dir / "preflight.json").write_bytes(raw)
    (receipt_dir / "preflight.sha256").write_text(hashlib.sha256(raw).hexdigest() + "  preflight.json\n")
    return receipt_dir


def _run_process(argv: list[str], cwd: Path, env: dict[str, str],
                 timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, timeout=timeout, check=False)


def execute(target: dict, profile: str, output: Path) -> int:
    output = output.resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    stage = "checkout"
    with tempfile.TemporaryDirectory(prefix="garcar-ci-private-") as tmp:
        root = Path(tmp)
        workspace = root / "workspace"
        workspace.mkdir()
        checkout = workspace / target["directory"]
        try:
            _clone(target, workspace, timeout=120)
            if git(checkout, "branch", "--show-current") != target["branch"] or \
                    git(checkout, "rev-parse", "HEAD") != target["expected_commit"]:
                raise RuntimeError("Remote source moved or is not the approved commit")
            env = scrubbed_env(root)
            if profile == "full":
                stage = "python-environment"
                venv = root / "venv"
                creation = _run_process([sys.executable, "-m", "venv", str(venv)], root, env, 120)
                if creation.returncode != 0:
                    raise RuntimeError("Isolated Python environment creation failed")
                interpreter = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
                stage = "dependency-install"
                requirements = checkout / "requirements.txt"
                if not requirements.is_file() or requirements.is_symlink():
                    raise RuntimeError("Trusted pinned requirements.txt missing or not a regular file")
                installation = _run_process([
                    str(interpreter), "-m", "pip", "install", "--no-input",
                    "--disable-pip-version-check", "-r", str(requirements)
                ], root, env, 600)
                if installation.returncode != 0:
                    raise RuntimeError("Isolated dependency installation failed (exit %s)" %
                                       installation.returncode)
            else:
                interpreter = Path(sys.executable)
            stage = "application-tests"
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(make_manifest(target)), encoding="utf-8")
            runner = Path(__file__).with_name("cicd_orchestrator.py")
            command = [str(interpreter), str(runner), "run",
                       "--manifest", str(manifest), "--workspace", str(workspace),
                       "--output", str(output), "--profile", profile]
            result = _run_process(command, root, env, 1200)
            # The runner writes report.json and SHA-256 evidence even on a failed test.
            try:
                summary = json.loads(result.stdout.strip().splitlines()[-1])
                evidence = Path(summary["report"])
                verified = evidence.is_dir() and evidence.parent == output and verify(evidence)
            except (IndexError, ValueError, KeyError, TypeError):
                verified = False
            if not verified:
                raise RuntimeError("Runner evidence is missing or failed integrity verification")
            report = json.loads((evidence / "report.json").read_text())
            passed = result.returncode == 0 and summary["status"] == "passed" and \
                report["status"] == "passed" and len(report["repositories"]) == 1 and \
                report["repositories"][0]["commit"] == target["expected_commit"]
            print(json.dumps({
                "runner": "garcar-private-pr-ci", "status": "passed" if passed else "failed_or_blocked",
                "profile": profile, "repository": target["github_repository"],
                "commit": target["expected_commit"], "report_directory": str(evidence),
                "evidence_verified": True, "deployment_performed": False,
                "merge_performed": False,
            }, sort_keys=True), flush=True)
            return 0 if passed else 1
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
            # Never return or log token values, command environment, or raw pip output.
            receipt = record_preflight(output, target, profile, stage,
                                       type(error).__name__ + ": " + str(error), None)
            print(json.dumps({
                "runner": "garcar-private-pr-ci", "status": "blocked",
                "stage": stage, "receipt_directory": str(receipt),
                "deployment_performed": False, "merge_performed": False,
            }, sort_keys=True), flush=True)
            return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a pinned private PR without GitHub Actions")
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--output", type=Path, default=Path("ci-results"))
    args = parser.parse_args(argv)
    try:
        target = validated_target(dict(os.environ))
    except ValueError as error:
        print(json.dumps({"runner": "garcar-private-pr-ci", "status": "blocked",
                          "reason": str(error), "deployment_performed": False}), flush=True)
        return 2
    return execute(target, args.profile, args.output)


if __name__ == "__main__":
    sys.exit(main())
