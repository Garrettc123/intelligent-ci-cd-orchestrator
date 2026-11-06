"""Garcar independent CI: genuine local checks, never synthetic outcomes.

Runs only trusted, checked-out worktrees with an operator-reviewed manifest.
Does not create GitHub status checks, make deployments, send mail, or charge.
Uses Python standard library only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid

ALLOWED_EXECUTABLES = {"python", "node", "npm"}
MAX_TIMEOUT = 3600


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(worktree: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(worktree), *args],
        capture_output=True, text=True, timeout=15, check=True,
    )
    return result.stdout.strip()


def resolve_project(workspace: Path, name: str) -> Path:
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError("Repository directory must be a single safe name")
    folder = (workspace / name).resolve()
    if not folder.is_relative_to(workspace.resolve()):
        raise ValueError("Repository path escapes workspace")
    return folder


def validate_command(command: object) -> list[str]:
    if (not isinstance(command, list) or not command or
            not all(isinstance(x, str) and x for x in command)):
        raise ValueError("Check command must be a nonempty argv list")
    if command[0] not in ALLOWED_EXECUTABLES:
        raise ValueError("Unapproved executable: " + command[0])
    if command[0] == "python" and "-c" in command:
        raise ValueError("Inline Python code is not permitted in check manifests")
    if command[0] == "node" and ("-e" in command or "--eval" in command):
        raise ValueError("Inline Node code is not permitted in check manifests")
    return command


def load_manifest(path: Path, profile: str) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    doc = json.loads(raw)
    repos = doc.get("repositories")
    if not isinstance(repos, list) or not repos:
        raise ValueError("Manifest must contain at least one repository")
    seen = set()
    for repo in repos:
        if not isinstance(repo, dict) or not isinstance(repo.get("directory"), str):
            raise ValueError("Repository must specify directory")
        if repo["directory"] in seen:
            raise ValueError("Duplicate repository directory")
        seen.add(repo["directory"])
        if not isinstance(repo.get("branch"), str) or not repo["branch"]:
            raise ValueError("Expected branch is required")
        checks = repo.get("checks", {}).get(profile)
        if not isinstance(checks, list) or not checks:
            raise ValueError(f"Missing checks for {repo['directory']}: {profile}")
        for check in checks:
            if not isinstance(check, dict) or not check.get("name"):
                raise ValueError("Checks require names")
            validate_command(check.get("command"))
            timeout = check.get("timeout", 180)
            if type(timeout) is not int or not 1 <= timeout <= MAX_TIMEOUT:
                raise ValueError("Check timeout must be 1 to 3600 seconds")
    return repos, sha256(raw)


def execute_check(folder: Path, check: dict, log_dir: Path) -> dict:
    argv = validate_command(check["command"])
    command = [sys.executable if argv[0] == "python" else argv[0], *argv[1:]]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / (uuid.uuid4().hex + ".log")
    started = utcnow()
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="garcar-ci-home-") as clean_home:
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": clean_home,
            "XDG_CONFIG_HOME": clean_home,
            "CI": "true",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "LANG": "C.UTF-8",
            "TMPDIR": clean_home,
        }
        if os.name == "nt":
            for key in ("SYSTEMROOT", "WINDIR", "PATHEXT", "COMSPEC"):
                if key in os.environ:
                    env[key] = os.environ[key]
        executable = command[0]
        if executable != sys.executable and shutil.which(executable, path=env["PATH"]) is None:
            result = (127, "Executable not installed: " + executable, "blocked")
        else:
            try:
                completed = subprocess.run(
                    command, cwd=folder, env=env,
                    capture_output=True, text=True, errors="replace",
                    timeout=check.get("timeout", 180), check=False,
                )
                output = (completed.stdout or "") + "\n" + (completed.stderr or "")
                result = (completed.returncode, output, "passed" if completed.returncode == 0 else "failed")
            except subprocess.TimeoutExpired as exc:
                result = (124, "Timed out after " + str(check.get("timeout", 180)) + "s", "failed")
            except OSError as exc:
                result = (127, f"Runner could not start process: {type(exc).__name__}", "blocked")
    log_bytes = result[1].encode("utf-8")
    log_file.write_bytes(log_bytes)
    return {
        "name": check["name"], "command": argv, "status": result[2],
        "exit_code": result[0], "started_at": started,
        "duration_seconds": round(time.monotonic() - t0, 3),
        "log": str(log_file.name), "log_sha256": sha256(log_bytes),
    }


def run(manifest: Path, workspace: Path, output: Path, profile: str = "smoke", *, allow_dirty: bool = False) -> tuple[dict, Path]:
    repos, manifest_hash = load_manifest(manifest, profile)
    output = output.resolve()
    run_dir = output / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for repo in repos:
        name = repo["directory"]
        record = {"repository": name, "expected_branch": repo["branch"], "status": "blocked", "commit": None, "checks": []}
        records.append(record)
        try:
            folder = resolve_project(workspace, name)
            if not folder.is_dir():
                raise ValueError("Checkout missing: " + str(folder))
            branch = git(folder, "branch", "--show-current")
            commit = git(folder, "rev-parse", "HEAD")
            record["commit"] = commit
            record["actual_branch"] = branch
            if branch != repo["branch"]:
                raise ValueError("Expected branch " + repo["branch"] + ", got " + branch)
            dirty = bool(git(folder, "status", "--porcelain"))
            record["dirty"] = dirty
            if dirty and not allow_dirty:
                raise ValueError("Checkout contains uncommitted changes")
            if dirty:
                record["source_verified"] = False
            else:
                record["source_verified"] = True
        except (ValueError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            record["reason"] = str(exc)
            continue
        for check in repo["checks"][profile]:
            outcome = execute_check(folder, check, run_dir)
            record["checks"].append(outcome)
            if outcome["status"] != "passed":
                break  # fail closed, preserve the first failure
        record["status"] = "passed" if len(record["checks"]) == len(repo["checks"][profile]) and all(x["status"] == "passed" for x in record["checks"]) else "failed"
        if record["status"] == "failed" and any(c["status"] == "blocked" for c in record["checks"]):
            record["status"] = "blocked"
    success = bool(records) and all(r["status"] == "passed" and r.get("source_verified", False) for r in records)
    report = {
        "schema_version": 1, "runner": "garcar-independent-ci", "profile": profile,
        "finished_at": utcnow(), "manifest_sha256": manifest_hash,
        "status": "passed" if success else "failed_or_blocked",
        "source_verified": success, "deployment_performed": False,
        "repositories": records,
    }
    data = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    (run_dir / "report.json").write_bytes(data)
    (run_dir / "report.sha256").write_text(sha256(data) + "  report.json\n")
    return report, run_dir


def verify(run_dir: Path) -> bool:
    try:
        report_data = (run_dir / "report.json").read_bytes()
        expected = (run_dir / "report.sha256").read_text().split()[0]
        if sha256(report_data) != expected:
            return False
        doc = json.loads(report_data)
        for repo in doc["repositories"]:
            for check in repo["checks"]:
                logname = check["log"]
                if Path(logname).name != logname or sha256((run_dir / logname).read_bytes()) != check["log_sha256"]:
                    return False
        return True
    except (OSError, ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError):
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independent CI for trusted Garcar workspaces; no GitHub Actions required")
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("run", "watch"):
        p = sub.add_parser(action)
        p.add_argument("--manifest", type=Path, default=Path("config/garcar-checks.json"))
        p.add_argument("--workspace", type=Path, required=True)
        p.add_argument("--output", type=Path, default=Path("ci-results"))
        p.add_argument("--profile", choices=("smoke", "full"), default="smoke")
        p.add_argument("--allow-dirty", action="store_true", help="Run experimental checks, never report verified success")
        if action == "watch":
            p.add_argument("--interval", type=int, default=120)
    v = sub.add_parser("verify")
    v.add_argument("report_directory", type=Path)
    args = parser.parse_args(argv)
    if args.action == "verify":
        ok = verify(args.report_directory)
        print("Evidence files valid" if ok else "Evidence files invalid or missing")
        return 0 if ok else 1
    if args.action == "watch":
        if args.interval < 10:
            parser.error("Watch interval must be at least 10 seconds")
        last = None
        try:
            while True:
                try:
                    repos, _ = load_manifest(args.manifest, args.profile)
                    state = tuple((r["directory"], git(resolve_project(args.workspace, r["directory"]), "rev-parse", "HEAD")) for r in repos)
                except (ValueError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    state = ("missing_source",)
                if state != last:
                    report, location = run(args.manifest, args.workspace, args.output, args.profile, allow_dirty=args.allow_dirty)
                    print(json.dumps({"status": report["status"], "report": str(location)}, sort_keys=True), flush=True)
                    last = state
                time.sleep(args.interval)
        except KeyboardInterrupt:
            return 130
    report, location = run(args.manifest, args.workspace, args.output, args.profile, allow_dirty=args.allow_dirty)
    print(json.dumps({"status": report["status"], "report": str(location)}, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())