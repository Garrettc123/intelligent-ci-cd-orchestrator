"""Independent CI executor for trusted Git worktrees.

Runs real checks; requires repository/branch/remote match; records source before
and after, process exit status and bounded logs. Never deploys, merges, or
reports unexecuted checks as successful. Not a container security boundary.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit
import uuid

ALLOWED_EXECUTABLES = {"python", "node", "npm"}
MAX_TIMEOUT = 3600
MAX_LOG_BYTES = 4 * 1024 * 1024
_GITHUB_NAME = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(worktree: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(worktree), *args], capture_output=True,
        text=True, timeout=15, check=True,
    )
    return result.stdout.strip()


def github_identity(remote: str) -> str:
    """Identify only github.com owner/repo remotes; reject embedded credentials.

    This validates local *configuration*, not cryptographic commit provenance.
    """
    if remote.startswith("git@github.com:"):
        identity = remote[len("git@github.com:"):]
    else:
        url = urlsplit(remote)
        if url.scheme not in ("https", "ssh") or url.hostname != "github.com":
            raise ValueError("Origin must point to github.com")
        if url.password or url.query or url.fragment or url.port:
            raise ValueError("Origin includes unsupported URL components")
        if url.username not in (None, "git") or (url.username and url.scheme != "ssh"):
            raise ValueError("Origin contains credentials or unexpected user")
        identity = url.path.lstrip("/")
    if identity.endswith(".git"):
        identity = identity[:-4]
    if not _GITHUB_NAME.fullmatch(identity):
        raise ValueError("Origin is not a canonical GitHub owner/repository URL")
    return identity.casefold()


def resolve_project(workspace: Path, name: str) -> Path:
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError("Repository directory must be a single safe name")
    folder = (workspace / name).resolve()
    if not folder.is_relative_to(workspace.resolve()):
        raise ValueError("Repository path escapes workspace")
    return folder


def validate_command(command: object) -> list[str]:
    if (not isinstance(command, list) or not command or
            not all(isinstance(x, str) and x and "\x00" not in x for x in command)):
        raise ValueError("Check command must be a nonempty argv list")
    if command[0] not in ALLOWED_EXECUTABLES:
        raise ValueError("Unapproved executable: " + command[0])
    if command[0] == "python" and any(x in ("-c", "-i", "--interactive") for x in command):
        raise ValueError("Inline/interactive Python code is not permitted in check manifests")
    if command[0] == "node" and any(x in ("-e", "--eval", "-p", "--print") for x in command):
        raise ValueError("Inline Node code is not permitted in check manifests")
    return command


def load_manifest(path: Path, profile: str) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    doc = json.loads(raw)
    if doc.get("schema_version") != 2:
        raise ValueError("Manifest schema_version must be 2")
    repos = doc.get("repositories")
    if not isinstance(repos, list) or not repos:
        raise ValueError("Manifest must contain at least one repository")
    seen = set()
    for repo in repos:
        if not isinstance(repo, dict) or not isinstance(repo.get("directory"), str):
            raise ValueError("Repository must specify directory")
        directory = repo["directory"]
        if directory in seen:
            raise ValueError("Duplicate repository directory")
        seen.add(directory)
        if not isinstance(repo.get("branch"), str) or not repo["branch"]:
            raise ValueError("Expected branch is required")
        identity = repo.get("github_repository")
        if not isinstance(identity, str) or not _GITHUB_NAME.fullmatch(identity):
            raise ValueError("Every repository must declare canonical github_repository")
        if type(repo.get("private", False)) is not bool:
            raise ValueError("private must be a boolean")
        expected = repo.get("expected_commit")
        if expected is not None and (not isinstance(expected, str)
                                     or not re.fullmatch(r"[0-9a-f]{40}", expected)):
            raise ValueError("expected_commit must be a lowercase 40-character SHA")
        checks = repo.get("checks", {}).get(profile)
        if not isinstance(checks, list) or not checks:
            raise ValueError(f"Missing checks for {directory}: {profile}")
        check_names = set()
        for check in checks:
            if not isinstance(check, dict) or not isinstance(check.get("name"), str) or not check["name"]:
                raise ValueError("Checks require names")
            if check["name"] in check_names:
                raise ValueError("Duplicate check name")
            check_names.add(check["name"])
            validate_command(check.get("command"))
            timeout = check.get("timeout", 180)
            if type(timeout) is not int or not 1 <= timeout <= MAX_TIMEOUT:
                raise ValueError("Check timeout must be 1 to 3600 seconds")
    return repos, sha256(raw)


def _snapshot(folder: Path) -> dict[str, str | bool]:
    """Snapshot tracked/untracked status, branch and HEAD without network I/O."""
    return {
        "branch": git(folder, "branch", "--show-current"),
        "commit": git(folder, "rev-parse", "HEAD"),
        "dirty": bool(git(folder, "status", "--porcelain", "--untracked-files=all")),
    }


def execute_check(folder: Path, check: dict, log_dir: Path) -> dict:
    argv = validate_command(check["command"])
    command = [sys.executable if argv[0] == "python" else argv[0], *argv[1:]]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / (uuid.uuid4().hex + ".log")
    started = utcnow()
    t0 = time.monotonic()
    truncated = False
    with tempfile.TemporaryDirectory(prefix="garcar-ci-home-") as clean_home:
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": clean_home,
            "XDG_CONFIG_HOME": clean_home,
            "CI": "true",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": os.path.join(clean_home, "pycache"),
            "PYTHONNOUSERSITE": "1",
            "LANG": "C.UTF-8",
            "TMPDIR": clean_home,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
        }
        if os.name == "nt":
            for key in ("SYSTEMROOT", "WINDIR", "PATHEXT", "COMSPEC"):
                if key in os.environ:
                    env[key] = os.environ[key]
        executable = command[0]
        if executable != sys.executable and shutil.which(executable, path=env["PATH"]) is None:
            code, status, message = 127, "blocked", "Executable not installed: " + executable
            log_file.write_text(message, encoding="utf-8")
        else:
            with tempfile.TemporaryFile(mode="w+b") as spool:
                try:
                    proc = subprocess.Popen(
                        command, cwd=folder, env=env, stdin=subprocess.DEVNULL,
                        stdout=spool, stderr=subprocess.STDOUT,
                        start_new_session=os.name == "posix",
                    )
                    try:
                        code = proc.wait(timeout=check.get("timeout", 180))
                        status = "passed" if code == 0 else "failed"
                    except subprocess.TimeoutExpired:
                        code, status = 124, "failed"
                        if os.name == "posix":
                            os.killpg(proc.pid, signal.SIGKILL)
                        else:
                            proc.kill()
                        proc.wait(timeout=10)
                except OSError as exc:
                    code, status = 127, "blocked"
                    spool.write(f"Failed to start check: {type(exc).__name__}".encode())
                spool.flush()
                size = spool.tell()
                spool.seek(0)
                with log_file.open("wb") as dst:
                    shutil.copyfileobj(spool, dst, MAX_LOG_BYTES + 1) if size <= MAX_LOG_BYTES else dst.write(spool.read(MAX_LOG_BYTES))
                    if size > MAX_LOG_BYTES:
                        truncated = True
                        dst.write(f"\n[Garcar CI: log truncated at {MAX_LOG_BYTES} bytes; {size} bytes captured in spool]\n".encode())
                    if code == 124:
                        dst.write(f"\n[Garcar CI: command exceeded {check.get('timeout', 180)} second timeout]\n".encode())
    log_digest = hashlib.sha256()
    with log_file.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            log_digest.update(chunk)
    return {
        "name": check["name"], "command": argv, "status": status,
        "exit_code": code, "started_at": started,
        "duration_seconds": round(time.monotonic() - t0, 3),
        "log": log_file.name, "log_sha256": log_digest.hexdigest(),
        "log_truncated": truncated,
    }


def run(manifest: Path, workspace: Path, output: Path, profile: str = "smoke", *, allow_dirty: bool = False) -> tuple[dict, Path]:
    repos, manifest_hash = load_manifest(manifest, profile)
    output = output.resolve()
    run_dir = output / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for repo in repos:
        name = repo["directory"]
        record = {"repository": name, "expected_branch": repo["branch"],
                  "expected_github_repository": repo["github_repository"],
                  "expected_commit": repo.get("expected_commit"),
                  "status": "blocked", "commit": None, "checks": []}
        records.append(record)
        try:
            folder = resolve_project(workspace, name)
            if not folder.is_dir():
                raise ValueError("Checkout missing: " + str(folder))
            before = _snapshot(folder)
            record.update({"commit": before["commit"], "actual_branch": before["branch"],
                           "dirty": before["dirty"]})
            if before["branch"] != repo["branch"]:
                raise ValueError("Expected branch " + repo["branch"] + ", got " + str(before["branch"]))
            if repo.get("expected_commit") and before["commit"] != repo["expected_commit"]:
                raise ValueError("Source HEAD does not match pinned approved commit")
            origin = git(folder, "config", "--get", "remote.origin.url")
            if github_identity(origin) != repo["github_repository"].casefold():
                raise ValueError("Git origin does not match approved repository")
            record["remote_config_verified"] = True
            if before["dirty"] and not allow_dirty:
                raise ValueError("Checkout contains uncommitted changes")
            record["source_verified"] = not before["dirty"]
        except (ValueError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            record["reason"] = "Origin not configured or source not eligible" if isinstance(exc, subprocess.CalledProcessError) else str(exc)
            continue
        for check in repo["checks"][profile]:
            outcome = execute_check(folder, check, run_dir)
            record["checks"].append(outcome)
            if outcome["status"] != "passed":
                break
        try:
            after = _snapshot(folder)
            if after["branch"] != before["branch"] or after["commit"] != before["commit"]:
                record["source_verified"] = False
                record["reason"] = "Source branch or commit changed while tests ran"
            elif after["dirty"]:
                record["source_verified"] = False
                record["reason"] = "Source checkout dirty after checks; results not eligible for verified pass"
            record["dirty_after"] = after["dirty"]
        except (ValueError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            record["source_verified"] = False
            record["reason"] = "Could not verify source state after checks"
        all_passed = len(record["checks"]) == len(repo["checks"][profile]) and all(
            outcome["status"] == "passed" for outcome in record["checks"]
        )
        record["status"] = ("passed" if all_passed and record["source_verified"] else
                            "blocked" if not record["source_verified"] or any(
                                outcome["status"] == "blocked" for outcome in record["checks"]
                            ) else "failed")
    success = bool(records) and all(r["status"] == "passed" and r.get("source_verified", False)
                                    and r.get("remote_config_verified", False) for r in records)
    report = {
        "schema_version": 2, "runner": "garcar-independent-ci", "profile": profile,
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
        if (run_dir / "report.json").is_symlink() or (run_dir / "report.sha256").is_symlink():
            return False
        report_data = (run_dir / "report.json").read_bytes()
        expected = (run_dir / "report.sha256").read_text().split()[0]
        if sha256(report_data) != expected:
            return False
        doc = json.loads(report_data)
        for repo in doc["repositories"]:
            for check in repo["checks"]:
                logname = check["log"]
                if Path(logname).name != logname or not re.fullmatch(r"[a-f0-9]{32}\.log", logname):
                    return False
                if (run_dir / logname).is_symlink():
                    return False
                digest = hashlib.sha256()
                with (run_dir / logname).open("rb") as stream:
                    for chunk in iter(lambda: stream.read(65536), b""):
                        digest.update(chunk)
                if digest.hexdigest() != check["log_sha256"]:
                    return False
        return True
    except (OSError, ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError):
        return False


def watch_state(manifest: Path, workspace: Path, profile: str) -> tuple:
    repos, digest = load_manifest(manifest, profile)
    state = [("manifest", digest)]
    for repo in repos:
        try:
            folder = resolve_project(workspace, repo["directory"])
            snap = _snapshot(folder)
            # Include content changes, not just the commit hash. This detects
            # working-tree edits that keep the same porcelain status category.
            changes = git(folder, "diff", "--binary", "HEAD")
            untracked = git(folder, "ls-files", "--others", "--exclude-standard")
            untracked_info = []
            for file in untracked.splitlines():
                p = folder / file
                try:
                    stat = p.lstat()
                    untracked_info.append((file, stat.st_size, stat.st_mtime_ns))
                except OSError:
                    untracked_info.append((file, "unavailable"))
            fingerprint = sha256((changes + json.dumps(untracked_info)).encode())
            state.append((repo["directory"], snap["branch"], snap["commit"], snap["dirty"], fingerprint))
        except (ValueError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            state.append((repo["directory"], "missing_or_blocked", type(exc).__name__))
    return tuple(state)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Garcar Independent CI: local, fail-closed, no deployment")
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("run", "watch"):
        p = sub.add_parser(action)
        p.add_argument("--manifest", type=Path, default=Path("config/garcar-checks.json"))
        p.add_argument("--workspace", type=Path, required=True)
        p.add_argument("--output", type=Path, default=Path("ci-results"))
        p.add_argument("--profile", choices=("smoke", "full"), default="smoke")
        p.add_argument("--allow-dirty", action="store_true", help="Run experimental checks; never report verified success")
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
                    state = watch_state(args.manifest, args.workspace, args.profile)
                except (ValueError, OSError, json.JSONDecodeError):
                    state = ("invalid_manifest_or_source",)
                if state != last:
                    report, location = run(args.manifest, args.workspace, args.output, args.profile,
                                           allow_dirty=args.allow_dirty)
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