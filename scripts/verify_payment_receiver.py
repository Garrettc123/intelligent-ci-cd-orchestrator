#!/usr/bin/env python3
"""Independently verify the pinned real Garcar payment receiver source.

Runs only the receiver entry from the reviewed multi-repository manifest
using the existing fail-closed clone, source verification, test execution and
SHA-256 evidence machinery. Does not deploy, handle customer data or use
payment credentials.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from launch import launch_once  # noqa: E402


def main() -> int:
    source = json.loads((ROOT / "config" / "garcar-checks.json").read_text())
    targets = [
        r for r in source["repositories"]
        if r.get("github_repository") == "Garrettc123/garcar-emergency-payments"
    ]
    if len(targets) != 1:
        print("Payment receiver target must occur exactly once", file=sys.stderr)
        return 2
    target = targets[0]
    if target.get("private") is not False:
        print("Receiver source must be an approved public repository", file=sys.stderr)
        return 2
    if target.get("branch") != "fix/subscription-invoice-dedupe-20261116" or (
        target.get("expected_commit") !=
        "a5feb1d51c39d7916b78ed8ea4972110e8367717"
    ):
        print("Pinned receiver branch or SHA changed without review", file=sys.stderr)
        return 2
    minimal_manifest = {
        "schema_version": source["schema_version"],
        "repositories": targets,
    }
    with tempfile.TemporaryDirectory(prefix="garcar-payment-ci-") as tmp:
        root = Path(tmp)
        manifest = root / "receiver-manifest.json"
        manifest.write_text(json.dumps(minimal_manifest, indent=2) + "\n")
        report, location = launch_once(manifest, root / "evidence", "smoke")
        record = report["repositories"][0]
        success = (
            report["status"] == "passed"
            and report["source_verified"] is True
            and report["deployment_performed"] is False
            and record["status"] == "passed"
            and record["source_verified"] is True
            and record["remote_config_verified"] is True
            and record["commit"] == target["expected_commit"]
            and len(record["checks"]) == len(target["checks"]["smoke"])
            and all(ch["status"] == "passed" and ch["exit_code"] == 0
                    for ch in record["checks"])
        )
        summary = {
            "repository": target["github_repository"],
            "branch": target["branch"],
            "commit": record.get("commit"),
            "status": "passed" if success else "failed_or_blocked",
            "evidence_verified": True,
            "deployment_performed": False,
            "checks": [
                {"name": c["name"], "status": c["status"],
                 "exit_code": c["exit_code"], "log_sha256": c["log_sha256"]}
                for c in record["checks"]
            ],
            "manifest_sha256": report["manifest_sha256"],
            "evidence_dir": location.name,
        }
        print(json.dumps(summary, sort_keys=True))
        return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
