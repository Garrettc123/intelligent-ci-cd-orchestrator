"""Offline coverage for approved-branch bootstrap and real command execution."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import launch
from cicd_orchestrator import verify


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.origin = self.root / "fixture"
        self.origin.mkdir()
        subprocess.run(
            ["git", "init", "-b", "main", str(self.origin)],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(self.origin), "config", "user.name", "Garcar CI Fixture"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.origin), "config", "user.email", "ci@local.invalid"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.origin), "remote", "add", "origin",
             "https://github.com/Garrettc123/fixture.git"],
            check=True,
        )
        (self.origin / "test_fixture.py").write_text(
            "import unittest\nclass T(unittest.TestCase):\n def test_pass(self): self.assertEqual(2, 2)\n"
        )
        subprocess.run(["git", "-C", str(self.origin), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(self.origin), "commit", "-m", "fixture"],
            check=True, capture_output=True,
        )
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({
            "schema_version": 2,
            "repositories": [{
                "directory": "fixture",
                "github_repository": "Garrettc123/fixture",
                "branch": "main",
                "checks": {"smoke": [{
                    "name": "fixture unit",
                    "command": ["python", "-m", "unittest", "-v", "test_fixture"],
                    "timeout": 30,
                }]},
            }],
        }))
        self.output = self.root / "output"

    def checkout_fixture(self, repo, workspace):
        shutil.copytree(self.origin, workspace / repo["directory"])

    def test_verified_local_fixture_execution(self):
        with patch.object(launch, "_clone", self.checkout_fixture):
            report, location = launch.launch_once(self.manifest, self.output)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["repositories"][0]["checks"][0]["exit_code"], 0)
        self.assertFalse(report["deployment_performed"])
        self.assertTrue(verify(location))

    def test_clone_failure_is_blocked_and_evidence_verifies(self):
        with patch.object(launch, "_clone", side_effect=RuntimeError("git offline")):
            report, location = launch.launch_once(self.manifest, self.output)
        self.assertEqual(report["status"], "failed_or_blocked")
        self.assertEqual(report["repositories"][0]["status"], "blocked")
        self.assertIn("fixture", report["source_fetch_errors"])
        self.assertTrue(verify(location))

    def test_exit_code_matches_actual_checks(self):
        with patch.object(launch, "_clone", self.checkout_fixture):
            status = launch.main([
                "--manifest", str(self.manifest), "--output", str(self.output)
            ])
        self.assertEqual(status, 0)
        with patch.object(launch, "_clone", side_effect=RuntimeError("git offline")):
            status = launch.main([
                "--manifest", str(self.manifest), "--output", str(self.output)
            ])
        self.assertEqual(status, 1)

    def test_fresh_checkout_refuses_replacement(self):
        workspace = self.root / "work"
        (workspace / "fixture").mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "Refusing to overwrite"):
            launch._clone({
                "github_repository": "Garrettc123/fixture",
                "branch": "main",
                "directory": "fixture",
            }, workspace)

    def test_rotates_only_old_verified_reports(self):
        with patch.object(launch, "_clone", self.checkout_fixture):
            for _ in range(14):
                result, location = launch.launch_once(self.manifest, self.output)
                self.assertEqual(result["status"], "passed")
                self.assertTrue(verify(location))
        report_paths = sorted(self.output.glob("*/report.json"))
        self.assertEqual(len(report_paths), 12)

    def test_rotation_preserves_invalid_evidence_for_investigation(self):
        failed = self.output / "20261109T170000Z-01234567"
        failed.mkdir(parents=True)
        (failed / "report.json").write_text("not-json")
        with patch.object(launch, "_clone", self.checkout_fixture):
            for _ in range(13):
                launch.launch_once(self.manifest, self.output)
        self.assertTrue(failed.exists())
        self.assertEqual(len(list(self.output.glob("*/report.json"))), 13)

    def test_short_monitoring_interval_rejected(self):
        with self.assertRaises(SystemExit) as error:
            launch.main([
                "--manifest", str(self.manifest), "--interval", "30"
            ])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
