"""Offline tests for pinned private CI source, scoped auth, honest evidence."""
from __future__ import annotations

import json
import os
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
import private_pr_ci as private_ci
from cicd_orchestrator import load_manifest, verify


class PrivateCIContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = {
            "GARCAR_CI_TARGET_REPOSITORY": "Garrettc123/fixture-private",
            "GARCAR_CI_TARGET_BRANCH": "security/fixed-sha",
            "GARCAR_CI_TARGET_SHA": "a" * 40,
        }

    def test_target_validation_and_no_embedded_secrets(self):
        target = private_ci.validated_target(self.env)
        self.assertEqual(target["github_repository"], "Garrettc123/fixture-private")
        self.assertTrue(target["private"])
        self.assertEqual(target["expected_commit"], "a" * 40)
        self.assertNotIn("token", json.dumps(target).lower())
        doc = private_ci.make_manifest(target)
        path = self.root / "manifest.json"
        path.write_text(json.dumps(doc))
        self.assertEqual(load_manifest(path, "full")[0][0]["expected_commit"], "a" * 40)

    def test_invalid_sha_and_source_rejected(self):
        for field, bad in (
            ("GARCAR_CI_TARGET_SHA", "main"),
            ("GARCAR_CI_TARGET_SHA", "A" * 40),
            ("GARCAR_CI_TARGET_REPOSITORY", "OtherOwner/fixture-private"),
            ("GARCAR_CI_TARGET_REPOSITORY", "Garrettc123/../../secret"),
            ("GARCAR_CI_TARGET_BRANCH", "../main"),
            ("GARCAR_CI_TARGET_BRANCH", "main..evil"),
        ):
            with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                private_ci.validated_target({**self.env, field: bad})

    def test_bootstrap_environment_drops_read_credentials(self):
        with patch.dict(os.environ, {
            "GARCAR_CI_GITHUB_READ_TOKEN": "private-ci-fake-token",
            "STRIPE_SECRET_KEY": "test-only-fake-key",
        }):
            clean = private_ci.scrubbed_env(self.root)
        self.assertNotIn("GARCAR_CI_GITHUB_READ_TOKEN", clean)
        self.assertNotIn("STRIPE_SECRET_KEY", clean)
        self.assertEqual(clean["GIT_TERMINAL_PROMPT"], "0")

    def test_private_clone_fails_closed_when_token_absent(self):
        folder = self.root / "checkout"
        folder.mkdir()
        target = private_ci.validated_target(self.env)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "not configured"):
                launch._clone(target, folder)

    def test_private_git_token_stays_out_of_url_and_argv(self):
        workspace = self.root / "checkout"
        workspace.mkdir()
        target = private_ci.validated_target(self.env)
        token = "secret-should-not-be-in-argv-1234567890"
        captured = {}
        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            captured["env"] = kwargs["env"]
            script = Path(kwargs["env"]["GIT_ASKPASS"])
            captured["script"] = script.read_text()
            self.assertTrue(script.exists())
            return subprocess.CompletedProcess(argv, 0, "", "")
        with patch.dict(os.environ, {"GARCAR_CI_GITHUB_READ_TOKEN": token}, clear=False):
            with patch.object(launch.subprocess, "run", side_effect=fake_run):
                launch._clone(target, workspace)
        self.assertNotIn(token, " ".join(captured["argv"]))
        self.assertNotIn(token, captured["script"])
        self.assertEqual(captured["env"]["GARCAR_CI_GITHUB_READ_TOKEN"], token)
        self.assertNotIn(token, "https://github.com/Garrettc123/fixture-private.git")
        self.assertFalse(Path(captured["env"]["GIT_ASKPASS"]).exists())

    def test_preflight_receipt_is_reproducible_and_not_a_fake_pass(self):
        target = private_ci.validated_target(self.env)
        output = self.root / "receipts"
        output.mkdir()
        receipt = private_ci.record_preflight(output, target, "full", "checkout", "Credential absent", None)
        data = (receipt / "preflight.json").read_bytes()
        expected = (receipt / "preflight.sha256").read_text().split()[0]
        import hashlib
        self.assertEqual(hashlib.sha256(data).hexdigest(), expected)
        self.assertEqual(json.loads(data)["status"], "blocked")
        self.assertFalse(json.loads(data)["deployment_performed"])

    def test_smoke_run_executes_real_python_check_and_reports_pass(self):
        target = private_ci.validated_target(self.env)
        fixture = self.root / "fixture"
        fixture.mkdir()
        subprocess.run(["git", "init", "-b", self.env["GARCAR_CI_TARGET_BRANCH"], str(fixture)],
                       check=True, capture_output=True)
        for k, value in (("user.email", "ci@local.invalid"), ("user.name", "CI Fixture")):
            subprocess.run(["git", "-C", str(fixture), "config", k, value], check=True)
        subprocess.run(["git", "-C", str(fixture), "remote", "add", "origin",
                        "https://github.com/Garrettc123/fixture-private.git"], check=True)
        (fixture / "main.py").write_text("print('fixture')\n")
        (fixture / "sync").mkdir()
        (fixture / "sync" / "__init__.py").write_text("")
        (fixture / "tests").mkdir()
        (fixture / "tests" / "test_fixture.py").write_text("def test_truth(): assert True\n")
        subprocess.run(["git", "-C", str(fixture), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(fixture), "commit", "-m", "fixture"], check=True, capture_output=True)
        sha = subprocess.run(["git", "-C", str(fixture), "rev-parse", "HEAD"],
                             check=True, capture_output=True, text=True).stdout.strip()
        target["expected_commit"] = sha
        def clone_fixture(_target, workspace, timeout=120):
            shutil.copytree(fixture, workspace / target["directory"])
        output = self.root / "out"
        with patch.object(private_ci, "_clone", side_effect=clone_fixture):
            exit_code = private_ci.execute(target, "smoke", output)
        self.assertEqual(exit_code, 0)
        receipts = list(output.glob("*/report.json"))
        self.assertEqual(len(receipts), 1)
        self.assertTrue(verify(receipts[0].parent))
        report = json.loads(receipts[0].read_text())
        self.assertEqual(report["repositories"][0]["commit"], sha)
        self.assertEqual(report["repositories"][0]["status"], "passed")
        self.assertFalse(report["deployment_performed"])


if __name__ == "__main__":
    unittest.main()
