"""Verification of real command execution and honest CI outcomes."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cicd_orchestrator import load_manifest, run, verify, validate_command, resolve_project


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.workspace = self.root / "workspace"
        self.checkout = self.workspace / "sample"
        self.checkout.mkdir(parents=True)
        subprocess.run(["git", "init", "-b", "main", str(self.checkout)], capture_output=True, check=True)
        subprocess.run(["git", "-C", str(self.checkout), "config", "user.email", "ci-test@local.invalid"], check=True)
        subprocess.run(["git", "-C", str(self.checkout), "config", "user.name", "CI Test"], check=True)
        (self.checkout / "test_ok.py").write_text("import unittest\nclass Yes(unittest.TestCase):\n def test_success(self): self.assertTrue(True)\n")
        self.commit()
        self.manifest = self.root / "manifest.json"
        self.write_manifest(["python", "-m", "unittest", "-v", "test_ok"])
        self.output = self.root / "out"

    def commit(self):
        subprocess.run(["git", "-C", str(self.checkout), "add", "-A"], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(self.checkout), "commit", "-m", "test fixture"], check=True, capture_output=True)

    def write_manifest(self, command, branch="main"):
        self.manifest.write_text(json.dumps({"repositories": [{"directory": "sample", "branch": branch,
            "checks": {"smoke": [{"name": "unit", "command": command, "timeout": 30}],
                       "full": [{"name": "unit", "command": command, "timeout": 30}]}}]}))

    def test_real_pass_and_evidence(self):
        report, path = run(self.manifest, self.workspace, self.output)
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["source_verified"])
        self.assertEqual(report["repositories"][0]["checks"][0]["exit_code"], 0)
        self.assertFalse(report["deployment_performed"])
        self.assertTrue(verify(path))

    def test_failure_is_failure(self):
        (self.checkout / "test_ok.py").write_text("import unittest\nclass No(unittest.TestCase):\n def test_failure(self): self.assertEqual(1,2)\n")
        self.commit()
        report, _ = run(self.manifest, self.workspace, self.output)
        self.assertEqual(report["status"], "failed_or_blocked")
        self.assertEqual(report["repositories"][0]["status"], "failed")

    def test_wrong_branch_blocks(self):
        self.write_manifest(["python", "-m", "unittest", "-v", "test_ok"], branch="feature")
        report, _ = run(self.manifest, self.workspace, self.output)
        self.assertEqual(report["repositories"][0]["status"], "blocked")
        self.assertEqual(report["repositories"][0]["checks"], [])

    def test_dirty_worktree_is_never_verified(self):
        (self.checkout / "dirty.txt").write_text("dirty")
        report, _ = run(self.manifest, self.workspace, self.output)
        self.assertEqual(report["repositories"][0]["status"], "blocked")
        report, _ = run(self.manifest, self.workspace, self.output, allow_dirty=True)
        self.assertEqual(report["repositories"][0]["status"], "passed")
        self.assertEqual(report["status"], "failed_or_blocked")

    def test_tampering_detected(self):
        _, path = run(self.manifest, self.workspace, self.output)
        record = json.loads((path / "report.json").read_text())
        log = record["repositories"][0]["checks"][0]["log"]
        (path / log).write_text("not the original output")
        self.assertFalse(verify(path))

    def test_manifest_rejects_shell_commands(self):
        with self.assertRaises(ValueError):
            validate_command(["bash", "-c", "exit 0"])
        with self.assertRaises(ValueError):
            validate_command(["python", "-c", "print('forbidden')"])
        self.write_manifest(["bash", "-c", "exit 0"])
        with self.assertRaises(ValueError):
            load_manifest(self.manifest, "smoke")

    def test_path_traversal_denied(self):
        for directory in ("../outside", "/tmp", "..", "."):
            with self.subTest(directory=directory), self.assertRaises(ValueError):
                resolve_project(self.workspace, directory)


if __name__ == "__main__":
    unittest.main()
