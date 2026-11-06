"""Verification of real command execution and honest CI outcomes."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cicd_orchestrator as ci
from cicd_orchestrator import load_manifest, run, verify, validate_command, resolve_project, github_identity, watch_state


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
        subprocess.run(["git", "-C", str(self.checkout), "remote", "add", "origin",
                        "https://github.com/Garrettc123/sample.git"], check=True)
        self.manifest = self.root / "manifest.json"
        self.write_manifest(["python", "-m", "unittest", "-v", "test_ok"])
        self.output = self.root / "out"

    def commit(self):
        subprocess.run(["git", "-C", str(self.checkout), "add", "-A"], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(self.checkout), "commit", "-m", "test fixture"], check=True, capture_output=True)

    def write_manifest(self, command, branch="main"):
        self.manifest.write_text(json.dumps({"schema_version": 2, "repositories": [{"directory": "sample", "branch": branch,
            "github_repository": "Garrettc123/sample",
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
        self.assertEqual(report["repositories"][0]["status"], "blocked")
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


    def test_origin_mismatch_blocks_check(self):
        subprocess.run(["git", "-C", str(self.checkout), "remote", "set-url", "origin",
                        "https://github.com/SomeoneElse/sample.git"], check=True)
        report, _ = run(self.manifest, self.workspace, self.output)
        self.assertEqual(report["status"], "failed_or_blocked")
        self.assertEqual(report["repositories"][0]["status"], "blocked")
        self.assertEqual(report["repositories"][0]["checks"], [])

    def test_bad_origin_credentials_rejected(self):
        with self.assertRaises(ValueError):
            github_identity("https://login:password@github.com/Garrettc123/sample.git")
        self.assertEqual(github_identity("git@github.com:Garrettc123/sample.git"), "garrettc123/sample")
        self.assertEqual(github_identity("ssh://git@github.com/Garrettc123/sample.git"), "garrettc123/sample")

    def test_postcheck_source_mutation_not_accepted(self):
        (self.checkout / "tracked.txt").write_text("original")
        (self.checkout / "test_ok.py").write_text(
            "import unittest\nfrom pathlib import Path\nclass Change(unittest.TestCase):\n"
            " def test_write(self): Path('tracked.txt').write_text('modified')\n")
        self.commit()
        report, path = run(self.manifest, self.workspace, self.output)
        record = report["repositories"][0]
        self.assertEqual(record["checks"][0]["status"], "passed")
        self.assertEqual(record["status"], "blocked")
        self.assertFalse(record["source_verified"])
        self.assertTrue(verify(path))

    def test_watcher_observes_worktree_and_manifest(self):
        initial = watch_state(self.manifest, self.workspace, "smoke")
        (self.checkout / "test_ok.py").write_text("print('changed')\n")
        changed = watch_state(self.manifest, self.workspace, "smoke")
        self.assertNotEqual(initial, changed)
        (self.checkout / "test_ok.py").write_text("print('different')\n")
        again = watch_state(self.manifest, self.workspace, "smoke")
        self.assertNotEqual(changed, again)
        self.write_manifest(["python", "-m", "unittest", "-v", "test_ok"], branch="newbranch")
        self.assertNotEqual(again, watch_state(self.manifest, self.workspace, "smoke"))

    def test_timeout_not_reported_as_pass(self):
        (self.checkout / "test_ok.py").write_text(
            "import time\nimport unittest\nclass Slow(unittest.TestCase):\n"
            " def test_sleep(self): time.sleep(15)\n")
        self.commit()
        doc = json.loads(self.manifest.read_text())
        for profile in ("smoke", "full"):
            doc["repositories"][0]["checks"][profile][0]["timeout"] = 1
        self.manifest.write_text(json.dumps(doc))
        report, _ = run(self.manifest, self.workspace, self.output)
        self.assertEqual(report["repositories"][0]["checks"][0]["exit_code"], 124)
        self.assertEqual(report["status"], "failed_or_blocked")

    def test_manifest_requires_repository_identity(self):
        doc = json.loads(self.manifest.read_text())
        del doc["repositories"][0]["github_repository"]
        self.manifest.write_text(json.dumps(doc))
        with self.assertRaises(ValueError):
            load_manifest(self.manifest, "smoke")

    def test_compilation_keeps_checkout_clean(self):
        self.write_manifest(["python", "-m", "compileall", "-q", "test_ok.py"])
        report, _ = run(self.manifest, self.workspace, self.output)
        self.assertEqual(report["status"], "passed")
        self.assertFalse((self.checkout / "__pycache__").exists())

    def test_evidence_truncation_still_verifies(self):
        old = ci.MAX_LOG_BYTES
        ci.MAX_LOG_BYTES = 128
        self.addCleanup(setattr, ci, "MAX_LOG_BYTES", old)
        (self.checkout / "test_ok.py").write_text(
            "import unittest\nclass Loud(unittest.TestCase):\n"
            " def test_loud(self): print('L' * 10000)\n")
        self.commit()
        report, path = run(self.manifest, self.workspace, self.output)
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["repositories"][0]["checks"][0]["log_truncated"])
        self.assertTrue(verify(path))

    def test_symlinked_evidence_rejected(self):
        report, path = run(self.manifest, self.workspace, self.output)
        logfile = path / report["repositories"][0]["checks"][0]["log"]
        target = path / "backup"
        logfile.rename(target)
        try:
            logfile.symlink_to(target)
        except (OSError, NotImplementedError):
            self.skipTest("Symlinks unsupported by this host")
        self.assertFalse(verify(path))


if __name__ == "__main__":
    unittest.main()
