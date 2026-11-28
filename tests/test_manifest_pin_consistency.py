"""Guard the reviewed public CI target inventory against silent weakening.

This is a static source regression. It does NOT prove network checkout, app
test success, production readiness, or that the live Railway cron has rerun.
"""
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config" / "garcar-checks.json"
SHA = re.compile(r"^[a-f0-9]{40}$")


class PublicTargetManifest(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.targets = self.payload["repositories"]

    def test_target_identity_and_strict_sha_pins(self):
        self.assertEqual(self.payload["schema_version"], 2)
        self.assertEqual(len(self.targets), 5)
        seen = set()
        for target in self.targets:
            self.assertRegex(target["github_repository"], r"^Garrettc123/[A-Za-z0-9_.-]+$")
            self.assertRegex(target["expected_commit"], SHA)
            self.assertIs(target["private"], False)
            self.assertNotIn(target["github_repository"], seen)
            seen.add(target["github_repository"])
            self.assertTrue(target["branch"].strip())
            self.assertTrue(target["checks"]["smoke"])

    def test_enterprise_main_pin_tracks_verified_read_only_status_commit(self):
        records = [t for t in self.targets
                   if t["github_repository"] == "Garrettc123/enterprise-unified-platform"]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["branch"], "main")
        self.assertEqual(
            records[0]["expected_commit"],
            "b93c77cd6a2368fd40b44f9f4231033df695e3df",
        )

    def test_no_payment_or_deploy_commands(self):
        for target in self.targets:
            for profile in ("smoke", "full"):
                for check in target["checks"][profile]:
                    argv = check["command"]
                    self.assertIsInstance(argv, list)
                    self.assertIn(argv[0], ("python", "node", "npm"))
                    joined = " ".join(argv).lower()
                    for forbidden in ("stripe ", "railway up", "kubectl", "terraform apply",
                                      "git push", "git reset", "curl", "wget", "bash -c"):
                        self.assertNotIn(forbidden, joined)


if __name__ == "__main__":
    unittest.main()
