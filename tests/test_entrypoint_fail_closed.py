"""Executable regression coverage for CI worker startup denial paths.

Run only branches that must return before launching any runner, network fetch,
volume bootstrap, or test process. No provider credentials are used.
"""
import os
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "deploy" / "entrypoint.sh"


def rejected(env_vars):
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "GARCAR_CI_MODE": "private",
    }
    env.update(env_vars)
    return subprocess.run(
        ["/bin/sh", str(ENTRYPOINT)],
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=5,
        check=False,
    )


class RailwayEntrypointContract(unittest.TestCase):
    def test_private_without_credential_exits_before_clone(self):
        result = rejected({
            "GARCAR_CI_TARGET_REPOSITORY": "Garrettc123/approved-private",
            "GARCAR_CI_TARGET_BRANCH": "review/pinned",
            "GARCAR_CI_TARGET_SHA": "a" * 40,
        })
        self.assertEqual(result.returncode, 3)
        self.assertIn("read_only_github_credential_not_configured", result.stderr)
        self.assertIn('"deployment_performed":false', result.stderr)
        self.assertEqual(result.stdout, "")

    def test_private_with_credential_but_missing_source_pin_exits(self):
        result = rejected({"GARCAR_CI_GITHUB_READ_TOKEN": "fixture-no-real-token"})
        self.assertEqual(result.returncode, 3)
        self.assertIn("approved_target_identity_or_sha_missing", result.stderr)
        self.assertNotIn("fixture-no-real-token", result.stderr)

    def test_invalid_mode_exits_without_clone(self):
        result = rejected({"GARCAR_CI_MODE": "unrestricted"})
        self.assertEqual(result.returncode, 2)
        self.assertIn("unsupported_ci_mode", result.stderr)
        self.assertNotIn("traceback", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()
