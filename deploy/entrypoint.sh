#!/bin/sh
set -eu
# Railway mounts the evidence volume as root. Bootstrap ownership as root,
# then drop ALL Git fetches, dependency installation and tests to runner.
# No credentials are printed; missing private auth fails closed.
MODE="${GARCAR_CI_MODE:-public}"
OUT="/home/runner/ci-results"
case "$MODE" in
  public)
    ;;
  private)
    if [ -z "${GARCAR_CI_GITHUB_READ_TOKEN:-}" ]; then
      echo '{"runner":"garcar-independent-ci","status":"blocked","reason":"read_only_github_credential_not_configured","deployment_performed":false}' >&2
      exit 3
    fi
    if [ -z "${GARCAR_CI_TARGET_REPOSITORY:-}" ] ||
       [ -z "${GARCAR_CI_TARGET_BRANCH:-}" ] ||
       [ -z "${GARCAR_CI_TARGET_SHA:-}" ]; then
      echo '{"runner":"garcar-independent-ci","status":"blocked","reason":"approved_target_identity_or_sha_missing","deployment_performed":false}' >&2
      exit 3
    fi
    set -- python src/private_pr_ci.py --profile full --output "$OUT"
    ;;
  *)
    echo '{"runner":"garcar-independent-ci","status":"blocked","reason":"unsupported_ci_mode","deployment_performed":false}' >&2
    exit 2
    ;;
esac

if [ "$(id -u)" -eq 0 ]; then
  mkdir -p "$OUT"
  chown runner:runner "$OUT"
  chmod 0700 "$OUT"
  echo '{"runner":"garcar-independent-ci","bootstrap":"volume_owned_by_runner","tests_as_root":false}' >&2
  exec gosu runner "$@"
fi
if [ ! -d "$OUT" ] || [ ! -w "$OUT" ]; then
  echo '{"runner":"garcar-independent-ci","status":"blocked","reason":"evidence_volume_not_writable_by_runner","deployment_performed":false}' >&2
  exit 3
fi
exec "$@"
