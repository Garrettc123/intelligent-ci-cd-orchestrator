#!/bin/sh
set -eu
# Railway volumes are root-owned; only permissions setup may run as root.
# All repo cloning, dependency setup and tests execute as unprivileged runner.
case "${GARCAR_CI_MODE:-public}" in
  public) ;;
  private)
    # This mode is deliberately opt-in and fails closed without approved
    # target identity, pinned commit and a read-only private repo credential.
    set -- python src/private_pr_ci.py --profile full --output /home/runner/ci-results
    ;;
  *)
    echo "Invalid GARCAR_CI_MODE; permitted: public or private" >&2
    exit 2
    ;;
esac
if [ "$(id -u)" -eq 0 ]; then
  mkdir -p /home/runner/ci-results
  chown runner:runner /home/runner/ci-results
  exec gosu runner "$@"
fi
exec "$@"
