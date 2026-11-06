#!/bin/sh
set -eu
# Railway volumes are root-owned; only this permissions setup runs as root.
# Source checkout and Python/Node tests execute under an unprivileged UID.
if [ "$(id -u)" -eq 0 ]; then
  mkdir -p /home/runner/ci-results
  chown runner:runner /home/runner/ci-results
  exec gosu runner "$@"
fi
exec "$@"
