# Independent, unprivileged test worker; no customer or payment secrets.
FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
      git nodejs npm ca-certificates gosu \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 10001 runner
WORKDIR /app
COPY --chown=runner:runner src/ ./src/
COPY --chown=runner:runner config/ ./config/
COPY --chown=runner:runner tests/ ./tests/
COPY deploy/entrypoint.sh /usr/local/bin/garcar-entrypoint
RUN chmod 755 /usr/local/bin/garcar-entrypoint
USER runner
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENTRYPOINT ["/usr/local/bin/garcar-entrypoint"]
CMD ["python", "src/launch.py", "--profile", "smoke", "--output", "/home/runner/ci-results"]
