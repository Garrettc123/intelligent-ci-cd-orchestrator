FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends git nodejs npm ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 10001 runner
WORKDIR /app
COPY --chown=runner:runner src/ ./src/
COPY --chown=runner:runner config/ ./config/
COPY --chown=runner:runner tests/ ./tests/
USER runner
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
CMD ["python", "src/launch.py", "--profile", "smoke", "--output", "/home/runner/ci-results"]
