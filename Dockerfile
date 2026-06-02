FROM python:3.12-slim AS base

# Install ffmpeg and runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e . 2>/dev/null || true

# Copy source
COPY src/ src/
COPY .env.example .env.example

# Install the package properly
RUN pip install --no-cache-dir -e ".[captions]"

# Default: run the API server
EXPOSE 8000
CMD ["openreel", "serve", "--host", "0.0.0.0"]
