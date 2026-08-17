# One image serves both the API and the ingestion worker; docker-compose picks
# which by overriding the command. They share every dependency and every model,
# so building twice would only double the image cache.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    # Model weights land here; compose mounts a volume so they download once.
    HF_HOME=/models/hf \
    XDG_CACHE_HOME=/models/cache \
    PATH="/app/.venv/bin:$PATH"

# libgomp is the OpenMP runtime that onnxruntime (fastembed) and CTranslate2
# (faster-whisper) both link against. Nothing else is needed: there is no torch
# in the default install, and PyMuPDF and PyAV ship self-contained wheels, so
# no ffmpeg or build toolchain either.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.24 /uv /uvx /bin/

WORKDIR /app

# Dependencies first, so editing source doesn't invalidate the install layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 1001 app \
    && mkdir -p /data /models \
    && chown -R app:app /app /data /models
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "rag.main:app", "--host", "0.0.0.0", "--port", "8000"]
