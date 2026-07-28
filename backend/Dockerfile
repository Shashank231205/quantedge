# Pinned to 3.11 per PRD; several quant libraries lag on newer runtimes.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Dependency layer cached separately from source.
COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install -e . 2>/dev/null || true

COPY . .
RUN pip install -e .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "quantedge.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
