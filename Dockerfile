# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

# libgomp is torch's OpenMP runtime; libglib/libgl are needed only by OpenCV,
# which comes with the optional 'yolo' extra. Installing them unconditionally
# costs a few megabytes and saves a confusing ImportError when the extra is
# switched on at build time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 libglib2.0-0 libgl1 curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Which extras to install. Default is the slim build with the classic
# silhouette backend; pass --build-arg EXTRAS=[yolo] for the production
# detector.
ARG EXTRAS=""

# torch's default PyPI wheel bundles the CUDA runtime and is several gigabytes.
# On a CPU-only host the CPU index gives the same package at a fraction of the
# size — but it is a separate host, so it stays opt-in rather than breaking
# builds behind a restrictive egress policy.
ARG TORCH_INDEX_URL=""

COPY pyproject.toml README.md LICENSE ./
COPY arcbody ./arcbody

RUN if [ -n "$TORCH_INDEX_URL" ]; then \
        pip install --index-url "$TORCH_INDEX_URL" torch; \
    fi \
    && pip install ".${EXTRAS}"

# Model weights. Baked in when present so the container needs no network at
# runtime; without them the service starts with an untrained encoder and says
# so on /healthz.
COPY weights ./weights

# SQLite lives here. Mount a volume over it to keep enrolled profiles.
RUN mkdir -p /app/var
ENV ARCBODY_GALLERY__DATABASE_PATH=/app/var/arcbody.sqlite3 \
    ARCBODY_EMBEDDING__WEIGHTS_PATH=/app/weights/arcbody.pt

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "arcbody.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
