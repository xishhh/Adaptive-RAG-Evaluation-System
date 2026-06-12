# ──────────────────────────────────────────────────────────────────────────────
# Adaptive RAG — Dockerfile
# ──────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ───────────────────────────────────────────────────────────
# Install dependencies in an isolated stage to keep the final image lean.
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system dependencies needed by Python packages
# (pypdf, chromadb, and others may need these at build time)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first — this layer is cached unless requirements change
COPY requirements.txt .

# Install into a prefix directory so we can copy cleanly to the final image
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ───────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN useradd --create-home appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY app/ ./app/

# Create runtime directories that the app writes to
RUN mkdir -p /app/chroma_db \
             /app/data/raw_documents \
             /app/data/processed_documents \
             /app/data/evaluation_dataset \
             /app/evaluation_results \
    && chown -R appuser:appuser /app

USER appuser

# Expose FastAPI default port
EXPOSE 8000

# Health check — Docker will mark the container unhealthy if this fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

# Start the application
# - workers=1 keeps ChromaDB state consistent (single-process)
# - reload is OFF in production; mount your source volume for dev reload
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]