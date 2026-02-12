# ============================================================================
# Multi-stage Dockerfile for ChatBot RAG Project
# ============================================================================

FROM python:3.12-slim as base

# Metadata
LABEL maintainer="ChatBot RAG Project"
LABEL description="RAG ChatBot with Ollama, HuggingFace Embeddings, and ChromaDB"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /app

# ============================================================================
# Stage 1: Dependencies Builder
# ============================================================================
FROM base as builder

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements (usar versión CPU para reducir tamaño)
COPY requirements.cpu.txt ./

# Install Python dependencies with optimizations
RUN pip install --upgrade pip setuptools wheel && \
    # Instalar todas las dependencias excepto torch
    pip install -r requirements.cpu.txt && \
    # Instalar PyTorch CPU desde índice oficial (más ligero)
    pip install torch==2.10.0 --extra-index-url https://download.pytorch.org/whl/cpu && \
    # Limpiar cache y archivos temporales
    pip cache purge && \
    find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type f -name "*.pyc" -delete && \
    find /opt/venv -type f -name "*.pyo" -delete && \
    rm -rf /root/.cache/pip

# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM base as runtime

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create necessary directories with proper permissions
RUN mkdir -p /app/data \
    /app/logs \
    /app/models \
    /app/vectorstore_data \
    /app/documents \
    && chmod -R 755 /app

# Copy application code
COPY --chown=nobody:nogroup . /app/

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Expose port (if needed for API in the future)
EXPOSE 8000

# Default command
CMD ["python", "main.py"]
