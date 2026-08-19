# Multi-stage build for PatchCore anomaly detection

# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Set PATH and environment
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/root/.cache/torch

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config/ ./config/

# Create directories for data and results
RUN mkdir -p data results/models results/figures results/inference_latest

# Pre-download DINOv2 model to avoid GitHub rate limit during runtime
RUN python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', trust_repo=True)" 2>/dev/null || echo "Model download skipped or cached"

# Default command (no ENTRYPOINT, let CMD handle everything)
CMD ["python", "scripts/train.py", "config/config.yaml"]
