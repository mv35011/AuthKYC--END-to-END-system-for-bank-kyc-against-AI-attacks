# ═══════════════════════════════════════════════
# AuthKYC — Production Dockerfile
# Multi-stage: Python 3.11 + OpenCV + PyTorch (CPU)
# ═══════════════════════════════════════════════
FROM python:3.11-slim

WORKDIR /app

# System dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY modules/ modules/
COPY frontend/ frontend/
COPY core_engine.py .
COPY main.py .

# Copy model weights (if present at build time)
COPY best_ftca_pad_model.pth* ./
COPY patent_ftca_v2.pth* ./

# Expose the FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]