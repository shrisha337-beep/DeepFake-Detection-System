# ==========================================================================
#  Deepfake Detection System — Hugging Face Spaces Dockerfile
# ==========================================================================
#  Build:  docker build -t deepfake-detection .
#  Run:    docker run -p 7860:7860 deepfake-detection
# ==========================================================================

FROM python:3.10-slim

# ── System dependencies (OpenCV + GLib) ──────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (cached layer) ──────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Application source code ─────────────────────────────────────────────
COPY . .

# ── Expose Gradio default port ──────────────────────────────────────────
EXPOSE 7860

# ── Environment variables ───────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# ── Launch the Gradio frontend (main entry-point for HF Spaces) ────────
CMD ["python", "frontend/app.py"]
