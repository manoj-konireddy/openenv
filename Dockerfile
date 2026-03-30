# ============================================================
# Customer Support Routing — OpenEnv
# Hugging Face Spaces compatible Dockerfile
# ============================================================

FROM python:3.11-slim

# HF Spaces runs as non-root user 1000
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Fix permissions for HF Spaces
RUN chown -R appuser:appuser /app

USER appuser

# Expose port expected by HF Spaces
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Launch FastAPI server
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
