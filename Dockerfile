FROM python:3.10-slim
WORKDIR /app
# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl && \
    rm -rf /var/lib/apt/lists/*
# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copy application
COPY . .
# Create data directory
RUN mkdir -p /app/data /app/models /app/logs
# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser
# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v3/health || exit 1
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "api_server.enterprise_main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
