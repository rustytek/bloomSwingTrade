# ==============================================================================
# SwingTrader — Home Assistant OS Add-on + Standalone Docker
#
# Uses python:3.12-slim directly (no hassio-addons base required).
# Config is read from /data/options.json (HAOS) or env vars (docker-compose).
# ==============================================================================
FROM python:3.12-slim

# Install system dependencies for the cryptography package
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        libffi-dev \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (cached layer — only rebuilds if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Expose HTTPS port (mirrors config.yaml ports declaration)
EXPOSE 8443

# Health check — works for both HAOS and standalone deployments
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,ssl; ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE; urllib.request.urlopen('https://localhost:8443/', context=ctx, timeout=5)" || exit 1

# haos_entry.py reads /data/options.json (HAOS) or falls through to env vars
CMD ["python", "haos_entry.py"]
