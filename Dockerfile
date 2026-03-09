# ==============================================================================
# SwingTrader — Home Assistant OS Add-on + Standalone Docker
#
# Uses python:3.12-slim directly. ubuntu-base:8.1.1 ships Python 3.8 which
# is incompatible with pandas 2.2+, numpy 2.x, and other modern dependencies.
# haos_entry.py reads /data/options.json (HAOS) or falls through to env vars
# (docker-compose standalone). HA handles container restarts on failure.
# ==============================================================================
FROM python:3.12-slim

# Install system dependencies for the cryptography + other C-extension packages
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

EXPOSE 8443

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,ssl; ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE; urllib.request.urlopen('https://localhost:8443/', context=ctx, timeout=5)" || exit 1

# haos_entry.py reads /data/options.json when running inside HA,
# or transparently falls through to env vars for docker-compose.
CMD ["python", "haos_entry.py"]
