# ==============================================================================
# SwingTrader — Home Assistant OS Add-on
# Follows the same pattern as marciogranzotto/addon-nightscout.
#
# BUILD_FROM defaults to the ubuntu-base so the image can also be built
# locally with plain "docker build ." without HA injecting the ARG.
# ==============================================================================
ARG BUILD_FROM=ghcr.io/hassio-addons/ubuntu-base:8.1.1
# hadolint ignore=DL3006
FROM ${BUILD_FROM}

# Set shell (ubuntu-base ships bash)
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install Python 3 and build dependencies via apt-get (ubuntu-base = Ubuntu 22.04)
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        gcc \
        libffi-dev \
        libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies (cached layer)
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Install s6-overlay service + init scripts
COPY rootfs /
RUN chmod a+x \
        /etc/cont-init.d/swingtrader.sh \
        /etc/services.d/swingtrader/run \
        /etc/services.d/swingtrader/finish

EXPOSE 8443

# Build arguments (injected by HA build system)
ARG BUILD_ARCH=amd64
ARG BUILD_DATE
ARG BUILD_REF
ARG BUILD_VERSION

LABEL \
    io.hass.name="SwingTrader" \
    io.hass.description="Self-hosted swing trading screener with live market data, technical indicators, watchlist and portfolio tracking." \
    io.hass.arch="${BUILD_ARCH}" \
    io.hass.type="addon" \
    io.hass.version="${BUILD_VERSION}" \
    maintainer="rustytek" \
    org.opencontainers.image.title="SwingTrader" \
    org.opencontainers.image.source="https://github.com/rustytek/bloomSwingTrade" \
    org.opencontainers.image.created="${BUILD_DATE}" \
    org.opencontainers.image.revision="${BUILD_REF}" \
    org.opencontainers.image.version="${BUILD_VERSION}"
