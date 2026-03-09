# ==============================================================================
# SwingTrader — Home Assistant OS Add-on
# Build base is injected by the HA build system via the BUILD_FROM arg.
# See build.yaml for the base image version.
# ==============================================================================
ARG BUILD_FROM
FROM ${BUILD_FROM}

# Install Python 3 and build dependencies
# The hassio base is Alpine Linux; use apk (not apt-get).
RUN apk add --no-cache \
        python3 \
        py3-pip \
        python3-dev \
        gcc \
        musl-dev \
        libffi-dev \
        openssl-dev

# Create an isolated virtual environment to avoid conflicts with system Python
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies first (layer-cached unless requirements change)
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Install s6-overlay service + init scripts
COPY rootfs /

RUN chmod a+x \
        /etc/cont-init.d/swingtrader.sh \
        /etc/services.d/swingtrader/run \
        /etc/services.d/swingtrader/finish

# Expose HTTPS port (also declared in config.yaml)
EXPOSE 8443
