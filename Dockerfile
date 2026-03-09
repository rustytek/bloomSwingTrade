# ==============================================================================
# SwingTrader — Home Assistant OS Add-on
# Build base is injected by the HA build system via the BUILD_FROM arg.
# See build.yaml for the base image version.
# ==============================================================================
ARG BUILD_FROM
FROM ${BUILD_FROM}

# Install Python 3 and build dependencies
# The hassio base is Ubuntu; python3 / python3-venv are in the default repos.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        gcc \
        libffi-dev \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

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
