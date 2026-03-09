#!/usr/bin/with-contenv bashio
# ==============================================================================
# SwingTrader — Container initialisation
# Reads add-on options from HA's /data/options.json via bashio and writes a
# chmod-600 env file to /data/ (HA's persistent volume) for the service runner.
# ==============================================================================
set -e

bashio::log.info "Configuring SwingTrader..."

# ── Persistent directories ────────────────────────────────────────────────────
mkdir -p /data/ssl
chmod 700 /data/ssl

# ── Read options from HA supervisor ──────────────────────────────────────────
SECRET_KEY="$(bashio::config 'secret_key')"
ADMIN_USER="$(bashio::config 'admin_user')"
ADMIN_PASS="$(bashio::config 'admin_pass')"
AI_PROVIDER="$(bashio::config 'ai_provider')"
AI_API_KEY="$(bashio::config 'ai_api_key')"
AI_MODEL="$(bashio::config 'ai_model')"

# ── SSL setup ─────────────────────────────────────────────────────────────────
if bashio::config.true 'use_ha_ssl'; then
    CERTFILE="$(bashio::config 'certfile')"
    KEYFILE="$(bashio::config 'keyfile')"
    if [[ -f "/ssl/${CERTFILE}" && -f "/ssl/${KEYFILE}" ]]; then
        bashio::log.info "Using Home Assistant SSL certificate: ${CERTFILE}"
        SSL_CERT="/ssl/${CERTFILE}"
        SSL_KEY="/ssl/${KEYFILE}"
    else
        bashio::log.warning "HA SSL files not found at /ssl/${CERTFILE} — falling back to self-signed"
        SSL_CERT="/data/ssl/cert.pem"
        SSL_KEY="/data/ssl/key.pem"
    fi
else
    bashio::log.info "Using self-signed SSL certificate (auto-generated on first run)"
    SSL_CERT="/data/ssl/cert.pem"
    SSL_KEY="/data/ssl/key.pem"
fi

# ── Write env file (chmod 600 — root-readable only inside container) ──────────
cat > /data/swingtrader.env << EOF
SECRET_KEY=${SECRET_KEY}
ADMIN_USER=${ADMIN_USER}
ADMIN_PASS=${ADMIN_PASS}
AI_PROVIDER=${AI_PROVIDER}
AI_API_KEY=${AI_API_KEY}
AI_MODEL=${AI_MODEL}
SSL_CERT=${SSL_CERT}
SSL_KEY=${SSL_KEY}
DATABASE_URL=sqlite:////data/swingtrader.db
HOST=0.0.0.0
PORT=8443
EOF

chmod 600 /data/swingtrader.env

bashio::log.info "SwingTrader configuration complete."
