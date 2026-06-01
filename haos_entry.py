"""
SwingTrader — HAOS add-on entry point.

When run inside a Home Assistant add-on container, HA writes the user's
configuration to /data/options.json before starting the container.  This
script reads that file, sets the appropriate environment variables, then
hands off to main.py exactly as if those variables had been set externally.

When /data/options.json is absent (plain Docker / docker-compose), the
script is a no-op and main.py uses whatever env vars Docker provided.
"""
import json
import os
import runpy

OPTS = "/data/options.json"

if os.path.exists(OPTS):
    with open(OPTS) as f:
        opts = json.load(f)

    # ── Core settings ─────────────────────────────────────────────────────────
    os.environ["SECRET_KEY"]  = str(opts.get("secret_key",  "change-me"))
    os.environ["ADMIN_USER"]  = str(opts.get("admin_user",  "admin"))
    os.environ["ADMIN_PASS"]  = str(opts.get("admin_pass",  "changeme"))
    os.environ["AI_PROVIDER"]  = str(opts.get("ai_provider",  "none"))
    os.environ["AI_API_KEY"]   = str(opts.get("ai_api_key",   ""))
    os.environ["AI_MODEL"]     = str(opts.get("ai_model",     ""))
    os.environ["FRED_API_KEY"] = str(opts.get("fred_api_key", ""))
    os.environ["OLLAMA_URL"]   = str(opts.get("ollama_url",   "http://192.168.10.21:11434"))
    os.environ["OLLAMA_MODEL"] = str(opts.get("ollama_model", "qwen35-moe:latest"))
    os.environ["REPORT_MODEL"] = str(opts.get("report_model", "deepseek-r1:8b"))
    os.environ["LITELLM_URL"]  = str(opts.get("litellm_url",  ""))
    os.environ["LITELLM_API_KEY"] = str(opts.get("litellm_api_key", ""))
    os.environ["HOST"]         = "0.0.0.0"
    os.environ["PORT"]         = "8443"

    # ── Database (HA persistent volume) ───────────────────────────────────────
    os.environ["DATABASE_URL"] = "sqlite:////data/swingtrader.db"

    # ── SSL ───────────────────────────────────────────────────────────────────
    use_ha_ssl = opts.get("use_ha_ssl", False)
    certfile   = opts.get("certfile", "fullchain.pem")
    keyfile    = opts.get("keyfile",  "privkey.pem")

    if use_ha_ssl and os.path.isfile(f"/ssl/{certfile}") and os.path.isfile(f"/ssl/{keyfile}"):
        # Use HA's trusted certificate (Let's Encrypt via Duck DNS / Nginx PM)
        os.environ["SSL_CERT"] = f"/ssl/{certfile}"
        os.environ["SSL_KEY"]  = f"/ssl/{keyfile}"
        print(f"[swingtrader] Using HA SSL certificate: {certfile}")
    else:
        if use_ha_ssl:
            print(f"[swingtrader] HA SSL files not found at /ssl/{certfile} — using self-signed")
        os.makedirs("/data/ssl", mode=0o700, exist_ok=True)
        os.environ["SSL_CERT"] = "/data/ssl/cert.pem"
        os.environ["SSL_KEY"]  = "/data/ssl/key.pem"

# ── Run main.py ───────────────────────────────────────────────────────────────
# runpy executes main.py with __name__ == "__main__" so the uvicorn block fires.
# All env vars are already set, so pydantic-settings picks them up on first use.
os.chdir(os.path.dirname(os.path.abspath(__file__)))
runpy.run_path("main.py", run_name="__main__")
