from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    # Security
    secret_key: str = "change-this-to-a-random-secret-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440  # 24 hours

    # Database
    database_url: str = "sqlite:///./data/swingtrader.db"

    # Initial admin credentials (used only on first run to seed DB)
    admin_user: str = "admin"
    admin_pass: str = "changeme"

    # Server
    host: str = "0.0.0.0"
    port: int = 8443
    ssl_cert: str = "ssl/cert.pem"
    ssl_key: str = "ssl/key.pem"

    # Cache TTLs (seconds)
    quote_cache_ttl: int = 86400     # 24 hours — refresh once per day
    history_cache_ttl: int = 86400   # 24 hours
    ai_cache_ttl: int = 3600         # 1 hour
    cache_max_age_days: int = 180    # purge entries older than 180 days

    # AI Provider ("none", "anthropic", "openai", "litellm")
    ai_provider: str = "litellm"
    ai_api_key: str = ""
    ai_model: str = "tooling_high"                 # LiteLLM tier alias, not a raw model name

    # FRED API (Federal Reserve Economic Data)
    fred_api_key: str = ""

    report_model: str = "tooling_high"     # report generation — LiteLLM tier alias

    # LiteLLM (OpenAI-compatible proxy — the only AI backend this app talks to)
    litellm_url: str = "http://192.168.0.21:4000"  # e.g. http://192.168.0.21:4000
    litellm_api_key: str = ""                      # LiteLLM virtual key for this user/app

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
