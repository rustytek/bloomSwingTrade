"""
SwingTrader — FastAPI entry point.

Startup sequence:
  1. Generate SSL cert if missing
  2. Initialize / migrate database (create tables, seed admin user)
  3. Mount API routers
  4. Serve static frontend for all non-API routes
  5. Start uvicorn with HTTPS
"""
import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from config import get_settings
from database.db import engine
from database.models import Base, User
from database.db import SessionLocal
from auth.utils import hash_password, verify_password
from auth.router import router as auth_router
from api.stocks import router as stocks_router
from api.screener import router as screener_router
from api.watchlist import router as watchlist_router
from api.portfolio import router as portfolio_router
from api.ai import router as ai_router
from api.charts import router as charts_router
from api.backtest import router as backtest_router
from api.settings import router as settings_router
from api.journal import router as journal_router
from api.today import router as today_router
from generate_ssl import generate_ssl_cert
from services.universe import UNIVERSE
from services.market_data import (
    refresh_universe,
    refresh_universe_once,
    is_market_open,
    cleanup_old_entries,
    invalidate_legacy_cache,
    invalidate_short_history_cache,
    invalidate_missing_swing_score_cache,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()
_scheduler = AsyncIOScheduler()


async def _scheduled_report_job():
    """Generate the daily report for every user at 05:30, each using their own
    LiteLLM key (falls back to the global key when a user has none)."""
    from services.report_service import generate_daily_report
    db = SessionLocal()
    try:
        users = db.query(User).all()
        if not users:
            logger.warning("Scheduler: no users found — skipping report")
            return
        logger.info("Scheduler: generating daily reports for %s users", len(users))
        for user in users:
            try:
                await generate_daily_report(
                    db, user.id, triggered_by="schedule", api_key=user.litellm_api_key,
                    system_prompt=user.report_system_prompt,
                )
            except Exception as e:
                logger.error("Scheduler: report failed for user %s: %s", user.username, e)
        logger.info("Scheduler: daily reports complete")
    except Exception as e:
        logger.error("Scheduler: report generation failed: %s", e)
    finally:
        db.close()


async def _scheduled_market_refresh_job():
    """Refresh market data every 15 minutes during regular trading hours."""
    if not is_market_open():
        logger.debug("Scheduler: skipped market refresh because market is closed")
        return
    logger.info("Scheduler: refreshing market data for %s tickers", len(UNIVERSE))
    result = await refresh_universe_once(UNIVERSE, db_factory=SessionLocal, force=True)
    logger.info(
        "Scheduler: market refresh complete — refreshed=%s skipped=%s hit_limit=%s",
        result.get("refreshed"),
        result.get("skipped"),
        result.get("hit_limit"),
    )
    from services.today import invalidate_cache
    invalidate_cache()


def init_db():
    """Create all tables and sync the configured admin account."""
    Base.metadata.create_all(bind=engine)
    ensure_schema_migrations()
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.username == settings.admin_user).first()
        if admin:
            changed = False
            if not verify_password(settings.admin_pass, admin.password_hash):
                admin.password_hash = hash_password(settings.admin_pass)
                changed = True
            if not admin.is_admin:
                admin.is_admin = True
                changed = True
            if changed:
                db.commit()
                logger.info("Synced configured admin user: %s", settings.admin_user)
        else:
            admin = User(
                username=settings.admin_user,
                password_hash=hash_password(settings.admin_pass),
                is_admin=True,
            )
            db.add(admin)
            db.commit()
            logger.info(f"Created admin user: {settings.admin_user}")
    finally:
        db.close()


async def _scheduled_watchlist_snapshot_job():
    """Capture each user's watchlist at the start of the trading week."""
    from services.backtest import create_watchlist_snapshot
    db = SessionLocal()
    try:
        users = db.query(User).all()
        for user in users:
            create_watchlist_snapshot(db, user.id, notes="weekly-schedule")
        logger.info("Scheduler: captured weekly watchlist snapshots for %s users", len(users))
    except Exception as e:
        logger.error("Scheduler: watchlist snapshot failed: %s", e)
    finally:
        db.close()


def _ensure_columns(conn, table: str, columns: dict[str, str]):
    """ALTER TABLE ADD COLUMN for any column in `columns` (name -> DDL) not yet present."""
    existing = {
        row[1]
        for row in conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()
    }
    for name, ddl in columns.items():
        if name not in existing:
            conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


def ensure_schema_migrations():
    """Add lightweight SQLite columns needed by older installs."""
    if engine.dialect.name != "sqlite":
        return

    with engine.begin() as conn:
        _ensure_columns(conn, "stock_cache", {
            "quote_cached_at": "DATETIME",
            "history_cached_at": "DATETIME",
        })
        _ensure_columns(conn, "users", {
            "account_size": "FLOAT DEFAULT 10000",
            "risk_pct": "FLOAT DEFAULT 1.0",
            "max_positions": "INTEGER DEFAULT 8",
            "atr_stop_mult": "FLOAT DEFAULT 2.5",
            "r_multiple": "FLOAT DEFAULT 2.0",
            "litellm_api_key": "VARCHAR(256)",
            "report_system_prompt": "TEXT",
            "chat_system_prompt": "TEXT",
        })
        _ensure_columns(conn, "portfolio_positions", {
            "stop_loss": "FLOAT",
            "target": "FLOAT",
            "entry_date": "DATE",
            "strategy": "VARCHAR(32)",
        })
        _ensure_columns(conn, "report_cache", {
            "model": "VARCHAR(128)",
        })

        # Repair any NULL trading-settings left behind by older migrations that
        # added these columns without a DEFAULT (those rows 500 the Today page
        # because build_trade_plan / position_flags can't compare None to a number).
        for col, default in (
            ("account_size", 10000),
            ("risk_pct", 1.0),
            ("max_positions", 8),
            ("atr_stop_mult", 2.5),
            ("r_multiple", 2.0),
        ):
            conn.exec_driver_sql(
                f"UPDATE users SET {col} = {default} WHERE {col} IS NULL"
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SwingTrader starting up…")
    # Purge entries older than 180 days
    db = SessionLocal()
    try:
        cleanup_old_entries(db)
        invalidate_legacy_cache(db)
        invalidate_short_history_cache(db)
        invalidate_missing_swing_score_cache(db)
    finally:
        db.close()
    # Kick off background universe data refresh (non-blocking)
    import asyncio
    asyncio.create_task(refresh_universe(UNIVERSE, db_factory=SessionLocal))
    logger.info(f"Background universe refresh started for {len(UNIVERSE)} tickers")

    # Schedule daily report at 05:30 local server time
    _scheduler.add_job(
        _scheduled_report_job,
        CronTrigger(hour=5, minute=30),
        id="daily_report",
        replace_existing=True,
    )
    _scheduler.add_job(
        _scheduled_watchlist_snapshot_job,
        CronTrigger(day_of_week="mon", hour=6, minute=0),
        id="weekly_watchlist_snapshot",
        replace_existing=True,
    )
    _scheduler.add_job(
        _scheduled_market_refresh_job,
        CronTrigger(day_of_week="mon-fri", hour="9-15", minute="*/15", timezone="America/New_York"),
        id="market_data_15m_refresh",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    _scheduler.start()
    logger.info("Scheduler started — daily report 05:30, market data every 15 minutes during trading hours")

    yield

    _scheduler.shutdown(wait=False)
    logger.info("SwingTrader shutting down.")


app = FastAPI(
    title="SwingTrader",
    description="Swing trading screener with AI analysis hooks",
    version="1.12.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS — allow the frontend origin (same host, different port only in dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API routers ──────────────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(stocks_router)
app.include_router(screener_router)
app.include_router(watchlist_router)
app.include_router(portfolio_router)
app.include_router(ai_router)
app.include_router(charts_router)
app.include_router(backtest_router)
app.include_router(settings_router)
app.include_router(journal_router)
app.include_router(today_router)


# ── Static files (React SPA) ─────────────────────────────────────────────────

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Serve shared static assets (e.g. /static/js/common.js)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/login")
async def login_page():
    return FileResponse(os.path.join(STATIC_DIR, "login.html"))


@app.get("/report")
async def report_page():
    return FileResponse(os.path.join(STATIC_DIR, "report.html"))


@app.get("/charts")
async def charts_page():
    return FileResponse(os.path.join(STATIC_DIR, "charts.html"))


@app.get("/backtest")
async def backtest_page():
    return FileResponse(os.path.join(STATIC_DIR, "backtest.html"))


@app.get("/screener")
async def screener_page():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/journal")
async def journal_page():
    return FileResponse(os.path.join(STATIC_DIR, "journal.html"))


@app.get("/today")
async def today_page():
    return FileResponse(os.path.join(STATIC_DIR, "today.html"))


@app.get("/admin")
async def admin_page():
    return FileResponse(os.path.join(STATIC_DIR, "admin.html"))


@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "today.html"))


# Catch-all: serve the Today dashboard for any unknown routes
@app.exception_handler(404)
async def spa_handler(request: Request, exc):
    # API routes should return proper 404 JSON
    if request.url.path.startswith(("/api/", "/auth/", "/static/")):
        return JSONResponse({"detail": "Not found"}, status_code=404)
    return FileResponse(os.path.join(STATIC_DIR, "today.html"))


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Generate SSL cert BEFORE uvicorn tries to load it
    generate_ssl_cert(settings.ssl_cert, settings.ssl_key)
    init_db()

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        ssl_keyfile=settings.ssl_key,
        ssl_certfile=settings.ssl_cert,
        reload=False,
        log_level="info",
    )
