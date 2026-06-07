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
    """Generate the daily report for the admin user at 05:30."""
    from services.report_service import generate_daily_report
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.is_admin.is_(True)).first()
        if admin:
            logger.info("Scheduler: generating daily report for admin user %s", admin.username)
            await generate_daily_report(db, admin.id, triggered_by="schedule")
            logger.info("Scheduler: daily report complete")
        else:
            logger.warning("Scheduler: no admin user found — skipping report")
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


def init_db():
    """Create all tables and sync the configured admin account."""
    Base.metadata.create_all(bind=engine)
    ensure_cache_columns()
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


def ensure_cache_columns():
    """Add lightweight SQLite columns needed by older installs."""
    if engine.dialect.name != "sqlite":
        return

    with engine.begin() as conn:
        existing = {
            row[1]
            for row in conn.exec_driver_sql("PRAGMA table_info(stock_cache)").fetchall()
        }
        for column in ("quote_cached_at", "history_cached_at"):
            if column not in existing:
                conn.exec_driver_sql(f"ALTER TABLE stock_cache ADD COLUMN {column} DATETIME")


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
    version="1.6.7",
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


# ── Static files (React SPA) ─────────────────────────────────────────────────

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


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


@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# Catch-all: serve index.html for any unknown routes (SPA routing)
@app.exception_handler(404)
async def spa_handler(request: Request, exc):
    # API routes should return proper 404 JSON
    if request.url.path.startswith(("/api/", "/auth/")):
        return JSONResponse({"detail": "Not found"}, status_code=404)
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


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
