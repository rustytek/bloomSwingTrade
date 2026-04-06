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
from auth.utils import hash_password
from auth.router import router as auth_router
from api.stocks import router as stocks_router
from api.screener import router as screener_router
from api.watchlist import router as watchlist_router
from api.portfolio import router as portfolio_router
from api.ai import router as ai_router
from api.charts import router as charts_router
from generate_ssl import generate_ssl_cert
from services.universe import UNIVERSE
from services.market_data import refresh_universe, cleanup_old_entries, invalidate_legacy_cache

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


def init_db():
    """Create all tables and seed the admin user if it doesn't exist."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        if not db.query(User).filter(User.username == settings.admin_user).first():
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SwingTrader starting up…")
    # Purge entries older than 180 days
    db = SessionLocal()
    try:
        cleanup_old_entries(db)
        invalidate_legacy_cache(db)
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
    _scheduler.start()
    logger.info("Scheduler started — daily report will run at 05:30")

    yield

    _scheduler.shutdown(wait=False)
    logger.info("SwingTrader shutting down.")


app = FastAPI(
    title="SwingTrader",
    description="Swing trading screener with AI analysis hooks",
    version="1.5.0",
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
