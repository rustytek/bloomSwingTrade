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
from api.weekly_plan import router as weekly_plan_router
from api.scorecard import router as scorecard_router
from api.jobs import router as jobs_router
from api.broker import router as broker_router
from api.history import router as history_router
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
    invalidate_legacy_schema_cache,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()
_scheduler = AsyncIOScheduler()


async def _scheduled_report_job():
    """Generate the daily report for every user at 05:30, each using their own
    LiteLLM key (falls back to the global key when a user has none)."""
    # Each report is built in the out-of-process job worker (see
    # report_service.run_daily_report_job) — running it on this event loop is
    # what got the add-on SIGKILLed by the watchdog at 05:31 on 2026-09-24.
    # Users are done one at a time so four workers never compete at once.
    from services.report_service import run_daily_report_job
    db = SessionLocal()
    try:
        users = db.query(User).all()
        if not users:
            logger.warning("Scheduler: no users found — skipping report")
            return
        logger.info("Scheduler: generating daily reports for %s users", len(users))
        for user in users:
            try:
                await run_daily_report_job(db, user.id, triggered_by="schedule")
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
    orphan_background_jobs()
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


def orphan_background_jobs():
    """Fail any job left mid-flight by a restart.

    Workers are child processes of this container, so a container restart kills
    every one of them. Anything still marked queued/running at startup is by
    definition dead — waiting for the 12-minute staleness reaper to notice would
    leave the edge-matrix build blocked (`enqueue` joins an "active" job rather
    than starting a second) for that whole window after every restart.

    This is the exact loop that produced the 2026-09-23 outage: the watchdog
    restarted the add-on mid-build, and the stuck row then swallowed the retry.
    """
    from database.models import BackgroundJob
    from datetime import datetime, timezone

    db = SessionLocal()
    try:
        stuck = db.query(BackgroundJob).filter(
            BackgroundJob.status.in_(("queued", "running"))).all()
        if not stuck:
            return
        for job in stuck:
            job.status = "failed"
            job.finished_at = datetime.now(timezone.utc)
            job.error = ("Interrupted by an add-on restart — the worker process did not "
                         "survive it. Nothing was corrupted; start the build again.")
        db.commit()
        logger.info("Reset %d background job(s) orphaned by a restart", len(stuck))
    except Exception:  # noqa: BLE001 — never block startup over bookkeeping
        logger.exception("Could not reset orphaned background jobs")
        db.rollback()
    finally:
        db.close()


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
            # Chosen open-R ceiling. Deliberately NO DEFAULT: NULL means "not
            # chosen", which is what makes the derived-vs-chosen distinction in
            # `budget_basis` honest.
            "max_open_r": "FLOAT",
            "use_evidence_regimes": "BOOLEAN DEFAULT 0",
        })
        _ensure_columns(conn, "portfolio_positions", {
            "stop_loss": "FLOAT",
            # Stop at entry — the R-multiple denominator. Stays NULL on legacy
            # rows; close_position() falls back to stop_loss when it is NULL.
            "initial_stop": "FLOAT",
            "target": "FLOAT",
            "entry_date": "DATE",
            "strategy": "VARCHAR(32)",
            # Trade-plan intent, previously buried in `notes`. See
            # services/trade_plan.py::parse_plan_notes and the backfill below.
            "planned_entry": "FLOAT",
            "planned_entry_high": "FLOAT",
            "thesis": "TEXT",
            "invalidation": "TEXT",
            "time_stop_days": "INTEGER",
        })
        _ensure_columns(conn, "closed_trades", {
            "initial_stop": "FLOAT",
            "planned_entry": "FLOAT",
            "planned_entry_high": "FLOAT",
            "thesis": "TEXT",
            "invalidation": "TEXT",
            "time_stop_days": "INTEGER",
        })
        _ensure_columns(conn, "report_cache", {
            "model": "VARCHAR(128)",
        })
        _ensure_columns(conn, "history_archive", {
            "source": "VARCHAR(16)",
            "requested_start": "VARCHAR(10)",
            "issues": "TEXT",
            "checked_at": "DATETIME",
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

        # `max_open_r` is deliberately NOT repaired to a value here: NULL is a
        # meaningful state ("not chosen — use the derived budget").

        backfill_plan_fields_from_notes(conn)


# Tables that carry the trade-plan intent columns, and can therefore be
# back-filled from their legacy prefixed `notes` text.
_PLAN_BACKFILL_TABLES = ("portfolio_positions", "closed_trades")
_PLAN_BACKFILL_COLUMNS = ("thesis", "invalidation", "time_stop_days", "planned_entry",
                          "planned_entry_high")


def backfill_plan_fields_from_notes(conn) -> dict[str, int]:
    """One-time, idempotent migration of plan intent out of `notes`.

    Before PortfolioPosition/ClosedTrade had real columns, the Plan-a-Trade
    modal packed the thesis, the invalidation condition, the time stop and the
    planned entry into `notes` as prefixed lines. This lifts those values into
    the new columns.

    Rules:
      * Only fills a column that is currently NULL — a value written by the API
        always wins over one re-parsed from prose.
      * `notes` is LEFT COMPLETELY INTACT. It is the only free-form record of a
        trade's reasoning and destroying it to "clean up" would be unrecoverable.
      * Therefore idempotent: a second run finds the columns already populated
        and moves nothing.

    Returns {table: rows_updated} and logs what it moved.
    """
    from services.trade_plan import parse_plan_notes

    moved: dict[str, int] = {}
    for table in _PLAN_BACKFILL_TABLES:
        try:
            present = {
                row[1]
                for row in conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()
            }
        except Exception:  # noqa: BLE001 — table absent on a brand-new DB
            continue
        cols = [c for c in _PLAN_BACKFILL_COLUMNS if c in present]
        if not cols or "notes" not in present:
            continue

        select = ", ".join(["id", "notes", *cols])
        null_filter = " OR ".join(f"{c} IS NULL" for c in cols)
        rows = conn.exec_driver_sql(
            f"SELECT {select} FROM {table} "
            f"WHERE notes IS NOT NULL AND notes != '' AND ({null_filter})"
        ).fetchall()

        updated = 0
        for row in rows:
            current = dict(zip(cols, row[2:]))
            parsed = parse_plan_notes(row[1])
            assigns = {
                c: parsed.get(c)
                for c in cols
                if current.get(c) is None and parsed.get(c) is not None
            }
            if not assigns:
                continue
            set_sql = ", ".join(f"{c} = ?" for c in assigns)
            conn.exec_driver_sql(
                f"UPDATE {table} SET {set_sql} WHERE id = ?",
                (*assigns.values(), row[0]),
            )
            updated += 1
            logger.info(
                "Plan backfill: %s id=%s <- %s",
                table, row[0], ", ".join(sorted(assigns)),
            )
        moved[table] = updated
        if updated:
            logger.info("Plan backfill: moved plan fields for %d %s row(s)", updated, table)
    return moved


async def _warm_chart_cache():
    """Populate the in-memory chart cache so /charts is never cold-fetched
    inside a user's request. Every failure is swallowed and logged: a warm-up
    is an optimisation, and must never stop the app from starting."""
    import asyncio
    try:
        from services import chart_service as cs
        # Sequential, not gathered: the point is to fill the cache without
        # competing with the universe refresh for yfinance bandwidth.
        for name, fn in (("VIX", cs.get_vix_data),
                         ("sectors", cs.get_sector_data),
                         ("ETF groups", cs.get_etf_group_data)):
            try:
                await fn()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Chart cache warm-up for %s failed: %s", name, exc)
            await asyncio.sleep(0)
        logger.info("Chart cache warm-up complete")
    except Exception:  # noqa: BLE001
        logger.exception("Chart cache warm-up could not run")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SwingTrader starting up…")
    # Surface a raw provider model name in AI_MODEL/REPORT_MODEL. These must be
    # LiteLLM tier aliases; a raw name pins the app to one model and fails as an
    # opaque proxy error at request time rather than here, where it is obvious.
    try:
        from services.ai_service import check_model_aliases
        for problem in check_model_aliases(settings):
            logger.warning("CONFIG: %s", problem)
    except Exception:  # noqa: BLE001 — a config lint must never block startup
        logger.exception("Could not check model alias configuration")
    # Purge entries older than 180 days
    db = SessionLocal()
    try:
        cleanup_old_entries(db)
        invalidate_legacy_cache(db)
        invalidate_short_history_cache(db)
        invalidate_missing_swing_score_cache(db)
        # Back-date any StockCache row whose quote predates the current quote
        # schema. get_quote already refuses to SERVE a stale-schema row, but
        # refresh_universe picks stale tickers by timestamp alone — without this
        # a pre-v2 row would only be rebuilt on an on-demand hit.
        invalidate_legacy_schema_cache(db)
    finally:
        db.close()
    # Kick off background universe data refresh (non-blocking)
    import asyncio
    asyncio.create_task(refresh_universe(UNIVERSE, db_factory=SessionLocal))
    logger.info(f"Background universe refresh started for {len(UNIVERSE)} tickers")

    # Warm the chart cache in the background.
    #
    # services/chart_service.py caches in MEMORY ONLY, so every container
    # restart leaves it empty and the next visitor to /charts pays for the
    # whole cold fetch (VIX + 11 sector ETFs + ETF groups + macro) inside their
    # own request. Through a Cloudflare tunnel that request can exceed the
    # proxy's patience and return 502 — which is exactly what the user saw on
    # 2026-09-23, when a watchdog restart wiped this cache mid-incident.
    # Warming it here moves that cost off the first request.
    asyncio.create_task(_warm_chart_cache())

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
    version="1.23.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS — explicit origin when PUBLIC_URL is set (credentialed); wildcard without credentials otherwise ("*" + credentials is rejected by browsers).
# The browser's Origin header never has a trailing slash, so strip one from PUBLIC_URL or it never matches.
_cors_origin = get_settings().public_url.rstrip("/")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_cors_origin] if _cors_origin else ["*"],
    allow_credentials=bool(_cors_origin),
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
app.include_router(weekly_plan_router)
app.include_router(scorecard_router)
app.include_router(jobs_router)
app.include_router(broker_router)
app.include_router(history_router)


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


@app.get("/scorecard")
async def scorecard_page():
    return FileResponse(os.path.join(STATIC_DIR, "scorecard.html"))


@app.get("/plan")
async def serve_plan():
    return FileResponse(os.path.join(STATIC_DIR, "plan.html"))


@app.get("/today")
async def today_page():
    return FileResponse(os.path.join(STATIC_DIR, "today.html"))


@app.get("/admin")
async def admin_page():
    return FileResponse(os.path.join(STATIC_DIR, "admin.html"))


@app.get("/trade")
async def trade_page():
    return FileResponse(os.path.join(STATIC_DIR, "trade.html"))


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
