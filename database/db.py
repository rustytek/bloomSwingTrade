from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from config import get_settings
import os

settings = get_settings()

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# How long a connection waits for another process's write lock before raising
# "database is locked". The web app and up to a few job-worker PROCESSES
# (services/job_worker.py) share this one SQLite file; the 20-year backfill
# writes multi-MB rows, and SQLite's 5 s default was not enough — on
# 2026-09-26 a daily-report worker could not even record its own failure
# while the backfill was writing, and its job stayed "running".
SQLITE_BUSY_TIMEOUT_S = 30


def configure_sqlite(dbapi_conn, _record=None) -> None:
    """Per-connection SQLite settings.

    WAL journal mode lets readers proceed while a writer is active (the
    default rollback journal blocks readers during every commit), so the web
    app's reads never wait on a worker's writes and only writer-vs-writer
    waits remain, bounded by busy_timeout. journal_mode=WAL is persistent in
    the database file; setting it on every connect is a cheap no-op after the
    first. synchronous=NORMAL is the recommended pairing with WAL (durable
    across app crashes; a power cut can lose only the last commits).
    """
    cur = dbapi_conn.cursor()
    try:
        cur.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_S * 1000}")
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
    finally:
        cur.close()


_is_sqlite = settings.database_url.startswith("sqlite")
_connect_args = {"check_same_thread": False, "timeout": SQLITE_BUSY_TIMEOUT_S} if _is_sqlite else {}
engine = create_engine(
    settings.database_url,
    connect_args=_connect_args,  # check_same_thread / timeout are SQLite-only
)
if _is_sqlite and ":memory:" not in settings.database_url:
    event.listen(engine, "connect", configure_sqlite)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency: yields a DB session and ensures it closes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
