from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from config import get_settings
import os

settings = get_settings()

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(
    settings.database_url,
    connect_args=_connect_args,  # check_same_thread is SQLite-only
)

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
