from sqlalchemy import (
    Column, Integer, String, Float, Boolean, Date, DateTime, Text,
    ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from database.db import Base


def utcnow():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    password_hash = Column(String(256), nullable=False)
    email = Column(String(128), nullable=True)
    is_admin = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=utcnow)
    last_login = Column(DateTime, nullable=True)

    # Per-user LiteLLM virtual key — keeps each user's AI token usage separate.
    # When null, the AI path falls back to the global key from config.
    litellm_api_key = Column(String(256), nullable=True)

    # Per-user AI system-prompt overrides. When null, the built-in default
    # (services/report_service._SYSTEM_PROMPT / api.ai.DEFAULT_CHAT_SYSTEM_PROMPT)
    # is used. The fixed report _TEMPLATE is never user-editable.
    report_system_prompt = Column(Text, nullable=True)
    chat_system_prompt = Column(Text, nullable=True)

    # Trading settings (fixed-fractional risk model)
    account_size = Column(Float, default=10000, nullable=False)
    risk_pct = Column(Float, default=1.0, nullable=False)
    max_positions = Column(Integer, default=8, nullable=False)
    atr_stop_mult = Column(Float, default=2.5, nullable=False)
    r_multiple = Column(Float, default=2.0, nullable=False)

    watchlist = relationship("WatchlistItem", back_populates="user", cascade="all, delete-orphan")
    portfolio = relationship("PortfolioPosition", back_populates="user", cascade="all, delete-orphan")

    @property
    def has_litellm_key(self) -> bool:
        return bool(self.litellm_api_key)


class WatchlistItem(Base):
    __tablename__ = "watchlist_items"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    ticker = Column(String(16), nullable=False)
    added_at = Column(DateTime, default=utcnow)
    notes = Column(Text, nullable=True)

    user = relationship("User", back_populates="watchlist")

    __table_args__ = (UniqueConstraint("user_id", "ticker", name="uq_watchlist_user_ticker"),)


class WatchlistSnapshot(Base):
    __tablename__ = "watchlist_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    week_start = Column(Date, nullable=False, index=True)
    tickers_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=utcnow)
    notes = Column(Text, nullable=True)

    user = relationship("User")

    __table_args__ = (UniqueConstraint("user_id", "week_start", name="uq_watchlist_snapshot_user_week"),)


class PortfolioPosition(Base):
    __tablename__ = "portfolio_positions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    ticker = Column(String(16), nullable=False)
    shares = Column(Float, nullable=False)
    avg_cost = Column(Float, nullable=False)
    added_at = Column(DateTime, default=utcnow)
    notes = Column(Text, nullable=True)
    stop_loss = Column(Float, nullable=True)
    target = Column(Float, nullable=True)
    entry_date = Column(Date, nullable=True)
    strategy = Column(String(32), nullable=True)

    user = relationship("User", back_populates="portfolio")

    __table_args__ = (UniqueConstraint("user_id", "ticker", name="uq_portfolio_user_ticker"),)


class ClosedTrade(Base):
    """Journal entry created when a portfolio position is closed (sold)."""
    __tablename__ = "closed_trades"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    ticker = Column(String(16), nullable=False)
    shares = Column(Float, nullable=False)
    avg_cost = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=False)
    entry_date = Column(Date, nullable=True)
    exit_date = Column(Date, nullable=True)
    stop_loss = Column(Float, nullable=True)
    target = Column(Float, nullable=True)
    strategy = Column(String(32), nullable=True)
    pnl = Column(Float, nullable=False)
    pnl_pct = Column(Float, nullable=False)
    r_multiple = Column(Float, nullable=True)   # null when no valid stop was set
    notes = Column(Text, nullable=True)
    opened_at = Column(DateTime, nullable=True)
    closed_at = Column(DateTime, default=utcnow)

    user = relationship("User")


class StockCache(Base):
    __tablename__ = "stock_cache"

    ticker = Column(String(16), primary_key=True)
    quote_json = Column(Text, nullable=True)
    history_json = Column(Text, nullable=True)
    cached_at = Column(DateTime, default=utcnow)
    quote_cached_at = Column(DateTime, nullable=True)
    history_cached_at = Column(DateTime, nullable=True)


class HistoryArchive(Base):
    """On-demand long-history store for arbitrary-era backtests.

    Separate from the rolling StockCache (which stays current for the live
    screener). Each row holds the widest [start_date, end_date] daily range
    fetched for a ticker; backtests over old eras fetch the missing span once
    here and reuse it thereafter.
    """
    __tablename__ = "history_archive"

    ticker = Column(String(16), primary_key=True)
    bars_json = Column(Text, nullable=False)
    start_date = Column(String(10), nullable=False)   # earliest bar date covered (YYYY-MM-DD)
    end_date = Column(String(10), nullable=False)     # latest bar date covered
    fetched_at = Column(DateTime, default=utcnow)


class AICache(Base):
    __tablename__ = "ai_cache"

    ticker = Column(String(16), primary_key=True)
    analysis_json = Column(Text, nullable=False)
    analyzed_at = Column(DateTime, default=utcnow)


class ReportCache(Base):
    __tablename__ = "report_cache"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    report_markdown = Column(Text, nullable=False)
    generated_at = Column(DateTime, default=utcnow)
    triggered_by = Column(String(16), default="user")   # "user" | "schedule"
    model = Column(String(128), nullable=True)          # resolved model that generated it
