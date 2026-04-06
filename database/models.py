from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text,
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

    watchlist = relationship("WatchlistItem", back_populates="user", cascade="all, delete-orphan")
    portfolio = relationship("PortfolioPosition", back_populates="user", cascade="all, delete-orphan")


class WatchlistItem(Base):
    __tablename__ = "watchlist_items"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    ticker = Column(String(16), nullable=False)
    added_at = Column(DateTime, default=utcnow)
    notes = Column(Text, nullable=True)

    user = relationship("User", back_populates="watchlist")

    __table_args__ = (UniqueConstraint("user_id", "ticker", name="uq_watchlist_user_ticker"),)


class PortfolioPosition(Base):
    __tablename__ = "portfolio_positions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    ticker = Column(String(16), nullable=False)
    shares = Column(Float, nullable=False)
    avg_cost = Column(Float, nullable=False)
    added_at = Column(DateTime, default=utcnow)
    notes = Column(Text, nullable=True)

    user = relationship("User", back_populates="portfolio")

    __table_args__ = (UniqueConstraint("user_id", "ticker", name="uq_portfolio_user_ticker"),)


class StockCache(Base):
    __tablename__ = "stock_cache"

    ticker = Column(String(16), primary_key=True)
    quote_json = Column(Text, nullable=True)
    history_json = Column(Text, nullable=True)
    cached_at = Column(DateTime, default=utcnow)


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
