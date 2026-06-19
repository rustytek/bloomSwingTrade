"""
On-demand long-history archive for arbitrary-era backtests.

The rolling StockCache holds ~5y and stays current for the live screener.
For backtests over older eras (e.g. "2 years, 10 years ago") this module
fetches the requested span from yfinance once, stores the widest range seen
per ticker in the `history_archive` table, and reuses it on later runs.
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

import yfinance as yf
from sqlalchemy.orm import Session

from database.models import HistoryArchive

logger = logging.getLogger(__name__)

# Calendar-day buffer fetched before the test start so the 200-day MA regime
# gate and strategy warmup (~220 bars) have data before the first rebalance.
WARMUP_BUFFER_DAYS = 480


def shift_days(iso: str, days: int) -> str:
    y, m, d = (int(x) for x in iso.split("-")[:3])
    return (date(y, m, d) + timedelta(days=days)).isoformat()


def _fetch_range_sync(ticker: str, start: str, end: str) -> list[dict]:
    """Fetch daily OHLCV for [start, end] (end inclusive) from yfinance."""
    try:
        t = yf.Ticker(ticker)
        # yfinance `end` is exclusive — nudge it forward a day to include it.
        hist = t.history(start=start, end=shift_days(end, 1), interval="1d", auto_adjust=True)
        if hist.empty:
            return []
        bars = []
        for i, (dt, row) in enumerate(hist.iterrows()):
            bars.append({
                "i": i,
                "date": dt.strftime("%Y-%m-%d"),
                "open": round(float(row["Open"]), 4),
                "high": round(float(row["High"]), 4),
                "low": round(float(row["Low"]), 4),
                "close": round(float(row["Close"]), 4),
                "vol": int(row["Volume"]),
            })
        return bars
    except Exception as e:
        logger.warning("Archive fetch error %s [%s..%s]: %s", ticker, start, end, e)
        return []


def ensure_archive(db: Session, ticker: str, need_start: str, need_end: str) -> list[dict]:
    """Return archived bars for `ticker` covering [need_start, need_end],
    fetching/widening from yfinance as needed. Returns the full stored range."""
    row = db.query(HistoryArchive).filter(HistoryArchive.ticker == ticker).first()
    if row and row.start_date <= need_start and row.end_date >= need_end:
        try:
            return json.loads(row.bars_json or "[]")
        except Exception:
            pass  # corrupt → refetch below

    # Fetch the union of what we need and what we already have (one wide pull).
    fetch_start = min(need_start, row.start_date) if row else need_start
    fetch_end = max(need_end, row.end_date) if row else need_end
    bars = _fetch_range_sync(ticker, fetch_start, fetch_end)
    if not bars:
        # Fall back to whatever we already had, if anything.
        if row:
            try:
                return json.loads(row.bars_json or "[]")
            except Exception:
                return []
        return []

    actual_start, actual_end = bars[0]["date"], bars[-1]["date"]
    payload = json.dumps(bars)
    if row:
        row.bars_json = payload
        row.start_date = actual_start
        row.end_date = actual_end
        row.fetched_at = datetime.now(timezone.utc)
    else:
        db.add(HistoryArchive(
            ticker=ticker, bars_json=payload,
            start_date=actual_start, end_date=actual_end,
            fetched_at=datetime.now(timezone.utc),
        ))
    db.commit()
    return bars


def load_archived_histories(
    db: Session, tickers: list[str], need_start: str, need_end: str, max_workers: int = 6,
) -> dict[str, list[dict]]:
    """Ensure + load archived histories for many tickers (concurrent fetch).

    Each ticker uses its own DB session so the threads don't share one Session.
    """
    from database.db import SessionLocal

    def _one(ticker: str):
        s = SessionLocal()
        try:
            return ticker, ensure_archive(s, ticker, need_start, need_end)
        finally:
            s.close()

    out: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for ticker, bars in pool.map(_one, tickers):
            if bars:
                out[ticker] = bars
    return out
