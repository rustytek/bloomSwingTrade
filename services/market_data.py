"""
yfinance data service with in-memory + SQLite caching.
All public methods are async-friendly (run blocking yfinance in thread pool).
"""
import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from functools import partial
from typing import Optional

import yfinance as yf
from sqlalchemy.orm import Session

from config import get_settings
from database.models import StockCache
from services.indicators import (
    calc_ma, calc_ema, calc_rsi, calc_macd, calc_bollinger,
    macd_signal_label, compute_score, compute_performance_metrics,
)

logger = logging.getLogger(__name__)
settings = get_settings()

# In-memory layer on top of SQLite cache
_mem_cache: dict[str, dict] = {}


def _last_market_close() -> datetime:
    """Return the most recent NYSE market close (4:00 PM ET, weekdays) as UTC.

    Data cached after this time is considered fresh; data cached before is stale.
    Ensures a single daily refresh after market close, not a rolling 24-hour timer.
    """
    try:
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
        use_tz = True
    except Exception:
        m = datetime.now(timezone.utc).month
        now_et = (datetime.now(timezone.utc) + timedelta(hours=-4 if 4 <= m <= 10 else -5)).replace(tzinfo=None)
        use_tz = False

    for days_back in range(8):
        day = now_et - timedelta(days=days_back)
        if day.weekday() >= 5:          # skip Saturday (5) and Sunday (6)
            continue
        close = day.replace(hour=16, minute=0, second=0, microsecond=0)
        now_cmp   = now_et.replace(tzinfo=None) if use_tz else now_et
        close_cmp = close.replace(tzinfo=None)  if use_tz else close
        if days_back == 0 and now_cmp < close_cmp:
            continue                    # today's close hasn't happened yet
        if use_tz:
            return close.astimezone(timezone.utc)
        offset_h = 4 if 4 <= close.month <= 10 else 5
        return close.replace(tzinfo=timezone.utc) + timedelta(hours=offset_h)

    return datetime.now(timezone.utc) - timedelta(hours=25)   # safety fallback


def _is_fresh(cached_at: datetime) -> bool:
    """Data is fresh if cached after the most recent NYSE market close (4pm ET).

    Refreshes once per day after market close — not on a rolling 24-hour timer.
    Manual refresh (force=True) bypasses this check.
    """
    if cached_at.tzinfo is None:
        cached_at = cached_at.replace(tzinfo=timezone.utc)
    return cached_at >= _last_market_close()


# ── Raw yfinance helpers (blocking — run in thread pool) ──────────────────

def _fetch_quote_sync(ticker: str) -> dict:
    """Fetch quote + fundamentals synchronously via yfinance."""
    t = yf.Ticker(ticker)
    info = t.fast_info
    full_info = {}
    try:
        full_info = t.info or {}
    except Exception as e:
        logger.warning(f"t.info failed for {ticker}: {e}")

    # fast_info fields (always available)
    price = getattr(info, "last_price", None) or full_info.get("currentPrice")
    prev_close = getattr(info, "previous_close", None) or full_info.get("previousClose")
    mkt_cap = getattr(info, "market_cap", None) or full_info.get("marketCap")

    chg_pct = None
    if price and prev_close and prev_close != 0:
        chg_pct = ((price - prev_close) / prev_close) * 100

    pe = full_info.get("trailingPE")
    fwd_pe = full_info.get("forwardPE")
    eps_grw = full_info.get("earningsGrowth")
    rev_grw = full_info.get("revenueGrowth")
    pm = full_info.get("profitMargins")
    debt_eq = full_info.get("debtToEquity")
    beta = full_info.get("beta")
    div_yield = full_info.get("dividendYield")
    fcf = full_info.get("freeCashflow")
    name = full_info.get("shortName") or full_info.get("longName") or ticker
    sector = full_info.get("sector") or "Unknown"
    industry = full_info.get("industry") or ""
    description = full_info.get("longBusinessSummary") or full_info.get("description") or ""
    if not description:
        try:
            description = t.get_info().get("longBusinessSummary") or ""
        except Exception:
            pass

    # Convert units
    if mkt_cap:
        mkt_cap = mkt_cap / 1e9  # → $B
    if pm is not None:
        pm = pm * 100  # → %
    if eps_grw is not None:
        eps_grw = eps_grw * 100
    if rev_grw is not None:
        rev_grw = rev_grw * 100
    if div_yield is not None:
        div_yield = div_yield * 100
    if debt_eq is not None:
        debt_eq = debt_eq / 100  # yfinance returns e.g. 152 for 1.52

    quote_type = full_info.get("quoteType", "EQUITY")  # EQUITY | ETF | MUTUALFUND

    return {
        "ticker": ticker.upper(),
        "name": name,
        "sector": sector,
        "industry": industry,
        "description": description,
        "quote_type": quote_type,
        "price": price,
        "prev_close": prev_close,
        "chg_pct": round(chg_pct, 2) if chg_pct is not None else None,
        "mkt_cap": round(mkt_cap, 2) if mkt_cap else None,
        "pe": round(pe, 2) if pe else None,
        "fwd_pe": round(fwd_pe, 2) if fwd_pe else None,
        "eps_grw": round(eps_grw, 2) if eps_grw is not None else None,
        "rev_grw": round(rev_grw, 2) if rev_grw is not None else None,
        "pm": round(pm, 2) if pm is not None else None,
        "debt_eq": round(debt_eq, 3) if debt_eq is not None else None,
        "beta": round(beta, 2) if beta is not None else None,
        "div": round(div_yield, 2) if div_yield is not None else None,
        "fcf_pos": fcf is not None and fcf > 0,
        "earn_beat": full_info.get("earningsBeat", False),
        "earn_soon": False,  # not reliably in yfinance free tier
    }


def _fetch_history_sync(ticker: str, period: str = "6mo") -> list[dict]:
    """Fetch OHLCV history synchronously."""
    t = yf.Ticker(ticker)
    hist = t.history(period=period, interval="1d", auto_adjust=True)
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


def _enrich_with_technicals(quote: dict, history: list[dict]) -> dict:
    """Compute all technical indicators and attach to quote dict."""
    if not history:
        return quote

    closes = [b["close"] for b in history]
    highs = [b["high"] for b in history]
    lows = [b["low"] for b in history]
    vols = [b["vol"] for b in history]

    rsi_series = calc_rsi(closes, 14)
    rsi_val = next((v for v in reversed(rsi_series) if v is not None), None)

    macd_data = calc_macd(closes)
    macd_sig = macd_signal_label(macd_data["macd_line"], macd_data["signal"])

    ma50 = calc_ma(closes, 50)
    ma200 = calc_ma(closes, 200)
    ma50_val = next((v for v in reversed(ma50) if v is not None), None)
    ma200_val = next((v for v in reversed(ma200) if v is not None), None)

    price = quote.get("price") or closes[-1]
    vs_ma200 = ((price - ma200_val) / ma200_val * 100) if ma200_val else None
    vs_ma50 = ((price - ma50_val) / ma50_val * 100) if ma50_val else None

    # Golden / Death Cross: MA50 crosses MA200
    gc = dc = False
    ma50_clean = [v for v in ma50 if v is not None]
    ma200_clean = [v for v in ma200 if v is not None]
    min_len = min(len(ma50_clean), len(ma200_clean))
    if min_len >= 2:
        if ma50_clean[-1] > ma200_clean[-1] and ma50_clean[-2] <= ma200_clean[-2]:
            gc = True
        elif ma50_clean[-1] < ma200_clean[-1] and ma50_clean[-2] >= ma200_clean[-2]:
            dc = True
        elif ma50_clean[-1] > ma200_clean[-1]:
            gc = True  # currently in golden cross territory
        else:
            dc = True  # currently in death cross territory

    # 52-week position
    if len(closes) >= 2:
        high52 = max(highs)
        low52 = min(lows)
        p52w = ((price - low52) / (high52 - low52) * 100) if high52 != low52 else 50
    else:
        p52w = None

    # Volume ratio (avg last 20 bars vs avg last 5 bars)
    avg_vol_20 = sum(vols[-20:]) / min(20, len(vols)) if vols else 0
    avg_vol_5 = sum(vols[-5:]) / min(5, len(vols)) if vols else 0
    vol_r = (avg_vol_5 / avg_vol_20) if avg_vol_20 > 0 else 1.0

    # SPY closes from memory cache (for Info Ratio) — best-effort
    spy_closes = None
    spy_entry = _mem_cache.get("history:SPY")
    if spy_entry and spy_entry.get("data"):
        spy_closes = [b["close"] for b in spy_entry["data"]]

    # Beta from yfinance (for Treynor)
    beta = quote.get("beta")

    perf = compute_performance_metrics(closes, spy_closes=spy_closes, beta=beta)

    # Sparkline: last 7 closes for in-table mini trend chart
    spark = [round(v, 2) for v in closes[-7:]]

    enriched = {
        **quote,
        "spark": spark,
        "rsi": round(rsi_val, 1) if rsi_val is not None else None,
        "macd_sig": macd_sig,
        "vs_ma50": round(vs_ma50, 1) if vs_ma50 is not None else None,
        "vs_ma200": round(vs_ma200, 1) if vs_ma200 is not None else None,
        "gc": gc,
        "dc": dc,
        "p52w": round(p52w, 1) if p52w is not None else None,
        "vol_r": round(vol_r, 2),
        **perf,
    }
    enriched["score"] = compute_score(enriched)
    return enriched


# ── Public async interface ─────────────────────────────────────────────────

async def get_quote(ticker: str, db: Session, force_refresh: bool = False) -> Optional[dict]:
    """Return enriched quote dict, using cache."""
    ticker = ticker.upper()
    cache_key = f"quote:{ticker}"

    # 1. Check memory cache
    if not force_refresh and cache_key in _mem_cache:
        entry = _mem_cache[cache_key]
        if _is_fresh(entry["cached_at"]):
            return entry["data"]

    # 2. Check SQLite cache
    row: Optional[StockCache] = db.query(StockCache).filter(StockCache.ticker == ticker).first()
    if not force_refresh and row and row.quote_json and _is_fresh(row.cached_at):
        data = json.loads(row.quote_json)
        _mem_cache[cache_key] = {"data": data, "cached_at": row.cached_at}
        return data

    # 3. Fetch from yfinance
    try:
        loop = asyncio.get_event_loop()
        quote = await loop.run_in_executor(None, _fetch_quote_sync, ticker)
        history = await get_history(ticker, db)
        if not history:
            history = []
        enriched = _enrich_with_technicals(quote, history)
    except Exception as e:
        logger.error(f"Failed to fetch {ticker}: {e}")
        # Write a failure marker so we don't retry until after next market close.
        # quote_json stays None so the ticker won't appear in screener results,
        # but cached_at is set so _is_fresh() treats it as "done for today".
        try:
            now = datetime.now(timezone.utc)
            fail_row = db.query(StockCache).filter(StockCache.ticker == ticker).first()
            if fail_row:
                fail_row.cached_at = now   # refresh timestamp, leave quote_json as-is
            else:
                db.add(StockCache(ticker=ticker, quote_json=None, cached_at=now))
            db.commit()
        except Exception:
            db.rollback()
        return None

    # 4. Persist to cache (re-query row — get_history may have inserted it)
    now = datetime.now(timezone.utc)
    quote_json = json.dumps(enriched)
    row = db.query(StockCache).filter(StockCache.ticker == ticker).first()
    if row:
        row.quote_json = quote_json
        row.cached_at = now
    else:
        row = StockCache(ticker=ticker, quote_json=quote_json, cached_at=now)
        db.add(row)
    db.commit()

    _mem_cache[cache_key] = {"data": enriched, "cached_at": now}
    return enriched


async def get_history(ticker: str, db: Session, period: str = "6mo", force_refresh: bool = False) -> list[dict]:
    """Return OHLCV history list, using cache."""
    ticker = ticker.upper()
    cache_key = f"history:{ticker}"

    if not force_refresh and cache_key in _mem_cache:
        entry = _mem_cache[cache_key]
        if _is_fresh(entry["cached_at"]):
            return entry["data"]

    row: Optional[StockCache] = db.query(StockCache).filter(StockCache.ticker == ticker).first()
    if not force_refresh and row and row.history_json and _is_fresh(row.cached_at):
        data = json.loads(row.history_json)
        _mem_cache[cache_key] = {"data": data, "cached_at": row.cached_at}
        return data

    try:
        loop = asyncio.get_event_loop()
        history = await loop.run_in_executor(None, partial(_fetch_history_sync, ticker, period))
    except Exception as e:
        logger.error(f"Failed to fetch history for {ticker}: {e}")
        return []

    now = datetime.now(timezone.utc)
    history_json = json.dumps(history)
    if row:
        row.history_json = history_json
        row.cached_at = now
    else:
        row = StockCache(ticker=ticker, history_json=history_json, cached_at=now)
        db.add(row)
    db.commit()

    _mem_cache[cache_key] = {"data": history, "cached_at": now}
    return history


async def get_batch(tickers: list[str], db: Session, force_refresh: bool = False) -> list[dict]:
    """Fetch multiple quotes concurrently."""
    tasks = [get_quote(t, db, force_refresh) for t in tickers]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if isinstance(r, dict)]


def get_macd_series(history: list[dict]) -> dict:
    """Return full MACD series for charting."""
    closes = [b["close"] for b in history]
    return calc_macd(closes)


def get_rsi_series(history: list[dict]) -> list:
    closes = [b["close"] for b in history]
    return calc_rsi(closes, 14)


def get_ma_series(history: list[dict], period: int) -> list:
    closes = [b["close"] for b in history]
    return calc_ma(closes, period)


def get_bb_series(history: list[dict]) -> list:
    closes = [b["close"] for b in history]
    return calc_bollinger(closes)


# ── Rate Limiting & Tracking ───────────────────────────────────────────────

class APIRateLimiter:
    def __init__(self, limit_per_hour: int = 2000):
        self.limit = limit_per_hour
        self.count = 0
        self.current_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        self.lock = asyncio.Lock()

    async def increment(self) -> bool:
        """Increments the counter. Returns True if successful, False if limit reached."""
        now_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        async with self.lock:
            if now_hour > self.current_hour:
                self.current_hour = now_hour
                self.count = 0
            
            if self.count >= self.limit:
                return False
            self.count += 1
            return True

    def get_status(self) -> dict:
        """Returns current usage dictionary"""
        now_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        if now_hour > self.current_hour:
            return {"used": 0, "limit": self.limit, "reset": (now_hour + timedelta(hours=1)).isoformat()}
        return {"used": self.count, "limit": self.limit, "reset": (self.current_hour + timedelta(hours=1)).isoformat()}

# Global rate limiter instance
limiter = APIRateLimiter(limit_per_hour=2000)

def is_market_open() -> bool:
    """Returns True if the US market is currently open"""
    try:
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        # Fallback if zoneinfo isn't available
        m = datetime.now(timezone.utc).month
        now_et = (datetime.now(timezone.utc) + timedelta(hours=-4 if 4 <= m <= 10 else -5)).replace(tzinfo=None)
        
    if now_et.weekday() >= 5: # Weekend
        return False
    
    current_time = now_et.time()
    market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= current_time <= market_close


# ── Universe background refresh ────────────────────────────────────────────

def cleanup_old_entries(db: Session) -> int:
    """Delete StockCache rows not updated in the last max_age_days days."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=settings.cache_max_age_days)
    deleted = db.query(StockCache).filter(StockCache.cached_at < cutoff).delete()
    db.commit()
    if deleted:
        logger.info(f"Cleaned up {deleted} stale cache entries (> {settings.cache_max_age_days} days old)")
    return deleted


def invalidate_legacy_cache(db: Session) -> int:
    """Reset cached_at for entries that pre-date performance metrics (missing 'sharpe' field).

    This forces refresh_universe to re-fetch those tickers so risk ratio fields
    (sharpe, sortino, calmar, etc.) get computed and stored. Safe to call on every
    startup — once entries are refreshed they will have the field and be skipped.
    """
    epoch = datetime(2000, 1, 1, tzinfo=timezone.utc)
    rows = db.query(StockCache).filter(StockCache.quote_json.isnot(None)).all()
    count = 0
    for row in rows:
        try:
            data = json.loads(row.quote_json)
            if "sharpe" not in data:
                row.cached_at = epoch
                count += 1
        except Exception:
            pass
    if count:
        db.commit()
        logger.info(f"Invalidated {count} legacy cache entries missing performance metrics — will re-fetch")
    return count


def get_universe_status(db: Session, universe: list[str]) -> dict:
    """Return how many universe tickers have been attempted (loaded) and have fresh data."""
    total = len(universe)
    universe_set = set(universe)
    # Count ALL rows including failure markers (quote_json=None) as "attempted/loaded"
    rows = db.query(StockCache).all()
    loaded = sum(1 for r in rows if r.ticker in universe_set)
    fresh  = sum(1 for r in rows if r.ticker in universe_set
                 and r.quote_json and _is_fresh(r.cached_at))
                 
    last_updated_time = None
    latest = max((r.cached_at for r in rows if r.ticker in universe_set and r.cached_at), default=None)
    if latest:
        try:
            from datetime import timezone
            dt = latest.replace(tzinfo=timezone.utc) if latest.tzinfo is None else latest
            last_updated_time = dt.astimezone().strftime("%I:%M %p").lstrip('0')
        except Exception:
            last_updated_time = latest.strftime("%I:%M %p").lstrip('0')
                 
    # Append the rate limiter status
    limit_status = limiter.get_status()
    return {"total": total, "loaded": loaded, "fresh": fresh, "rate_limit": limit_status, "last_updated_time": last_updated_time}


async def refresh_universe(universe: list[str], db_factory, force: bool = False) -> None:
    """
    Background continuous loop that polls the API while respecting rate limits.
    If market is open, it will query across the hour avoiding limits.
    If market is closed, it sweeps once and then waits.
    """
    # Max concurrent fetches (keep low so rate limit checks are granular)
    sem = asyncio.Semaphore(5)

    async def fetch_one(ticker: str, force_flag: bool):
        # Stop and wait if rate limit is hit
        if not await limiter.increment():
            return False
            
        async with sem:
            db = db_factory()
            try:
                await get_quote(ticker, db, force_refresh=force_flag)
            except Exception as e:
                logger.debug(f"Background fetch skipped {ticker}: {e}")
            finally:
                db.close()
            # Distribute requests to about 2000 per hour (1.8s delay total on average)
            await asyncio.sleep(1.8)
            return True

    while True:
        db = db_factory()
        try:
            existing = {
                r.ticker: r.cached_at
                for r in db.query(StockCache).all()
            }
        finally:
            db.close()

        # Re-check stale tickers
        stale = universe if force else [
            t for t in universe
            if t not in existing or not _is_fresh(existing[t])
        ]

        if not stale:
            # Everything is fresh. Sleep a bit then check again.
            await asyncio.sleep(60)
            force = False # clear force flag after one loop
            continue
            
        logger.info(f"Background universe loop: {len(stale)}/{len(universe)} tickers need updating")
        
        market_open = is_market_open()
        
        for ticker in stale:
            res = await fetch_one(ticker, force)
            if not res:
                logger.warning(f"Hourly API rate limit reached ({limiter.limit}). Pausing background polling until next window.")
                # We hit the limit. Sleep until the next hour begins.
                now = datetime.now(timezone.utc)
                next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
                await asyncio.sleep((next_hour - now).total_seconds())
                break # break the for-loop, while-loop restarts the stale sweep
        
        # If the market was closed, we do not want to constantly run. 
        # By definition, if stale was empty, it skips above. 
        # So we just wait 5 minutes before the next stale check.
        await asyncio.sleep(300)
        force = False
