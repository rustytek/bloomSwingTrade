import json
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from database.db import get_db
from database.models import User, WatchlistItem, StockCache
from auth.deps import get_current_user
from services.tickers import normalize_ticker

router = APIRouter(prefix="/api/watchlist", tags=["watchlist"])


class AddTickerRequest(BaseModel):
    ticker: str
    notes: str | None = None


class AddBatchRequest(BaseModel):
    tickers: list[str]


@router.get("")
def get_watchlist(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    items = db.query(WatchlistItem).filter(WatchlistItem.user_id == user.id).all()
    # Build cache lookup for sector/name enrichment
    tickers = [item.ticker for item in items]
    cache_rows = db.query(StockCache).filter(StockCache.ticker.in_(tickers)).all() if tickers else []
    cache_map = {}
    for row in cache_rows:
        if row.quote_json:
            try:
                q = json.loads(row.quote_json)
                cache_map[row.ticker] = {"sector": q.get("sector"), "name": q.get("name"), "price": q.get("price")}
            except Exception:
                pass
    return [
        {
            "ticker": item.ticker,
            "added_at": item.added_at.isoformat(),
            "notes": item.notes,
            "sector": cache_map.get(item.ticker, {}).get("sector"),
            "name": cache_map.get(item.ticker, {}).get("name"),
            "price": cache_map.get(item.ticker, {}).get("price"),
        }
        for item in items
    ]


@router.post("", status_code=status.HTTP_201_CREATED)
def add_ticker(
    req: AddTickerRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    ticker = normalize_ticker(req.ticker)

    existing = (
        db.query(WatchlistItem)
        .filter(WatchlistItem.user_id == user.id, WatchlistItem.ticker == ticker)
        .first()
    )
    if existing:
        return {"ticker": ticker, "added_at": existing.added_at.isoformat(), "notes": existing.notes}

    item = WatchlistItem(user_id=user.id, ticker=ticker, notes=req.notes)
    db.add(item)
    db.commit()
    db.refresh(item)
    return {"ticker": item.ticker, "added_at": item.added_at.isoformat(), "notes": item.notes}


@router.post("/batch", status_code=status.HTTP_201_CREATED)
def add_batch(
    req: AddBatchRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    added = []
    for raw in req.tickers:
        try:
            ticker = normalize_ticker(raw)
        except HTTPException:
            continue
        existing = (
            db.query(WatchlistItem)
            .filter(WatchlistItem.user_id == user.id, WatchlistItem.ticker == ticker)
            .first()
        )
        if not existing:
            item = WatchlistItem(user_id=user.id, ticker=ticker)
            db.add(item)
            added.append(ticker)
    db.commit()
    return {"added": added}


@router.delete("/{ticker}", status_code=status.HTTP_204_NO_CONTENT)
def remove_ticker(
    ticker: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    item = (
        db.query(WatchlistItem)
        .filter(WatchlistItem.user_id == user.id, WatchlistItem.ticker == normalize_ticker(ticker))
        .first()
    )
    if not item:
        raise HTTPException(status_code=404, detail="Ticker not in watchlist")
    db.delete(item)
    db.commit()


@router.post("/auto-populate", status_code=status.HTTP_200_OK)
def auto_populate_watchlist(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Score all cached stocks/ETFs and add the top 30 by composite quality to the watchlist.
    Uses all filter dimensions: fundamentals, technical, momentum, risk ratios, and score.
    Enforces max 5 per sector for diversification.
    """
    rows = db.query(StockCache).filter(StockCache.quote_json.isnot(None)).all()

    stocks = []
    for row in rows:
        try:
            q = json.loads(row.quote_json)
            if q and q.get("ticker"):
                stocks.append(q)
        except Exception:
            pass

    def composite_rank(q: dict) -> float:
        sc = q.get("score") or {}
        overall = sc.get("o") or 0
        score_f = sc.get("f") or 0
        score_t = sc.get("t") or 0
        score_m = sc.get("m") or 0
        sharpe = q.get("sharpe") or 0
        sortino = q.get("sortino") or q.get("gain_sharpe") or 0
        calmar = q.get("calmar") or 0
        info_ratio = q.get("info_ratio") or 0
        rsi = q.get("rsi") or 50
        vs_ma200 = q.get("vs_ma200") or 0
        vs_ma50 = q.get("vs_ma50") or 0
        vol_r = q.get("vol_r") or 0
        p52w = q.get("p52w") or 0
        max_dd = q.get("max_dd_1m") or 50
        macd = q.get("macd_sig") or ""
        gc = q.get("gc") or False

        rank = overall * 3.0           # 0-15 from composite score
        rank += score_f * 0.5          # fundamentals bonus
        rank += score_t * 0.5          # technicals bonus
        rank += score_m * 0.5          # momentum bonus
        rank += min(2.0, max(0, sharpe))         # Sharpe up to +2
        rank += min(1.5, max(0, sortino * 0.5))  # Sortino up to +1.5
        rank += min(1.0, max(0, calmar * 0.3))   # Calmar up to +1
        rank += min(0.5, max(0, info_ratio * 0.3))  # Info ratio up to +0.5
        rank += 1.0 if 40 <= rsi <= 65 else 0   # RSI sweet spot
        rank += 1.0 if vs_ma200 > 5 else 0.5 if vs_ma200 > 0 else 0  # Above MA200
        rank += 0.5 if vs_ma50 > 0 else 0        # Above MA50
        rank += 0.5 if vol_r > 1.2 else 0        # Volume confirmation
        rank += 0.5 if p52w > 60 else 0          # Upper half of 52W range
        rank += 0.5 if macd == "bullish" else 0   # MACD bullish
        rank += 0.5 if gc else 0                  # Golden cross
        rank -= min(3.0, max_dd * 0.1)            # Penalize drawdown
        return rank

    # Minimum quality threshold
    qualified = [q for q in stocks if (q.get("score") or {}).get("o", 0) >= 3]
    if len(qualified) < 15:
        qualified = [q for q in stocks if (q.get("score") or {}).get("o", 0) >= 2]

    qualified.sort(key=composite_rank, reverse=True)

    # Sector diversity: max 5 per sector
    sector_count: dict = {}
    selected = []
    for q in qualified:
        sector = q.get("sector") or "Unknown"
        if sector_count.get(sector, 0) >= 5:
            continue
        selected.append(q)
        sector_count[sector] = sector_count.get(sector, 0) + 1
        if len(selected) >= 30:
            break

    # Add to watchlist (skip existing)
    added = []
    skipped = []
    for q in selected:
        try:
            ticker = normalize_ticker(q.get("ticker", ""))
        except HTTPException:
            continue
        existing = (
            db.query(WatchlistItem)
            .filter(WatchlistItem.user_id == user.id, WatchlistItem.ticker == ticker)
            .first()
        )
        if existing:
            skipped.append(ticker)
        else:
            item = WatchlistItem(user_id=user.id, ticker=ticker,
                                 notes="Auto-added by SwingTrader top-30 screener")
            db.add(item)
            added.append(ticker)

    db.commit()
    return {"added": added, "skipped": skipped, "total": len(selected)}
