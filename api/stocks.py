from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel
from database.db import get_db
from database.models import User
from auth.deps import get_current_user
from services import market_data
from services.tickers import normalize_ticker, normalize_tickers

router = APIRouter(prefix="/api/stocks", tags=["stocks"])


class BatchRequest(BaseModel):
    tickers: list[str]
    force_refresh: bool = False


@router.get("/{ticker}")
async def get_stock(
    ticker: str,
    force_refresh: bool = Query(False),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    ticker = normalize_ticker(ticker)
    data = await market_data.get_quote(ticker, db, force_refresh=force_refresh)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Could not fetch data for {ticker}")
    return data


@router.get("/{ticker}/history")
async def get_history(
    ticker: str,
    period: str = Query("2y", description="yfinance period: 1mo, 3mo, 6mo, 1y, 2y"),
    force_refresh: bool = Query(False),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    ticker = normalize_ticker(ticker)
    history = await market_data.get_history(ticker, db, period=period, force_refresh=force_refresh)
    if not history:
        raise HTTPException(status_code=404, detail=f"No history found for {ticker}")

    # Attach indicator series for charting
    rsi = market_data.get_rsi_series(history)
    macd = market_data.get_macd_series(history)
    ma50 = market_data.get_ma_series(history, 50)
    ma200 = market_data.get_ma_series(history, 200)
    bb = market_data.get_bb_series(history)

    return {
        "ticker": ticker,
        "bars": history,
        "indicators": {
            "rsi": rsi,
            "macd": macd,
            "ma50": ma50,
            "ma200": ma200,
            "bb": bb,
        },
    }


@router.post("/batch")
async def get_batch(
    req: BatchRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if not req.tickers:
        return []
    tickers = normalize_tickers(req.tickers, limit=100)
    results = await market_data.get_batch(tickers, db, force_refresh=req.force_refresh)
    return results
