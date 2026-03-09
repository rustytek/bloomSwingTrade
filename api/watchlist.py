from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from database.db import get_db
from database.models import User, WatchlistItem
from auth.deps import get_current_user

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
    return [
        {
            "ticker": item.ticker,
            "added_at": item.added_at.isoformat(),
            "notes": item.notes,
        }
        for item in items
    ]


@router.post("", status_code=status.HTTP_201_CREATED)
def add_ticker(
    req: AddTickerRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    ticker = req.ticker.upper().strip()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker cannot be empty")

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
        ticker = raw.upper().strip()
        if not ticker:
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
        .filter(WatchlistItem.user_id == user.id, WatchlistItem.ticker == ticker.upper())
        .first()
    )
    if not item:
        raise HTTPException(status_code=404, detail="Ticker not in watchlist")
    db.delete(item)
    db.commit()
