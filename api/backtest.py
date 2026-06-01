from datetime import date

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from auth.deps import get_current_user
from database.db import get_db
from database.models import User
from services.backtest import (
    build_decision_cockpit,
    build_watchlist_replay,
    create_watchlist_snapshot,
    run_walk_forward_backtest,
)


router = APIRouter(prefix="/api/backtest", tags=["backtest"])


@router.get("/walk-forward")
def walk_forward(
    source: str = Query("watchlist", pattern="^(watchlist|portfolio|both)$"),
    top_n: int = Query(5, ge=1, le=20),
    rebalance_days: int = Query(5, ge=5, le=21),
    cost_bps: float = Query(10, ge=0, le=100),
    spy_regime: bool = Query(True),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return run_walk_forward_backtest(
        db=db,
        user_id=user.id,
        source=source,
        top_n=top_n,
        rebalance_days=rebalance_days,
        cost_bps=cost_bps,
        spy_regime=spy_regime,
    )


@router.get("/cockpit")
def cockpit(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return build_decision_cockpit(db, user.id)


@router.post("/watchlist-replay/snapshot")
def snapshot_watchlist(
    week_start: date | None = Query(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return create_watchlist_snapshot(db, user.id, week_start=week_start, notes="manual")


@router.get("/watchlist-replay")
def watchlist_replay(
    weeks: int = Query(8, ge=1, le=26),
    top_n: int = Query(10, ge=1, le=20),
    spy_regime: bool = Query(True),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return build_watchlist_replay(db, user.id, weeks=weeks, top_n=top_n, spy_regime=spy_regime)
