from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from auth.deps import get_current_user
from database.db import get_db
from database.models import ClosedTrade, User

router = APIRouter(prefix="/api/journal", tags=["journal"])


def _trade_dict(t: ClosedTrade) -> dict:
    return {
        "id": t.id,
        "ticker": t.ticker,
        "shares": t.shares,
        "avg_cost": t.avg_cost,
        "exit_price": t.exit_price,
        "entry_date": t.entry_date.isoformat() if t.entry_date else None,
        "exit_date": t.exit_date.isoformat() if t.exit_date else None,
        "stop_loss": t.stop_loss,
        "target": t.target,
        "strategy": t.strategy,
        "pnl": t.pnl,
        "pnl_pct": t.pnl_pct,
        "r_multiple": t.r_multiple,
        "notes": t.notes,
        "closed_at": t.closed_at.isoformat() if t.closed_at else None,
    }


def _stats(trades: list[ClosedTrade]) -> dict:
    if not trades:
        return {
            "count": 0, "win_rate": None, "avg_r": None, "expectancy_r": None,
            "total_pnl": 0.0, "avg_win_pct": None, "avg_loss_pct": None,
        }
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    r_values = [t.r_multiple for t in trades if t.r_multiple is not None]
    win_rate = len(wins) / len(trades) * 100
    avg_r = sum(r_values) / len(r_values) if r_values else None
    # Expectancy in R using realized R where available
    expectancy_r = avg_r
    return {
        "count": len(trades),
        "win_rate": round(win_rate, 1),
        "avg_r": round(avg_r, 2) if avg_r is not None else None,
        "expectancy_r": round(expectancy_r, 2) if expectancy_r is not None else None,
        "total_pnl": round(sum(t.pnl for t in trades), 2),
        "avg_win_pct": round(sum(t.pnl_pct for t in wins) / len(wins), 2) if wins else None,
        "avg_loss_pct": round(sum(t.pnl_pct for t in losses) / len(losses), 2) if losses else None,
    }


def _equity_curve(trades: list[ClosedTrade]) -> list[dict]:
    """Cumulative realized P&L over time, oldest→newest, from ALL closed trades.
    Each point: {date, pnl, cum_pnl}. Ordered by exit_date (fallback closed_at)."""
    from datetime import date as _date
    ordered = sorted(
        trades,
        key=lambda t: (t.exit_date or (t.closed_at.date() if t.closed_at else None) or _date.min, t.id),
    )
    curve = []
    cum = 0.0
    for t in ordered:
        cum += t.pnl
        when = t.exit_date or (t.closed_at.date() if t.closed_at else None)
        curve.append({
            "date": when.isoformat() if when else None,
            "ticker": t.ticker,
            "pnl": round(t.pnl, 2),
            "cum_pnl": round(cum, 2),
        })
    return curve


@router.get("")
def get_journal(
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    trades = (
        db.query(ClosedTrade)
        .filter(ClosedTrade.user_id == user.id)
        .order_by(ClosedTrade.closed_at.desc())
        .limit(limit)
        .all()
    )
    # Equity curve uses ALL closed trades, not just the limited page.
    all_trades = db.query(ClosedTrade).filter(ClosedTrade.user_id == user.id).all()
    # Per-strategy breakdown
    by_strategy: dict[str, list[ClosedTrade]] = {}
    for t in trades:
        by_strategy.setdefault(t.strategy or "unspecified", []).append(t)
    return {
        "trades": [_trade_dict(t) for t in trades],
        "stats": _stats(trades),
        "by_strategy": {k: _stats(v) for k, v in by_strategy.items()},
        "equity_curve": _equity_curve(all_trades),
    }


@router.delete("/{trade_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_trade(
    trade_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    trade = (
        db.query(ClosedTrade)
        .filter(ClosedTrade.id == trade_id, ClosedTrade.user_id == user.id)
        .first()
    )
    if not trade:
        raise HTTPException(status_code=404, detail="Journal entry not found")
    db.delete(trade)
    db.commit()
