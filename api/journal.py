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
        # The stop the R-multiple was measured against (null on legacy rows,
        # where stop_loss was used as the denominator instead).
        "initial_stop": t.initial_stop,
        "target": t.target,
        "strategy": t.strategy,
        "pnl": t.pnl,
        "pnl_pct": t.pnl_pct,
        "r_multiple": t.r_multiple,
        # The plan's intent, captured at entry. `planned_entry` is null on
        # legacy rows and on entries made outside the Plan-a-Trade flow — null
        # means UNKNOWN, not "filled exactly at plan".
        "planned_entry": t.planned_entry,
        "planned_entry_high": t.planned_entry_high,
        "thesis": t.thesis,
        "invalidation": t.invalidation,
        "time_stop_days": t.time_stop_days,
        "notes": t.notes,
        "closed_at": t.closed_at.isoformat() if t.closed_at else None,
    }


def _stats(trades: list[ClosedTrade]) -> dict:
    if not trades:
        return {
            "count": 0, "win_rate": None, "wins": 0, "losses": 0, "scratch": 0,
            "avg_r": None, "expectancy_r": None, "r_sample": 0,
            "total_pnl": 0.0, "avg_win_pct": None, "avg_loss_pct": None,
        }
    # A breakeven exit (pnl == 0) is neither a win nor a loss — counting it as a
    # loss dragged win_rate down. Split it out as `scratch` and compute
    # win_rate = wins / (wins + losses), excluding scratches from the denominator.
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]
    scratch = [t for t in trades if t.pnl == 0]
    decided = len(wins) + len(losses)
    win_rate = (len(wins) / decided * 100) if decided else None

    # NOTE ON MIXED DENOMINATORS: r_multiple is null whenever no valid stop below
    # cost was recorded, so avg_r / expectancy_r are computed over only the
    # subset of trades that HAVE an R — while win_rate spans every decided trade.
    # `r_sample` makes that subset size explicit so the two aren't read as
    # describing the same sample.
    r_values = [t.r_multiple for t in trades if t.r_multiple is not None]
    avg_r = sum(r_values) / len(r_values) if r_values else None
    # Expectancy in R: currently just the mean realized R over that subset.
    expectancy_r = avg_r
    return {
        "count": len(trades),
        "win_rate": round(win_rate, 1) if win_rate is not None else None,
        "wins": len(wins),
        "losses": len(losses),
        "scratch": len(scratch),
        "avg_r": round(avg_r, 2) if avg_r is not None else None,
        "expectancy_r": round(expectancy_r, 2) if expectancy_r is not None else None,
        "r_sample": len(r_values),
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
    # `trades` above is the LIMIT-ed display page. Every aggregate — stats,
    # per-strategy breakdown and the equity curve — is computed from ALL of the
    # user's closed trades, so the headline numbers don't silently change when
    # the page limit changes or the journal grows past it.
    all_trades = db.query(ClosedTrade).filter(ClosedTrade.user_id == user.id).all()
    # Per-strategy breakdown (full history, not the page)
    by_strategy: dict[str, list[ClosedTrade]] = {}
    for t in all_trades:
        by_strategy.setdefault(t.strategy or "unspecified", []).append(t)
    return {
        "trades": [_trade_dict(t) for t in trades],
        "stats": _stats(all_trades),
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
