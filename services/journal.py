"""Journal writes shared by the portfolio API and the broker fill sync.

`journal_close` / `compute_r_multiple` used to live in `api/portfolio.py`,
which forced `services/broker_service.py` to import from the API layer
(services -> api inversion, one refactor away from a real import cycle).
They now live here; `api/portfolio.py` re-exports both names so existing
call sites keep working unchanged.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy.orm import Session

from database.models import ClosedTrade, PortfolioPosition


def compute_r_multiple(avg_cost: float, exit_price: float,
                       initial_stop: float | None, stop_loss: float | None):
    """Realized R for a closed long, measured against the INITIAL stop.

    R must reflect the risk actually taken when the trade was opened. Using the
    current (possibly trailed-up) stop as the denominator shrinks the
    denominator over the life of the trade and inflates every recorded R.
    `initial_stop` is null on legacy rows written before the column existed —
    those fall back to stop_loss. Returns None when no valid stop below cost
    exists (a stop at or above cost has no meaningful R).
    """
    r_stop = initial_stop if initial_stop is not None else stop_loss
    if r_stop is None or avg_cost - r_stop <= 0:
        return None, r_stop
    return round((exit_price - avg_cost) / (avg_cost - r_stop), 2), r_stop


def journal_close(db: Session, pos: PortfolioPosition, exit_price: float,
                  exit_date: date | None = None, notes: str | None = None) -> ClosedTrade:
    """Archive `pos` to the journal at `exit_price` and delete it. Commits.

    The single implementation behind POST /api/portfolio/{ticker}/close and the
    broker fill sync (services/broker_service.py), so a Robinhood fill and a
    manual close write an identical journal row.
    """
    cost_basis = pos.shares * pos.avg_cost
    pnl = pos.shares * exit_price - cost_basis
    pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0.0

    # R-multiple against the INITIAL stop (legacy rows fall back to stop_loss).
    r_multiple, r_stop = compute_r_multiple(
        pos.avg_cost, exit_price, pos.initial_stop, pos.stop_loss
    )

    trade = ClosedTrade(
        user_id=pos.user_id,
        ticker=pos.ticker,
        shares=pos.shares,
        avg_cost=pos.avg_cost,
        exit_price=exit_price,
        entry_date=pos.entry_date,
        exit_date=exit_date or date.today(),
        stop_loss=pos.stop_loss,
        initial_stop=r_stop,
        target=pos.target,
        strategy=pos.strategy,
        pnl=round(pnl, 2),
        pnl_pct=round(pnl_pct, 2),
        r_multiple=r_multiple,
        # Carry the plan's intent into the journal — without planned_entry the
        # entry-chasing execution metric has nothing to measure the fill against.
        planned_entry=pos.planned_entry,
        planned_entry_high=pos.planned_entry_high,
        thesis=pos.thesis,
        invalidation=pos.invalidation,
        time_stop_days=pos.time_stop_days,
        notes=notes or pos.notes,
        opened_at=pos.added_at,
        closed_at=datetime.now(timezone.utc),
    )
    db.add(trade)
    db.delete(pos)
    db.commit()
    db.refresh(trade)
    return trade
