"""Account-level circuit breaker endpoints (services/circuit_breaker.py, ML4T gap 7)."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from auth.deps import get_current_user
from database.db import get_db
from database.models import EquitySnapshot, User
from services import circuit_breaker, today as today_svc

router = APIRouter(prefix="/api/risk", tags=["risk"])


@router.get("/breaker")
def breaker_state(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Current breaker level, the reasons, the equity metrics behind them, and
    the recent equity snapshots. Records today's intraday snapshot if missing
    (cheap: cached prices + one journal sum, no network)."""
    state = circuit_breaker.current_state(db, user, record=True)
    rows = (db.query(EquitySnapshot)
            .filter(EquitySnapshot.user_id == user.id)
            .order_by(EquitySnapshot.date.desc()).limit(60).all())
    state["snapshots"] = [
        {"date": r.date, "equity": r.equity, "source": r.source,
         "realized_pnl": r.realized_pnl, "unrealized_pnl": r.unrealized_pnl}
        for r in reversed(rows)
    ]
    return state


@router.post("/breaker/acknowledge")
def acknowledge_breaker(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Resume after a HALT — at REDUCED size, recorded with who and when. The
    halt re-engages if the drawdown deepens a further few points."""
    try:
        state = circuit_breaker.acknowledge(db, user)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    # The Playbook payload embeds the breaker state; drop its cached copy.
    today_svc.invalidate_cache(user.id)
    return state
