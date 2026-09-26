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
from services.strategies import STRATEGIES, strategy_catalog
from services import trial_log


router = APIRouter(prefix="/api/backtest", tags=["backtest"])

# Built from the registry so newly added strategies are accepted automatically.
# Only *actionable* strategies are backtestable: the walk-forward engine models a
# long-only equal-weight rotation, so a watchlist-only bearish signal (e.g.
# bear_reversal_watch, actionable=False) would produce a meaningless long equity
# curve. Non-actionable strategies stay in GET /api/backtest/strategies (the
# catalog) flagged `backtestable: false`.
BACKTESTABLE_STRATEGIES = [sid for sid, s in STRATEGIES.items() if s.actionable]
_STRATEGY_PATTERN = "^(" + "|".join(BACKTESTABLE_STRATEGIES) + ")$"


@router.get("/strategies")
def strategies():
    catalog = strategy_catalog()
    for row in catalog:
        # Visible in the catalog either way — just flagged so the UI can grey out
        # the "Run backtest" control for watchlist-only signals.
        row["backtestable"] = bool(row.get("actionable"))
    return {"strategies": catalog, "backtestable": BACKTESTABLE_STRATEGIES}


@router.get("/walk-forward")
def walk_forward(
    strategy: str = Query("momentum_rotation", pattern=_STRATEGY_PATTERN),
    source: str = Query("watchlist", pattern="^(watchlist|portfolio|both|universe)$"),
    top_n: int = Query(5, ge=1, le=20),
    rebalance_days: int = Query(5, ge=5, le=21),
    cost_bps: float = Query(10, ge=0, le=100),
    spy_regime: bool = Query(True),
    regime_ma: int = Query(200, ge=20, le=200),
    period: str = Query("all", pattern="^(1Y|2Y|5Y|all)$"),
    start_date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    end_date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    archive: bool = Query(False),
    # ── Trade-plan simulation ──────────────────────────────────────────────
    # mode=rotation is the historical behaviour and stays the default, so every
    # existing caller/bookmark keeps working. mode=trade_plan simulates what the
    # user actually does live (ATR stop, R target, fixed-fractional sizing,
    # limited slots). The risk overrides below default to the authenticated
    # user's saved settings, matching the Today dashboard.
    mode: str = Query("rotation", pattern="^(rotation|trade_plan)$"),
    account_size: float | None = Query(None, gt=0),
    risk_pct: float | None = Query(None, gt=0, le=100),
    max_positions: int | None = Query(None, ge=1, le=50),
    atr_stop_mult: float | None = Query(None, gt=0, le=10),
    r_multiple: float | None = Query(None, gt=0, le=20),
    exit_rules: str | None = Query(None, pattern=r"^[a-z,]+$"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    # Always draws on the 20-year archive (+ the daily cache), so start_date
    # can sit anywhere in the last 20 years — but the TESTED window is capped
    # at MAX_WINDOW_YEARS inside the engine. Each ticker is loaded only from
    # shortly before the window, so a run costs about what a 5-year run always has.
    result = run_walk_forward_backtest(
        db=db,
        user_id=user.id,
        strategy_id=strategy,
        source=source,
        top_n=top_n,
        rebalance_days=rebalance_days,
        cost_bps=cost_bps,
        spy_regime=spy_regime,
        regime_ma=regime_ma,
        period=period,
        start_date=start_date,
        end_date=end_date,
        archive=archive,
        mode=mode,
        account_size=account_size if account_size is not None else user.account_size,
        risk_pct=risk_pct if risk_pct is not None else user.risk_pct,
        max_positions=max_positions if max_positions is not None else user.max_positions,
        atr_stop_mult=atr_stop_mult if atr_stop_mult is not None else user.atr_stop_mult,
        r_multiple=r_multiple if r_multiple is not None else user.r_multiple,
        exit_rules=exit_rules,
        history="20y",
    )
    # Every run is a trial: record it and deflate the Sharpe for the search
    # (ML4T gap 2b). Never raises — see trial_log.annotate.
    return trial_log.annotate(db, user.id, result)


@router.get("/trials")
def trials(
    strategy: str | None = Query(None, pattern=_STRATEGY_PATTERN),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Every distinct Strategy Lab configuration this user has tried — the
    trial count behind the Deflated Sharpe Ratio. Read-only: there is
    deliberately no delete, since hiding tries would defeat the correction."""
    rows = trial_log.list_trials(db, user.id, strategy)
    return {"trials": rows, "count": len(rows)}


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
