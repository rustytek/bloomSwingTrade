"""
Evidence-layer API: the strategy scorecard and the strategy x regime edge matrix.

The edge matrix is SLOW (one full walk-forward per actionable strategy over the
whole universe). A cold GET therefore never builds it — it returns an explicit
"not computed yet" payload so the page can render instantly and offer a button.
Only `?refresh=true` does the work.
"""
from __future__ import annotations

import anyio
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from auth.deps import get_current_user
from database.db import get_db
from database.models import User
from services import edge_matrix as em
from services.regime import evidence_adjustment
from services.scorecard import build_scorecard, execution_quality

router = APIRouter(tags=["scorecard"])


@router.get("/api/scorecard")
async def get_scorecard(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Realized vs expected per strategy, plus ranked execution-quality leaks."""
    return {
        "scorecard": build_scorecard(db, user.id),
        "execution_quality": execution_quality(db, user.id),
    }


@router.get("/api/edge-matrix")
async def get_edge_matrix(
    refresh: bool = Query(False, description="Rebuild the matrix (SLOW — runs one "
                                             "walk-forward backtest per strategy)"),
    source: str = Query("universe", description="universe | watchlist | portfolio"),
    mode: str | None = Query(None, description="Backtest mode; defaults to trade_plan "
                                               "when services/backtest.py supports it"),
    period: str = Query("all", description="1Y | 2Y | all"),
    quadrant: str | None = Query(None, description="Preview the evidence override for "
                                                   "this quadrant"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    kwargs = {"period": period}
    if mode:
        kwargs["mode"] = mode

    if refresh:
        # Run the blocking build off the event loop so one refresh does not stall
        # every other request on the worker.
        matrix = await anyio.to_thread.run_sync(
            lambda: em.build_edge_matrix(db, user.id, source=source, **kwargs)
        )
        status = "computed"
    else:
        matrix = em.get_cached_matrix(user.id, source=source, **kwargs) \
            or em.get_cached_matrix(user.id)
        status = "cached" if matrix else "not_computed"

    if matrix is None:
        return {
            "status": "not_computed",
            "matrix": None,
            "message": (
                "The edge matrix has not been computed yet. It runs a full walk-forward "
                "backtest for every actionable strategy, which takes minutes — call this "
                "endpoint again with ?refresh=true to build it."
            ),
            "strategies": em.actionable_strategy_ids(),
            "quadrants": [{"id": q, "label": em.QUADRANT_INFO[q]["label"]} for q in em.QUADRANTS],
            "thresholds": {
                "min_periods_judge": em.MIN_PERIODS_JUDGE,
                "min_periods_high_confidence": em.MIN_PERIODS_HIGH,
                "min_periods_promote": em.MIN_PERIODS_PROMOTE,
            },
        }

    payload = {"status": status, "matrix": matrix}
    if quadrant:
        payload["evidence_override_preview"] = evidence_adjustment(quadrant, matrix, force=True)
    return payload


@router.post("/api/edge-matrix/invalidate")
async def invalidate_edge_matrix(user: User = Depends(get_current_user)):
    em.invalidate_cache(user.id)
    return {"ok": True}
