"""
Evidence-layer API: the strategy scorecard and the strategy x regime edge matrix.

The edge matrix is SLOW (one full walk-forward per actionable strategy over the
whole universe). A cold GET therefore never builds it — it returns an explicit
"not computed yet" payload so the page can render instantly and offer a button.

`?refresh=true` does NOT build it either. It ENQUEUES a build in a separate
process and returns immediately with a job id to poll. Running the build inside
the request — even on a worker thread — starved uvicorn's event loop badly
enough that TLS handshakes timed out and Cloudflare 502'd every endpoint in the
app, after which the Supervisor watchdog restarted the container and threw the
work away. See database/models.py::BackgroundJob. Do not put it back.

Results are read from the job table first and the in-memory cache second, so a
completed matrix now survives a restart.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from auth.deps import get_current_user
from database.db import get_db
from database.models import User
from services import edge_matrix as em
from services import jobs as jobsvc
from services.regime import evidence_adjustment
from services.scorecard import build_scorecard, execution_quality

EDGE_JOB_KIND = "edge_matrix"

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
    job_params = {"source": source, **kwargs}
    job_key = jobsvc.params_key(job_params)

    if refresh:
        # Start (or join) a background build and return AT ONCE. The response is
        # a job handle, never a matrix — building here is what took the add-on
        # down. `enqueue` reuses an in-flight job, so a double-click or a
        # double-firing page cannot run the build twice.
        job, created = jobsvc.enqueue(db, user.id, EDGE_JOB_KIND, job_params)
        return {
            "status": "building",
            "matrix": None,
            "job": jobsvc.to_dict(job),
            "started": created,
            "message": (
                "Build started — this runs one full walk-forward per strategy and takes "
                "minutes. Poll /api/jobs/{id}; the page can be closed and the build will "
                "carry on."
                if created else
                "A build with these settings is already running — following that one "
                "instead of starting a second."
            ),
        }

    # Cached read. The JOB TABLE is consulted first because it is the only copy
    # that survives a restart; `em._cache` is in-memory and empties every time
    # the container bounces.
    matrix = None
    status = "not_computed"
    outdated = False
    done = jobsvc.latest_done(db, user.id, EDGE_JOB_KIND, job_key) \
        or jobsvc.latest_done(db, user.id, EDGE_JOB_KIND)
    payload = jobsvc.result_of(done)
    if payload and isinstance(payload.get("matrix"), dict):
        if em.is_current(payload["matrix"]):
            matrix = payload["matrix"]
            status = "cached"
        else:
            # Built by an older measurement method (see em.METHOD_VERSION) —
            # never serve it as current evidence.
            outdated = True
    if matrix is None:
        matrix = em.get_cached_matrix(user.id, source=source, **kwargs) \
            or em.get_cached_matrix(user.id)
        status = "cached" if matrix else "not_computed"

    if matrix is None:
        # An in-flight build must not read as "nothing here" — that is what makes
        # a user hit refresh again and pile a second build onto a loaded box.
        active = jobsvc.find_active(db, user.id, EDGE_JOB_KIND, job_key)
        if active is not None:
            return {
                "status": "building",
                "matrix": None,
                "job": jobsvc.to_dict(active),
                "message": "A build is already running. Poll /api/jobs/{id} for progress.",
                "strategies": em.actionable_strategy_ids(),
                "quadrants": [{"id": q, "label": em.QUADRANT_INFO[q]["label"]} for q in em.QUADRANTS],
            }
        failed = jobsvc.latest_failed(db, user.id, EDGE_JOB_KIND)
        return {
            "status": "not_computed",
            "matrix": None,
            "outdated": outdated,
            "last_error": (failed.error if failed is not None else None),
            "message": (
                "The last edge matrix was built with an older method that scored periods spent "
                "entirely in cash (the SPY 200-day filter blocked every entry in bear regimes) "
                "as 0% results. It is not shown. Rebuild it to get corrected numbers."
                if outdated else
                "The edge matrix has not been computed yet. It runs a full walk-forward "
                "backtest for every actionable strategy, which takes minutes — call this "
                "endpoint again with ?refresh=true to start a background build."
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
