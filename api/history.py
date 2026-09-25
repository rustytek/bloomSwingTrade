"""
/api/history — the 20-year price archive behind long backtests and the edge
matrix (services/long_history.py).

The backfill downloads ~540 tickers x 20 years; it ALWAYS runs in the job
worker, never in a request. The archive is shared by every user, so starting
it is admin-only; anyone can read the coverage status.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from auth.deps import get_current_admin, get_current_user
from config import get_settings
from database.db import get_db
from database.models import BackgroundJob, User
from services import jobs as jobsvc
from services import long_history

BACKFILL_KIND = "history_backfill"

router = APIRouter(prefix="/api/history", tags=["history"])


def _latest_backfill(db: Session) -> BackgroundJob | None:
    return (db.query(BackgroundJob)
            .filter(BackgroundJob.kind == BACKFILL_KIND)
            .order_by(BackgroundJob.created_at.desc())
            .first())


@router.get("/status")
def history_status(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    jobsvc.reap_stale(db)
    job = _latest_backfill(db)
    last = jobsvc.result_of(job) if job is not None and job.status == "done" else None
    return {
        **long_history.status(db),
        "tiingo_configured": bool((get_settings().tiingo_api_key or "").strip()),
        "job": jobsvc.to_dict(job),
        "last_result": last,
        "can_start": bool(user.is_admin),
    }


@router.post("/backfill")
def start_backfill(
    force: bool = Query(False, description="Re-download tickers that are already complete"),
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    """Start (or join) the 20-year download in the background. Returns at once."""
    running = (db.query(BackgroundJob)
               .filter(BackgroundJob.kind == BACKFILL_KIND,
                       BackgroundJob.status.in_(jobsvc.ACTIVE_STATUSES))
               .first())
    if running is not None:
        # One download at a time across ALL users — two would fight over the
        # same rows and double the load on Yahoo/Tiingo.
        return {"status": "running", "started": False, "job": jobsvc.to_dict(running)}
    job, created = jobsvc.enqueue(db, admin.id, BACKFILL_KIND, {"force": bool(force)})
    return {"status": "running", "started": created, "job": jobsvc.to_dict(job)}
