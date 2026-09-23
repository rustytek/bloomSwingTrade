"""Job status endpoints — the polling half of the out-of-process build.

These handlers are deliberately trivial: a poll happens every couple of seconds
while a build runs, so it must be a single indexed row read and nothing more.
The result payload is NOT included (it can be megabytes) — the feature endpoint
that owns the job serves it once the job reports "done".
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from auth.deps import get_current_user
from database.db import get_db
from database.models import BackgroundJob, User
from services import jobs as jobsvc

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("")
def list_jobs(
    kind: str | None = None,
    limit: int = 20,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """This user's recent jobs, newest first. Reaps stale rows on the way past,
    so a poll is also what unsticks a job whose worker was killed."""
    jobsvc.reap_stale(db)
    q = db.query(BackgroundJob).filter(BackgroundJob.user_id == user.id)
    if kind:
        q = q.filter(BackgroundJob.kind == kind)
    rows = q.order_by(BackgroundJob.created_at.desc()).limit(max(1, min(limit, 100))).all()
    return {"jobs": [jobsvc.to_dict(j) for j in rows]}


@router.get("/{job_id}")
def get_job(
    job_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    jobsvc.reap_stale(db)
    job = (db.query(BackgroundJob)
           .filter(BackgroundJob.id == job_id,
                   BackgroundJob.user_id == user.id)   # never leak another user's job
           .first())
    if job is None:
        raise HTTPException(status_code=404, detail="No such job.")
    return jobsvc.to_dict(job)


@router.post("/{job_id}/cancel")
def cancel_job(
    job_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Mark a job cancelled so it stops blocking new ones.

    HONESTY: this does not kill the worker process — it is detached by design,
    and the build is CPU-bound with no safe interruption point. The row stops
    being 'active' immediately (so a fresh build can be started), and the worker
    finishes into a row that is no longer wanted. Said plainly in the response
    rather than implying a hard stop.
    """
    job = (db.query(BackgroundJob)
           .filter(BackgroundJob.id == job_id, BackgroundJob.user_id == user.id)
           .first())
    if job is None:
        raise HTTPException(status_code=404, detail="No such job.")
    if job.status not in jobsvc.ACTIVE_STATUSES:
        return {"ok": True, "status": job.status, "note": "Job had already finished."}
    job.status = "cancelled"
    job.finished_at = jobsvc._utcnow()
    job.error = "Cancelled by the user."
    db.commit()
    return {
        "ok": True,
        "status": "cancelled",
        "note": ("Marked cancelled — you can start a new build now. The worker process "
                 "is not force-killed; it will run to completion in the background and "
                 "its result will simply be discarded."),
    }
