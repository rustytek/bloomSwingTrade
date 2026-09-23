"""Background job registry — the API side of the out-of-process build.

See database/models.py::BackgroundJob for WHY this exists (the 2026-09-23
event-loop-starvation outage). This module is what the HTTP handlers touch:
it never runs a build itself, it only records intent and launches a worker.

Design rules, each one a direct response to how the outage actually unfolded:

  * **One job per (user, kind, params).** The cloudflared log shows two
    concurrent `?refresh=true` requests, so the build ran twice and doubled the
    CPU load. `enqueue()` returns the EXISTING job instead of starting a second.
  * **The worker is a separate PROCESS,** not a thread. A thread would still
    contend for the GIL, which is precisely what starved the event loop.
  * **Spawned detached, never awaited.** The HTTP handler returns immediately;
    it must never hold a socket open for the length of a build.
  * **Stale jobs are reaped.** A worker killed mid-build (watchdog restart) would
    otherwise leave a row stuck at "running" that blocks every future refresh.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from database.models import BackgroundJob

# A worker writes heartbeat_at on every progress tick. If one has been silent
# for longer than this it is presumed dead. Generous, because a single strategy's
# walk-forward over the full universe can legitimately run for minutes without a
# tick — reaping a job that is merely slow would be worse than the bug.
STALE_AFTER = timedelta(minutes=12)

# Jobs are pruned rather than kept forever; the result of a superseded build is
# noise. Keeps the table from growing without bound on a daily-refresh habit.
KEEP_FINISHED = timedelta(days=3)

ACTIVE_STATUSES = ("queued", "running")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(dt: datetime | None) -> datetime | None:
    """SQLite hands back naive datetimes; compare them as UTC.

    Without this the staleness check raises TypeError (naive vs aware) and the
    reaper silently never runs — which is exactly the failure mode it exists to
    prevent, so it must not depend on how the row was written.
    """
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def params_key(params: dict | None) -> str:
    """Canonical key for a parameter set. Sorted so key order cannot create a
    duplicate job, and truncated to the column width."""
    return json.dumps(params or {}, sort_keys=True, separators=(",", ":"))[:512]


def reap_stale(db: Session) -> int:
    """Fail any job whose worker has gone silent, and prune old finished rows.

    Returns the number of jobs reaped. Called before every enqueue/poll so a
    dead job can never permanently block the feature it belongs to.
    """
    now = _utcnow()
    reaped = 0
    for job in db.query(BackgroundJob).filter(
            BackgroundJob.status.in_(ACTIVE_STATUSES)).all():
        last = _aware(job.heartbeat_at) or _aware(job.started_at) or _aware(job.created_at)
        if last is None or (now - last) <= STALE_AFTER:
            continue
        job.status = "failed"
        job.finished_at = now
        job.error = (
            "The worker process stopped reporting and is presumed dead — usually a "
            "container restart (the Home Assistant watchdog) or the host running out "
            "of memory mid-build. Nothing was corrupted; start the build again."
        )
        reaped += 1

    cutoff = now - KEEP_FINISHED
    db.query(BackgroundJob).filter(
        BackgroundJob.status.notin_(ACTIVE_STATUSES),
        BackgroundJob.finished_at.isnot(None),
        BackgroundJob.finished_at < cutoff,
    ).delete(synchronize_session=False)

    if reaped:
        db.commit()
    else:
        db.commit()
    return reaped


def find_active(db: Session, user_id: int, kind: str, key: str) -> BackgroundJob | None:
    return (db.query(BackgroundJob)
            .filter(BackgroundJob.user_id == user_id,
                    BackgroundJob.kind == kind,
                    BackgroundJob.params_key == key,
                    BackgroundJob.status.in_(ACTIVE_STATUSES))
            .order_by(BackgroundJob.created_at.desc())
            .first())


def latest_done(db: Session, user_id: int, kind: str, key: str | None = None) -> BackgroundJob | None:
    """Most recent SUCCESSFUL job, so a result survives a restart.

    `edge_matrix._cache` is in-memory only: before this existed, a completed
    matrix was lost on every container restart and the page fell back to "not
    computed yet" with no way for the user to tell the difference.
    """
    q = (db.query(BackgroundJob)
         .filter(BackgroundJob.user_id == user_id,
                 BackgroundJob.kind == kind,
                 BackgroundJob.status == "done"))
    if key is not None:
        q = q.filter(BackgroundJob.params_key == key)
    return q.order_by(BackgroundJob.finished_at.desc()).first()


def latest_failed(db: Session, user_id: int, kind: str) -> BackgroundJob | None:
    """Most recent failure, so a page can say WHY the last attempt produced
    nothing instead of showing an indistinguishable "not computed yet"."""
    return (db.query(BackgroundJob)
            .filter(BackgroundJob.user_id == user_id,
                    BackgroundJob.kind == kind,
                    BackgroundJob.status == "failed")
            .order_by(BackgroundJob.finished_at.desc())
            .first())


def enqueue(db: Session, user_id: int, kind: str, params: dict | None = None,
            *, launch: bool = True) -> tuple[BackgroundJob, bool]:
    """Return (job, created). An identical in-flight job is REUSED, not duplicated."""
    reap_stale(db)
    key = params_key(params)

    existing = find_active(db, user_id, kind, key)
    if existing is not None:
        return existing, False

    job = BackgroundJob(
        user_id=user_id,
        kind=kind,
        params_key=key,
        params_json=json.dumps(params or {}, sort_keys=True),
        status="queued",
        progress=0.0,
        progress_detail="Queued — waiting for the worker to start.",
        created_at=_utcnow(),
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    if launch:
        try:
            launch_worker(job.id)
        except Exception as exc:  # noqa: BLE001
            # A job that cannot be launched must fail LOUDLY and immediately,
            # not sit at "queued" until the reaper notices 12 minutes later.
            job.status = "failed"
            job.error = f"Could not start the worker process: {exc}"
            job.finished_at = _utcnow()
            db.commit()
    return job, True


def launch_worker(job_id: int) -> int:
    """Start `python -m services.job_worker <job_id>` detached, and return its pid.

    Deliberately a fresh interpreter rather than a fork: forking a process that
    is already running an asyncio loop plus several background threads risks
    inheriting held locks, and `spawn` would re-import main.py and try to start a
    second copy of the web app. `-m` on a module with no import side effects is
    the only option with neither problem.
    """
    cmd = [sys.executable, "-m", "services.job_worker", str(job_id)]
    kwargs: dict = {
        "cwd": os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "stdin": subprocess.DEVNULL,
        # Inherit stdout/stderr so worker tracebacks land in the add-on log,
        # where the user can actually see them.
        "env": {**os.environ, "PYTHONUNBUFFERED": "1"},
    }
    if os.name == "posix":
        # Detach from the parent's process group so a uvicorn reload or a
        # SIGINT to the web process does not take the build down with it.
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(cmd, **kwargs)  # noqa: S603
    return proc.pid


def to_dict(job: BackgroundJob | None) -> dict | None:
    """JSON-safe view. `result` is deliberately EXCLUDED — it can be megabytes,
    and a poll must stay cheap. Callers fetch the result explicitly."""
    if job is None:
        return None
    return {
        "id": job.id,
        "kind": job.kind,
        "status": job.status,
        "progress": round(float(job.progress or 0.0), 4),
        "progress_detail": job.progress_detail,
        "error": job.error,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
    }


def result_of(job: BackgroundJob | None) -> dict | None:
    if job is None or job.status != "done" or not job.result_json:
        return None
    try:
        return json.loads(job.result_json)
    except (ValueError, TypeError):
        return None
