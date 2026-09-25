"""Out-of-process build worker: `python -m services.job_worker <job_id>`.

This module MUST have no import side effects beyond the app's own modules —
it is launched as a fresh interpreter by services/jobs.py::launch_worker, and
anything it runs at import time would run on every job.

It owns its own DB session and its own GIL. That second point is the entire
reason it exists: running these builds inside the web process starved uvicorn's
event loop badly enough that TLS handshakes timed out and Cloudflare 502'd the
whole add-on. See database/models.py::BackgroundJob for the full account.

Contract with the API side:
  * Every job ends in exactly one terminal status — "done" or "failed". A crash
    anywhere below is caught and recorded, because a job stuck at "running"
    blocks the feature until the reaper times it out.
  * `heartbeat_at` is written on every tick. The reaper uses it to distinguish
    "still working" from "killed by the watchdog".
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _run(job_id: int) -> int:
    # Imported inside the function so a missing/broken dependency is reported
    # against the job row rather than killing the interpreter before we have a
    # session to record it with.
    from database.db import SessionLocal
    from database.models import BackgroundJob

    db = SessionLocal()
    try:
        job = db.query(BackgroundJob).filter(BackgroundJob.id == job_id).first()
        if job is None:
            print(f"[job_worker] job {job_id} not found", file=sys.stderr)
            return 2
        if job.status not in ("queued", "running"):
            print(f"[job_worker] job {job_id} already {job.status}; nothing to do")
            return 0

        job.status = "running"
        job.started_at = _utcnow()
        job.heartbeat_at = _utcnow()
        job.pid = None
        try:
            import os
            job.pid = os.getpid()
        except Exception:  # noqa: BLE001
            pass
        job.progress_detail = "Starting…"
        db.commit()

        def tick(fraction: float, detail: str) -> None:
            """Progress callback handed to the builder. Commits immediately so
            the polling page sees movement while the build is still running."""
            try:
                job.progress = max(0.0, min(1.0, float(fraction)))
                job.progress_detail = str(detail)[:256]
                job.heartbeat_at = _utcnow()
                db.commit()
            except Exception:  # noqa: BLE001
                db.rollback()   # progress is best-effort; never fail the build over it

        params = {}
        try:
            params = json.loads(job.params_json or "{}")
        except (ValueError, TypeError):
            params = {}

        handler = HANDLERS.get(job.kind)
        if handler is None:
            raise ValueError(f"No handler registered for job kind {job.kind!r}")

        result = handler(db, job.user_id, params, tick)

        job.result_json = json.dumps(result, default=str)
        job.status = "done"
        job.progress = 1.0
        job.progress_detail = "Complete."
        job.heartbeat_at = _utcnow()
        job.finished_at = _utcnow()
        db.commit()
        print(f"[job_worker] job {job_id} ({job.kind}) done")
        return 0

    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        try:
            db.rollback()
            job = db.query(BackgroundJob).filter(BackgroundJob.id == job_id).first()
            if job is not None:
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"[:4000]
                job.finished_at = _utcnow()
                job.heartbeat_at = _utcnow()
                db.commit()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        return 1
    finally:
        db.close()


# ── Handlers ────────────────────────────────────────────────────────────────
# Each takes (db, user_id, params, tick) and returns a JSON-serialisable dict.

def _edge_matrix(db, user_id: int, params: dict, tick) -> dict:
    """Build the strategy x regime edge matrix.

    Progress is reported per strategy. `build_edge_matrix` does not currently
    accept a progress callback, so we pass one only if its signature has grown
    one — the builder is maintained separately and must not be forced to change
    in lockstep with this worker.
    """
    import inspect
    from services import edge_matrix as em

    source = params.get("source", "universe")
    kwargs = {k: v for k, v in params.items() if k not in ("source",)}

    tick(0.02, "Loading cached history for the universe…")

    sig = inspect.signature(em.build_edge_matrix)
    if "progress" in sig.parameters:
        matrix = em.build_edge_matrix(db, user_id, source=source, progress=tick, **kwargs)
    else:
        # No progress hook available: report the coarse phase honestly rather
        # than faking a percentage that does not correspond to anything.
        tick(0.05, "Running one walk-forward per strategy — this is the slow part.")
        matrix = em.build_edge_matrix(db, user_id, source=source, **kwargs)

    tick(0.99, "Saving results…")
    return {"matrix": matrix, "source": source}


def _selftest(db, user_id: int, params: dict, tick) -> dict:
    """Prove the whole pipeline works without waiting minutes for a real build.

    Exercises exactly the parts that can break independently of any strategy
    code: the subprocess launches, it opens its own DB session, progress ticks
    commit and are visible to the polling API, and a terminal status plus a
    result payload are written. Kept in the shipped code deliberately — after
    an outage caused by this layer, being able to verify it on the real box in
    a few seconds is worth the handful of lines.
    """
    import time
    steps = max(1, min(int(params.get("steps", 4)), 20))
    delay = max(0.0, min(float(params.get("delay", 0.25)), 2.0))
    for i in range(steps):
        tick(i / steps, f"self-test step {i + 1}/{steps}")
        time.sleep(delay)
    return {"ok": True, "steps": steps, "user_id": user_id}


def _daily_report(db, user_id: int, params: dict, tick) -> dict:
    """Generate one user's daily AI report out of the web process.

    Moved here after the 2026-09-24 05:31 outage: the 05:30 scheduled report
    ran inside uvicorn's event loop — per-ticker quote enrichment and 5-year
    history parsing run synchronously there, while an expired chart cache was
    refilled by yfinance threads competing for the same GIL — and the add-on
    stopped answering its health check for over a minute. The Supervisor
    watchdog then SIGKILLed it (exit 137).

    The LLM call can legitimately take up to its 900 s timeout, longer than
    jobs.STALE_AFTER (12 min), so a heartbeat coroutine ticks every 60 s while
    the report is generated — otherwise the reaper would fail a healthy job.
    The report uses its OWN session; `db` (and `tick`) belong to the job row.
    """
    import asyncio
    import logging

    if not logging.getLogger().handlers:
        # Fresh interpreter: without this the report's INFO/ERROR lines would be
        # dropped instead of landing in the add-on log next to the web app's.
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: [report worker] %(message)s")

    from database.db import SessionLocal
    from database.models import User
    from services.report_service import generate_daily_report

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise ValueError(f"user {user_id} not found")
    # The key and prompt are read from the user row HERE rather than passed
    # in params, so a LiteLLM key is never copied into background_jobs.params_json.
    api_key = user.litellm_api_key
    system_prompt = user.report_system_prompt

    async def _go():
        report_db = SessionLocal()

        async def heartbeat():
            while True:
                await asyncio.sleep(60)
                tick(0.5, "Waiting for the AI model to finish the report…")

        beat = asyncio.create_task(heartbeat())
        try:
            tick(0.1, "Gathering market context…")
            return await generate_daily_report(
                report_db, user_id,
                triggered_by=str(params.get("triggered_by") or "schedule"),
                model=params.get("model") or None,
                api_key=api_key,
                system_prompt=system_prompt,
            )
        finally:
            beat.cancel()
            report_db.close()

    result = asyncio.run(_go())
    tick(0.99, "Saving report…")
    return {"markdown": result.get("markdown"), "model": result.get("model")}


def _history_backfill(db, user_id: int, params: dict, tick) -> dict:
    """Download 20 years of daily history into the archive (yfinance, with a
    Tiingo fallback when TIINGO_API_KEY is set). Resumable: finished tickers
    are skipped, so re-running continues where the last run stopped. Progress
    ticks per ticker, well inside jobs.STALE_AFTER."""
    import logging
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: [history worker] %(message)s")
    from config import get_settings
    from services.long_history import run_backfill
    tick(0.01, "Starting the 20-year download…")
    return run_backfill(db, tiingo_key=(get_settings().tiingo_api_key or "").strip() or None,
                        force=bool(params.get("force")), progress=tick)



HANDLERS = {
    "edge_matrix": _edge_matrix,
    "selftest": _selftest,
    "daily_report": _daily_report,
    "history_backfill": _history_backfill,
}


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python -m services.job_worker <job_id>", file=sys.stderr)
        return 64
    try:
        job_id = int(argv[1])
    except ValueError:
        print(f"invalid job id: {argv[1]!r}", file=sys.stderr)
        return 64
    return _run(job_id)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
