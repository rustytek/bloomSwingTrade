"""Background-job tests — the layer that exists because of the 2026-09-23 outage.

These are NOT unit tests of a pure function: the whole point of this layer is
that it launches a real separate process, so the end-to-end test launches one.
A test that mocked the subprocess would pass while the thing it guards is
broken, which is the failure mode that caused the outage in the first place.

Self-contained: runs against a throwaway SQLite file in a temp directory, no
network, and never touches data/swingtrader.db.

    python test_jobs.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

# Point the app at a throwaway DB BEFORE anything imports config/database.
_TMP = tempfile.mkdtemp(prefix="swingtrader-jobs-")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_TMP, "t.db").replace("\\", "/")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_TESTS = []


def test(fn):
    _TESTS.append(fn)
    return fn


from database.db import Base, SessionLocal, engine          # noqa: E402
from database.models import BackgroundJob, User             # noqa: E402
from services import jobs as jobsvc                         # noqa: E402

Base.metadata.create_all(bind=engine)

_USER_ID = 1


def _fresh_db():
    db = SessionLocal()
    db.query(BackgroundJob).delete()
    if not db.query(User).filter(User.id == _USER_ID).first():
        db.add(User(id=_USER_ID, username="jobtest", password_hash="x"))
    db.commit()
    return db


def _wait(db, job_id, want=("done", "failed", "cancelled"), timeout=90):
    """Poll like the page does. Expires the session each time so we read what
    the WORKER PROCESS committed, not this session's stale identity map."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        db.expire_all()
        job = db.query(BackgroundJob).filter(BackgroundJob.id == job_id).first()
        last = job
        if job is not None and job.status in want:
            return job
        time.sleep(0.25)
    return last


# ── The end-to-end path ─────────────────────────────────────────────────────
@test
def test_worker_subprocess_runs_and_reports_progress():
    """A real subprocess launches, ticks progress, and writes a result."""
    db = _fresh_db()
    try:
        job, created = jobsvc.enqueue(
            db, _USER_ID, "selftest", {"steps": 4, "delay": 0.2})
        assert created is True
        assert job.status in ("queued", "running"), job.status

        done = _wait(db, job.id)
        assert done is not None, "job row vanished"
        assert done.status == "done", (
            "worker did not finish cleanly: status=%s error=%s"
            % (done.status, done.error))
        assert done.progress == 1.0, done.progress
        assert done.finished_at is not None
        assert done.pid, "worker never recorded its pid — did it actually run?"

        payload = jobsvc.result_of(done)
        assert payload == {"ok": True, "steps": 4, "user_id": _USER_ID}, payload
    finally:
        db.close()


@test
def test_progress_is_visible_while_the_job_is_still_running():
    """The page polls for movement; a build that reports nothing until it is
    finished is indistinguishable from one that has hung."""
    db = _fresh_db()
    try:
        job, _ = jobsvc.enqueue(db, _USER_ID, "selftest", {"steps": 8, "delay": 0.4})
        seen_running = False
        seen_detail = None
        deadline = time.time() + 40
        while time.time() < deadline:
            db.expire_all()
            row = db.query(BackgroundJob).filter(BackgroundJob.id == job.id).first()
            if row.status == "running" and row.heartbeat_at is not None:
                seen_running = True
                if row.progress_detail and "step" in row.progress_detail:
                    seen_detail = row.progress_detail
                    break
            if row.status in ("done", "failed"):
                break
            time.sleep(0.15)
        assert seen_running, "never observed the job in 'running' — no heartbeat"
        assert seen_detail, "progress_detail never named the step being worked on"
        _wait(db, job.id)
    finally:
        db.close()


# ── Dedupe: the specific bug in the outage log ──────────────────────────────
@test
def test_identical_request_joins_the_running_job_instead_of_starting_a_second():
    """The cloudflared log shows ?refresh=true arriving TWICE concurrently, so
    the build ran twice and doubled the CPU load that starved the event loop."""
    db = _fresh_db()
    try:
        params = {"source": "universe", "period": "all"}
        first, created_a = jobsvc.enqueue(db, _USER_ID, "selftest",
                                          {**params, "steps": 6, "delay": 0.3})
        second, created_b = jobsvc.enqueue(db, _USER_ID, "selftest",
                                           {**params, "steps": 6, "delay": 0.3})
        assert created_a is True and created_b is False, (created_a, created_b)
        assert first.id == second.id, (first.id, second.id)

        total = db.query(BackgroundJob).filter(
            BackgroundJob.kind == "selftest").count()
        assert total == 1, f"{total} job rows — the build would run twice"
        _wait(db, first.id)
    finally:
        db.close()


@test
def test_different_params_are_different_jobs():
    """Dedupe must key on the PARAMETERS, or a universe build would swallow a
    watchlist build and silently return the wrong matrix."""
    db = _fresh_db()
    try:
        a, ca = jobsvc.enqueue(db, _USER_ID, "selftest",
                               {"source": "universe", "steps": 1, "delay": 0}, launch=False)
        b, cb = jobsvc.enqueue(db, _USER_ID, "selftest",
                               {"source": "watchlist", "steps": 1, "delay": 0}, launch=False)
        assert ca and cb
        assert a.id != b.id
    finally:
        db.close()


@test
def test_params_key_is_order_independent():
    """Key order must not create a duplicate job."""
    assert jobsvc.params_key({"a": 1, "b": 2}) == jobsvc.params_key({"b": 2, "a": 1})
    assert jobsvc.params_key({"a": 1}) != jobsvc.params_key({"a": 2})


# ── Recovery: a dead worker must never block the feature ────────────────────
@test
def test_stale_running_job_is_reaped_so_it_cannot_block_forever():
    """A worker killed by the watchdog leaves a row at 'running'. If that row
    stayed active, `enqueue` would join it forever and the user could never
    rebuild — turning a transient restart into a permanent outage."""
    from datetime import timedelta
    db = _fresh_db()
    try:
        stale = BackgroundJob(
            user_id=_USER_ID, kind="edge_matrix", params_key="{}", params_json="{}",
            status="running", progress=0.3,
            created_at=jobsvc._utcnow() - timedelta(hours=2),
            started_at=jobsvc._utcnow() - timedelta(hours=2),
            heartbeat_at=jobsvc._utcnow() - jobsvc.STALE_AFTER - timedelta(minutes=5),
        )
        db.add(stale)
        db.commit()
        db.refresh(stale)

        reaped = jobsvc.reap_stale(db)
        assert reaped == 1, reaped
        db.expire_all()
        row = db.query(BackgroundJob).filter(BackgroundJob.id == stale.id).first()
        assert row.status == "failed", row.status
        assert "presumed dead" in (row.error or "")

        # ...and a new build can now be started.
        fresh, created = jobsvc.enqueue(db, _USER_ID, "edge_matrix", {}, launch=False)
        assert created is True and fresh.id != stale.id
    finally:
        db.close()


@test
def test_a_recent_heartbeat_is_never_reaped():
    """Negative control: reaping a job that is merely slow would be worse than
    the bug this reaper fixes."""
    db = _fresh_db()
    try:
        live = BackgroundJob(
            user_id=_USER_ID, kind="edge_matrix", params_key="{}", params_json="{}",
            status="running", progress=0.5,
            created_at=jobsvc._utcnow(), started_at=jobsvc._utcnow(),
            heartbeat_at=jobsvc._utcnow(),
        )
        db.add(live)
        db.commit()
        db.refresh(live)
        assert jobsvc.reap_stale(db) == 0
        db.expire_all()
        assert db.query(BackgroundJob).filter(
            BackgroundJob.id == live.id).first().status == "running"
    finally:
        db.close()


@test
def test_naive_timestamps_do_not_break_the_reaper():
    """SQLite returns naive datetimes. Comparing them to an aware `now` raises
    TypeError, which would make the reaper silently never run."""
    from datetime import datetime, timedelta
    db = _fresh_db()
    try:
        job = BackgroundJob(
            user_id=_USER_ID, kind="edge_matrix", params_key="{}", params_json="{}",
            status="running",
            created_at=datetime.utcnow() - timedelta(hours=3),      # naive, on purpose
            started_at=datetime.utcnow() - timedelta(hours=3),
            heartbeat_at=datetime.utcnow() - timedelta(hours=3),
        )
        db.add(job)
        db.commit()
        assert jobsvc.reap_stale(db) == 1   # would raise TypeError without _aware()
    finally:
        db.close()


@test
def test_startup_resets_jobs_orphaned_by_a_restart():
    """Workers die with the container. Anything still 'running' at startup is
    dead, and must not block the next build for the 12-minute stale window."""
    db = _fresh_db()
    try:
        job = BackgroundJob(
            user_id=_USER_ID, kind="edge_matrix", params_key="{}", params_json="{}",
            status="running", created_at=jobsvc._utcnow(),
            started_at=jobsvc._utcnow(), heartbeat_at=jobsvc._utcnow(),
        )
        db.add(job)
        db.commit()
        job_id = job.id
        db.close()

        import main
        main.orphan_background_jobs()

        db2 = SessionLocal()
        row = db2.query(BackgroundJob).filter(BackgroundJob.id == job_id).first()
        assert row.status == "failed", row.status
        assert "restart" in (row.error or "").lower()
        db2.close()
    finally:
        pass


# ── Failure handling ────────────────────────────────────────────────────────
@test
def test_unknown_job_kind_fails_the_job_rather_than_hanging():
    """Every job must reach a terminal status. One stuck at 'running' blocks
    the feature until the reaper times it out."""
    db = _fresh_db()
    try:
        job, _ = jobsvc.enqueue(db, _USER_ID, "no_such_kind_at_all", {})
        row = _wait(db, job.id, want=("failed",), timeout=60)
        assert row is not None and row.status == "failed", (
            row.status if row else None)
        assert "no handler" in (row.error or "").lower(), row.error
    finally:
        db.close()


@test
def test_to_dict_never_leaks_the_result_payload():
    """A poll happens every couple of seconds; the matrix is megabytes."""
    db = _fresh_db()
    try:
        job = BackgroundJob(
            user_id=_USER_ID, kind="edge_matrix", params_key="{}", params_json="{}",
            status="done", progress=1.0, result_json='{"matrix": {"huge": true}}',
            created_at=jobsvc._utcnow(), finished_at=jobsvc._utcnow(),
        )
        db.add(job)
        db.commit()
        d = jobsvc.to_dict(job)
        assert "result" not in d and "result_json" not in d, d
        assert set(d) >= {"id", "kind", "status", "progress", "progress_detail"}
        # ...but it is retrievable on purpose.
        assert jobsvc.result_of(job) == {"matrix": {"huge": True}}
    finally:
        db.close()


@test
def test_result_of_is_none_unless_the_job_actually_succeeded():
    """A failed job with a half-written payload must never look like a result."""
    job = BackgroundJob(user_id=_USER_ID, kind="edge_matrix", params_key="{}",
                        status="failed", result_json='{"matrix": {}}')
    assert jobsvc.result_of(job) is None
    job.status = "done"
    job.result_json = "{not json"
    assert jobsvc.result_of(job) is None


# ── Wiring guards ───────────────────────────────────────────────────────────
@test
def test_edge_matrix_endpoint_no_longer_builds_inline():
    """The regression that caused the outage. `?refresh=true` must ENQUEUE."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "api", "scorecard.py"), encoding="utf-8").read()
    assert "jobsvc.enqueue" in src, "refresh no longer enqueues a job"
    assert "anyio.to_thread" not in src, (
        "api/scorecard.py runs the build in-process again — this starves the "
        "event loop and 502s the whole add-on")
    # The build call itself must not appear in the request path at all.
    assert "em.build_edge_matrix" not in src, (
        "api/scorecard.py calls build_edge_matrix directly; it belongs in "
        "services/job_worker.py")


@test
def test_build_edge_matrix_accepts_a_progress_callback():
    """Without it the worker cannot heartbeat, and a long build gets reaped
    as dead while it is still working."""
    import inspect
    from services import edge_matrix as em
    assert "progress" in inspect.signature(em.build_edge_matrix).parameters


@test
def test_progress_callback_failure_cannot_sink_a_build():
    """Progress is best-effort. A DB hiccup while ticking must not lose minutes
    of completed work."""
    import inspect
    from services import edge_matrix as em
    src = inspect.getsource(em.build_edge_matrix)
    assert "except Exception" in src and "_tick" in src, (
        "build_edge_matrix does not guard its progress callback")


@test
def test_worker_module_has_no_import_side_effects():
    """It is spawned as a fresh interpreter for every job; anything at import
    time runs every time. It must also never import main.py, which would try to
    start a second copy of the web app."""
    import ast
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "services", "job_worker.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        assert isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                                 ast.ClassDef, ast.Assign, ast.AnnAssign,
                                 ast.Expr, ast.If)), (
            f"top-level {type(node).__name__} in job_worker.py runs on every job")
    src = open(path, encoding="utf-8").read()
    assert "import main" not in src and "from main" not in src


def main_runner() -> int:
    passed = failed = 0
    failures = []
    for fn in _TESTS:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            failures.append((fn.__name__, exc))
            print("FAIL  " + fn.__name__ + "  --  " + repr(exc))
        else:
            passed += 1
            print("PASS  " + fn.__name__)
    print("\n" + "=" * 56)
    print(f"  {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 56)
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        code = main_runner()
    finally:
        engine.dispose()
        shutil.rmtree(_TMP, ignore_errors=True)
    sys.exit(code)
