"""
20-year history archive tests — no network.

Covers services/history_sources.py (cleaning, assessment, source choice,
Tiingo parsing via httpx.MockTransport), services/long_history.py (splice and
rebase across a split, backfill: yfinance first, Tiingo fallback, rate-limit
stop, resumability, never shrinking a stored series, VIX refresh, status),
the 20-year path through the walk-forward engine (archive actually used,
real-VIX regime tagging), and the API surface (20y runs are queued, never run
in the request; the backfill is admin-only).

    python test_history.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import traceback
from datetime import date, timedelta

_TMP = tempfile.mkdtemp(prefix="swingtrader-history-")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_TMP, "t.db").replace("\\", "/")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import httpx  # noqa: E402

from database.db import Base, SessionLocal, engine  # noqa: E402
from database.models import HistoryArchive, StockCache, User  # noqa: E402
from services import history_sources as src  # noqa: E402
from services import long_history as lh  # noqa: E402

Base.metadata.create_all(bind=engine)
src.pause = lambda s: None          # no real sleeping in tests

_TESTS = []


def test(fn):
    _TESTS.append(fn)
    return fn


# ── synthetic data ─────────────────────────────────────────────────────────
def trading_days(start: str, n: int) -> list[str]:
    d, out = date.fromisoformat(start), []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def make_bars(days: list[str], start_px=100.0, drift=0.0004, wiggle=0.01, scale=1.0, vol=1_000_000):
    out, px = [], start_px
    for i, d in enumerate(days):
        px *= 1 + drift + wiggle * math.sin(i / 3.0)
        c = round(px * scale, 4)
        out.append({"i": i, "date": d, "open": c, "high": round(c * 1.01, 4),
                    "low": round(c * 0.99, 4), "close": c, "vol": vol})
    return out


def put_cache(db, ticker, bars):
    row = db.query(StockCache).filter(StockCache.ticker == ticker).first()
    if row is None:
        row = StockCache(ticker=ticker)
        db.add(row)
    row.history_json = json.dumps(bars)
    db.commit()


def put_archive(db, ticker, bars, requested_start=None):
    row = db.query(HistoryArchive).filter(HistoryArchive.ticker == ticker).first()
    if row is None:
        row = HistoryArchive(ticker=ticker, bars_json="[]", start_date="", end_date="")
        db.add(row)
    row.bars_json = json.dumps(bars)
    row.start_date, row.end_date = bars[0]["date"], bars[-1]["date"]
    row.requested_start = requested_start
    db.commit()


def wipe(db):
    db.query(HistoryArchive).delete()
    db.query(StockCache).delete()
    db.commit()


# ── history_sources ────────────────────────────────────────────────────────
@test
def test_clean_bars_sorts_dedupes_and_drops_bad_closes():
    bars, dropped = src.clean_bars([
        {"date": "2024-01-03", "close": 2, "open": 2, "high": 2, "low": 2, "vol": 5},
        {"date": "2024-01-02", "close": 1},
        {"date": "2024-01-03", "close": 3},                 # duplicate: last wins
        {"date": "2024-01-04", "close": 0},                 # not a price
        {"date": "2024-01-05", "close": float("nan")},
    ])
    assert [b["date"] for b in bars] == ["2024-01-02", "2024-01-03"] and dropped == 2
    assert bars[1]["close"] == 3 and bars[0]["high"] == 1, "missing OHLC falls back to close, nothing invented beyond that"


@test
def test_assess_flags_truncated_stale_and_holed():
    days = trading_days("2024-01-01", 300)
    good = make_bars(days)
    today = days[-1]
    assert src.assess(good, target_start="2006-01-01", today=today, calendar=days)["ok"]
    trunc = src.assess(good[100:], target_start="2006-01-01", today=today, known_first=days[0])
    assert not trunc["ok"] and "truncated" in trunc["issues"][0]
    stale = src.assess(good[:200], target_start="2006-01-01", today=today)
    assert not stale["ok"] and "old" in stale["issues"][0]
    holed = [b for i, b in enumerate(good) if i % 10]
    assert not src.assess(holed, target_start="2006-01-01", today=today, calendar=days)["ok"]
    assert not src.assess([], target_start="2006-01-01", today=today)["ok"]


@test
def test_better_prefers_usable_then_earlier_then_fewer_gaps():
    ok_late = ([{"date": "2010-01-01"}], {"ok": True, "first": "2010-01-01", "missing": 0})
    ok_early = ([{"date": "2006-01-01"}], {"ok": True, "first": "2006-01-01", "missing": 3})
    bad_early = ([{"date": "2000-01-01"}], {"ok": False, "first": "2000-01-01", "missing": 0})
    assert src.better(ok_late, bad_early) is ok_late
    assert src.better(ok_late, ok_early) is ok_early
    assert src.better(ok_early, ok_early) is ok_early


@test
def test_tiingo_uses_adjusted_fields_and_reports_rate_limit():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        if "brk-b" in str(request.url):
            return httpx.Response(200, json=[
                {"date": "2024-01-02T00:00:00.000Z", "close": 400, "adjClose": 200, "adjOpen": 199,
                 "adjHigh": 201, "adjLow": 198, "adjVolume": 10}])
        if "limited" in str(request.url):
            return httpx.Response(429, json={"detail": "rate"})
        return httpx.Response(404, json={"detail": "not found"})

    tr = httpx.MockTransport(handler)
    bars = src.fetch_tiingo("BRK.B", "2024-01-01", "2024-01-05", "KEY", transport=tr)
    assert bars and bars[0]["close"] == 200 and bars[0]["open"] == 199, "must use adjusted prices"
    assert "brk-b" in seen["url"] and seen["auth"] == "Token KEY"
    assert src.fetch_tiingo("NOPE", "2024-01-01", "2024-01-05", "KEY", transport=tr) == []
    assert src.fetch_tiingo("^VIX", "2024-01-01", "2024-01-05", "KEY", transport=tr) == [], "no indices on Tiingo"
    try:
        src.fetch_tiingo("LIMITED", "2024-01-01", "2024-01-05", "KEY", transport=tr)
        raise AssertionError("429 must raise")
    except src.TiingoRateLimited:
        pass


# ── splice ─────────────────────────────────────────────────────────────────
@test
def test_splice_rebases_across_a_split_without_a_fake_jump():
    days = trading_days("2015-01-01", 800)
    archive = make_bars(days[:700], scale=2.0)          # downloaded BEFORE a 2:1 split
    cache = make_bars(days, scale=1.0)[300:]             # cache: split-adjusted, starts later
    bars, meta = lh.splice(archive, cache)
    assert meta["used_archive_bars"] == 300 and abs(meta["ratio"] - 0.5) < 1e-6, meta
    assert [b["date"] for b in bars] == days
    assert abs(bars[0]["vol"] - 2_000_000) < 100, "split-like ratio adjusts volume too"
    worst = max(abs(bars[i]["close"] / bars[i - 1]["close"] - 1) for i in range(1, len(bars)))
    assert worst < 0.03, f"splice created a jump of {worst:.1%}"


@test
def test_splice_refuses_sources_that_disagree():
    days = trading_days("2015-01-01", 400)
    archive = make_bars(days[:300])
    cache = [dict(b) for b in make_bars(days)[200:]]
    for i, b in enumerate(cache[:10]):                    # inconsistent overlap
        b["close"] = round(b["close"] * (1 + 0.05 * (i % 2)), 4)
    bars, meta = lh.splice(archive, cache)
    assert bars is cache and "disagree" in meta["reason"]
    assert lh.splice([], cache)[0] is cache
    assert lh.splice(archive, [])[0] is archive
    no_overlap, meta = lh.splice(make_bars(days[:100]), make_bars(days)[200:])
    assert meta["used_archive_bars"] == 0 and "overlap" in meta["reason"]


# ── backfill ───────────────────────────────────────────────────────────────
class FakeSources:
    def __init__(self, yf: dict, tiingo: dict | None = None, rate_limit_after: int | None = None):
        self.yf, self.tiingo = yf, tiingo or {}
        self.calls = {"yf": [], "tiingo": []}
        self.rate_limit_after = rate_limit_after

    def fetch_yf(self, t, start, end):
        self.calls["yf"].append((t, start))
        return [dict(b) for b in self.yf.get(t, [])]

    def fetch_tiingo(self, t, start, end, key, transport=None):
        self.calls["tiingo"].append(t)
        if self.rate_limit_after is not None and len(self.calls["tiingo"]) > self.rate_limit_after:
            raise src.TiingoRateLimited("rate limited")
        return [dict(b) for b in self.tiingo.get(t, [])]


def install(fake):
    src.fetch_yfinance = fake.fetch_yf
    src.fetch_tiingo = fake.fetch_tiingo


TODAY = date(2026, 9, 24)
LONG_DAYS = trading_days(lh.target_start(TODAY), 5230)
LONG_DAYS = [d for d in LONG_DAYS if d <= TODAY.isoformat()]


@test
def test_backfill_yfinance_first_tiingo_for_truncated_and_resumable():
    db = SessionLocal()
    wipe(db)
    full = make_bars(LONG_DAYS)
    put_cache(db, "AAA", full[-1250:])
    put_cache(db, "BBB", full[-1250:])
    fake = FakeSources(
        yf={"SPY": full, "^VIX": make_bars(LONG_DAYS, start_px=20, drift=0), "AAA": full,
            "BBB": full[-600:]},                       # Yahoo truncated BBB
        tiingo={"BBB": full},
    )
    install(fake)
    r = lh.run_backfill(db, tiingo_key="K", today=TODAY, tickers=["SPY", "^VIX", "AAA", "BBB"])
    assert r["counts"]["yfinance"] == 3 and r["counts"]["tiingo"] == 1, r
    row = db.query(HistoryArchive).filter(HistoryArchive.ticker == "BBB").one()
    assert row.source == "tiingo" and row.start_date == full[0]["date"]
    assert fake.calls["tiingo"] == ["BBB"], "Tiingo only for the ticker Yahoo got wrong"
    st = lh.status(db)
    assert st["spy_start"] == full[0]["date"] and st["ready"]
    # Second run: everything complete -> nothing downloaded.
    fake.calls = {"yf": [], "tiingo": []}
    r2 = lh.run_backfill(db, tiingo_key="K", today=TODAY, tickers=["SPY", "^VIX", "AAA", "BBB"])
    assert r2["counts"]["skipped"] == 4 and not fake.calls["yf"], r2
    db.close()


@test
def test_backfill_failure_is_recorded_retried_later_and_never_shrinks():
    db = SessionLocal()
    wipe(db)
    full = make_bars(LONG_DAYS)
    fake = FakeSources(yf={"SPY": full, "CCC": []})
    install(fake)
    r = lh.run_backfill(db, tiingo_key=None, today=TODAY, tickers=["SPY", "CCC"])
    assert r["counts"]["failed"] == 1 and r["problems"][0]["ticker"] == "CCC"
    fake.calls = {"yf": [], "tiingo": []}
    lh.run_backfill(db, tiingo_key=None, today=TODAY, tickers=["SPY", "CCC"])
    assert ("CCC", lh.target_start(TODAY)) not in fake.calls["yf"], "a recent failure is not hammered again"
    # A stored full series must survive a later, shorter download.
    put_archive(db, "DDD", full, requested_start=None)
    install(FakeSources(yf={"DDD": full[-300:]}))
    lh.run_backfill(db, tiingo_key=None, today=TODAY, tickers=["DDD"], force=True)
    row = db.query(HistoryArchive).filter(HistoryArchive.ticker == "DDD").one()
    assert row.start_date == full[0]["date"], "never shrink a stored series"
    db.close()


@test
def test_backfill_stops_using_tiingo_when_rate_limited():
    db = SessionLocal()
    wipe(db)
    full = make_bars(LONG_DAYS)
    fake = FakeSources(yf={"SPY": full}, tiingo={"E1": full, "E2": full, "E3": full}, rate_limit_after=1)
    install(fake)
    r = lh.run_backfill(db, tiingo_key="K", today=TODAY, tickers=["SPY", "E1", "E2", "E3"])
    assert fake.calls["tiingo"] == ["E1", "E2"], fake.calls
    assert r["tiingo_note"] and r["counts"]["tiingo"] == 1
    db.close()


@test
def test_vix_archive_refreshes_when_behind():
    db = SessionLocal()
    wipe(db)
    full = make_bars(LONG_DAYS)
    vix_old = make_bars(LONG_DAYS[:-30], start_px=20, drift=0)
    put_archive(db, "^VIX", vix_old, requested_start=lh.target_start(TODAY))
    fake = FakeSources(yf={"SPY": full, "^VIX": make_bars(LONG_DAYS, start_px=20, drift=0)})
    install(fake)
    lh.run_backfill(db, today=TODAY, tickers=["SPY", "^VIX"])
    assert any(t == "^VIX" for t, _ in fake.calls["yf"]), "stale VIX must be refreshed"
    db.close()


# ── engine: 20-year path ───────────────────────────────────────────────────
def seed_engine_world(db, vix_level=None):
    wipe(db)
    days = trading_days("2014-01-01", 2600)
    spy = make_bars(days, drift=0.0006, wiggle=0.004)
    put_archive(db, "SPY", spy[:2000])
    put_cache(db, "SPY", spy[1300:])
    from services.universe import UNIVERSE
    names = [t for t in UNIVERSE if t != "SPY"][:6]
    for k, t in enumerate(names):
        b = make_bars(days, start_px=50 + k * 10, drift=0.0005 + k * 0.0001, wiggle=0.012)
        put_archive(db, t, b[:2000])
        put_cache(db, t, b[1300:])
    if vix_level is not None:
        put_archive(db, "^VIX", make_bars(days, start_px=vix_level, drift=0, wiggle=0))
    return days


@test
def test_20y_run_uses_the_archive_and_reports_it():
    from services.backtest import run_walk_forward_backtest
    db = SessionLocal()
    days = seed_engine_world(db)
    five = run_walk_forward_backtest(db, 1, strategy_id="momentum_rotation", source="universe",
                                     max_window_years=None)
    long = run_walk_forward_backtest(db, 1, strategy_id="momentum_rotation", source="universe",
                                     history="20y", max_window_years=None)
    assert five["data_quality"]["history_first_date"] == days[1300]
    assert long["data_quality"]["history_first_date"] == days[0], long["data_quality"]
    assert long["data_quality"]["tickers_spliced_with_archive"] >= 5
    assert len(long["equity"]) > len(five["equity"]) + 150
    assert long["parameters"]["history"] == "20y" and "history" not in five["parameters"]
    ticks = []
    run_walk_forward_backtest(db, 1, strategy_id="momentum_rotation", source="universe",
                              history="20y", progress=lambda f, d: ticks.append(f))
    assert ticks and all(0 <= f <= 1 for f in ticks), "progress must be reported"
    db.close()


@test
def test_test_window_is_capped_at_five_years_anywhere_in_the_archive():
    """The database keeps 20 years; a Strategy Lab run tests at most 5 of them,
    and start_date picks which. The edge matrix (max_window_years=None) sees all."""
    from services.backtest import MAX_WINDOW_YEARS, run_walk_forward_backtest
    db = SessionLocal()
    days = seed_engine_world(db)
    kw = dict(strategy_id="momentum_rotation", source="universe", history="20y")
    recent = run_walk_forward_backtest(db, 1, **kw)["data_quality"]
    assert recent["test_years"] <= MAX_WINDOW_YEARS and recent["test_last_date"] >= days[-10], recent
    early = run_walk_forward_backtest(db, 1, start_date="2015-06-01", **kw)
    dq = early["data_quality"]
    assert dq["test_first_date"] >= "2015-06-01" and dq["test_last_date"] <= "2020-06-01", dq
    assert dq["test_years"] > 4.5, "an early 5-year window is fully usable — warm-up comes from before it"
    too_long = run_walk_forward_backtest(db, 1, start_date="2015-06-01", end_date="2023-01-01", **kw)
    assert too_long["data_quality"]["test_last_date"] <= "2020-06-01", "an explicit end cannot exceed the cap"
    full = run_walk_forward_backtest(db, 1, max_window_years=None, **kw)["data_quality"]
    assert full["test_years"] > MAX_WINDOW_YEARS
    # Trimming each ticker to the window must not change the answer.
    import services.backtest as bt
    saved = bt._PRE_WINDOW_EXTRA_BARS
    try:
        bt._PRE_WINDOW_EXTRA_BARS = 5000                  # i.e. load everything
        untrimmed = run_walk_forward_backtest(db, 1, start_date="2015-06-01", **kw)
    finally:
        bt._PRE_WINDOW_EXTRA_BARS = saved
    assert untrimmed["equity"] == early["equity"] and untrimmed["trades"] == early["trades"]
    db.close()


@test
def test_real_vix_drives_the_crisis_rule_when_available():
    from services.backtest import run_walk_forward_backtest
    db = SessionLocal()
    seed_engine_world(db, vix_level=40.0)
    hot = run_walk_forward_backtest(db, 1, strategy_id="momentum_rotation", source="universe",
                                    history="20y", spy_regime=False)
    seed_engine_world(db, vix_level=12.0)
    calm = run_walk_forward_backtest(db, 1, strategy_id="momentum_rotation", source="universe",
                                     history="20y", spy_regime=False)
    hot_q = {t["quadrant"] for t in hot["trades"]}
    calm_q = {t["quadrant"] for t in calm["trades"]}
    assert "trending_bull" not in hot_q, "VIX 40 >= VIX_CRISIS can never be a calm trending bull"
    assert "trending_bull" in calm_q, calm_q
    db.close()


# ── API ────────────────────────────────────────────────────────────────────
@test
def test_api_runs_capped_windows_directly_and_backfill_is_admin_only():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.backtest import router as bt_router
    from api.history import router as hist_router
    from auth.deps import get_current_user
    from database.db import get_db
    from services import jobs as jobsvc

    launched = []
    jobsvc.launch_worker = lambda job_id: launched.append(job_id) or 0

    db = SessionLocal()
    seed_engine_world(db)
    admin = User(username="adm_h", password_hash="x", is_admin=True)
    plain = User(username="usr_h", password_hash="x", is_admin=False)
    db.add_all([admin, plain])
    db.commit()
    who = {"u": plain}
    app = FastAPI()
    app.include_router(bt_router)
    app.include_router(hist_router)
    app.dependency_overrides[get_current_user] = lambda: db.merge(who["u"])
    c = TestClient(app)

    r = c.get("/api/backtest/walk-forward", params={"strategy": "momentum_rotation",
                                                    "source": "universe", "start_date": "2015-06-01"}).json()
    assert "equity" in r and not launched, "a capped window answers directly — no background job"
    assert r["data_quality"]["test_first_date"] >= "2015-06-01", "start dates reach into the archive"
    assert r["data_quality"]["test_years"] <= 5
    assert c.get("/api/backtest/walk-forward", params={"strategy": "momentum_rotation",
                                                       "period": "10Y"}).status_code == 422

    assert c.post("/api/history/backfill").status_code == 403
    st = c.get("/api/history/status").json()
    assert st["can_start"] is False and "tickers_total" in st
    who["u"] = admin
    b = c.post("/api/history/backfill").json()
    assert b["status"] == "running" and b["started"] is True
    b2 = c.post("/api/history/backfill").json()
    assert b2["started"] is False, "one download at a time"
    db.close()


@test
def test_worker_registers_the_new_job_kinds():
    from services.job_worker import HANDLERS
    assert {"history_backfill", "edge_matrix"} <= set(HANDLERS)


def main() -> int:
    passed = failed = 0
    failures = []
    for fn in _TESTS:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            failures.append((fn.__name__, traceback.format_exc()))
            print(f"FAIL  {fn.__name__}: {exc}")
        else:
            passed += 1
            print(f"PASS  {fn.__name__}")
    print("\n" + "=" * 56)
    print(f"  {passed} passed, {failed} failed, {len(_TESTS)} total")
    print("=" * 56)
    for name, tb in failures:
        print(f"\n[{name}]\n{tb}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
