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


# ── ML4T gap 9: columnar storage, data versions, revisions ─────────────────
from services import bar_store as bs  # noqa: E402


def _nan_eq(a, b) -> bool:
    """Deep equality where NaN == NaN and -0.0 is distinguished from 0.0."""
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return a == b and math.copysign(1, a) == math.copysign(1, b)
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return list(a) == list(b) and all(_nan_eq(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_nan_eq(x, y) for x, y in zip(a, b))
    return a == b


@test
def test_codec_round_trips_exactly_including_nan_none_and_big_volumes():
    bars = [
        {"i": 0, "date": "2024-01-02", "open": 1.1, "high": float("nan"), "low": -0.0,
         "close": 1.23456789012345, "vol": 9_000_000_000_123},
        {"i": 1, "date": "2024-01-03", "open": None, "high": float("inf"), "low": 5e-324,
         "close": 2.0, "vol": None},
        {"i": 2, "date": None, "open": 3.0, "high": float("-inf"), "low": 1e308,
         "close": 0.1 + 0.2, "vol": 0},
    ]
    blob = bs.encode_bars(bars)
    assert blob is not None and bs.is_blob(blob) and bs.blob_count(blob) == 3
    back = bs.decode_bars(blob)
    assert _nan_eq(back, bars), back
    assert all(type(b["i"]) is int for b in back) and type(back[0]["vol"]) is int
    # Key order is preserved (the archive's order differs from the cache's).
    arch = [{"date": "2024-01-02", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "vol": 5, "i": 0}]
    assert list(bs.decode_bars(bs.encode_bars(arch))[0]) == list(arch[0])
    assert bs.decode_bars(bs.encode_bars([])) == []


@test
def test_codec_refuses_what_it_cannot_hold_exactly_and_writers_fall_back_to_json():
    base = {"date": "2024-01-02", "close": 1.0, "vol": 1}
    for bad in (
        [dict(base, vol=True)],                                 # bool is not an int here
        [base, dict(base, vol=1.5)],                            # mixed int/float column
        [base, {"date": "2024-01-03", "close": 1.0}],           # rows with different keys
        [dict(base, date="2024-01-0\u00e9")],                   # non-ASCII string
        [dict(base, vol=2 ** 70)],                              # beyond int64
        "not a list",
    ):
        assert bs.encode_bars(bad) is None, bad
    blob, text, version = bs.payload_for([dict(base, vol=True)])
    assert blob is None and json.loads(text) == [dict(base, vol=True)]
    assert version == bs.bars_version([dict(base, vol=True)])


@test
def test_read_bars_prefers_blob_falls_back_to_json_and_trims_like_json():
    bars = make_bars(trading_days("2020-01-01", 300))
    text = json.dumps(bars)
    blob = bs.encode_bars(bars)
    assert bs.read_bars(blob, None) == bars
    assert bs.read_bars(None, text) == bars                      # not migrated yet
    assert bs.read_bars(b"SBR1garbage", text) == bars            # unreadable BLOB -> JSON
    since = bars[200]["date"]
    assert bs.read_bars(blob, None, since=since, keep_before=50) == bars[150:]
    assert bs.read_bars(None, text, since=since, keep_before=50) == bars[150:]
    assert bs.read_bars(blob, None, since=since, keep_before=500) == bars
    assert bs.bar_count(blob, None) == 300 and bs.bar_count(None, text) == 300


@test
def test_data_version_is_the_same_before_and_after_migration():
    bars = make_bars(trading_days("2021-01-01", 50))
    text = json.dumps(bars)
    v = bs.bars_version(bars)
    assert bs.text_version(text) == v
    assert bs.row_version(bs.encode_bars(bars), None, None) == v
    assert bs.row_version(None, text, None) == v
    assert bs.row_version(None, text, "stored") == "stored"
    bars2 = [dict(b) for b in bars]
    bars2[10]["close"] += 0.01
    assert bs.bars_version(bars2) != v


def _legacy_rows(db, n=4):
    """JSON rows exactly as pre-gap-9 versions of the app wrote them."""
    wipe(db)
    days = trading_days("2019-01-01", 400)
    series = {}
    for k in range(n):
        t = f"L{k}"
        bars = make_bars(days, start_px=20 + k)
        series[t] = bars
        db.add(StockCache(ticker=t, history_json=json.dumps(bars[200:])))
        db.add(HistoryArchive(ticker=t, bars_json=json.dumps(bars), start_date=days[0], end_date=days[-1]))
    db.commit()
    return series


@test
def test_migration_is_verified_resumable_and_idempotent():
    db = SessionLocal()
    series = _legacy_rows(db)
    before = {t: (lh.load_cache_bars(db, t), lh.load_archive_bars(db, t)) for t in series}
    first = bs.migrate(db, max_rows=3)                             # interrupted part-way
    assert first["cache"] + first["archive"] == 3 and first["remaining"] == 5, first
    second = bs.migrate(db)                                       # resumes
    assert second["remaining"] == 0 and second["cache"] + second["archive"] == 5, second
    third = bs.migrate(db)                                        # nothing left to do
    assert third["cache"] == third["archive"] == 0 and third["remaining"] == 0
    db.expire_all()
    for t in series:
        c = db.query(StockCache).filter(StockCache.ticker == t).one()
        a = db.query(HistoryArchive).filter(HistoryArchive.ticker == t).one()
        assert c.history_json is None and c.history_blob and a.bars_json == "" and a.bars_blob
        assert c.history_version == bs.bars_version(before[t][0])
        assert lh.load_cache_bars(db, t) == before[t][0]
        assert lh.load_archive_bars(db, t) == before[t][1]
    st = bs.storage_status(db)
    assert st["migrated"] and st["cache_columnar"] == 4 and st["archive_columnar"] == 4
    db.close()


@test
def test_migration_keeps_json_it_cannot_reproduce_and_never_overwrites_a_newer_write():
    db = SessionLocal()
    wipe(db)
    odd = [{"date": "2024-01-02", "close": 1.0, "vol": True}]      # the codec refuses bools
    db.add(StockCache(ticker="ODD", history_json=json.dumps(odd)))
    good = make_bars(trading_days("2024-01-01", 30))
    db.add(StockCache(ticker="RACE", history_json=json.dumps(good)))
    db.commit()
    newer = make_bars(trading_days("2024-01-01", 31), start_px=77)
    real = bs._verified_blob

    def racing(text):
        # The web process refreshes RACE between this run's read and its write.
        if text == json.dumps(good):
            other = SessionLocal()
            row = other.query(StockCache).filter(StockCache.ticker == "RACE").one()
            row.history_json = json.dumps(newer)
            other.commit()
            other.close()
        return real(text)

    bs._verified_blob = racing
    try:
        res = bs.migrate(db)
    finally:
        bs._verified_blob = real
    assert {"store": "cache", "ticker": "ODD"} in res["kept_json"]
    assert res["raced"] >= 1, res
    db.expire_all()
    assert json.loads(db.query(StockCache).filter(StockCache.ticker == "ODD").one().history_json) == odd
    assert lh.load_cache_bars(db, "RACE") == newer, "the newer write must survive"
    assert bs.migrate(db)["cache"] == 1, "the raced row converts on the next run"
    assert lh.load_cache_bars(db, "RACE") == newer
    db.close()


@test
def test_compare_series_classifies_adjustments_corrections_and_ignores_growth():
    days = trading_days("2023-01-02", 120)
    old = make_bars(days[:100])
    assert bs.compare_series(old, old) is None
    grown = old + make_bars(days[100:], start_px=old[-1]["close"])
    assert bs.compare_series(old, grown) is None, "a longer series is not a revision"
    rolled = old[20:] + make_bars(days[100:], start_px=old[-1]["close"])
    assert bs.compare_series(old, rolled) is None, "a rolled-forward window is not a revision"
    last_changed = [dict(b) for b in old]
    last_changed[-1]["close"] *= 1.05                             # an in-progress session settled
    assert bs.compare_series(old, last_changed) is None
    # A dividend: every bar before the ex-date scaled by the same factor.
    div = [dict(b) for b in old]
    for b in div[:60]:
        for k in ("open", "high", "low", "close"):
            b[k] = round(b[k] * 0.99, 4)
    ch = bs.compare_series(old, div)
    assert ch and ch["kind"] == "adjustment" and ch["changed"] == 60 and abs(ch["factor"] - 0.99) < 1e-3, ch
    split = [dict(b) for b in old]
    for b in split[:60]:
        for k in ("open", "high", "low", "close"):
            b[k] = round(b[k] / 4, 4)
    assert bs.compare_series(old, split)["kind"] == "split"
    fixed = [dict(b) for b in old]
    fixed[40]["high"] *= 1.2
    ch = bs.compare_series(old, fixed)
    assert ch["kind"] == "correction" and ch["changed"] == 1, ch
    tiny = [dict(b) for b in old]
    tiny[40]["close"] *= 1.0001                                   # rounding noise
    tiny[41]["vol"] += 12345                                      # volume revisions are routine
    assert bs.compare_series(old, tiny) is None


@test
def test_archive_rewrite_keeps_the_previous_series_and_identical_refetch_does_not():
    from database.models import HistoryRevision
    db = SessionLocal()
    wipe(db)
    db.query(HistoryRevision).delete()
    db.commit()
    days = trading_days("2020-01-01", 200)
    v1 = make_bars(days)
    lh._store(db, "ADJ", v1, "yfinance", days[0], [])
    lh._store(db, "ADJ", v1, "yfinance", days[0], [])              # identical re-download
    assert db.query(HistoryRevision).count() == 0
    v2 = [dict(b) for b in v1]
    for b in v2[:150]:
        for k in ("open", "high", "low", "close"):
            b[k] = round(b[k] * 0.985, 4)
    lh._store(db, "ADJ", v2, "yfinance", days[0], [])
    revs = db.query(HistoryRevision).all()
    assert len(revs) == 1 and revs[0].kind == "adjustment" and revs[0].store == "archive", revs
    assert bs.revision_bars(revs[0]) == v1, "the old series must be reconstructable"
    assert revs[0].prev_version == bs.bars_version(v1) and revs[0].new_version == bs.bars_version(v2)
    row = db.query(HistoryArchive).filter(HistoryArchive.ticker == "ADJ").one()
    assert row.bars_blob and row.data_version == bs.bars_version(v2) and lh.load_archive_bars(db, "ADJ") == v2
    db.close()


@test
def test_cache_refetch_records_a_revision_and_stores_columnar():
    import asyncio
    from database.models import HistoryRevision
    from services import market_data as md
    db = SessionLocal()
    wipe(db)
    db.query(HistoryRevision).delete()
    db.commit()
    days = trading_days("2021-01-01", 300)
    v1 = make_bars(days[:299])
    db.add(StockCache(ticker="CACHEX", history_json=json.dumps(v1)))   # legacy JSON row
    db.commit()
    v2 = [dict(b) for b in v1]
    for b in v2[:250]:
        b["close"] = round(b["close"] * 0.99, 4)
    v2.append(make_bars(days[299:], start_px=v1[-1]["close"])[0])
    real = md._fetch_history_sync
    md._fetch_history_sync = lambda ticker, period="5y": v2
    md._mem_cache.pop("history:CACHEX", None)
    try:
        got = asyncio.run(md.get_history("CACHEX", db, force_refresh=True))
    finally:
        md._fetch_history_sync = real
    assert got == v2
    db.expire_all()
    row = db.query(StockCache).filter(StockCache.ticker == "CACHEX").one()
    assert row.history_blob and row.history_json is None and row.history_version == bs.bars_version(v2)
    rev = db.query(HistoryRevision).filter(HistoryRevision.ticker == "CACHEX").one()
    assert rev.store == "cache" and bs.revision_bars(rev) == v1
    assert rev.prev_version == bs.text_version(json.dumps(v1))
    db.close()


@test
def test_revision_retention_per_series_and_total_bytes():
    from database.models import HistoryRevision
    db = SessionLocal()
    db.query(HistoryRevision).delete()
    db.commit()
    bars = make_bars(trading_days("2022-01-03", 50))
    change = {"kind": "correction", "reason": "test"}
    for _ in range(bs.REVISIONS_PER_SERIES + 3):
        bs.record_revision(db, "KEEP", "cache", bars, change, "a", "b")
    db.commit()
    assert db.query(HistoryRevision).filter(HistoryRevision.ticker == "KEEP").count() == bs.REVISIONS_PER_SERIES
    old_cap = bs.REVISIONS_MAX_BYTES
    try:
        size = db.query(HistoryRevision).first().payload_bytes
        bs.REVISIONS_MAX_BYTES = size * 4 + 1
        for t in ("B1", "B2", "B3"):
            bs.record_revision(db, t, "cache", bars, change, "a", "b")
        db.commit()
        total = sum(r.payload_bytes for r in db.query(HistoryRevision).all())
        assert total <= bs.REVISIONS_MAX_BYTES, total
        assert db.query(HistoryRevision).filter(HistoryRevision.ticker == "B3").count() == 1, \
            "the newest revision survives; the oldest are pruned first"
    finally:
        bs.REVISIONS_MAX_BYTES = old_cap
    db.close()


@test
def test_walk_forward_reports_a_stable_data_fingerprint_and_sources():
    from services.backtest import run_walk_forward_backtest
    db = SessionLocal()
    seed_engine_world(db)
    kw = dict(strategy_id="momentum_rotation", source="universe", history="20y", max_window_years=None)
    a = run_walk_forward_backtest(db, 1, **kw)
    b = run_walk_forward_backtest(db, 1, **kw)
    dv = a["data_versions"]
    assert dv["fingerprint"] and dv["fingerprint"] == b["data_versions"]["fingerprint"]
    assert dv["tickers"] >= 5 and dv["unversioned"] == 0
    assert any(s.startswith("archive:") for s in dv["sources"]), dv["sources"]
    assert a["data_quality"]["sources"] and any(c["id"] == "data_sources" for c in a["caveats"])
    # Migrating to the columnar format must not change the data version...
    bs.migrate(db)
    c = run_walk_forward_backtest(db, 1, **kw)
    assert c["data_versions"]["fingerprint"] == dv["fingerprint"]
    assert c["equity"] == a["equity"] and c["trades"] == a["trades"] and c["metrics"] == a["metrics"]
    # ...but changing one stored series must.
    t = next(x for x in dv["versions"] if x not in ("SPY", "^VIX"))
    bars = lh.load_cache_bars(db, t)
    bars[-2]["close"] = round(bars[-2]["close"] * 1.03, 4)
    put_cache(db, t, bars)
    row = db.query(StockCache).filter(StockCache.ticker == t).one()
    row.history_blob, row.history_version = None, None
    db.commit()
    d = run_walk_forward_backtest(db, 1, **kw)
    assert d["data_versions"]["fingerprint"] != dv["fingerprint"]
    assert d["data_versions"]["versions"][t]["cache"] != dv["versions"][t]["cache"]
    five = run_walk_forward_backtest(db, 1, strategy_id="momentum_rotation", source="universe")
    assert five["data_versions"]["fingerprint"] and five["data_versions"]["sources"].get("cache")
    db.close()


@test
def test_storage_status_and_migration_endpoint_are_wired():
    from services.job_worker import HANDLERS
    assert "bar_storage_migration" in HANDLERS
    db = SessionLocal()
    st = bs.storage_status(db)
    assert {"cache_columnar", "archive_columnar", "pending_cache", "pending_archive",
            "revisions", "revision_limits"} <= set(st)
    db.close()


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
