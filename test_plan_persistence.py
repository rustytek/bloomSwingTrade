"""
Pure-function, no-network tests for trade-plan persistence and the features it
unlocks.

Covers:
  1. services/trade_plan.py::parse_plan_notes   — the legacy `notes` decoder
  2. main.py::backfill_plan_fields_from_notes   — the one-time migration,
     against a REAL temporary SQLite file (never the live DB), including
     idempotency and "notes are never destroyed"
  3. services/scorecard.py::_chasing            — entry_chasing, and above all
     the honesty rule: a missing planned_entry is UNKNOWN, never "clean"
  4. api/settings.py::resolve_max_open_r        — chosen vs derived budget
  5. services/regime.py::strategies_for_regime  — UNCHANGED for all four
     quadrants with the evidence override off, and with a missing/malformed
     matrix when it is on
  6. services/exits.py::build_exit_rules        — time stop defaults to the
     strategy's horizon_days, falls back to 20

Run:  python test_plan_persistence.py
No network. The only DB touched is a throwaway file in the system temp dir.
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import math
import os
import sqlite3
import sys
import tempfile
import traceback
import types

# ──────────────────────────────────────────────────────────────────────────
# Web-dependency stubs (same block as test_passes.py)
# ──────────────────────────────────────────────────────────────────────────
# Some helpers under test live in api/*.py, which import fastapi (and jose /
# passlib through auth.deps). Permissive stand-ins are appended to the END of
# sys.meta_path, so a real install always wins and these are inert when the
# packages are present.
_STUBBED_PACKAGES = ("fastapi", "jose", "passlib", "bcrypt")


class _Stub:
    def __init__(self, name: str):
        self._name = name

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and not kwargs and callable(args[0]) and not isinstance(args[0], _Stub):
            return args[0]
        return _Stub(f"{self._name}()")

    def __getattr__(self, item):
        return _Stub(f"{self._name}.{item}")

    def __repr__(self):
        return f"<stub {self._name}>"


class _StubModule(types.ModuleType):
    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        return _Stub(f"{self.__name__}.{item}")


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in _STUBBED_PACKAGES:
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        module = _StubModule(spec.name)
        module.__path__ = []
        return module

    def exec_module(self, module):
        pass


_STUB_FINDER = _StubFinder()
sys.meta_path.append(_STUB_FINDER)

# Point the ORM at a throwaway DB BEFORE anything imports database/db.py, so
# importing main can never touch ./data/swingtrader.db.
_TMPDIR = tempfile.mkdtemp(prefix="swingtrader-test-")
_TEST_DB = os.path.join(_TMPDIR, "plan_persistence_test.db").replace("\\", "/")
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB}"

from services.trade_plan import parse_plan_notes                      # noqa: E402
from services.exits import (                                          # noqa: E402
    DEFAULT_TIME_STOP_BARS, TimeStop, build_exit_rules, strategy_horizon_days,
)
from services.regime import (                                         # noqa: E402
    QUADRANTS, evidence_adjustment, set_evidence_override,
    strategies_for_regime, strategies_for_regime_with_evidence,
)
from services import scorecard                                        # noqa: E402


# The web-dependency stub above is import-time scaffolding only. Drop the
# finder and purge any stub modules it provided, so later imports in this
# process resolve the real packages (or raise a real ImportError) instead of
# silently reusing stubs bound for these tests.
try:
    sys.meta_path.remove(_STUB_FINDER)
except ValueError:
    pass
for _stubbed in [m for m in list(sys.modules) if m.split(".")[0] in _STUBBED_PACKAGES]:
    del sys.modules[_stubbed]


# ──────────────────────────────────────────────────────────────────────────
# Tiny test runner
# ──────────────────────────────────────────────────────────────────────────
_TESTS = []


def test(fn):
    _TESTS.append(fn)
    return fn


def approx(a, b, tol=1e-6):
    return a is not None and b is not None and abs(a - b) <= tol


class _Obj:
    """Duck-typed stand-in for a ClosedTrade row (no DB, no ORM)."""

    _FIELDS = ("id", "ticker", "strategy", "avg_cost", "exit_price", "planned_entry",
               "planned_entry_high", "initial_stop", "stop_loss", "r_multiple", "pnl",
               "entry_date", "exit_date")

    def __init__(self, **kw):
        for f in self._FIELDS:
            setattr(self, f, None)
        for k, v in kw.items():
            setattr(self, k, v)


# ──────────────────────────────────────────────────────────────────────────
# 1. The legacy notes parser
# ──────────────────────────────────────────────────────────────────────────
# This is exactly what static/today.html's Plan-a-Trade modal writes today.
FULL_NOTES = (
    "THESIS: pullback to the rising 50MA, volume drying up on the dip\n"
    "INVALIDATION: daily close below 118.00\n"
    "TIME STOP: 12 days\n"
    "PLAN: entry 123.45 stop 118.20 target 133.90 2.0R\n"
    "STRATEGY: Pullback to 50MA"
)
PARTIAL_NOTES = (
    "THESIS: 60-day-high breakout on 2x volume\n"
    "PLAN: entry 88.10 stop 82.00 target 100.30 2.0R"
)
PLAIN_NOTES = "bought a starter position, will add on strength"


@test
def test_parse_full_plan_notes():
    p = parse_plan_notes(FULL_NOTES)
    assert p["thesis"] == "pullback to the rising 50MA, volume drying up on the dip", p
    assert p["invalidation"] == "daily close below 118.00", p
    assert p["time_stop_days"] == 12, p
    assert approx(p["planned_entry"], 123.45), p
    assert p["planned_entry_high"] is None, p       # the legacy line has no zone
    assert p["strategy_name"] == "Pullback to 50MA", p


@test
def test_parse_partial_plan_notes_leaves_absent_fields_none():
    p = parse_plan_notes(PARTIAL_NOTES)
    assert p["thesis"] == "60-day-high breakout on 2x volume", p
    assert approx(p["planned_entry"], 88.10), p
    # Absent prefixes must be None — NOT an empty string, and NOT a guess.
    assert p["invalidation"] is None, p
    assert p["time_stop_days"] is None, p


@test
def test_parse_notes_with_no_prefixes_yields_nothing():
    p = parse_plan_notes(PLAIN_NOTES)
    assert all(v is None for v in p.values()), p
    assert all(v is None for v in parse_plan_notes(None).values())
    assert all(v is None for v in parse_plan_notes("").values())


@test
def test_parse_multiline_thesis_is_not_split_into_a_new_field():
    notes = ("THESIS: first line of reasoning\nsecond line, no prefix\n"
             "INVALIDATION: close below 50\nPLAN: entry 10.5 stop 9 target 13 2.0R")
    p = parse_plan_notes(notes)
    assert p["thesis"] == "first line of reasoning\nsecond line, no prefix", p
    assert p["invalidation"] == "close below 50", p
    assert approx(p["planned_entry"], 10.5), p


@test
def test_parse_time_stop_without_a_number_is_none_not_a_guess():
    p = parse_plan_notes("TIME STOP: until the thesis breaks")
    assert p["time_stop_days"] is None, p
    # ...but a bare number or "N days" both work.
    assert parse_plan_notes("TIME STOP: 7")["time_stop_days"] == 7
    assert parse_plan_notes("TIME STOP: 21 trading days")["time_stop_days"] == 21


@test
def test_parse_entry_zone_when_one_is_present():
    p = parse_plan_notes("PLAN: entry 100.00 zone 99.00-101.50 stop 95 target 110 2.0R")
    assert approx(p["planned_entry"], 100.0), p
    assert approx(p["planned_entry_high"], 101.5), p


# ──────────────────────────────────────────────────────────────────────────
# 2. The backfill, against a real SQLite file
# ──────────────────────────────────────────────────────────────────────────
def _fresh_db() -> str:
    """A throwaway DB with the legacy-shaped tables plus the new columns."""
    path = os.path.join(tempfile.mkdtemp(prefix="swingtrader-backfill-"), "t.db")
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE portfolio_positions (
            id INTEGER PRIMARY KEY, user_id INTEGER, ticker TEXT, shares FLOAT,
            avg_cost FLOAT, notes TEXT,
            planned_entry FLOAT, planned_entry_high FLOAT,
            thesis TEXT, invalidation TEXT, time_stop_days INTEGER
        );
        CREATE TABLE closed_trades (
            id INTEGER PRIMARY KEY, user_id INTEGER, ticker TEXT, notes TEXT,
            planned_entry FLOAT, planned_entry_high FLOAT,
            thesis TEXT, invalidation TEXT, time_stop_days INTEGER
        );
        """
    )
    con.commit()
    con.close()
    return path


def _seed(path):
    con = sqlite3.connect(path)
    con.executemany(
        "INSERT INTO portfolio_positions (id,user_id,ticker,shares,avg_cost,notes) "
        "VALUES (?,?,?,?,?,?)",
        [
            (1, 1, "AAPL", 10, 123.90, FULL_NOTES),      # every prefix
            (2, 1, "MSFT", 5, 88.50, PARTIAL_NOTES),     # only some prefixes
            (3, 1, "NVDA", 2, 500.00, PLAIN_NOTES),      # no prefixes at all
            (4, 1, "TSLA", 1, 200.00, None),             # no notes at all
        ],
    )
    con.execute("INSERT INTO closed_trades (id,user_id,ticker,notes) VALUES (?,?,?,?)",
                (1, 1, "AMD", FULL_NOTES))
    con.commit()
    con.close()


def _rows(path, table="portfolio_positions"):
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    rows = {r["id"]: dict(r) for r in con.execute(f"SELECT * FROM {table}")}
    con.close()
    return rows


def _run_backfill(path):
    """Run the real migration function from main.py against `path`."""
    import main
    from sqlalchemy import create_engine
    engine = create_engine(f"sqlite:///{path.replace(chr(92), '/')}")
    with engine.begin() as conn:
        return main.backfill_plan_fields_from_notes(conn)


@test
def test_backfill_moves_full_and_partial_notes_and_leaves_the_rest_alone():
    path = _fresh_db()
    _seed(path)
    before = _rows(path)
    moved = _run_backfill(path)
    after = _rows(path)

    assert moved["portfolio_positions"] == 2, moved   # rows 1 and 2 only
    assert moved["closed_trades"] == 1, moved

    r1 = after[1]
    assert approx(r1["planned_entry"], 123.45), r1
    assert r1["time_stop_days"] == 12, r1
    assert r1["thesis"].startswith("pullback to the rising 50MA"), r1
    assert r1["invalidation"] == "daily close below 118.00", r1

    r2 = after[2]
    assert approx(r2["planned_entry"], 88.10), r2
    assert r2["thesis"] == "60-day-high breakout on 2x volume", r2
    # A prefix that was never written stays NULL — nothing is invented.
    assert r2["invalidation"] is None, r2
    assert r2["time_stop_days"] is None, r2

    # Rows with no plan prefixes, and rows with no notes, are untouched.
    for rid in (3, 4):
        for col in ("planned_entry", "planned_entry_high", "thesis",
                    "invalidation", "time_stop_days"):
            assert after[rid][col] is None, (rid, col, after[rid])

    # notes is NEVER modified — it is the only copy of the reasoning.
    for rid in before:
        assert after[rid]["notes"] == before[rid]["notes"], rid

    # Core position values are untouched too.
    for rid in before:
        for col in ("user_id", "ticker", "shares", "avg_cost"):
            assert after[rid][col] == before[rid][col], (rid, col)


@test
def test_backfill_is_idempotent():
    path = _fresh_db()
    _seed(path)
    _run_backfill(path)
    first = _rows(path)
    moved2 = _run_backfill(path)
    second = _rows(path)
    assert moved2["portfolio_positions"] == 0, moved2   # nothing left to move
    assert moved2["closed_trades"] == 0, moved2
    assert first == second, "second backfill run changed data"


@test
def test_backfill_never_overwrites_a_value_the_api_already_wrote():
    path = _fresh_db()
    _seed(path)
    con = sqlite3.connect(path)
    # The API wrote a real planned_entry; the notes text says something else.
    con.execute("UPDATE portfolio_positions SET planned_entry = 999.0 WHERE id = 1")
    con.commit()
    con.close()
    _run_backfill(path)
    after = _rows(path)
    assert approx(after[1]["planned_entry"], 999.0), after[1]
    # The other NULL columns on that same row still get filled.
    assert after[1]["time_stop_days"] == 12, after[1]


# ──────────────────────────────────────────────────────────────────────────
# 3. entry_chasing — and the "unavailable is not zero" honesty rule
# ──────────────────────────────────────────────────────────────────────────
@test
def test_entry_chasing_flags_a_fill_above_the_planned_entry():
    trades = [
        # chased: paid 105 against a 100 plan, 1R of risk = 10/share
        _Obj(id=1, ticker="AAPL", strategy="pullback_50ma", avg_cost=105.0,
             planned_entry=100.0, initial_stop=95.0, r_multiple=1.2),
        # disciplined: filled at the plan
        _Obj(id=2, ticker="MSFT", strategy="pullback_50ma", avg_cost=100.0,
             planned_entry=100.0, initial_stop=95.0, r_multiple=0.4),
        # better than plan
        _Obj(id=3, ticker="NVDA", strategy="pullback_50ma", avg_cost=98.0,
             planned_entry=100.0, initial_stop=95.0, r_multiple=2.0),
    ]
    m = scorecard._chasing(trades)
    assert m["available"] is True, m
    assert m["value"] == 1, m
    assert m["trades_examined"] == 3, m
    assert m["trades_total"] == 3, m
    assert m["trades_missing_planned_entry"] == 0, m
    assert approx(m["avg_overpay_pct"], 5.0, 0.01), m
    # overpay 5.00 / risk-per-share 10.00 = 0.5R
    assert approx(m["r_cost"], 0.5, 0.01), m
    assert m["offenders"][0]["ticker"] == "AAPL", m


@test
def test_entry_chasing_uses_the_top_of_the_zone_when_one_was_planned():
    trades = [
        # inside the planned zone -> not chasing
        _Obj(id=1, ticker="AAPL", avg_cost=101.0, planned_entry=100.0,
             planned_entry_high=102.0, initial_stop=95.0, r_multiple=1.0),
        # above the zone -> chasing
        _Obj(id=2, ticker="MSFT", avg_cost=104.0, planned_entry=100.0,
             planned_entry_high=102.0, initial_stop=95.0, r_multiple=1.0),
    ]
    m = scorecard._chasing(trades)
    assert m["value"] == 1, m
    assert m["offenders"][0]["ticker"] == "MSFT", m
    assert approx(m["offenders"][0]["reference_price"], 102.0), m["offenders"][0]


@test
def test_entry_chasing_is_unavailable_when_no_row_has_planned_entry():
    """THE honesty rule: no data must never read as a clean bill of health."""
    trades = [
        _Obj(id=1, ticker="AAPL", avg_cost=105.0, initial_stop=95.0, r_multiple=1.0),
        _Obj(id=2, ticker="MSFT", avg_cost=100.0, initial_stop=95.0, r_multiple=0.5),
    ]
    m = scorecard._chasing(trades)
    assert m["available"] is False, m
    assert m["status"] == "UNAVAILABLE", m
    # It must NOT report zero offenders as though the field were present.
    assert m["value"] is None, m
    assert m.get("r_cost") is None, m
    assert "planned_entry" in m["needs"], m
    assert m["observed"]["trades_examined"] == 0, m
    assert m["observed"]["trades_total"] == 2, m
    assert m["observed"]["trades_missing_planned_entry"] == 2, m


@test
def test_entry_chasing_reports_partial_coverage_honestly():
    """Only SOME rows carry a plan: the metric computes, but says so loudly."""
    trades = [
        _Obj(id=1, ticker="AAPL", avg_cost=105.0, planned_entry=100.0,
             initial_stop=95.0, r_multiple=1.0),                       # planned, chased
        _Obj(id=2, ticker="MSFT", avg_cost=100.0, initial_stop=95.0),  # legacy, no plan
        _Obj(id=3, ticker="NVDA", avg_cost=700.0, initial_stop=650.0), # legacy, no plan
    ]
    m = scorecard._chasing(trades)
    assert m["available"] is True, m
    assert m["value"] == 1, m
    # The two legacy rows are NOT counted as "not chasing".
    assert m["trades_examined"] == 1, m
    assert m["trades_total"] == 3, m
    assert m["trades_missing_planned_entry"] == 2, m
    assert "NOT counted as clean" in m["coverage_note"], m
    assert m["trades_examined"] < m["trades_total"], m


@test
def test_entry_chasing_zero_offenders_is_only_reported_over_examined_rows():
    """A clean result is only ever stated about rows that HAVE a plan."""
    trades = [
        _Obj(id=1, ticker="AAPL", avg_cost=99.0, planned_entry=100.0,
             initial_stop=95.0, r_multiple=1.0),
        _Obj(id=2, ticker="MSFT", avg_cost=100.0),   # no plan at all
    ]
    m = scorecard._chasing(trades)
    assert m["value"] == 0 and m["available"] is True, m
    assert m["trades_examined"] == 1 and m["trades_total"] == 2, m
    assert "1 planned trades" in m["detail"], m["detail"]
    assert m["trades_missing_planned_entry"] == 1, m


@test
def test_entry_chasing_r_cost_needs_a_recorded_r_multiple():
    """No r_multiple -> the overpay is still reported, the R cost is not made up."""
    trades = [
        _Obj(id=1, ticker="AAPL", avg_cost=110.0, planned_entry=100.0,
             initial_stop=None, stop_loss=None, r_multiple=None),
    ]
    m = scorecard._chasing(trades)
    assert m["value"] == 1, m
    assert approx(m["avg_overpay_pct"], 10.0, 0.01), m
    assert m["r_cost"] == 0.0, m            # nothing priced, so nothing accumulated
    assert m["unpriced_offenders"] == 1, m
    assert m["offenders"][0]["r_cost"] is None, m
    assert m["offenders"][0]["r_cost_note"], m


@test
def test_entry_chasing_numbers_are_all_finite():
    """Starlette serializes with allow_nan=False — one NaN 500s the response."""
    trades = [
        _Obj(id=1, ticker="A", avg_cost=float("nan"), planned_entry=100.0),
        _Obj(id=2, ticker="B", avg_cost=105.0, planned_entry=float("nan")),
        _Obj(id=3, ticker="C", avg_cost=105.0, planned_entry=0.0),
        _Obj(id=4, ticker="D", avg_cost=105.0, planned_entry=100.0,
             initial_stop=105.0, r_multiple=1.0),   # zero risk per share
    ]
    m = scorecard._chasing(trades)

    def walk(o):
        if isinstance(o, float):
            assert math.isfinite(o), o
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(m)
    # A planned_entry of NaN is not a planned entry at all.
    assert m["trades_examined"] == 3, m


# ──────────────────────────────────────────────────────────────────────────
# 4. max_open_r: chosen vs derived
# ──────────────────────────────────────────────────────────────────────────
class _User:
    def __init__(self, max_positions=8, risk_pct=1.0, max_open_r=None):
        self.max_positions = max_positions
        self.risk_pct = risk_pct
        self.max_open_r = max_open_r
        self.account_size = 10000.0
        self.atr_stop_mult = 2.5
        self.r_multiple = 2.0
        self.use_evidence_regimes = False


@test
def test_max_open_r_is_derived_when_not_chosen():
    from api.settings import resolve_max_open_r
    budget, basis = resolve_max_open_r(_User(max_positions=8, risk_pct=1.0))
    assert approx(budget, 8.0), budget
    assert "derived" in basis.lower(), basis
    assert "chosen ceiling" not in basis, basis


@test
def test_max_open_r_is_honoured_when_chosen():
    from api.settings import resolve_max_open_r
    budget, basis = resolve_max_open_r(
        _User(max_positions=8, risk_pct=1.0, max_open_r=5.0))
    assert approx(budget, 5.0), budget
    assert "chosen ceiling" in basis, basis
    assert "derived" not in basis.split("not derived")[0].lower(), basis


@test
def test_max_open_r_nonsense_values_fall_back_to_derived():
    from api.settings import resolve_max_open_r
    for bad in (0, -3.0, "abc", float("nan")):
        budget, basis = resolve_max_open_r(
            _User(max_positions=10, risk_pct=1.0, max_open_r=bad))
        assert approx(budget, 10.0), (bad, budget)
        assert "chosen ceiling" not in basis, (bad, basis)


@test
def test_portfolio_heat_budget_basis_reflects_which_budget_is_in_play():
    from api.settings import resolve_max_open_r
    from api.portfolio import _with_basis
    from services import portfolio_risk

    for user, expect_chosen in ((_User(), False), (_User(max_open_r=4.0), True)):
        budget, basis = resolve_max_open_r(user)
        heat = _with_basis(
            portfolio_risk.portfolio_heat([], {}, budget, risk_unit=100.0,
                                          max_positions=user.max_positions),
            basis,
        )
        assert approx(heat["max_open_r"], 4.0 if expect_chosen else 8.0), heat
        assert ("chosen ceiling" in heat["budget_basis"]) is expect_chosen, heat


# ──────────────────────────────────────────────────────────────────────────
# 5. The evidence override must not change today's behaviour
# ──────────────────────────────────────────────────────────────────────────
MALFORMED_MATRICES = [
    None,
    {},
    {"strategies": None},
    {"strategies": []},
    {"strategies": {"momentum_rotation": None}},
    {"strategies": {"momentum_rotation": {"cells": None}}},
    {"strategies": {"momentum_rotation": {"cells": {"trending_bull": "nope"}}}},
    {"strategies": {"momentum_rotation": {"cells": {"trending_bull": {"verdict": 7}}}}},
    {"nonsense": True},
]


@test
def test_strategies_for_regime_unchanged_with_the_override_off():
    set_evidence_override(False)
    try:
        for q in list(QUADRANTS) + [None]:
            base = strategies_for_regime(q)
            assert strategies_for_regime_with_evidence(q) == base, q
            # Even a PERFECTLY VALID matrix must do nothing while off.
            matrix = {"strategies": {
                sid: {"cells": {q: {"verdict": "mis-tagged"}}} for sid in base
            }} if q else {}
            assert strategies_for_regime_with_evidence(q, matrix) == base, q
    finally:
        set_evidence_override(False)


@test
def test_strategies_for_regime_unchanged_when_on_but_matrix_is_unusable():
    set_evidence_override(True)
    try:
        for q in list(QUADRANTS) + [None]:
            base = strategies_for_regime(q)
            for matrix in MALFORMED_MATRICES:
                got = strategies_for_regime_with_evidence(q, matrix)
                assert got == base, (q, matrix, sorted(got), sorted(base))
                adj = evidence_adjustment(q, matrix, force=True)
                assert adj["excluded"] == [] and adj["added"] == [], (q, matrix, adj)
    finally:
        set_evidence_override(False)


@test
def test_evidence_can_never_blank_the_playbook():
    """Even valid evidence condemning EVERY strategy degrades to the tags."""
    for q in QUADRANTS:
        base = strategies_for_regime(q)
        if not base:
            continue
        matrix = {"strategies": {sid: {"cells": {q: {"verdict": "mis-tagged"}}}
                                 for sid in base}}
        adj = evidence_adjustment(q, matrix, force=True)
        assert set(adj["result"]) == base, (q, adj)
        assert adj["fallback"] is True, (q, adj)
        assert adj["applied"] is False, (q, adj)


@test
def test_today_evidence_helper_degrades_silently_on_a_broken_matrix():
    """services/today.py must never surface a junk cell as a verdict."""
    from services.today import _evidence_cell
    for matrix in MALFORMED_MATRICES:
        cell = _evidence_cell(matrix, "trending_bull", "momentum_rotation")
        assert cell is None or cell["verdict"] is None, (matrix, cell)
    good = {"strategies": {"momentum_rotation": {"cells": {
        "trending_bull": {"verdict": "confirmed", "periods": 42}}}}}
    cell = _evidence_cell(good, "trending_bull", "momentum_rotation")
    assert cell == {"verdict": "confirmed", "n": 42, "detail": None}, cell


# ──────────────────────────────────────────────────────────────────────────
# 6. The time stop defaults from the strategy
# ──────────────────────────────────────────────────────────────────────────
def _time_stop(rules):
    for r in rules:
        if isinstance(r, TimeStop):
            return r
    return None


@test
def test_build_exit_rules_time_stop_defaults_to_the_strategy_horizon():
    for sid, expected in (("mean_reversion", 10), ("momentum_rotation", 21),
                          ("dual_momentum", 63), ("volatility_breakout", 30)):
        assert strategy_horizon_days(sid) == expected, sid
        ts = _time_stop(build_exit_rules("fixed,time", strategy_id=sid))
        assert ts is not None and ts.max_bars == expected, (sid, ts)


@test
def test_build_exit_rules_time_stop_falls_back_to_twenty():
    assert DEFAULT_TIME_STOP_BARS == 20
    # No strategy at all.
    assert _time_stop(build_exit_rules("time")).max_bars == 20
    # A strategy with no machine-readable horizon (prose only).
    assert strategy_horizon_days("low_vol_trend") is None
    assert _time_stop(build_exit_rules("time", strategy_id="low_vol_trend")).max_bars == 20
    # An unknown id must not explode.
    assert strategy_horizon_days("no_such_strategy") is None
    assert _time_stop(build_exit_rules("time", strategy_id="no_such_strategy")).max_bars == 20


@test
def test_an_explicit_time_stop_bars_always_wins_over_the_horizon():
    ts = _time_stop(build_exit_rules("time", time_stop_bars=7, strategy_id="dual_momentum"))
    assert ts.max_bars == 7, ts
    # And every existing call site (which passes no strategy_id) is unchanged.
    assert _time_stop(build_exit_rules("fixed,time", atr_mult=2.5)).max_bars == 20


@test
def test_build_exit_rules_still_always_includes_a_hard_stop():
    from services.exits import FixedStopTarget
    for spec in (None, "", "time", ["trail"], "bogus"):
        rules = build_exit_rules(spec, strategy_id="mean_reversion")
        assert any(isinstance(r, FixedStopTarget) for r in rules), spec


# ──────────────────────────────────────────────────────────────────────────
def main_runner() -> int:
    passed = failed = 0
    for fn in _TESTS:
        try:
            fn()
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"ok    {fn.__name__}")
    print(f"\n{passed} passed, {failed} failed  ({len(_TESTS)} total)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main_runner())
