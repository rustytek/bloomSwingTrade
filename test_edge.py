"""
Pure-function, no-network tests for the evidence layer.

Covers:
  1. services/regime.py  — strategies_for_regime is UNCHANGED, and the evidence
     override can never hand back an empty playbook
  2. services/edge_matrix.py — classify_verdict (all four verdicts + the
     low-sample guard), Wilson intervals, cell construction, JSON safety
  3. services/scorecard.py — realized-vs-expected drift on synthetic
     ClosedTrade rows, the small-sample guard, and the UNAVAILABLE contract

Run:
    python test_edge.py

No network, no DB, no HTTP. Deterministic.
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import math
import sys
import time
import traceback
import types
from datetime import date

# ──────────────────────────────────────────────────────────────────────────
# Web-dependency stubs (same mechanism as test_passes.py — appended to the END
# of sys.meta_path so a real install always wins; inert when one is present).
# ──────────────────────────────────────────────────────────────────────────
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

from database.models import ClosedTrade, StockCache  # noqa: E402
from services import edge_matrix as em  # noqa: E402
from services import regime as rg  # noqa: E402
from services import scorecard as sc  # noqa: E402
from services.strategies import STRATEGIES  # noqa: E402


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


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


# ──────────────────────────────────────────────────────────────────────────
# Fakes
# ──────────────────────────────────────────────────────────────────────────
class FakeTrade:
    """Stands in for a ClosedTrade row. Only the attributes the code reads."""

    def __init__(self, **kw):
        defaults = dict(
            id=1, ticker="AAA", shares=10, avg_cost=100.0, exit_price=95.0,
            entry_date=None, exit_date=None, stop_loss=None, initial_stop=None,
            target=None, strategy=None, pnl=-50.0, pnl_pct=-5.0, r_multiple=None,
            notes=None, opened_at=None, closed_at=None,
        )
        defaults.update(kw)
        for k, v in defaults.items():
            setattr(self, k, v)


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *a, **k):
        return self

    def order_by(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def all(self):
        return list(self._rows)

    def first(self):
        return self._rows[0] if self._rows else None


class FakeDB:
    """Serves ClosedTrade rows; every other model comes back empty (which is
    exactly what makes the SPY-regime metric report UNAVAILABLE)."""

    def __init__(self, trades=None, stock_rows=None):
        self._trades = trades or []
        self._stock = stock_rows or []

    def query(self, model, *rest):
        if model is ClosedTrade:
            return _FakeQuery(self._trades)
        if model is StockCache:
            return _FakeQuery(self._stock)
        return _FakeQuery([])


def cell(n, avg, cum, ci_low=60.0, quadrant="trending_bull", q_value=0.01,
         disc_avg="same", hold_avg="same", seg_n=20):
    """A cell that, by default, clears the v4 gates (significant, and both
    segments agreeing in sign) — so each test varies exactly one thing."""
    return {
        "quadrant": quadrant, "label": "x", "n": n, "avg_period_return": avg,
        "win_rate": 55.0, "win_rate_ci_low": ci_low, "win_rate_ci_high": 90.0,
        "cum_return": cum, "confidence": em.cell_confidence(n), "q_value": q_value,
        "discovery": {"n": seg_n, "avg_period_return": avg if disc_avg == "same" else disc_avg},
        "holdout": {"n": seg_n, "avg_period_return": avg if hold_avg == "same" else hold_avg},
    }


def fake_matrix(verdict_by_strategy: dict[str, dict[str, str]], mode="trade_plan",
                overall: dict | None = None) -> dict:
    """{strategy: {quadrant: verdict}} -> a matrix shaped like build_edge_matrix's."""
    strategies = {}
    for sid, by_q in verdict_by_strategy.items():
        cells = {}
        for q in rg.QUADRANTS:
            c = cell(50, 0.5, 20.0, quadrant=q)
            c["tagged"] = q in getattr(STRATEGIES.get(sid), "regimes", []) if sid in STRATEGIES else False
            c["verdict"] = by_q.get(q, "unproven")
            cells[q] = c
        strategies[sid] = {
            "id": sid, "name": sid, "tagged_regimes": [], "cells": cells,
            "overall": (overall or {}).get(sid, {}), "error": None,
        }
    return {"method_version": em.METHOD_VERSION, "generated_at": "2026-01-01T00:00:00Z",
            "mode": mode, "strategies": strategies}


# ──────────────────────────────────────────────────────────────────────────
# 1. regime.py — the pinned contract must not move
# ──────────────────────────────────────────────────────────────────────────
@test
def test_strategies_for_regime_unchanged():
    """The hand-tag lookup must still be exactly the literal in strategies.py."""
    for q in rg.QUADRANTS:
        expected = {sid for sid, s in STRATEGIES.items() if q in getattr(s, "regimes", [])}
        assert rg.strategies_for_regime(q) == expected, q
        assert isinstance(rg.strategies_for_regime(q), set)
    assert rg.strategies_for_regime(None) == set(STRATEGIES)
    assert rg.strategies_for_regime("") == set(STRATEGIES)


@test
def test_with_evidence_no_matrix_is_identical():
    """No matrix -> byte-identical to today's behaviour, for every quadrant."""
    assert rg.EVIDENCE_OVERRIDE_ENABLED is False, "override must ship OFF"
    for q in list(rg.QUADRANTS) + [None]:
        assert rg.strategies_for_regime_with_evidence(q) == rg.strategies_for_regime(q), q
        assert rg.strategies_for_regime_with_evidence(q, None) == rg.strategies_for_regime(q), q
        assert rg.strategies_for_regime_with_evidence(q, {}) == rg.strategies_for_regime(q), q


@test
def test_override_off_ignores_a_valid_matrix():
    m = fake_matrix({sid: {q: "mis-tagged" for q in rg.QUADRANTS} for sid in STRATEGIES})
    for q in rg.QUADRANTS:
        assert rg.strategies_for_regime_with_evidence(q, m) == rg.strategies_for_regime(q)


@test
def test_empty_or_stale_matrix_never_empties_the_playbook():
    """SAFETY INVARIANT: no matrix state may blank the user's strategy set."""
    all_mis = fake_matrix({sid: {q: "mis-tagged" for q in rg.QUADRANTS} for sid in STRATEGIES})
    garbage = [
        None, {}, {"strategies": None}, {"strategies": {}},
        {"strategies": {"momentum_rotation": "not-a-dict"}},
        {"strategies": {"momentum_rotation": {"cells": None}}},
        {"strategies": {"momentum_rotation": {"cells": {"trending_bull": {"verdict": 7}}}}},
        all_mis,
        fake_matrix({"ghost_strategy_that_does_not_exist": {"trending_bull": "untagged-edge"}}),
    ]
    for q in rg.QUADRANTS:
        base = rg.strategies_for_regime(q)
        if not base:
            continue
        for m in garbage:
            got = rg.strategies_for_regime_with_evidence(q, m, force=True)
            assert got, f"empty strategy set for quadrant={q} matrix={str(m)[:60]}"
            if m is all_mis:
                # Every strategy demoted -> fall back to the hand tags, flagged.
                adj = rg.evidence_adjustment(q, m, force=True)
                assert adj["fallback"] is True
                assert set(adj["result"]) == base


@test
def test_evidence_override_excludes_and_promotes():
    q = "trending_bull"
    base = rg.strategies_for_regime(q)
    tagged_one = sorted(base)[0]
    untagged_one = sorted(set(STRATEGIES) - base)[0]
    m = fake_matrix({
        tagged_one: {q: "mis-tagged"},
        untagged_one: {q: "untagged-edge"},
    })
    adj = rg.evidence_adjustment(q, m, force=True)
    assert adj["applied"] is True and adj["fallback"] is False
    assert tagged_one in adj["excluded"]
    assert untagged_one in adj["added"]
    result = rg.strategies_for_regime_with_evidence(q, m, force=True)
    assert tagged_one not in result and untagged_one in result
    # Unproven cells change nothing.
    m2 = fake_matrix({tagged_one: {q: "unproven"}})
    assert rg.strategies_for_regime_with_evidence(q, m2, force=True) == base


@test
def test_set_evidence_override_toggles_and_restores():
    q = "trending_bull"
    base = rg.strategies_for_regime(q)
    tagged_one = sorted(base)[0]
    m = fake_matrix({tagged_one: {q: "mis-tagged"}})
    try:
        rg.set_evidence_override(True)
        assert tagged_one not in rg.strategies_for_regime_with_evidence(q, m)
        # The pinned function is untouched by the flag.
        assert rg.strategies_for_regime(q) == base
    finally:
        rg.set_evidence_override(False)
    assert rg.EVIDENCE_OVERRIDE_ENABLED is False
    assert rg.strategies_for_regime_with_evidence(q, m) == base


# ──────────────────────────────────────────────────────────────────────────
# 2. edge_matrix.py
# ──────────────────────────────────────────────────────────────────────────
@test
def test_verdict_confirmed():
    assert em.classify_verdict(cell(60, 0.8, 30.0), True) == "confirmed"


@test
def test_verdict_mis_tagged():
    assert em.classify_verdict(cell(60, -0.6, -18.0), True) == "mis-tagged"


@test
def test_verdict_untagged_edge():
    c = cell(40, 0.6, 22.0, ci_low=58.0)
    assert em.classify_verdict(c, False) == "untagged-edge"


@test
def test_verdict_unproven_low_sample():
    """THE critical guard: a LOSING cell with too few periods is 'unproven',
    never 'mis-tagged'. We do not strip a strategy out on noise."""
    losing_thin = cell(em.MIN_PERIODS_JUDGE - 1, -3.0, -25.0)
    assert em.classify_verdict(losing_thin, True) == "unproven"
    assert em.classify_verdict(losing_thin, False) == "unproven"
    # ...and a winning thin cell cannot be promoted either.
    winning_thin = cell(em.MIN_PERIODS_PROMOTE - 1, 2.0, 40.0, ci_low=80.0)
    assert em.classify_verdict(winning_thin, False) == "unproven"
    assert em.classify_verdict(cell(0, None, None), True) == "unproven"


@test
def test_verdict_edge_cases_are_conservative():
    # Mixed signs (positive mean, negative compounding) -> we can't tell.
    assert em.classify_verdict(cell(60, 0.4, -5.0), True) == "unproven"
    assert em.classify_verdict(cell(60, -0.4, 5.0), True) == "unproven"
    # Flat.
    assert em.classify_verdict(cell(60, 0.0, 0.0), True) == "unproven"
    # Untagged, positive but the Wilson lower bound is still a coin flip.
    assert em.classify_verdict(cell(60, 0.6, 22.0, ci_low=49.9), False) == "unproven"
    # Untagged, good CI but the mean is too small to matter.
    assert em.classify_verdict(cell(60, 0.1, 5.0, ci_low=70.0), False) == "unproven"
    # NaN must never become a confident verdict.
    assert em.classify_verdict(cell(60, float("nan"), 30.0), True) == "unproven"
    assert em.classify_verdict(cell(60, 0.8, float("inf")), True) == "unproven"


@test
def test_verdict_needs_multiple_testing_significance():
    """Gap 2: a positive cell that is not significant after the BH correction
    across the whole matrix is unproven — and so is one never tested (q None)."""
    for verdict_tagged, c in (
        (True, cell(60, 0.8, 30.0, q_value=0.30)),
        (True, cell(60, -0.6, -18.0, q_value=0.30)),
        (False, cell(40, 0.6, 22.0, ci_low=58.0, q_value=0.30)),
        (True, cell(60, 0.8, 30.0, q_value=None)),
    ):
        v, why = em.judge_cell(c, verdict_tagged)
        assert v == "unproven" and "noise" in why, (v, why)
    assert em.classify_verdict(cell(60, 0.8, 30.0, q_value=em.FDR_Q), True) == "confirmed"


@test
def test_verdict_needs_discovery_and_holdout_to_agree():
    # Holdout points the other way -> not confirmed, and not demoted either.
    v, why = em.judge_cell(cell(60, 0.8, 30.0, hold_avg=-0.2), True)
    assert v == "unproven" and "holdout" in why, (v, why)
    v, why = em.judge_cell(cell(60, -0.6, -18.0, disc_avg=0.1), True)
    assert v == "unproven" and "discovery" in why, (v, why)
    # A zero segment mean is not agreement.
    assert em.classify_verdict(cell(60, 0.8, 30.0, hold_avg=0.0), True) == "unproven"
    # A segment too thin to speak cannot confirm.
    v, why = em.judge_cell(cell(60, 0.8, 30.0, seg_n=em.MIN_SEGMENT_PERIODS - 1), True)
    assert v == "unproven" and "periods" in why, (v, why)
    # A cell without segments (built without dates) is never confident.
    bare = cell(60, 0.8, 30.0)
    bare.pop("discovery"); bare.pop("holdout")
    assert em.classify_verdict(bare, True) == "unproven"


@test
def test_cell_splits_discovery_and_holdout_by_date():
    rets = [1.0, 2.0, -1.0, 3.0]
    dates = ["2012-03-05", "2018-12-24", em.HOLDOUT_START, "2023-06-05"]
    c = em._cell_from_returns("trending_bull", rets, 0, dates)
    assert c["discovery"]["n"] == 2 and c["discovery"]["avg_period_return"] == 1.5, c
    assert c["holdout"]["n"] == 2 and c["holdout"]["avg_period_return"] == 1.0, c
    assert c["q_value"] is None and 0.0 <= c["p_value"] <= 1.0
    json.dumps(c, allow_nan=False)


@test
def test_apply_verdicts_corrects_across_the_whole_matrix():
    """One strong cell among many noise cells survives; the same cell's raw
    p-value is inflated by the number of cells judged (BH)."""
    strong = cell(60, 0.8, 30.0)
    strong.update({"p_value": 0.001, "q_value": None, "tagged": True})
    noise = []
    for i in range(9):
        c = cell(60, 0.3, 5.0)
        c.update({"p_value": 0.04 + i * 0.01, "q_value": None, "tagged": True})
        noise.append(c)
    thin = cell(5, 0.9, 5.0)
    thin.update({"p_value": 0.0001, "q_value": None, "tagged": True})
    strategies = {"s": {"cells": {"a": strong, "t": thin, **{f"n{i}": c for i, c in enumerate(noise)}}}}
    tested = em.apply_verdicts(strategies)
    assert tested == 10, tested                      # the thin cell is not a test
    assert thin["q_value"] is None and thin["verdict"] == "unproven"
    assert strong["q_value"] == 0.01 and strong["verdict"] == "confirmed", strong
    # 0.04 alone would pass at q=0.10; among ten tests it does not.
    assert noise[0]["q_value"] > em.FDR_Q and noise[0]["verdict"] == "unproven", noise[0]
    assert all("verdict_reason" in c for c in [strong, thin, *noise])


@test
def test_matrix_without_pre_holdout_history_confirms_nothing():
    real = em.run_walk_forward_backtest

    def fake_run(**kw):
        trades = [{"date": f"2023-{1 + i // 4:02d}-{1 + (i % 4) * 7:02d}", "quadrant": "trending_bull",
                   "holdings": [{"ticker": "X"}], "exits": 0, "period_return": 1.0 + (i % 3)}
                  for i in range(40)]
        return {"trades": trades, "metrics": {}, "data_quality": {}}

    em.run_walk_forward_backtest = fake_run
    try:
        m = em.build_edge_matrix(None, 996)
    finally:
        em.run_walk_forward_backtest = real
    ids = {c["id"] for c in m["caveats"]}
    assert "no_discovery_segment" in ids and "multiple_testing" in ids, ids
    verdicts = {c["verdict"] for s in m["strategies"].values() for c in s["cells"].values()}
    assert verdicts == {"unproven"}, verdicts
    assert m["tests_corrected_for"] >= 1 and m["holdout_start"] == em.HOLDOUT_START
    json.dumps(m, allow_nan=False)
    em.invalidate_cache(996)


@test
def test_stats_student_t_and_bh_match_reference_values():
    from services import stats
    assert approx(stats.t_two_sided_p(2.228, 10), 0.05, 1e-3)
    assert approx(stats.t_two_sided_p(2.0, 10), 0.0734, 1e-3)
    assert approx(stats.t_two_sided_p(1.96, 1e6), 0.05, 1e-3)
    assert stats.t_two_sided_p(0.0, 5) == 1.0
    q = stats.bh_adjust([0.01, 0.04, 0.03, 0.2])
    assert [round(x, 4) for x in q] == [0.04, 0.0533, 0.0533, 0.2], q


@test
def test_effective_n_shrinks_for_autocorrelation_and_never_grows():
    from services import stats
    trending = [1.0 + 0.1 * i for i in range(40)]            # strongly autocorrelated
    n_eff, rho = stats.effective_n(trending)
    assert rho > 0.5 and n_eff < 20, (n_eff, rho)
    alternating = [1.0, -1.0] * 20                            # negative autocorrelation
    n_eff, rho = stats.effective_n(alternating)
    assert rho == 0.0 and n_eff == 40, (n_eff, rho)


@test
def test_wilson_interval_hand_computed():
    """7 of 10 -> (39.68%, 89.22%), the standard published Wilson 95% values."""
    low, high = em.wilson_interval(7, 10)
    assert approx(low, 39.6809, 0.01), low
    assert approx(high, 89.2185, 0.01), high
    # 0 of 30: lower bound pinned at ~0, upper bound still 11.35% — a wide
    # interval is exactly what a small sample should produce.
    low0, high0 = em.wilson_interval(0, 30)
    assert abs(low0) < 0.01, low0
    assert approx(high0, 11.3512, 0.01), high0
    # 30 of 30 stays inside [0, 100] (the reason Wilson is used over normal).
    low1, high1 = em.wilson_interval(30, 30)
    assert 0.0 <= low1 <= 100.0 and high1 <= 100.0
    assert approx(high1, 100.0, 1e-9)
    # n == 0 -> maximally uninformative, never a crash.
    assert em.wilson_interval(0, 0) == (0.0, 100.0)
    # An interval always brackets the point estimate.
    lo, hi = em.wilson_interval(13, 20)
    assert lo < 65.0 < hi


@test
def test_cell_confidence_thresholds():
    assert em.cell_confidence(0) == "insufficient"
    assert em.cell_confidence(em.MIN_PERIODS_JUDGE - 1) == "insufficient"
    assert em.cell_confidence(em.MIN_PERIODS_JUDGE) == "low"
    assert em.cell_confidence(em.MIN_PERIODS_HIGH - 1) == "low"
    assert em.cell_confidence(em.MIN_PERIODS_HIGH) == "high"


@test
def test_cell_from_returns_math_and_json_safety():
    c = em._cell_from_returns("trending_bull", [1.0, -2.0, 3.0, 0.0])
    assert c["n"] == 4
    assert approx(c["avg_period_return"], 0.5, 1e-9)
    assert approx(c["win_rate"], 50.0, 1e-9)   # 2 of 4 strictly positive
    expected_cum = ((1.01 * 0.98 * 1.03 * 1.0) - 1) * 100
    assert approx(c["cum_return"], round(expected_cum, 2), 1e-9)
    empty = em._cell_from_returns("choppy_calm", [])
    assert empty["n"] == 0 and empty["confidence"] == "insufficient"
    # Starlette serializes with allow_nan=False — nothing here may be NaN/Inf.
    json.dumps([c, empty], allow_nan=False)


@test
def test_evidence_backed_regimes():
    m = fake_matrix({
        "momentum_rotation": {"trending_bull": "confirmed", "choppy_calm": "untagged-edge",
                              "trending_bear": "mis-tagged", "choppy_volatile": "unproven"},
    })
    earned = em.evidence_backed_regimes(m)
    assert earned["momentum_rotation"] == ["trending_bull", "choppy_calm"]
    assert em.evidence_backed_regimes(None) == {}
    assert em.evidence_backed_regimes({}) == {}


@test
def test_actionable_strategies_exclude_watchlist_only():
    ids = em.actionable_strategy_ids()
    assert "bear_reversal_watch" not in ids, "non-actionable strategy must not be backtested"
    assert "momentum_rotation" in ids
    assert len(ids) >= 5


@test
def test_matrix_cache_roundtrip_and_invalidate():
    em.invalidate_cache()
    assert em.get_cached_matrix(999) is None
    m = fake_matrix({"momentum_rotation": {}})
    em._cache[(999, "__latest__")] = (m, time.time())
    assert em.get_cached_matrix(999) is m
    # Stale entries expire rather than being served forever.
    em._cache[(999, "__latest__")] = (m, time.time() - em._TTL_SECONDS - 1)
    assert em.get_cached_matrix(999) is None
    em._cache[(999, "__latest__")] = (m, time.time())
    em.invalidate_cache(999)
    assert em.get_cached_matrix(999) is None


@test
def test_old_method_matrix_is_never_served():
    """A matrix built before METHOD_VERSION 2 scored cash periods as results —
    it must read as absent, not as evidence."""
    em.invalidate_cache()
    old = fake_matrix({"momentum_rotation": {}})
    old.pop("method_version")
    em._cache[(998, "__latest__")] = (old, time.time())
    assert em.get_cached_matrix(998) is None
    assert not em.is_current(old) and em.is_current(fake_matrix({}))


@test
def test_matrix_builds_without_spy_filter_and_excludes_idle_periods():
    """The SPY>200MA filter blocks every entry in the bear regimes, so the matrix
    must run with it OFF, and a period that held nothing must not be scored."""
    seen = {}
    real = em.run_walk_forward_backtest

    def fake_run(**kw):
        seen.update(kw)
        return {"trades": [
            {"quadrant": "trending_bear", "holdings": [], "exits": 0, "period_return": 0.0},
            {"quadrant": "trending_bear", "holdings": [], "exits": 0, "period_return": 0.0},
            {"quadrant": "trending_bear", "holdings": [{"ticker": "X"}], "exits": 0, "period_return": 2.0},
            # Exited during the period, empty at its end: that IS a measurement.
            {"quadrant": "trending_bear", "holdings": [], "exits": 1, "period_return": -1.0},
        ], "metrics": {}, "data_quality": {"test_first_date": "2022-01-03"}}

    em.run_walk_forward_backtest = fake_run
    try:
        m = em.build_edge_matrix(None, 997)
    finally:
        em.run_walk_forward_backtest = real
    assert seen.get("spy_regime") is False, seen
    c = next(iter(m["strategies"].values()))["cells"]["trending_bear"]
    assert c["n"] == 2 and c["idle_periods"] == 2, c
    assert c["avg_period_return"] == 0.5, c
    assert m["method_version"] == em.METHOD_VERSION and m["data_quality"]
    assert any(cv["id"] == "no_regime_filter" for cv in m["caveats"])
    em.invalidate_cache(997)


# ──────────────────────────────────────────────────────────────────────────
# 3. scorecard.py
# ──────────────────────────────────────────────────────────────────────────
def _seed_matrix(user_id: int, expectancy: float | None, mode="trade_plan"):
    em.invalidate_cache(user_id)
    m = fake_matrix({"momentum_rotation": {}}, mode=mode,
                    overall={"momentum_rotation": {"expectancy_r": expectancy}})
    em._cache[(user_id, "__latest__")] = (m, time.time())
    return m


@test
def test_scorecard_drift_underperforming():
    _seed_matrix(1, 1.0)
    trades = [FakeTrade(id=i, strategy="momentum_rotation", pnl=(10 if i % 2 else -10),
                        pnl_pct=1.0, r_multiple=(0.6 if i % 2 else -0.2))
              for i in range(12)]
    out = sc.build_scorecard(FakeDB(trades), 1)
    row = out["strategies"][0]
    assert row["trades"] == 12 and row["r_sample"] == 12
    assert approx(row["avg_r"], 0.2, 1e-9), row["avg_r"]
    assert approx(row["expected_r"], 1.0, 1e-9)
    assert approx(row["drift"], -0.8, 1e-9)
    assert row["recommendation"]["verdict"] == "underperforming"
    json.dumps(out, allow_nan=False)


@test
def test_scorecard_small_sample_is_never_broken():
    _seed_matrix(2, 1.0)
    trades = [FakeTrade(id=i, strategy="momentum_rotation", pnl=-100.0, pnl_pct=-9.0,
                        r_multiple=-1.0) for i in range(4)]
    out = sc.build_scorecard(FakeDB(trades), 2)
    row = out["strategies"][0]
    assert row["trades"] == 4
    assert approx(row["drift"], -2.0, 1e-9)  # the number is still reported...
    assert row["recommendation"]["verdict"] == "monitoring"  # ...but not judged
    assert "too few to judge" in row["recommendation"]["text"]


@test
def test_scorecard_thin_r_sample_is_not_judged():
    _seed_matrix(3, 1.0)
    # 12 trades, but only 3 carry an r_multiple.
    trades = [FakeTrade(id=i, strategy="momentum_rotation", pnl=-100.0, pnl_pct=-9.0,
                        r_multiple=(-1.0 if i < 3 else None)) for i in range(12)]
    out = sc.build_scorecard(FakeDB(trades), 3)
    row = out["strategies"][0]
    assert row["r_sample"] == 3 and row["trades"] == 12
    assert row["recommendation"]["verdict"] == "monitoring"


@test
def test_scorecard_scratch_split_matches_journal_convention():
    _seed_matrix(4, None)
    trades = [
        FakeTrade(id=1, strategy="s", pnl=10.0, pnl_pct=1.0),
        FakeTrade(id=2, strategy="s", pnl=-10.0, pnl_pct=-1.0),
        FakeTrade(id=3, strategy="s", pnl=0.0, pnl_pct=0.0),
    ]
    out = sc.build_scorecard(FakeDB(trades), 4)
    row = out["strategies"][0]
    assert (row["wins"], row["losses"], row["scratch"]) == (1, 1, 1)
    assert approx(row["win_rate"], 50.0, 1e-9)  # scratch excluded from denominator
    assert row["expected_r"] is None
    assert row["recommendation"]["verdict"] in ("monitoring", "no_benchmark")


@test
def test_scorecard_no_matrix_has_no_fabricated_expectation():
    em.invalidate_cache()
    trades = [FakeTrade(id=i, strategy="momentum_rotation", pnl=10.0, pnl_pct=1.0,
                        r_multiple=0.5) for i in range(20)]
    out = sc.build_scorecard(FakeDB(trades), 77)
    row = out["strategies"][0]
    assert row["expected_r"] is None and row["drift"] is None
    assert row["recommendation"]["verdict"] == "no_benchmark"
    assert out["edge_matrix"]["available"] is False


@test
def test_scorecard_rotation_mode_gives_no_pseudo_r():
    """A rotation-mode matrix has no per-trade R. It must report None, not a
    period return dressed up as an R."""
    _seed_matrix(5, None, mode="rotation")
    trades = [FakeTrade(id=i, strategy="momentum_rotation", pnl=10.0, pnl_pct=1.0,
                        r_multiple=0.5) for i in range(20)]
    row = sc.build_scorecard(FakeDB(trades), 5)["strategies"][0]
    assert row["expected_r"] is None
    assert "trade_plan" in row["expected_r_source"]


# ── execution quality ───────────────────────────────────────────────────────
@test
def test_execution_chasing_is_unavailable_not_zero():
    """The planned entry price is not persisted anywhere, so this metric must
    come back explicitly UNAVAILABLE — never 0, never estimated."""
    trades = [FakeTrade(id=1, strategy="momentum_rotation", avg_cost=100.0)]
    out = sc.execution_quality(FakeDB(trades), 9)
    names = {m["metric"] for m in out["unavailable"]}
    assert "entry_chasing" in names
    m = next(m for m in out["unavailable"] if m["metric"] == "entry_chasing")
    assert m["available"] is False and m["status"] == "UNAVAILABLE"
    assert m["value"] is None and m["r_cost"] is None
    assert "planned_entry" in m["needs"]
    json.dumps(out, allow_nan=False)


@test
def test_execution_stop_loosening_unavailable_without_initial_stop():
    trades = [FakeTrade(id=1, strategy="s", initial_stop=None, stop_loss=90.0)]
    out = sc.execution_quality(FakeDB(trades), 10)
    m = next(m for m in out["unavailable"] if m["metric"] == "stop_loosening")
    assert m["status"] == "UNAVAILABLE" and m["r_cost"] is None
    assert m["observed"]["trades_with_both_stops"] == 0


@test
def test_execution_stop_loosening_computes_r_cost():
    trades = [
        # Stop widened 95 -> 90 and exited at 88: avoidable loss (95-88)/(100-95) = 1.4R
        FakeTrade(id=1, strategy="s", avg_cost=100.0, initial_stop=95.0,
                  stop_loss=90.0, exit_price=88.0),
        # Stop trailed UP — correct behaviour, not an offender.
        FakeTrade(id=2, strategy="s", avg_cost=100.0, initial_stop=95.0,
                  stop_loss=98.0, exit_price=110.0),
        # Widened but exited above the initial stop: an offender with no
        # realized cost — reported as None, not as a made-up number.
        FakeTrade(id=3, strategy="s", avg_cost=100.0, initial_stop=95.0,
                  stop_loss=90.0, exit_price=101.0),
    ]
    out = sc.execution_quality(FakeDB(trades), 11)
    m = next(m for m in out["ranked"] if m["metric"] == "stop_loosening")
    assert m["available"] is True
    assert m["value"] == 2, m["offenders"]
    assert approx(m["r_cost"], 1.4, 1e-9), m["r_cost"]
    no_cost = next(o for o in m["offenders"] if o["id"] == 3)
    assert no_cost["r_cost"] is None and no_cost["r_cost_note"]


@test
def test_execution_horizon_is_live_now_that_strategies_declare_horizon_days():
    """`Strategy.horizon_days` was added (momentum_rotation = 21 trading days),
    which switches this metric ON. It used to be UNAVAILABLE because
    `details["horizon"]` was prose and parsing an int out of English is a guess.

    A 59-CALENDAR-day hold is ~41 trading days, well past a 21-trading-day
    horizon, so this trade must be flagged as an offender."""
    trades = [FakeTrade(id=1, strategy="momentum_rotation",
                        entry_date=date(2025, 1, 2), exit_date=date(2025, 3, 2))]
    out = sc.execution_quality(FakeDB(trades), 12)
    assert not any(m["metric"] == "overstayed_horizon" for m in out["unavailable"]), \
        "horizon_days is declared now — the metric must no longer report UNAVAILABLE"
    m = next(m for m in out["ranked"] if m["metric"] == "overstayed_horizon")
    assert m["available"] is True
    assert m["value"] == 1, m
    off = m["offenders"][0]
    assert off["days_held"] == 59 and off["horizon_days"] == 21, off
    # No r_multiple on the trade -> no invented cost.
    assert off["r_cost"] is None, off


@test
def test_execution_horizon_stays_unavailable_for_open_ended_strategies():
    """low_vol_trend is deliberately horizon_days=None ("hold while the trend
    persists"). A strategy with no time-based exit must NOT be graded against
    an invented deadline — it simply contributes no offenders."""
    from services.strategies import STRATEGIES
    assert STRATEGIES["low_vol_trend"].horizon_days is None
    trades = [FakeTrade(id=1, strategy="low_vol_trend",
                        entry_date=date(2025, 1, 2), exit_date=date(2025, 12, 1))]
    out = sc.execution_quality(FakeDB(trades), 12)
    flagged = [m for m in out["ranked"] if m["metric"] == "overstayed_horizon"]
    assert not flagged or flagged[0]["value"] == 0, flagged


@test
def test_every_actionable_strategy_declares_a_usable_horizon():
    """Guard the contract: a new actionable strategy that forgets horizon_days
    silently drops out of the overstayed-horizon check rather than failing loudly."""
    from services.strategies import STRATEGIES
    missing = [
        sid for sid, s in STRATEGIES.items()
        if s.actionable and s.horizon_days is None and sid != "low_vol_trend"
    ]
    assert not missing, f"actionable strategies with no horizon_days: {missing}"
    for sid, s in STRATEGIES.items():
        if s.horizon_days is not None:
            assert isinstance(s.horizon_days, int) and s.horizon_days > 0, sid


@test
def test_execution_off_regime_unavailable_without_spy_history():
    trades = [FakeTrade(id=1, strategy="momentum_rotation", entry_date=date(2025, 1, 2))]
    out = sc.execution_quality(FakeDB(trades), 13)
    m = next(m for m in out["unavailable"] if m["metric"] == "off_regime_entries")
    assert m["status"] == "UNAVAILABLE" and m["r_cost"] is None
    assert "SPY" in m["needs"]


@test
def test_execution_quality_ranks_by_r_cost_and_is_json_safe():
    trades = [
        FakeTrade(id=1, strategy="s", avg_cost=100.0, initial_stop=95.0,
                  stop_loss=90.0, exit_price=88.0),
    ]
    out = sc.execution_quality(FakeDB(trades), 14)
    costs = [m.get("r_cost") or 0.0 for m in out["ranked"]]
    assert costs == sorted(costs, reverse=True)
    assert out["total_r_cost"] >= 1.4
    assert out["trades_examined"] == 1
    assert all(m["available"] is False for m in out["unavailable"])
    assert "not estimated" in out["honesty_note"].lower() or \
           "NOT zero" in out["honesty_note"]
    json.dumps(out, allow_nan=False)


@test
def test_numeric_guards_reject_nan():
    assert sc._num(float("nan")) is None
    assert sc._num(float("inf")) is None
    assert sc._num(None) is None
    assert sc._num("abc") is None
    assert sc._round(float("nan"), 2, 0.0) == 0.0
    assert em._num(float("nan")) is None
    assert em._round(float("-inf"), 2) is None


@test
def test_universe_caveat_is_disclosed():
    from services.universe import UNIVERSE_AS_OF, UNIVERSE_CAVEAT
    assert "survived" in UNIVERSE_CAVEAT
    assert "point-in-time" in UNIVERSE_CAVEAT
    assert UNIVERSE_AS_OF
    out = sc.build_scorecard(FakeDB([]), 99)
    assert UNIVERSE_CAVEAT in out["caveats"]


# ──────────────────────────────────────────────────────────────────────────
# 4. Gap 2b — Sharpe inference, trial log, Deflated Sharpe
# ──────────────────────────────────────────────────────────────────────────
def _equity(returns, start=100.0):
    eq, v = [{"date": "2020-01-06", "value": start}], start
    for i, r in enumerate(returns):
        v *= 1 + r
        eq.append({"date": f"2020-{1 + (i // 4) % 12:02d}-{6 + (i % 4) * 5:02d}", "value": v})
    return eq


def _trial_db():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from database.db import Base
    import database.models  # noqa: F401 — registers the tables
    engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def _fake_result(strategy, sharpe_annual, params):
    return {
        "mode": "rotation", "parameters": {"strategy": strategy, **params},
        "equity": [{"date": "2021-01-04", "value": 1.0}, {"date": "2025-12-29", "value": 2.0}],
        "metrics": {"cagr": 10.0},
        "sharpe_inference": {"periods": 250, "periods_per_year": 50.4,
                             "sharpe_period": sharpe_annual / math.sqrt(50.4),
                             "sharpe_annual": sharpe_annual, "skew": 0.0, "kurtosis": 3.0},
        "caveats": [],
    }


@test
def test_sharpe_inference_ci_contains_point_and_widens_with_fat_tails():
    from services.backtest import sharpe_inference
    import random
    rng = random.Random(7)
    calm = [0.004 + rng.gauss(0, 0.02) for _ in range(200)]
    si = sharpe_inference(_equity(calm), 50.4)
    assert si["ci95_low"] < si["sharpe_annual"] < si["ci95_high"], si
    assert 0.0 <= si["psr_vs_zero"] <= 1.0
    # Same Sharpe, fatter tails or negative skew -> a larger standard error.
    from services.stats import sharpe_se
    base = sharpe_se(0.15, 200, 0.0, 3.0)
    assert sharpe_se(0.15, 200, 0.0, 9.0) > base
    assert sharpe_se(0.15, 200, -2.0, 3.0) > base
    assert sharpe_inference(_equity([0.01]), 50.4) is None
    json.dumps(si, allow_nan=False)


@test
def test_deflated_sharpe_falls_as_trials_grow():
    from services import stats
    sr, n = 0.15, 250
    one = stats.deflated_sharpe(sr, n, 0.0, 3.0, 1, 0.0)
    assert approx(one, stats.probabilistic_sharpe(sr, 0.0, n, 0.0, 3.0), 1e-12)
    ten = stats.deflated_sharpe(sr, n, 0.0, 3.0, 10, 0.01)
    hundred = stats.deflated_sharpe(sr, n, 0.0, 3.0, 100, 0.01)
    assert one > ten > hundred, (one, ten, hundred)
    assert approx(stats.expected_max_sharpe(10, 1.0), 1.5746, 1e-3)


@test
def test_trial_log_counts_distinct_configurations_only():
    from services import trial_log
    db = _trial_db()
    r1 = trial_log.annotate(db, 1, _fake_result("momentum_rotation", 1.0, {"top_n": 5}))
    assert r1["selection_bias"]["trials"] == 1, r1["selection_bias"]
    # Re-running the identical configuration is NOT a new trial.
    r1b = trial_log.annotate(db, 1, _fake_result("momentum_rotation", 1.0, {"top_n": 5}))
    assert r1b["selection_bias"]["trials"] == 1
    # A different window/parameter IS one — and it raises the bar.
    r2 = trial_log.annotate(db, 1, _fake_result("momentum_rotation", 0.2, {"top_n": 3}))
    r3 = trial_log.annotate(db, 1, _fake_result("momentum_rotation", 1.0, {"top_n": 7}))
    assert r3["selection_bias"]["trials"] == 3
    assert r3["selection_bias"]["deflated_sharpe"] < r1["selection_bias"]["deflated_sharpe"]
    assert r3["selection_bias"]["luck_hurdle_sharpe_annual"] > 0
    # Other strategies and other users are separate searches.
    other = trial_log.annotate(db, 1, _fake_result("pullback_50ma", 1.0, {"top_n": 5}))
    assert other["selection_bias"]["trials"] == 1
    assert trial_log.annotate(db, 2, _fake_result("momentum_rotation", 1.0, {"top_n": 5}))["selection_bias"]["trials"] == 1
    rows = trial_log.list_trials(db, 1, "momentum_rotation")
    assert len(rows) == 3 and max(r["runs"] for r in rows) == 2, rows
    json.dumps(r3["selection_bias"], allow_nan=False)


@test
def test_trial_log_flags_luck_and_never_breaks_a_finished_run():
    from services import trial_log
    db = _trial_db()
    weak = trial_log.annotate(db, 1, _fake_result("breakout_volume", 0.05, {"top_n": 5}))
    assert weak["selection_bias"]["label"] != "likely_real"
    assert any(c["id"] == "selection_bias" for c in weak["caveats"])
    # No Sharpe (e.g. not enough data): nothing recorded, result untouched.
    bare = {"parameters": {"strategy": "breakout_volume"}, "sharpe_inference": None}
    assert "selection_bias" not in trial_log.annotate(db, 1, bare)

    class Broken:
        def query(self, *a, **k):
            raise RuntimeError("db is down")

        def rollback(self):
            pass

    out = trial_log.annotate(Broken(), 1, _fake_result("breakout_volume", 1.0, {}))
    assert "error" in out["selection_bias"] and out["metrics"]["cagr"] == 10.0


# ──────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────
def main() -> int:
    passed = failed = 0
    failures = []
    for fn in _TESTS:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            failures.append((fn.__name__, exc, traceback.format_exc()))
            print(f"FAIL  {fn.__name__}: {exc}")
        else:
            passed += 1
            print(f"PASS  {fn.__name__}")

    print("\n" + "=" * 56)
    print(f"  {passed} passed, {failed} failed, {len(_TESTS)} total")
    print("=" * 56)
    if failures:
        print("\n--- failure tracebacks ---")
        for name, _exc, tb in failures:
            print(f"\n[{name}]\n{tb}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
