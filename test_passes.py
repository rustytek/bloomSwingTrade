"""
Pure-function, no-network smoke tests for SwingTrader core logic.

Covers:
  1. services/indicators.py::calc_atr  (Wilder ATR, None-padding)
  2. services/trade_plan.py::build_trade_plan  (sizing, stop/target, max cap)
  3. services/strategies.py::STRATEGIES  (candidate + scan for all three)
  4. ClosedTrade R-multiple math (formula mirrored from api/portfolio.py)

Run with the Python 3.14 venv (system 3.9 won't parse `X | None` runtime annots):
    .venv-test/Scripts/python.exe test_passes.py

No network, no DB, no HTTP. Deterministic (randomness seeded).
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import math
import random
import sys
import traceback
import types

# ──────────────────────────────────────────────────────────────────────────
# Web-dependency stubs
# ──────────────────────────────────────────────────────────────────────────
# Some pure helpers under test live in api/*.py, which import fastapi (and,
# transitively, jose/passlib through auth.deps). This runner is deliberately
# dependency-light, so those packages may not be installed. Install permissive
# stand-ins for them — appended to the END of sys.meta_path, so a real install
# always wins. The stubs only have to survive import time: every function these
# tests actually call is plain arithmetic.
_STUBBED_PACKAGES = ("fastapi", "jose", "passlib", "bcrypt")


class _Stub:
    """Stands in for any attribute of a stubbed module."""

    def __init__(self, name: str):
        self._name = name

    def __call__(self, *args, **kwargs):
        # Decorator use: @router.get("/x") -> the inner call gets the function
        # and must return it unchanged so the module still defines real symbols.
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
        module.__path__ = []          # allow `import pkg.submodule`
        return module

    def exec_module(self, module):
        pass


sys.meta_path.append(_StubFinder())

from services.indicators import calc_atr
from services.trade_plan import build_trade_plan
from services.strategies import STRATEGIES, Setup


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
# Synthetic bar builders (deterministic)
# ──────────────────────────────────────────────────────────────────────────
def _bar(o, h, l, c, v=1_000_000):
    return {"open": o, "high": h, "low": l, "close": c, "vol": v}


def make_uptrend_bars(n=60, start=100.0, step=0.5, seed=1):
    """Smooth, low-noise rising series with sane OHLC and volume."""
    rng = random.Random(seed)
    bars = []
    price = start
    for _ in range(n):
        price += step
        noise = rng.uniform(-0.05, 0.05)
        close = price + noise
        open_ = close - step * 0.5
        high = max(open_, close) + abs(noise) + 0.2
        low = min(open_, close) - abs(noise) - 0.2
        bars.append(_bar(open_, high, low, close))
    return bars


# ──────────────────────────────────────────────────────────────────────────
# 1. calc_atr
# ──────────────────────────────────────────────────────────────────────────
@test
def test_atr_padding_and_alignment():
    n = 40
    period = 14
    highs = [10 + i * 0.3 + 0.5 for i in range(n)]
    lows = [10 + i * 0.3 - 0.5 for i in range(n)]
    closes = [10 + i * 0.3 for i in range(n)]
    atr = calc_atr(highs, lows, closes, period)

    assert len(atr) == n, f"length {len(atr)} != input {n}"
    # Precise None-padding: indices 0..period-1 None, index `period` first float.
    none_count = sum(1 for v in atr[:period] if v is None)
    assert none_count == period, f"expected {period} leading None, got {none_count}"
    assert all(v is None for v in atr[:period]), "padding region must be all None"
    assert atr[period] is not None, "first ATR must be at index == period"
    assert isinstance(atr[period], float)
    assert isinstance(atr[-1], float) and atr[-1] > 0, "trailing ATR must be positive float"


@test
def test_atr_constant_series_is_zero():
    n = 30
    val = 50.0
    highs = [val] * n
    lows = [val] * n
    closes = [val] * n
    atr = calc_atr(highs, lows, closes, 14)
    assert atr[-1] is not None
    assert approx(atr[-1], 0.0, tol=1e-9), f"constant series ATR should be ~0, got {atr[-1]}"


@test
def test_atr_wilder_seed_matches_simple_average():
    """First ATR value (at index == period) is the simple mean of the first
    `period` true ranges. Use a tiny hand-computable series with period=3."""
    period = 3
    # Construct so true ranges are easy: TR_i = max(H-L, |H-prevC|, |L-prevC|)
    closes = [10.0, 11.0, 10.5, 12.0, 11.5, 13.0]
    highs = [10.5, 11.5, 11.0, 12.5, 12.0, 13.5]
    lows = [9.5, 10.5, 10.0, 11.0, 11.0, 12.5]
    atr = calc_atr(highs, lows, closes, period)

    # Recompute true ranges (i from 1..n-1) by hand via the same formula
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    seed = sum(trs[:period]) / period
    assert atr[period] is not None
    assert approx(atr[period], seed, tol=1e-9), (
        f"Wilder seed mismatch: got {atr[period]} expected {seed}"
    )
    # Length / padding sanity on this tiny series too
    assert len(atr) == len(closes)
    assert all(v is None for v in atr[:period])


@test
def test_atr_too_short_returns_all_none():
    atr = calc_atr([1, 2, 3], [0, 1, 2], [0.5, 1.5, 2.5], period=14)
    assert len(atr) == 3
    assert all(v is None for v in atr)


# ──────────────────────────────────────────────────────────────────────────
# 2. build_trade_plan
# ──────────────────────────────────────────────────────────────────────────
REQUIRED_PLAN_KEYS = {
    "entry", "stop", "target", "shares", "risk_dollars",
    "position_value", "position_pct", "capped_by_max_position",
    "entry_zone", "r_multiple",
}


@test
def test_trade_plan_shape_and_ordering():
    bars = make_uptrend_bars(60, seed=7)
    plan = build_trade_plan(bars, account_size=100_000, risk_pct=1.0)
    assert plan is not None, "plan should be produced for a valid 60-bar series"
    missing = REQUIRED_PLAN_KEYS - set(plan)
    assert not missing, f"missing keys: {missing}"
    assert plan["stop"] < plan["entry"] < plan["target"], (
        f"ordering wrong: stop={plan['stop']} entry={plan['entry']} target={plan['target']}"
    )
    assert isinstance(plan["shares"], int) and plan["shares"] >= 0
    # entry_zone brackets entry
    lo, hi = plan["entry_zone"]
    assert lo <= plan["entry"] <= hi
    assert plan["r_multiple"] == 2.0


@test
def test_trade_plan_risk_budget_not_capped():
    """When NOT capped, risk_dollars == shares*risk_per_share, must be <= budget
    and within one share's risk of the budget (shares is floored)."""
    bars = make_uptrend_bars(60, seed=3)
    account = 100_000
    risk_pct = 1.0
    # Lift the position cap so the floored risk-budget path is what's exercised
    # (with a tight ATR stop, risk-per-share is a small fraction of price, so the
    # share count from a 1% risk budget can otherwise brush the default 25% cap).
    plan = build_trade_plan(
        bars, account_size=account, risk_pct=risk_pct, max_position_pct=100.0
    )
    assert plan is not None
    assert plan["capped_by_max_position"] is False, "should not be capped at 100% max position"
    budget = account * risk_pct / 100.0  # 1000
    risk_per_share = plan["entry"] - plan["stop"]
    assert plan["risk_dollars"] <= budget + 1e-6, (
        f"risk_dollars {plan['risk_dollars']} exceeds budget {budget}"
    )
    # within one share's worth of risk of the budget (floor effect)
    assert budget - plan["risk_dollars"] <= risk_per_share + 1e-6, (
        f"risk_dollars {plan['risk_dollars']} more than one share below budget {budget}"
    )


@test
def test_trade_plan_max_position_cap_triggers():
    """Tiny account so the full risk-budget share count would blow past the
    position-size cap -> capped flag set, position_value within the cap."""
    bars = make_uptrend_bars(60, seed=11)
    account = 1_000          # small
    risk_pct = 5.0           # would want a big position relative to account
    max_pct = 25.0
    plan = build_trade_plan(
        bars, account_size=account, risk_pct=risk_pct, max_position_pct=max_pct
    )
    assert plan is not None
    assert plan["capped_by_max_position"] is True, "expected the max-position cap to trigger"
    cap_value = account * max_pct / 100.0
    assert plan["position_value"] <= cap_value + 1e-6, (
        f"position_value {plan['position_value']} exceeds cap {cap_value}"
    )


@test
def test_trade_plan_rejects_bad_inputs():
    bars = make_uptrend_bars(60)
    assert build_trade_plan([], 100_000, 1.0) is None, "empty bars -> None"
    # too short to compute ATR (need > atr_period bars)
    assert build_trade_plan(make_uptrend_bars(5), 100_000, 1.0) is None, "too-short -> None"
    assert build_trade_plan(bars, 100_000, 0) is None, "risk_pct<=0 -> None"
    assert build_trade_plan(bars, 100_000, -1.0) is None, "negative risk_pct -> None"
    assert build_trade_plan(bars, 0, 1.0) is None, "account<=0 -> None"


# ──────────────────────────────────────────────────────────────────────────
# 3. strategies
# ──────────────────────────────────────────────────────────────────────────
def _assert_setup_like(setup, strat_id):
    assert setup is not None, f"{strat_id}.scan returned None on triggering series"
    assert isinstance(setup, Setup)
    assert hasattr(setup, "state") and setup.state in {"triggered", "forming"}
    assert hasattr(setup, "reasons") and isinstance(setup.reasons, list)
    assert hasattr(setup, "score") and isinstance(setup.score, (int, float))
    d = setup.to_dict()
    assert d["strategy"] == strat_id
    assert d["state"] in {"triggered", "forming"}
    assert isinstance(d["score"], (int, float))


def _assert_candidate(result, strat_id):
    assert result is not None, f"{strat_id}.candidate returned None on triggering series"
    assert isinstance(result.get("score"), (int, float)), f"{strat_id}: score must be numeric"
    assert result.get("state") in {"triggered", "forming"}, f"{strat_id}: bad state"


# ---- momentum_rotation: smooth uptrend above 100MA, no >15% single-day move ----
def make_momentum_series(n=280, start=50.0, daily=0.004, seed=21):
    rng = random.Random(seed)
    bars = []
    price = start
    for _ in range(n):
        price *= (1 + daily + rng.uniform(-0.003, 0.003))  # smooth, well under 15%
        close = price
        open_ = close * (1 - daily * 0.5)
        high = max(open_, close) * 1.003
        low = min(open_, close) * 0.997
        bars.append(_bar(open_, high, low, close, v=2_000_000))
    return bars


@test
def test_momentum_rotation_triggers():
    bars = make_momentum_series()
    strat = STRATEGIES["momentum_rotation"]
    res = strat.candidate(bars, len(bars) - 1)
    _assert_candidate(res, "momentum_rotation")
    assert res["score"] > 0
    setup = strat.scan("TEST", bars, {})
    _assert_setup_like(setup, "momentum_rotation")


# ---- pullback_50ma: uptrend (close>MA200, MA50>MA200), dip to ~50MA, close > prior high ----
def make_pullback_series(seed=33):
    """Long uptrend establishing MA50>MA200 and close>MA200, then a pullback that
    tags the 50MA, then a confirmation bar closing above the prior bar's high."""
    rng = random.Random(seed)
    bars = []
    price = 50.0
    # 250 bars of steady uptrend
    for _ in range(250):
        price *= (1 + 0.0035 + rng.uniform(-0.002, 0.002))
        close = price
        open_ = close * 0.999
        high = close * 1.004
        low = close * 0.996
        bars.append(_bar(open_, high, low, close, v=1_500_000))
    # Pullback: a few down bars dropping toward the 50-day MA region
    for _ in range(8):
        price *= (1 - 0.012)
        close = price
        open_ = close * 1.004
        high = close * 1.006
        low = close * 0.985          # dip low tags the rising 50MA
        bars.append(_bar(open_, high, low, close, v=1_400_000))
    # Confirmation bar: strong up close above the prior bar's high
    prev_high = bars[-1]["high"]
    close = prev_high * 1.02
    bars.append(_bar(prev_high, close * 1.005, bars[-1]["close"] * 0.999, close, v=2_500_000))
    return bars


@test
def test_pullback_50ma_triggers():
    bars = make_pullback_series()
    strat = STRATEGIES["pullback_50ma"]
    res = strat.candidate(bars, len(bars) - 1)
    _assert_candidate(res, "pullback_50ma")
    setup = strat.scan("TEST", bars, {})
    _assert_setup_like(setup, "pullback_50ma")


# ---- breakout_volume: new 60-day high, vol>=1.5x 20d avg, close>MA50>MA200, 52w>=70 ----
def make_breakout_series(seed=44):
    rng = random.Random(seed)
    bars = []
    price = 40.0
    # Long, gently rising base to set MA50>MA200 and a 52w range
    for _ in range(260):
        price *= (1 + 0.0025 + rng.uniform(-0.0015, 0.0015))
        close = price
        open_ = close * 0.999
        high = close * 1.005
        low = close * 0.995
        bars.append(_bar(open_, high, low, close, v=1_000_000))
    # Breakout bar: new 60-day high on a big volume spike
    prior_high = max(b["high"] for b in bars[-61:-1])
    close = prior_high * 1.03
    bars.append(_bar(prior_high * 1.001, close * 1.002, prior_high, close, v=3_000_000))
    return bars


@test
def test_breakout_volume_triggers():
    bars = make_breakout_series()
    strat = STRATEGIES["breakout_volume"]
    res = strat.candidate(bars, len(bars) - 1)
    _assert_candidate(res, "breakout_volume")
    assert res["state"] == "triggered", f"expected triggered, got {res['state']}"
    setup = strat.scan("TEST", bars, {})
    _assert_setup_like(setup, "breakout_volume")


# ---- mean_reversion: uptrend (close>MA200), sharp washout driving RSI(2) < 10 ----
def make_mean_reversion_series(seed=77):
    """Long uptrend keeping close>MA200, then a sharp multi-day selloff that
    crushes RSI(2) and stretches price below the lower Bollinger Band."""
    rng = random.Random(seed)
    bars = []
    price = 50.0
    # 255 bars of steady uptrend so MA200 sits well below price
    for _ in range(255):
        price *= (1 + 0.0035 + rng.uniform(-0.002, 0.002))
        close = price
        bars.append(_bar(close * 0.999, close * 1.004, close * 0.996, close, v=1_500_000))
    # Washout: five straight down days (~-1.5%/day) — RSI(2) pins near 0,
    # but price stays comfortably above the 200-day MA
    for _ in range(5):
        price *= (1 - 0.015)
        close = price
        bars.append(_bar(close * 1.006, close * 1.008, close * 0.995, close, v=1_800_000))
    return bars


@test
def test_mean_reversion_triggers():
    bars = make_mean_reversion_series()
    strat = STRATEGIES["mean_reversion"]
    res = strat.candidate(bars, len(bars) - 1)
    _assert_candidate(res, "mean_reversion")
    assert res["state"] == "triggered", f"expected triggered, got {res['state']}"
    assert res["rsi"] < 10, f"expected RSI(2) < 10, got {res['rsi']}"
    assert res["score"] > 0
    setup = strat.scan("TEST", bars, {})
    _assert_setup_like(setup, "mean_reversion")


EXPECTED_STRATEGY_IDS = {
    "momentum_rotation", "pullback_50ma", "breakout_volume", "mean_reversion",
    "dual_momentum", "volatility_breakout", "sector_rotation",
    "bear_reversal_watch", "low_vol_trend",
}


@test
def test_strategies_registry_keys():
    assert set(STRATEGIES) == EXPECTED_STRATEGY_IDS, (
        f"registry drift: extra={set(STRATEGIES) - EXPECTED_STRATEGY_IDS} "
        f"missing={EXPECTED_STRATEGY_IDS - set(STRATEGIES)}"
    )
    # The dict key must be the Strategy's own .id — a mismatch silently breaks
    # STRATEGIES[setup.strategy] lookups in today.py / backtest.py.
    for key, strat in STRATEGIES.items():
        assert strat.id == key, f"registry key {key!r} != strategy.id {strat.id!r}"


@test
def test_every_strategy_declares_regimes_and_actionable():
    from services.regime import QUADRANTS
    for sid, strat in STRATEGIES.items():
        assert isinstance(strat.regimes, list) and strat.regimes, (
            f"{sid}: must declare a non-empty regimes list"
        )
        for q in strat.regimes:
            assert q in QUADRANTS, f"{sid}: unknown regime quadrant {q!r}"
        assert isinstance(strat.actionable, bool), (
            f"{sid}: actionable must be a bool, got {type(strat.actionable).__name__}"
        )


# ---- non-actionable strategies must not be backtestable as longs ----
@test
def test_non_actionable_strategy_not_backtestable():
    """bear_reversal_watch is an explicit 'do NOT buy' signal (actionable=False).
    The long-only walk-forward engine must never accept it."""
    from api.backtest import BACKTESTABLE_STRATEGIES, _STRATEGY_PATTERN
    import re

    non_actionable = [sid for sid, s in STRATEGIES.items() if not s.actionable]
    assert non_actionable, "expected at least one non-actionable strategy (bear_reversal_watch)"
    assert "bear_reversal_watch" in non_actionable

    for sid in non_actionable:
        assert sid not in BACKTESTABLE_STRATEGIES, f"{sid} must not be backtestable"
        assert not re.match(_STRATEGY_PATTERN, sid), (
            f"{sid} must be rejected by the walk-forward query pattern"
        )
    for sid, s in STRATEGIES.items():
        if s.actionable:
            assert sid in BACKTESTABLE_STRATEGIES
            assert re.match(_STRATEGY_PATTERN, sid)

    # Defensive second gate: calling the engine directly returns a clear note,
    # not a 500 and not an equity curve. db/user are never touched on this path.
    from services.backtest import run_walk_forward_backtest
    res = run_walk_forward_backtest(db=None, user_id=1, strategy_id="bear_reversal_watch")
    assert res["equity"] == [], "non-actionable strategy must produce no equity curve"
    assert res["metrics"] == {}
    assert any("watchlist-only" in n for n in res["notes"]), res["notes"]

    # …and it stays visible in the catalog, just flagged.
    from services.strategies import strategy_catalog
    cat = {row["id"]: row for row in strategy_catalog()}
    assert "bear_reversal_watch" in cat, "must stay visible in the strategy catalog"
    assert cat["bear_reversal_watch"]["actionable"] is False


# ---- negative: flat / declining series should NOT trigger ----
def make_flat_series(n=280, seed=55):
    rng = random.Random(seed)
    bars = []
    for _ in range(n):
        close = 100.0 + rng.uniform(-0.2, 0.2)
        bars.append(_bar(close, close + 0.3, close - 0.3, close, v=1_000_000))
    return bars


def make_declining_series(n=280, start=200.0, seed=66):
    rng = random.Random(seed)
    bars = []
    price = start
    for _ in range(n):
        price *= (1 - 0.003 + rng.uniform(-0.001, 0.001))
        close = price
        bars.append(_bar(close * 1.001, close * 1.004, close * 0.996, close, v=900_000))
    return bars


@test
def test_strategies_do_not_trigger_on_non_setups():
    flat = make_flat_series()
    decline = make_declining_series()
    idx_flat = len(flat) - 1
    idx_dec = len(decline) - 1
    # Momentum needs an uptrend above 100MA -> flat & decline must fail
    assert STRATEGIES["momentum_rotation"].candidate(flat, idx_flat) is None
    assert STRATEGIES["momentum_rotation"].candidate(decline, idx_dec) is None
    # Pullback / breakout need close>MA200 & MA50>MA200 -> a decline fails both
    assert STRATEGIES["pullback_50ma"].candidate(decline, idx_dec) is None
    assert STRATEGIES["breakout_volume"].candidate(decline, idx_dec) is None
    # Mean reversion needs close>MA200 -> a decline fails the trend gate
    assert STRATEGIES["mean_reversion"].candidate(decline, idx_dec) is None


# ──────────────────────────────────────────────────────────────────────────
# 4. ClosedTrade R-multiple math (mirrors api/portfolio.py::close_position)
# ──────────────────────────────────────────────────────────────────────────
def closed_trade_r_multiple(avg_cost, exit_price, stop_loss):
    """Replicates the formula used in api/portfolio.py close_position."""
    if stop_loss is not None and avg_cost - stop_loss > 0:
        return round((exit_price - avg_cost) / (avg_cost - stop_loss), 2)
    return None


@test
def test_r_multiple_uses_initial_stop_not_trailed_stop():
    """R must be measured against the stop the trade was OPENED with. Trailing a
    stop up shrinks (avg_cost - stop) and would otherwise inflate recorded R."""
    from api.portfolio import compute_r_multiple

    # Entry 100, initial stop 90 (1R = $10), stop later trailed up to 98, exit 120.
    r, r_stop = compute_r_multiple(100, 120, initial_stop=90, stop_loss=98)
    assert r_stop == 90, "must use the initial stop as the denominator"
    assert r == 2.0, f"expected 2.0R against the initial stop, got {r}"
    # Against the trailed stop it would have been (120-100)/(100-98) = 10.0R.
    inflated, _ = compute_r_multiple(100, 120, initial_stop=None, stop_loss=98)
    assert inflated == 10.0, "sanity: the trailed stop really does inflate R"
    assert r != inflated

    # Legacy row: initial_stop is NULL -> fall back to stop_loss.
    r_legacy, legacy_stop = compute_r_multiple(100, 120, initial_stop=None, stop_loss=90)
    assert legacy_stop == 90 and r_legacy == 2.0, "legacy null must fall back to stop_loss"

    # Exact -1R loss measured on the initial stop, even with a trailed stop set.
    assert compute_r_multiple(100, 90, initial_stop=90, stop_loss=95)[0] == -1.0
    # No stop at all -> None
    assert compute_r_multiple(100, 120, None, None)[0] is None
    # Stop at/above cost -> None (non-positive denominator)
    assert compute_r_multiple(100, 120, 110, None)[0] is None
    assert compute_r_multiple(100, 120, 100, None)[0] is None


# ──────────────────────────────────────────────────────────────────────────
# 5. Backtest: turnover cost + risk-free-adjusted Sharpe
# ──────────────────────────────────────────────────────────────────────────
@test
def test_turnover_cost_is_not_clipped():
    """A full rotation must cost twice a half rotation. The old
    min(1.0, turnover) clip charged both the same."""
    from services.backtest import turnover_cost

    cost = 0.001          # 10 bps
    top_n = 5
    prev = {"A", "B", "C", "D", "E"}

    # No change -> no cost
    assert turnover_cost(set(prev), prev, top_n, cost) == 0.0

    # Full rotation: sym_diff = 10 = 2*top_n -> 2 * cost (sell all + buy all)
    full = {"F", "G", "H", "I", "J"}
    assert approx(turnover_cost(full, prev, top_n, cost), 2 * cost)

    # Half rotation: swap 2 of 5 -> sym_diff = 4 -> 0.8 * cost, and it must be
    # strictly less than the full-rotation charge (the bug made them equal).
    half = {"A", "B", "C", "F", "G"}
    assert approx(turnover_cost(half, prev, top_n, cost), 4 / 5 * cost)
    assert turnover_cost(half, prev, top_n, cost) < turnover_cost(full, prev, top_n, cost)

    # Regression guard for the exact bug: the old code was
    #   min(1.0, sym_diff/top_n) * cost
    # which capped at 1*cost. A full rotation must now cost strictly more.
    old_clipped = min(1.0, len(full.symmetric_difference(prev)) / top_n) * cost
    assert turnover_cost(full, prev, top_n, cost) > old_clipped


@test
def test_metrics_sharpe_subtracts_risk_free_rate():
    """_metrics must charge the same 5% risk-free rate indicators.py does,
    de-annualized to the rebalance period."""
    import math as _math
    from services.backtest import _metrics, RISK_FREE_RATE
    from statistics import pstdev, mean as _mean

    assert RISK_FREE_RATE == 0.05, "must match indicators.compute_performance_metrics"

    ppy = 252 / 5
    # 20 weekly periods, deterministic alternating-ish returns
    vals = [1.0]
    rets = [0.01, -0.004, 0.012, 0.002, -0.008] * 4
    dates = []
    from datetime import date as _d, timedelta as _td
    day = _d(2024, 1, 1)
    dates.append(day.isoformat())
    for r in rets:
        vals.append(round(vals[-1] * (1 + r), 6))
        day += _td(days=5)
        dates.append(day.isoformat())
    equity = [{"date": d, "value": v} for d, v in zip(dates, vals)]

    m = _metrics(equity, ppy)
    actual_rets = [equity[i + 1]["value"] / equity[i]["value"] - 1 for i in range(len(equity) - 1)]
    vol = pstdev(actual_rets)
    rf_period = (1 + RISK_FREE_RATE) ** (1 / ppy) - 1
    expected = _mean([r - rf_period for r in actual_rets]) / vol * _math.sqrt(ppy)
    assert m["sharpe"] == round(expected, 2), f"got {m['sharpe']} expected {round(expected, 2)}"

    # And it must differ from the old no-rf definition on this series.
    naive = _mean(actual_rets) / vol * _math.sqrt(ppy)
    assert round(naive, 2) != m["sharpe"], "rf subtraction had no effect — check the formula"
    assert m["sharpe"] < round(naive, 2), "excess-return Sharpe must be lower than the raw one"
    assert m["sortino"] is not None


# ──────────────────────────────────────────────────────────────────────────
# 5b. Quote schema v2 consumers (screener filters, watchlist ranking)
# ──────────────────────────────────────────────────────────────────────────
def _v2_quote(**over):
    """A minimal schema-v2 quote. Every range filter treats None as 'no value',
    so only the fields a given test cares about need to be set."""
    q = {
        "ticker": "TEST", "schema_v": 2, "quote_type": "EQUITY",
        "ma_state": "bull", "gc": False, "dc": False,
        "gc_event": False, "dc_event": False,
        "vol_r": 1.0, "vol_r_5d": 1.0, "p52w": 50.0,
    }
    q.update(over)
    return q


@test
def test_screener_earn_beat_filter_is_gone():
    """earn_beat no longer exists on a quote — leaving the filter in place made
    it match nothing and return zero rows."""
    from api.screener import ScreenerFilters
    assert not hasattr(ScreenerFilters(), "earn_beat"), "earn_beat must be removed"
    import inspect
    from api import screener
    assert "earn_beat" not in inspect.getsource(screener._passes), (
        "_passes must no longer reference earn_beat"
    )
    # earn_soon is a different field and must survive.
    assert hasattr(ScreenerFilters(), "earn_soon")


@test
def test_screener_ma_state_and_vol_r_5d_filters():
    from api.screener import ScreenerFilters, _passes

    bull = _v2_quote(ma_state="bull")
    bear = _v2_quote(ma_state="bear")
    none_state = _v2_quote(ma_state=None)

    # No filter set -> everything passes (empty string must behave as "no filter")
    for f in (ScreenerFilters(), ScreenerFilters(ma_state=""), ScreenerFilters(ma_state=None)):
        assert _passes(bull, f) and _passes(bear, f) and _passes(none_state, f)

    f_bull = ScreenerFilters(ma_state="bull")
    assert _passes(bull, f_bull)
    assert not _passes(bear, f_bull)
    assert not _passes(none_state, f_bull)
    assert _passes(bear, ScreenerFilters(ma_state="bear"))

    # vol_r_5d_min is the SMOOTHED ratio and must be independent of vol_r.
    hi5 = _v2_quote(vol_r=0.5, vol_r_5d=1.8)
    lo5 = _v2_quote(vol_r=3.0, vol_r_5d=0.9)
    assert _passes(hi5, ScreenerFilters(vol_r_5d_min=1.2))
    assert not _passes(lo5, ScreenerFilters(vol_r_5d_min=1.2))
    # …and vol_r_min still filters the single-bar ratio.
    assert _passes(lo5, ScreenerFilters(vol_r_min=1.2))
    assert not _passes(hi5, ScreenerFilters(vol_r_min=1.2))


@test
def test_screener_gc_filter_is_an_event_not_a_state():
    """gc/dc filter on a CROSS within the last 5 bars. An established uptrend
    (ma_state bull, no recent cross) must NOT satisfy the gc filter."""
    from api.screener import ScreenerFilters, _passes

    established = _v2_quote(ma_state="bull", gc=False, gc_event=False)
    fresh_cross = _v2_quote(ma_state="bull", gc=True, gc_event=True)
    f_gc = ScreenerFilters(gc=True)
    assert not _passes(established, f_gc), "an established trend is not a cross event"
    assert _passes(fresh_cross, f_gc)
    # ma_state is the way to ask for the standing trend instead.
    assert _passes(established, ScreenerFilters(ma_state="bull"))

    f_dc = ScreenerFilters(dc=True)
    assert not _passes(_v2_quote(dc=False, dc_event=False), f_dc)
    assert _passes(_v2_quote(dc=True, dc_event=True), f_dc)


@test
def test_screener_filters_accept_new_frontend_keys():
    """The frontend POSTs ma_state and vol_r_5d_min. They must be real model
    fields, not silently-dropped extras."""
    from api.screener import ScreenerFilters
    f = ScreenerFilters(**{"ma_state": "bull", "vol_r_5d_min": 1.3})
    assert f.ma_state == "bull"
    assert f.vol_r_5d_min == 1.3


@test
def test_watchlist_rank_uses_ma_state_not_gc_event():
    """The 0.5 trend bonus must key off the standing MA state. Keying it off the
    5-bar `gc` event meant it essentially never fired and reordered rankings."""
    from api.watchlist import composite_rank

    # Quotes identical except for the MA trend state / cross event.
    base = {"ticker": "X", "score": {"o": 5, "f": 3, "t": 3, "m": 3}, "max_dd_1m": 0}
    bull = {**base, "ma_state": "bull", "gc": False, "gc_event": False}
    bear = {**base, "ma_state": "bear", "gc": False, "gc_event": False}
    fresh = {**base, "ma_state": "bull", "gc": True, "gc_event": True}
    nostate = {**base, "ma_state": None, "gc": False, "gc_event": False}

    # The trend-STATE bonus keeps its original 0.5 weight.
    assert approx(composite_rank(bull) - composite_rank(bear), 0.5)
    assert approx(composite_rank(bull) - composite_rank(nostate), 0.5)
    # A fresh cross is a SEPARATE, smaller bonus on top of the state bonus.
    assert approx(composite_rank(fresh) - composite_rank(bull), 0.25)
    assert composite_rank(fresh) > composite_rank(bull) > composite_rank(bear)

    # Regression: before the fix an established uptrend with no recent cross
    # (gc False) scored the same as a downtrend on this term. It must not now.
    assert composite_rank(bull) != composite_rank(bear)

    # vol_r (single-bar ratio) keeps its 1.2 threshold and 0.5 weight.
    assert approx(
        composite_rank({**bull, "vol_r": 1.5}) - composite_rank({**bull, "vol_r": 1.0}), 0.5
    )


@test
def test_ai_technicals_whitelist_covers_schema_v2():
    import inspect
    from api import ai

    src = inspect.getsource(ai)
    idx = src.index("technicals = {k: data.get(k) for k in [")
    block = src[idx: idx + 400]
    for field in ("ma_state", "ann_ret_1m", "rsi", "macd_sig", "vs_ma200", "gc", "dc",
                  "vol_r", "p52w", "score"):
        assert f'"{field}"' in block, f"technicals whitelist missing {field}"
    assert "earn_beat" not in src, "api/ai.py must not reference the removed earn_beat"


# ──────────────────────────────────────────────────────────────────────────
# 6. Journal stats: win-rate / scratch split
# ──────────────────────────────────────────────────────────────────────────
class _FakeTrade:
    """Duck-types the ClosedTrade fields api/journal.py::_stats reads."""
    def __init__(self, pnl, pnl_pct, r_multiple=None, strategy=None):
        self.pnl = pnl
        self.pnl_pct = pnl_pct
        self.r_multiple = r_multiple
        self.strategy = strategy


@test
def test_journal_stats_scratch_split_and_r_sample():
    from api.journal import _stats

    trades = [
        _FakeTrade(100, 5.0, r_multiple=2.0),
        _FakeTrade(50, 2.0),              # win, no R recorded (no valid stop)
        _FakeTrade(-40, -2.0, r_multiple=-1.0),
        _FakeTrade(0, 0.0),               # breakeven scratch — neither win nor loss
    ]
    s = _stats(trades)
    assert s["count"] == 4
    assert s["wins"] == 2
    assert s["losses"] == 1
    assert s["scratch"] == 1, "pnl == 0 must be counted as scratch, not a loss"
    # wins / (wins + losses) = 2/3, NOT 2/4 (the old pnl <= 0 bucketing)
    assert s["win_rate"] == round(2 / 3 * 100, 1), f"got {s['win_rate']}"
    assert s["win_rate"] != 50.0, "scratch must not be counted in the loss bucket"
    # avg_loss_pct must not be diluted by the breakeven trade
    assert s["avg_loss_pct"] == -2.0

    # Mixed denominators are now explicit: only 2 of 4 trades carry an R.
    assert s["r_sample"] == 2
    assert s["avg_r"] == 0.5 and s["expectancy_r"] == 0.5

    # All-scratch edge case -> win_rate undefined rather than 0%
    assert _stats([_FakeTrade(0, 0.0)])["win_rate"] is None
    assert _stats([])["count"] == 0


@test
def test_journal_aggregates_use_all_trades_not_the_page():
    """stats/by_strategy must be built from every closed trade, not the LIMIT-ed
    display page — otherwise the headline numbers change with the page size."""
    import inspect
    from api import journal

    src = inspect.getsource(journal.get_journal)
    assert '"stats": _stats(all_trades)' in src, "stats must be computed from all_trades"
    assert "for t in all_trades:" in src, "by_strategy must be built from all_trades"
    assert '"stats": _stats(trades)' not in src, "stats must not use the limited page"


# ──────────────────────────────────────────────────────────────────────────
# 7. ClosedTrade R-multiple math (legacy formula mirror)
# ──────────────────────────────────────────────────────────────────────────
@test
def test_r_multiple_known_cases():
    # +2R winner: cost 100, stop 90, exit 120 -> (120-100)/(100-90)=2.0
    assert closed_trade_r_multiple(100, 120, 90) == 2.0
    # exact -1R loss: exit at stop
    assert closed_trade_r_multiple(100, 90, 90) == -1.0
    # no stop -> None
    assert closed_trade_r_multiple(100, 120, None) is None
    # stop above cost (non-positive denominator) -> None
    assert closed_trade_r_multiple(100, 120, 110) is None
    # stop equal to cost -> None (denominator 0)
    assert closed_trade_r_multiple(100, 120, 100) is None
    # fractional R rounds to 2 dp: cost 50, stop 45, exit 57.5 -> 7.5/5 = 1.5
    assert closed_trade_r_multiple(50, 57.5, 45) == 1.5


# ──────────────────────────────────────────────────────────────────────────
# Shared page chrome
#
# The top bar used to be hand-copied into charts.html, report.html and
# index.html. Both the markup AND its CSS drifted: those pages still said
# "Today" and "Backtest", had no Scorecard link at all, and sat on an older
# palette — so clicking Charts visibly jumped to a different-looking app with
# a stale menu. journal.html/admin.html drifted too, invisibly: they took
# their LINKS from common.js but kept their own green/red, so P&L was
# literally a different green there than on the Playbook.
#
# Everything now comes from static/js/common.js + static/css/common.css.
# These tests keep it that way.
# ──────────────────────────────────────────────────────────────────────────

import os as _os
import re as _re
import subprocess as _sp

_STATIC = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "static")
_ROOT = _os.path.dirname(_STATIC)

# Pages that load common.js and therefore share its global scope.
_SHARED_PAGES = ("today.html", "backtest.html", "scorecard.html", "journal.html",
                 "admin.html", "charts.html", "report.html", "index.html")


def _read_static(*parts):
    with open(_os.path.join(_STATIC, *parts), encoding="utf-8") as fh:
        return fh.read()


def _common_js_top_level_names():
    """Names declared at column 0 in common.js — the shared global surface."""
    js = _read_static("js", "common.js")
    pat = r"^(?:const|let|var|async function|function|class)\s+([A-Za-z_$][\w$]*)"
    return set(_re.findall(pat, js, _re.M))


@test
def test_every_page_uses_the_shared_chrome():
    """No page may hand-write the nav again; all must load the shared files."""
    for page in _SHARED_PAGES:
        html = _read_static(page)
        assert _re.search(r'<script[^>]+src="[^"]*common\.js"', html), \
            page + " does not load /static/js/common.js"
        assert 'href="/static/css/common.css"' in html, \
            page + " does not link /static/css/common.css"
        assert 'class="navlink"' not in html, \
            page + " still has hand-written .navlink nav markup"


@test
def test_nav_is_rendered_from_nav_links_only():
    """NAV_LINKS is the single source: pages call initHeader, or (index.html,
    which is React and must not let initHeader touch its DOM) read the array."""
    # Match an actual CALL — `initHeader('/charts', …)` — not a prose mention of
    # the name, which index.html legitimately contains in an explanatory comment.
    call = _re.compile(r"initHeader\(\s*['\"]")
    for page in ("today.html", "backtest.html", "scorecard.html", "journal.html",
                 "admin.html", "charts.html", "report.html"):
        html = _read_static(page)
        assert call.search(html), page + " never calls initHeader('<path>')"
    idx = _read_static("index.html")
    assert "NAV_LINKS" in idx, "index.html no longer reads NAV_LINKS"
    assert not call.search(idx), \
        "index.html must NOT call initHeader — it uses innerHTML and fights React"
    # The old hand-mirrored array must be gone for good.
    assert not _re.search(r"\['/'\s*,\s*'Today'\]", idx), \
        "index.html still carries the old hand-mirrored nav array"


@test
def test_every_nav_target_has_a_route():
    """A nav entry pointing at a route main.py does not serve is a dead link."""
    js = _read_static("js", "common.js")
    body = js.split("const NAV_LINKS = [", 1)[1].split("];", 1)[0]
    links = _re.findall(r"\['(/[a-z]*)',\s*'([^']+)'", body)
    assert len(links) >= 6, links
    with open(_os.path.join(_ROOT, "main.py"), encoding="utf-8") as fh:
        main_src = fh.read()
    for href, label in links:
        assert ('@app.get("' + href + '")') in main_src, \
            "no route in main.py for " + href + " (" + label + ")"


@test
def test_palette_lives_only_in_common_css():
    """The header palette must be defined once. Pages asserting the hexes
    themselves is what let journal/admin drift to their own colours."""
    css = _read_static("css", "common.css")
    for token in ("#121215", "#232327", "#3fbf7f", "#e5484d"):
        assert token in css, "common.css lost " + token
    # No page may re-declare a .topbar background/border of its own.
    for page in _SHARED_PAGES:
        html = _read_static(page)
        block = html.split("<style>", 1)[-1].split("</style>", 1)[0]
        for rule in _re.findall(r"\.topbar\s*\{([^}]*)\}", block):
            assert "background" not in rule, \
                page + " re-declares .topbar background; common.css owns it"


@test
def test_no_js_binding_collides_with_common_js():
    """A page `const`/`let` that reuses ANY common.js top-level name is a fatal
    SyntaxError — including over one of its `function`s, because a global
    function declaration creates a non-configurable global property. This cost
    two real outages during the refactor (`_tdChart`, `fmtPct`)."""
    names = _common_js_top_level_names()
    assert {"token", "authHdr", "NAV_LINKS", "_tdChart", "fmtPct"} <= names, names
    for page in _SHARED_PAGES:
        html = _read_static(page)
        for blk in _re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>",
                               html, _re.S):
            # Column-0 / low-indent declarations are the ones sharing global scope.
            for kw, name in _re.findall(
                    r"^\s{0,6}(const|let)\s+([A-Za-z_$][\w$]*)", blk, _re.M):
                assert name not in names, (
                    page + " declares `" + kw + " " + name + "` which collides "
                    "with a common.js top-level binding — this is a SyntaxError "
                    "that blanks the page"
                )


@test
def test_pages_compile_together_with_common_js():
    """Definitive check: compile common.js + each page's inline script in one
    scope, exactly as the browser does. Skips cleanly when node is absent."""
    try:
        _sp.run(["node", "--version"], capture_output=True, check=True, timeout=20)
    except Exception:
        return  # node not available in this environment; the static guards above still ran
    common = _read_static("js", "common.js")
    for page in _SHARED_PAGES:
        html = _read_static(page)
        blocks = _re.findall(
            r"<script(?![^>]*\bsrc=)([^>]*)>(.*?)</script>", html, _re.S)
        if any("text/babel" in attrs for attrs, _ in blocks):
            continue  # JSX needs Babel; covered by the static guard above
        combined = common + "\n;\n" + "\n".join(body for _, body in blocks)
        # Must go through a UTF-8 file: the sources contain non-cp1252
        # characters, and piping via stdin fails on Windows' default codec.
        import tempfile
        fd, tmp = tempfile.mkstemp(suffix=".js")
        try:
            with _os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(combined)
            proc = _sp.run(["node", "--check", tmp],
                           capture_output=True, text=True, timeout=60)
        finally:
            _os.unlink(tmp)
        assert proc.returncode == 0, (
            page + " does not compile alongside common.js:\n"
            + (proc.stderr or "")[:600]
        )

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
