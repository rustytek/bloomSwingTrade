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

import math
import random
import traceback

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


@test
def test_strategies_registry_keys():
    assert set(STRATEGIES) == {
        "momentum_rotation", "pullback_50ma", "breakout_volume", "mean_reversion",
    }


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
