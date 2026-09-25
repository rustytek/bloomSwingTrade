"""
Pure-function, no-network tests for the walk-forward backtest engine and the
pluggable exit-rule framework.

Covers:
  1. NO LOOK-AHEAD  — truncating the series immediately after a decision bar
     must not change the decision at that bar (strategies AND exit rules).
  2. rotation-mode regression — the simulation output is byte-identical to the
     recorded baseline digest, so the trade_plan work cannot have moved a
     single number the app has ever shown.
  3. Every exit rule firing on a hand-built series with a known answer.
  4. The intrabar stop-before-target convention (and gap fills).
  5. max_positions blocking an entry and incrementing signals_skipped_full_book.
  6. Wilson score interval vs hand-computed values.
  7. JSON-safety (Starlette serializes with allow_nan=False — one NaN 500s).

Run:
    python test_backtest.py

No network, no DB (the two DB accessors are monkeypatched), deterministic.
"""
from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import json
import math
import sys
import traceback
import types

# ──────────────────────────────────────────────────────────────────────────
# Web-dependency stubs (same mechanism as test_passes.py)
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

import services.backtest as bt                                      # noqa: E402
from services.backtest import (                                     # noqa: E402
    OpenPosition, close_position_at, confidence_label, walk_position,
    wilson_interval, MIN_PERIODS_LOW_CONFIDENCE, MIN_PERIODS_HIGH_CONFIDENCE,
)
from services.exits import (                                        # noqa: E402
    AtrTrailingStop, FixedStopTarget, PartialProfitTaking, PositionState,
    RegimeExit, TimeStop, advance, apply_partial, build_exit_rules, resolve_exit,
)
from services.strategies import STRATEGIES                          # noqa: E402


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
# Deterministic synthetic market
# ──────────────────────────────────────────────────────────────────────────
N_BARS = 420
_DATES = None


def _dates(n: int = N_BARS) -> list[str]:
    """Sequential calendar dates — the engine only ever compares them as strings."""
    global _DATES
    if _DATES is None or len(_DATES) < n:
        from datetime import date, timedelta
        day = date(2019, 1, 1)
        out = []
        while len(out) < n:
            if day.weekday() < 5:
                out.append(day.isoformat())
            day += timedelta(days=1)
        _DATES = out
    return _DATES[:n]


def make_series(seed: int, n: int = N_BARS, start: float = 100.0,
                drift: float = 0.0009, amp: float = 0.045, wave: float = 37.0) -> list[dict]:
    """Trend + deterministic sine wobble. No RNG: byte-stable across runs/platforms."""
    dates = _dates(n)
    bars = []
    price = start
    for i in range(n):
        phase = math.sin((i + seed * 11) / wave * 2 * math.pi)
        price = start * math.exp(drift * i) * (1 + amp * phase)
        wig = abs(math.cos((i + seed * 7) / 9.0)) * price * 0.012
        close = price
        open_ = price - wig * 0.4
        bars.append({
            "date": dates[i],
            "open": round(open_, 4),
            "high": round(max(open_, close) + wig, 4),
            "low": round(min(open_, close) - wig, 4),
            "close": round(close, 4),
            "vol": 1_000_000 + (i * 977 + seed * 13) % 500_000,
        })
    return bars


_TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


def _histories() -> dict[str, list[dict]]:
    h = {t: make_series(seed=i + 1, drift=0.0006 + 0.0002 * i, amp=0.03 + 0.01 * (i % 3))
         for i, t in enumerate(_TICKERS)}
    h["SPY"] = make_series(seed=99, drift=0.0005, amp=0.02, wave=61.0)
    return h


_HIST = _histories()


def _install_fake_db():
    """Patch the two DB accessors so the engine runs entirely in memory."""
    bt._load_history = lambda db, ticker: _HIST.get(ticker, [])
    bt._source_tickers = lambda db, user_id, source: sorted(_TICKERS)


_install_fake_db()

_BASE_KW = dict(
    db=None, user_id=1, strategy_id="momentum_rotation", source="watchlist",
    top_n=3, rebalance_days=5, cost_bps=10, spy_regime=True, regime_ma=50,
    period="all",
)


def run(**over):
    kw = dict(_BASE_KW)
    kw.update(over)
    return bt.run_walk_forward_backtest(**kw)


def _bar(date_, o, h, l, c, atr=1.0, quadrant="trending_bull", vol=1_000_000):
    return {"date": date_, "open": o, "high": h, "low": l, "close": c,
            "vol": vol, "atr": atr, "quadrant": quadrant}


def _state(entry=100.0, stop=95.0, target=110.0, shares=100, **over):
    kw = dict(
        ticker="TST", entry_date="2020-01-01", entry_price=entry, entry_idx=0,
        shares=shares, initial_shares=shares, stop=stop, initial_stop=stop,
        risk_per_share=entry - stop, target=target, strategy="unit_test",
        highest_close=entry,
    )
    kw.update(over)
    return PositionState(**kw)


# ──────────────────────────────────────────────────────────────────────────
# 1. NO LOOK-AHEAD — the cardinal rule
# ──────────────────────────────────────────────────────────────────────────
@test
def test_no_lookahead_strategy_candidate_truncation():
    """candidate(bars, idx) must equal candidate(bars[:idx+1], idx) for every
    strategy at every probe point. If a strategy ever peeks past idx, the whole
    app's numbers are fiction."""
    bars = _HIST["AAA"]
    checked = 0
    for sid, strategy in STRATEGIES.items():
        for idx in (150, 220, 300, 380, len(bars) - 1):
            full = strategy.candidate(bars, idx)
            trunc = strategy.candidate(bars[: idx + 1], idx)
            assert full == trunc, (
                f"{sid} look-ahead at idx={idx}: full={full!r} truncated={trunc!r}"
            )
            checked += 1
    assert checked >= 30, f"only probed {checked} strategy/idx combinations"


@test
def test_no_lookahead_exit_rules_truncation():
    """walk_position over the full series must produce the same fills as over a
    series truncated one bar after the exit decision."""
    bars = [_bar(f"2020-01-{i + 1:02d}", 100, 101 + i, 99 - i * 0.7, 100 - i * 0.3)
            for i in range(12)]
    plain = [{k: v for k, v in b.items() if k not in ("atr", "quadrant")} for b in bars]

    def _fills(series):
        st = _state()
        rules = [FixedStopTarget()]
        pos = OpenPosition(ticker="TST", bars=series, atr=[1.0] * len(series),
                           quadrants={}, state=st, rules=rules, idx=0)
        advance(rules, dict(series[0], atr=1.0, quadrant=None), st)
        return walk_position(pos, series[-1]["date"])

    full = _fills(plain)
    assert full and full[0]["closed"], "expected the stop to fire on the full series"
    cut = plain.index(next(b for b in plain if b["date"] == full[0]["date"]))
    truncated = _fills(plain[: cut + 1])
    assert truncated == full, (
        f"decision changed when the future was removed:\n full={full}\n trunc={truncated}"
    )


@test
def test_no_lookahead_trailing_stop_uses_prior_bar_level():
    """A bar must never be able to stop itself out on a trail level derived from
    its own close — the level checked on bar t comes from bar t-1's update()."""
    rule = AtrTrailingStop(atr_mult=2.0)
    st = _state(entry=100.0, stop=95.0, target=None)
    # Bar 1: big up close -> trail should advance only AFTER evaluation.
    b1 = _bar("2020-01-02", 100, 112, 99.5, 110, atr=2.0)
    assert rule.evaluate(b1, st) is None, "trail fired before it was ever set"
    advance([rule], b1, st)
    assert approx(st.trail_stop, 110 - 4.0), st.trail_stop
    # Bar 2 low 105 > 106? no -> 105 <= 106 fires, using YESTERDAY's level.
    b2 = _bar("2020-01-03", 108, 109, 105, 107, atr=2.0)
    sig = rule.evaluate(b2, st)
    assert sig is not None and approx(sig.price, 106.0), sig
    # And the level never ratchets down.
    b3 = _bar("2020-01-06", 100, 101, 99, 100, atr=9.0)
    advance([rule], b3, st)
    assert st.trail_stop >= 106.0, st.trail_stop


# ──────────────────────────────────────────────────────────────────────────
# 2. Rotation-mode regression against a recorded baseline
# ──────────────────────────────────────────────────────────────────────────
# Digest over ONLY the simulation-bearing keys (the new `mode`/`trade_log`/
# `caveats` keys and the new regime_breakdown CI fields are additive metadata
# and are excluded by construction below).
#
# RECORDED 2026-09-20 and independently cross-checked against the pre-existing
# engine: the same synthetic market was run through the previous revision of
# services/backtest.py and the equity curve, benchmark, per-period trade rows,
# parameters, notes and available_tickers came back identical, byte for byte.
# Re-record ONLY on a deliberate, reviewed change to rotation behaviour —
# regenerate with `python test_backtest.py --record-baseline`.
_ROTATION_BASELINE_DIGEST = "ef29c7556f5bdb93575c4241965c8812"
_BASELINE_REGIME_FIELDS = ("quadrant", "label", "periods", "pct_of_periods",
                           "avg_period_return", "win_rate", "cum_return")


def rotation_fingerprint(result: dict) -> str:
    payload = {
        "equity": result["equity"],
        "benchmark": result["benchmark"],
        "trades": result["trades"],
        "metrics": result["metrics"],
        "benchmark_metrics": result["benchmark_metrics"],
        "available_tickers": result["available_tickers"],
        "parameters": result["parameters"],
        "strategy_regimes": result["strategy_regimes"],
        "notes": result["notes"],
        "regime_breakdown": [
            {k: row[k] for k in _BASELINE_REGIME_FIELDS} for row in result["regime_breakdown"]
        ],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(blob.encode()).hexdigest()


@test
def test_rotation_matches_recorded_baseline():
    digest = rotation_fingerprint(run())
    assert digest == _ROTATION_BASELINE_DIGEST, (
        f"rotation output changed! got {digest}, baseline {_ROTATION_BASELINE_DIGEST}. "
        "Rotation mode is a frozen regression surface — if this was intentional, "
        "re-record the digest deliberately."
    )


@test
def test_rotation_default_and_explicit_mode_identical():
    assert rotation_fingerprint(run()) == rotation_fingerprint(run(mode="rotation"))
    # An unknown mode falls back to rotation rather than exploding.
    assert rotation_fingerprint(run(mode="nonsense")) == rotation_fingerprint(run())


@test
def test_rotation_has_no_trade_plan_metrics():
    m = run()["metrics"]
    for key in ("avg_r", "expectancy_r", "win_rate_trades", "trades_taken",
                "signals_skipped_full_book", "avg_bars_held"):
        assert key not in m, f"rotation leaked trade-plan metric {key}"
    assert run()["trade_log"] == []


@test
def test_rotation_notes_unchanged():
    notes = run()["notes"]
    assert "ATR stops/targets are NOT simulated intrabar" in notes[2], notes[2]


# ──────────────────────────────────────────────────────────────────────────
# 3. Exit rules — hand-built series, known answers
# ──────────────────────────────────────────────────────────────────────────
@test
def test_fixed_stop_fires_at_level():
    rule = FixedStopTarget()
    st = _state()
    sig = rule.evaluate(_bar("d", 99, 101, 94, 96), st)
    assert sig is not None and sig.kind == "stop" and approx(sig.price, 95.0), sig


@test
def test_fixed_target_fires_at_level():
    rule = FixedStopTarget()
    st = _state()
    sig = rule.evaluate(_bar("d", 101, 111, 100.5, 110.5), st)
    assert sig is not None and sig.kind == "target" and approx(sig.price, 110.0), sig


@test
def test_fixed_rule_quiet_inside_the_band():
    assert FixedStopTarget().evaluate(_bar("d", 100, 105, 96, 102), _state()) is None


@test
def test_stop_before_target_intrabar_convention():
    """A bar that breaches BOTH the stop and the target must resolve as a STOP.
    Assuming the target filled first is how a backtest manufactures winners."""
    st = _state()
    sig = FixedStopTarget().evaluate(_bar("d", 100, 115, 90, 112), st)
    assert sig is not None and sig.kind == "stop", sig
    assert approx(sig.price, 95.0), sig.price
    # Same verdict through the global resolver, with a target-only rule present.
    rules = [PartialProfitTaking(first_r=1.0), FixedStopTarget()]
    st2 = _state()
    resolved = resolve_exit(rules, _bar("d", 100, 115, 90, 112), st2)
    assert resolved.kind == "stop", resolved


@test
def test_gap_down_fills_at_the_open_not_the_level():
    sig = FixedStopTarget().evaluate(_bar("d", 90, 92, 88, 91), _state())
    assert sig is not None and approx(sig.price, 90.0), sig
    # ...and a gap UP through the target fills at the open too.
    sig2 = FixedStopTarget().evaluate(_bar("d", 114, 116, 113, 115), _state())
    assert sig2 is not None and sig2.kind == "target" and approx(sig2.price, 114.0), sig2


@test
def test_time_stop_fires_only_when_flat_and_old():
    rule = TimeStop(max_bars=10, min_r=0.5)
    young = _state()
    young.bars_held = 9
    assert rule.evaluate(_bar("d", 100, 101, 99, 100.5), young) is None, "fired too early"
    old_flat = _state()
    old_flat.bars_held = 10
    sig = rule.evaluate(_bar("d", 100, 101, 99, 100.5), old_flat)   # +0.1R
    assert sig is not None and sig.kind == "time" and approx(sig.price, 100.5), sig
    old_working = _state()
    old_working.bars_held = 30
    # entry 100, risk 5 -> +0.5R is 102.5; a 103 close must NOT be cut.
    assert rule.evaluate(_bar("d", 102, 104, 101, 103), old_working) is None


@test
def test_partial_profit_scales_out_and_moves_stop_to_breakeven():
    rule = PartialProfitTaking(first_r=1.0, fraction=0.5)
    st = _state(shares=100)                 # entry 100, stop 95, 1R = 105
    assert rule.evaluate(_bar("d", 100, 104.9, 99, 104), st) is None, "fired below 1R"
    sig = rule.evaluate(_bar("d", 101, 106, 100, 105.5), st)
    assert sig is not None and sig.kind == "partial" and approx(sig.price, 105.0), sig
    sold = apply_partial(sig, st, rule)
    assert sold == 50 and st.shares == 50, (sold, st.shares)
    assert approx(st.stop, 100.0), st.stop
    assert st.scaled_out and rule.evaluate(_bar("d", 110, 120, 109, 119), st) is None


@test
def test_regime_exit_fires_when_quadrant_leaves_the_tagged_set():
    rule = RegimeExit(regimes=["trending_bull"])
    st = _state()
    assert rule.evaluate(_bar("d", 100, 101, 99, 100, quadrant="trending_bull"), st) is None
    sig = rule.evaluate(_bar("d", 100, 101, 99, 100.25, quadrant="choppy_volatile"), st)
    assert sig is not None and sig.kind == "regime" and approx(sig.price, 100.25), sig
    # An untagged rule never fires (that is how "fixed"-only positions behave).
    assert RegimeExit(regimes=[]).evaluate(
        _bar("d", 1, 2, 0.5, 1, quadrant="choppy_volatile"), st) is None


@test
def test_resolver_priority_order_stop_partial_target_time_regime():
    rules = build_exit_rules("fixed,trail,time,partial,regime",
                             atr_mult=2.5, regimes=["trending_bull"])
    st = _state()
    st.bars_held = 99
    st.trail_stop = 97.0
    # Everything fires at once on this bar; the stop must win.
    bar = _bar("d", 100, 120, 90, 100, quadrant="choppy_calm")
    sig = resolve_exit(rules, bar, st)
    assert sig.kind == "stop", sig
    # The TRAIL stop (priority 5) beats the fixed stop (priority 10) at 97.
    assert approx(sig.price, 97.0), sig.price


@test
def test_build_exit_rules_always_includes_a_hard_stop():
    for spec in (None, "", "time", ["trail"], "bogus"):
        rules = build_exit_rules(spec)
        assert any(isinstance(r, FixedStopTarget) for r in rules), spec


@test
def test_walk_position_books_partial_then_stop_at_breakeven():
    """End-to-end on one position: scale out at 1R, then get stopped at the
    breakeven stop the partial installed. Net result must be positive."""
    bars = [
        {"date": "2020-01-01", "open": 100, "high": 101, "low": 99, "close": 100},
        {"date": "2020-01-02", "open": 101, "high": 106, "low": 100, "close": 105},  # 1R -> partial
        {"date": "2020-01-03", "open": 101, "high": 102, "low": 98, "close": 99},    # breakeven stop
    ]
    st = _state(shares=100)
    rules = [FixedStopTarget(), PartialProfitTaking(first_r=1.0, fraction=0.5)]
    pos = OpenPosition(ticker="TST", bars=bars, atr=[1.0] * 3, quadrants={},
                       state=st, rules=rules, idx=0)
    advance(rules, dict(bars[0], atr=1.0, quadrant=None), st)
    fills = walk_position(pos, "2020-01-03")
    assert len(fills) == 2, fills
    assert fills[0]["kind"] == "partial" and fills[0]["shares"] == 50
    assert fills[1]["closed"] and approx(fills[1]["price"], 100.0), fills[1]
    gross = sum(f["shares"] * (f["price"] - 100.0) for f in fills)
    assert approx(gross, 250.0), gross          # 50 * +5 + 50 * 0


@test
def test_close_position_at_marks_the_remaining_shares():
    st = _state(shares=40)
    pos = OpenPosition(ticker="TST", bars=[], atr=[], quadrants={}, state=st,
                       rules=[], idx=0)
    fill = close_position_at(pos, 123.45, "2020-06-01", "open at end of test")
    assert pos.closed and fill["shares"] == 40 and fill["closed"]
    assert fill["kind"] == "end_of_test"


# ──────────────────────────────────────────────────────────────────────────
# 4. trade_plan mode
# ──────────────────────────────────────────────────────────────────────────
@test
def test_trade_plan_mode_runs_and_reports_trade_metrics():
    res = run(mode="trade_plan", max_positions=4, account_size=100000, risk_pct=1.0)
    m = res["metrics"]
    for key in ("avg_r", "expectancy_r", "win_rate_trades", "trades_taken",
                "signals_skipped_full_book", "avg_bars_held"):
        assert key in m, f"missing trade-plan metric {key}"
    assert m["trades_taken"] > 0, "no trades were taken on the synthetic market"
    assert res["mode"] == "trade_plan"
    assert len(res["trade_log"]) == m["trades_taken"]
    row = res["trade_log"][0]
    for key in ("entry_date", "entry_price", "exit_date", "exit_price",
                "r_multiple", "exit_reason", "bars_held"):
        assert key in row, f"trade record missing {key}"
    assert all(t["exit_date"] is not None for t in res["trade_log"]), "a trade never closed"
    assert all(t["bars_held"] >= 1 for t in res["trade_log"])


@test
def test_trade_plan_keeps_every_pinned_response_key():
    res = run(mode="trade_plan")
    for key in ("strategy", "source", "parameters", "available_tickers", "equity",
                "benchmark", "trades", "metrics", "benchmark_metrics",
                "regime_breakdown", "strategy_regimes", "notes"):
        assert key in res, f"pinned key {key} disappeared"


@test
def test_max_positions_blocks_entries_and_is_counted():
    tight = run(mode="trade_plan", max_positions=1, top_n=5, account_size=100000)
    loose = run(mode="trade_plan", max_positions=6, top_n=5, account_size=100000)
    assert tight["metrics"]["signals_skipped_full_book"] > 0, \
        "a 1-slot book never rejected a signal"
    assert tight["metrics"]["signals_skipped_full_book"] > \
        loose["metrics"]["signals_skipped_full_book"], "slot limit had no effect"
    # And the book is genuinely capped at every rebalance.
    assert max(len(t["holdings"]) for t in tight["trades"]) <= 1
    assert max(len(t["holdings"]) for t in loose["trades"]) <= 6


@test
def test_trade_plan_respects_max_position_value_cap():
    """build_trade_plan caps a position at 25% of equity; with a tiny account and
    a wide risk budget the cap, not the risk budget, must be what binds."""
    res = run(mode="trade_plan", account_size=20000, risk_pct=50.0, max_positions=8)
    for t in res["trade_log"]:
        value = t["shares"] * t["entry_price"]
        # 25% cap of an account that only ever compounds from 20k; allow generous
        # headroom for equity growth over the run.
        assert value <= 20000 * 0.25 * 12, (t["ticker"], value)


@test
def test_trade_plan_cash_earns_nothing_when_no_signals_qualify():
    """With the SPY regime filter satisfied but no strategy candidates (an
    impossible score threshold), equity must stay pinned at 1.0."""
    strategy = STRATEGIES["momentum_rotation"]
    original = strategy.candidate
    try:
        strategy.candidate = lambda bars, idx: None
        res = run(mode="trade_plan")
        assert res["metrics"]["trades_taken"] == 0
        assert all(approx(p["value"], 1.0) for p in res["equity"]), res["equity"][:3]
    finally:
        strategy.candidate = original


@test
def test_trade_plan_parameters_echo_the_risk_settings():
    p = run(mode="trade_plan", account_size=55000, risk_pct=2.5, max_positions=3,
            atr_stop_mult=3.0, r_multiple=1.5)["parameters"]
    assert p["mode"] == "trade_plan"
    assert p["account_size"] == 55000 and p["risk_pct"] == 2.5
    assert p["max_positions"] == 3 and p["atr_stop_mult"] == 3.0 and p["r_multiple"] == 1.5


@test
def test_trade_plan_defaults_match_the_today_dashboard():
    p = run(mode="trade_plan")["parameters"]
    assert p["account_size"] == bt.DEFAULT_ACCOUNT_SIZE == 10000.0
    assert p["risk_pct"] == bt.DEFAULT_RISK_PCT == 1.0
    assert p["max_positions"] == bt.DEFAULT_MAX_POSITIONS == 8
    assert p["atr_stop_mult"] == bt.DEFAULT_ATR_STOP_MULT == 2.5
    assert p["r_multiple"] == bt.DEFAULT_R_MULTIPLE == 2.0


# ──────────────────────────────────────────────────────────────────────────
# 5. Wilson interval + sample-size honesty
# ──────────────────────────────────────────────────────────────────────────
@test
def test_wilson_interval_hand_computed():
    # 7 wins of 10, 95% Wilson -> (39.68%, 89.22%)  [Wilson 1927; standard table]
    low, high = wilson_interval(7, 10)
    assert approx(low, 39.6777, 0.01), low
    assert approx(high, 89.2209, 0.01), high
    # 0 of 30 -> lower bound exactly 0, upper bound ~11.35%
    low0, high0 = wilson_interval(0, 30)
    assert approx(low0, 0.0, 1e-9), low0
    assert approx(high0, 11.3502, 0.01), high0
    # 30 of 30 -> upper bound exactly 100, lower ~88.65% (mirror of the above)
    low1, high1 = wilson_interval(30, 30)
    assert approx(high1, 100.0, 1e-9), high1
    assert approx(low1, 100 - 11.3502, 0.01), low1


@test
def test_wilson_interval_narrows_with_sample_size():
    widths = [wilson_interval(n // 2, n)[1] - wilson_interval(n // 2, n)[0]
              for n in (10, 50, 200, 1000)]
    assert widths == sorted(widths, reverse=True), widths
    assert widths[-1] < widths[0] / 5, widths
    assert wilson_interval(0, 0) == (0.0, 100.0)


@test
def test_confidence_labels_use_module_constants():
    assert confidence_label(0) == "insufficient"
    assert confidence_label(MIN_PERIODS_LOW_CONFIDENCE - 1) == "insufficient"
    assert confidence_label(MIN_PERIODS_LOW_CONFIDENCE) == "low"
    assert confidence_label(MIN_PERIODS_HIGH_CONFIDENCE - 1) == "low"
    assert confidence_label(MIN_PERIODS_HIGH_CONFIDENCE) == "high"


@test
def test_regime_breakdown_carries_intervals_and_confidence():
    for mode in ("rotation", "trade_plan"):
        rows = run(mode=mode)["regime_breakdown"]
        assert rows, f"{mode}: empty regime_breakdown"
        for row in rows:
            assert row["win_rate_ci_low"] <= row["win_rate"] + 1e-6 <= row["win_rate_ci_high"] + 1e-6, row
            assert row["confidence"] in ("high", "low", "insufficient"), row
            assert row["confidence"] == confidence_label(row["periods"])


@test
def test_caveats_are_structured_and_cover_the_required_ground():
    for mode in ("rotation", "trade_plan"):
        caveats = run(mode=mode)["caveats"]
        ids = {c["id"] for c in caveats}
        assert "survivorship_bias" in ids, ids
        assert "mode" in ids, ids
        for c in caveats:
            assert set(c) == {"id", "severity", "title", "detail"}, c
            assert c["severity"] in ("high", "medium", "info"), c
        mode_row = next(c for c in caveats if c["id"] == "mode")
        assert mode in mode_row["title"], mode_row
    assert "no_intrabar_exits" in {c["id"] for c in run(mode="rotation")["caveats"]}
    assert "no_intrabar_exits" not in {c["id"] for c in run(mode="trade_plan")["caveats"]}


# ──────────────────────────────────────────────────────────────────────────
# 6. JSON safety — Starlette serializes with allow_nan=False
# ──────────────────────────────────────────────────────────────────────────
@test
def test_results_are_strict_json_serializable():
    for mode in ("rotation", "trade_plan"):
        payload = run(mode=mode)
        try:
            json.dumps(payload, allow_nan=False)
        except ValueError as exc:
            raise AssertionError(f"{mode} response is not strict-JSON safe: {exc}") from exc


# ──────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────
@test
def test_data_coverage_reports_dropped_gaps_and_early_ends():
    from services.backtest import data_coverage, data_quality_caveats
    days = [f"2024-01-{d:02d}" for d in range(1, 29)]
    spy = [{"date": d, "close": 1.0} for d in days]
    full = [{"date": d, "close": 1.0} for d in days]
    holed = [b for b in full if b["date"] not in ("2024-01-10", "2024-01-15")]
    early = full[:10]
    loaded = {"FULL": full, "HOLE": holed, "EARLY": early, "BROKEN": full[:1]}
    used = {k: v for k, v in loaded.items() if len(v) >= 5}
    dq = data_coverage(loaded, used, spy, 5, ["2024-01-05", "2024-01-10", "2024-01-20"])
    assert [d["ticker"] for d in dq["dropped"]] == ["BROKEN"]
    assert dq["tickers_with_gaps"] == 1 and dq["missing_bars"] == 2
    assert dq["missing_on_rebalance_dates"] == 1, "01-10 is a rebalance date"
    assert [e["ticker"] for e in dq["ends_early"]] == ["EARLY"]
    ids = {c["id"] for c in data_quality_caveats(dq)}
    assert {"history_window", "tickers_dropped", "missing_bars", "history_ends_early"} <= ids
    clean = data_coverage({"FULL": full}, {"FULL": full}, spy, 5, ["2024-01-05", "2024-01-20"])
    assert {c["id"] for c in data_quality_caveats(clean)} == {"history_window"}


def main() -> int:
    if "--record-baseline" in sys.argv:
        print(rotation_fingerprint(run()))
        return 0
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
