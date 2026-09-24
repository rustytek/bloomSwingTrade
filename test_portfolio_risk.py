"""
Pure-function, no-network tests for portfolio-level risk management.

Covers:
  1. services/portfolio_risk.py::correlation_matrix / pair_correlation
     (returns-not-prices, date alignment, minimum overlap)
  2. services/portfolio_risk.py::open_risk        (free roll, missing stop)
  3. services/portfolio_risk.py::concentration / portfolio_heat
  4. services/portfolio_risk.py::assess_new_position (block vs warn thresholds)
  5. services/trade_plan.py::build_trade_plan edge_multiplier
     (default 1.0 reproduces the pre-change output exactly; clamping at both ends)

Run:
    python test_portfolio_risk.py

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
from datetime import date, timedelta

# ──────────────────────────────────────────────────────────────────────────
# Web-dependency stubs (mirrors test_passes.py). Appended to the END of
# sys.meta_path so a real install always wins; inert when the packages exist.
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


sys.meta_path.append(_StubFinder())

from services import portfolio_risk as pr
from services.trade_plan import (
    EDGE_MULTIPLIER_MAX,
    EDGE_MULTIPLIER_MIN,
    build_trade_plan,
    clamp_edge_multiplier,
)


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
# Synthetic data builders (deterministic)
# ──────────────────────────────────────────────────────────────────────────
def _dates(n: int, start: date = date(2024, 1, 1)) -> list[str]:
    """n consecutive weekday-ish date strings (calendar days are fine here)."""
    return [(start + timedelta(days=i)).isoformat() for i in range(n)]


def bars_from_closes(closes, start: date = date(2024, 1, 1)) -> list[dict]:
    ds = _dates(len(closes), start)
    out = []
    for d, c in zip(ds, closes):
        out.append({"date": d, "open": c, "high": c * 1.01, "low": c * 0.99,
                    "close": c, "vol": 1_000_000})
    return out


def random_walk(n: int, seed: int, start: float = 100.0, drift: float = 0.0008,
                vol: float = 0.012) -> list[float]:
    """An INDEPENDENT rising random walk — same drift, unrelated shocks."""
    rng = random.Random(seed)
    price = start
    out = []
    for _ in range(n):
        price *= (1.0 + drift + rng.gauss(0.0, vol))
        out.append(price)
    return out


def plan_bars(n=80, seed=7, start=100.0, step=0.5) -> list[dict]:
    """The exact bar series used to pin build_trade_plan's baseline output."""
    rng = random.Random(seed)
    bars = []
    p = start
    for _ in range(n):
        p += step
        noise = rng.uniform(-0.4, 0.4)
        c = p + noise
        o = c - 0.25
        bars.append({"open": o, "high": max(o, c) + 0.6, "low": min(o, c) - 0.6,
                     "close": c, "vol": 1_000_000})
    return bars


def pos(ticker, shares, avg_cost, stop=None):
    return {"ticker": ticker, "shares": shares, "avg_cost": avg_cost, "stop_loss": stop}


def quote(ticker, price, sector="Technology", beta=1.0):
    return {"ticker": ticker, "price": price, "sector": sector, "beta": beta}


# ══════════════════════════════════════════════════════════════════════════
# 1. Correlation
# ══════════════════════════════════════════════════════════════════════════
@test
def test_identical_series_correlate_one():
    closes = random_walk(120, seed=1)
    h = {"AAA": bars_from_closes(closes), "BBB": bars_from_closes(closes)}
    m = pr.correlation_matrix(h)
    c = m["matrix"]["AAA"]["BBB"]
    assert c is not None and approx(c, 1.0, 1e-9), f"identical series -> {c}"
    assert m["matrix"]["AAA"]["AAA"] == 1.0


@test
def test_negated_returns_correlate_minus_one():
    """A series whose DAILY RETURNS are the exact negation of another."""
    base = random_walk(120, seed=2)
    rets = [base[i] / base[i - 1] - 1.0 for i in range(1, len(base))]
    mirror = [100.0]
    for r in rets:
        mirror.append(mirror[-1] * (1.0 - r))
    h = {"AAA": bars_from_closes(base), "BBB": bars_from_closes(mirror)}
    c = pr.correlation_matrix(h)["matrix"]["AAA"]["BBB"]
    assert c is not None and approx(c, -1.0, 1e-6), f"negated series -> {c}"


@test
def test_returns_not_prices_independent_walks():
    """THE bug this module must avoid: two independent RISING series must not
    look like the same trade. On price LEVELS they correlate ~0.9+; on returns
    they must land near zero."""
    # Strong shared drift, unrelated shocks: both series rise ~4x over the
    # window, so PRICE LEVELS correlate ~0.99 while RETURNS do not.
    a = random_walk(250, seed=11, drift=0.006, vol=0.010)
    b = random_walk(250, seed=99, drift=0.006, vol=0.010)
    c = pr.correlation_matrix({"AAA": bars_from_closes(a), "BBB": bars_from_closes(b)})
    corr = c["matrix"]["AAA"]["BBB"]
    assert corr is not None
    assert abs(corr) < 0.3, f"independent walks correlated at {corr} — prices, not returns?"
    # And prove the levels really would have been misleading:
    lvl = pr._pearson(a, b)
    assert lvl is not None and lvl > 0.7, f"price-level corr only {lvl}; test is not probing the bug"
    assert not c["high_pairs"]


@test
def test_date_alignment_partial_overlap():
    """Different start dates: the correct answer uses only the SHARED dates.
    Index-zipping would compare mismatched days and give a different number."""
    closes_a = random_walk(200, seed=5)
    bars_a = bars_from_closes(closes_a, start=date(2024, 1, 1))
    # BBB starts 100 calendar days later and shares the last 100 dates, with
    # closes derived from AAA's SAME-DATE returns -> true correlation 1.0.
    shared_dates = [b["date"] for b in bars_a[100:]]
    shared_rets = [closes_a[i] / closes_a[i - 1] - 1.0 for i in range(100, len(closes_a))]
    price = 50.0
    bars_b = []
    for d, r in zip(shared_dates, shared_rets):
        price *= (1.0 + r)
        bars_b.append({"date": d, "open": price, "high": price, "low": price,
                       "close": price, "vol": 1})
    c = pr.correlation_matrix({"AAA": bars_a, "BBB": bars_b})["matrix"]["AAA"]["BBB"]
    assert c is not None and approx(c, 1.0, 1e-6), f"date-aligned overlap -> {c}"

    # Same BBB, but shifted one day off AAA's dates -> no shared dates at all.
    bars_b_shift = [dict(b, date=(date.fromisoformat(b["date"]) + timedelta(days=400)).isoformat())
                    for b in bars_b]
    c2 = pr.correlation_matrix({"AAA": bars_a, "BBB": bars_b_shift})["matrix"]["AAA"]["BBB"]
    assert c2 is None, f"disjoint dates should be None, got {c2}"


@test
def test_below_minimum_overlap_returns_none():
    closes = random_walk(200, seed=6)
    bars_a = bars_from_closes(closes)
    # Only 10 shared dates -> below MIN_OVERLAP (30) -> None, not a garbage number.
    bars_b = bars_from_closes(closes[:11], start=date(2024, 1, 1))
    m = pr.correlation_matrix({"AAA": bars_a, "BBB": bars_b})
    assert m["matrix"]["AAA"]["BBB"] is None
    assert ["AAA", "BBB"] in m["insufficient"]
    assert pr.MIN_OVERLAP == 30


@test
def test_flat_series_has_no_correlation():
    flat = bars_from_closes([100.0] * 120)
    other = bars_from_closes(random_walk(120, seed=8))
    c = pr.correlation_matrix({"AAA": flat, "BBB": other})["matrix"]["AAA"]["BBB"]
    assert c is None, f"zero-variance series should be None, got {c}"


@test
def test_correlation_window_is_respected():
    """Only the most recent `window` shared dates are used."""
    n = 300
    a = random_walk(n, seed=21)
    b = random_walk(n, seed=22)
    full = pr.correlation_matrix({"A": bars_from_closes(a), "B": bars_from_closes(b)},
                                 window=290)["matrix"]["A"]["B"]
    short = pr.correlation_matrix({"A": bars_from_closes(a), "B": bars_from_closes(b)},
                                  window=40)["matrix"]["A"]["B"]
    assert full is not None and short is not None
    assert full != short, "window had no effect"


@test
def test_nan_and_junk_bars_are_dropped():
    closes = random_walk(120, seed=9)
    bars = bars_from_closes(closes)
    bars.append({"date": "2024-09-01", "close": float("nan")})
    bars.append({"date": None, "close": 100.0})
    bars.append({"date": "2024-09-02", "close": 0})
    rets = pr.daily_returns(bars)
    assert all(math.isfinite(v) for v in rets.values())
    assert len(rets) == len(closes) - 1


# ══════════════════════════════════════════════════════════════════════════
# 2. Open risk
# ══════════════════════════════════════════════════════════════════════════
@test
def test_open_risk_basic_and_r_units():
    positions = [pos("AAA", 100, 50.0, stop=45.0)]
    quotes = [quote("AAA", 52.0)]
    r = pr.open_risk(positions, quotes, risk_unit=700.0)
    assert approx(r["open_risk_dollars"], 700.0), r["open_risk_dollars"]
    assert approx(r["open_r"], 1.0), r["open_r"]
    assert r["positions_at_risk"] == 1
    assert r["positions_without_stop"] == []


@test
def test_open_risk_free_roll_floors_at_zero():
    """Stop trailed ABOVE the current price: the trade cannot lose — risk is 0,
    never a negative number that would offset real risk elsewhere."""
    positions = [pos("AAA", 100, 50.0, stop=60.0)]
    quotes = [quote("AAA", 55.0)]
    r = pr.open_risk(positions, quotes, risk_unit=1000.0)
    assert r["open_risk_dollars"] == 0.0
    assert r["open_r"] == 0.0
    assert r["positions_at_risk"] == 0
    assert r["positions"][0]["free_roll"] is True


@test
def test_open_risk_missing_stop_is_flagged_not_zero():
    positions = [pos("AAA", 100, 50.0, stop=45.0), pos("BBB", 10, 200.0, stop=None)]
    quotes = [quote("AAA", 52.0), quote("BBB", 210.0)]
    r = pr.open_risk(positions, quotes, risk_unit=700.0)
    assert r["positions_without_stop"] == ["BBB"]
    assert r["unstopped_count"] == 1
    assert "unbounded" in r["unstopped_warning"]
    assert approx(r["open_risk_dollars"], 700.0), "unstopped risk must not be counted as 0 dollars"
    bbb = [d for d in r["positions"] if d["ticker"] == "BBB"][0]
    assert bbb["unbounded"] is True and bbb["risk_dollars"] is None


@test
def test_open_risk_nan_quote_is_safe():
    positions = [pos("AAA", 100, 50.0, stop=45.0)]
    quotes = [{"ticker": "AAA", "price": float("nan"), "sector": "Tech", "beta": float("nan")}]
    r = pr.open_risk(positions, quotes, risk_unit=500.0)
    # price falls back to avg_cost; nothing NaN escapes (Starlette allow_nan=False)
    assert approx(r["open_risk_dollars"], 500.0)
    c = pr.concentration(positions, quotes)
    assert c["weighted_beta"] is None
    assert all(isinstance(v, float) and math.isfinite(v) for v in c["sector_weights"].values())


# ══════════════════════════════════════════════════════════════════════════
# 3. Concentration + heat
# ══════════════════════════════════════════════════════════════════════════
@test
def test_concentration_weights_and_beta():
    positions = [pos("AAA", 100, 50.0), pos("BBB", 100, 50.0), pos("CCC", 100, 50.0)]
    quotes = [quote("AAA", 100.0, "Technology", 1.5),
              quote("BBB", 100.0, "Technology", 1.5),
              quote("CCC", 100.0, "Utilities", 0.6)]
    c = pr.concentration(positions, quotes)
    assert approx(c["sector_weights"]["Technology"], 66.67, 0.01)
    assert approx(c["sector_weights"]["Utilities"], 33.33, 0.01)
    assert approx(c["weighted_beta"], 1.2, 0.001)
    assert [f["sector"] for f in c["flagged_sectors"]] == ["Technology", "Utilities"]


@test
def test_portfolio_heat_budget_and_slots():
    positions = [pos("AAA", 100, 50.0, 45.0), pos("BBB", 100, 50.0, 45.0)]
    quotes = [quote("AAA", 50.0), quote("BBB", 50.0)]
    budget = pr.implied_max_open_r(8, 1.0)          # 8 slots x 1% = 8R
    assert approx(budget, 8.0)
    h = pr.portfolio_heat(positions, quotes, budget, risk_unit=500.0, max_positions=8)
    assert approx(h["open_r"], 2.0)                  # 2 x $500 = 1R each
    assert approx(h["heat_pct"], 25.0)
    assert h["over_budget"] is False
    assert h["slots_used"] == 2 and h["slots_available"] == 6
    assert h["at_max_positions"] is False


# ══════════════════════════════════════════════════════════════════════════
# 4. assess_new_position
# ══════════════════════════════════════════════════════════════════════════
_SETTINGS = {"account_size": 100_000.0, "risk_pct": 1.0, "max_positions": 8}


def _corr_histories(n=150, seed_a=31, clone_noise=0.0):
    """AAA plus a NEW ticker whose returns are AAA's (optionally + noise)."""
    a = random_walk(n, seed=seed_a)
    rets = [a[i] / a[i - 1] - 1.0 for i in range(1, len(a))]
    rng = random.Random(777)
    p = 80.0
    clone = [p]
    for r in rets:
        p *= (1.0 + r + (rng.gauss(0.0, clone_noise) if clone_noise else 0.0))
        clone.append(p)
    return bars_from_closes(a), bars_from_closes(clone)


@test
def test_assess_blocks_on_high_correlation():
    a_bars, new_bars = _corr_histories(clone_noise=0.0)     # corr == 1.0 >= 0.85
    positions = [pos("AAA", 10, 100.0, 90.0)]
    quotes = [quote("AAA", 100.0, "Technology", 1.0), quote("NEW", 80.0, "Healthcare", 1.0)]
    plan = {"entry": 80.0, "stop": 72.0, "shares": 10}
    res = pr.assess_new_position("NEW", plan, positions, quotes,
                                 {"AAA": a_bars, "NEW": new_bars}, _SETTINGS)
    corr_blocks = [w for w in res["warnings"]
                   if w["code"] == "correlation" and w["level"] == "block"]
    assert corr_blocks, res["warnings"]
    assert corr_blocks[0]["corr"] >= pr.CORR_BLOCK
    assert res["ok"] is False and res["verdict"] == "block"


@test
def test_assess_warns_in_the_correlation_warn_band():
    """Tuned so the pair lands between CORR_WARN (0.70) and CORR_BLOCK (0.85)."""
    a_bars, new_bars = _corr_histories(clone_noise=0.010)
    positions = [pos("AAA", 10, 100.0, 90.0)]
    quotes = [quote("AAA", 100.0, "Technology", 1.0), quote("NEW", 80.0, "Healthcare", 1.0)]
    plan = {"entry": 80.0, "stop": 72.0, "shares": 10}
    res = pr.assess_new_position("NEW", plan, positions, quotes,
                                 {"AAA": a_bars, "NEW": new_bars}, _SETTINGS)
    c = [row for row in res["correlations"] if row["ticker"] == "AAA"][0]["corr"]
    assert pr.CORR_WARN <= c < pr.CORR_BLOCK, f"corr {c} not in the warn band; retune the test"
    levels = {w["level"] for w in res["warnings"] if w["code"] == "correlation"}
    assert levels == {"warn"}, res["warnings"]


@test
def test_assess_sector_block_and_before_after():
    """Two Tech names held, a third Tech add pushes the sector past 40%."""
    positions = [pos("AAA", 100, 100.0, 90.0), pos("BBB", 100, 100.0, 90.0),
                 pos("CCC", 100, 100.0, 90.0)]
    quotes = [quote("AAA", 100.0, "Technology"), quote("BBB", 100.0, "Utilities"),
              quote("CCC", 100.0, "Energy"), quote("NEW", 100.0, "Technology")]
    plan = {"entry": 100.0, "stop": 95.0, "shares": 100}
    # Fully invested account: the book IS the account, so book weight = account weight.
    res = pr.assess_new_position("NEW", plan, positions, quotes, {},
                                 dict(_SETTINGS, account_size=40000))
    sect = [w for w in res["warnings"] if w["code"] == "sector_concentration"]
    assert sect and sect[0]["level"] == "block", res["warnings"]
    assert approx(res["before"]["sector_weights"]["Technology"], 33.33, 0.01)
    assert approx(res["after"]["sector_weights"]["Technology"], 50.0, 0.01)
    assert res["thresholds"]["sector_block_pct"] == pr.SECTOR_BLOCK_PCT


@test
def test_assess_sector_warn_band():
    positions = [pos("AAA", 100, 100.0, 90.0), pos("BBB", 100, 100.0, 90.0),
                 pos("CCC", 100, 100.0, 90.0)]
    quotes = [quote("AAA", 100.0, "Utilities"), quote("BBB", 100.0, "Energy"),
              quote("CCC", 100.0, "Staples"), quote("NEW", 100.0, "Technology")]
    plan = {"entry": 100.0, "stop": 95.0, "shares": 150}   # 15k/45k = 33.3% Tech after
    res = pr.assess_new_position("NEW", plan, positions, quotes, {},
                                 dict(_SETTINGS, account_size=45000))
    sect = [w for w in res["warnings"] if w["code"] == "sector_concentration"]
    assert sect and sect[0]["level"] == "warn", res["warnings"]
    assert pr.SECTOR_WARN_PCT <= sect[0]["after_pct"] < pr.SECTOR_BLOCK_PCT


@test
def test_first_position_in_empty_book_is_not_a_sector_block():
    """Regression: sector weight used to be measured against the invested book
    only, so the FIRST trade in an empty account was '100% of the book' and was
    blocked — a new account could never open anything. It is now measured
    against max(book, account_size)."""
    quotes = [quote("NEW", 100.0, "Technology")]
    plan = {"entry": 100.0, "stop": 95.0, "shares": 20}      # $2k of a $10k account
    res = pr.assess_new_position("NEW", plan, [], quotes, {},
                                 dict(_SETTINGS, account_size=10000))
    sect = [w for w in res["warnings"] if w["code"] == "sector_concentration"]
    assert not sect, res["warnings"]


@test
def test_assess_blocks_when_open_r_exceeds_budget():
    """max_positions 2 x risk_pct 1.0 = 2R budget; two 1R positions + a 1R add."""
    settings = {"account_size": 100_000.0, "risk_pct": 1.0, "max_positions": 3}
    positions = [pos("AAA", 100, 50.0, 40.0), pos("BBB", 100, 50.0, 40.0)]
    quotes = [quote("AAA", 50.0, "Utilities"), quote("BBB", 50.0, "Energy"),
              quote("NEW", 50.0, "Staples")]
    plan = {"entry": 50.0, "stop": 40.0, "shares": 200}    # $2000 = 2R
    res = pr.assess_new_position("NEW", plan, positions, quotes, {}, settings)
    budget = [w for w in res["warnings"] if w["code"] == "open_risk_budget"]
    assert budget and budget[0]["level"] == "block", res["warnings"]
    assert approx(res["before"]["open_r"], 2.0)
    assert approx(res["after"]["open_r"], 4.0)
    assert res["max_open_r"] == 3.0


@test
def test_assess_blocks_when_book_is_full():
    settings = {"account_size": 100_000.0, "risk_pct": 1.0, "max_positions": 2}
    positions = [pos("AAA", 10, 50.0, 45.0), pos("BBB", 10, 50.0, 45.0)]
    quotes = [quote("AAA", 50.0, "Utilities"), quote("BBB", 50.0, "Energy"),
              quote("NEW", 50.0, "Staples")]
    plan = {"entry": 50.0, "stop": 45.0, "shares": 1}
    res = pr.assess_new_position("NEW", plan, positions, quotes, {}, settings)
    codes = [w["code"] for w in res["warnings"] if w["level"] == "block"]
    assert "max_positions" in codes, res["warnings"]


@test
def test_assess_clean_trade_has_no_blocks():
    settings = {"account_size": 100_000.0, "risk_pct": 1.0, "max_positions": 8}
    positions = [pos("AAA", 200, 50.0, 45.0), pos("BBB", 200, 50.0, 45.0)]
    a_bars = bars_from_closes(random_walk(150, seed=41))
    b_bars = bars_from_closes(random_walk(150, seed=43))
    n_bars = bars_from_closes(random_walk(150, seed=42))
    quotes = [quote("AAA", 50.0, "Utilities", 0.7), quote("BBB", 50.0, "Energy", 0.8),
              quote("NEW", 50.0, "Healthcare", 0.9)]
    plan = {"entry": 50.0, "stop": 45.0, "shares": 100}     # $500 = 0.5R, 20% weight
    res = pr.assess_new_position("NEW", plan, positions, quotes,
                                 {"AAA": a_bars, "BBB": b_bars, "NEW": n_bars}, settings)
    assert res["ok"] is True, res["warnings"]
    assert res["verdict"] == "ok", res["warnings"]
    assert res["block_count"] == 0
    assert approx(res["before"]["open_r"], 2.0)
    assert approx(res["after"]["open_r"], 2.5)


@test
def test_assess_blocks_plan_without_stop():
    plan = {"entry": 50.0, "stop": None, "shares": 100}
    res = pr.assess_new_position("NEW", plan, [], [quote("NEW", 50.0)], {}, _SETTINGS)
    assert any(w["code"] == "no_stop" and w["level"] == "block" for w in res["warnings"])


# ══════════════════════════════════════════════════════════════════════════
# 5. Edge-weighted sizing
# ══════════════════════════════════════════════════════════════════════════
# Captured from build_trade_plan BEFORE edge_multiplier existed, on plan_bars().
_BASELINE = {
    "entry": 139.97, "entry_zone": [139.24, 140.7], "stop": 136.32, "stop_pct": -2.61,
    "target": 147.28, "target_pct": 5.22, "atr": 1.461, "atr_mult": 2.5,
    "risk_per_share": 3.65, "risk_dollars": 650.23, "shares": 178,
    "position_value": 24914.55, "position_pct": 24.91, "r_multiple": 2.0,
    "capped_by_max_position": True,
}


@test
def test_edge_multiplier_default_reproduces_baseline_exactly():
    bars = plan_bars()
    default = build_trade_plan(bars, 100_000, 1.0)
    explicit = build_trade_plan(bars, 100_000, 1.0, edge_multiplier=1.0)
    assert default == explicit
    for k, v in _BASELINE.items():
        assert default[k] == v, f"{k}: {default[k]!r} != pre-change {v!r}"
    # Only additive keys may appear.
    assert set(default) - set(_BASELINE) == {"initial_stop", "edge_multiplier"}
    assert default["edge_multiplier"] == 1.0


@test
def test_initial_stop_mirrors_stop():
    bars = plan_bars()
    plan = build_trade_plan(bars, 100_000, 1.0)
    assert plan["initial_stop"] == plan["stop"]


@test
def test_edge_multiplier_scales_risk_budget():
    """Uncapped case: shares must scale with the multiplier."""
    bars = plan_bars()
    base = build_trade_plan(bars, 100_000, 0.5, max_position_pct=100.0)
    up = build_trade_plan(bars, 100_000, 0.5, max_position_pct=100.0, edge_multiplier=1.5)
    down = build_trade_plan(bars, 100_000, 0.5, max_position_pct=100.0, edge_multiplier=0.5)
    assert up["shares"] > base["shares"] > down["shares"]
    assert up["shares"] == math.floor(1.5 * 100_000 * 0.005 / base["risk_per_share"])
    assert down["shares"] == math.floor(0.5 * 100_000 * 0.005 / base["risk_per_share"])


@test
def test_edge_multiplier_clamps_at_both_ends():
    assert clamp_edge_multiplier(99.0) == EDGE_MULTIPLIER_MAX
    assert clamp_edge_multiplier(-3.0) == EDGE_MULTIPLIER_MIN
    assert clamp_edge_multiplier(0.0) == EDGE_MULTIPLIER_MIN
    assert clamp_edge_multiplier(float("nan")) == 1.0
    assert clamp_edge_multiplier(None) == 1.0
    assert clamp_edge_multiplier(1.25) == 1.25

    bars = plan_bars()
    hi = build_trade_plan(bars, 100_000, 0.5, max_position_pct=100.0, edge_multiplier=10.0)
    at_max = build_trade_plan(bars, 100_000, 0.5, max_position_pct=100.0,
                              edge_multiplier=EDGE_MULTIPLIER_MAX)
    lo = build_trade_plan(bars, 100_000, 0.5, max_position_pct=100.0, edge_multiplier=0.01)
    at_min = build_trade_plan(bars, 100_000, 0.5, max_position_pct=100.0,
                              edge_multiplier=EDGE_MULTIPLIER_MIN)
    assert hi == at_max, "multiplier above the band must clamp, not scale"
    assert lo == at_min, "multiplier below the band must clamp, not scale"
    assert hi["edge_multiplier"] == EDGE_MULTIPLIER_MAX
    assert lo["edge_multiplier"] == EDGE_MULTIPLIER_MIN


@test
def test_implied_max_open_r_guards():
    assert pr.implied_max_open_r(8, 1.0) == 8.0
    assert pr.implied_max_open_r(0, 1.0) is None
    assert pr.implied_max_open_r(8, None) is None
    assert pr.implied_max_open_r(float("nan"), 1.0) is None


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
