"""
No-network tests for the Weekly Plan decision layer and strategy rationale.

Covers:
  1. services/weekly_plan.py::_review_position — stop hit / target hit / trend
     break / overstayed horizon / trailing-stop suggestion, and the sell-order
     contract the Trade page consumes.
  2. services/weekly_plan.py::_buy_orders — the risk-aware greedy pick: a buy
     that is highly correlated with one ALREADY PICKED this week is blocked,
     forming setups and mis-tagged strategies are not pre-selected, and the
     weekly buy cap holds.
  3. services/strategy_rationale.py — every registered strategy has written
     rationale for every regime quadrant.

Run:
    python test_weekly_plan.py
"""
from __future__ import annotations

import random
import sys
import traceback
from datetime import date, timedelta
from types import SimpleNamespace

from services import weekly_plan as wp
from services.regime import QUADRANTS
from services.strategies import STRATEGIES
from services.strategy_rationale import RATIONALE, rationale_for

_TESTS = []


def test(fn):
    _TESTS.append(fn)
    return fn


ORDER_KEYS = {"id", "kind", "side", "ticker", "shares", "limit_price", "est_value",
              "recommended", "skip_reason", "headline", "why", "strategy", "plan"}
SETTINGS = {"account_size": 10000.0, "risk_pct": 1.0, "max_positions": 8,
            "atr_stop_mult": 2.5, "r_multiple": 2.0, "max_open_r": 8.0}


def _pos_row(**kw):
    row = {"ticker": "AAA", "name": "Aaa Inc", "shares": 10, "avg_cost": 100.0, "price": 100.0,
           "pnl_pct": 0.0, "stop_loss": 95.0, "target": 110.0, "entry_date": None,
           "strategy": "pullback_50ma", "open_r": 0.0, "status": "ok", "actions": [],
           "suggested_stop": None}
    row.update(kw)
    return row


def _bars(seed: int, n: int = 260, rets=None):
    rng = random.Random(seed)
    start = date(2025, 1, 1)
    price = 100.0
    out = []
    for i in range(n):
        r = rets[i] if rets is not None else rng.gauss(0.0005, 0.015)
        price *= 1 + r
        out.append({"date": (start + timedelta(days=i)).isoformat(), "open": price,
                    "high": price * 1.01, "low": price * 0.99, "close": price, "volume": 1e6})
    return out


def _setup(ticker, strategy="momentum_rotation", state="triggered", shares=10, entry=100.0):
    return {
        "ticker": ticker, "name": ticker + " Corp", "sector": "Tech-" + ticker,
        "strategy": strategy, "strategy_name": STRATEGIES[strategy].name, "state": state,
        "score": 1.0, "reasons": ["a reason"], "rank": 1, "pool_size": 5, "actionable": True,
        "plan": {"entry": entry, "entry_zone": [entry - 1, entry + 1], "stop": entry - 5,
                 "initial_stop": entry - 5, "stop_pct": -5.0, "target": entry + 10,
                 "target_pct": 10.0, "atr": 2.0, "atr_mult": 2.5, "risk_per_share": 5.0,
                 "risk_dollars": 5.0 * shares, "shares": shares, "position_value": entry * shares,
                 "position_pct": 10.0, "r_multiple": 2.0, "capped_by_max_position": False,
                 "edge_multiplier": 1.0},
    }


def _cache(tickers_bars):
    return {t: {"bars": b, "quote": {"ticker": t, "price": b[-1]["close"], "sector": "Tech-" + t}}
            for t, b in tickers_bars.items()}


def _today(setups):
    return {"setups": setups, "strategy_regime_status": [
        {"id": s, "rationale": rationale_for(s, "trending_bull")} for s in STRATEGIES]}


# ── 1. Position reviews ──────────────────────────────────────────────────────

@test
def test_stop_hit_is_a_recommended_full_sell():
    review, order = wp._review_position(_pos_row(status="stop_hit", price=94.0), None)
    assert review["verdict"] == "sell"
    assert order and order["recommended"] and order["kind"] == "sell" and order["side"] == "sell"
    assert order["shares"] == 10
    assert order["limit_price"] == round(94.0 * 0.99, 2)
    assert ORDER_KEYS <= set(order), ORDER_KEYS - set(order)
    assert order["id"] == "sell:AAA"


@test
def test_target_hit_trims_half():
    _r, order = wp._review_position(_pos_row(status="target_hit", price=111.0, shares=9), None)
    assert order["kind"] == "trim" and order["recommended"]
    assert order["shares"] == 4
    assert order["id"] == "trim:AAA"


@test
def test_trend_break_sell_is_optional():
    _r, order = wp._review_position(_pos_row(status="trend_break"), None)
    assert order and order["recommended"] is False and order["skip_reason"]


@test
def test_own_time_stop_overstayed_is_preselected():
    entry = (date.today() - timedelta(days=60)).isoformat()
    orm = SimpleNamespace(time_stop_days=10)
    review, order = wp._review_position(_pos_row(entry_date=entry, open_r=0.1), orm)
    assert review["verdict"] == "sell" and order["recommended"] is True
    # A strategy horizon (not the user's own time stop) is offered, not pre-selected.
    _r2, order2 = wp._review_position(_pos_row(entry_date=entry, open_r=0.1),
                                      SimpleNamespace(time_stop_days=None))
    assert order2 and order2["recommended"] is False


@test
def test_trailing_stop_suggestion_without_order():
    review, order = wp._review_position(_pos_row(suggested_stop=97.5), None)
    assert review["verdict"] == "raise_stop" and review["raise_stop_to"] == 97.5
    assert order is None


@test
def test_healthy_position_holds():
    review, order = wp._review_position(_pos_row(), None)
    assert review["verdict"] == "hold" and order is None and review["why"]


# ── 2. Buys ──────────────────────────────────────────────────────────────────

@test
def test_correlated_second_buy_is_blocked():
    base = _bars(1)
    twin_rets = [(base[i]["close"] / base[i - 1]["close"] - 1) if i else 0.0 for i in range(len(base))]
    cache = _cache({"AAA": base, "BBB": _bars(1, rets=twin_rets), "CCC": _bars(7)})
    today = _today([_setup("AAA"), _setup("BBB"), _setup("CCC")])
    orders = wp._buy_orders(today, cache, [], set(), SETTINGS, {})
    rec = {o["ticker"]: o["recommended"] for o in orders}
    assert rec == {"AAA": True, "BBB": False, "CCC": True}, rec
    bbb = next(o for o in orders if o["ticker"] == "BBB")
    assert "correlated" in bbb["skip_reason"], bbb["skip_reason"]
    assert all(ORDER_KEYS <= set(o) for o in orders)


@test
def test_forming_and_mistagged_not_preselected():
    cache = _cache({"AAA": _bars(2), "BBB": _bars(3)})
    today = _today([_setup("AAA", state="forming"), _setup("BBB", strategy="pullback_50ma")])
    cells = {"pullback_50ma": {"verdict": "mis-tagged", "n": 120}}
    orders = {o["ticker"]: o for o in wp._buy_orders(today, cache, [], set(), SETTINGS, cells)}
    assert orders["AAA"]["recommended"] is False and "FORMING" in orders["AAA"]["skip_reason"]
    assert orders["BBB"]["recommended"] is False and "not worked" in orders["BBB"]["skip_reason"]


@test
def test_weekly_buy_cap():
    tickers = [f"T{i}" for i in range(wp.MAX_NEW_BUYS_PER_WEEK + 2)]
    cache = _cache({t: _bars(10 + i) for i, t in enumerate(tickers)})
    settings = dict(SETTINGS, max_positions=50, max_open_r=50.0)
    orders = wp._buy_orders(_today([_setup(t) for t in tickers]), cache, [], set(), settings, {})
    assert sum(o["recommended"] for o in orders) == wp.MAX_NEW_BUYS_PER_WEEK


@test
def test_full_book_blocks_buys():
    cache = _cache({"AAA": _bars(4), "HELD": _bars(5)})
    held = [SimpleNamespace(ticker="HELD", shares=10, avg_cost=100.0, stop_loss=95.0)]
    orders = wp._buy_orders(_today([_setup("AAA")]), cache, held, set(),
                            dict(SETTINGS, max_positions=1), {})
    assert orders[0]["recommended"] is False and "max_positions" in orders[0]["skip_reason"]


@test
def test_buy_carries_plan_intent_and_limit_at_zone_top():
    cache = _cache({"AAA": _bars(6)})
    o = wp._buy_orders(_today([_setup("AAA", entry=50.0)]), cache, [], set(), SETTINGS, {})[0]
    assert o["limit_price"] == 51.0 and o["side"] == "buy" and o["id"] == "buy:AAA"
    assert o["plan"]["stop"] == 45.0 and o["plan"]["planned_entry"] == 50.0
    assert o["plan"]["planned_entry_high"] == 51.0 and o["plan"]["strategy"] == "momentum_rotation"


# ── 3. Rationale ─────────────────────────────────────────────────────────────

@test
def test_every_strategy_has_rationale_for_every_quadrant():
    for sid in STRATEGIES:
        r = RATIONALE.get(sid)
        assert r, f"no rationale for {sid}"
        for key in ("why_it_works", "best_when", "fails_when"):
            assert r.get(key), f"{sid} missing {key}"
        for q in QUADRANTS:
            assert r["regime_fit"].get(q), f"{sid} missing regime_fit[{q}]"
        assert rationale_for(sid, QUADRANTS[0])["fit_now"]


@test
def test_unknown_strategy_rationale_is_empty_not_error():
    r = rationale_for("nope", "trending_bull")
    assert r["why_it_works"] == "" and r["fit_now"] == ""


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
    sys.exit(main())
