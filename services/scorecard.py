"""
Strategy scorecard — realized vs expected, and execution quality.

TWO QUESTIONS THIS ANSWERS
--------------------------
1. EDGE DECAY. services/edge_matrix.py says what each strategy was *supposed*
   to deliver. The user's ClosedTrade journal says what it *actually*
   delivered. Nothing in the app compared them, so a strategy whose edge had
   quietly died looked exactly like one that was working. `build_scorecard`
   joins the two and reports the drift.

2. EXECUTION QUALITY. Often the strategy is fine and the *execution* is the
   leak: chasing entries, loosening stops, overstaying the horizon, trading a
   strategy in a regime it was stood down for. `execution_quality` measures the
   ones that are measurable and ranks them by what they cost in R.

HONESTY RULE (non-negotiable)
-----------------------------
If the data needed for a metric does not exist, the metric is returned with
`"available": false` and a `reason` plus the exact field that would be needed.
It is NEVER estimated, inferred from a proxy, or filled with a plausible
number. A fabricated execution number is worse than no number: the user would
act on it.

SAMPLE SIZE
-----------
A 4-trade strategy is not "broken", it is unmeasured. Nothing here emits a
negative verdict below MIN_TRADES_TO_JUDGE closed trades.
"""
from __future__ import annotations

import math
from datetime import date as _date

from sqlalchemy.orm import Session

from database.models import ClosedTrade
from services.edge_matrix import get_cached_matrix
from services.strategies import STRATEGIES
from services.universe import UNIVERSE_CAVEAT

# Below this many closed trades a strategy is "monitoring" — never "broken".
MIN_TRADES_TO_JUDGE = 10
# R-multiple is null whenever no valid stop was recorded, so avg_r is computed
# over a SUBSET. Below this subset size the drift number is reported but is not
# allowed to drive a negative recommendation.
MIN_R_SAMPLE = 8
# Drift bands, in R, applied only once both samples above are met.
DRIFT_UNDERPERFORM = -0.35
DRIFT_OUTPERFORM = 0.35


def _num(value, default=None):
    """Non-finite -> default. Starlette serializes with allow_nan=False."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _round(value, digits=2, default=None):
    f = _num(value)
    return round(f, digits) if f is not None else default


def _unavailable(metric: str, reason: str, needs: str, observed: dict | None = None) -> dict:
    """The one and only way this module reports a metric it cannot compute."""
    out = {
        "metric": metric,
        "available": False,
        "status": "UNAVAILABLE",
        "value": None,
        "r_cost": None,
        "reason": reason,
        "needs": needs,
    }
    if observed:
        out["observed"] = observed
    return out


# ──────────────────────────────────────────────────────────────────────────
# 1. Realized vs expected
# ──────────────────────────────────────────────────────────────────────────
def _realized_stats(trades: list[ClosedTrade]) -> dict:
    """Mirrors api/journal.py::_stats conventions: scratch (pnl == 0) split out
    of losses, and r_sample reported separately because avg_r only covers the
    trades that carry an r_multiple."""
    wins = [t for t in trades if (t.pnl or 0) > 0]
    losses = [t for t in trades if (t.pnl or 0) < 0]
    scratch = [t for t in trades if (t.pnl or 0) == 0]
    decided = len(wins) + len(losses)
    r_values = [_num(t.r_multiple) for t in trades if _num(t.r_multiple) is not None]
    avg_r = sum(r_values) / len(r_values) if r_values else None
    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "scratch": len(scratch),
        "win_rate": _round(len(wins) / decided * 100, 1) if decided else None,
        "avg_r": _round(avg_r, 2),
        "r_sample": len(r_values),
        "total_pnl": _round(sum(_num(t.pnl, 0.0) for t in trades), 2, 0.0),
    }


def _expected_r(matrix: dict | None, strategy_id: str) -> tuple[float | None, str]:
    """Expected R for a strategy from the edge matrix, plus its source label.

    Only trade_plan-mode runs produce a real per-trade expectancy (stops,
    targets and sizing simulated). A rotation-mode run has no per-trade R at
    all, so we return None rather than converting a period return into a
    pseudo-R — that conversion would be a fabricated number.
    """
    if not matrix:
        return None, "no edge matrix computed yet"
    entry = (matrix.get("strategies") or {}).get(strategy_id)
    if not entry:
        return None, "strategy not present in the edge matrix"
    overall = entry.get("overall") or {}
    exp = _num(overall.get("expectancy_r"))
    if exp is None:
        exp = _num(overall.get("avg_r"))
        if exp is not None:
            return exp, "edge matrix avg_r (trade_plan mode)"
        return None, (
            f"edge matrix was built in '{matrix.get('mode')}' mode, which has no per-trade "
            "R — rebuild with mode=trade_plan for an expected R"
        )
    return exp, "edge matrix expectancy_r (trade_plan mode)"


def _recommendation(stats: dict, expected: float | None, drift: float | None) -> dict:
    n = stats["trades"]
    if n < MIN_TRADES_TO_JUDGE:
        return {
            "verdict": "monitoring",
            "text": (f"{n} closed trade(s) — too few to judge. Keep the strategy running and "
                     f"revisit at {MIN_TRADES_TO_JUDGE}."),
        }
    if expected is None or drift is None:
        return {
            "verdict": "no_benchmark",
            "text": ("No expected R to compare against yet — build the edge matrix in "
                     "trade_plan mode, then this becomes a real drift check."),
        }
    if stats["r_sample"] < MIN_R_SAMPLE:
        return {
            "verdict": "monitoring",
            "text": (f"Only {stats['r_sample']} of {n} trades carry an R-multiple, so the "
                     "drift is measured on a thin subset. Record a stop on every entry to "
                     "make this meaningful."),
        }
    if drift <= DRIFT_UNDERPERFORM:
        return {
            "verdict": "underperforming",
            "text": (f"Realized {stats['avg_r']}R vs {round(expected, 2)}R expected "
                     f"({round(drift, 2)}R drift) over {stats['r_sample']} trades. Either the "
                     "edge has decayed or execution is leaking — check execution_quality "
                     "before dropping the strategy."),
        }
    if drift >= DRIFT_OUTPERFORM:
        return {
            "verdict": "outperforming",
            "text": (f"Realized {stats['avg_r']}R vs {round(expected, 2)}R expected "
                     f"(+{round(drift, 2)}R). Above backtest — likely favourable regime or "
                     "selection; do not size up on this alone."),
        }
    return {
        "verdict": "on_track",
        "text": (f"Realized {stats['avg_r']}R vs {round(expected, 2)}R expected "
                 f"({round(drift, 2)}R drift) — tracking the backtest."),
    }


def build_scorecard(db: Session, user_id: int) -> dict:
    """Per-strategy realized results joined against edge-matrix expectancy."""
    trades = db.query(ClosedTrade).filter(ClosedTrade.user_id == user_id).all()
    matrix = get_cached_matrix(user_id)

    by_strategy: dict[str, list[ClosedTrade]] = {}
    for t in trades:
        by_strategy.setdefault(t.strategy or "unspecified", []).append(t)

    rows = []
    for sid, group in sorted(by_strategy.items()):
        stats = _realized_stats(group)
        expected, source = _expected_r(matrix, sid)
        drift = None
        if expected is not None and stats["avg_r"] is not None:
            drift = _round(stats["avg_r"] - expected, 2)
        strat = STRATEGIES.get(sid)
        rows.append({
            "strategy": sid,
            "name": getattr(strat, "name", sid) if strat else sid,
            **stats,
            "expected_r": _round(expected, 3),
            "expected_r_source": source,
            "drift": drift,
            "recommendation": _recommendation(stats, expected, drift),
        })

    overall = _realized_stats(trades)
    return {
        "strategies": rows,
        "overall": overall,
        "edge_matrix": {
            "available": matrix is not None,
            "generated_at": (matrix or {}).get("generated_at"),
            "mode": (matrix or {}).get("mode"),
            "note": None if matrix else
                    "No edge matrix cached — call GET /api/edge-matrix?refresh=true first.",
        },
        "thresholds": {
            "min_trades_to_judge": MIN_TRADES_TO_JUDGE,
            "min_r_sample": MIN_R_SAMPLE,
            "drift_underperform": DRIFT_UNDERPERFORM,
            "drift_outperform": DRIFT_OUTPERFORM,
        },
        "caveats": [UNIVERSE_CAVEAT],
    }


# ──────────────────────────────────────────────────────────────────────────
# 2. Execution quality
# ──────────────────────────────────────────────────────────────────────────
def _chasing(trades: list[ClosedTrade]) -> dict:
    """Entries filled ABOVE the planned entry price.

    Computable since ClosedTrade gained `planned_entry` / `planned_entry_high`
    (written from services/trade_plan.build_trade_plan at commit time). The fill
    is `avg_cost`; the reference is the TOP OF THE PLANNED ZONE when one was
    recorded, otherwise the single planned entry price. Paying above that
    reference is chasing.

    HONESTY RULE — the reason this function is written the way it is:
    a row with no `planned_entry` is UNKNOWN, not "not chasing". Legacy rows and
    entries made outside the Plan-a-Trade flow carry no plan, and silently
    counting them as clean would turn a data gap into a clean bill of health.
    So only rows that HAVE the field are examined, `trades_examined` is always
    reported against `trades_total`, and when NOT ONE row has it the metric
    stays UNAVAILABLE rather than reporting "0 offenders".
    """
    usable = [t for t in trades if _num(getattr(t, "planned_entry", None)) is not None]
    missing = len(trades) - len(usable)
    if not usable:
        return _unavailable(
            "entry_chasing",
            "No closed trade carries a planned_entry, so there is nothing to compare the "
            "fill against. This is NOT a clean result — it is an absence of data.",
            "ClosedTrade.planned_entry (written from services/trade_plan.build_trade_plan "
            "when the position is committed via the Plan-a-Trade flow, and carried to the "
            "journal on close). Entries added by hand will never have one.",
            observed={"trades_examined": 0, "trades_total": len(trades),
                      "trades_missing_planned_entry": missing},
        )

    offenders, total_r_cost, overpays, unpriced = [], 0.0, [], 0
    for t in usable:
        planned = _num(t.planned_entry)
        high = _num(getattr(t, "planned_entry_high", None))
        # The zone's top is the fair reference when a zone was planned; paying
        # inside the zone is exactly what the plan asked for.
        ref = high if (high is not None and high >= planned) else planned
        fill = _num(t.avg_cost)
        if fill is None or ref is None or ref <= 0 or fill <= ref:
            continue
        overpay_pct = (fill - ref) / ref * 100.0
        overpays.append(overpay_pct)

        # R cost of the overpay, only where an R-multiple was actually recorded
        # (i.e. the trade had a valid stop, so a per-share risk exists).
        r_cost = None
        r_mult = _num(t.r_multiple)
        init = _num(t.initial_stop)
        if init is None:
            init = _num(t.stop_loss)
        risk = (fill - init) if (init is not None) else None
        if r_mult is not None and risk is not None and risk > 0:
            r_cost = (fill - ref) / risk
            if math.isfinite(r_cost):
                total_r_cost += r_cost
            else:
                r_cost = None
        if r_cost is None:
            unpriced += 1

        offenders.append({
            "id": t.id, "ticker": t.ticker, "strategy": t.strategy,
            "planned_entry": _round(planned, 4),
            "planned_entry_high": _round(high, 4),
            "reference_price": _round(ref, 4),
            "fill_price": _round(fill, 4),
            "overpay_pct": _round(overpay_pct, 2),
            "r_multiple": _round(r_mult, 2),
            "r_cost": _round(r_cost, 2),
            "r_cost_note": None if r_cost is not None else
                           "no r_multiple / positive risk recorded, R cost not computable",
        })

    avg_overpay = (sum(overpays) / len(overpays)) if overpays else None
    coverage_note = (
        f"{len(usable)} of {len(trades)} closed trades carry a planned_entry; the other "
        f"{missing} were NOT counted as clean — they have no plan to be judged against."
        if missing else
        f"All {len(usable)} closed trades carry a planned_entry."
    )
    return {
        "metric": "entry_chasing",
        "available": True,
        "status": "OK",
        "value": len(offenders),
        "r_cost": _round(total_r_cost, 2, 0.0),
        "avg_overpay_pct": _round(avg_overpay, 2),
        "trades_examined": len(usable),
        "trades_total": len(trades),
        "trades_missing_planned_entry": missing,
        "unpriced_offenders": unpriced,
        "offenders": sorted(offenders, key=lambda o: -(o["r_cost"] or 0)),
        "detail": (
            f"{len(offenders)} of {len(usable)} planned entries were filled above the plan, "
            f"by {round(avg_overpay, 2)}% on average, costing {round(total_r_cost, 2)}R."
            if offenders else
            f"No entry was filled above its planned price across {len(usable)} planned trades."
        ),
        "coverage_note": coverage_note,
    }


def _stop_loosening(trades: list[ClosedTrade]) -> dict:
    """Stops moved DOWN after entry — computable from initial_stop vs stop_loss.

    database/models.py documents `initial_stop` as "the FIRST stop ever set,
    written once and never overwritten", while `stop_loss` is the stop in force
    at exit. stop_loss < initial_stop therefore means the stop was widened
    (moved away from price) after entry — the classic way a planned 1R loss
    becomes a 2R loss.

    R cost: for a trade that exited BELOW its initial stop, the avoidable part
    of the loss is (initial_stop - exit_price) per share, expressed in R against
    the original risk (avg_cost - initial_stop).
    """
    usable = [t for t in trades
              if _num(t.initial_stop) is not None and _num(t.stop_loss) is not None]
    if not usable:
        return _unavailable(
            "stop_loosening",
            "No closed trade carries both an initial_stop and an exit stop_loss. Legacy "
            "rows predate the initial_stop column and positions closed without a stop "
            "have nothing to compare.",
            "ClosedTrade.initial_stop AND ClosedTrade.stop_loss populated — set a stop when "
            "opening the position so initial_stop is written at entry.",
            observed={"trades_examined": len(trades), "trades_with_both_stops": 0},
        )

    offenders, total_r_cost, unpriced = [], 0.0, 0
    for t in usable:
        init, final = _num(t.initial_stop), _num(t.stop_loss)
        if final >= init:
            continue  # unchanged or trailed UP — that is correct behaviour
        cost = _num(t.avg_cost)
        exitp = _num(t.exit_price)
        risk = (cost - init) if (cost is not None and init is not None) else None
        r_cost = None
        if risk and risk > 0 and exitp is not None and exitp < init:
            r_cost = (init - exitp) / risk
            total_r_cost += r_cost
        elif not risk or risk <= 0:
            unpriced += 1
        offenders.append({
            "id": t.id, "ticker": t.ticker, "strategy": t.strategy,
            "initial_stop": _round(init, 4), "final_stop": _round(final, 4),
            "exit_price": _round(exitp, 4),
            "r_cost": _round(r_cost, 2),
            "r_cost_note": None if r_cost is not None else
                           ("exited above the initial stop — the loosening cost nothing "
                            "realized" if exitp is not None and init is not None and exitp >= init
                            else "no positive initial risk recorded, R cost not computable"),
        })

    return {
        "metric": "stop_loosening",
        "available": True,
        "status": "OK",
        "value": len(offenders),
        "r_cost": _round(total_r_cost, 2, 0.0),
        "trades_examined": len(trades),
        "trades_with_both_stops": len(usable),
        "unpriced_offenders": unpriced,
        "offenders": sorted(offenders, key=lambda o: -(o["r_cost"] or 0)),
        "detail": (
            f"{len(offenders)} of {len(usable)} comparable trades had the stop moved DOWN "
            f"after entry, costing {round(total_r_cost, 2)}R of avoidable loss."
            if offenders else
            f"No stop was widened after entry across {len(usable)} comparable trades."
        ),
    }


def _horizon_days(strategy_id: str) -> int | None:
    """A MACHINE-READABLE max hold, if a strategy exposes one.

    `Strategy.details["horizon"]` is prose ("3-10 day snap-back"). Parsing a
    number out of English would be guessing, so this only reads an explicit
    numeric attribute and returns None otherwise.
    """
    strat = STRATEGIES.get(strategy_id)
    if strat is None:
        return None
    for attr in ("horizon_days", "max_hold_days"):
        val = _num(getattr(strat, attr, None))
        if val is not None and val > 0:
            return int(val)
    return None


def _overstayed(trades: list[ClosedTrade]) -> dict:
    """Positions held past the strategy's documented horizon."""
    held = []
    for t in trades:
        if t.entry_date and t.exit_date:
            held.append((t, (t.exit_date - t.entry_date).days))
    have_horizon = {t.strategy for t, _ in held if _horizon_days(t.strategy) is not None}

    if not held:
        return _unavailable(
            "overstayed_horizon",
            "No closed trade has both an entry_date and an exit_date, so hold length "
            "cannot be measured.",
            "ClosedTrade.entry_date and ClosedTrade.exit_date populated on close.",
            observed={"trades_examined": len(trades)},
        )
    if not have_horizon:
        days = [d for _, d in held]
        return _unavailable(
            "overstayed_horizon",
            "Hold length IS measurable, but no strategy exposes a machine-readable "
            "horizon — Strategy.details['horizon'] is prose (e.g. '3-10 day snap-back') "
            "and parsing a number out of it would be a guess.",
            "A numeric `horizon_days` (int) class attribute on each Strategy in "
            "services/strategies.py, alongside the existing prose details['horizon'].",
            observed={
                "trades_with_dates": len(held),
                "avg_days_held": _round(sum(days) / len(days), 1),
                "max_days_held": max(days),
            },
        )

    offenders, total_r_cost = [], 0.0
    for t, days in held:
        limit = _horizon_days(t.strategy)
        if limit is None or days <= limit:
            continue
        r = _num(t.r_multiple)
        # Only a LOSING overstayed trade has a realized R cost. A winner held
        # long is not evidence of a cost, and inventing the counterfactual
        # "what it would have made at the horizon" is not permitted here.
        r_cost = -r if (r is not None and r < 0) else None
        if r_cost:
            total_r_cost += r_cost
        offenders.append({
            "id": t.id, "ticker": t.ticker, "strategy": t.strategy,
            "days_held": days, "horizon_days": limit,
            "r_multiple": _round(r, 2), "r_cost": _round(r_cost, 2),
        })
    return {
        "metric": "overstayed_horizon",
        "available": True,
        "status": "OK",
        "value": len(offenders),
        "r_cost": _round(total_r_cost, 2, 0.0),
        "trades_examined": len(held),
        "offenders": sorted(offenders, key=lambda o: -(o["r_cost"] or 0)),
        "detail": f"{len(offenders)} trades were held past their strategy's horizon.",
        "partial_note": (
            "Only strategies exposing a numeric horizon_days were checked; the rest were "
            "skipped rather than guessed at."
        ),
    }


def _spy_quadrants(db: Session) -> dict[str, str]:
    """{date -> quadrant} from cached SPY history, or {} if not derivable.

    Mirrors services/backtest.py's per-bar regime map (same classify_quadrant /
    trend_strength_label helpers), so "which regime was I in" means the same
    thing here as it does in a backtest.
    """
    try:
        from services.backtest import _load_history
        from services.indicators import calc_adx, calc_ma
        from services.regime import classify_quadrant, trend_strength_label

        bars = _load_history(db, "SPY")
        if len(bars) < 220:
            return {}
        closes = [b["close"] for b in bars]
        ma200 = calc_ma(closes, 200)
        adx = calc_adx([b["high"] for b in bars], [b["low"] for b in bars], closes, 14)["series"]
        out: dict[str, str] = {}
        for i, bar in enumerate(bars):
            ma = ma200[i] if i < len(ma200) else None
            if ma is None:
                continue
            a = adx[i]
            adx_v = a.get("adx") if isinstance(a, dict) else None
            window = closes[max(0, i - 20):i + 1]
            crisis = False
            if len(window) >= 2:
                rets = [(window[k] / window[k - 1] - 1) for k in range(1, len(window))]
                rv = (sum(r * r for r in rets) / len(rets)) ** 0.5 * (252 ** 0.5) * 100
                crisis = rv >= 22.0
            out[bar["date"]] = classify_quadrant(
                trend_strength_label(adx_v), closes[i] > ma, crisis
            )
        return out
    except Exception:  # noqa: BLE001 — a missing/short cache is not an error path
        return {}


def _off_regime(db: Session, trades: list[ClosedTrade]) -> dict:
    """Trades taken while the strategy was stood down for that regime."""
    dated = [t for t in trades if t.entry_date and t.strategy]
    if not dated:
        return _unavailable(
            "off_regime_entries",
            "No closed trade records both an entry_date and a strategy, so the regime at "
            "entry cannot be looked up.",
            "ClosedTrade.entry_date and ClosedTrade.strategy populated on every entry.",
            observed={"trades_examined": len(trades)},
        )
    quads = _spy_quadrants(db)
    if not quads:
        return _unavailable(
            "off_regime_entries",
            "Cached SPY history is missing or shorter than the 200-day MA + ADX warmup, so "
            "the market regime on each entry date cannot be reconstructed.",
            "A StockCache row for SPY with at least ~220 bars of history_json (run a "
            "screener refresh).",
            observed={"trades_with_entry_date": len(dated)},
        )

    offenders, total_r_cost, unknown_dates = [], 0.0, 0
    for t in dated:
        key = t.entry_date.isoformat() if isinstance(t.entry_date, _date) else str(t.entry_date)
        quad = quads.get(key)
        if quad is None:
            unknown_dates += 1
            continue
        strat = STRATEGIES.get(t.strategy)
        tagged = list(getattr(strat, "regimes", []) or []) if strat else []
        if not tagged or quad in tagged:
            continue
        r = _num(t.r_multiple)
        r_cost = -r if (r is not None and r < 0) else None
        if r_cost:
            total_r_cost += r_cost
        offenders.append({
            "id": t.id, "ticker": t.ticker, "strategy": t.strategy,
            "entry_date": key, "quadrant_at_entry": quad, "tagged_regimes": tagged,
            "r_multiple": _round(r, 2), "r_cost": _round(r_cost, 2),
            "pnl": _round(t.pnl, 2),
        })
    return {
        "metric": "off_regime_entries",
        "available": True,
        "status": "OK",
        "value": len(offenders),
        "r_cost": _round(total_r_cost, 2, 0.0),
        "trades_examined": len(dated),
        "entry_dates_not_in_spy_cache": unknown_dates,
        "offenders": sorted(offenders, key=lambda o: -(o["r_cost"] or 0)),
        "detail": (
            f"{len(offenders)} entries were taken in a regime the strategy is not tagged "
            f"for, costing {round(total_r_cost, 2)}R realized."
        ),
        "note": ("Regime is reconstructed from cached SPY bars with no look-ahead, using the "
                 "same classifier the backtest uses. Only realized losses count as an R cost; "
                 "no counterfactual is invented for the winners."),
    }


def execution_quality(db: Session, user_id: int) -> dict:
    """Four execution leaks, ranked by realized R cost.

    Anything not computable comes back in `unavailable` with the exact field it
    would need. Nothing is ever estimated to fill a gap.
    """
    trades = db.query(ClosedTrade).filter(ClosedTrade.user_id == user_id).all()
    metrics = [
        _chasing(trades),
        _stop_loosening(trades),
        _overstayed(trades),
        _off_regime(db, trades),
    ]
    available = [m for m in metrics if m.get("available")]
    unavailable = [m for m in metrics if not m.get("available")]
    ranked = sorted(available, key=lambda m: -(m.get("r_cost") or 0.0))
    total = sum((m.get("r_cost") or 0.0) for m in available)
    return {
        "trades_examined": len(trades),
        "ranked": ranked,
        "unavailable": unavailable,
        "total_r_cost": _round(total, 2, 0.0),
        "summary": (
            f"{len(available)} of {len(metrics)} execution metrics are computable from the "
            f"data the app records today; {len(unavailable)} need fields that are not stored. "
            f"Measured leaks cost {round(total, 2)}R."
        ),
        "honesty_note": (
            "Metrics listed under `unavailable` are NOT zero and NOT estimated — the data to "
            "compute them does not exist. Each one names the field required."
        ),
    }
