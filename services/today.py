"""
"Today" dashboard aggregation — the daily guided-workflow payload.

Combines four things the user needs each morning:
  1. Market regime light (SPY trend, VIX, breadth) → trade or sit out
  2. Position health (stop/target/trend checks) → what to sell
  3. Top setups across the 3 strategies with full trade plans → what to buy
  4. A guided checklist + capacity vs max positions

The payload is cached per-user with a 15-minute TTL; the scheduler also
invalidates it after each market refresh / daily report.
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from database.models import PortfolioPosition, StockCache, User, ClosedTrade
from services.indicators import calc_atr, calc_ma, compute_swing_score
from services.market_data import _is_fresh
from services.strategies import STRATEGIES
from services.trade_plan import build_trade_plan
from services.universe import UNIVERSE
from services import chart_service
from services import regime as regime_mod
from services.regime import classify_regime, strategies_for_regime, QUADRANT_INFO
from services.strategy_rationale import rationale_for

_TTL_SECONDS = 15 * 60
_cache: dict[int, tuple[dict, float]] = {}


def _json_safe(obj):
    """Recursively replace NaN/Inf floats with None. Starlette serializes with
    allow_nan=False, so a single non-finite float anywhere in the payload would
    500 the response — cached quote fields (RSI, vs_ma*, etc.) can carry NaN."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    return obj


def invalidate_cache(user_id: int | None = None) -> None:
    if user_id is None:
        _cache.clear()
    else:
        _cache.pop(user_id, None)


# ── Shared position-health helper (also used by build_decision_cockpit) ──────

def position_flags(pos: PortfolioPosition, quote: dict, bars: list[dict] | None = None,
                   atr_mult: float = 2.5) -> dict:
    """Evaluate one holding. Returns status, actions, exit reasons, suggested
    trailing stop and open R-multiple. `bars` (OHLCV) enables the ATR trail."""
    close = quote.get("price")
    stop = pos.stop_loss
    target = pos.target

    # Quality/trend reasons (extracted from the original decision cockpit)
    reasons: list[str] = []
    if (quote.get("vs_ma200") or 0) < 0:
        reasons.append("below 200MA")
    # ann_ret_1m = the 21-bar (1-month) annualized %. Quote field `ann_ret` is now
    # the FULL-HISTORY annualized return (what Sharpe/Sortino/Calmar use), which is
    # NOT the intent here — this flag is about recent momentum going soft.
    if (quote.get("ann_ret_1m") or 0) < 10:
        reasons.append("weak 1M annualized return")
    if quote.get("sharpe") is not None and quote["sharpe"] < 0.5:
        reasons.append("low Sharpe")
    if quote.get("max_dd_1m") is not None and quote["max_dd_1m"] > 8:
        reasons.append("drawdown pressure")

    status = "ok"
    actions: list[str] = []
    if close is not None:
        if stop is not None and close <= stop:
            status = "stop_hit"
            actions.append(
                "SELL NOW — price dropped to your stop. Exit to cap the loss before it grows."
            )
        elif target is not None and close >= target:
            status = "target_hit"
            actions.append(
                "TAKE PROFITS — price hit your target. Sell part of the position (or all of it), "
                "or raise your stop up to lock in the gain."
            )
        elif stop is not None and close <= stop * 1.03:
            status = "near_stop"
            actions.append(
                "GET READY TO SELL — within 3% of your stop. Set a price alert and exit if it "
                "closes below the stop."
            )
        elif (quote.get("vs_ma200") or 0) < 0:
            status = "trend_break"
            actions.append(
                "CONSIDER SELLING — it closed below its 200-day average, so the long-term uptrend "
                "is broken. Re-check why you own it and tighten your stop."
            )
    if not actions and reasons:
        actions.append(
            "HOLD, BUT WATCH — okay to keep for now, but it's weakening (" + ", ".join(reasons) + ")."
        )
    if not actions:
        actions.append("HOLD — on track. Nothing to do; leave your stop where it is.")

    # ATR trailing-stop suggestion (never below an existing stop)
    suggested_stop = None
    if bars and close:
        highs = [b["high"] for b in bars]
        lows = [b["low"] for b in bars]
        closes = [b["close"] for b in bars]
        atr_series = calc_atr(highs, lows, closes, 14)
        atr = next((v for v in reversed(atr_series) if v is not None), None)
        if atr:
            trail = close - atr_mult * atr
            suggested_stop = round(max(stop or 0.0, trail), 2)

    open_r = None
    if stop is not None and close is not None and pos.avg_cost - stop > 0:
        open_r = round((close - pos.avg_cost) / (pos.avg_cost - stop), 2)

    return {
        "status": status,
        "actions": actions,
        "reasons": reasons,
        "suggested_stop": suggested_stop,
        "open_r": open_r,
    }


# ── Bulk cache loaders ───────────────────────────────────────────────────────

def _load_all_cache(db: Session) -> dict[str, dict]:
    """One query → {ticker: {"quote": dict|None, "bars": list|None}}."""
    rows = db.query(
        StockCache.ticker, StockCache.quote_json, StockCache.history_json
    ).all()
    out: dict[str, dict] = {}
    for ticker, quote_json, history_json in rows:
        quote = None
        bars = None
        if quote_json:
            try:
                quote = json.loads(quote_json)
            except Exception:
                quote = None
        if history_json:
            try:
                bars = json.loads(history_json)
            except Exception:
                bars = None
        out[ticker] = {"quote": quote, "bars": bars}
    return out


# ── Regime ───────────────────────────────────────────────────────────────────

_QUADRANT_TO_LIGHT = {
    "trending_bull": "green",
    "choppy_calm": "yellow",
    "trending_bear": "red",
    "choppy_volatile": "red",
    None: "yellow",
}


async def _build_regime(cache: dict[str, dict]) -> dict:
    spy = cache.get("SPY", {}).get("bars") or []
    closes = [b["close"] for b in spy] if spy else []
    spy_price = closes[-1] if closes else None
    ma50 = calc_ma(closes, 50)[-1] if len(closes) >= 50 else None
    ma200 = calc_ma(closes, 200)[-1] if len(closes) >= 200 else None
    vs_ma50 = ((spy_price - ma50) / ma50 * 100) if (spy_price and ma50) else None
    vs_ma200 = ((spy_price - ma200) / ma200 * 100) if (spy_price and ma200) else None

    # VIX
    vix_last = vix_chg5 = None
    try:
        vix = await chart_service.get_vix_data()
        if vix:
            vix_last = vix[-1]["value"]
            if len(vix) >= 6:
                vix_chg5 = round(vix_last - vix[-6]["value"], 2)
    except Exception:
        pass

    # Breadth — % of universe quotes above their 200-day MA
    above = total = 0
    for ticker in UNIVERSE:
        q = cache.get(ticker, {}).get("quote")
        if q and q.get("vs_ma200") is not None:
            total += 1
            if q["vs_ma200"] > 0:
                above += 1
    breadth_pct = round(above / total * 100, 1) if total else None

    classified = classify_regime(spy, vix_last=vix_last, breadth_pct=breadth_pct)
    light = _QUADRANT_TO_LIGHT.get(classified["quadrant"], "yellow")

    reasons = list(classified["reasons"])
    if vix_last is not None and "VIX" not in " ".join(reasons):
        reasons.append(f"VIX {vix_last:.1f}")

    return {
        "light": light,
        "quadrant": classified["quadrant"],
        "quadrant_label": classified["quadrant_label"],
        "quadrant_description": classified.get("quadrant_description"),
        "adx": classified["adx"],
        "adx_trend": classified["adx_trend"],
        "above_200ma": classified.get("above_200ma"),
        "transition": classified["transition"],
        "spy": {
            "price": round(spy_price, 2) if spy_price else None,
            "vs_ma50": round(vs_ma50, 2) if vs_ma50 is not None else None,
            "vs_ma200": round(vs_ma200, 2) if vs_ma200 is not None else None,
        },
        "vix": {"last": vix_last, "chg_5d": vix_chg5},
        "breadth": {"pct_above_ma200": breadth_pct, "sample": total},
        "reasons": reasons,
    }


# ── Positions ──────────────────────────────────────────────────────────────

def _build_positions(positions: list[PortfolioPosition], cache: dict[str, dict],
                     atr_mult: float) -> list[dict]:
    out = []
    for pos in positions:
        entry = cache.get(pos.ticker, {})
        quote = entry.get("quote") or {}
        bars = entry.get("bars")
        price = quote.get("price")
        cost_basis = pos.shares * pos.avg_cost
        pnl_pct = ((price - pos.avg_cost) / pos.avg_cost * 100) if (price and pos.avg_cost) else None
        flags = position_flags(pos, quote, bars, atr_mult)
        out.append({
            "ticker": pos.ticker,
            "name": quote.get("name"),
            "shares": pos.shares,
            "avg_cost": pos.avg_cost,
            "price": price,
            "pnl_pct": round(pnl_pct, 2) if pnl_pct is not None else None,
            "stop_loss": pos.stop_loss,
            "target": pos.target,
            "entry_date": pos.entry_date.isoformat() if pos.entry_date else None,
            "strategy": pos.strategy,
            "open_r": flags["open_r"],
            "status": flags["status"],
            "actions": flags["actions"],
            "suggested_stop": flags["suggested_stop"],
        })
    # Surface positions that need action first
    order = {"stop_hit": 0, "near_stop": 1, "target_hit": 2, "trend_break": 3, "ok": 4}
    out.sort(key=lambda p: (order.get(p["status"], 5), -(p["pnl_pct"] or 0)))
    return out


# ── Setups ─────────────────────────────────────────────────────────────────

def _build_setups(cache: dict[str, dict], held: set[str], account_size: float,
                  risk_pct: float, atr_mult: float, r_multiple: float,
                  active_ids: set[str] | None = None,
                  per_strategy: int = 3, overall: int = 8,
                  selection: dict | None = None) -> list[dict]:
    """Top setups across the active strategies.

    `selection`, when passed, is filled in with HOW the list was chosen — how
    many tickers were scanned, how many passed each strategy's rules, and the
    cut-offs applied — so the Playbook can say "ranked #2 of 31 that qualified"
    instead of presenting the list as if it fell from the sky.
    """
    strategies = {sid: s for sid, s in STRATEGIES.items() if active_ids is None or sid in active_ids}
    by_strategy: dict[str, list] = {sid: [] for sid in strategies}
    scanned = 0
    for ticker in UNIVERSE:
        if ticker in held:
            continue
        entry = cache.get(ticker, {})
        bars = entry.get("bars")
        quote = entry.get("quote") or {}
        if not bars:
            continue
        scanned += 1
        for sid, strat in strategies.items():
            setup = strat.scan(ticker, bars, quote)
            if setup:
                by_strategy[sid].append((setup, bars, quote))

    selected = []
    pool_size: dict[str, int] = {}
    rank_of: dict[tuple[str, str], int] = {}
    for sid, items in by_strategy.items():
        items.sort(key=lambda x: x[0].score, reverse=True)
        pool_size[sid] = len(items)
        for i, (setup, _b, _q) in enumerate(items):
            rank_of[(sid, setup.ticker)] = i + 1
        selected.extend(items[:per_strategy])
    # Triggered before forming, then by score
    selected.sort(key=lambda x: (x[0].state != "triggered", -x[0].score))
    selected = selected[:overall]

    if selection is not None:
        selection.update({
            "universe_size": len(UNIVERSE),
            "scanned": scanned,
            "skipped_held": len(held),
            "per_strategy_cap": per_strategy,
            "overall_cap": overall,
            "passed_by_strategy": dict(pool_size),
            "method": (
                f"Scanned {scanned} tickers with cached price history (skipping the "
                f"{len(held)} you already hold). Each running strategy applied its own rules; "
                f"the survivors were ranked by that strategy's score, the top {per_strategy} per "
                f"strategy kept, then setups that have TRIGGERED were put ahead of ones still "
                f"FORMING, capped at {overall} overall."
            ),
        })

    out = []
    for setup, bars, quote in selected:
        strat = STRATEGIES[setup.strategy]
        plan = build_trade_plan(
            bars,
            account_size=account_size,
            risk_pct=risk_pct,
            atr_mult=atr_mult,
            r_multiple=r_multiple,
        ) if strat.actionable else None
        swing = compute_swing_score(quote) if quote else None
        out.append({
            "ticker": setup.ticker,
            "name": quote.get("name"),
            "sector": quote.get("sector"),
            "strategy": setup.strategy,
            "strategy_name": setup.strategy_name,
            "state": setup.state,
            "score": round(setup.score, 4),
            "reasons": setup.reasons,
            # Why THIS ticker rather than another that also passed the rules.
            "rank": rank_of.get((setup.strategy, setup.ticker)),
            "pool_size": pool_size.get(setup.strategy),
            "metrics": setup.metrics,
            "price": quote.get("price"),
            "swing_score": {"score": swing["score"], "grade": swing["grade"]} if swing else None,
            "plan": plan,
            "actionable": strat.actionable,
        })
    return out


# ── Evidence-aware regime filtering ──────────────────────────────────────────
# `Strategy.regimes` is a hand-written opinion about which quadrants a strategy
# suits. services/edge_matrix.py turns the walk-forward backtest into evidence
# about whether that opinion survives contact with data. This wires the two
# together for the Playbook — under three hard safety constraints:
#
#   1. NEVER BUILD THE MATRIX HERE. get_cached_matrix() reads the 12h per-user
#      cache and returns None on a miss; a cold build takes minutes and would
#      block the dashboard. A miss means "no evidence", not "wait".
#   2. NEVER CHANGE WHAT THE USER SEES on a missing, stale or malformed matrix.
#      Every failure path degrades to exactly today's tag-based behaviour, and
#      every exception is swallowed into that same fallback.
#   3. SHIP OFF. Requires the per-user `use_evidence_regimes` setting (or the
#      module-level services/regime.set_evidence_override opt-in).

def _evidence_cell(matrix, quadrant, strategy_id) -> dict | None:
    """One strategy's evidence cell for a quadrant, tolerating any shape."""
    try:
        cell = matrix["strategies"][strategy_id]["cells"][quadrant]
    except Exception:  # noqa: BLE001 — a malformed matrix is "no evidence"
        return None
    if not isinstance(cell, dict):
        return None
    verdict = cell.get("verdict")
    n = cell.get("periods", cell.get("n"))
    if not isinstance(n, (int, float)) or not math.isfinite(n):
        n = None
    return {
        "verdict": verdict if isinstance(verdict, str) else None,
        "n": int(n) if n is not None else None,
        "detail": cell.get("verdict_detail") if isinstance(cell.get("verdict_detail"), str)
                  else None,
    }


def _resolve_active_strategies(db: Session, user, quadrant: str | None) -> tuple[set, dict, dict]:
    """(active_ids, evidence_summary, matrix) for the Playbook.

    Returns the plain tag-based set unless the evidence override is opted into
    AND a usable cached matrix exists AND applying it leaves a non-empty
    playbook.
    """
    base = strategies_for_regime(quadrant)
    opted_in = bool(getattr(user, "use_evidence_regimes", False)) or \
        bool(getattr(regime_mod, "EVIDENCE_OVERRIDE_ENABLED", False))
    summary = {
        "enabled": opted_in,
        "applied": False,
        "fallback": False,
        "matrix_available": False,
        "matrix_generated_at": None,
        "excluded": [],
        "added": [],
        "reason": ("Evidence override is off — the Playbook uses the hand-written "
                   "Strategy.regimes tags."),
    }
    if not opted_in:
        return base, summary, None

    matrix = None
    try:
        from services.edge_matrix import get_cached_matrix
        matrix = get_cached_matrix(user.id)   # cache read only — never builds
    except Exception:  # noqa: BLE001
        matrix = None
    if not isinstance(matrix, dict):
        matrix = None
    summary["matrix_available"] = matrix is not None
    summary["matrix_generated_at"] = (matrix or {}).get("generated_at")
    if matrix is None:
        summary["fallback"] = True
        summary["reason"] = ("Evidence override is on but no edge matrix is cached — "
                             "using the hand-written tags. Build one on the Backtest page.")
        return base, summary, None

    try:
        adj = evidence_adjustment_safe(quadrant, matrix)
    except Exception:  # noqa: BLE001
        adj = None
    if not adj or not adj.get("result"):
        summary["fallback"] = True
        summary["reason"] = ("The cached edge matrix could not be applied — using the "
                             "hand-written tags.")
        return base, summary, matrix

    summary["applied"] = bool(adj.get("applied"))
    summary["fallback"] = bool(adj.get("fallback"))
    summary["excluded"] = list(adj.get("excluded") or [])
    summary["added"] = list(adj.get("added") or [])
    summary["reason"] = adj.get("reason") or summary["reason"]
    return set(adj["result"]), summary, matrix


def evidence_adjustment_safe(quadrant, matrix):
    """`services.regime.evidence_adjustment` with force=True.

    `force` is correct here: the opt-in decision was already made per-user in
    `_resolve_active_strategies`, and the module-level flag must not have to be
    flipped globally for one user's dashboard.
    """
    return regime_mod.evidence_adjustment(quadrant, matrix, force=True)


# ── Top-level builder ────────────────────────────────────────────────────────

async def build_today(db: Session, user_id: int, force: bool = False) -> dict:
    if not force:
        hit = _cache.get(user_id)
        if hit and time.time() - hit[1] < _TTL_SECONDS:
            return hit[0]

    user = db.query(User).filter(User.id == user_id).first()
    # Coerce trading settings to safe defaults. Older installs can carry NULLs
    # (columns added without a DEFAULT), and None would crash the numeric paths.
    account_size = user.account_size or 10000.0
    risk_pct = user.risk_pct or 1.0
    max_positions = user.max_positions or 8
    atr_stop_mult = user.atr_stop_mult or 2.5
    r_multiple = user.r_multiple or 2.0

    positions = db.query(PortfolioPosition).filter(PortfolioPosition.user_id == user_id).all()
    cache = _load_all_cache(db)

    regime = await _build_regime(cache)
    tagged_ids = strategies_for_regime(regime["quadrant"])
    active_ids, evidence, matrix = _resolve_active_strategies(db, user, regime["quadrant"])
    pos_rows = _build_positions(positions, cache, atr_stop_mult)
    held = {p.ticker for p in positions}
    selection: dict = {}
    setups = _build_setups(cache, held, account_size, risk_pct, atr_stop_mult, r_multiple,
                           active_ids=active_ids, selection=selection)

    strategy_regime_status = []
    for s in STRATEGIES.values():
        is_active = s.id in active_ids
        was_tagged = s.id in tagged_ids
        cell = _evidence_cell(matrix, regime["quadrant"], s.id) if matrix else None
        # What the evidence did to THIS strategy, so the UI can explain itself
        # instead of silently reordering the playbook.
        if not evidence["enabled"]:
            effect, effect_note = None, None
        elif s.id in evidence["excluded"]:
            effect = "excluded"
            effect_note = ("Tagged for this regime, but the edge matrix marks it mis-tagged "
                           "here — the evidence stood it down.")
        elif s.id in evidence["added"]:
            effect = "promoted"
            effect_note = ("Not tagged for this regime, but the edge matrix found a real "
                           "edge here — the evidence promoted it.")
        elif evidence["applied"]:
            effect, effect_note = "unchanged", "Evidence agrees with the hand-written tag."
        else:
            effect = "unchanged"
            effect_note = "Evidence not applied — the hand-written tag stands."

        base_reason = (
            f"Matches current regime ({regime['quadrant_label']})" if was_tagged
            else f"Built for {', '.join(QUADRANT_INFO[q]['label'] for q in s.regimes) or 'no tagged regime'} — "
                 f"current regime is {regime['quadrant_label']}, so this strategy sits out."
        )
        if effect == "excluded":
            reason = f"{base_reason} Overridden: {effect_note}"
        elif effect == "promoted":
            reason = f"{base_reason} Overridden: {effect_note}"
        else:
            reason = base_reason

        strategy_regime_status.append({
            "id": s.id,
            "name": s.name,
            "active": is_active,
            "tagged_for_regime": was_tagged,
            "regimes": s.regimes,
            "actionable": s.actionable,
            "reason": reason,
            # WHY you would use this strategy at all, and why now — written
            # opinion (services/strategy_rationale.py), shown alongside, never
            # instead of, the tested evidence below.
            "rationale": rationale_for(s.id, regime["quadrant"]),
            "description": s.description,
            "horizon": (s.details or {}).get("horizon"),
            "how": (s.details or {}).get("how"),
            "horizon_days": s.horizon_days,
            "evidence": {
                "override_enabled": evidence["enabled"],
                "override_applied": evidence["applied"],
                "effect": effect,
                "note": effect_note,
                "verdict": (cell or {}).get("verdict"),
                "n": (cell or {}).get("n"),
                "verdict_detail": (cell or {}).get("detail"),
            },
        })

    # Data freshness from SPY cache row
    spy_row = db.query(StockCache).filter(StockCache.ticker == "SPY").first()
    spy_cached = spy_row.quote_cached_at or spy_row.cached_at if spy_row else None
    data_fresh = bool(spy_cached and _is_fresh(spy_cached))

    alerts = sum(1 for p in pos_rows if p["status"] != "ok")
    open_journal = db.query(ClosedTrade).filter(ClosedTrade.user_id == user_id).count()

    # "Suppressed" no longer hides the setups list outright — bear/crisis
    # regimes have their own tagged strategies (dual momentum's defensive
    # rotation, the bear-reversal watchlist) that should still surface. It
    # now just flags that fresh, un-tagged risk should not be added.
    suppressed = regime["light"] == "red"
    checklist = [
        {"id": "data", "label": "Market data fresh", "done": data_fresh,
         "detail": "refreshed after last close" if data_fresh else "stale — refresh on the Screener page"},
        {"id": "regime", "label": f"Check regime ({regime['quadrant_label']}, ADX {regime['adx']})", "done": True},
        {"id": "positions", "label": f"Review {alerts} position alert(s)" if alerts else "No position alerts",
         "done": alerts == 0, "count": alerts},
        {"id": "setups", "label": "Review today's top setups" if not suppressed else "Regime risk-off — favor defensive/watchlist-only setups",
         "done": False, "count": len(setups)},
        {"id": "journal", "label": "Journal any closed trades", "done": False, "count": open_journal},
    ]

    payload = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "regime": regime,
        "positions": pos_rows,
        "setups": setups,
        "selection": selection,
        "strategy_regime_status": strategy_regime_status,
        "evidence_regimes": evidence,
        "suppressed_by_regime": suppressed,
        "checklist": checklist,
        "capacity": {
            "open_positions": len(positions),
            "max_positions": max_positions,
            "slots_free": max(0, max_positions - len(positions)),
        },
        "settings_used": {
            "account_size": account_size,
            "risk_pct": risk_pct,
            "max_positions": max_positions,
            "atr_stop_mult": atr_stop_mult,
            "r_multiple": r_multiple,
        },
        "disclaimer": "Decision-support only. Verify position sizing, taxes, liquidity, and "
                      "your own risk tolerance before placing any order.",
    }
    payload = _json_safe(payload)
    _cache[user_id] = (payload, time.time())
    return payload
