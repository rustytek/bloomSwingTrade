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
import time
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from database.models import PortfolioPosition, StockCache, User, WatchlistItem, ClosedTrade
from services.indicators import calc_atr, calc_ma, compute_swing_score
from services.market_data import _is_fresh
from services.strategies import STRATEGIES
from services.trade_plan import build_trade_plan
from services.universe import UNIVERSE
from services import chart_service

_TTL_SECONDS = 15 * 60
_cache: dict[int, tuple[dict, float]] = {}


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
    if (quote.get("ann_ret") or 0) < 10:
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

    reasons: list[str] = []
    above_200 = vs_ma200 is not None and vs_ma200 > 0
    above_50 = vs_ma50 is not None and vs_ma50 > 0
    if above_200:
        reasons.append("SPY above 200-day MA")
    else:
        reasons.append("SPY below 200-day MA")
    if above_50:
        reasons.append("SPY above 50-day MA")
    if vix_last is not None:
        reasons.append(f"VIX {vix_last:.1f}")
    if breadth_pct is not None:
        reasons.append(f"{breadth_pct:.0f}% of names above 200-day MA")

    if not above_200 or (vix_last is not None and vix_last > 30):
        light = "red"
    elif (vix_last is not None and 25 <= vix_last <= 30) or not above_50:
        light = "yellow"
    else:
        light = "green"

    return {
        "light": light,
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
                  per_strategy: int = 3, overall: int = 8) -> list[dict]:
    by_strategy: dict[str, list] = {sid: [] for sid in STRATEGIES}
    for ticker in UNIVERSE:
        if ticker in held:
            continue
        entry = cache.get(ticker, {})
        bars = entry.get("bars")
        quote = entry.get("quote") or {}
        if not bars:
            continue
        for sid, strat in STRATEGIES.items():
            setup = strat.scan(ticker, bars, quote)
            if setup:
                by_strategy[sid].append((setup, bars, quote))

    selected = []
    for sid, items in by_strategy.items():
        items.sort(key=lambda x: x[0].score, reverse=True)
        selected.extend(items[:per_strategy])
    # Triggered before forming, then by score
    selected.sort(key=lambda x: (x[0].state != "triggered", -x[0].score))
    selected = selected[:overall]

    out = []
    for setup, bars, quote in selected:
        plan = build_trade_plan(
            bars,
            account_size=account_size,
            risk_pct=risk_pct,
            atr_mult=atr_mult,
            r_multiple=r_multiple,
        )
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
            "swing_score": {"score": swing["score"], "grade": swing["grade"]} if swing else None,
            "plan": plan,
        })
    return out


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
    pos_rows = _build_positions(positions, cache, atr_stop_mult)
    held = {p.ticker for p in positions}
    setups = _build_setups(cache, held, account_size, risk_pct, atr_stop_mult, r_multiple)

    # Data freshness from SPY cache row
    spy_row = db.query(StockCache).filter(StockCache.ticker == "SPY").first()
    spy_cached = spy_row.quote_cached_at or spy_row.cached_at if spy_row else None
    data_fresh = bool(spy_cached and _is_fresh(spy_cached))

    alerts = sum(1 for p in pos_rows if p["status"] != "ok")
    open_journal = db.query(ClosedTrade).filter(ClosedTrade.user_id == user_id).count()

    suppressed = regime["light"] == "red"
    checklist = [
        {"id": "data", "label": "Market data fresh", "done": data_fresh,
         "detail": "refreshed after last close" if data_fresh else "stale — refresh on the Screener page"},
        {"id": "regime", "label": f"Check regime ({regime['light'].upper()})", "done": True},
        {"id": "positions", "label": f"Review {alerts} position alert(s)" if alerts else "No position alerts",
         "done": alerts == 0, "count": alerts},
        {"id": "setups", "label": "Review today's top setups" if not suppressed else "Regime risk-off — hold cash",
         "done": False, "count": len(setups)},
        {"id": "journal", "label": "Journal any closed trades", "done": False, "count": open_journal},
    ]

    payload = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "regime": regime,
        "positions": pos_rows,
        "setups": setups,
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
    _cache[user_id] = (payload, time.time())
    return payload
