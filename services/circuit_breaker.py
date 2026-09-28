"""
Account-level circuit breakers (ML4T 3e §19.8, §26.5 — gap 7).

WHY: every other risk check in the app is per TRADE (stop, size, correlation,
sector, open R). Nothing looked at the ACCOUNT: a losing streak or a deep
drawdown never slowed the app down — the Weekly Plan kept recommending full-
size buys into it. The book's answer is a graduated kill switch: warn, then
size down, then stop opening new risk, with an explicit, recorded human
decision to resume.

EQUITY (the app does not track cash):

    equity = account_size + realized P&L (every journal row)
                          + unrealized P&L (open positions × (last cached price − avg cost))

A position with no cached price contributes 0 unrealized P&L and is counted in
`unpriced` (reported, never silently assumed flat without saying so). Prices
come from the StockCache only — computing equity never triggers a network
fetch, so it is safe inside a request and in the scheduler.

SNAPSHOTS: one `EquitySnapshot` per user per trading session (US-Eastern date;
weekends map to the preceding Friday). The after-close scheduler job writes a
`close` row; loading the Playbook writes an `intraday` row when today's close
row doesn't exist yet. A `close` row is never overwritten by `intraday`.

LEVELS (graduated; the most severe trigger wins):
  ok     — nothing tripped.
  warn   — daily loss >= breaker_daily_loss_pct (default 2 %). Informational.
  reduce — drawdown from the equity peak >= breaker_drawdown_reduce_pct (8 %),
           or breaker_loss_streak (5) closed losing trades in a row.
           New trades are sized at REDUCE_RISK_MULTIPLIER (0.5) of normal risk.
  halt   — drawdown >= breaker_drawdown_halt_pct (12 %). No NEW entries:
           Weekly Plan buys are not pre-selected, Plan-a-Trade's commit is
           blocked, live broker BUY orders are refused. Sells, trims and stop
           raises are NEVER blocked — cutting risk must always be possible.

HALT LATCH (BreakerState): once halted, the halt persists even if equity
bounces a little. It clears
  * automatically when the drawdown recovers below the REDUCE threshold (the
    re-arm level) — the level then follows the live metrics again; or
  * when the user ACKNOWLEDGES it (POST /api/risk/breaker/acknowledge; who and
    when are recorded). Acknowledging lifts the halt to REDUCE only — never
    straight back to full size — and the halt re-engages if the drawdown
    deepens REHALT_STEP_PCT (3) points beyond where it was acknowledged.

FAILURE MODE: if the breaker cannot be evaluated (bad data, DB error), the
state is `unknown` with the error shown, and nothing is blocked — a bug in the
safety layer must not freeze all trading silently, but it must be visible.
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta, timezone

from sqlalchemy.orm import Session

from database.models import (
    BreakerState, ClosedTrade, EquitySnapshot, PortfolioPosition, StockCache, User,
)

DEFAULT_THRESHOLDS = {
    "daily_loss_pct": 2.0,
    "drawdown_reduce_pct": 8.0,
    "drawdown_halt_pct": 12.0,
    "loss_streak": 5,
}
_USER_FIELDS = {
    "daily_loss_pct": "breaker_daily_loss_pct",
    "drawdown_reduce_pct": "breaker_drawdown_reduce_pct",
    "drawdown_halt_pct": "breaker_drawdown_halt_pct",
    "loss_streak": "breaker_loss_streak",
}
REDUCE_RISK_MULTIPLIER = 0.5
REHALT_STEP_PCT = 3.0
LEVELS = ("ok", "warn", "reduce", "halt")
_RANK = {lvl: i for i, lvl in enumerate(LEVELS)}
SNAPSHOT_LOOKBACK_DAYS = 3 * 366   # the peak is taken over this window
STREAK_LOOKBACK_TRADES = 50

EQUITY_DEFINITION = (
    "Equity = account size (Settings) + realized P&L from the journal + unrealized P&L on open "
    "positions at their last cached price. The app doesn't track cash, so deposits and withdrawals "
    "only show up if you change the account size."
)


def _f(value, default=None):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _naive_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt


# ── thresholds ──────────────────────────────────────────────────────────────
def thresholds_for(user) -> dict:
    """The user's thresholds, NULL -> default. Always internally consistent:
    the halt threshold is kept above the reduce threshold."""
    out = {}
    for key, default in DEFAULT_THRESHOLDS.items():
        raw = getattr(user, _USER_FIELDS[key], None) if user is not None else None
        val = _f(raw)
        if val is None or val <= 0:
            val = default
        out[key] = int(val) if key == "loss_streak" else float(val)
    if out["drawdown_halt_pct"] <= out["drawdown_reduce_pct"]:
        out["drawdown_halt_pct"] = out["drawdown_reduce_pct"] + 1.0
    out["reduce_risk_multiplier"] = REDUCE_RISK_MULTIPLIER
    out["rehalt_step_pct"] = REHALT_STEP_PCT
    out["user_set"] = {k: getattr(user, f, None) is not None for k, f in _USER_FIELDS.items()} if user else {}
    return out


# ── session date ────────────────────────────────────────────────────────────
def session_date(now: datetime | None = None) -> date:
    """The US-Eastern trading date for `now`; Saturday/Sunday map to Friday.
    Holidays are not modelled — a holiday snapshot simply repeats the prior
    equity (a 0 % day), which trips nothing."""
    now = now or _utcnow()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        local = now.astimezone(ZoneInfo("America/New_York"))
    except Exception:  # noqa: BLE001 — no tzdata: approximate EDT/EST
        offset = -4 if 3 <= now.month <= 10 else -5
        local = now + timedelta(hours=offset)
    d = local.date()
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


# ── pure metrics ────────────────────────────────────────────────────────────
def compute_equity(account_size, realized_pnl, positions: list[dict]) -> dict:
    """Pure. positions: [{ticker, shares, avg_cost, price|None}]."""
    base = _f(account_size, 0.0) or 0.0
    realized = _f(realized_pnl, 0.0) or 0.0
    unrealized = open_value = 0.0
    unpriced: list[str] = []
    for p in positions:
        shares = _f(p.get("shares"), 0.0) or 0.0
        cost = _f(p.get("avg_cost"), 0.0) or 0.0
        price = _f(p.get("price"))
        if price is None or price <= 0:
            unpriced.append(str(p.get("ticker")))
            open_value += shares * cost
            continue
        open_value += shares * price
        unrealized += shares * (price - cost)
    return {
        "equity": round(base + realized + unrealized, 2),
        "account_size": round(base, 2),
        "realized_pnl": round(realized, 2),
        "unrealized_pnl": round(unrealized, 2),
        "open_value": round(open_value, 2),
        "unpriced": unpriced,
    }


def loss_streak(pnls_newest_first: list[float]) -> int:
    """Consecutive losing closed trades, newest first. A scratch (pnl == 0)
    neither extends nor breaks the streak; the first win ends it."""
    n = 0
    for p in pnls_newest_first:
        v = _f(p)
        if v is None or v == 0:
            continue
        if v < 0:
            n += 1
        else:
            break
    return n


def evaluate(snapshots: list[tuple[str, float]], current_equity: float, today: str,
             streak: int, thresholds: dict, latch: dict | None = None) -> dict:
    """Pure breaker evaluation.

    snapshots: [(iso_date, equity)] — any order; rows dated `today` are
    superseded by `current_equity`. latch: {halted_since, ack_at, ack_by,
    ack_drawdown_pct} from BreakerState (all optional).

    Returns the state dict plus `_latch` (the latch to persist)."""
    latch = dict(latch or {})
    cur = _f(current_equity)
    history = sorted((d, _f(e)) for d, e in snapshots if _f(e) is not None and d != today)
    prior = [e for d, e in history if d < today]
    prev = prior[-1] if prior else None
    daily = ((cur - prev) / prev * 100.0) if (cur is not None and prev and prev > 0) else None
    peak_candidates = [e for _, e in history] + ([cur] if cur is not None else [])
    peak = max(peak_candidates) if peak_candidates else None
    dd = ((peak - cur) / peak * 100.0) if (cur is not None and peak and peak > 0) else 0.0
    dd = max(0.0, dd)

    t = thresholds
    reasons: list[dict] = []
    if daily is not None and -daily >= t["daily_loss_pct"]:
        reasons.append({"code": "daily_loss", "level": "warn",
                        "message": f"Equity is down {-daily:.1f}% since the last session "
                                   f"(warn at {t['daily_loss_pct']:g}%). Slow down; review before adding risk."})
    if streak >= t["loss_streak"]:
        reasons.append({"code": "loss_streak", "level": "reduce",
                        "message": f"{streak} losing trades in a row (limit {t['loss_streak']}). "
                                   f"New trades are sized at {int(REDUCE_RISK_MULTIPLIER * 100)}% "
                                   "until a winner breaks the streak."})
    if dd >= t["drawdown_halt_pct"]:
        reasons.append({"code": "drawdown_halt", "level": "halt",
                        "message": f"Drawdown {dd:.1f}% from the equity peak (halt at "
                                   f"{t['drawdown_halt_pct']:g}%). New entries are paused; "
                                   "sells, trims and stop raises still work."})
    elif dd >= t["drawdown_reduce_pct"]:
        reasons.append({"code": "drawdown_reduce", "level": "reduce",
                        "message": f"Drawdown {dd:.1f}% from the equity peak (size-down at "
                                   f"{t['drawdown_reduce_pct']:g}%). New trades are sized at "
                                   f"{int(REDUCE_RISK_MULTIPLIER * 100)}%."})

    raw = "ok"
    for r in reasons:
        if _RANK[r["level"]] > _RANK[raw]:
            raw = r["level"]

    level = raw
    acknowledged = None
    rearm = t["drawdown_reduce_pct"]
    ack_dd = _f(latch.get("ack_drawdown_pct"))
    acked = latch.get("ack_at") is not None and ack_dd is not None
    if raw == "halt":
        if acked and dd < ack_dd + REHALT_STEP_PCT:
            level = "reduce"
        else:
            if acked:   # deepened past the acknowledged level: a NEW halt
                reasons.append({"code": "rehalt", "level": "halt",
                                "message": f"Drawdown deepened past {ack_dd + REHALT_STEP_PCT:.1f}% after the "
                                           "halt was acknowledged — halted again."})
                latch.update(ack_at=None, ack_by=None, ack_drawdown_pct=None)
                acked = False
            if not latch.get("halted_since"):
                latch["halted_since"] = _utcnow()
    elif latch.get("halted_since"):
        if dd < rearm:
            # Recovered below the re-arm level: the latch releases itself.
            reasons.append({"code": "halt_cleared", "level": "ok",
                            "message": f"The earlier halt cleared: drawdown recovered to {dd:.1f}% "
                                       f"(below the {rearm:g}% re-arm level)."})
            latch.update(halted_since=None, ack_at=None, ack_by=None, ack_drawdown_pct=None)
            acked = False
        elif acked:
            level = "reduce" if _RANK[raw] <= _RANK["reduce"] else raw
        else:
            level = "halt"
            reasons.append({"code": "halt_latched", "level": "halt",
                            "message": f"Still halted: the halt stays on until drawdown recovers below "
                                       f"{rearm:g}% or you acknowledge it (acknowledging resumes at "
                                       f"{int(REDUCE_RISK_MULTIPLIER * 100)}% size)."})
    if acked and latch.get("halted_since"):
        acknowledged = {
            "at": latch["ack_at"].isoformat() if hasattr(latch.get("ack_at"), "isoformat") else latch.get("ack_at"),
            "by": latch.get("ack_by"),
            "drawdown_pct": round(ack_dd, 2),
            "rehalt_at_drawdown_pct": round(ack_dd + REHALT_STEP_PCT, 2),
        }
        reasons.append({"code": "halt_acknowledged", "level": "reduce",
                        "message": f"Halt acknowledged by {latch.get('ack_by') or 'you'} at "
                                   f"{acknowledged['drawdown_pct']:.1f}% drawdown — trading resumed at "
                                   f"{int(REDUCE_RISK_MULTIPLIER * 100)}% size; halts again at "
                                   f"{acknowledged['rehalt_at_drawdown_pct']:.1f}%."})
        if level == "ok" or level == "warn":
            level = "reduce"

    multiplier = {"ok": 1.0, "warn": 1.0, "reduce": REDUCE_RISK_MULTIPLIER, "halt": 0.0}[level]
    return {
        "level": level,
        "raw_level": raw,
        "risk_multiplier": multiplier,
        "blocks_new_entries": level == "halt",
        "can_acknowledge": level == "halt",
        "reasons": reasons,
        "acknowledged": acknowledged,
        "metrics": {
            "equity": round(cur, 2) if cur is not None else None,
            "previous_equity": round(prev, 2) if prev is not None else None,
            "daily_change_pct": round(daily, 2) if daily is not None else None,
            "peak_equity": round(peak, 2) if peak is not None else None,
            "drawdown_pct": round(dd, 2),
            "loss_streak": int(streak),
            "sessions_recorded": len(history) + 1,
        },
        "thresholds": {k: v for k, v in thresholds.items() if k != "user_set"},
        "_latch": latch,
    }


# ── order gating (pure; used by the broker layer) ───────────────────────────
def gate_orders(rows: list[dict], state: dict | None, live: bool) -> None:
    """Mutates broker validation rows. HALT: live BUY rows are refused; paper
    buys are allowed but flagged. REDUCE/WARN: buys get a warning. Sell rows
    are never touched — reducing risk is always allowed."""
    if not state:
        return
    level = state.get("level")
    for r in rows:
        if r.get("side") != "buy":
            continue
        if level == "halt":
            msg = ("Circuit breaker HALT — new entries are paused ("
                   + "; ".join(x["message"] for x in state.get("reasons") or [] if x.get("level") == "halt")[:300]
                   + "). Acknowledge it on the Playbook to resume at reduced size.")
            if live:
                if r.get("ok"):
                    r["ok"] = False
                r.setdefault("errors", []).append(msg)
            else:
                r.setdefault("warnings", []).append("Paper only — would be refused live. " + msg)
        elif level == "reduce":
            r.setdefault("warnings", []).append(
                f"Circuit breaker REDUCE — new positions should be sized at "
                f"{int(REDUCE_RISK_MULTIPLIER * 100)}% of normal risk; check this order's size.")
        elif level == "warn":
            r.setdefault("warnings", []).append(
                "Circuit breaker WARN — " + "; ".join(x["message"] for x in state.get("reasons") or [])[:300])


# ── DB layer ────────────────────────────────────────────────────────────────
def _cached_prices(db: Session, tickers: list[str]) -> dict[str, float]:
    if not tickers:
        return {}
    out: dict[str, float] = {}
    rows = db.query(StockCache.ticker, StockCache.quote_json).filter(StockCache.ticker.in_(tickers)).all()
    for ticker, quote_json in rows:
        try:
            price = _f((json.loads(quote_json) if quote_json else {}).get("price"))
        except (ValueError, TypeError):
            price = None
        if price is not None:
            out[ticker] = price
    return out


def equity_now(db: Session, user: User) -> dict:
    positions = db.query(PortfolioPosition).filter(PortfolioPosition.user_id == user.id).all()
    prices = _cached_prices(db, sorted({p.ticker for p in positions}))
    realized = sum(_f(r[0], 0.0) or 0.0 for r in
                   db.query(ClosedTrade.pnl).filter(ClosedTrade.user_id == user.id).all())
    return compute_equity(
        user.account_size or 10000.0, realized,
        [{"ticker": p.ticker, "shares": p.shares, "avg_cost": p.avg_cost, "price": prices.get(p.ticker)}
         for p in positions],
    )


def record_snapshot(db: Session, user: User, source: str = "intraday",
                    now: datetime | None = None, eq: dict | None = None) -> EquitySnapshot | None:
    """Upsert today's snapshot. A `close` row is never replaced by `intraday`."""
    day = session_date(now).isoformat()
    eq = eq or equity_now(db, user)
    row = (db.query(EquitySnapshot)
           .filter(EquitySnapshot.user_id == user.id, EquitySnapshot.date == day).first())
    if row is not None and row.source == "close" and source != "close":
        return row
    if row is None:
        row = EquitySnapshot(user_id=user.id, date=day, equity=eq["equity"], source=source)
        db.add(row)
    row.equity = eq["equity"]
    row.account_size = eq["account_size"]
    row.realized_pnl = eq["realized_pnl"]
    row.unrealized_pnl = eq["unrealized_pnl"]
    row.open_value = eq["open_value"]
    row.unpriced = len(eq["unpriced"])
    row.source = source
    row.recorded_at = _utcnow()
    db.commit()
    return row


def record_close_snapshots(db: Session) -> int:
    """Scheduler entry point (after the close): one `close` snapshot per user.
    Cheap — cached prices and one journal sum per user, no network."""
    n = 0
    for user in db.query(User).all():
        try:
            record_snapshot(db, user, source="close")
            n += 1
        except Exception:  # noqa: BLE001 — one user must not stop the rest
            db.rollback()
    return n


def _latch_of(row: BreakerState | None) -> dict:
    if row is None:
        return {}
    return {"halted_since": row.halted_since, "ack_at": row.ack_at, "ack_by": row.ack_by,
            "ack_drawdown_pct": row.ack_drawdown_pct}


def current_state(db: Session, user: User, record: bool = True, now: datetime | None = None) -> dict:
    """Evaluate (and persist the latch of) this user's breaker. Never raises."""
    try:
        eq = equity_now(db, user)
        if record:
            record_snapshot(db, user, source="intraday", now=now, eq=eq)
        today = session_date(now).isoformat()
        since = (session_date(now) - timedelta(days=SNAPSHOT_LOOKBACK_DAYS)).isoformat()
        snaps = [(r.date, r.equity) for r in db.query(EquitySnapshot.date, EquitySnapshot.equity)
                 .filter(EquitySnapshot.user_id == user.id, EquitySnapshot.date >= since).all()]
        pnls = [r[0] for r in db.query(ClosedTrade.pnl).filter(ClosedTrade.user_id == user.id)
                .order_by(ClosedTrade.closed_at.desc(), ClosedTrade.id.desc())
                .limit(STREAK_LOOKBACK_TRADES).all()]
        row = db.query(BreakerState).filter(BreakerState.user_id == user.id).first()
        state = evaluate(snaps, eq["equity"], today, loss_streak(pnls), thresholds_for(user), _latch_of(row))
        latch = state.pop("_latch")
        if row is None:
            row = BreakerState(user_id=user.id)
            db.add(row)
        stamp = _naive_utc(now or _utcnow())
        if row.level != state["level"]:
            row.level, row.level_since = state["level"], stamp
        hs = latch.get("halted_since")
        row.halted_since = _naive_utc(hs) if isinstance(hs, datetime) else hs
        row.ack_at = _naive_utc(latch.get("ack_at")) if isinstance(latch.get("ack_at"), datetime) else latch.get("ack_at")
        row.ack_by = latch.get("ack_by")
        row.ack_drawdown_pct = latch.get("ack_drawdown_pct")
        row.updated_at = stamp
        db.commit()
        state["since"] = row.level_since.isoformat() if row.level_since else None
        state["halted_since"] = row.halted_since.isoformat() if row.halted_since else None
        state["metrics"]["unpriced_positions"] = eq["unpriced"]
        state["metrics"]["realized_pnl"] = eq["realized_pnl"]
        state["metrics"]["unrealized_pnl"] = eq["unrealized_pnl"]
        state["metrics"]["account_size"] = eq["account_size"]
        state["equity_definition"] = EQUITY_DEFINITION
        state["as_of"] = _utcnow().isoformat()
        return state
    except Exception as exc:  # noqa: BLE001
        try:
            db.rollback()
        except Exception:  # noqa: BLE001
            pass
        return unknown_state(f"{type(exc).__name__}: {exc}")


def unknown_state(error: str) -> dict:
    return {
        "level": "unknown", "raw_level": "unknown", "risk_multiplier": 1.0,
        "blocks_new_entries": False, "can_acknowledge": False,
        "reasons": [{"code": "breaker_error", "level": "warn",
                     "message": "The circuit breaker could not be evaluated, so nothing is being "
                                "blocked by it — check the add-on log. " + error[:300]}],
        "acknowledged": None, "metrics": {}, "thresholds": DEFAULT_THRESHOLDS,
        "equity_definition": EQUITY_DEFINITION, "error": error[:500],
    }


def acknowledge(db: Session, user: User, now: datetime | None = None) -> dict:
    """Record the user's decision to resume after a HALT (at reduced size)."""
    state = current_state(db, user, record=True, now=now)
    if state.get("level") != "halt":
        raise ValueError("There is no active halt to acknowledge.")
    row = db.query(BreakerState).filter(BreakerState.user_id == user.id).first()
    row.ack_at = _naive_utc(now or _utcnow())
    row.ack_by = user.username
    row.ack_drawdown_pct = (state.get("metrics") or {}).get("drawdown_pct")
    db.commit()
    return current_state(db, user, record=False, now=now)


def plan_risk_multiplier(state: dict | None) -> float:
    """Multiplier to apply to risk % when building NEW trade plans. A halt
    still shows plans (sized as if reduced) so the page stays informative —
    the halt itself blocks acting on them."""
    level = (state or {}).get("level")
    if level in ("reduce", "halt"):
        return REDUCE_RISK_MULTIPLIER
    return 1.0
