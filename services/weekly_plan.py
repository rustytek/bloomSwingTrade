"""
Weekly Plan — the guided "what should I do this week, and why" walkthrough.

Built ON TOP of the Playbook payload (services/today.py::build_today) rather
than beside it, so the two pages can never disagree about the regime, the
running strategies or the setups. What this module adds is the DECISION layer:

  1. Position reviews   — for every holding: sell / trim / raise stop / hold,
                          each with the reasons in plain English.
  2. Proposed orders    — concrete, sized limit orders (the contract the Trade
                          page consumes), each with a `why[]` list.
  3. A risk-aware pick  — buys are walked in priority order through
                          services/portfolio_risk.assess_new_position against a
                          HYPOTHETICAL book that already contains the buys picked
                          before them, so two highly-correlated names, a sector
                          pile-up or a blown open-R budget can't all be
                          recommended at once.

Order ids are stable strings — "buy:AAPL", "sell:MSFT", "trim:NVDA" — so the
Trade page can carry a selection across page loads.

Nothing here places an order. It is decision support; the Trade page (and the
user's explicit confirmation) is the only path to a broker.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone

from sqlalchemy.orm import Session

from database.models import PortfolioPosition, User
from services import portfolio_risk
from services.strategies import STRATEGIES
from services.strategy_rationale import rationale_for
from services.today import build_today, _load_all_cache, _json_safe

# A sell is priced a hair under the last price so it fills at the open like a
# market order would, but can't be dumped into a gap far below. 1% is the
# conventional "marketable limit" cushion.
SELL_LIMIT_CUSHION = 0.01
# Setups ranked beyond this many buys are listed but not pre-selected, even if
# slots and budget remain: a weekly review should add a few positions, not
# restock the whole book in one go.
MAX_NEW_BUYS_PER_WEEK = 4


def week_of(today: date | None = None) -> str:
    """ISO date of the Monday of the current week."""
    d = today or date.today()
    return (d - timedelta(days=d.weekday())).isoformat()


def _f(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _money(v) -> str:
    x = _f(v)
    return "--" if x is None else f"${x:,.2f}"


def _evidence_cells(user_id: int, quadrant: str | None) -> dict[str, dict]:
    """{strategy_id: cell} from the CACHED edge matrix; {} when not built.

    Never builds (a cold build takes minutes) — same rule as the Playbook.
    """
    if not quadrant:
        return {}
    try:
        from services.edge_matrix import get_cached_matrix
        matrix = get_cached_matrix(user_id)
        strats = (matrix or {}).get("strategies") or {}
        out = {}
        for sid, s in strats.items():
            cell = ((s or {}).get("cells") or {}).get(quadrant)
            if isinstance(cell, dict):
                out[sid] = cell
        return out
    except Exception:  # noqa: BLE001 — no evidence is a valid state
        return {}


def _evidence_line(cell: dict | None) -> str | None:
    if not cell:
        return None
    verdict = cell.get("verdict")
    n = cell.get("n", cell.get("periods"))
    wr = _f(cell.get("win_rate"))
    avg = _f(cell.get("avg_period_return"))
    bits = []
    if avg is not None:
        bits.append(f"avg {avg:+.2f}% per period")
    if wr is not None:
        bits.append(f"win rate {wr:.0f}%")
    if n is not None:
        bits.append(f"n={n}")
    tail = f" ({', '.join(bits)})" if bits else ""
    if verdict == "confirmed":
        return f"Backtest CONFIRMS this strategy has worked in this regime{tail}."
    if verdict == "mis-tagged":
        return f"Warning: the backtest says this strategy has NOT worked in this regime{tail}."
    if verdict == "untagged-edge":
        return f"The backtest found an edge here even though it is not tagged for this regime{tail}."
    if verdict == "unproven":
        return f"Backtest evidence is thin in this regime{tail} — treat it as unproven."
    return None


# ── Position reviews ─────────────────────────────────────────────────────────

def _days_held(entry_date: str | None) -> int | None:
    if not entry_date:
        return None
    try:
        d = date.fromisoformat(str(entry_date)[:10])
    except ValueError:
        return None
    return max(0, (date.today() - d).days)


def _review_position(p: dict, orm: PortfolioPosition | None) -> tuple[dict, dict | None]:
    """(review, order-or-None) for one position row from build_today."""
    ticker = p["ticker"]
    price = _f(p.get("price"))
    shares = _f(p.get("shares")) or 0.0
    stop = _f(p.get("stop_loss"))
    target = _f(p.get("target"))
    open_r = _f(p.get("open_r"))
    status = p.get("status") or "ok"
    strat = STRATEGIES.get(p.get("strategy") or "")
    held_days = _days_held(p.get("entry_date"))
    # Trading-day horizon -> rough calendar days (5 of every 7).
    time_stop_td = (orm.time_stop_days if orm is not None and orm.time_stop_days else None)
    horizon_td = time_stop_td or (strat.horizon_days if strat else None)
    horizon_cal = int(horizon_td * 7 / 5) if horizon_td else None

    why: list[str] = list(p.get("actions") or [])
    verdict, kind, recommended, sell_shares = "hold", None, False, 0

    if status == "stop_hit":
        verdict, kind, recommended, sell_shares = "sell", "sell", True, shares
        why.append(
            f"Price {_money(price)} is at or below your stop {_money(stop)}. The stop is the price at "
            f"which you decided in advance that the trade idea was wrong — honouring it is what keeps "
            f"one loss from becoming a big one."
        )
    elif status == "target_hit":
        verdict, kind, recommended = "trim", "trim", True
        sell_shares = math.floor(shares / 2) if shares >= 2 else shares
        why.append(
            f"Price {_money(price)} reached your target {_money(target)}"
            + (f" (+{open_r:.1f}R)" if open_r is not None else "")
            + ". Selling half locks in the planned profit; keep the rest with the stop raised to at "
              "least your entry so the trade can no longer lose money."
        )
    elif status == "trend_break":
        verdict, kind, recommended, sell_shares = "sell", "sell", False, shares
        why.append(
            "It closed below its 200-day average — the long-term uptrend that justified owning it is "
            "broken. Not pre-selected because your stop has not been hit; tick it if you agree the "
            "reason you bought it no longer holds."
        )
    elif horizon_cal and held_days is not None and held_days > horizon_cal and (open_r is None or open_r < 0.5):
        verdict, kind, sell_shares = "sell", "sell", shares
        recommended = bool(time_stop_td)   # the user's OWN time stop is pre-selected
        why.append(
            f"Held {held_days} days — past the {'time stop you set' if time_stop_td else (strat.name + ' strategy horizon' if strat else 'planned horizon')} "
            f"(~{horizon_cal} calendar days) without making meaningful progress"
            + (f" ({open_r:+.1f}R)" if open_r is not None else "")
            + ". Capital stuck in a trade that isn't working can't be used on one that is."
        )
    elif status == "near_stop":
        verdict = "watch"
        why.append("Close to the stop. No order yet — if it closes below the stop, sell next session.")

    suggested = _f(p.get("suggested_stop"))
    raise_to = None
    if verdict in ("hold", "watch", "trim") and suggested is not None and (stop is None or suggested > stop):
        raise_to = round(suggested, 2)
        if verdict == "hold":
            verdict = "raise_stop"
        why.append(
            f"Raise the stop to {_money(raise_to)} (a {p.get('atr_mult') or 'standard'}-ATR trail under "
            f"the price). It locks in more of the gain while leaving room for normal day-to-day noise."
            if stop is not None else
            f"This position has NO stop. Set one at {_money(raise_to)} — without it the risk is unlimited."
        )
    if verdict == "hold" and not why:
        why.append("On track — nothing to do. Leave the stop where it is.")

    order = None
    if kind and sell_shares and price:
        limit = round(price * (1 - SELL_LIMIT_CUSHION), 2)
        order = {
            "id": f"{kind}:{ticker}",
            "kind": kind,
            "side": "sell",
            "ticker": ticker,
            "name": p.get("name"),
            "shares": int(sell_shares) if float(sell_shares).is_integer() else sell_shares,
            "limit_price": limit,
            "est_value": round(limit * sell_shares, 2),
            "recommended": recommended,
            "skip_reason": None if recommended else "Optional — your call. Not pre-selected.",
            "headline": (
                f"{'Sell all' if kind == 'sell' else 'Sell half'} {int(sell_shares)} {ticker} "
                f"(limit {_money(limit)})"
            ),
            "why": why + [
                f"Limit price {_money(limit)} is 1% under the last price so it fills like a market "
                f"order at the open but can't sell into a gap far below."
            ],
            "strategy": p.get("strategy"),
            "strategy_name": strat.name if strat else None,
            "plan": None,
        }

    review = {
        "ticker": ticker,
        "name": p.get("name"),
        "status": status,
        "verdict": verdict,
        "price": price,
        "shares": shares,
        "avg_cost": _f(p.get("avg_cost")),
        "pnl_pct": p.get("pnl_pct"),
        "open_r": open_r,
        "stop_loss": stop,
        "target": target,
        "raise_stop_to": raise_to,
        "held_days": held_days,
        "strategy": p.get("strategy"),
        "strategy_name": strat.name if strat else None,
        "why": why,
        "order_id": order["id"] if order else None,
    }
    return review, order


# ── Buys ─────────────────────────────────────────────────────────────────────

def _buy_why(s: dict, status: dict | None, cell: dict | None, settings: dict) -> list[str]:
    plan = s.get("plan") or {}
    rat = (status or {}).get("rationale") or rationale_for(s.get("strategy") or "")
    why: list[str] = []
    rank, pool = s.get("rank"), s.get("pool_size")
    if rank and pool:
        why.append(
            f"Ranked #{rank} of {pool} stocks that passed {s.get('strategy_name')}'s rules today "
            f"(state: {s.get('state')})."
        )
    for r in s.get("reasons") or []:
        why.append(f"Signal: {r}.")
    if rat.get("fit_now"):
        why.append(f"Why this strategy now: {rat['fit_now']}")
    ev = _evidence_line(cell)
    if ev:
        why.append(ev)
    if plan:
        acct = _f(settings.get("account_size")) or 0
        rp = _f(settings.get("risk_pct")) or 0
        why.append(
            f"Sizing: risking {_money(plan.get('risk_dollars'))} ({rp:g}% of your {_money(acct)} account). "
            f"The stop sits {abs(_f(plan.get('stop_pct')) or 0):.1f}% below entry at {_money(plan.get('stop'))} "
            f"({_f(plan.get('atr_mult')) or 0:g}× the stock's average daily range), so "
            f"{plan.get('shares')} shares loses exactly that amount if the stop is hit."
            + (" Size was capped by the max-position-value limit." if plan.get("capped_by_max_position") else "")
        )
        why.append(
            f"Target {_money(plan.get('target'))} (+{_f(plan.get('target_pct')) or 0:.1f}%) = "
            f"{_f(plan.get('r_multiple')) or 0:g}R: you aim to make {_f(plan.get('r_multiple')) or 0:g}× what you risk."
        )
    # The strategy's full why_it_works / fails_when essay lives once in the
    # Weekly Plan's "Strategies in play" step — repeating it on every buy
    # buried the ticker-specific reasons under identical paragraphs.
    return why


def _buy_orders(today: dict, cache: dict, positions: list[PortfolioPosition],
                sells: set[str], settings: dict, cells: dict) -> list[dict]:
    statuses = {s["id"]: s for s in today.get("strategy_regime_status") or []}
    quotes = {t: (cache.get(t) or {}).get("quote") or {} for t in cache}
    histories = {t: (cache.get(t) or {}).get("bars") or [] for t in cache}

    # The hypothetical book: current positions minus full sells proposed above.
    book: list[dict] = [
        {"ticker": p.ticker, "shares": p.shares, "avg_cost": p.avg_cost, "stop_loss": p.stop_loss}
        for p in positions if p.ticker not in sells
    ]
    picked = 0
    out = []
    for s in today.get("setups") or []:
        plan = s.get("plan")
        if not s.get("actionable") or not plan or not plan.get("shares"):
            continue
        ticker = s["ticker"]
        cell = cells.get(s.get("strategy") or "")
        why = _buy_why(s, statuses.get(s.get("strategy")), cell, settings)

        try:
            assess = portfolio_risk.assess_new_position(
                ticker, plan, book, quotes, histories, settings)
        except Exception as e:  # noqa: BLE001 — a failed check must not sink the plan
            assess = {"warnings": [{"level": "warn", "code": "assess_failed",
                                    "message": f"Risk check failed: {e}"}]}
        warnings = [w for w in (assess.get("warnings") or []) if w.get("level") in ("block", "warn")]
        blocks = [w for w in warnings if w.get("level") == "block"]

        skip = None
        if blocks:
            skip = "Blocked by a risk check: " + "; ".join(w.get("message", "") for w in blocks)
        elif (cell or {}).get("verdict") == "mis-tagged":
            skip = "The backtest says this strategy has not worked in the current regime."
        elif s.get("state") != "triggered":
            skip = "Setup is still FORMING — the entry signal hasn't fired yet. Watch it; don't buy early."
        elif picked >= MAX_NEW_BUYS_PER_WEEK:
            skip = (f"Lower priority — already {MAX_NEW_BUYS_PER_WEEK} buys pre-selected this week. "
                    f"Adding positions gradually spreads out timing risk.")
        for w in warnings:
            why.append(("BLOCKED: " if w.get("level") == "block" else "Risk note: ") + w.get("message", ""))

        zone = plan.get("entry_zone") or []
        limit = _f(zone[1]) if len(zone) == 2 else _f(plan.get("entry"))
        shares = int(plan["shares"])
        why.append(
            f"Limit {_money(limit)} is the top of the entry zone. If it opens above that, the order "
            f"simply won't fill — that's deliberate: chasing a gap wrecks the risk/reward."
        )
        recommended = skip is None
        if recommended:
            picked += 1
            book.append({"ticker": ticker, "shares": shares, "avg_cost": plan.get("entry"),
                         "stop_loss": plan.get("stop")})

        strat = STRATEGIES.get(s.get("strategy") or "")
        out.append({
            "id": f"buy:{ticker}",
            "kind": "buy",
            "side": "buy",
            "ticker": ticker,
            "name": s.get("name"),
            "sector": s.get("sector"),
            "shares": shares,
            "limit_price": limit,
            "est_value": round((limit or 0) * shares, 2),
            "recommended": recommended,
            "skip_reason": skip,
            "headline": f"Buy {shares} {ticker} (limit {_money(limit)}) — {s.get('strategy_name')}",
            "why": why,
            "strategy": s.get("strategy"),
            "strategy_name": s.get("strategy_name"),
            "state": s.get("state"),
            "evidence_verdict": (cell or {}).get("verdict"),
            "risk_warnings": warnings,
            "plan": {
                "stop": plan.get("stop"),
                "target": plan.get("target"),
                "strategy": s.get("strategy"),
                "strategy_name": s.get("strategy_name"),
                "planned_entry": plan.get("entry"),
                "planned_entry_high": limit,
                "thesis": f"{s.get('strategy_name')}: " + "; ".join(s.get("reasons") or []),
                "invalidation": f"Daily close below the stop at {_money(plan.get('stop'))}.",
                "time_stop_days": strat.horizon_days if strat else None,
            },
            "trade_plan": plan,
        })
    return out


# ── Top level ────────────────────────────────────────────────────────────────

async def build_weekly_plan(db: Session, user: User, force: bool = False) -> dict:
    today = await build_today(db, user.id, force=force)
    positions = db.query(PortfolioPosition).filter(PortfolioPosition.user_id == user.id).all()
    by_ticker = {p.ticker: p for p in positions}
    regime = today.get("regime") or {}
    settings_used = today.get("settings_used") or {}

    budget, basis = portfolio_risk.resolve_max_open_r(user)
    settings = dict(settings_used, max_open_r=budget)

    # 1. Positions
    reviews, sell_orders = [], []
    for p in today.get("positions") or []:
        p = dict(p, atr_mult=settings_used.get("atr_stop_mult"))
        review, order = _review_position(p, by_ticker.get(p["ticker"]))
        reviews.append(review)
        if order:
            sell_orders.append(order)
    full_sells = {o["ticker"] for o in sell_orders if o["kind"] == "sell" and o["recommended"]}

    # 2. Buys
    cells = _evidence_cells(user.id, regime.get("quadrant"))
    cache = _load_all_cache(db)
    buy_orders = _buy_orders(today, cache, positions, full_sells, settings, cells)

    # 3. Strategies in play, with the WHY
    counts: dict[str, int] = {}
    for s in today.get("setups") or []:
        counts[s.get("strategy")] = counts.get(s.get("strategy"), 0) + 1
    running, standing_down = [], []
    for st in today.get("strategy_regime_status") or []:
        row = {
            "id": st["id"], "name": st["name"], "actionable": st.get("actionable"),
            "rationale": st.get("rationale"), "how": st.get("how"), "horizon": st.get("horizon"),
            "reason": st.get("reason"), "setups": counts.get(st["id"], 0),
            "evidence": _evidence_line(cells.get(st["id"])),
            "verdict": (cells.get(st["id"]) or {}).get("verdict"),
        }
        (running if st.get("active") else standing_down).append(row)

    orders = sell_orders + buy_orders
    rec_sells = [o for o in sell_orders if o["recommended"]]
    rec_buys = [o for o in buy_orders if o["recommended"]]
    stop_raises = [r for r in reviews if r.get("raise_stop_to")]

    light = regime.get("light")
    stance = {
        "green": "Risk-on: the market backdrop supports adding positions.",
        "yellow": "Caution: be selective and keep position sizes honest.",
        "red": "Risk-off: protect capital first; new buys only from the defensive strategies.",
    }.get(light, "Regime unclear: be selective.")

    steps = [
        {"n": 1, "title": "Read the market",
         "detail": f"{regime.get('quadrant_label', 'Unknown')}. {stance}"},
        {"n": 2, "title": "Deal with what you own first",
         "detail": (f"{len(rec_sells)} sell/trim order(s) recommended, {len(stop_raises)} stop(s) to raise."
                    if (rec_sells or stop_raises) else "No holdings need action.")},
        {"n": 3, "title": "Pick new trades",
         "detail": (f"{len(rec_buys)} buy(s) recommended out of {len(buy_orders)} candidates; "
                    f"{today.get('capacity', {}).get('slots_free', 0)} slot(s) free.")},
        {"n": 4, "title": "Send the orders",
         "detail": "Tick the trades you agree with, then review and place them on the Trade tab."},
    ]
    headline = (
        f"Week of {week_of()}: {regime.get('quadrant_label', 'market regime unknown')} — "
        f"{len(rec_sells)} to sell, {len(stop_raises)} stops to raise, {len(rec_buys)} to buy."
    )

    payload = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "week_of": week_of(),
        "regime": {
            "quadrant": regime.get("quadrant"),
            "quadrant_label": regime.get("quadrant_label"),
            "quadrant_description": regime.get("quadrant_description"),
            "light": light,
            "stance": stance,
            "adx": regime.get("adx"),
            "adx_trend": regime.get("adx_trend"),
            "spy": regime.get("spy"),
            "vix": regime.get("vix"),
            "breadth": regime.get("breadth"),
            "reasons": regime.get("reasons") or [],
            "transition": regime.get("transition"),
        },
        "summary": {"headline": headline, "steps": steps},
        "strategies_running": running,
        "strategies_standing_down": standing_down,
        "selection": today.get("selection") or {},
        "position_reviews": reviews,
        "orders": orders,
        "capacity": today.get("capacity") or {},
        "risk_budget": {"max_open_r": budget, "basis": basis},
        "settings_used": settings_used,
        "evidence_available": bool(cells),
        "disclaimer": ("Decision support, not financial advice. Every order is a limit order you "
                       "choose to place; verify it against your own judgement first."),
    }
    return _json_safe(payload)
