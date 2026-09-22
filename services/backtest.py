import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from statistics import mean, pstdev

import numpy as np
from sqlalchemy.orm import Session

from database.models import PortfolioPosition, StockCache, WatchlistItem, WatchlistSnapshot
from services.indicators import calc_ma, calc_adx, calc_atr
from services.strategies import STRATEGIES
from services.regime import QUADRANTS, QUADRANT_INFO, trend_strength_label, classify_quadrant
from services.exits import (
    ExitRule, PartialProfitTaking, PositionState,
    advance, apply_partial, build_exit_rules, resolve_exit,
)
from services.trade_plan import build_trade_plan
from services.universe import UNIVERSE_CAVEAT


# Risk-free rate used for Sharpe/Sortino here. MUST stay in sync with
# services/indicators.py::compute_performance_metrics, which subtracts the same
# annual rate — otherwise the Sharpe shown on the Backtest page is not
# comparable to the Sharpe shown on the Screener/detail pages.
RISK_FREE_RATE = 0.05

# ── Simulation modes ──────────────────────────────────────────────────────
# "rotation"   — the original equal-weight top-N rotation with a turnover cost.
#                Kept byte-identical; every number the app has ever shown came
#                from this path.
# "trade_plan" — simulates what the user ACTUALLY does live: ATR stop, R target,
#                fixed-fractional sizing, a max-position-value cap, a finite
#                number of slots, and per-bar exit evaluation between rebalances.
MODES = ("rotation", "trade_plan")

# Defaults mirror services/today.py::build_today so a backtest and the live
# dashboard size the same trade the same way when the user has saved nothing.
DEFAULT_ACCOUNT_SIZE = 10000.0
DEFAULT_RISK_PCT = 1.0
DEFAULT_MAX_POSITIONS = 8
DEFAULT_ATR_STOP_MULT = 2.5
DEFAULT_R_MULTIPLE = 2.0
ATR_PERIOD = 14

# ── Sample-size honesty ───────────────────────────────────────────────────
# A 62% win rate over 11 periods and over 400 periods are not the same claim.
# Every regime_breakdown row carries a Wilson 95% interval and a confidence
# label derived from these thresholds — no magic numbers scattered inline.
WILSON_Z_95 = 1.959963984540054
MIN_PERIODS_LOW_CONFIDENCE = 30      # below this: "insufficient"
MIN_PERIODS_HIGH_CONFIDENCE = 100    # at/above this: "high"; between: "low"


def wilson_interval(successes: int, n: int, z: float = WILSON_Z_95) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion, in PERCENT.

    Preferred over the normal approximation because it stays inside [0, 1] and
    behaves at the extremes (0 or n successes) where a small sample is exactly
    where an honest backtest most needs a wide interval.
    """
    if n <= 0:
        return (0.0, 100.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (centre - half)) * 100, min(1.0, (centre + half)) * 100)


def confidence_label(n: int) -> str:
    """Map a sample size to high/low/insufficient."""
    if n < MIN_PERIODS_LOW_CONFIDENCE:
        return "insufficient"
    if n < MIN_PERIODS_HIGH_CONFIDENCE:
        return "low"
    return "high"


def _safe_float(value):
    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    except (TypeError, ValueError):
        return None


def _load_history(db: Session, ticker: str) -> list[dict]:
    row = db.query(StockCache).filter(StockCache.ticker == ticker).first()
    if not row or not row.history_json:
        return []
    try:
        bars = json.loads(row.history_json)
    except Exception:
        return []
    out = []
    for b in bars:
        close = _safe_float(b.get("close"))
        if not b.get("date") or not close:
            continue
        # Carry OHLC so high/low-dependent strategies (pullback, breakout) work.
        # Fall back to close when a legacy bar lacks high/low/open.
        out.append({
            "date": b.get("date"),
            "open": _safe_float(b.get("open")) or close,
            "high": _safe_float(b.get("high")) or close,
            "low": _safe_float(b.get("low")) or close,
            "close": close,
            "vol": _safe_float(b.get("vol")) or 0,
        })
    return out


def _load_quote(db: Session, ticker: str) -> dict:
    row = db.query(StockCache).filter(StockCache.ticker == ticker).first()
    if not row or not row.quote_json:
        return {"ticker": ticker}
    try:
        data = json.loads(row.quote_json)
        data["ticker"] = data.get("ticker") or ticker
        return data
    except Exception:
        return {"ticker": ticker}


def _source_tickers(db: Session, user_id: int, source: str) -> list[str]:
    tickers = set()
    if source == "universe":
        from services.universe import UNIVERSE
        cached = {t for (t,) in db.query(StockCache.ticker).filter(StockCache.history_json.isnot(None)).all()}
        tickers.update(cached & set(UNIVERSE))
    if source in ("watchlist", "both"):
        tickers.update(t for (t,) in db.query(WatchlistItem.ticker).filter(WatchlistItem.user_id == user_id).all())
    if source in ("portfolio", "both"):
        tickers.update(t for (t,) in db.query(PortfolioPosition.ticker).filter(PortfolioPosition.user_id == user_id).all())
    return sorted(tickers)


def _max_drawdown(equity: list[dict]) -> float:
    peak = equity[0]["value"] if equity else 1.0
    max_dd = 0.0
    for point in equity:
        value = point["value"]
        peak = max(peak, value)
        if peak > 0:
            max_dd = max(max_dd, (peak - value) / peak)
    return max_dd * 100


def _metrics(equity: list[dict], periods_per_year: float = 252 / 5) -> dict:
    if len(equity) < 2:
        return {"total_return": 0, "cagr": 0, "max_drawdown": 0, "sharpe": None, "sortino": None, "win_rate": None}
    returns = []
    for prev, curr in zip(equity, equity[1:]):
        if prev["value"] > 0:
            returns.append(curr["value"] / prev["value"] - 1)
    total_return = equity[-1]["value"] / equity[0]["value"] - 1
    start = datetime.fromisoformat(equity[0]["date"])
    end = datetime.fromisoformat(equity[-1]["date"])
    years = max((end - start).days / 365.25, 1 / 252)
    cagr = (equity[-1]["value"] / equity[0]["value"]) ** (1 / years) - 1
    vol = pstdev(returns) if len(returns) > 1 else 0
    # De-annualize the risk-free rate to one rebalance period so excess returns
    # are measured on the same clock as `returns` (compounding, not /ppy).
    rf_period = (1 + RISK_FREE_RATE) ** (1 / periods_per_year) - 1
    excess = [r - rf_period for r in returns]
    sharpe = (mean(excess) / vol * math.sqrt(periods_per_year)) if vol > 0 else None
    # Downside deviation of the EXCESS return relative to 0 — only periods that
    # underperform cash count as risk.
    downside = math.sqrt(mean([min(r, 0.0) ** 2 for r in excess])) if excess else 0
    sortino = (mean(excess) / downside * math.sqrt(periods_per_year)) if downside > 0 else None
    wins = [r for r in returns if r > 0]
    return {
        "total_return": round(total_return * 100, 2),
        "cagr": round(cagr * 100, 2),
        "max_drawdown": round(_max_drawdown(equity), 2),
        "sharpe": round(sharpe, 2) if sharpe is not None else None,
        "sortino": round(sortino, 2) if sortino is not None else None,
        "win_rate": round(len(wins) / len(returns) * 100, 1) if returns else None,
    }


def turnover_cost(holdings: set, prev_holdings: set, top_n: int, cost: float) -> float:
    """Round-trip transaction cost for rotating `prev_holdings` into `holdings`.

    symmetric_difference counts BOTH sides of the rotation (names sold + names
    bought), so one-way turnover is len(sym_diff) / (2 * top_n) and each side
    pays `cost`. The round-trip charge is therefore

        (len(sym_diff) / top_n) * cost

    with NO clip: it is already bounded at 2*cost for a complete rotation
    (sym_diff == 2*top_n → sell everything + buy everything). The previous
    min(1.0, turnover) clip charged a full rotation exactly the same as a 50%
    rotation, systematically understating trading costs for high-turnover
    strategies.
    """
    sym_diff = len(holdings.symmetric_difference(prev_holdings))
    return sym_diff / max(top_n, 1) * cost


def _value_on_or_before(series: list[dict], date: str) -> tuple[int, float] | None:
    for idx in range(len(series) - 1, -1, -1):
        if series[idx]["date"] <= date:
            return idx, series[idx]["close"]
    return None


def _week_start(value: date | None = None) -> date:
    value = value or datetime.now(timezone.utc).date()
    return value - timedelta(days=value.weekday())


def _parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _value_on_or_after(series: list[dict], day: date) -> tuple[int, float, str] | None:
    target = day.isoformat()
    for idx, point in enumerate(series):
        if point["date"] >= target:
            return idx, point["close"], point["date"]
    return None


def _latest_on_or_before(series: list[dict], day: date) -> tuple[int, float, str] | None:
    target = day.isoformat()
    for idx in range(len(series) - 1, -1, -1):
        if series[idx]["date"] <= target:
            return idx, series[idx]["close"], series[idx]["date"]
    return None


def _rank_candidate(series: list[dict], idx: int) -> dict | None:
    """Momentum ranking at a historical bar — delegates to the shared strategy."""
    return STRATEGIES["momentum_rotation"].candidate(series, idx)


def create_watchlist_snapshot(db: Session, user_id: int, week_start: date | None = None, notes: str | None = None) -> dict:
    week_start = _week_start(week_start)
    tickers = _source_tickers(db, user_id, "watchlist")
    row = (
        db.query(WatchlistSnapshot)
        .filter(WatchlistSnapshot.user_id == user_id, WatchlistSnapshot.week_start == week_start)
        .first()
    )
    if row:
        row.tickers_json = json.dumps(tickers)
        row.created_at = datetime.now(timezone.utc)
        row.notes = notes
    else:
        row = WatchlistSnapshot(
            user_id=user_id,
            week_start=week_start,
            tickers_json=json.dumps(tickers),
            notes=notes,
        )
        db.add(row)
    db.commit()
    db.refresh(row)
    return {
        "id": row.id,
        "week_start": row.week_start.isoformat(),
        "ticker_count": len(tickers),
        "tickers": tickers,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "notes": row.notes,
    }


def _evaluate_snapshot(db: Session, snapshot: WatchlistSnapshot, top_n: int, spy_regime: bool) -> dict:
    tickers = json.loads(snapshot.tickers_json or "[]")
    start_day = snapshot.week_start
    scheduled_end = start_day + timedelta(days=4)
    histories = {ticker: _load_history(db, ticker) for ticker in tickers}
    histories = {ticker: bars for ticker, bars in histories.items() if len(bars) >= 75}
    spy = _load_history(db, "SPY")
    spy_start = _value_on_or_after(spy, start_day)
    spy_end = _latest_on_or_before(spy, scheduled_end)
    today = datetime.now(timezone.utc).date()
    if spy_end and date.fromisoformat(spy_end[2]) < scheduled_end and today <= scheduled_end:
        status = "in_progress"
    else:
        status = "complete"

    if not spy_start or not spy_end or not histories:
        return {
            "id": snapshot.id,
            "week_start": start_day.isoformat(),
            "week_end": scheduled_end.isoformat(),
            "status": "waiting_for_data",
            "ticker_count": len(tickers),
            "available_tickers": sorted(histories),
            "tickers": tickers,
            "selected": [],
            "all_returns": [],
            "model_return": None,
            "spy_return": None,
            "notes": ["Need cached SPY history and at least 75 bars for saved tickers."],
        }

    spy_closes = [b["close"] for b in spy]
    spy_ma = calc_ma(spy_closes, 50)
    spy_idx, spy_start_px, start_date = spy_start
    _, spy_end_px, end_date = spy_end
    regime_ok = not spy_regime or (spy_ma[spy_idx] is not None and spy_start_px > spy_ma[spy_idx])

    ranked = []
    all_returns = []
    for ticker, bars in histories.items():
        start_point = _value_on_or_after(bars, start_day)
        end_point = _latest_on_or_before(bars, scheduled_end)
        if not start_point or not end_point or end_point[0] <= start_point[0]:
            continue
        start_idx, start_px, _ = start_point
        _, end_px, _ = end_point
        perf = (end_px / start_px - 1) * 100 if start_px else None
        rank = _rank_candidate(bars, start_idx)
        all_returns.append({
            "ticker": ticker,
            "return": round(perf, 2) if perf is not None else None,
            "selected": False,
            "score": round(rank["score"], 4) if rank else None,
            "rsi": round(rank["rsi"], 1) if rank and rank.get("rsi") is not None else None,
        })
        if regime_ok and rank:
            ranked.append((ticker, rank, perf))

    ranked.sort(key=lambda item: item[1]["score"], reverse=True)
    selected = ranked[: max(1, min(top_n, 20))]
    selected_tickers = {ticker for ticker, _, _ in selected}
    for row in all_returns:
        row["selected"] = row["ticker"] in selected_tickers
    all_returns.sort(key=lambda row: row["return"] if row["return"] is not None else -999, reverse=True)
    selected_rows = [
        {
            "ticker": ticker,
            "score": round(rank["score"], 4),
            "return": round(perf, 2) if perf is not None else None,
            "ret_21": rank.get("ret_21"),
            "ret_63": rank.get("ret_63"),
            "slope_ann": rank.get("slope_ann"),
            "r2": rank.get("r2"),
            "rsi": round(rank["rsi"], 1) if rank.get("rsi") is not None else None,
        }
        for ticker, rank, perf in selected
    ]
    model_values = [row["return"] for row in selected_rows if row["return"] is not None]

    return {
        "id": snapshot.id,
        "week_start": start_day.isoformat(),
        "week_end": scheduled_end.isoformat(),
        "actual_start": start_date,
        "actual_end": end_date,
        "status": status,
        "ticker_count": len(tickers),
        "available_tickers": sorted(histories),
        "tickers": tickers,
        "regime": "risk_on" if regime_ok else "cash",
        "selected": selected_rows,
        "all_returns": all_returns,
        "model_return": round(mean(model_values), 2) if model_values else 0.0,
        "spy_return": round((spy_end_px / spy_start_px - 1) * 100, 2),
        "notes": [
            "Replay uses the tickers saved in that week's snapshot, not today's edited watchlist.",
            "In-progress weeks use the latest cached close available so you can monitor performance before Friday.",
        ],
    }


def build_watchlist_replay(db: Session, user_id: int, weeks: int = 8, top_n: int = 10, spy_regime: bool = True) -> dict:
    rows = (
        db.query(WatchlistSnapshot)
        .filter(WatchlistSnapshot.user_id == user_id)
        .order_by(WatchlistSnapshot.week_start.desc())
        .limit(max(1, min(weeks, 26)))
        .all()
    )
    return {
        "current_week_start": _week_start().isoformat(),
        "snapshots": [_evaluate_snapshot(db, row, top_n=top_n, spy_regime=spy_regime) for row in rows],
        "parameters": {"weeks": weeks, "top_n": top_n, "spy_regime": spy_regime},
    }


# ──────────────────────────────────────────────────────────────────────────
# Trade-plan simulation machinery
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class OpenPosition:
    """One live simulated position plus the cursor into its own bar series."""
    ticker: str
    bars: list[dict]
    atr: list                       # ATR(14) series aligned 1:1 with `bars`
    quadrants: dict                 # date -> regime quadrant (shared, read-only)
    state: PositionState
    rules: list[ExitRule]
    idx: int                        # index of the LAST bar already processed
    entry_value: float = 0.0        # cash paid at entry, incl. cost
    proceeds: float = 0.0           # cash booked back from partials + exit
    fills: list = field(default_factory=list)
    closed: bool = False


def _bar_view(bar: dict, atr, quadrant) -> dict:
    """The enriched, single-bar view an exit rule is allowed to see.

    A COPY — the cached history dicts are shared across tickers/users and must
    never be mutated. This is also the structural no-look-ahead guarantee: a
    rule is handed one bar, never the series.
    """
    view = dict(bar)
    view["atr"] = atr
    view["quadrant"] = quadrant
    return view


def walk_position(pos: OpenPosition, until_date: str) -> list[dict]:
    """Advance one position bar-by-bar through `until_date` (inclusive).

    Returns the fills produced on the way. Exactly one exit signal is honoured
    per bar (priority-resolved by services/exits.resolve_exit), which is why a
    partial and a target can never both book on the same bar.
    """
    fills: list[dict] = []
    bars = pos.bars
    while not pos.closed and pos.idx + 1 < len(bars) and bars[pos.idx + 1]["date"] <= until_date:
        pos.idx += 1
        bar = bars[pos.idx]
        atr = pos.atr[pos.idx] if pos.idx < len(pos.atr) else None
        view = _bar_view(bar, atr, pos.quadrants.get(bar["date"]))
        signal = resolve_exit(pos.rules, view, pos.state)
        if signal is not None:
            if signal.fraction < 1.0:
                rule = next(
                    (r for r in pos.rules
                     if r.id == signal.rule and isinstance(r, PartialProfitTaking)),
                    None,
                )
                sold = apply_partial(signal, pos.state, rule)
                if sold:
                    fills.append({
                        "date": bar["date"], "price": signal.price, "shares": sold,
                        "reason": signal.reason, "kind": signal.kind, "closed": False,
                    })
            else:
                fills.append({
                    "date": bar["date"], "price": signal.price, "shares": pos.state.shares,
                    "reason": signal.reason, "kind": signal.kind, "closed": True,
                })
                pos.closed = True
                break
        advance(pos.rules, view, pos.state)
    pos.fills.extend(fills)
    return fills


def close_position_at(pos: OpenPosition, price: float, day: str, reason: str) -> dict:
    """Force-close at `price` (used for positions still open at test end)."""
    fill = {"date": day, "price": price, "shares": pos.state.shares,
            "reason": reason, "kind": "end_of_test", "closed": True}
    pos.closed = True
    pos.fills.append(fill)
    return fill


def _trade_record(pos: OpenPosition) -> dict:
    """Collapse a position's fills into one JSON-safe trade-log row."""
    st = pos.state
    gross = sum(f["shares"] * (f["price"] - st.entry_price) for f in pos.fills)
    denom = st.initial_shares * st.risk_per_share
    last = pos.fills[-1] if pos.fills else None
    pnl = pos.proceeds - pos.entry_value
    return {
        "ticker": pos.ticker,
        "strategy": st.strategy,
        "entry_date": st.entry_date,
        "entry_price": round(st.entry_price, 2),
        "exit_date": last["date"] if last else None,
        "exit_price": round(last["price"], 2) if last else None,
        "shares": st.initial_shares,
        "stop": round(st.initial_stop, 2),
        "target": round(st.target, 2) if st.target else None,
        "bars_held": st.bars_held,
        "exit_reason": last["reason"] if last else None,
        "exit_kind": last["kind"] if last else None,
        "scaled_out": st.scaled_out,
        "gross_pnl": round(gross, 2),
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl / pos.entry_value * 100, 2) if pos.entry_value else None,
        "r_multiple": round(gross / denom, 3) if denom else None,
    }


def run_walk_forward_backtest(
    db: Session,
    user_id: int,
    strategy_id: str = "momentum_rotation",
    source: str = "watchlist",
    top_n: int = 5,
    rebalance_days: int = 5,
    cost_bps: float = 10,
    spy_regime: bool = True,
    regime_ma: int = 200,
    period: str = "all",
    start_date: str | None = None,
    end_date: str | None = None,
    archive: bool = False,
    mode: str = "rotation",
    account_size: float | None = None,
    risk_pct: float | None = None,
    max_positions: int | None = None,
    atr_stop_mult: float | None = None,
    r_multiple: float | None = None,
    exit_rules: str | list[str] | None = None,
) -> dict:
    strategy = STRATEGIES.get(strategy_id) or STRATEGIES["momentum_rotation"]
    strategy_meta = {"id": strategy.id, "name": strategy.name}
    mode = mode if mode in MODES else "rotation"
    params = {
        "strategy": strategy.id,
        "source": source,
        "top_n": top_n,
        "rebalance_days": rebalance_days,
        "cost_bps": cost_bps,
        "spy_regime": spy_regime,
        "regime_ma": regime_ma,
        "period": period,
        "start_date": start_date,
        "end_date": end_date,
        "archive": archive,
    }
    # Trade-plan knobs only appear in `parameters` for trade_plan runs so a
    # rotation response stays byte-for-byte what it has always been.
    if mode == "trade_plan":
        account_size = float(account_size or DEFAULT_ACCOUNT_SIZE)
        risk_pct = float(risk_pct or DEFAULT_RISK_PCT)
        max_positions = int(max_positions or DEFAULT_MAX_POSITIONS)
        atr_stop_mult = float(atr_stop_mult or DEFAULT_ATR_STOP_MULT)
        r_multiple = float(r_multiple or DEFAULT_R_MULTIPLE)
        params.update({
            "mode": mode,
            "account_size": account_size,
            "risk_pct": risk_pct,
            "max_positions": max_positions,
            "atr_stop_mult": atr_stop_mult,
            "r_multiple": r_multiple,
            "exit_rules": exit_rules or "fixed",
        })
    # Non-actionable strategies (e.g. bear_reversal_watch) are watchlist-only
    # signals. This engine models a LONG-ONLY equal-weight rotation, so running
    # one here would produce a long equity curve for a "do not buy" signal.
    # api/backtest.py already excludes them from the accepted query pattern —
    # this is the defensive second gate for direct/internal callers.
    if not strategy.actionable:
        return {
            "strategy": strategy_meta, "source": source, "parameters": params,
            "available_tickers": [], "equity": [], "benchmark": [], "trades": [],
            "metrics": {}, "benchmark_metrics": {}, "regime_breakdown": [],
            "strategy_regimes": strategy.regimes,
            "notes": [
                f"{strategy.name} is a watchlist-only signal (actionable=False) and is not "
                "backtestable here: the walk-forward engine simulates a long-only "
                "equal-weight rotation, so a bearish/informational signal would produce a "
                "meaningless long equity curve.",
                "Pick an actionable strategy to run a walk-forward backtest.",
            ],
        }

    # Warm-up: enough bars for both the regime MA and the strategy's own filters
    warmup = max(regime_ma, strategy.min_bars) + 5

    tickers = _source_tickers(db, user_id, source)

    if archive:
        # Arbitrary-era mode: fetch/reuse a long-history archive for the exact
        # [start_date, end_date] window (plus a warmup buffer before it).
        if not (start_date and end_date):
            return {
                "strategy": strategy_meta, "source": source, "parameters": params,
                "available_tickers": [], "equity": [], "benchmark": [], "trades": [],
                "metrics": {}, "benchmark_metrics": {},
                "notes": ["Archive mode requires both a start date and an end date."],
            }
        from services.history_archive import (
            load_archived_histories, ensure_archive, shift_days, WARMUP_BUFFER_DAYS,
        )
        buf_start = shift_days(start_date, -WARMUP_BUFFER_DAYS)
        histories = load_archived_histories(db, tickers, buf_start, end_date)
        spy = ensure_archive(db, "SPY", buf_start, end_date)
    else:
        histories = {ticker: _load_history(db, ticker) for ticker in tickers}
        spy = _load_history(db, "SPY")

    histories = {ticker: bars for ticker, bars in histories.items() if len(bars) >= warmup}
    dates = [b["date"] for b in spy] if len(spy) >= warmup else []
    dates = dates[warmup:: max(1, rebalance_days)]

    # Restrict to the requested window: a period preset (1Y/2Y/all) and/or a
    # custom start date. A custom start with a period bounds both ends
    # (start → start+period); a period alone uses the most recent window.
    full_dates = [b["date"] for b in spy]
    latest = full_dates[-1] if full_dates else None
    years = {"1Y": 1, "2Y": 2}.get((period or "all").upper())

    def _shift_year(iso: str, yrs: int) -> str:
        from datetime import date as _d
        y, m, d = (int(x) for x in iso.split("-")[:3])
        try:
            return _d(y + yrs, m, d).isoformat()
        except ValueError:          # e.g. Feb 29 → Feb 28
            return _d(y + yrs, m, 28).isoformat()

    win_start = win_end = None
    if start_date:
        win_start = start_date
        if years:
            win_end = _shift_year(start_date, years)
    elif years and latest:
        win_start = _shift_year(latest, -years)
    if end_date:                    # explicit end overrides the period-derived end
        win_end = end_date
    if win_start:
        dates = [d for d in dates if d >= win_start]
    if win_end:
        dates = [d for d in dates if d <= win_end]

    if len(dates) < 4 or not histories:
        return {
            "strategy": strategy_meta,
            "source": source,
            "parameters": params,
            "available_tickers": sorted(histories),
            "equity": [],
            "benchmark": [],
            "trades": [],
            "metrics": {},
            "benchmark_metrics": {},
            "notes": [
                f"Need cached SPY history and at least {warmup} bars for selected tickers.",
                "Tip: use the Full Universe source after the 2-year history backfill completes.",
            ],
        }

    equity = [{"date": dates[0], "value": 1.0}]
    benchmark = [{"date": dates[0], "value": 1.0}]
    trades = []
    prev_holdings: set[str] = set()
    cost = cost_bps / 10000

    spy_closes = [b["close"] for b in spy]
    spy_ma = calc_ma(spy_closes, regime_ma)
    spy_ma200 = calc_ma(spy_closes, 200)
    adx_full = calc_adx(
        [b["high"] for b in spy], [b["low"] for b in spy], spy_closes, 14
    )["series"]
    # Rolling 20-day realized volatility — the same VIX-proxy classify_regime()
    # falls back to, precomputed once so per-period regime tagging is O(1).
    vol20 = [None] * len(spy_closes)
    for i in range(20, len(spy_closes)):
        window = spy_closes[i - 20: i + 1]
        rets = [(window[j] / window[j - 1] - 1) for j in range(1, len(window))]
        if rets:
            vol20[i] = (sum(r * r for r in rets) / len(rets)) ** 0.5 * (252 ** 0.5) * 100

    def _quadrant_at(idx: int) -> tuple[str, float | None]:
        """Regime quadrant + ADX at a SPY bar index (no look-ahead: index only)."""
        adx_v = adx_full[idx]["adx"] if adx_full[idx] else None
        above = spy_ma200[idx] is not None and spy_closes[idx] > spy_ma200[idx]
        crisis = vol20[idx] is not None and vol20[idx] >= 22.0
        return classify_quadrant(trend_strength_label(adx_v), above, crisis), adx_v

    trade_log: list[dict] = []
    tp_stats: dict = {}

    if mode == "trade_plan":
        # Per-bar regime map, precomputed once — RegimeExit needs a quadrant on
        # every bar, not just rebalance dates.
        quadrants_by_date = {}
        for i in range(len(spy)):
            quadrants_by_date[spy[i]["date"]] = _quadrant_at(i)[0]

        atr_cache: dict[str, list] = {}

        def _atr_for(ticker: str, bars: list[dict]) -> list:
            series = atr_cache.get(ticker)
            if series is None:
                # ONCE per ticker for the whole run — never per bar.
                series = calc_atr([b["high"] for b in bars], [b["low"] for b in bars],
                                  [b["close"] for b in bars], ATR_PERIOD)
                atr_cache[ticker] = series
            return series

        cash = float(account_size)
        open_positions: dict[str, OpenPosition] = {}
        skipped_full_book = 0
        skipped_no_cash = 0
        skipped_no_plan = 0

        for i, start in enumerate(dates):
            spy_start = _value_on_or_before(spy, start)
            if not spy_start:
                continue
            spy_idx, spy_start_px = spy_start
            regime_ok = not spy_regime or (spy_ma[spy_idx] is not None and spy_start_px > spy_ma[spy_idx])
            quadrant, adx_val = _quadrant_at(spy_idx)

            # ── 1. ENTRIES at the close of `start`.
            # The decision uses strategy.candidate(bars, idx), which only reads
            # bars[:idx+1], and the fill is THAT bar's close. Nothing later is
            # visible at this point in the simulation.
            equity_now = cash + sum(
                p.state.shares * p.bars[p.idx]["close"] for p in open_positions.values()
            )
            if regime_ok and len(open_positions) < max_positions:
                ranked = []
                for ticker, bars in histories.items():
                    if ticker in open_positions or not strategy.applies_to(ticker):
                        continue
                    point = _value_on_or_before(bars, start)
                    # Require a bar ON the rebalance date: filling at a stale
                    # close from an earlier session would be a fabricated price.
                    if not point or bars[point[0]]["date"] != start:
                        continue
                    rank = strategy.candidate(bars, point[0])
                    if rank:
                        ranked.append((ticker, rank, point[0]))
                ranked.sort(key=lambda item: item[1]["score"], reverse=True)

                for ticker, rank, idx in ranked[: max(1, min(top_n, 20))]:
                    if len(open_positions) >= max_positions:
                        skipped_full_book += 1
                        continue
                    bars = histories[ticker]
                    plan = build_trade_plan(
                        bars[: idx + 1], equity_now, risk_pct,
                        bars[idx]["close"], atr_stop_mult, r_multiple,
                    )
                    if not plan or not plan.get("shares"):
                        skipped_no_plan += 1
                        continue
                    entry_px = float(plan["entry"])
                    stop_px = float(plan["stop"])
                    risk_ps = entry_px - stop_px
                    if risk_ps <= 0 or entry_px <= 0:
                        skipped_no_plan += 1
                        continue
                    shares = int(plan["shares"])
                    outlay = shares * entry_px * (1 + cost)
                    if outlay > cash:
                        shares = int(cash / (entry_px * (1 + cost)))
                        if shares < 1:
                            skipped_no_cash += 1
                            continue
                        outlay = shares * entry_px * (1 + cost)
                    cash -= outlay

                    atr_series = _atr_for(ticker, bars)
                    rules = build_exit_rules(
                        exit_rules, atr_mult=atr_stop_mult, regimes=strategy.regimes,
                        # Lets the time stop default to THIS strategy's own
                        # horizon_days instead of a flat 20 bars.
                        strategy_id=strategy.id,
                    )
                    state = PositionState(
                        ticker=ticker, entry_date=start, entry_price=entry_px,
                        entry_idx=idx, shares=shares, initial_shares=shares,
                        stop=stop_px, initial_stop=stop_px, risk_per_share=risk_ps,
                        target=float(plan["target"]), strategy=strategy.id,
                    )
                    pos = OpenPosition(
                        ticker=ticker, bars=bars, atr=atr_series,
                        quadrants=quadrants_by_date, state=state, rules=rules,
                        idx=idx, entry_value=outlay,
                    )
                    # Seed path-dependent rule state (highest close, trail level)
                    # from the ENTRY bar, so the level checked tomorrow was
                    # derived from data available today.
                    advance(rules, _bar_view(bars[idx], atr_series[idx] if idx < len(atr_series) else None,
                                             quadrants_by_date.get(start)), state)
                    open_positions[ticker] = pos

            end = dates[i + 1] if i + 1 < len(dates) else None
            if end is None:
                break
            spy_end = _value_on_or_before(spy, end)
            if not spy_end:
                continue
            _, spy_end_px = spy_end

            # ── 2. Walk every open position bar-by-bar through `end`.
            exits_this_period = 0
            for ticker, pos in list(open_positions.items()):
                for fill in walk_position(pos, end):
                    proceeds = fill["shares"] * fill["price"] * (1 - cost)
                    cash += proceeds
                    pos.proceeds += proceeds
                if pos.closed:
                    trade_log.append(_trade_record(pos))
                    open_positions.pop(ticker, None)
                    exits_this_period += 1

            # ── 3. Mark to market.
            total = cash + sum(
                p.state.shares * p.bars[p.idx]["close"] for p in open_positions.values()
            )
            value = total / account_size
            prev_value = equity[-1]["value"]
            period_ret = (value / prev_value - 1) if prev_value else 0.0
            spy_ret = (spy_end_px / spy_start_px - 1) if spy_start_px else 0.0

            equity.append({"date": end, "value": round(value, 6)})
            benchmark.append({"date": end, "value": round(benchmark[-1]["value"] * (spy_end_px / spy_start_px), 6)})
            trades.append({
                "date": start,
                "holdings": [
                    {"ticker": p.ticker, "shares": p.state.shares,
                     "entry_date": p.state.entry_date,
                     "open_r": round(p.state.r_at(p.bars[p.idx]["close"]), 2)}
                    for p in open_positions.values()
                ],
                "regime": "risk_on" if regime_ok else "cash",
                "quadrant": quadrant,
                "quadrant_label": QUADRANT_INFO[quadrant]["label"],
                "adx": round(adx_val, 1) if adx_val is not None else None,
                "period_return": round(period_ret * 100, 2),
                "spy_return": round(spy_ret * 100, 2),
                "excess_return": round((period_ret - spy_ret) * 100, 2),
                "exits": exits_this_period,
                "cash_pct": round(cash / total * 100, 1) if total else None,
            })

        # Force-close anything still open at the end of the test so the trade
        # stats describe every dollar committed, not just the closed trades.
        for ticker, pos in list(open_positions.items()):
            px = pos.bars[pos.idx]["close"]
            fill = close_position_at(pos, px, pos.bars[pos.idx]["date"], "open at end of test")
            pos.proceeds += fill["shares"] * fill["price"] * (1 - cost)
            trade_log.append(_trade_record(pos))
            open_positions.pop(ticker, None)

        rs = [t["r_multiple"] for t in trade_log if t["r_multiple"] is not None]
        wins = [t for t in trade_log if (t["pnl"] or 0) > 0]
        losses = [t for t in trade_log if (t["pnl"] or 0) <= 0]
        win_rs = [t["r_multiple"] for t in wins if t["r_multiple"] is not None]
        loss_rs = [t["r_multiple"] for t in losses if t["r_multiple"] is not None]
        p_win = (len(win_rs) / len(rs)) if rs else 0.0
        expectancy = (p_win * mean(win_rs) if win_rs else 0.0) + \
                     ((1 - p_win) * mean(loss_rs) if loss_rs else 0.0)
        tp_stats = {
            "avg_r": round(mean(rs), 3) if rs else None,
            "expectancy_r": round(expectancy, 3) if rs else None,
            "win_rate_trades": round(len(wins) / len(trade_log) * 100, 1) if trade_log else None,
            "trades_taken": len(trade_log),
            "signals_skipped_full_book": skipped_full_book,
            "signals_skipped_no_cash": skipped_no_cash,
            "signals_skipped_no_plan": skipped_no_plan,
            "avg_bars_held": round(mean([t["bars_held"] for t in trade_log]), 1) if trade_log else None,
            "ending_cash_pct": round(cash / (equity[-1]["value"] * account_size) * 100, 1)
                               if equity and equity[-1]["value"] else None,
        }
        rebalance_windows: list = []
    else:
        rebalance_windows = list(zip(dates, dates[1:]))

    for start, end in rebalance_windows:
        spy_start = _value_on_or_before(spy, start)
        spy_end = _value_on_or_before(spy, end)
        if not spy_start or not spy_end:
            continue
        spy_idx, spy_start_px = spy_start
        _, spy_end_px = spy_end
        regime_ok = not spy_regime or (spy_ma[spy_idx] is not None and spy_start_px > spy_ma[spy_idx])

        adx_val = adx_full[spy_idx]["adx"] if adx_full[spy_idx] else None
        above_200_here = spy_ma200[spy_idx] is not None and spy_start_px > spy_ma200[spy_idx]
        crisis_vol_here = vol20[spy_idx] is not None and vol20[spy_idx] >= 22.0
        quadrant = classify_quadrant(trend_strength_label(adx_val), above_200_here, crisis_vol_here)

        ranked = []
        if regime_ok:
            for ticker, bars in histories.items():
                if not strategy.applies_to(ticker):
                    continue
                start_point = _value_on_or_before(bars, start)
                end_point = _value_on_or_before(bars, end)
                if not start_point or not end_point or end_point[0] <= start_point[0]:
                    continue
                rank = strategy.candidate(bars, start_point[0])
                if rank:
                    ranked.append((ticker, rank, start_point[1], end_point[1]))
        ranked.sort(key=lambda item: item[1]["score"], reverse=True)
        selected = ranked[: max(1, min(top_n, 20))]
        holdings = {t for t, *_ in selected}

        period_ret = 0.0
        if selected:
            period_ret = mean((end_px / start_px - 1) for _, _, start_px, end_px in selected)
        # Round-trip turnover cost, uncapped — see turnover_cost() above.
        period_ret -= turnover_cost(holdings, prev_holdings, top_n, cost)
        spy_ret = (spy_end_px / spy_start_px - 1) if spy_start_px else 0.0

        equity.append({"date": end, "value": round(equity[-1]["value"] * (1 + period_ret), 6)})
        benchmark.append({"date": end, "value": round(benchmark[-1]["value"] * (spy_end_px / spy_start_px), 6)})
        trades.append({
            "date": start,
            "holdings": [
                {
                    "ticker": ticker,
                    "score": round(rank["score"], 4),
                    "ret_21": rank.get("ret_21"),
                    "ret_63": rank.get("ret_63"),
                    "rsi": round(rank["rsi"], 1) if rank.get("rsi") is not None else None,
                }
                for ticker, rank, _, _ in selected
            ],
            "regime": "risk_on" if regime_ok else "cash",
            "quadrant": quadrant,
            "quadrant_label": QUADRANT_INFO[quadrant]["label"],
            "adx": round(adx_val, 1) if adx_val is not None else None,
            "period_return": round(period_ret * 100, 2),
            "spy_return": round(spy_ret * 100, 2),
            "excess_return": round((period_ret - spy_ret) * 100, 2),
        })
        prev_holdings = holdings

    ppy = 252 / max(1, rebalance_days)
    metrics = _metrics(equity, ppy)
    benchmark_metrics = _metrics(benchmark, ppy)
    # Exposure: share of rebalance periods actually holding positions, and the
    # average return earned in just those periods (fair view for strategies
    # that sit in cash a lot, e.g. mean reversion).
    invested = [t for t in trades if t["holdings"]]
    metrics["exposure"] = round(len(invested) / len(trades) * 100, 1) if trades else None
    metrics["avg_invested_return"] = round(mean(t["period_return"] for t in invested), 2) if invested else None
    benchmark_metrics["exposure"] = 100.0 if trades else None
    # Per-TRADE statistics only exist in trade_plan mode — a rotation has no
    # discrete trades, only rebalance periods, so these keys stay absent there.
    metrics.update(tp_stats)

    # Per-regime breakdown — this is the "does the strategy actually earn its
    # keep in the regime it's tagged for" check. A strategy tagged for
    # trending_bull that loses money in trending_bull periods here is a red
    # flag no aggregate Sharpe number would show.
    regime_breakdown = []
    for q in QUADRANTS:
        rets = [t["period_return"] for t in trades if t["quadrant"] == q]
        if not rets:
            continue
        wins = [r for r in rets if r > 0]
        cum = 1.0
        for r in rets:
            cum *= (1 + r / 100)
        # Sample-size honesty: a win rate with no interval around it invites
        # reading 7-of-11 as an edge. Wilson handles small n and the extremes.
        ci_low, ci_high = wilson_interval(len(wins), len(rets))
        regime_breakdown.append({
            "quadrant": q,
            "label": QUADRANT_INFO[q]["label"],
            "periods": len(rets),
            "pct_of_periods": round(len(rets) / len(trades) * 100, 1) if trades else 0,
            "avg_period_return": round(mean(rets), 2),
            "win_rate": round(len(wins) / len(rets) * 100, 1),
            "cum_return": round((cum - 1) * 100, 2),
            "win_rate_ci_low": round(ci_low, 1),
            "win_rate_ci_high": round(ci_high, 1),
            "confidence": confidence_label(len(rets)),
        })

    caveats = [
        {
            "id": "survivorship_bias",
            "severity": "high",
            "title": "Survivorship bias inflates every number here",
            # ONE source of truth for this text: services/universe.py, so the
            # backtest, the edge matrix and the scorecard cannot drift apart on
            # how serious the bias is. The {id, severity, title, detail} shape
            # and the stable `id` are what the front end keys off.
            "detail": UNIVERSE_CAVEAT,
        },
        {
            "id": "mode",
            "severity": "info",
            "title": f"Produced in '{mode}' mode",
            "detail": (
                "rotation: equal-weight top-N rebalanced on a fixed cadence with a "
                "turnover cost; positions are marked to market only on rebalance dates."
                if mode == "rotation" else
                "trade_plan: each entry is sized by services/trade_plan.py (fixed-fractional "
                "risk + max-position cap), capped at max_positions slots, and exits are "
                "evaluated bar-by-bar between rebalances by services/exits.py."
            ),
        },
    ]
    if mode == "rotation":
        caveats.append({
            "id": "no_intrabar_exits",
            "severity": "high",
            "title": "Stops and targets are NOT simulated",
            "detail": (
                "Rotation mode holds every selected name for the full rebalance period. "
                "It models no ATR stop, no R target, no position sizing and no slot limit, "
                "so its returns are not what the live trade plans would have produced. "
                "Run mode=trade_plan for that comparison."
            ),
        })
    else:
        caveats.append({
            "id": "intrabar_assumption",
            "severity": "medium",
            "title": "Intrabar fills assume the worst",
            "detail": (
                "Daily bars do not reveal whether the high or the low came first. When a "
                "bar breaches the stop and reaches the target, the STOP is assumed to fill "
                "first, and a gap through a level fills at the open, not the level. This is "
                "deliberately pessimistic; the opposite assumption manufactures winners."
            ),
        })
    if metrics.get("trades_taken") is not None and metrics["trades_taken"] < MIN_PERIODS_LOW_CONFIDENCE:
        caveats.append({
            "id": "few_trades",
            "severity": "high",
            "title": f"Only {metrics['trades_taken']} trades",
            "detail": (
                f"Fewer than {MIN_PERIODS_LOW_CONFIDENCE} closed trades. Expectancy and win "
                "rate over this few samples are noise, not evidence."
            ),
        })

    return {
        "strategy": strategy_meta,
        "source": source,
        "parameters": params,
        "available_tickers": sorted(histories),
        "equity": equity,
        "benchmark": benchmark,
        "trades": trades,
        "metrics": metrics,
        "benchmark_metrics": benchmark_metrics,
        "regime_breakdown": regime_breakdown,
        "strategy_regimes": strategy.regimes,
        "mode": mode,
        "trade_log": trade_log,
        "caveats": caveats,
        "notes": [
            "Hypothetical backtest using cached adjusted daily closes only.",
            f"Strategy: {strategy.name}. {strategy.description}",
            "Modeled as an equal-weight top-N rotation rebalanced on the chosen cadence "
            "with turnover cost and an optional SPY regime filter. ATR stops/targets are "
            "NOT simulated intrabar — live trade plans add those on top."
            if mode == "rotation" else
            "Modeled as the live trade plan: fixed-fractional sizing off current equity, "
            f"{atr_stop_mult}x ATR stop, {r_multiple}R target, a max-position-value cap, at "
            f"most {max_positions} concurrent positions, and per-bar exit evaluation between "
            "rebalances. Cash earns nothing.",
            "regime_breakdown classifies each rebalance period by ADX(14) trend strength + "
            "200-day MA direction (+ realized volatility as a VIX proxy) — compare it "
            "against this strategy's tagged regimes to see if the edge actually shows up "
            "where it's supposed to.",
        ],
    }


def build_decision_cockpit(db: Session, user_id: int) -> dict:
    positions = db.query(PortfolioPosition).filter(PortfolioPosition.user_id == user_id).all()
    watchlist = db.query(WatchlistItem).filter(WatchlistItem.user_id == user_id).all()
    portfolio_tickers = [p.ticker for p in positions]
    watch_tickers = [w.ticker for w in watchlist]
    quotes = {ticker: _load_quote(db, ticker) for ticker in sorted(set(portfolio_tickers + watch_tickers))}

    def setup_score(q: dict) -> float:
        sc = q.get("score") or {}
        score = (sc.get("o") or 0) * 10
        score += 8 if 40 <= (q.get("rsi") or 0) <= 65 else 0
        score += 6 if (q.get("vs_ma200") or 0) > 0 else -8
        # NOTE: vol_r semantics changed — it is now today's volume / prior 20-day
        # average (a conventional single-bar ratio), where it used to be a
        # 5-day/20-day average ratio (now quote["vol_r_5d"]). The 1.15 threshold
        # is kept deliberately: it still reads as "today traded meaningfully
        # above its recent average", just on a single bar instead of a 5-day
        # smoothed window, so it is noisier but more timely.
        score += 4 if (q.get("vol_r") or 0) > 1.15 else 0
        score += 4 if (q.get("sharpe") or 0) > 1 else 0
        score -= min(8, (q.get("max_dd_1m") or 0) * 0.4)
        return round(score, 2)

    opportunities = sorted(
        [
            {
                "ticker": t,
                "name": quotes[t].get("name"),
                "sector": quotes[t].get("sector"),
                "setup_score": setup_score(quotes[t]),
                "price": quotes[t].get("price"),
                "rsi": quotes[t].get("rsi"),
                "vol_r": quotes[t].get("vol_r"),
                "score": quotes[t].get("score"),
                "why": [
                    label for ok, label in [
                        (40 <= (quotes[t].get("rsi") or 0) <= 65, "RSI in swing zone"),
                        ((quotes[t].get("vs_ma200") or 0) > 0, "above 200MA"),
                        ((quotes[t].get("vol_r") or 0) > 1.15, "volume confirmation"),
                        ((quotes[t].get("sharpe") or 0) > 1, "strong risk-adjusted return"),
                    ] if ok
                ],
            }
            for t in watch_tickers
        ],
        key=lambda row: row["setup_score"],
        reverse=True,
    )[:12]

    from services.today import position_flags
    exits = []
    for pos in positions:
        q = quotes.get(pos.ticker, {})
        reasons = position_flags(pos, q)["reasons"]
        if reasons:
            exits.append({
                "ticker": pos.ticker,
                "reasons": reasons,
                "price": q.get("price"),
                "rsi": q.get("rsi"),
                # ann_ret is now the FULL-HISTORY annualized % (the one Sharpe/
                # Sortino/Calmar use); ann_ret_1m is the 21-bar annualized % that
                # position_flags() flags on. Surface both so the UI can't confuse them.
                "ann_ret": q.get("ann_ret"),
                "ann_ret_1m": q.get("ann_ret_1m"),
                "sharpe": q.get("sharpe"),
                "vs_ma200": q.get("vs_ma200"),
            })

    sectors: dict[str, float] = {}
    total_mv = 0.0
    weighted_beta = 0.0
    for pos in positions:
        q = quotes.get(pos.ticker, {})
        price = q.get("price") or pos.avg_cost
        mv = pos.shares * price
        total_mv += mv
        sectors[q.get("sector") or "Unknown"] = sectors.get(q.get("sector") or "Unknown", 0) + mv
        if q.get("beta") is not None:
            weighted_beta += mv * q["beta"]

    sector_concentration = [
        {"sector": sector, "weight": round(value / total_mv * 100, 1) if total_mv else 0}
        for sector, value in sorted(sectors.items(), key=lambda item: item[1], reverse=True)
    ]

    risk = {
        "positions": len(positions),
        "market_value": round(total_mv, 2),
        "weighted_beta": round(weighted_beta / total_mv, 2) if total_mv else None,
        "top_sector": sector_concentration[0] if sector_concentration else None,
        "sector_concentration": sector_concentration,
        "exit_flags": len(exits),
    }

    changes = []
    for ticker, q in quotes.items():
        # gc/dc are now CROSSOVER EVENTS (a cross in the last 5 bars), not the
        # standing MA50-vs-MA200 state — use the explicit *_event names and say
        # "crossed". The standing state lives in quote["ma_state"] ("bull"/"bear").
        if q.get("gc_event"):
            changes.append({"ticker": ticker, "type": "trend",
                            "message": "Golden cross — MA50 crossed above MA200"})
        if q.get("dc_event"):
            changes.append({"ticker": ticker, "type": "trend",
                            "message": "Death cross — MA50 crossed below MA200"})
        if (q.get("vol_r") or 0) > 1.5:
            changes.append({"ticker": ticker, "type": "volume", "message": f"volume ratio {q.get('vol_r')}"})
        if q.get("rsi") is not None and (q["rsi"] > 70 or q["rsi"] < 30):
            changes.append({"ticker": ticker, "type": "rsi", "message": f"RSI {q['rsi']}"})

    return {
        "risk": risk,
        "opportunities": opportunities,
        "exit_pressure": exits,
        "changes": changes[:20],
        "disclaimer": "Decision-support data only. Review position sizing, taxes, liquidity, and personal risk tolerance before acting.",
    }
