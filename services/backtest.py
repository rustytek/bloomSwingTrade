import json
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from bisect import bisect_left, bisect_right
from datetime import date, datetime, timedelta, timezone
from statistics import mean, pstdev

import numpy as np
from sqlalchemy.orm import Session

from database.models import PortfolioPosition, StockCache, WatchlistItem, WatchlistSnapshot
from services.indicators import calc_ma, calc_adx, calc_atr
from services.strategies import STRATEGIES
from services.regime import (
    QUADRANTS, QUADRANT_INFO, VIX_CRISIS, trend_strength_label, classify_quadrant,
)
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

# ── Price history depth vs test window ────────────────────────────────────
# `history` is which DATA the run can draw on:
#   "5y"  — the rolling StockCache only.
#   "20y" — services/long_history.py: the 20-year archive spliced onto the
#           cache (falls back to the cache per ticker until the backfill ran).
# The TEST WINDOW is separate and capped at MAX_WINDOW_YEARS for Strategy Lab
# runs: pick where the 5 years sit with start_date (e.g. 2007 → 2012 to test
# through 2008). The edge matrix passes max_window_years=None — its whole job
# is to see every regime across all 20 years, and it runs in the job worker.
HISTORY_MODES = ("5y", "20y")
PERIOD_YEARS = {"1Y": 1, "2Y": 2, "5Y": 5}
MAX_WINDOW_YEARS = 5
# Bars loaded before the window starts, beyond the strategy warm-up, so the
# ATR (Wilder-smoothed from the first bar it sees) has settled by day one.
_PRE_WINDOW_EXTRA_BARS = 400

# How many tickers' full bar series the simulation keeps loaded at once. The
# scan holds ONE ticker at a time; only open positions need full bars later.
_BAR_LRU_SIZE = 24

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
    return _normalize_bars(bars)


_INF = (math.inf, -math.inf)


def _num(value):
    """_safe_float with a fast path: JSON already gives finite floats almost
    always, and the per-field function call dominated a 20-year load."""
    if type(value) is float and value == value and value not in _INF:
        return value
    return _safe_float(value)


def _normalize_bars(bars: list[dict]) -> list[dict]:
    out = []
    append = out.append
    for b in bars:
        close = _num(b.get("close"))
        if not b.get("date") or not close:
            continue
        # Carry OHLC so high/low-dependent strategies (pullback, breakout) work.
        # Fall back to close when a legacy bar lacks high/low/open.
        append({
            "date": b.get("date"),
            "open": _num(b.get("open")) or close,
            "high": _num(b.get("high")) or close,
            "low": _num(b.get("low")) or close,
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


def sharpe_inference(equity: list[dict], periods_per_year: float = 252 / 5) -> dict | None:
    """How much the run's Sharpe ratio can be trusted (ML4T 3e §16.7).

    Uses the same per-period excess returns as `_metrics`. The standard error
    allows for skew and fat tails (Mertens), so a strategy with rare big losses
    gets a wider interval than a normal-returns formula would give it.
    `psr_vs_zero` is the probability the true Sharpe is above zero; the
    Deflated Sharpe Ratio (which also allows for how many variants were tried)
    is added by services/trial_log.py, since only it knows that number.
    Kept OUT of `metrics` so the frozen rotation digest is unaffected."""
    from services.stats import moments, probabilistic_sharpe, sharpe_se
    returns = [curr["value"] / prev["value"] - 1
               for prev, curr in zip(equity, equity[1:]) if prev["value"] > 0]
    n = len(returns)
    if n < 3:
        return None
    rf_period = (1 + RISK_FREE_RATE) ** (1 / periods_per_year) - 1
    excess = [r - rf_period for r in returns]
    vol = pstdev(excess)
    if vol <= 0:
        return None
    sr = mean(excess) / vol
    skew, kurt = moments(excess)
    se = sharpe_se(sr, n, skew, kurt)
    scale = math.sqrt(periods_per_year)
    return {
        "periods": n,
        "periods_per_year": round(periods_per_year, 4),
        "sharpe_period": round(sr, 6),
        "sharpe_annual": round(sr * scale, 3),
        "ci95_low": round((sr - WILSON_Z_95 * se) * scale, 3),
        "ci95_high": round((sr + WILSON_Z_95 * se) * scale, 3),
        "psr_vs_zero": round(probabilistic_sharpe(sr, 0.0, n, skew, kurt), 4),
        "skew": round(skew, 3),
        "kurtosis": round(kurt, 3),
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


def _bar_date(bar: dict) -> str:
    return bar["date"]


# Bars are always date-sorted ascending, so these lookups binary-search instead
# of scanning. The scan was ~25% of a universe backtest at 5 years and grows
# with history length — at 20 years it would dominate.
def _value_on_or_before(series: list[dict], date: str) -> tuple[int, float] | None:
    idx = bisect_right(series, date, key=_bar_date) - 1
    if idx < 0:
        return None
    return idx, series[idx]["close"]


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
    idx = bisect_left(series, day.isoformat(), key=_bar_date)
    if idx >= len(series):
        return None
    return idx, series[idx]["close"], series[idx]["date"]


def _latest_on_or_before(series: list[dict], day: date) -> tuple[int, float, str] | None:
    idx = bisect_right(series, day.isoformat(), key=_bar_date) - 1
    if idx < 0:
        return None
    return idx, series[idx]["close"], series[idx]["date"]


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
    # (reason, kind) of an exit decided at a close, to be filled at the NEXT
    # session's open. See DEFER_TO_NEXT_OPEN.
    pending_exit: tuple | None = None


# ── Fill convention (ML4T 3e §16.2: "same-bar execution ... fuses past and
# future") ────────────────────────────────────────────────────────────────
# Decisions are made from a bar's CLOSE; the fill happens LATER:
#   * Entries: a day limit order at the top of the ATR entry zone, working the
#     next session — exactly what the Weekly Plan tells you to place. Fills at
#     min(open, limit) if that session trades down to the limit; no fill if it
#     gaps above and never comes back (counted as a missed entry).
#   * Exits decided at a close (time stop, regime exit): next session's open.
#   * Stops / targets / partials are resting orders already working in the
#     market, so they keep filling intrabar (stop first, gaps at the open).
DEFER_TO_NEXT_OPEN = ("time", "regime")


def next_day_limit_fill(bar: dict, limit: float, stop: float) -> tuple[float | None, str]:
    """Fill a buy limit order placed after the prior close. Pure.

    Returns (price, "") on a fill, or (None, why) when it does not fill:
    "gap" — the session never traded down to the limit; "through_stop" — the
    fill would be at/below the planned stop (the setup was invalidated
    overnight; nobody enters a trade already past its stop)."""
    if bar["low"] > limit:
        return None, "gap"
    price = min(bar["open"], limit)
    if price <= stop:
        return None, "through_stop"
    return price, ""


def _next_open(series: list[dict], idx: int, until: str | None = None) -> float | None:
    """Open of the session after `idx` (optionally only if it is <= until)."""
    j = idx + 1
    if j < len(series) and (until is None or series[j]["date"] <= until):
        return series[j]["open"]
    return None


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
        if pos.pending_exit is not None:
            # Decided at yesterday's close; executed at today's open.
            reason, kind = pos.pending_exit
            fills.append({
                "date": bar["date"], "price": bar["open"], "shares": pos.state.shares,
                "reason": f"{reason} (filled next open)", "kind": kind, "closed": True,
            })
            pos.closed = True
            break
        atr = pos.atr[pos.idx] if pos.idx < len(pos.atr) else None
        view = _bar_view(bar, atr, pos.quadrants.get(bar["date"]))
        signal = resolve_exit(pos.rules, view, pos.state)
        if signal is not None and signal.fraction >= 1.0 and signal.kind in DEFER_TO_NEXT_OPEN:
            # Known only once this bar has closed: sell at the next open. Held
            # overnight, so a gap moves the fill — as it would live.
            pos.pending_exit = (signal.reason, signal.kind)
            advance(pos.rules, view, pos.state)
            continue
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


class _BarLoader:
    """Loads one ticker's normalized bars on demand, from the 5-year cache,
    the 20-year archive+cache splice, or a preloaded dict (arbitrary-era
    archive mode). A small LRU serves the simulation's re-reads of tickers it
    actually buys; the scan itself never caches."""

    def __init__(self, db: Session, history: str = "5y", preloaded: dict | None = None):
        self.db = db
        self.history = history
        self.preloaded = preloaded
        # (window_start, bars_to_keep_before_it) — set once the window is known.
        # Deterministic, so the scan and the simulation see identical series
        # and a stored bar index means the same bar in both.
        self.trim_before: tuple[str, int] | None = None
        self._lru: "OrderedDict[str, list[dict]]" = OrderedDict()

    def load_with_meta(self, ticker: str) -> tuple[list[dict], dict | None]:
        if self.preloaded is not None:
            return self.preloaded.get(ticker, []), None
        if self.history == "20y":
            from services.long_history import load_long_bars
            trim = self.trim_before if ticker != "SPY" else None
            # Trimming the ARCHIVE before the splice avoids rebasing 20 years
            # of bars to test a 5-year window (the dominant cost when profiled).
            raw, meta = load_long_bars(self.db, ticker, trim_before=trim)
            if trim:
                start, keep = trim
                cut = bisect_left(raw, start, key=_bar_date) - keep
                if cut > 0:
                    raw = raw[cut:]
            return _normalize_bars(raw), meta
        return _load_history(self.db, ticker), None

    def load(self, ticker: str) -> list[dict]:
        hit = self._lru.get(ticker)
        if hit is not None:
            self._lru.move_to_end(ticker)
            return hit
        bars, _ = self.load_with_meta(ticker)
        self._lru[ticker] = bars
        if len(self._lru) > _BAR_LRU_SIZE:
            self._lru.popitem(last=False)
        return bars


def _keep_top(lst: list, item, k: int, final: bool = False) -> None:
    """Append `item` and keep only the best `k` by rank score.

    Pruning is a STABLE sort (reverse=True keeps equal scores in insertion
    order), and tickers are appended in the same order the old all-in-memory
    loop visited them, so the kept top-k is exactly what sorting the full list
    would have produced — ties included."""
    if item is not None:
        lst.append(item)
    if final or len(lst) > 4 * k:
        lst.sort(key=lambda it: it[1]["score"], reverse=True)
        del lst[k:]


def _load_vix(db: Session) -> dict[str, float]:
    """date -> VIX close from the archive; {} when there is none (tests, or no
    backfill yet), which leaves the realized-volatility proxy in charge."""
    try:
        from services.long_history import load_vix_by_date
        return load_vix_by_date(db)
    except Exception:  # noqa: BLE001 — missing table/rows must not break a backtest
        return {}


def data_coverage(loaded: dict[str, list[dict]], used: dict[str, list[dict]],
                  spy: list[dict], warmup: int, rebalance_dates: list[str]) -> dict:
    """What price data the run actually had — so a thin or holed dataset is
    REPORTED instead of silently shaping the result. Pure; no DB.

    - `dropped`: tickers excluded because they had fewer than `warmup` bars
      (a broken/truncated cache looks exactly like this).
    - gaps: trading days SPY has but a ticker lacks, INSIDE that ticker's own
      date span. On a rebalance date a missing bar means the ticker was
      silently skipped for entry that period.
    - `ends_early`: tickers whose history stops > 5 SPY sessions before SPY's
      last bar (stale cache or delisted), so they vanish from the test.
    """
    acc = _Coverage(spy, warmup, rebalance_dates)
    for t, bars in loaded.items():
        acc.add(t, bars, t in used)
    return acc.result()


class _Coverage:
    """Streaming form of data_coverage(): fed one ticker at a time by the
    per-ticker scan, so the report never needs every history in memory."""

    def __init__(self, spy: list[dict], warmup: int, rebalance_dates: list[str]):
        self.spy_dates = [b["date"] for b in spy]
        self.spy_last = self.spy_dates[-1] if self.spy_dates else None
        self.rebal = set(rebalance_dates)
        self.rebalance_dates = rebalance_dates
        self.warmup = warmup
        self.requested = 0
        self.used = 0
        self.dropped: list[dict] = []
        self.missing_bars = 0
        self.missing_on_rebalance = 0
        self.gap_tickers: list[dict] = []
        self.ends_early: list[dict] = []
        self.spliced = 0
        self.splice_problems: list[dict] = []

    def add(self, ticker: str, bars: list[dict], used: bool, splice_meta: dict | None = None) -> None:
        self.requested += 1
        if splice_meta is not None:
            if splice_meta.get("used_archive_bars"):
                self.spliced += 1
            elif any(w in (splice_meta.get("reason") or "") for w in ("disagree", "cannot rebase")) \
                    and len(self.splice_problems) < 25:
                # Only genuine join failures. "Archive adds nothing" is normal:
                # young listings, and recent windows whose archive was trimmed.
                self.splice_problems.append({"ticker": ticker, "reason": splice_meta["reason"]})
        if not used:
            self.dropped.append({"ticker": ticker, "bars": len(bars)})
            return
        self.used += 1
        if not bars:
            return
        have = {b["date"] for b in bars}
        lo, hi = bars[0]["date"], bars[-1]["date"]
        span = self.spy_dates[bisect_left(self.spy_dates, lo):bisect_right(self.spy_dates, hi)]
        miss = [d for d in span if d not in have]
        if miss:
            self.missing_bars += len(miss)
            on_rebal = sum(1 for d in miss if d in self.rebal)
            self.missing_on_rebalance += on_rebal
            self.gap_tickers.append({"ticker": ticker, "missing": len(miss),
                                     "on_rebalance_dates": on_rebal, "first_missing": miss[0]})
        if self.spy_last and hi < self.spy_last:
            behind = len(self.spy_dates) - bisect_right(self.spy_dates, hi)
            if behind > 5:
                self.ends_early.append({"ticker": ticker, "last_bar": hi, "sessions_behind": behind})

    def result(self) -> dict:
        dropped = sorted(self.dropped, key=lambda d: (d["bars"], d["ticker"]))
        gap_tickers = sorted(self.gap_tickers, key=lambda g: (-g["missing"], g["ticker"]))
        ends_early = sorted(self.ends_early, key=lambda e: (-e["sessions_behind"], e["ticker"]))
        first = self.rebalance_dates[0] if self.rebalance_dates else None
        last = self.rebalance_dates[-1] if self.rebalance_dates else None
        years = None
        if first and last:
            years = round((date.fromisoformat(last) - date.fromisoformat(first)).days / 365.25, 1)
        return {
            "history_first_date": self.spy_dates[0] if self.spy_dates else None,
            "history_last_date": self.spy_last,
            "warmup_bars": self.warmup,
            "test_first_date": first,
            "test_last_date": last,
            "test_years": years,
            "tickers_requested": self.requested,
            "tickers_used": self.used,
            "dropped": dropped,
            "tickers_with_gaps": len(gap_tickers),
            "missing_bars": self.missing_bars,
            "missing_on_rebalance_dates": self.missing_on_rebalance,
            "gap_examples": gap_tickers[:10],
            "ends_early": ends_early[:25],
            "ends_early_count": len(ends_early),
            "tickers_spliced_with_archive": self.spliced,
            "splice_problems": self.splice_problems,
        }


def data_quality_caveats(dq: dict) -> list[dict]:
    out = []
    if dq.get("test_first_date"):
        out.append({
            "id": "history_window",
            "severity": "info",
            "title": f"Tested {dq['test_first_date']} → {dq.get('test_last_date')} (~{dq.get('test_years')} years)",
            "detail": (
                f"The price cache starts {dq.get('history_first_date')}; the first "
                f"{dq.get('warmup_bars')} bars are warm-up for the 200-day MA and the strategy's own "
                "lookbacks, so the test starts later than the data. A regime that barely occurred "
                "in this window cannot be judged from it, however the numbers look."
            ),
        })
    if dq.get("dropped"):
        names = ", ".join(f"{d['ticker']} ({d['bars']})" for d in dq["dropped"][:12])
        out.append({
            "id": "tickers_dropped",
            "severity": "medium",
            "title": f"{len(dq['dropped'])} ticker(s) excluded for too little history",
            "detail": (
                f"Fewer than {dq.get('warmup_bars')} cached bars: {names}"
                + (" …" if len(dq["dropped"]) > 12 else "")
                + ". Recently listed names are expected here; an established company with "
                "almost no bars means its cached history is broken and it was left out."
            ),
        })
    if dq.get("missing_bars"):
        out.append({
            "id": "missing_bars",
            "severity": "medium" if dq.get("missing_on_rebalance_dates") else "info",
            "title": f"{dq.get('tickers_with_gaps')} ticker(s) are missing {dq['missing_bars']} trading day(s)",
            "detail": (
                "Days SPY traded but the cached history has no bar. Nothing is filled in: "
                f"on {dq.get('missing_on_rebalance_dates', 0)} (ticker, rebalance date) pair(s) "
                "the ticker was skipped for entry because there was no real price that day."
            ),
        })
    if dq.get("ends_early_count"):
        out.append({
            "id": "history_ends_early",
            "severity": "medium",
            "title": f"{dq['ends_early_count']} ticker(s) stop before the end of the test",
            "detail": (
                "Their cached history ends more than 5 sessions before SPY's (stale cache or "
                "delisted), so they drop out of the later part of the test."
            ),
        })
    return out


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
    history: str = "5y",
    progress=None,
    max_window_years: float | None = MAX_WINDOW_YEARS,
) -> dict:
    """Walk-forward backtest. `history="20y"` draws on the 20-year archive
    spliced onto the cache (services/long_history.py). The tested window is
    capped at `max_window_years` (None = uncapped — the edge matrix, which
    runs in the job worker). `progress(fraction, detail)` is an optional
    heartbeat callback."""
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
    history = history if history in HISTORY_MODES else "5y"
    if history != "5y":
        # Only non-default depths appear in `parameters`, so a default run's
        # response stays byte-for-byte what it has always been.
        params["history"] = history

    def _tick(fraction: float, detail: str) -> None:
        if progress is None:
            return
        try:
            progress(fraction, detail)
        except Exception:  # noqa: BLE001 — progress must never sink a run
            pass

    tickers = _source_tickers(db, user_id, source)

    preloaded = None
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
        preloaded = load_archived_histories(db, tickers, buf_start, end_date)
        spy = ensure_archive(db, "SPY", buf_start, end_date)
    loader = _BarLoader(db, history, preloaded)
    if not archive:
        spy = loader.load_with_meta("SPY")[0]
    vix_by_date = _load_vix(db)

    dates = [b["date"] for b in spy] if len(spy) >= warmup else []
    dates = dates[warmup:: max(1, rebalance_days)]

    # Restrict to the requested window: a period preset (1Y/2Y/5Y/10Y/20Y/all)
    # and/or a custom start date. A custom start with a period bounds both ends
    # (start → start+period); a period alone uses the most recent window.
    full_dates = [b["date"] for b in spy]
    latest = full_dates[-1] if full_dates else None
    years = PERIOD_YEARS.get((period or "all").upper())

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
    if max_window_years:
        # Never test more than max_window_years: anchor at the start when one
        # is given, else at the end (or the latest bar).
        cap = int(max_window_years)
        if win_start:
            limit = _shift_year(win_start, cap)
            if not win_end or win_end > limit:
                win_end = limit
        else:
            anchor = win_end or latest
            if anchor:
                win_start = _shift_year(anchor, -cap)
    if win_start:
        dates = [d for d in dates if d >= win_start]
    if win_end:
        dates = [d for d in dates if d <= win_end]
    if win_start and history == "20y" and preloaded is None:
        # Load each ticker only from shortly before the window: warm-up plus
        # settling bars. Twenty years of bars for a five-year window would
        # quadruple the parse cost for nothing.
        loader.trim_before = (win_start, warmup + _PRE_WINDOW_EXTRA_BARS)

    have_window = len(dates) >= 4
    if have_window:
        spy_closes = [b["close"] for b in spy]
        spy_ma = calc_ma(spy_closes, regime_ma)
        spy_ma200 = calc_ma(spy_closes, 200)
        adx_full = calc_adx(
            [b["high"] for b in spy], [b["low"] for b in spy], spy_closes, 14
        )["series"]
        # Rolling 20-day realized volatility — the VIX proxy classify_regime()
        # falls back to, used on any date without a real VIX close.
        vol20 = [None] * len(spy_closes)
        for i in range(20, len(spy_closes)):
            window = spy_closes[i - 20: i + 1]
            rets = [(window[j] / window[j - 1] - 1) for j in range(1, len(window))]
            if rets:
                vol20[i] = (sum(r * r for r in rets) / len(rets)) ** 0.5 * (252 ** 0.5) * 100

        # The SPY>MA entry gate depends only on SPY, so it is known per date
        # before any ticker is scanned — and no candidate is computed on a
        # date where nothing could be bought anyway.
        regime_ok_at: dict[str, bool] = {}
        for d in dates:
            p = _value_on_or_before(spy, d)
            regime_ok_at[d] = bool(p) and (
                not spy_regime or (spy_ma[p[0]] is not None and p[1] > spy_ma[p[0]]))

    def _crisis_at(idx: int) -> bool:
        """Real VIX (live rule: VIX >= VIX_CRISIS) when the archive has that
        date, else the realized-volatility proxy — never look-ahead."""
        v = vix_by_date.get(spy[idx]["date"]) if vix_by_date else None
        if v is not None:
            return v >= VIX_CRISIS
        return vol20[idx] is not None and vol20[idx] >= 22.0

    # ── SCAN: one ticker in memory at a time ─────────────────────────────
    # strategy.candidate(bars, idx) reads only that ticker's bars[:idx+1], so
    # every candidate can be computed ticker-by-ticker up front with results
    # identical to computing them date-by-date — without holding the whole
    # universe (≈1.25 GB at 20 years) in memory. Per date only the top
    # `k_keep` are kept: rotation takes the top k_sel; trade_plan takes the
    # top k_sel after skipping names already held (at most max_positions).
    k_sel = max(1, min(top_n, 20))
    k_keep = k_sel if mode != "trade_plan" else k_sel + int(max_positions or 0)
    cands: dict[str, list] = {d: [] for d in dates}
    coverage = _Coverage(spy, warmup, dates)
    used_tickers: list[str] = []
    scan_list = list(preloaded.keys()) if preloaded is not None else tickers
    pairs = list(zip(dates, dates[1:]))
    for n, ticker in enumerate(scan_list):
        if n % 10 == 0:
            _tick(0.03 + 0.85 * n / max(1, len(scan_list)),
                  f"Scanning {ticker} ({n + 1}/{len(scan_list)})")
        bars, splice_meta = loader.load_with_meta(ticker)
        usable = len(bars) >= warmup
        coverage.add(ticker, bars, usable, splice_meta)
        if not usable:
            continue
        used_tickers.append(ticker)
        if not have_window or not strategy.applies_to(ticker):
            continue
        if mode == "trade_plan":
            for start in dates:
                if not regime_ok_at.get(start):
                    continue
                point = _value_on_or_before(bars, start)
                # Require a bar ON the rebalance date: filling at a stale
                # close from an earlier session would be a fabricated price.
                if not point or bars[point[0]]["date"] != start:
                    continue
                rank = strategy.candidate(bars, point[0])
                if rank:
                    _keep_top(cands[start], (ticker, rank, point[0]), k_keep)
        else:
            for start, end in pairs:
                if not regime_ok_at.get(start):
                    continue
                start_point = _value_on_or_before(bars, start)
                end_point = _value_on_or_before(bars, end)
                if not start_point or not end_point or end_point[0] <= start_point[0]:
                    continue
                rank = strategy.candidate(bars, start_point[0])
                if rank:
                    # Traded at the open of the session AFTER each decision
                    # date (never at the close the decision was made from).
                    # The final period has no next session: marked at its close.
                    buy_px = _next_open(bars, start_point[0], until=end)
                    if buy_px is None:
                        continue
                    sell_px = _next_open(bars, end_point[0]) or end_point[1]
                    _keep_top(cands[start], (ticker, rank, buy_px, sell_px), k_keep)
    for lst in cands.values():
        _keep_top(lst, None, k_keep, final=True)
    _tick(0.9, "Simulating…")

    if not have_window or not used_tickers:
        return {
            "strategy": strategy_meta,
            "source": source,
            "parameters": params,
            "available_tickers": sorted(used_tickers),
            "equity": [],
            "benchmark": [],
            "trades": [],
            "metrics": {},
            "benchmark_metrics": {},
            "data_quality": coverage.result(),
            "notes": [
                f"Need cached SPY history and at least {warmup} bars for selected tickers "
                f"(SPY has {len(spy)}; {len(used_tickers)} of {coverage.requested} tickers qualify).",
                "Tip: use the Full Universe source after the 5-year history backfill completes.",
            ],
        }

    equity = [{"date": dates[0], "value": 1.0}]
    benchmark = [{"date": dates[0], "value": 1.0}]
    trades = []
    prev_holdings: set[str] = set()
    cost = cost_bps / 10000

    def _quadrant_at(idx: int) -> tuple[str, float | None]:
        """Regime quadrant + ADX at a SPY bar index (no look-ahead: index only)."""
        adx_v = adx_full[idx]["adx"] if adx_full[idx] else None
        above = spy_ma200[idx] is not None and spy_closes[idx] > spy_ma200[idx]
        return classify_quadrant(trend_strength_label(adx_v), above, _crisis_at(idx)), adx_v

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
                # Once per held ticker — never per bar. Dropped when the
                # position closes so memory tracks the open book only.
                series = calc_atr([b["high"] for b in bars], [b["low"] for b in bars],
                                  [b["close"] for b in bars], ATR_PERIOD)
                atr_cache[ticker] = series
            return series

        cash = float(account_size)
        open_positions: dict[str, OpenPosition] = {}
        skipped_full_book = 0
        skipped_no_cash = 0
        skipped_no_plan = 0
        missed_gap = 0            # next session never traded down to the limit
        missed_through_stop = 0   # would have filled at/below the planned stop
        missed_no_session = 0     # no next session inside the test window

        for i, start in enumerate(dates):
            spy_start = _value_on_or_before(spy, start)
            if not spy_start:
                continue
            spy_idx, spy_start_px = spy_start
            regime_ok = not spy_regime or (spy_ma[spy_idx] is not None and spy_start_px > spy_ma[spy_idx])
            quadrant, adx_val = _quadrant_at(spy_idx)

            # ── 1. ORDERS decided at the close of `start`.
            # The decision uses strategy.candidate(bars, idx), which only reads
            # bars[:idx+1]. Nothing is FILLED here: each pick becomes a day
            # limit order for the next session (see DEFER_TO_NEXT_OPEN notes).
            equity_now = cash + sum(
                p.state.shares * p.bars[p.idx]["close"] for p in open_positions.values()
            )
            pending: list[tuple] = []
            if regime_ok and len(open_positions) < max_positions:
                ranked = [c for c in cands.get(start, ()) if c[0] not in open_positions]

                for ticker, rank, idx in ranked[:k_sel]:
                    if len(open_positions) + len(pending) >= max_positions:
                        skipped_full_book += 1
                        continue
                    bars = loader.load(ticker)
                    plan = build_trade_plan(
                        bars[: idx + 1], equity_now, risk_pct,
                        bars[idx]["close"], atr_stop_mult, r_multiple,
                    )
                    if not plan or not plan.get("shares"):
                        skipped_no_plan += 1
                        continue
                    pending.append((ticker, idx, plan, bars))

            end = dates[i + 1] if i + 1 < len(dates) else None
            if end is None:
                # Orders placed on the last decision date have no next session
                # inside the test; they never fill.
                missed_no_session += len(pending)
                break
            spy_end = _value_on_or_before(spy, end)
            if not spy_end:
                continue
            _, spy_end_px = spy_end

            exits_this_period = 0
            # ── 2a. FILL yesterday's orders in the next session.
            for ticker, idx, plan, bars in pending:
                fidx = idx + 1
                if fidx >= len(bars) or bars[fidx]["date"] > end:
                    missed_no_session += 1
                    continue
                bar = bars[fidx]
                stop_px = float(plan["stop"])
                entry_px, why = next_day_limit_fill(bar, float(plan["entry_zone"][1]), stop_px)
                if entry_px is None:
                    if why == "gap":
                        missed_gap += 1
                    else:
                        missed_through_stop += 1
                    continue
                risk_ps = entry_px - stop_px
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
                    ticker=ticker, entry_date=bar["date"], entry_price=entry_px,
                    entry_idx=fidx, shares=shares, initial_shares=shares,
                    stop=stop_px, initial_stop=stop_px, risk_per_share=risk_ps,
                    target=float(plan["target"]), strategy=strategy.id,
                )
                pos = OpenPosition(
                    ticker=ticker, bars=bars, atr=atr_series,
                    quadrants=quadrants_by_date, state=state, rules=rules,
                    idx=fidx, entry_value=outlay,
                )
                if bar["low"] <= stop_px:
                    # The rest of the fill session traded through the stop. The
                    # intrabar order is unknown, so assume the worst: stopped
                    # out the same day, and no same-day target credit.
                    fill = {"date": bar["date"], "price": stop_px, "shares": shares,
                            "reason": "stop hit on the entry day", "kind": "stop", "closed": True}
                    pos.fills.append(fill)
                    pos.closed = True
                    proceeds = shares * stop_px * (1 - cost)
                    cash += proceeds
                    pos.proceeds += proceeds
                    trade_log.append(_trade_record(pos))
                    exits_this_period += 1
                    continue
                # Seed path-dependent rule state (highest close, trail level)
                # from the ENTRY bar, so the level checked tomorrow was
                # derived from data available today.
                advance(rules, _bar_view(bar, atr_series[fidx] if fidx < len(atr_series) else None,
                                         quadrants_by_date.get(bar["date"])), state)
                open_positions[ticker] = pos

            # ── 2b. Walk every open position bar-by-bar through `end`.
            for ticker, pos in list(open_positions.items()):
                for fill in walk_position(pos, end):
                    proceeds = fill["shares"] * fill["price"] * (1 - cost)
                    cash += proceeds
                    pos.proceeds += proceeds
                if pos.closed:
                    trade_log.append(_trade_record(pos))
                    open_positions.pop(ticker, None)
                    atr_cache.pop(ticker, None)
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
            # Orders placed but never filled under the next-session limit rule.
            "entries_missed_gap": missed_gap,
            "entries_missed_through_stop": missed_through_stop,
            "entries_missed_no_session": missed_no_session,
            "avg_bars_held": round(mean([t["bars_held"] for t in trade_log]), 1) if trade_log else None,
            "ending_cash_pct": round(cash / (equity[-1]["value"] * account_size) * 100, 1)
                               if equity and equity[-1]["value"] else None,
        }
        rebalance_windows: list = []
    else:
        rebalance_windows = pairs

    for start, end in rebalance_windows:
        spy_start = _value_on_or_before(spy, start)
        spy_end = _value_on_or_before(spy, end)
        if not spy_start or not spy_end:
            continue
        spy_idx, spy_start_px = spy_start
        spy_end_idx, spy_end_px = spy_end
        # Decisions read the close; the SPY GATE is a decision, so it keeps the
        # close. Returns (strategy and benchmark alike) run open-to-open.
        regime_ok = not spy_regime or (spy_ma[spy_idx] is not None and spy_start_px > spy_ma[spy_idx])
        spy_buy = _next_open(spy, spy_idx, until=end) or spy_start_px
        spy_sell = _next_open(spy, spy_end_idx) or spy_end_px

        adx_val = adx_full[spy_idx]["adx"] if adx_full[spy_idx] else None
        above_200_here = spy_ma200[spy_idx] is not None and spy_start_px > spy_ma200[spy_idx]
        quadrant = classify_quadrant(trend_strength_label(adx_val), above_200_here, _crisis_at(spy_idx))

        ranked = cands.get(start, []) if regime_ok else []
        selected = ranked[:k_sel]
        holdings = {t for t, *_ in selected}

        period_ret = 0.0
        if selected:
            period_ret = mean((end_px / start_px - 1) for _, _, start_px, end_px in selected)
        # Round-trip turnover cost, uncapped — see turnover_cost() above.
        period_ret -= turnover_cost(holdings, prev_holdings, top_n, cost)
        spy_ret = (spy_sell / spy_buy - 1) if spy_buy else 0.0

        equity.append({"date": end, "value": round(equity[-1]["value"] * (1 + period_ret), 6)})
        benchmark.append({"date": end, "value": round(benchmark[-1]["value"] * (spy_sell / spy_buy), 6)})
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
    caveats.append({
        "id": "fill_convention",
        "severity": "info",
        "title": "Trades fill in the session AFTER the decision",
        "detail": (
            "Signals are computed from a day's close; nothing is bought or sold at that same "
            "close. "
            + ("Rotation holdings are bought and sold at the next session's open (the SPY "
               "benchmark uses the same open-to-open convention)."
               if mode == "rotation" else
               "Entries are day limit orders at the top of the ATR entry zone for the next "
               "session — filled at the lower of the open and the limit, and not filled at all "
               "if the stock gaps above it. Time-stop and regime exits sell at the next open; "
               "stops and targets are resting orders and fill intrabar.")
        ),
    })
    if mode == "trade_plan" and metrics.get("entries_missed_gap"):
        missed = metrics["entries_missed_gap"] + metrics.get("entries_missed_through_stop", 0)
        caveats.append({
            "id": "entries_missed",
            "severity": "info",
            "title": f"{missed} order(s) never filled",
            "detail": (
                f"{metrics['entries_missed_gap']} gapped above the limit and never traded back "
                f"down to it; {metrics.get('entries_missed_through_stop', 0)} would have filled at "
                "or below the planned stop. A backtest that fills every signal at the signal "
                "close counts these as trades — they are the optimism this convention removes."
            ),
        })
    data_quality = coverage.result()
    caveats.extend(data_quality_caveats(data_quality))
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
        "available_tickers": sorted(used_tickers),
        "equity": equity,
        "benchmark": benchmark,
        "trades": trades,
        "metrics": metrics,
        "benchmark_metrics": benchmark_metrics,
        "regime_breakdown": regime_breakdown,
        "strategy_regimes": strategy.regimes,
        "mode": mode,
        "sharpe_inference": sharpe_inference(equity, ppy),
        "trade_log": trade_log,
        "caveats": caveats,
        "data_quality": data_quality,
        "notes": [
            "Hypothetical backtest using cached adjusted daily closes only.",
            f"Strategy: {strategy.name}. {strategy.description}",
            "Modeled as an equal-weight top-N rotation rebalanced on the chosen cadence "
            "with turnover cost and an optional SPY regime filter, traded at the open of the "
            "session after each rebalance decision. ATR stops/targets are NOT simulated "
            "intrabar — live trade plans add those on top."
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
