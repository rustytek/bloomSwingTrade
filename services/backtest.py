import json
import math
from datetime import date, datetime, timedelta, timezone
from statistics import mean, pstdev

import numpy as np
from sqlalchemy.orm import Session

from database.models import PortfolioPosition, StockCache, WatchlistItem, WatchlistSnapshot
from services.indicators import calc_ma, calc_rsi


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
    return [
        {"date": b.get("date"), "close": _safe_float(b.get("close")), "vol": _safe_float(b.get("vol"))}
        for b in bars
        if b.get("date") and _safe_float(b.get("close"))
    ]


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


def _metrics(equity: list[dict]) -> dict:
    if len(equity) < 2:
        return {"total_return": 0, "cagr": 0, "max_drawdown": 0, "sharpe": None, "win_rate": None}
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
    sharpe = (mean(returns) / vol * math.sqrt(252 / 5)) if vol > 0 else None
    wins = [r for r in returns if r > 0]
    return {
        "total_return": round(total_return * 100, 2),
        "cagr": round(cagr * 100, 2),
        "max_drawdown": round(_max_drawdown(equity), 2),
        "sharpe": round(sharpe, 2) if sharpe is not None else None,
        "win_rate": round(len(wins) / len(returns) * 100, 1) if returns else None,
    }


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
    closes = [b["close"] for b in series[: idx + 1]]
    if len(closes) < 70:
        return None
    ret_21 = closes[-1] / closes[-22] - 1
    ret_63 = closes[-1] / closes[-64] - 1
    daily = np.diff(np.array(closes[-64:], dtype=float)) / np.array(closes[-64:-1], dtype=float)
    vol_63 = float(np.std(daily) * math.sqrt(252)) if len(daily) else 0
    ma50 = calc_ma(closes, 50)[-1]
    rsi = calc_rsi(closes, 14)[-1]
    if ma50 is None or closes[-1] <= ma50 or ret_21 <= 0:
        return None
    if rsi is not None and not (40 <= rsi <= 72):
        return None
    score = (ret_21 * 0.45) + (ret_63 * 0.55) - (vol_63 * 0.15)
    return {
        "score": score,
        "ret_21": ret_21,
        "ret_63": ret_63,
        "vol_63": vol_63,
        "rsi": rsi,
    }


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
            "rsi": round(rank["rsi"], 1) if rank and rank["rsi"] is not None else None,
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
            "ret_21": round(rank["ret_21"] * 100, 2),
            "ret_63": round(rank["ret_63"] * 100, 2),
            "rsi": round(rank["rsi"], 1) if rank["rsi"] is not None else None,
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


def run_walk_forward_backtest(
    db: Session,
    user_id: int,
    source: str = "watchlist",
    top_n: int = 5,
    rebalance_days: int = 5,
    cost_bps: float = 10,
    spy_regime: bool = True,
) -> dict:
    tickers = _source_tickers(db, user_id, source)
    histories = {ticker: _load_history(db, ticker) for ticker in tickers}
    histories = {ticker: bars for ticker, bars in histories.items() if len(bars) >= 75}
    spy = _load_history(db, "SPY")
    dates = [b["date"] for b in spy] if len(spy) >= 75 else []
    dates = dates[70:: max(1, rebalance_days)]
    if len(dates) < 4 or not histories:
        return {
            "strategy": "Regime Relative Strength",
            "source": source,
            "parameters": {"top_n": top_n, "rebalance_days": rebalance_days, "cost_bps": cost_bps, "spy_regime": spy_regime},
            "available_tickers": sorted(histories),
            "equity": [],
            "benchmark": [],
            "trades": [],
            "metrics": {},
            "benchmark_metrics": {},
            "notes": ["Need cached SPY history and at least 75 bars for selected tickers."],
        }

    equity = [{"date": dates[0], "value": 1.0}]
    benchmark = [{"date": dates[0], "value": 1.0}]
    trades = []
    prev_holdings: set[str] = set()
    cost = cost_bps / 10000

    spy_closes = [b["close"] for b in spy]
    spy_ma = calc_ma(spy_closes, 50)

    for start, end in zip(dates, dates[1:]):
        spy_start = _value_on_or_before(spy, start)
        spy_end = _value_on_or_before(spy, end)
        if not spy_start or not spy_end:
            continue
        spy_idx, spy_start_px = spy_start
        _, spy_end_px = spy_end
        regime_ok = not spy_regime or (spy_ma[spy_idx] is not None and spy_start_px > spy_ma[spy_idx])

        ranked = []
        if regime_ok:
            for ticker, bars in histories.items():
                start_point = _value_on_or_before(bars, start)
                end_point = _value_on_or_before(bars, end)
                if not start_point or not end_point or end_point[0] <= start_point[0]:
                    continue
                rank = _rank_candidate(bars, start_point[0])
                if rank:
                    ranked.append((ticker, rank, start_point[1], end_point[1]))
        ranked.sort(key=lambda item: item[1]["score"], reverse=True)
        selected = ranked[: max(1, min(top_n, 20))]
        holdings = {t for t, *_ in selected}

        period_ret = 0.0
        if selected:
            period_ret = mean((end_px / start_px - 1) for _, _, start_px, end_px in selected)
        turnover = len(holdings.symmetric_difference(prev_holdings)) / max(top_n, 1)
        period_ret -= min(1.0, turnover) * cost

        equity.append({"date": end, "value": round(equity[-1]["value"] * (1 + period_ret), 6)})
        benchmark.append({"date": end, "value": round(benchmark[-1]["value"] * (spy_end_px / spy_start_px), 6)})
        trades.append({
            "date": start,
            "holdings": [
                {
                    "ticker": ticker,
                    "score": round(rank["score"], 4),
                    "ret_21": round(rank["ret_21"] * 100, 2),
                    "ret_63": round(rank["ret_63"] * 100, 2),
                    "rsi": round(rank["rsi"], 1) if rank["rsi"] is not None else None,
                }
                for ticker, rank, _, _ in selected
            ],
            "regime": "risk_on" if regime_ok else "cash",
            "period_return": round(period_ret * 100, 2),
        })
        prev_holdings = holdings

    return {
        "strategy": "Regime Relative Strength",
        "source": source,
        "parameters": {"top_n": top_n, "rebalance_days": rebalance_days, "cost_bps": cost_bps, "spy_regime": spy_regime},
        "available_tickers": sorted(histories),
        "equity": equity,
        "benchmark": benchmark,
        "trades": trades,
        "metrics": _metrics(equity),
        "benchmark_metrics": _metrics(benchmark),
        "notes": [
            "Hypothetical backtest using cached adjusted daily closes only.",
            "Rules use trailing momentum, volatility, moving-average trend, RSI filter, transaction costs, and optional SPY regime filter.",
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

    exits = []
    for pos in positions:
        q = quotes.get(pos.ticker, {})
        reasons = []
        if (q.get("vs_ma200") or 0) < 0:
            reasons.append("below 200MA")
        if (q.get("ann_ret") or 0) < 10:
            reasons.append("weak 1M annualized return")
        if q.get("sharpe") is not None and q.get("sharpe") < 0.5:
            reasons.append("low Sharpe")
        if q.get("max_dd_1m") is not None and q.get("max_dd_1m") > 8:
            reasons.append("drawdown pressure")
        if reasons:
            exits.append({
                "ticker": pos.ticker,
                "reasons": reasons,
                "price": q.get("price"),
                "rsi": q.get("rsi"),
                "ann_ret": q.get("ann_ret"),
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
        if q.get("gc"):
            changes.append({"ticker": ticker, "type": "trend", "message": "Golden cross / MA50 above MA200"})
        if q.get("dc"):
            changes.append({"ticker": ticker, "type": "trend", "message": "Death cross / MA50 below MA200"})
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
