"""
Technical indicator calculations.
Pure Python / numpy — mirrors the logic from the HTML template.
"""
from __future__ import annotations

import math
import numpy as np
from typing import Optional


def _safe(v):
    """Return None if v is NaN, Inf, or not a number — ensures JSON safety."""
    if v is None:
        return None
    try:
        if math.isnan(v) or math.isinf(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def calc_ma(closes: list[float], period: int) -> list[Optional[float]]:
    arr = np.array(closes, dtype=float)
    result = [None] * len(arr)
    for i in range(period - 1, len(arr)):
        result[i] = float(np.mean(arr[i - period + 1 : i + 1]))
    return result


def calc_ema(closes: list[float], period: int) -> list[Optional[float]]:
    arr = np.array(closes, dtype=float)
    result = [None] * len(arr)
    if len(arr) < period:
        return result
    k = 2.0 / (period + 1)
    sma = float(np.mean(arr[:period]))
    result[period - 1] = sma
    for i in range(period, len(arr)):
        result[i] = (arr[i] - result[i - 1]) * k + result[i - 1]
    return result


def calc_rsi(closes: list[float], period: int = 14) -> list[Optional[float]]:
    arr = np.array(closes, dtype=float)
    result = [None] * len(arr)
    if len(arr) < period + 1:
        return result
    deltas = np.diff(arr)
    avg_gain = float(np.mean(np.where(deltas[:period] > 0, deltas[:period], 0)))
    avg_loss = float(np.mean(np.where(deltas[:period] < 0, -deltas[:period], 0)))
    result[period] = 100 - 100 / (1 + (1e9 if avg_loss == 0 else avg_gain / avg_loss))
    for i in range(period, len(deltas)):
        g = max(deltas[i], 0)
        l = max(-deltas[i], 0)
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
        result[i + 1] = 100 - 100 / (1 + (1e9 if avg_loss == 0 else avg_gain / avg_loss))
    return result


def calc_macd(closes: list[float]) -> dict:
    ema12 = calc_ema(closes, 12)
    ema26 = calc_ema(closes, 26)
    macd_line = [
        (e12 - e26) if e12 is not None and e26 is not None else None
        for e12, e26 in zip(ema12, ema26)
    ]
    valid_macd = [v for v in macd_line if v is not None]
    raw_signal = calc_ema(valid_macd, 9)
    signal_line = [None] * len(closes)
    si = 0
    for i, v in enumerate(macd_line):
        if v is not None and si < len(raw_signal):
            signal_line[i] = raw_signal[si]
            si += 1
    histogram = [
        (m - s) if m is not None and s is not None else None
        for m, s in zip(macd_line, signal_line)
    ]
    return {"macd_line": macd_line, "signal": signal_line, "histogram": histogram}


def calc_bollinger(closes: list[float], period: int = 20, mult: float = 2.0) -> list[Optional[dict]]:
    arr = np.array(closes, dtype=float)
    result = [None] * len(arr)
    for i in range(period - 1, len(arr)):
        window = arr[i - period + 1 : i + 1]
        mean = float(np.mean(window))
        std = float(np.std(window))
        result[i] = {"upper": mean + mult * std, "middle": mean, "lower": mean - mult * std}
    return result


def macd_signal_label(macd_line: list, signal_line: list) -> str:
    """Returns 'bullish', 'bearish', or 'neutral' based on recent MACD crossover."""
    valid = [(m, s) for m, s in zip(macd_line, signal_line) if m is not None and s is not None]
    if len(valid) < 3:
        return "neutral"
    recent = valid[-5:]
    last_m, last_s = recent[-1]
    prev_m, prev_s = recent[-2]
    if last_m > last_s and prev_m <= prev_s:
        return "bullish"
    if last_m < last_s and prev_m >= prev_s:
        return "bearish"
    if last_m > last_s:
        return "bullish"
    if last_m < last_s:
        return "bearish"
    return "neutral"


def compute_score(stock: dict) -> dict:
    """
    Mirrors the computeScore() function from the HTML template.
    Returns {f, t, m, o} each 0-5.
    """
    pe = stock.get("pe")
    pm = stock.get("pm") or 0
    fwd_pe = stock.get("fwd_pe")
    fcf_pos = stock.get("fcf_pos", False)
    debt_eq = stock.get("debt_eq")
    rsi = stock.get("rsi") or 0
    vs_ma200 = stock.get("vs_ma200") or 0
    gc = stock.get("gc", False)
    dc = stock.get("dc", False)
    chg_pct = stock.get("chg_pct") or 0
    vol_r = stock.get("vol_r") or 0
    p52w = stock.get("p52w") or 0
    earn_beat = stock.get("earn_beat", False)

    # Fundamental
    f = 0
    if pe and pe < 20: f += 1
    if pe and pe < 15: f += 1
    if pm > 20: f += 1
    if fcf_pos: f += 1
    if debt_eq is not None and debt_eq < 0.5: f += 1

    # Technical
    t = 0
    if 40 <= rsi <= 65: t += 1
    if vs_ma200 > 0: t += 1
    if vs_ma200 > 10: t += 1
    if gc: t += 1
    if not dc: t += 1

    # Momentum
    m = 0
    if chg_pct > 0: m += 1
    if chg_pct > 1: m += 1
    if vol_r > 1.2: m += 1
    if p52w > 70: m += 1
    if earn_beat: m += 1

    f = min(5, f)
    t = min(5, t)
    m = min(5, m)
    o = round((f + t + m) / 3)
    return {"f": f, "t": t, "m": m, "o": o}


def compute_swing_score(stock: dict) -> dict:
    """Return an explainable 0-100 swing-trade setup score.

    Component weights:
    trend 25, momentum/relative strength 25, risk 20,
    liquidity/volume 15, fundamentals 15.
    """
    reasons: list[tuple[int, str]] = []

    def num(key: str, default: float | None = None) -> float | None:
        value = stock.get(key)
        return value if isinstance(value, (int, float)) and not math.isnan(value) else default

    def add_reason(points: int, text: str) -> None:
        if points > 0:
            reasons.append((points, text))

    # Trend: is price in a useful uptrend without being purely a one-day pop?
    trend = 0
    vs_ma50 = num("vs_ma50", 0) or 0
    vs_ma200 = num("vs_ma200", 0) or 0
    rsi = num("rsi")
    if vs_ma200 > 0:
        trend += 8
        add_reason(8, f"above 200-day MA ({vs_ma200:.1f}%)")
    if vs_ma50 > 0:
        trend += 5
        add_reason(5, f"above 50-day MA ({vs_ma50:.1f}%)")
    if vs_ma200 > 10:
        trend += 4
        add_reason(4, "strong long-term trend")
    if rsi is not None and 45 <= rsi <= 65:
        trend += 4
        add_reason(4, f"RSI in swing zone ({rsi:.1f})")
    elif rsi is not None and 40 <= rsi <= 70:
        trend += 2
    if stock.get("macd_sig") == "bullish":
        trend += 3
        add_reason(3, "bullish MACD")
    if not stock.get("dc"):
        trend += 1
    trend = min(25, trend)

    # Momentum / relative strength: does it have multi-week strength and alpha?
    momentum = 0
    ret_5d = num("ret_5d")
    ret_21d = num("ret_21d")
    ret_63d = num("ret_63d")
    rel_spy_21d = num("rel_spy_21d")
    rel_spy_63d = num("rel_spy_63d")
    p52w = num("p52w")
    if ret_21d is not None and ret_21d > 0:
        pts = 6 + (4 if ret_21d >= 5 else 0)
        momentum += pts
        add_reason(pts, f"21D momentum {ret_21d:+.1f}%")
    if ret_5d is not None and ret_5d > 0:
        momentum += 3
        add_reason(3, f"5D momentum {ret_5d:+.1f}%")
    if ret_63d is not None and ret_63d > 0:
        momentum += 4
        add_reason(4, f"63D trend {ret_63d:+.1f}%")
    if rel_spy_21d is not None and rel_spy_21d > 0:
        pts = 5 + (2 if rel_spy_21d >= 3 else 0)
        momentum += pts
        add_reason(pts, f"beating SPY 21D by {rel_spy_21d:+.1f}%")
    elif rel_spy_63d is not None and rel_spy_63d > 0:
        momentum += 3
        add_reason(3, f"beating SPY 63D by {rel_spy_63d:+.1f}%")
    if p52w is not None and 60 <= p52w <= 92:
        momentum += 4
        add_reason(4, f"healthy 52-week position ({p52w:.0f}%)")
    elif p52w is not None and p52w > 92:
        momentum += 2
    momentum = min(25, momentum)

    # Risk: prefer names that move cleanly and recover from dips.
    risk = 0
    sortino = num("sortino")
    sharpe = num("sharpe")
    calmar = num("calmar")
    max_dd_1m = num("max_dd_1m")
    vol_1m = num("vol_1m")
    if sortino is not None and sortino > 1:
        pts = 5 + (3 if sortino > 2 else 0)
        risk += pts
        add_reason(pts, f"strong Sortino ({sortino:.2f})")
    if max_dd_1m is not None:
        if max_dd_1m <= 5:
            risk += 5
            add_reason(5, f"shallow 1M drawdown ({max_dd_1m:.1f}%)")
        elif max_dd_1m <= 10:
            risk += 3
    if vol_1m is not None:
        if vol_1m <= 30:
            risk += 4
            add_reason(4, f"manageable 1M volatility ({vol_1m:.1f}%)")
        elif vol_1m <= 45:
            risk += 2
    if sharpe is not None and sharpe > 1:
        risk += 3
    if calmar is not None and calmar > 1:
        risk += 3
    risk = min(20, risk)

    # Liquidity / volume: can a normal user enter/exit without chasing.
    liquidity = 0
    avg_dollar_vol_m = num("avg_dollar_vol_m")
    vol_r = num("vol_r", 1) or 1
    mkt_cap = num("mkt_cap")
    if avg_dollar_vol_m is not None:
        if avg_dollar_vol_m >= 50:
            liquidity += 6
            add_reason(6, f"liquid trading (${avg_dollar_vol_m:.0f}M avg)")
        elif avg_dollar_vol_m >= 10:
            liquidity += 4
    if 1.1 <= vol_r <= 2.5:
        liquidity += 5
        add_reason(5, f"volume confirmation ({vol_r:.2f}x)")
    elif vol_r > 2.5:
        liquidity += 2
    if mkt_cap is not None:
        if mkt_cap >= 10:
            liquidity += 4
        elif mkt_cap >= 2:
            liquidity += 2
    liquidity = min(15, liquidity)

    # Fundamentals: enough business quality to reduce single-chart risk.
    fundamentals = 0
    pe = num("pe")
    fwd_pe = num("fwd_pe")
    pm = num("pm")
    eps_grw = num("eps_grw")
    rev_grw = num("rev_grw")
    debt_eq = num("debt_eq")
    if stock.get("fcf_pos"):
        fundamentals += 3
        add_reason(3, "positive free cash flow")
    if pm is not None and pm >= 15:
        fundamentals += 3
        add_reason(3, f"healthy margin ({pm:.1f}%)")
    if rev_grw is not None and rev_grw >= 10:
        fundamentals += 3
        add_reason(3, f"revenue growth {rev_grw:.1f}%")
    if eps_grw is not None and eps_grw >= 10:
        fundamentals += 3
        add_reason(3, f"EPS growth {eps_grw:.1f}%")
    if debt_eq is not None and debt_eq <= 1:
        fundamentals += 2
    valuation = fwd_pe if fwd_pe is not None else pe
    if valuation is not None and 0 < valuation <= 30:
        fundamentals += 2
    fundamentals = min(15, fundamentals)

    components = {
        "trend": trend,
        "momentum": momentum,
        "risk": risk,
        "liquidity": liquidity,
        "fundamentals": fundamentals,
    }
    total = int(round(sum(components.values())))
    if total >= 80:
        grade = "A"
    elif total >= 65:
        grade = "B"
    elif total >= 50:
        grade = "C"
    elif total >= 35:
        grade = "D"
    else:
        grade = "F"

    top_reasons = [text for _, text in sorted(reasons, key=lambda item: item[0], reverse=True)[:5]]
    return {
        "score": max(0, min(100, total)),
        "grade": grade,
        "components": components,
        "reasons": top_reasons,
    }


def compute_performance_metrics(
    closes: list[float],
    spy_closes: list[float] | None = None,
    beta: float | None = None,
) -> dict:
    """Annual return, volatility, Sharpe, Sortino, Calmar, Info Ratio, Treynor.

    Args:
        closes:    full price history for the stock
        spy_closes: SPY price history (same length requested); used for Info Ratio
        beta:      pre-computed beta from yfinance; used for Treynor
    """
    _empty = {
        "ann_ret": None, "vol": None, "sharpe": None,
        "gain_sharpe": None, "sortino": None, "calmar": None,
        "info_ratio": None, "treynor": None,
        "vol_1m": None, "max_dd_1m": None,
    }
    # Strip NaN/Inf/zero prices before any calculation
    closes = [c for c in closes if c and not math.isnan(c) and not math.isinf(c) and c > 0]
    if len(closes) < 2:
        return _empty
    arr = np.array(closes, dtype=float)
    n = len(arr)
    daily_ret = np.diff(arr) / arr[:-1]
    total_ret = (arr[-1] - arr[0]) / arr[0]
    ann_ret = ((1 + total_ret) ** (252 / n) - 1) * 100
    vol = float(np.std(daily_ret) * np.sqrt(252) * 100)
    rf = 0.05  # 5% risk-free rate

    # ── Sharpe: (R − rf) / σ_total ───────────────────────────────────────
    sharpe = (ann_ret / 100 - rf) / (vol / 100) if vol > 0 else None

    # ── Sortino: (R − rf) / σ_downside ───────────────────────────────────
    neg = daily_ret[daily_ret < 0]
    down_vol = float(np.sqrt(np.sum(neg ** 2) / max(len(daily_ret), 1) * 252) * 100) if len(neg) > 0 else 0.01
    sortino = (ann_ret / 100 - rf) / (down_vol / 100) if down_vol > 0 else None

    # ── Calmar: ann_ret / max_drawdown (full history) ─────────────────────
    calmar = None
    peak = float(arr[0])
    max_dd_full = 0.0
    for p in arr:
        if p > peak:
            peak = p
        dd = (peak - p) / peak
        if dd > max_dd_full:
            max_dd_full = dd
    if max_dd_full > 0:
        calmar = (ann_ret / 100) / max_dd_full

    # ── Information Ratio: (R_stock − R_spy) / tracking_error ─────────────
    info_ratio = None
    if spy_closes and len(spy_closes) >= 2:
        # Align lengths
        spy_arr = np.array(spy_closes[-n:] if len(spy_closes) >= n else spy_closes, dtype=float)
        stock_sub = arr[-len(spy_arr):]
        if len(spy_arr) >= 2 and len(spy_arr) == len(stock_sub):
            spy_ret = np.diff(spy_arr) / spy_arr[:-1]
            stock_ret_sub = np.diff(stock_sub) / stock_sub[:-1]
            excess_daily = stock_ret_sub - spy_ret
            tracking_err = float(np.std(excess_daily) * np.sqrt(252))
            spy_total = (spy_arr[-1] - spy_arr[0]) / spy_arr[0]
            spy_ann = ((1 + spy_total) ** (252 / len(spy_arr)) - 1)
            excess_ann = ann_ret / 100 - spy_ann
            if tracking_err > 0:
                info_ratio = excess_ann / tracking_err

    # ── Treynor: (R − rf) / β ────────────────────────────────────────────
    treynor = None
    if beta is not None and abs(beta) > 0.001:
        treynor = (ann_ret / 100 - rf) / beta

    # ── 1-month slice (last 21 trading days) ──────────────────────────────
    BARS_1M = 21
    arr_1m = arr[-BARS_1M:] if len(arr) >= BARS_1M else arr
    n1 = len(arr_1m)
    vol_1m = max_dd_1m = ann_ret_1m = None
    if n1 >= 2:
        dr_1m = np.diff(arr_1m) / arr_1m[:-1]
        total_1m = (arr_1m[-1] - arr_1m[0]) / arr_1m[0]
        ann_ret_1m = ((1 + total_1m) ** (252 / n1) - 1) * 100
        vol_1m = float(np.std(dr_1m) * np.sqrt(252) * 100)
        peak_1m = arr_1m[0]
        max_dd_1m_raw = 0.0
        for p in arr_1m:
            if p > peak_1m:
                peak_1m = p
            dd = (peak_1m - p) / peak_1m
            if dd > max_dd_1m_raw:
                max_dd_1m_raw = dd
        max_dd_1m = round(max_dd_1m_raw * 100, 2)  # as positive %

    raw = {
        "ann_ret": round(ann_ret_1m, 2) if ann_ret_1m is not None else None,
        "vol": round(vol, 2),
        "sharpe": round(sharpe, 3) if sharpe is not None else None,
        "gain_sharpe": round(sortino, 3) if sortino is not None else None,  # legacy key
        "sortino": round(sortino, 3) if sortino is not None else None,
        "calmar": round(calmar, 3) if calmar is not None else None,
        "info_ratio": round(info_ratio, 3) if info_ratio is not None else None,
        "treynor": round(treynor, 3) if treynor is not None else None,
        "vol_1m": round(vol_1m, 2) if vol_1m is not None else None,
        "max_dd_1m": max_dd_1m,
    }
    # Final NaN/Inf guard — ensures JSON serialisation never fails
    return {k: _safe(v) for k, v in raw.items()}
