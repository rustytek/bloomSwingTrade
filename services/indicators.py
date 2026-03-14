"""
Technical indicator calculations.
Pure Python / numpy — mirrors the logic from the HTML template.
"""
import numpy as np
from typing import Optional


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
    if len(closes) < 2:
        return {
            "ann_ret": None, "vol": None, "sharpe": None,
            "gain_sharpe": None, "sortino": None, "calmar": None,
            "info_ratio": None, "treynor": None,
            "vol_1m": None, "max_dd_1m": None,
        }
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

    return {
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
