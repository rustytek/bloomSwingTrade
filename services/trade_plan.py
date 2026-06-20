"""
Trade plan generation — entry zone, ATR stop, fixed-fractional position size, R target.

Sizing follows the standard fixed-fractional model: risk a configured percent of
account equity per trade, with the stop distance (entry − stop) defining risk per
share. Position value is additionally capped at max_position_pct of the account.
"""
from __future__ import annotations

import math

from services.indicators import calc_atr


def build_trade_plan(
    bars: list[dict],
    account_size: float,
    risk_pct: float,
    entry: float | None = None,
    atr_mult: float = 2.5,
    r_multiple: float = 2.0,
    max_position_pct: float = 25.0,
    atr_period: int = 14,
) -> dict | None:
    """Build a complete trade plan from OHLCV bars and user risk settings.

    Returns None when there isn't enough data to compute an ATR or when the
    inputs can't produce a sane plan (zero/negative risk per share).
    """
    if not bars or not account_size or account_size <= 0 or not risk_pct or risk_pct <= 0:
        return None

    highs = [b.get("high") for b in bars]
    lows = [b.get("low") for b in bars]
    closes = [b.get("close") for b in bars]
    if any(v is None for v in (highs[-1], lows[-1], closes[-1])):
        return None

    atr_series = calc_atr(highs, lows, closes, atr_period)
    atr = next((v for v in reversed(atr_series) if v is not None), None)
    if atr is None or atr <= 0:
        return None

    entry = float(entry) if entry else float(closes[-1])
    if entry <= 0:
        return None

    stop = entry - atr_mult * atr
    risk_per_share = entry - stop
    if stop <= 0 or risk_per_share <= 0:
        return None

    target = entry + r_multiple * risk_per_share
    risk_dollars = account_size * risk_pct / 100.0
    shares = math.floor(risk_dollars / risk_per_share)

    capped = False
    max_value = account_size * max_position_pct / 100.0
    if shares * entry > max_value:
        shares = math.floor(max_value / entry)
        capped = True
    if shares < 1:
        shares = 0

    position_value = shares * entry
    return {
        "entry": round(entry, 2),
        "entry_zone": [round(entry - 0.5 * atr, 2), round(entry + 0.5 * atr, 2)],
        "stop": round(stop, 2),
        "stop_pct": round((stop - entry) / entry * 100, 2),
        "target": round(target, 2),
        "target_pct": round((target - entry) / entry * 100, 2),
        "atr": round(atr, 3),
        "atr_mult": atr_mult,
        "risk_per_share": round(risk_per_share, 2),
        "risk_dollars": round(shares * risk_per_share, 2) if shares else round(risk_dollars, 2),
        "shares": shares,
        "position_value": round(position_value, 2),
        "position_pct": round(position_value / account_size * 100, 2),
        "r_multiple": r_multiple,
        "capped_by_max_position": capped,
    }
