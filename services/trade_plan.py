"""
Trade plan generation — entry zone, ATR stop, fixed-fractional position size, R target.

Sizing follows the standard fixed-fractional model: risk a configured percent of
account equity per trade, with the stop distance (entry − stop) defining risk per
share. Position value is additionally capped at max_position_pct of the account.
"""
from __future__ import annotations

import math

from services.indicators import calc_atr


#: Edge-weighted sizing is clamped to this band. A conviction/edge estimate is
#: itself an estimate — letting it run free means one bad backtest number can
#: triple position size. 0.5x–1.5x is meaningful tilt without blow-up risk.
EDGE_MULTIPLIER_MIN = 0.5
EDGE_MULTIPLIER_MAX = 1.5


def _finite(v) -> bool:
    """True only for real, finite numbers (rejects None, NaN, ±inf)."""
    return isinstance(v, (int, float)) and math.isfinite(v)


def clamp_edge_multiplier(edge_multiplier) -> float:
    """Clamp an edge multiplier into [EDGE_MULTIPLIER_MIN, EDGE_MULTIPLIER_MAX].

    Non-numeric / NaN / None all fall back to 1.0 (neutral = today's behavior).
    """
    if not _finite(edge_multiplier):
        return 1.0
    return max(EDGE_MULTIPLIER_MIN, min(EDGE_MULTIPLIER_MAX, float(edge_multiplier)))


def build_trade_plan(
    bars: list[dict],
    account_size: float,
    risk_pct: float,
    entry: float | None = None,
    atr_mult: float = 2.5,
    r_multiple: float = 2.0,
    max_position_pct: float = 25.0,
    atr_period: int = 14,
    edge_multiplier: float = 1.0,
) -> dict | None:
    """Build a complete trade plan from OHLCV bars and user risk settings.

    `edge_multiplier` scales the per-trade risk budget
    (`risk_dollars = account_size * risk_pct/100 * edge_multiplier`) so a setup
    with a tested edge can be sized up and a marginal one sized down. It is
    clamped to [0.5, 1.5]; the default of 1.0 reproduces the un-weighted
    fixed-fractional sizing exactly.

    Returns None when there isn't enough data to compute an ATR or when the
    inputs can't produce a sane plan (zero/negative risk per share).
    """
    if not bars or not account_size or account_size <= 0 or not risk_pct or risk_pct <= 0:
        return None

    highs = [b.get("high") for b in bars]
    lows = [b.get("low") for b in bars]
    closes = [b.get("close") for b in bars]
    # Cached bars can hold NaN floats, not just None — NaN slips past every
    # comparison below (NaN <= 0 is False) and only blows up at math.floor().
    if not all(_finite(v) for v in (highs[-1], lows[-1], closes[-1])):
        return None

    atr_series = calc_atr(highs, lows, closes, atr_period)
    atr = next((v for v in reversed(atr_series) if _finite(v)), None)
    if atr is None or atr <= 0:
        return None

    entry = float(entry) if entry else float(closes[-1])
    if not _finite(entry) or entry <= 0:
        return None

    stop = entry - atr_mult * atr
    risk_per_share = entry - stop
    if not _finite(stop) or not _finite(risk_per_share) or stop <= 0 or risk_per_share <= 0:
        return None

    target = entry + r_multiple * risk_per_share
    edge = clamp_edge_multiplier(edge_multiplier)
    risk_dollars = account_size * risk_pct / 100.0 * edge
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
        # Same value as `stop`, carried separately so the entry path can persist
        # PortfolioPosition.initial_stop — the R denominator, which must never be
        # rewritten when the live stop is trailed up.
        "initial_stop": round(stop, 2),
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
        "edge_multiplier": edge,
    }


# ──────────────────────────────────────────────────────────────────────────
# Legacy plan-notes parser
# ──────────────────────────────────────────────────────────────────────────
# Before PortfolioPosition gained real `thesis` / `invalidation` /
# `time_stop_days` / `planned_entry` columns, the Plan-a-Trade modal packed the
# plan's intent into `notes` as prefixed lines (static/today.html):
#
#     THESIS: <free text>
#     INVALIDATION: <free text>
#     TIME STOP: <free text, usually a day count>
#     PLAN: entry 123.45 stop 118.20 target 133.90 2.0R
#     STRATEGY: <strategy name>
#
# `parse_plan_notes` is the ONE place that shape is decoded — used by the
# one-time backfill in main.py and unit-tested by test_plan_persistence.py.
# It never guesses: a prefix that is absent yields None, and a TIME STOP line
# with no integer in it yields None rather than a made-up horizon.

_PLAN_NOTE_PREFIXES = {
    "THESIS:": "thesis",
    "INVALIDATION:": "invalidation",
    "TIME STOP:": "time_stop",
    "PLAN:": "plan",
    "STRATEGY:": "strategy",
}

def _plan_number(text: str, keyword: str):
    """First finite number following `keyword` in `text`, else None."""
    import re
    m = re.search(rf"\b{keyword}\s+\$?(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if not m:
        return None
    try:
        val = float(m.group(1))
    except ValueError:
        return None
    return val if math.isfinite(val) else None


def parse_plan_notes(notes: str | None) -> dict:
    """Decode the legacy prefixed-notes format into the real column values.

    Returns a dict with the keys `thesis`, `invalidation`, `time_stop_days`,
    `planned_entry`, `planned_entry_high` and `strategy_name`. Every key is
    present; a value is None when the corresponding prefix was absent or
    carried nothing parseable. `notes` itself is never modified — the caller
    keeps it as the only free-form record of the trade's reasoning.

    Lines that do not start with a known prefix are treated as a continuation
    of the previous prefixed field (a multi-line thesis), never as a new one.
    """
    out = {
        "thesis": None,
        "invalidation": None,
        "time_stop_days": None,
        "planned_entry": None,
        "planned_entry_high": None,
        "strategy_name": None,
    }
    if not notes or not isinstance(notes, str):
        return out

    buckets: dict[str, list[str]] = {}
    current: str | None = None
    for raw in notes.splitlines():
        line = raw.strip()
        if not line:
            if current:
                buckets.setdefault(current, []).append("")
            continue
        matched = None
        upper = line.upper()
        for prefix, key in _PLAN_NOTE_PREFIXES.items():
            if upper.startswith(prefix):
                matched = key
                line = line[len(prefix):].strip()
                break
        if matched:
            current = matched
            buckets.setdefault(current, []).append(line)
        elif current:
            buckets.setdefault(current, []).append(line)
        # A leading un-prefixed line is ordinary free-form notes: ignored here.

    def _joined(key):
        if key not in buckets:
            return None
        text = "\n".join(buckets[key]).strip()
        return text or None

    out["thesis"] = _joined("thesis")
    out["invalidation"] = _joined("invalidation")
    out["strategy_name"] = _joined("strategy")

    time_text = _joined("time_stop")
    if time_text:
        import re
        m = re.search(r"-?\d+", time_text)
        if m:
            try:
                days = int(m.group(0))
            except ValueError:
                days = None
            if days is not None and days > 0:
                out["time_stop_days"] = days

    plan_text = _joined("plan")
    if plan_text:
        out["planned_entry"] = _plan_number(plan_text, "entry")
        # The legacy line never carried a zone, but a "zone 120.00-124.50" or
        # "zone high 124.50" variant is decoded if one is ever present.
        high = _plan_number(plan_text, "zone high")
        if high is None:
            import re
            m = re.search(r"\bzone\s+\$?-?\d+(?:\.\d+)?\s*[-–]\s*\$?(-?\d+(?:\.\d+)?)",
                          plan_text, re.IGNORECASE)
            if m:
                try:
                    cand = float(m.group(1))
                    high = cand if math.isfinite(cand) else None
                except ValueError:
                    high = None
        out["planned_entry_high"] = high

    return out
