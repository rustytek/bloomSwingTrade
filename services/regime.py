"""
Market regime classifier.

Two independent axes decide which strategies should be active:
  - DIRECTION: is SPY above or below its 200-day MA (risk-on/off)?
  - TREND STRENGTH: is the market actually trending, or just chopping?
    This is the axis the old red/yellow/green light was missing, and it is
    what should trigger a strategy switch even when direction hasn't
    changed — a choppy uptrend chews up trend-followers just as badly as a
    choppy downtrend chews up dip-buyers.

Trend strength is measured with ADX(14) on SPY, the standard indicator for
"is there a trend to trade at all" (as opposed to which way price is
pointed):
  - ADX >= 25  -> trending  (momentum/breakout/dual-momentum/rotation edges hold up)
  - ADX < 20   -> choppy    (mean-reversion/rotation edges hold up; trend-followers whipsaw)
  - 20-25      -> transition band

Combined with direction and a VIX-based volatility overlay, this produces
four quadrants. Each Strategy declares which quadrant(s) it's built for
(`Strategy.regimes`), so the "why should I stop using this" question has a
concrete, mechanical answer: the quadrant changed, or ADX crossed a
threshold.
"""
from __future__ import annotations

from services.indicators import calc_adx, calc_ma

QUADRANTS = ["trending_bull", "trending_bear", "choppy_calm", "choppy_volatile"]

QUADRANT_INFO = {
    "trending_bull": {
        "label": "Trending Bull",
        "description": "SPY above its 200-day MA with ADX >= 25 — a real uptrend worth riding.",
    },
    "trending_bear": {
        "label": "Trending Bear",
        "description": "SPY below its 200-day MA with ADX >= 25 — a real downtrend; capital preservation over new longs.",
    },
    "choppy_calm": {
        "label": "Choppy / Range",
        "description": "ADX < 20, VIX not elevated — sideways, low-drama market. Mean-reversion and rotation edges hold up here; trend-followers whipsaw.",
    },
    "choppy_volatile": {
        "label": "Choppy / Volatile (Crisis)",
        "description": "ADX < 25 with elevated VIX, or price broke the 200-day MA into a volatile chop — the highest-risk quadrant. Reduce size or sit in cash.",
    },
}

VIX_CRISIS = 25.0
ADX_TRENDING = 25.0
ADX_CHOPPY = 20.0


def trend_strength_label(adx_val: float | None) -> str:
    if adx_val is None:
        return "unknown"
    if adx_val >= ADX_TRENDING:
        return "trending"
    if adx_val < ADX_CHOPPY:
        return "choppy"
    return "transition"


def classify_quadrant(trend_strength: str, above_200: bool, crisis_vol: bool) -> str:
    """The shared decision tree — used both for the live regime light and for
    classifying every historical bar in a backtest, so "why change" reasoning
    stays identical between what you see live and what a backtest reports."""
    if trend_strength in ("trending", "transition") and above_200 and not crisis_vol:
        return "trending_bull"
    if trend_strength in ("trending", "transition") and not above_200:
        return "trending_bear"
    if crisis_vol or not above_200:
        return "choppy_volatile"
    return "choppy_calm"


def _adx_series(bars: list[dict]) -> dict:
    highs = [b["high"] for b in bars]
    lows = [b["low"] for b in bars]
    closes = [b["close"] for b in bars]
    return calc_adx(highs, lows, closes, 14)


def _detect_transition(adx_series: list[dict | None]) -> dict:
    """Look at the last few valid ADX readings for a threshold cross —
    this is the concrete "time to change strategy" signal."""
    valid = [(i, v["adx"]) for i, v in enumerate(adx_series) if v and v.get("adx") is not None]
    if len(valid) < 2:
        return {"changed": False, "reason": None}

    recent = valid[-6:]
    events = []
    for (_, a1), (_, a2) in zip(recent, recent[1:]):
        if a1 < ADX_TRENDING <= a2:
            events.append((
                "strengthening",
                f"ADX rose above {ADX_TRENDING:.0f} ({a1:.0f} -> {a2:.0f}) — the market "
                "started trending. Momentum, breakout, and dual-momentum setups become "
                "favored; mean-reversion setups get run over.",
            ))
        elif a1 >= ADX_TRENDING > a2:
            events.append((
                "weakening",
                f"ADX fell below {ADX_TRENDING:.0f} ({a1:.0f} -> {a2:.0f}) — the trend is "
                "losing strength. Trim trend-following size and start watching for a "
                "range-bound setup instead.",
            ))
        elif a1 < ADX_CHOPPY <= a2:
            events.append((
                "strengthening",
                f"ADX rose above {ADX_CHOPPY:.0f} ({a1:.0f} -> {a2:.0f}) — chop is "
                "starting to resolve into a directional move.",
            ))
        elif a1 >= ADX_CHOPPY > a2:
            events.append((
                "collapsing",
                f"ADX fell below {ADX_CHOPPY:.0f} ({a1:.0f} -> {a2:.0f}) — the trend has "
                "exhausted. Trend-following/breakout strategies now have the highest false-"
                "signal rate; favor mean-reversion or sector rotation instead.",
            ))
    if not events:
        return {"changed": False, "reason": None}
    direction, reason = events[-1]
    return {"changed": True, "direction": direction, "reason": reason}


def classify_regime(
    spy_bars: list[dict],
    vix_last: float | None = None,
    breadth_pct: float | None = None,
) -> dict:
    """Classify the current market into one of QUADRANTS, with the ADX
    reading and a "why change" transition event when the trend-strength
    regime has just flipped."""
    closes = [b["close"] for b in spy_bars] if spy_bars else []
    if len(closes) < 60:
        return {
            "quadrant": None,
            "quadrant_label": "Insufficient data",
            "adx": None,
            "adx_trend": None,
            "above_200ma": None,
            "transition": {"changed": False, "reason": None},
            "reasons": ["Not enough SPY history cached yet."],
        }

    ma200 = calc_ma(closes, 200)
    price = closes[-1]
    above_200 = ma200[-1] is not None and price > ma200[-1]

    adx_data = _adx_series(spy_bars)
    adx_now = adx_data["adx"]
    transition = _detect_transition(adx_data["series"])

    if vix_last is not None:
        crisis_vol = vix_last >= VIX_CRISIS
    else:
        # Backtests over historical dates don't have a per-date VIX series
        # cached — fall back to SPY's own realized volatility as a proxy for
        # "the tape is unstable" (roughly VIX-scale: annualized stdev of
        # daily returns over the last 20 bars).
        window = closes[-21:]
        if len(window) >= 2:
            rets = [(window[i] / window[i - 1] - 1) for i in range(1, len(window))]
            realized_vol = (sum(r * r for r in rets) / len(rets)) ** 0.5 * (252 ** 0.5) * 100
        else:
            realized_vol = 0.0
        crisis_vol = realized_vol >= 22.0

    trend_strength = trend_strength_label(adx_now)
    quadrant = classify_quadrant(trend_strength, above_200, crisis_vol)

    reasons = [
        f"SPY {'above' if above_200 else 'below'} its 200-day MA (direction)",
        f"ADX(14) {adx_now:.1f} — {trend_strength} (trend strength)" if adx_now is not None else "ADX unavailable",
    ]
    if vix_last is not None:
        reasons.append(f"VIX {vix_last:.1f}" + (" — elevated" if crisis_vol else ""))
    if breadth_pct is not None:
        reasons.append(f"{breadth_pct:.0f}% of universe above its 200-day MA")

    return {
        "quadrant": quadrant,
        "quadrant_label": QUADRANT_INFO[quadrant]["label"],
        "quadrant_description": QUADRANT_INFO[quadrant]["description"],
        "adx": round(adx_now, 1) if adx_now is not None else None,
        "adx_trend": trend_strength,
        "above_200ma": above_200,
        "transition": transition,
        "reasons": reasons,
    }


def strategies_for_regime(quadrant: str | None) -> set[str]:
    """Strategy ids tagged as a good fit for the given quadrant."""
    from services.strategies import STRATEGIES
    if not quadrant:
        return set(STRATEGIES)
    return {sid for sid, strat in STRATEGIES.items() if quadrant in getattr(strat, "regimes", [])}
