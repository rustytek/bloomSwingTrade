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
    """Strategy ids tagged as a good fit for the given quadrant.

    PINNED CONTRACT — services/today.py and services/backtest.py both call this
    with exactly one positional argument and expect a set back. The evidence
    override lives in `strategies_for_regime_with_evidence` below precisely so
    this signature and its behaviour never move.
    """
    from services.strategies import STRATEGIES
    if not quadrant:
        return set(STRATEGIES)
    return {sid for sid, strat in STRATEGIES.items() if quadrant in getattr(strat, "regimes", [])}


# ──────────────────────────────────────────────────────────────────────────
# Evidence override (opt-in)
# ──────────────────────────────────────────────────────────────────────────
# `Strategy.regimes` is a hand-written opinion. services/edge_matrix.py turns
# the backtest into evidence about whether that opinion holds per quadrant.
# When a matrix is supplied AND the override is enabled, we:
#   - EXCLUDE strategies the evidence marks "mis-tagged" here (tagged, but
#     losing money on an adequate sample), and
#   - INCLUDE strategies marked "untagged-edge" here (not tagged, but a solid
#     positive edge) as candidates.
# "unproven" changes nothing — the hand tag stands, because "we have no
# evidence" must never be read as "the evidence is bad".
#
# SAFETY INVARIANT: this function can never return an empty set when the plain
# tag-based lookup would have returned a non-empty one. A stale, empty or
# corrupt matrix degrades to today's behaviour; it cannot blank the user's
# playbook. `evidence_adjustment()` reports when that fallback fired.
EVIDENCE_OVERRIDE_ENABLED = False


def set_evidence_override(enabled: bool) -> None:
    """Module-level opt-in for the evidence override. Off by default so the
    live app behaves exactly as it does today until someone turns it on."""
    global EVIDENCE_OVERRIDE_ENABLED
    EVIDENCE_OVERRIDE_ENABLED = bool(enabled)


def _verdicts_for(matrix: dict | None, quadrant: str | None) -> dict[str, str]:
    """{strategy_id: verdict} for one quadrant, tolerating any malformed matrix."""
    if not matrix or not quadrant or not isinstance(matrix, dict):
        return {}
    strategies = matrix.get("strategies")
    if not isinstance(strategies, dict):
        return {}
    out: dict[str, str] = {}
    for sid, entry in strategies.items():
        if not isinstance(entry, dict):
            continue
        cells = entry.get("cells")
        if not isinstance(cells, dict):
            continue
        cell = cells.get(quadrant)
        if isinstance(cell, dict) and isinstance(cell.get("verdict"), str):
            out[sid] = cell["verdict"]
    return out


def evidence_adjustment(quadrant: str | None, matrix: dict | None = None,
                        force: bool = False) -> dict:
    """What the evidence would change for `quadrant`, and whether it applied.

    Returns: base, excluded, added, result, applied, fallback, reason.
    `force=True` applies the override regardless of EVIDENCE_OVERRIDE_ENABLED
    (used by the API/tests so a preview never needs a global flag flip).
    """
    from services.strategies import STRATEGIES

    base = strategies_for_regime(quadrant)
    enabled = force or EVIDENCE_OVERRIDE_ENABLED

    if not enabled:
        return {"base": sorted(base), "excluded": [], "added": [], "result": sorted(base),
                "applied": False, "fallback": False, "reason": "Evidence override is off."}

    verdicts = _verdicts_for(matrix, quadrant)
    if not verdicts:
        return {"base": sorted(base), "excluded": [], "added": [], "result": sorted(base),
                "applied": False, "fallback": True,
                "reason": "No usable edge matrix for this quadrant — using the hand-written tags."}

    excluded = sorted(s for s in base if verdicts.get(s) == "mis-tagged")
    added = sorted(
        s for s, v in verdicts.items()
        if v == "untagged-edge" and s not in base and s in STRATEGIES
    )
    result = (base - set(excluded)) | set(added)

    fallback = False
    reason = "Evidence applied."
    if not result and base:
        # SAFETY INVARIANT: never hand back an empty playbook.
        result = set(base)
        excluded, added, fallback = [], [], True
        reason = ("Evidence would have excluded every strategy for this regime — "
                  "ignored and fell back to the hand-written tags.")
    return {"base": sorted(base), "excluded": excluded, "added": added,
            "result": sorted(result), "applied": not fallback, "fallback": fallback,
            "reason": reason}


def strategies_for_regime_with_evidence(quadrant: str | None, matrix: dict | None = None,
                                        force: bool = False) -> set[str]:
    """Evidence-aware version of `strategies_for_regime`.

    With no matrix (or the override disabled) the result is IDENTICAL to
    `strategies_for_regime(quadrant)` — that equivalence is covered by
    test_edge.py for all four quadrants plus the None case.
    """
    return set(evidence_adjustment(quadrant, matrix, force=force)["result"])
