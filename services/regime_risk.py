"""
Regimes as a RISK LENS, not only a switch (ML4T 3e §1.4, §9.5 — gap 4).

services/regime.py classifies the market into one of four quadrants with hard
thresholds (ADX 20/25, SPY vs its 200-day MA, VIX 25). Used as an on/off
switch that is fragile: ADX 19.9 and 20.1 are the same market, yet one side
of the line turns a strategy fully on and the other fully off, and position
size never reflects how hostile the tape actually is.

This module adds two CONTINUOUS readings of the same inputs:

  * `market_risk(...)` — a 0..1 hostility score and a `multiplier` in
    [REGIME_MULTIPLIER_MIN, 1.0]. Regime can only CUT risk, never lever it
    up: a calm bull market sizes exactly as today (1.0), a volatile bear
    sizes down toward half. Every input enters through a logistic ramp, so
    the multiplier is continuous and monotone — there is no step at ADX
    20/25, at the 200-day MA or at the VIX threshold.
  * `quadrant_memberships(...)` / `strategy_fit(...)` — a fuzzy version of
    services/regime.py::classify_quadrant. The four memberships always sum
    to 1, and at the extremes (far from every threshold) they collapse to
    exactly the hard quadrant, so the fit of a strategy is 1.0 where it is
    tagged and 0.0 where it is not — the hard behaviour. Near a boundary a
    strategy tagged for the neighbouring quadrant gets a partial weight
    ("fading in" / "fading out").

Nothing here decides which strategies are ACTIVE — the hand tags still do
(services/regime.py::strategies_for_regime). The continuous numbers only
change position SIZE, and only for users who turn on `User.regime_sizing`
(ships OFF: it changes real position sizes). `sizing_multiplier()` is the
one function the live path and the backtest share.

Pure functions, no DB, no network.
"""
from __future__ import annotations

import math

from services.regime import ADX_CHOPPY, ADX_TRENDING, QUADRANTS, VIX_CRISIS
# Floor/ceiling of the regime multiplier live with the sizing code that
# enforces them: regime alone never takes a position below half its normal
# risk, and never above it (1.0 is the ceiling).
from services.trade_plan import REGIME_MULTIPLIER_MAX, REGIME_MULTIPLIER_MIN  # noqa: F401

#: Realized-volatility proxy threshold used when no VIX reading exists (the
#: same 22 the backtest uses in services/backtest.py for historical bars).
REALIZED_VOL_CRISIS = 22.0

# Logistic ramp widths — the distance over which each input moves from ~12%
# to ~88% of its effect (≈ ±2 widths covers the whole transition).
_DIR_WIDTH_PCT = 1.0      # SPY % distance from its 200-day MA
_VOL_WIDTH = 2.0          # VIX / realized-vol points
_ADX_WIDTH = 1.25         # ADX points

# Hostility weights (combined multiplicatively, so each stays monotone and the
# total stays in [0, 1)): below the 200-day MA, elevated volatility, and a
# directionless tape each take a bite out of the risk budget.
_W_BEAR = 0.45
_W_VOL = 0.40
_W_CHOP = 0.10
_SCORE_MAX = 1.0 - (1.0 - _W_BEAR) * (1.0 - _W_VOL) * (1.0 - _W_CHOP)


def _logistic(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _degrees(adx, spy_vs_ma200_pct, vol, vol_source: str) -> tuple[float, float, float]:
    """(trending, above, stressed) degrees in [0, 1].

    - trending: ADX relative to ADX_CHOPPY (20) — the hard quadrant boundary
      in classify_quadrant (the 20-25 "transition" band already counts as
      trending there). Unknown ADX -> 0.5.
    - above: SPY's % distance from its 200-day MA. Unknown -> 0.5.
    - stressed: VIX vs VIX_CRISIS (25), or the realized-vol proxy vs 22.
      Unknown -> 0.5.
    """
    trending = _logistic((float(adx) - ADX_CHOPPY) / _ADX_WIDTH) if _finite(adx) else 0.5
    above = _logistic(float(spy_vs_ma200_pct) / _DIR_WIDTH_PCT) if _finite(spy_vs_ma200_pct) else 0.5
    if _finite(vol):
        thr = VIX_CRISIS if vol_source == "vix" else REALIZED_VOL_CRISIS
        stressed = _logistic((float(vol) - thr) / _VOL_WIDTH)
    else:
        stressed = 0.5
    return trending, above, stressed


def market_risk(adx=None, spy_vs_ma200_pct=None, vol=None, vol_source: str = "vix") -> dict:
    """Continuous hostility score and risk multiplier for the current tape.

    Returns {score, multiplier, components, explanation}; `score` is 0 (benign)
    .. 1 (most hostile the inputs allow), `multiplier` is in
    [REGIME_MULTIPLIER_MIN, 1.0] and falls monotonically as the score rises.
    Choppiness is measured against ADX_TRENDING (25) — a market in the 20-25
    band is still somewhat directionless, so it takes a partial bite.
    """
    trending, above, stressed = _degrees(adx, spy_vs_ma200_pct, vol, vol_source)
    bear = 1.0 - above
    strong = _logistic((float(adx) - ADX_TRENDING) / _ADX_WIDTH) if _finite(adx) else 0.5
    chop = 1.0 - strong
    raw = 1.0 - (1.0 - _W_BEAR * bear) * (1.0 - _W_VOL * stressed) * (1.0 - _W_CHOP * chop)
    score = max(0.0, min(1.0, raw / _SCORE_MAX))
    multiplier = REGIME_MULTIPLIER_MAX - (REGIME_MULTIPLIER_MAX - REGIME_MULTIPLIER_MIN) * score
    multiplier = max(REGIME_MULTIPLIER_MIN, min(REGIME_MULTIPLIER_MAX, multiplier))

    parts = []
    if bear >= 0.5:
        parts.append(f"SPY below its 200-day MA ({bear:.0%} of the bear weight)")
    elif bear > 0.12:
        parts.append(f"SPY close to its 200-day MA ({bear:.0%} of the bear weight)")
    if stressed >= 0.5:
        parts.append(f"elevated volatility ({stressed:.0%} of the volatility weight)")
    elif stressed > 0.12:
        parts.append(f"volatility near the stress line ({stressed:.0%})")
    if chop >= 0.5:
        parts.append("no strong trend (ADX below 25)")
    explanation = (
        f"Regime risk multiplier {multiplier:.2f}: "
        + ("; ".join(parts) if parts else "calm, trending, above the 200-day MA — full size")
        + ". Regime can only cut risk, never raise it."
    )
    return {
        "score": round(score, 4),
        "multiplier": round(multiplier, 4),
        "floor": REGIME_MULTIPLIER_MIN,
        "components": {
            "direction": {"spy_vs_ma200_pct": _round(spy_vs_ma200_pct), "bear": round(bear, 4),
                          "weight": _W_BEAR},
            "volatility": {"value": _round(vol), "source": vol_source if _finite(vol) else None,
                           "stress": round(stressed, 4), "weight": _W_VOL},
            "trend": {"adx": _round(adx), "chop": round(chop, 4), "weight": _W_CHOP},
        },
        "explanation": explanation,
    }


def _round(v, digits: int = 2):
    return round(float(v), digits) if _finite(v) else None


def quadrant_memberships(adx=None, spy_vs_ma200_pct=None, vol=None,
                         vol_source: str = "vix") -> dict[str, float]:
    """Fuzzy version of services/regime.py::classify_quadrant. Sums to 1.

    With t = trending, u = above the 200-day MA, v = stressed, the hard tree
      trending & above & calm   -> trending_bull
      trending & below          -> trending_bear
      crisis vol or below       -> choppy_volatile   (incl. trending & above & stressed)
      otherwise                 -> choppy_calm
    becomes the products below; at t, u, v in {0, 1} it IS the hard tree.
    """
    t, u, v = _degrees(adx, spy_vs_ma200_pct, vol, vol_source)
    m = {
        "trending_bull": t * u * (1 - v),
        "trending_bear": t * (1 - u),
        "choppy_calm": (1 - t) * u * (1 - v),
        "choppy_volatile": (1 - t) * (1 - u * (1 - v)) + t * u * v,
    }
    return {q: round(m[q], 6) for q in QUADRANTS}


def strategy_fit(tagged_regimes, memberships: dict[str, float]) -> dict:
    """How much of the current market sits in the quadrants a strategy is
    tagged for: `weight` in [0, 1], plus a label for the UI.

    `full` (>= 0.9) and `out` (<= 0.1) are the hard behaviour; `fading_out`
    (0.5-0.9) means most of the market still fits but it is near an edge,
    `fading_in` (0.1-0.5) that only a minority fits — typically a strategy
    whose neighbouring quadrant the market is drifting toward. The label never
    changes whether the strategy is active — only, when regime sizing is on,
    how big it trades.
    """
    tagged = [q for q in (tagged_regimes or []) if q in memberships]
    weight = max(0.0, min(1.0, sum(memberships.get(q, 0.0) for q in tagged)))
    if weight >= 0.9:
        label = "full"
    elif weight <= 0.1:
        label = "out"
    elif weight >= 0.5:
        label = "fading_out"
    else:
        label = "fading_in"
    return {"weight": round(weight, 4), "label": label}


def sizing_multiplier(market_multiplier, fit_weight) -> float:
    """The per-setup regime factor used when `regime_sizing` is on:
    market multiplier × (0.5 + 0.5 × fit), clamped to
    [REGIME_MULTIPLIER_MIN, 1.0]. A strategy fully in its regime in a calm
    market sizes at 1.0 (today's size); a strategy at the edge of its regime
    or in a hostile tape sizes down, never below half. It is combined with
    the edge multiplier in services/trade_plan.py (see there for the bound
    on the product)."""
    m = float(market_multiplier) if _finite(market_multiplier) else 1.0
    f = float(fit_weight) if _finite(fit_weight) else 1.0
    f = max(0.0, min(1.0, f))
    return round(max(REGIME_MULTIPLIER_MIN, min(REGIME_MULTIPLIER_MAX, m * (0.5 + 0.5 * f))), 4)


def fit_sentence(name: str, fit: dict | None) -> str | None:
    """Plain-English line for the Weekly Plan / Playbook, or None when the
    strategy is squarely in or out of its regime (nothing to say)."""
    if not fit:
        return None
    w = fit.get("weight")
    if fit.get("label") == "fading_out":
        return (f"{name} is near the edge of its regime — about {w:.0%} of today's market "
                f"looks like the conditions it is built for.")
    if fit.get("label") == "fading_in":
        return (f"{name} is only partly in its regime — about {w:.0%} of today's market looks "
                f"like the conditions it is built for. Its hand-written tag still decides "
                f"whether it runs; this only affects size when regime sizing is on.")
    return None
