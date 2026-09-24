"""
Plain-English "why" for every strategy — the reasoning the Playbook, the
Weekly Plan and the Strategy Lab show next to a strategy's rules.

`Strategy.details` (services/strategies.py) says HOW a strategy works. This
module says WHY you would use it at all, and why it is (or is not) the right
tool for the market you are in right now:

  why_it_works — the behavioural/structural reason the edge exists
  best_when    — the conditions it is built for
  fails_when   — how it loses money, so the user knows what "normal" pain is
  regime_fit   — one sentence per regime quadrant (services/regime.py QUADRANTS)

This is written opinion, not evidence. Evidence lives in the edge matrix
(services/edge_matrix.py); the UI shows both side by side and never presents
this text as a tested result. Pure data — no imports from the app, so it is
safe to use from any layer.
"""
from __future__ import annotations

RATIONALE: dict[str, dict] = {
    "momentum_rotation": {
        "why_it_works": (
            "Investors under-react to good news and pile in late, so stocks that have trended "
            "up smoothly tend to keep trending for weeks to months. Ranking by trend strength "
            "x smoothness (R²) favours steady institutional buying over one-off news spikes."
        ),
        "best_when": "The broad market is in a clear uptrend and leadership is persistent.",
        "fails_when": (
            "Sharp trend reversals and choppy markets — leaders get sold hardest in a sudden "
            "risk-off turn, and a sideways market keeps rotating you in and out at a loss."
        ),
        "regime_fit": {
            "trending_bull": "Home turf: a rising tide with persistent leaders is exactly what momentum rides.",
            "trending_bear": "Stands down: in a downtrend the 'strongest' names are usually just falling slower.",
            "choppy_calm": "Stands down: without a trend, momentum ranks churn and whipsaw.",
            "choppy_volatile": "Stands down: momentum crashes happen in exactly this kind of volatile reversal.",
        },
    },
    "pullback_50ma": {
        "why_it_works": (
            "Healthy uptrends breathe. Buying a dip to the rising 50-day average gets you into a "
            "proven trend at a better price, with a natural, nearby place to be wrong (the average "
            "breaking), so the risk per share is small relative to the upside."
        ),
        "best_when": "An established uptrend (price above a rising 200-day MA) is taking a normal rest.",
        "fails_when": "The dip is the start of a real trend change — the 50MA breaks and the stop gets hit.",
        "regime_fit": {
            "trending_bull": "Home turf: dips in a bull trend get bought, which is what this strategy relies on.",
            "trending_bear": "Stands down: in a bear market dips keep dipping.",
            "choppy_calm": "Stands down: the 50MA is not reliable support when nothing is trending.",
            "choppy_volatile": "Stands down: volatile markets slice straight through moving averages.",
        },
    },
    "breakout_volume": {
        "why_it_works": (
            "A new 60-day high on heavy volume means buyers with size have overwhelmed the sellers "
            "who were capping the price. Those big buyers rarely finish in one day, so the move "
            "tends to follow through."
        ),
        "best_when": "A bull market where fresh highs are being rewarded rather than sold.",
        "fails_when": "False breakouts in weak or choppy markets — price pops above resistance and falls back.",
        "regime_fit": {
            "trending_bull": "Home turf: breakouts follow through when the whole market is lifting.",
            "trending_bear": "Stands down: breakouts in a bear market are usually traps.",
            "choppy_calm": "Stands down: range-bound markets fade breakouts back into the range.",
            "choppy_volatile": "Stands down: volume spikes here are fear, not accumulation.",
        },
    },
    "mean_reversion": {
        "why_it_works": (
            "When a stock in a long-term uptrend gets sold hard for a few days (RSI(2) under 10), "
            "the selling is usually short-term panic or forced liquidation rather than a change in "
            "the business, and price tends to snap back toward its average within days."
        ),
        "best_when": "Calm, sideways markets where overreactions reverse quickly.",
        "fails_when": (
            "The start of a real crash — a 'washout' keeps washing out. That is why it only buys "
            "names still above their 200-day MA, and holds for days, not weeks."
        ),
        "regime_fit": {
            "trending_bull": "Sits out by default: it works here too, but trend strategies get first call on the slots.",
            "trending_bear": "Stands down: oversold keeps getting more oversold in a downtrend.",
            "choppy_calm": "Home turf: sideways, low-drama markets are where short-term overreactions revert fastest.",
            "choppy_volatile": "Stands down: in a crisis the snap-back often does not come.",
        },
    },
    "dual_momentum": {
        "why_it_works": (
            "Two filters in one: only own things that have beaten cash over the past year "
            "(absolute momentum — keeps you out of falling markets), then pick the ones with the "
            "best return per unit of risk (relative momentum). It is designed to sidestep big bear "
            "markets rather than ride them down."
        ),
        "best_when": "Any trending market — up or down — because the cash test does the defending.",
        "fails_when": "Fast V-shaped reversals, where the 12-month lookback is too slow to catch the turn.",
        "regime_fit": {
            "trending_bull": "Runs: it holds the strongest risk-adjusted trends.",
            "trending_bear": "Runs defensively: the 'must beat cash' rule leaves only the few names still genuinely rising.",
            "choppy_calm": "Stands down: its slow lookback adds little in a directionless market.",
            "choppy_volatile": "Stands down: lookbacks lag badly when the market whipsaws.",
        },
    },
    "volatility_breakout": {
        "why_it_works": (
            "Ranges end with a burst of volatility. A close above the 20-day high while ATR is "
            "expanding catches a new trend near its start, before slower momentum rankings notice it."
        ),
        "best_when": "Early in a bull trend, when fresh names are breaking out of bases.",
        "fails_when": "Volatility expands because of fear, not demand — breakouts then reverse quickly.",
        "regime_fit": {
            "trending_bull": "Home turf: expanding volatility in an uptrend usually means new buyers arriving.",
            "trending_bear": "Stands down: expanding volatility in a downtrend is usually selling pressure.",
            "choppy_calm": "Stands down: without a market tailwind most range breaks fail.",
            "choppy_volatile": "Stands down: every day looks like a 'breakout' when volatility is high.",
        },
    },
    "sector_rotation": {
        "why_it_works": (
            "Money moves between sectors in waves that last months (e.g. into energy, out of tech). "
            "Owning the sector ETFs with the strongest, smoothest trend spreads the bet across a "
            "whole industry, so single-stock surprises matter much less."
        ),
        "best_when": "Bull or calm-sideways markets where leadership rotates rather than disappears.",
        "fails_when": "Everything falls together in a crisis — sector diversification stops helping.",
        "regime_fit": {
            "trending_bull": "Runs: it keeps you in the sectors leading the advance.",
            "trending_bear": "Stands down: in a broad downtrend there is rarely a sector worth owning long.",
            "choppy_calm": "Runs: in a sideways index, rotation between sectors is where the movement is.",
            "choppy_volatile": "Stands down: correlations jump toward 1 and every sector drops at once.",
        },
    },
    "bear_reversal_watch": {
        "why_it_works": (
            "A sharp bounce inside a confirmed downtrend is usually a relief rally that fails. This "
            "is a WATCHLIST signal only — the app is long-only, so it tells you what NOT to buy "
            "(and what may be about to roll over) rather than giving you a trade."
        ),
        "best_when": "Bear or volatile markets, where bounces are for selling, not buying.",
        "fails_when": "A genuine bottom — occasionally the bounce is the start of a new uptrend.",
        "regime_fit": {
            "trending_bull": "Stands down: bounces in a bull market are usually real.",
            "trending_bear": "Runs (watch only): flags relief rallies you should not chase.",
            "choppy_calm": "Stands down: not enough downtrend to fade.",
            "choppy_volatile": "Runs (watch only): volatile markets throw off many failing bounces.",
        },
    },
    "low_vol_trend": {
        "why_it_works": (
            "Calm, steadily rising stocks have historically delivered similar returns to exciting "
            "ones with far smaller drawdowns — investors overpay for lottery-ticket volatility. In "
            "rough markets that makes the calm names the ones you can actually hold."
        ),
        "best_when": "Choppy or nervous markets, as a defensive way to stay invested.",
        "fails_when": "Strong, speculative bull runs — the calm names lag the high-flyers.",
        "regime_fit": {
            "trending_bull": "Sits out: in a strong bull market the higher-momentum strategies should get the slots.",
            "trending_bear": "Stands down: few names are still trending up at all.",
            "choppy_calm": "Runs: a defensive way to stay invested while trends are weak.",
            "choppy_volatile": "Runs: the calmest uptrending names are the least likely to gap against you.",
        },
    },
}


def rationale_for(strategy_id: str, quadrant: str | None = None) -> dict:
    """The rationale for one strategy, with `fit_now` resolved for `quadrant`.

    Never raises: an unknown strategy returns empty strings so a new strategy
    added without text degrades to "no rationale written" rather than a 500.
    """
    r = RATIONALE.get(strategy_id) or {}
    fit = (r.get("regime_fit") or {})
    return {
        "why_it_works": r.get("why_it_works", ""),
        "best_when": r.get("best_when", ""),
        "fails_when": r.get("fails_when", ""),
        "regime_fit": dict(fit),
        "fit_now": fit.get(quadrant, "") if quadrant else "",
    }
