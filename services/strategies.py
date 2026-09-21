"""
Swing-trading strategy framework.

Each strategy implements two hooks sharing the same rules:
  - candidate(bars, idx): backtest hook — evaluate at historical bar `idx`
    using only bars[:idx+1]. Returns {"score": float, ...display metrics} or
    None when the setup's filters fail.
  - scan(ticker, bars, quote): live hook — evaluate the latest bar and return
    a Setup with explainable reasons, or None.

Strategies on daily bars:
  momentum_rotation — Clenow-style exponential regression momentum ranking
  pullback_50ma     — buy pullbacks to the 50-day MA inside an uptrend
  breakout_volume   — 60-day-high breakouts confirmed by volume expansion
  mean_reversion    — Connors-style RSI(2) washouts above the 200-day MA

The first three target 2-week to 2-month holds; mean_reversion is a shorter
3–10 day snap-back toward the 20-day mean.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from services.indicators import calc_rsi, calc_atr


@dataclass
class Setup:
    ticker: str
    strategy: str
    strategy_name: str
    score: float
    state: str                      # "triggered" | "forming"
    reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "strategy": self.strategy,
            "strategy_name": self.strategy_name,
            "score": round(self.score, 4),
            "state": self.state,
            "reasons": self.reasons,
            "metrics": self.metrics,
        }


def _tail_closes(bars: list[dict], idx: int, window: int) -> list[float]:
    lo = max(0, idx + 1 - window)
    return [b["close"] for b in bars[lo : idx + 1]]


def _sma_at(closes: list[float], period: int) -> float | None:
    if len(closes) < period:
        return None
    return float(np.mean(closes[-period:]))


class Strategy(ABC):
    id: str
    name: str
    description: str
    details: dict = {}          # horizon / how / rules / scoring / parameters
    min_bars: int = 220
    # Regime quadrants (see services/regime.py QUADRANTS) this strategy is
    # built for. Used to filter/label setups by current market regime.
    regimes: list[str] = []
    # False = informational/watchlist only — never gets a live trade_plan
    # built for it (e.g. a long-only account can't act on a bearish signal).
    actionable: bool = True

    def applies_to(self, ticker: str) -> bool:
        """Ticker-universe restriction (e.g. Sector Rotation only trades the
        11 SPDR sector ETFs). Checked by both scan() (live) and the
        walk-forward backtest, so a restricted strategy can't be silently
        run against the wrong tickers in either place."""
        return True

    @abstractmethod
    def candidate(self, bars: list[dict], idx: int) -> dict | None:
        """Score this ticker at bar `idx` using bars[:idx+1] only."""

    def scan(self, ticker: str, bars: list[dict], quote: dict) -> Setup | None:
        """Live evaluation of the latest bar. Default wraps candidate()."""
        if not self.applies_to(ticker) or len(bars) < self.min_bars:
            return None
        result = self.candidate(bars, len(bars) - 1)
        if not result:
            return None
        return Setup(
            ticker=ticker,
            strategy=self.id,
            strategy_name=self.name,
            score=result.pop("score"),
            state=result.pop("state", "triggered"),
            reasons=result.pop("reasons", []),
            metrics=result,
        )


class MomentumRotation(Strategy):
    id = "momentum_rotation"
    name = "Momentum Rotation"
    description = (
        "Clenow-style momentum: rank by 90-day exponential regression slope "
        "(annualized) x R². Smooth, persistent uptrends rank highest; gappy or "
        "choppy charts are filtered out. Hold the top names, rotate on rebalance."
    )
    details = {
        "horizon": "Trend-following; rotate on each rebalance (weekly–monthly).",
        "how": "Ranks every candidate by how strong AND how smooth its 90-day uptrend is, then holds the top N.",
        "rules": [
            "Skip names with any single-day move > 15% over the window (avoids gap-driven scores).",
            "Trend filter: price must be above its 100-day moving average.",
            "Fit a linear regression to the last 90 daily log-closes.",
            "Annualize the slope: exp(slope × 252) − 1.",
            "Multiply by R² so choppy, low-fit trends are penalized.",
            "Rank all candidates by score; hold the top N; re-rank at each rebalance.",
        ],
        "scoring": "score = (exp(slope × 252) − 1) × R²   — annualized 90-day log-price slope weighted by fit quality (R²). Must be > 0.",
        "parameters": [
            ["Regression window", "90 bars"],
            ["Trend filter", "above 100-day MA"],
            ["Gap filter", "reject if any 1-day move > 15%"],
            ["Min history", "110 bars"],
        ],
    }
    min_bars = 110
    REGRESSION_BARS = 90
    regimes = ["trending_bull"]

    def candidate(self, bars: list[dict], idx: int) -> dict | None:
        closes = _tail_closes(bars, idx, self.REGRESSION_BARS + 20)
        if len(closes) < self.min_bars or closes[-1] <= 0:
            return None
        window = closes[-self.REGRESSION_BARS:]
        if any(c <= 0 for c in window):
            return None

        # Filter: avoid names driven by a single huge gap
        rets = np.diff(np.array(window)) / np.array(window[:-1])
        if np.max(np.abs(rets)) > 0.15:
            return None

        # Trend filter: above 100-day MA
        ma100 = _sma_at(closes, 100)
        if ma100 is None or closes[-1] <= ma100:
            return None

        # Exponential regression on log prices
        y = np.log(np.array(window))
        x = np.arange(len(window), dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        y_fit = slope * x + intercept
        ss_res = float(np.sum((y - y_fit) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        annualized = math.exp(slope * 252) - 1
        score = annualized * max(0.0, r2)
        if score <= 0:
            return None

        ret_63 = (closes[-1] / closes[-64] - 1) * 100 if len(closes) >= 64 else None
        ret_21 = (closes[-1] / closes[-22] - 1) * 100 if len(closes) >= 22 else None
        return {
            "score": score,
            "state": "triggered",
            "reasons": [
                f"90D trend annualizes to {annualized * 100:+.0f}%",
                f"smooth trend (R² {r2:.2f})",
                "above 100-day MA",
            ],
            "slope_ann": round(annualized * 100, 1),
            "r2": round(r2, 3),
            "ret_21": round(ret_21, 2) if ret_21 is not None else None,
            "ret_63": round(ret_63, 2) if ret_63 is not None else None,
        }


class Pullback50MA(Strategy):
    id = "pullback_50ma"
    name = "Pullback to 50MA"
    description = (
        "Buy the dip inside a confirmed uptrend: price above the 200-day MA with "
        "the 50-day above the 200-day, a pullback to the 50-day MA (or an RSI "
        "reset below 45), then entry when price closes back above the prior "
        "day's high."
    )
    details = {
        "horizon": "Buy-the-dip swing entry inside an established uptrend (2 weeks–2 months).",
        "how": "Waits for a healthy uptrend to pull back to support, then triggers when price turns back up.",
        "rules": [
            "Uptrend gate: price > 200-day MA AND 50-day MA > 200-day MA.",
            "Momentum gate: 63-day (≈3-month) return must be positive.",
            "Pullback: price tagged the 50-day MA within ~2% recently, OR RSI(14) reset below 45 after having been above 60.",
            "State = 'triggered' when price closes back above the prior day's high (confirmation); otherwise 'forming'.",
        ],
        "scoring": "score = ret_63 / 100 − distance_to_50MA / 200   — rewards a stronger 3-month trend and a tighter pullback to the 50-day MA.",
        "parameters": [
            ["Trend filter", "close > 200MA, 50MA > 200MA"],
            ["Pullback band", "within ~2% of 50-day MA"],
            ["RSI reset", "RSI(14) < 45 after > 60"],
            ["Confirmation", "close > prior day's high"],
            ["Min history", "220 bars"],
        ],
    }
    min_bars = 220
    regimes = ["trending_bull"]

    def candidate(self, bars: list[dict], idx: int) -> dict | None:
        closes = _tail_closes(bars, idx, 260)
        if len(closes) < self.min_bars:
            return None
        lo = max(0, idx + 1 - 260)
        window = bars[lo : idx + 1]
        lows = [b["low"] for b in window]
        highs = [b["high"] for b in window]
        close = closes[-1]

        ma200 = _sma_at(closes, 200)
        ma50 = _sma_at(closes, 50)
        if ma200 is None or ma50 is None or close <= ma200 or ma50 <= ma200:
            return None
        ret_63 = (close / closes[-64] - 1) * 100 if len(closes) >= 64 else None
        if ret_63 is None or ret_63 <= 0:
            return None

        # Pullback test: tagged the 50MA recently, or RSI reset after strength
        touched_ma50 = any(
            low <= float(np.mean(closes[: len(closes) - k][-50:])) * 1.02
            for k, low in ((2, lows[-3]), (1, lows[-2]), (0, lows[-1]))
            if len(closes) - k >= 50
        )
        rsi_series = calc_rsi(closes[-80:], 14)
        rsi_now = rsi_series[-1]
        rsi_window = [v for v in rsi_series[-16:-1] if v is not None]
        rsi_reset = (
            rsi_now is not None and rsi_now < 45
            and rsi_window and max(rsi_window) > 60
        )
        if not touched_ma50 and not rsi_reset:
            return None

        # Confirmation: close back above the prior bar's high
        confirmed = len(highs) >= 2 and close > highs[-2]

        dist_ma50 = abs(close - ma50) / ma50 * 100
        score = ret_63 / 100 - dist_ma50 / 200
        reasons = ["above 200-day MA, 50MA > 200MA", f"63D trend {ret_63:+.1f}%"]
        reasons.append("pulled back to 50-day MA" if touched_ma50 else "RSI reset below 45 after strength")
        if confirmed:
            reasons.append("resumed above prior day's high")
        return {
            "score": score,
            "state": "triggered" if confirmed else "forming",
            "reasons": reasons,
            "dist_ma50": round(dist_ma50, 2),
            "rsi": round(rsi_now, 1) if rsi_now is not None else None,
            "ret_63": round(ret_63, 2),
        }


class BreakoutVolume(Strategy):
    id = "breakout_volume"
    name = "Breakout + Volume"
    description = (
        "Buy strength: a close at a new 60-day high on volume at least 1.5x the "
        "20-day average, with the trend template intact (close > 50MA > 200MA, "
        "upper third of the 52-week range)."
    )
    details = {
        "horizon": "Momentum breakout entry; ride strength for 2 weeks–2 months.",
        "how": "Buys a fresh 60-day high confirmed by a surge in volume, only while the broader trend is healthy.",
        "rules": [
            "Trend template: close > 50-day MA > 200-day MA.",
            "Range filter: in the upper third of the 52-week range (position ≥ 70%).",
            "Breakout: close at/above the highest high of the prior 60 bars.",
            "Volume confirmation: today's volume ≥ 1.5× the 20-day average → 'triggered'.",
            "Within 2% of the breakout level on ≥ 1.1× volume → 'forming'.",
        ],
        "scoring": "score = volume_ratio × (1 + max(0, ret_63)/100)   — rewards a bigger volume surge and a stronger 3-month trend behind the breakout.",
        "parameters": [
            ["Breakout lookback", "60-day high"],
            ["Volume confirm", "≥ 1.5× 20-day avg"],
            ["52-week position", "≥ 70%"],
            ["Trend template", "close > 50MA > 200MA"],
            ["Min history", "220 bars"],
        ],
    }
    min_bars = 220
    LOOKBACK = 60
    VOL_RATIO_MIN = 1.5
    regimes = ["trending_bull"]

    def candidate(self, bars: list[dict], idx: int) -> dict | None:
        closes = _tail_closes(bars, idx, 260)
        if len(closes) < self.min_bars:
            return None
        lo = max(0, idx + 1 - 260)
        window = bars[lo : idx + 1]
        highs = [b["high"] for b in window]
        lows = [b["low"] for b in window]
        vols = [b["vol"] for b in window]
        close = closes[-1]

        ma200 = _sma_at(closes, 200)
        ma50 = _sma_at(closes, 50)
        if ma200 is None or ma50 is None or not (close > ma50 > ma200):
            return None

        # 52-week range position (use available window, ~1y)
        yr_high = max(highs[-252:]) if len(highs) >= 252 else max(highs)
        yr_low = min(lows[-252:]) if len(lows) >= 252 else min(lows)
        p52w = (close - yr_low) / (yr_high - yr_low) * 100 if yr_high > yr_low else 0
        if p52w < 70:
            return None

        prior_high = max(highs[-(self.LOOKBACK + 1):-1])
        avg_vol = float(np.mean(vols[-21:-1])) if len(vols) >= 21 else None
        if not avg_vol or avg_vol <= 0:
            return None
        vol_ratio = vols[-1] / avg_vol
        ret_63 = (close / closes[-64] - 1) * 100 if len(closes) >= 64 else 0

        breakout = close >= prior_high
        near_breakout = close >= prior_high * 0.98
        if breakout and vol_ratio >= self.VOL_RATIO_MIN:
            state = "triggered"
        elif near_breakout and vol_ratio >= 1.1:
            state = "forming"
        else:
            return None

        score = vol_ratio * (1 + max(0.0, ret_63) / 100)
        reasons = [
            f"{'new' if breakout else 'approaching'} {self.LOOKBACK}-day high",
            f"volume {vol_ratio:.1f}x 20-day average",
            f"trend template intact (52w position {p52w:.0f}%)",
        ]
        return {
            "score": score,
            "state": state,
            "reasons": reasons,
            "breakout_level": round(prior_high, 2),
            "vol_ratio": round(vol_ratio, 2),
            "p52w": round(p52w, 1),
            "ret_63": round(ret_63, 2),
        }


class MeanReversionRSI2(Strategy):
    id = "mean_reversion"
    name = "Mean Reversion (RSI-2)"
    description = (
        "Connors-style dip buying: when a stock in a long-term uptrend (above "
        "its 200-day MA) gets washed out short-term — RSI(2) below 10 — buy "
        "the snap-back toward the 20-day mean. A close at/below the lower "
        "Bollinger Band (20, 2σ) confirms the stretch."
    )
    details = {
        "horizon": "Short swing: 3–10 day snap-back toward the 20-day mean.",
        "how": "Buys sharp short-term washouts inside long-term uptrends and sells the bounce back toward the mean.",
        "rules": [
            "Trend gate: price must be above its 200-day moving average (only buy dips in uptrends).",
            "Washout trigger: RSI(2) below 10 → 'triggered'.",
            "RSI(2) between 10 and 20 with a close at/below the lower Bollinger Band (20, 2σ) → 'forming'.",
            "Exit is implicit: once price reverts to the mean the name stops qualifying and drops out at the next rebalance.",
            "Backtest tip: use the Weekly (5-day) rebalance cadence — exits only register at rebalance, so longer cadences blur the short snap-back edge.",
        ],
        "scoring": "score = (20 − RSI2)/20 + max(0, z)/2   — deeper RSI(2) washouts and bigger stretches below the 20-day mean (z in σ) rank first.",
        "parameters": [
            ["Trend filter", "close > 200-day MA"],
            ["RSI period", "2 bars"],
            ["Trigger", "RSI(2) < 10"],
            ["Bollinger confirm", "20-day, 2σ lower band"],
            ["Min history", "220 bars"],
        ],
    }
    min_bars = 220
    RSI_TRIGGER = 10.0
    RSI_FORMING = 20.0
    regimes = ["choppy_calm"]

    def candidate(self, bars: list[dict], idx: int) -> dict | None:
        closes = _tail_closes(bars, idx, 260)
        if len(closes) < self.min_bars or closes[-1] <= 0:
            return None
        close = closes[-1]

        # Trend gate: only fade dips while the long-term uptrend is intact
        ma200 = _sma_at(closes, 200)
        if ma200 is None or close <= ma200:
            return None

        rsi2 = calc_rsi(closes[-30:], 2)[-1]
        if rsi2 is None:
            return None

        window20 = np.array(closes[-20:], dtype=float)
        mid = float(np.mean(window20))
        sd = float(np.std(window20))
        lower_band = mid - 2 * sd
        band_touch = sd > 0 and close <= lower_band
        z = (mid - close) / sd if sd > 0 else 0.0

        if rsi2 < self.RSI_TRIGGER:
            state = "triggered"
        elif rsi2 < self.RSI_FORMING and band_touch:
            state = "forming"
        else:
            return None

        score = (self.RSI_FORMING - rsi2) / self.RSI_FORMING + max(0.0, z) / 2

        ret_21 = (close / closes[-22] - 1) * 100 if len(closes) >= 22 else None
        ret_63 = (close / closes[-64] - 1) * 100 if len(closes) >= 64 else None
        reasons = [
            f"RSI(2) at {rsi2:.0f} — short-term washout",
            "above 200-day MA (long-term uptrend intact)",
        ]
        if band_touch:
            reasons.append("closed at/below the lower Bollinger Band (20, 2σ)")
        return {
            "score": score,
            "state": state,
            "reasons": reasons,
            "rsi": round(rsi2, 1),
            "zscore": round(z, 2),
            "dist_ma200": round((close / ma200 - 1) * 100, 2),
            "ret_21": round(ret_21, 2) if ret_21 is not None else None,
            "ret_63": round(ret_63, 2) if ret_63 is not None else None,
        }


class DualMomentum(Strategy):
    id = "dual_momentum"
    name = "Dual Momentum"
    description = (
        "Antonacci-style dual momentum: requires ABSOLUTE momentum (12-1 month "
        "total return positive — the market must be beating cash) before ranking "
        "by RELATIVE, risk-adjusted momentum (that return divided by its own "
        "volatility). Works in any trend direction since it's cash-gated, not "
        "direction-gated — the strongest trending regime strategy."
    )
    details = {
        "horizon": "Trend rotation; hold 1-3 months, re-rank on rebalance.",
        "how": "Skips names that haven't beaten cash over the last year, then ranks survivors by return-per-unit-of-risk rather than raw return.",
        "rules": [
            "12-1 month momentum: total return from 252 bars ago to 21 bars ago (skips the most recent month, which tends to mean-revert).",
            "Absolute momentum gate: that return must be > 0 — if it isn't beating cash, skip it entirely (this is what rotates a basket into bonds/cash in a bear market).",
            "Relative momentum score: 12-1 return divided by annualized daily-return volatility over the same window.",
            "Rank all candidates by score; hold the top N.",
        ],
        "scoring": "score = (P[t-21]/P[t-252] − 1) / annualized_vol(daily returns, 231-bar window)   — risk-adjusted 12-1 month momentum. Must be > 0.",
        "parameters": [
            ["Momentum window", "252 -> 21 bars ago (12-1 month)"],
            ["Absolute momentum gate", "12-1 return > 0"],
            ["Min history", "260 bars"],
        ],
    }
    min_bars = 260
    regimes = ["trending_bull", "trending_bear"]

    def candidate(self, bars: list[dict], idx: int) -> dict | None:
        closes = _tail_closes(bars, idx, 260)
        if len(closes) < 253:
            return None
        start_px = closes[-253]
        end_px = closes[-22]
        if start_px <= 0:
            return None
        mom_12_1 = end_px / start_px - 1
        if mom_12_1 <= 0:
            return None
        window = np.array(closes[-232:-1], dtype=float)
        rets = np.diff(window) / window[:-1]
        vol = float(np.std(rets) * math.sqrt(252)) if len(rets) > 1 else 0.0
        if vol <= 0:
            return None
        score = mom_12_1 / vol
        ret_63 = (closes[-1] / closes[-64] - 1) * 100 if len(closes) >= 64 else None
        return {
            "score": score,
            "state": "triggered",
            "reasons": [
                f"12-1 month momentum {mom_12_1 * 100:+.1f}% (beats cash)",
                f"risk-adjusted score {score:.2f}",
            ],
            "mom_12_1": round(mom_12_1 * 100, 2),
            "ann_vol": round(vol * 100, 1),
            "ret_63": round(ret_63, 2) if ret_63 is not None else None,
        }


class VolatilityBreakout(Strategy):
    id = "volatility_breakout"
    name = "Volatility Breakout (Donchian)"
    description = (
        "Turtle-style Donchian channel breakout: buy a close above the prior "
        "20-day high, confirmed by ATR expansion (volatility waking up, not "
        "just a single noisy print). Distinct from Breakout+Volume — this "
        "triggers earlier, off a shorter channel, and confirms with "
        "volatility rather than share volume."
    )
    details = {
        "horizon": "Catches an emerging trend early; 2-6 week hold.",
        "how": "Buys the first close above a rolling 20-day high once ATR shows volatility is expanding, which is when a range starts becoming a trend.",
        "rules": [
            "Donchian breakout: close >= highest high of the prior 20 bars.",
            "Volatility expansion confirm: current ATR(14) > its own 20-day average ATR (vol is rising, not fading).",
            "Trend filter: close above the 50-day MA (skip breakouts fighting the intermediate trend).",
        ],
        "scoring": "score = (close/prior_20d_high − 1) × (ATR/avg_ATR_20)   — rewards a cleaner breakout thrust with more volatility expansion behind it.",
        "parameters": [
            ["Donchian channel", "20-day high"],
            ["Volatility confirm", "ATR(14) > 20-day avg ATR"],
            ["Trend filter", "close > 50-day MA"],
            ["Min history", "90 bars"],
        ],
    }
    min_bars = 90
    CHANNEL = 20
    regimes = ["trending_bull"]

    def candidate(self, bars: list[dict], idx: int) -> dict | None:
        lo = max(0, idx + 1 - 120)
        window = bars[lo: idx + 1]
        if len(window) < self.min_bars:
            return None
        closes = [b["close"] for b in window]
        highs = [b["high"] for b in window]
        lows = [b["low"] for b in window]
        close = closes[-1]

        ma50 = _sma_at(closes, 50)
        if ma50 is None or close <= ma50:
            return None

        prior_high = max(highs[-(self.CHANNEL + 1):-1])
        if close < prior_high:
            return None

        atr = calc_atr(highs, lows, closes, 14)
        valid_atr = [a for a in atr if a is not None]
        if len(valid_atr) < 20 or not valid_atr[-1]:
            return None
        atr_now = valid_atr[-1]
        avg_atr20 = float(np.mean(valid_atr[-20:]))
        if avg_atr20 <= 0 or atr_now <= avg_atr20:
            return None

        breakout_pct = close / prior_high - 1
        vol_ratio = atr_now / avg_atr20
        score = breakout_pct * vol_ratio
        if score <= 0:
            return None
        return {
            "score": score,
            "state": "triggered",
            "reasons": [
                f"new {self.CHANNEL}-day Donchian high",
                f"ATR expanding ({vol_ratio:.2f}x 20-day avg)",
                "above 50-day MA",
            ],
            "breakout_level": round(prior_high, 2),
            "vol_expansion": round(vol_ratio, 2),
        }


SECTOR_ETFS = {"XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLB", "XLC"}


class SectorRotation(Strategy):
    id = "sector_rotation"
    name = "Sector Rotation"
    description = (
        "Ranks the 11 SPDR sector ETFs by smooth 63-day momentum (same "
        "regression-slope-x-R² idea as Momentum Rotation, shorter window) and "
        "rotates into the strongest sectors. Reads leadership rotation across "
        "the market rather than picking individual stocks — useful when "
        "single names are noisy but sector flows are clear."
    )
    details = {
        "horizon": "Sector-level rotation; hold 1-2 months, re-rank monthly.",
        "how": "Fits a trend line to each sector ETF's last 63 days and ranks by how strong AND smooth that trend is, same as Momentum Rotation but tuned to sector-speed moves.",
        "rules": [
            "Universe restricted to the 11 SPDR sector ETFs (XLK, XLF, XLV, XLE, XLI, XLY, XLP, XLU, XLRE, XLB, XLC).",
            "Trend filter: sector price above its own 50-day MA.",
            "Fit a linear regression to the last 63 daily log-closes.",
            "Annualize the slope and weight by R² (fit quality), exactly like Momentum Rotation.",
        ],
        "scoring": "score = (exp(slope × 252) − 1) × R²  over a 63-day window — same formula as Momentum Rotation, shorter lookback for faster sector rotation.",
        "parameters": [
            ["Universe", "11 SPDR sector ETFs only"],
            ["Regression window", "63 bars"],
            ["Trend filter", "above 50-day MA"],
            ["Min history", "80 bars"],
        ],
    }
    min_bars = 80
    REGRESSION_BARS = 63
    regimes = ["trending_bull", "choppy_calm"]

    def applies_to(self, ticker: str) -> bool:
        return ticker in SECTOR_ETFS

    def candidate(self, bars: list[dict], idx: int) -> dict | None:
        closes = _tail_closes(bars, idx, self.REGRESSION_BARS + 20)
        if len(closes) < self.min_bars or closes[-1] <= 0:
            return None
        window = closes[-self.REGRESSION_BARS:]
        if any(c <= 0 for c in window):
            return None
        ma50 = _sma_at(closes, 50)
        if ma50 is None or closes[-1] <= ma50:
            return None

        y = np.log(np.array(window))
        x = np.arange(len(window), dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        y_fit = slope * x + intercept
        ss_res = float(np.sum((y - y_fit) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        annualized = math.exp(slope * 252) - 1
        score = annualized * max(0.0, r2)
        if score <= 0:
            return None
        ret_21 = (closes[-1] / closes[-22] - 1) * 100 if len(closes) >= 22 else None
        return {
            "score": score,
            "state": "triggered",
            "reasons": [
                f"63D sector trend annualizes to {annualized * 100:+.0f}%",
                f"smooth trend (R² {r2:.2f})",
                "leading sector, above 50-day MA",
            ],
            "slope_ann": round(annualized * 100, 1),
            "r2": round(r2, 3),
            "ret_21": round(ret_21, 2) if ret_21 is not None else None,
        }


class BearReversalWatch(Strategy):
    id = "bear_reversal_watch"
    name = "Bear Reversal (Watchlist)"
    description = (
        "Warning signal, not a buy signal: RSI(2) above 90 (a sharp, "
        "short-term overbought bounce) inside a confirmed downtrend (below "
        "the 200-day MA). These bounces statistically fail more often than "
        "they continue. Shown watchlist-only — this app is long-only, so "
        "there's no trade plan for it, only a flag to avoid chasing the "
        "bounce or to tighten stops on anything already held."
    )
    details = {
        "horizon": "Informational only — no position is opened from this signal.",
        "how": "Flags stocks that just had a sharp bounce while still in a confirmed downtrend, which is the classic setup for the bounce to fail.",
        "rules": [
            "Trend gate: price below its 200-day moving average (confirmed downtrend).",
            "Trigger: RSI(2) above 90 — a stretched, short-term overbought bounce.",
            "No trade plan is generated; this only ever appears as a watchlist caution.",
        ],
        "scoring": "score = (RSI2 − 90) / 10 + max(0, dist_below_200ma)/20   — bigger overbought stretch inside a deeper downtrend ranks first.",
        "parameters": [
            ["Trend filter", "close < 200-day MA"],
            ["RSI period", "2 bars"],
            ["Trigger", "RSI(2) > 90"],
            ["Min history", "220 bars"],
        ],
    }
    min_bars = 220
    regimes = ["trending_bear", "choppy_volatile"]
    actionable = False

    def candidate(self, bars: list[dict], idx: int) -> dict | None:
        closes = _tail_closes(bars, idx, 260)
        if len(closes) < self.min_bars or closes[-1] <= 0:
            return None
        close = closes[-1]
        ma200 = _sma_at(closes, 200)
        if ma200 is None or close >= ma200:
            return None
        rsi2 = calc_rsi(closes[-30:], 2)[-1]
        if rsi2 is None or rsi2 <= 90:
            return None
        dist_below = (ma200 - close) / ma200 * 100
        score = (rsi2 - 90) / 10 + max(0.0, dist_below) / 20
        return {
            "score": score,
            "state": "triggered",
            "reasons": [
                f"RSI(2) at {rsi2:.0f} — sharp overbought bounce",
                f"below 200-day MA by {dist_below:.1f}% (downtrend intact)",
                "watchlist only — bounces like this fail more than they continue",
            ],
            "rsi": round(rsi2, 1),
            "dist_below_ma200": round(dist_below, 2),
        }


class LowVolTrend(Strategy):
    id = "low_vol_trend"
    name = "Low-Volatility Trend"
    description = (
        "Defensive tilt for choppy or high-VIX markets: still requires an "
        "uptrend (above the 200-day MA), but ranks candidates by the LOWEST "
        "realized volatility instead of the fastest movers. Trades a smaller "
        "edge for a much steadier ride when the broad tape is unstable."
    )
    details = {
        "horizon": "Defensive core holding; hold as long as the trend and low-vol profile persist.",
        "how": "Filters to names still in an uptrend, then prefers the calmest ones instead of the strongest movers — the opposite bias from momentum strategies.",
        "rules": [
            "Trend gate: price above its 200-day moving average.",
            "Realized volatility: annualized stdev of daily returns over the last 63 bars.",
            "Rank ascending by volatility among trend-qualified names (lowest vol first).",
        ],
        "scoring": "score = 1 / annualized_vol(daily returns, 63-bar window)   — only computed for names above their 200-day MA; higher score = calmer trend.",
        "parameters": [
            ["Trend filter", "close > 200-day MA"],
            ["Volatility window", "63 bars"],
            ["Min history", "220 bars"],
        ],
    }
    min_bars = 220
    regimes = ["choppy_calm", "choppy_volatile"]

    def candidate(self, bars: list[dict], idx: int) -> dict | None:
        closes = _tail_closes(bars, idx, 260)
        if len(closes) < self.min_bars or closes[-1] <= 0:
            return None
        close = closes[-1]
        ma200 = _sma_at(closes, 200)
        if ma200 is None or close <= ma200:
            return None
        window = np.array(closes[-64:], dtype=float)
        if len(window) < 64:
            return None
        rets = np.diff(window) / window[:-1]
        vol = float(np.std(rets) * math.sqrt(252))
        if vol <= 0:
            return None
        score = 1 / vol
        ret_63 = (close / closes[-64] - 1) * 100 if len(closes) >= 64 else None
        return {
            "score": score,
            "state": "triggered",
            "reasons": [
                f"low realized volatility ({vol * 100:.1f}% annualized)",
                "above 200-day MA (uptrend intact)",
            ],
            "ann_vol": round(vol * 100, 1),
            "ret_63": round(ret_63, 2) if ret_63 is not None else None,
        }


STRATEGIES: dict[str, Strategy] = {
    s.id: s
    for s in (
        MomentumRotation(), Pullback50MA(), BreakoutVolume(), MeanReversionRSI2(),
        DualMomentum(), VolatilityBreakout(), SectorRotation(), BearReversalWatch(),
        LowVolTrend(),
    )
}


def strategy_catalog() -> list[dict]:
    from services.regime import QUADRANT_INFO
    return [
        {
            "id": s.id,
            "name": s.name,
            "description": s.description,
            "details": s.details,
            "regimes": s.regimes,
            "regime_labels": [QUADRANT_INFO[q]["label"] for q in s.regimes if q in QUADRANT_INFO],
            "actionable": s.actionable,
        }
        for s in STRATEGIES.values()
    ]
