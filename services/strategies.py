"""
Swing-trading strategy framework.

Each strategy implements two hooks sharing the same rules:
  - candidate(bars, idx): backtest hook — evaluate at historical bar `idx`
    using only bars[:idx+1]. Returns {"score": float, ...display metrics} or
    None when the setup's filters fail.
  - scan(ticker, bars, quote): live hook — evaluate the latest bar and return
    a Setup with explainable reasons, or None.

All three strategies target 2-week to 2-month holds on daily bars:
  momentum_rotation — Clenow-style exponential regression momentum ranking
  pullback_50ma     — buy pullbacks to the 50-day MA inside an uptrend
  breakout_volume   — 60-day-high breakouts confirmed by volume expansion
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from services.indicators import calc_rsi


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

    @abstractmethod
    def candidate(self, bars: list[dict], idx: int) -> dict | None:
        """Score this ticker at bar `idx` using bars[:idx+1] only."""

    def scan(self, ticker: str, bars: list[dict], quote: dict) -> Setup | None:
        """Live evaluation of the latest bar. Default wraps candidate()."""
        if len(bars) < self.min_bars:
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


STRATEGIES: dict[str, Strategy] = {
    s.id: s for s in (MomentumRotation(), Pullback50MA(), BreakoutVolume())
}


def strategy_catalog() -> list[dict]:
    return [
        {"id": s.id, "name": s.name, "description": s.description, "details": s.details}
        for s in STRATEGIES.values()
    ]
