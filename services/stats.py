"""
Inference helpers for the evidence layer (ML4T 3e, Ch 6 / §7.4 / §16.7).

Pure functions, standard library only — the add-on image has no scipy, and
none of this is heavy enough to need it.

WHY: a backtest statistic is a point estimate from one path of history. Before
the app acts on one (the edge matrix's verdicts un-tick Weekly Plan buys, and
the evidence override can switch strategies on and off) it must ask two
questions the raw numbers cannot answer:

  * Is it distinguishable from noise, given serial dependence between
    periods? -> `mean_test` (t-test with an AR(1) effective sample size)
  * Is it still distinguishable once we admit how many things were tried?
    -> `bh_adjust` (Benjamini-Hochberg across the matrix's cells) and
       `deflated_sharpe` (Bailey & Lopez de Prado, across Strategy Lab runs)
"""
from __future__ import annotations

import math
from statistics import NormalDist, mean, pstdev, stdev

_N = NormalDist()
EULER_GAMMA = 0.5772156649015329

# Lag-1 autocorrelation is clipped to this range before it shrinks the
# effective sample size. Negative autocorrelation would INFLATE n_eff above n;
# we never let dependence make a sample look bigger than it is.
AR1_RHO_MAX = 0.9


# ── Student t distribution (via the regularized incomplete beta) ────────────
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Numerical Recipes 6.4)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c
        c = c if abs(c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c
        c = c if abs(c) > 1e-300 else 1e-300
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                + a * math.log(x) + b * math.log(1.0 - x))
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: float) -> float:
    """Two-sided p-value of a Student t statistic."""
    if not math.isfinite(t):
        return 0.0
    if df <= 0:
        return 1.0
    return max(0.0, min(1.0, _betainc(df / 2.0, 0.5, df / (df + t * t))))


# ── serial dependence ──────────────────────────────────────────────────────
def lag1_autocorr(xs: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    m = mean(xs)
    den = sum((x - m) ** 2 for x in xs)
    if den <= 0:
        return 0.0
    return sum((xs[i] - m) * (xs[i - 1] - m) for i in range(1, n)) / den


def effective_n(xs: list[float]) -> tuple[float, float]:
    """(n_eff, rho) under an AR(1) approximation: n·(1−ρ)/(1+ρ), ρ clipped to
    [0, AR1_RHO_MAX]. Consecutive holding periods share positions and regimes,
    so they are not independent draws; this is the conservative first cut
    (ML4T gap 3 refines it with episode-level clustering)."""
    rho = max(0.0, min(AR1_RHO_MAX, lag1_autocorr(xs)))
    return len(xs) * (1.0 - rho) / (1.0 + rho), rho


def mean_test(xs: list[float]) -> dict:
    """Is the mean of `xs` distinguishable from zero?

    One-sample t-test whose sample size is the AR(1) effective n, so a run of
    correlated periods is not counted as many independent observations.
    Returns {t_stat, p_value, n_eff, rho}; p_value is two-sided, 1.0 when
    there is nothing to test."""
    n = len(xs)
    if n < 3:
        return {"t_stat": None, "p_value": 1.0, "n_eff": float(n), "rho": 0.0}
    n_eff, rho = effective_n(xs)
    sd = stdev(xs)
    if sd <= 0:
        m = mean(xs)
        return {"t_stat": None, "p_value": 1.0 if m == 0 else 0.0, "n_eff": n_eff, "rho": rho}
    t = mean(xs) / (sd / math.sqrt(n_eff))
    return {"t_stat": t, "p_value": t_two_sided_p(t, max(1.0, n_eff - 1.0)),
            "n_eff": n_eff, "rho": rho}


# ── multiple testing ───────────────────────────────────────────────────────
def bh_adjust(pvalues: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (q-values), same order as input.

    A cell is a discovery at false-discovery rate q when its adjusted value is
    <= q. Controls the expected share of false "edges" among those declared."""
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [1.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, pvalues[i] * m / rank)
        adj[i] = min(1.0, running)
    return adj


# ── Sharpe ratio inference ─────────────────────────────────────────────────
def moments(xs: list[float]) -> tuple[float, float]:
    """(skewness, kurtosis) — kurtosis NOT excess (normal = 3)."""
    n = len(xs)
    if n < 3:
        return 0.0, 3.0
    m = mean(xs)
    sd = pstdev(xs)
    if sd <= 0:
        return 0.0, 3.0
    skew = sum(((x - m) / sd) ** 3 for x in xs) / n
    kurt = sum(((x - m) / sd) ** 4 for x in xs) / n
    return skew, kurt


def sharpe_se(sr: float, n: int, skew: float, kurt: float) -> float:
    """Standard error of a per-period Sharpe ratio under non-normal returns
    (Mertens 2002; the denominator of the Probabilistic Sharpe Ratio)."""
    var = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    return math.sqrt(max(var, 1e-12) / max(1, n - 1))


def probabilistic_sharpe(sr: float, sr_benchmark: float, n: int, skew: float, kurt: float) -> float:
    """P(true Sharpe > sr_benchmark) — all Sharpes PER PERIOD, not annualized."""
    if n < 3:
        return 0.5
    return _N.cdf((sr - sr_benchmark) / sharpe_se(sr, n, skew, kurt))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum per-period Sharpe among `n_trials` unskilled trials
    whose Sharpes have variance `sr_variance` (Bailey & Lopez de Prado 2014).
    Zero for a single trial: nothing was selected, so there is no deflation."""
    if n_trials <= 1 or sr_variance <= 0:
        return 0.0
    n = float(n_trials)
    return math.sqrt(sr_variance) * (
        (1.0 - EULER_GAMMA) * _N.inv_cdf(1.0 - 1.0 / n)
        + EULER_GAMMA * _N.inv_cdf(1.0 - 1.0 / (n * math.e))
    )


def deflated_sharpe(sr: float, n: int, skew: float, kurt: float,
                    n_trials: int, sr_variance: float) -> float:
    """Deflated Sharpe Ratio: the probability the per-period Sharpe `sr` beats
    what the best of `n_trials` skill-less tries would show by luck."""
    return probabilistic_sharpe(sr, expected_max_sharpe(n_trials, sr_variance), n, skew, kurt)
