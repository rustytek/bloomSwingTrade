"""
Pure-function, no-network regression tests for the calculation fixes in
services/indicators.py, services/market_data.py and services/strategies.py.

Covers (one test group per fixed bug):
  1. p52w uses the last 252 bars, not the whole (5-year) cache
  2. calc_adx +DI/-DI alignment (di_idx = idx - period), vs an independent Wilder calc
  3. gc/dc are crossover EVENTS; ma_state carries the trend STATE
  4. compute_performance_metrics returns ann_ret (full history) AND ann_ret_1m,
     both annualized over n-1 return periods
  5. earn_beat is gone from the quote dict and from compute_score
  6. vol_r is today-vs-prior-20; vol_r_5d preserves the old smoothed ratio
  7. dividendYield unit auto-detection (fraction vs already-percent)
  8. DualMomentum volatility window matches the documented 12-1 span
  9. VolatilityBreakout ATR average excludes today
 10. schema_v == 2 stamped on enriched quotes; old cached quotes are stale

Run with:
    python test_indicators.py

No network, no DB, no HTTP. Deterministic (randomness seeded).
"""
from __future__ import annotations

import math
import random
import traceback

import numpy as np

# Settings uses extra="ignore", so a stale .env key can no longer hard-fail
# the import chain (see config.py). No global model_config override here: it
# would leak into every other test module in a shared pytest process.

from services.indicators import (  # noqa: E402
    calc_adx, compute_performance_metrics, compute_score,
)
import services.market_data as md  # noqa: E402
from services.strategies import STRATEGIES  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────
# Tiny test runner (same pattern as test_passes.py)
# ──────────────────────────────────────────────────────────────────────────
_TESTS = []


def test(fn):
    _TESTS.append(fn)
    return fn


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


# ──────────────────────────────────────────────────────────────────────────
# Synthetic bar builders (deterministic)
# ──────────────────────────────────────────────────────────────────────────
def _bar(o, h, l, c, v=1_000_000, i=0, date="2024-01-01"):
    return {"i": i, "date": date, "open": o, "high": h, "low": l, "close": c, "vol": v}


def make_bars(n: int, seed: int = 1, start: float = 100.0, drift: float = 0.0005,
              vol_base: int = 1_000_000) -> list[dict]:
    rng = random.Random(seed)
    bars = []
    price = start
    for i in range(n):
        price *= (1 + drift + rng.uniform(-0.01, 0.01))
        close = price
        open_ = close * (1 - drift * 0.5)
        high = max(open_, close) * 1.006
        low = min(open_, close) * 0.994
        bars.append(_bar(open_, high, low, close, v=vol_base, i=i))
    return bars


# ──────────────────────────────────────────────────────────────────────────
# BUG 1 — p52w over the last 252 bars only
# ──────────────────────────────────────────────────────────────────────────
@test
def test_p52w_uses_last_252_bars_only():
    """A huge spike 3 years back must not widen today's 52-week range.

    Bars 0..999 build a 5-year cache. The oldest 700 bars contain a 300.0 high
    and a 10.0 low; the most recent 252 bars live in a tight 100..110 band with
    price near the bottom. True 52w position must be low (~2%), not ~30%.
    """
    bars = []
    # Ancient era: enormous range that must be ignored
    for i in range(700):
        c = 300.0 if i % 2 == 0 else 10.0
        bars.append(_bar(c, 300.0, 10.0, c, i=i))
    # Recent 252 bars: tight 100..110 range, ending near the low
    for j in range(252):
        c = 110.0 if j < 251 else 100.2
        bars.append(_bar(c, 110.0, 100.0, c, i=700 + j))

    quote = {"ticker": "TEST", "price": 100.2, "beta": None}
    enriched = md._enrich_with_technicals(quote, bars)
    p52w = enriched["p52w"]
    # (100.2 - 100) / (110 - 100) * 100 = 2.0
    assert approx(p52w, 2.0, tol=0.05), f"expected ~2.0 from the 252-bar range, got {p52w}"
    # And the buggy 5-year version would have been (100.2-10)/(300-10)*100 = 31.1
    assert p52w < 10, f"p52w still leaking the full history: {p52w}"


@test
def test_p52w_falls_back_to_full_window_when_short():
    bars = [_bar(c, c + 1, c - 1, c, i=i) for i, c in enumerate([10.0, 20.0, 30.0])]
    enriched = md._enrich_with_technicals({"ticker": "T", "price": 30.0}, bars)
    # full window: high 31, low 9 -> (30-9)/(31-9)*100 = 95.45
    assert approx(enriched["p52w"], 95.5, tol=0.1), enriched["p52w"]


# ──────────────────────────────────────────────────────────────────────────
# BUG 2 — calc_adx +DI / -DI alignment
# ──────────────────────────────────────────────────────────────────────────
def _independent_wilder_di(highs, lows, closes, period=14):
    """Independent Wilder +DI/-DI for the FINAL bar, written from the
    definition rather than reusing calc_adx's internals."""
    n = len(closes)
    tr_sum = plus_sum = minus_sum = 0.0
    trs, pdms, mdms = [], [], []
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        dn = lows[i - 1] - lows[i]
        pdms.append(up if (up > dn and up > 0) else 0.0)
        mdms.append(dn if (dn > up and dn > 0) else 0.0)
        trs.append(max(highs[i] - lows[i],
                       abs(highs[i] - closes[i - 1]),
                       abs(lows[i] - closes[i - 1])))
    tr_sum = sum(trs[:period])
    plus_sum = sum(pdms[:period])
    minus_sum = sum(mdms[:period])
    for k in range(period, len(trs)):
        tr_sum = tr_sum - tr_sum / period + trs[k]
        plus_sum = plus_sum - plus_sum / period + pdms[k]
        minus_sum = minus_sum - minus_sum / period + mdms[k]
    if tr_sum <= 0:
        return None, None
    return 100 * plus_sum / tr_sum, 100 * minus_sum / tr_sum


@test
def test_adx_di_final_bar_matches_independent_wilder():
    bars = make_bars(300, seed=9, drift=0.001)
    highs = [b["high"] for b in bars]
    lows = [b["low"] for b in bars]
    closes = [b["close"] for b in bars]
    period = 14

    res = calc_adx(highs, lows, closes, period)
    exp_plus, exp_minus = _independent_wilder_di(highs, lows, closes, period)

    assert res["plus_di"] is not None, "+DI at the final bar must not be None (was, pre-fix)"
    assert res["minus_di"] is not None, "-DI at the final bar must not be None (was, pre-fix)"
    assert approx(res["plus_di"], exp_plus, tol=1e-9), (
        f"+DI {res['plus_di']} != independent Wilder {exp_plus}"
    )
    assert approx(res["minus_di"], exp_minus, tol=1e-9), (
        f"-DI {res['minus_di']} != independent Wilder {exp_minus}"
    )
    # The last series element must be the last bar (no trailing None DI gap)
    assert res["series"][-1] is not None
    assert approx(res["series"][-1]["plus_di"], exp_plus, tol=1e-9)


@test
def test_adx_first_valid_index_unchanged_and_no_lookahead():
    """ADX still first appears at 2*period-1, and the DI reported there is the
    DI computed from bars[:2*period], not a future bar's value."""
    bars = make_bars(120, seed=4)
    highs = [b["high"] for b in bars]
    lows = [b["low"] for b in bars]
    closes = [b["close"] for b in bars]
    period = 14
    series = calc_adx(highs, lows, closes, period)["series"]

    first_idx = next(i for i, v in enumerate(series) if v is not None)
    assert first_idx == 2 * period - 1, f"first ADX index {first_idx} != {2 * period - 1}"

    # No look-ahead: truncating the input right after the first valid bar must
    # not change the values reported AT that bar. (calc_adx needs 2*period+1
    # bars before it emits anything, so truncate to first_idx + 2.)
    cut = first_idx + 2
    trunc = calc_adx(highs[:cut], lows[:cut], closes[:cut], period)["series"]
    assert trunc[first_idx] is not None
    assert approx(trunc[first_idx]["plus_di"], series[first_idx]["plus_di"], tol=1e-9), (
        "DI at the first valid bar depends on future bars — look-ahead"
    )
    assert approx(trunc[first_idx]["minus_di"], series[first_idx]["minus_di"], tol=1e-9)
    assert approx(trunc[first_idx]["adx"], series[first_idx]["adx"], tol=1e-9)


# ──────────────────────────────────────────────────────────────────────────
# BUG 3 — gc/dc are EVENTS, ma_state is the STATE
# ──────────────────────────────────────────────────────────────────────────
def _cross_bars(n_down: int, n_up: int) -> list[dict]:
    """Long decline then a sharp rally, so MA50 crosses above MA200 near the end."""
    bars = []
    price = 200.0
    i = 0
    for _ in range(n_down):
        price *= (1 - 0.004)
        bars.append(_bar(price, price * 1.002, price * 0.998, price, i=i)); i += 1
    for _ in range(n_up):
        price *= (1 + 0.022)
        bars.append(_bar(price, price * 1.002, price * 0.998, price, i=i)); i += 1
    return bars


@test
def test_gc_dc_are_events_not_states():
    """A long, boring uptrend with no recent cross must report BOTH gc and dc
    False while ma_state says 'bull'. Pre-fix, gc was always True here."""
    bars = make_bars(600, seed=5, drift=0.0025)
    enriched = md._enrich_with_technicals({"ticker": "T", "price": bars[-1]["close"]}, bars)
    assert enriched["ma_state"] == "bull", enriched["ma_state"]
    assert enriched["gc_event"] is False, "no cross in the last 5 bars, gc_event must be False"
    assert enriched["dc_event"] is False
    assert enriched["gc"] is enriched["gc_event"], "gc must alias gc_event"
    assert enriched["dc"] is enriched["dc_event"], "dc must alias dc_event"
    assert not (enriched["gc"] and enriched["dc"]), "gc and dc can never both be True"


@test
def test_scores_read_ma_state_not_the_event_flags():
    """compute_score / compute_swing_score grade the TREND STATE. With gc/dc now
    being events, a long-running bull trend (gc_event False) must still earn the
    trend points, and ma_state must be what decides them."""
    from services.indicators import compute_swing_score

    bull = {"rsi": 50, "vs_ma200": 12, "ma_state": "bull",
            "gc": False, "dc": False, "vol_r": 1.5, "p52w": 80, "chg_pct": 2.0}
    bear = dict(bull, ma_state="bear")
    assert compute_score(bull)["t"] > compute_score(bear)["t"], (
        "ma_state must drive the Technical trend points"
    )
    # Equivalence with the old state-style flags (pre-v2 cached quotes)
    legacy_bull = {k: v for k, v in bull.items() if k != "ma_state"}
    legacy_bull["gc"] = True
    assert compute_score(legacy_bull)["t"] == compute_score(bull)["t"]

    assert compute_swing_score(bull)["components"]["trend"] > \
        compute_swing_score(bear)["components"]["trend"]


@test
def test_gc_event_fires_within_5_bars_of_a_real_cross():
    """Find the exact bar where MA50 crosses above MA200, then confirm gc_event
    is True on that bar and stays True for 5 bars, and is False on bar +6."""
    from services.indicators import calc_ma

    bars = _cross_bars(400, 120)
    closes = [b["close"] for b in bars]
    ma50 = calc_ma(closes, 50)
    ma200 = calc_ma(closes, 200)
    cross_idx = None
    for i in range(200, len(closes)):
        if ma50[i] is not None and ma200[i] is not None and ma50[i - 1] is not None:
            if ma50[i] > ma200[i] and ma50[i - 1] <= ma200[i - 1]:
                cross_idx = i
                break
    assert cross_idx is not None, "test fixture failed to produce a golden cross"

    def at(idx):
        sub = bars[: idx + 1]
        return md._enrich_with_technicals({"ticker": "T", "price": sub[-1]["close"]}, sub)

    on_cross = at(cross_idx)
    assert on_cross["gc_event"] is True, "gc_event must fire on the cross bar"
    assert on_cross["ma_state"] == "bull"

    assert at(cross_idx + 4)["gc_event"] is True, "gc_event must persist 5 bars"
    later = at(cross_idx + 6)
    assert later["gc_event"] is False, "gc_event must expire after 5 bars"
    assert later["ma_state"] == "bull", "state stays bull even after the event expires"


# ──────────────────────────────────────────────────────────────────────────
# BUG 4 — ann_ret (full history) vs ann_ret_1m, over n-1 periods
# ──────────────────────────────────────────────────────────────────────────
@test
def test_ann_ret_is_full_history_and_uses_n_minus_1():
    # 253 prices == 252 return periods == exactly one year -> ann_ret == total_ret
    closes = [100.0 * (1.001 ** i) for i in range(253)]
    m = compute_performance_metrics(closes)
    total = (closes[-1] - closes[0]) / closes[0] * 100
    assert approx(m["ann_ret"], round(total, 2), tol=0.02), (
        f"ann_ret {m['ann_ret']} should equal the 1-year total return {total:.2f}"
    )
    assert m["ann_ret_1m"] is not None

    # Same prices but with a distinctly different final month — the two keys
    # must then report visibly different numbers.
    mixed = list(closes) + [closes[-1] * (0.99 ** k) for k in range(1, 22)]
    m2 = compute_performance_metrics(mixed)
    assert m2["ann_ret_1m"] != m2["ann_ret"], "the two windows must be reported separately"
    assert m2["ann_ret_1m"] < 0 < m2["ann_ret"]

    # ann_ret_1m annualizes 21 prices == 20 return periods
    total_1m = (mixed[-1] - mixed[-21]) / mixed[-21]
    expect_1m = ((1 + total_1m) ** (252 / 20) - 1) * 100
    assert approx(m2["ann_ret_1m"], round(expect_1m, 2), tol=0.02), (
        f"ann_ret_1m {m2['ann_ret_1m']} != n-1 annualization {expect_1m:.2f}"
    )


@test
def test_ann_ret_sign_agrees_with_sharpe():
    """The reported ann_ret must be the SAME number Sharpe was computed from.
    Pre-fix a down last month could report ann_ret < 0 alongside sharpe > 0."""
    rng = random.Random(99)
    closes = [100.0]
    for _ in range(500):                       # long, strongly positive history
        closes.append(closes[-1] * (1 + 0.0025 + rng.uniform(-0.002, 0.002)))
    for _ in range(21):                        # sharp final month drawdown
        closes.append(closes[-1] * (1 - 0.01))

    m = compute_performance_metrics(closes)
    assert m["ann_ret"] > 0, f"full-history ann_ret should be positive, got {m['ann_ret']}"
    assert m["ann_ret_1m"] < 0, f"1M ann_ret should be negative, got {m['ann_ret_1m']}"
    assert m["sharpe"] is not None
    # Reconstruct sharpe from the reported ann_ret + vol and confirm they match.
    rebuilt = (m["ann_ret"] / 100 - 0.05) / (m["vol"] / 100)
    assert approx(rebuilt, m["sharpe"], tol=0.02), (
        f"sharpe {m['sharpe']} not reproducible from reported ann_ret {m['ann_ret']}"
    )


@test
def test_perf_metrics_keys_preserved():
    closes = [100.0 * (1.001 ** i) for i in range(300)]
    m = compute_performance_metrics(closes)
    for key in ("ann_ret", "ann_ret_1m", "vol", "sharpe", "gain_sharpe", "sortino",
                "calmar", "info_ratio", "treynor", "vol_1m", "max_dd_1m"):
        assert key in m, f"missing key {key}"
    assert m["gain_sharpe"] == m["sortino"], "legacy gain_sharpe must still mirror sortino"
    # Empty-result shape must carry the same keys
    empty = compute_performance_metrics([])
    assert "ann_ret_1m" in empty and empty["ann_ret_1m"] is None


# ──────────────────────────────────────────────────────────────────────────
# BUG 5 — earn_beat removed
# ──────────────────────────────────────────────────────────────────────────
@test
def test_earn_beat_removed_from_quote_and_score():
    import inspect
    src = inspect.getsource(md._fetch_quote_sync)
    assert "earningsBeat" not in src, "quote dict still reads yfinance earningsBeat"

    score_src = inspect.getsource(compute_score)
    assert "earn_beat = stock.get" not in score_src, "compute_score still reads earn_beat"

    # An injected earn_beat must no longer change the score.
    base = {"chg_pct": 2.0, "vol_r": 1.5, "p52w": 80, "rsi": 50,
            "vs_ma200": 12, "gc": True, "dc": False, "pe": 10, "pm": 30,
            "fcf_pos": True, "debt_eq": 0.2}
    with_flag = dict(base, earn_beat=True)
    assert compute_score(base) == compute_score(with_flag), (
        "earn_beat still influences compute_score"
    )


# ──────────────────────────────────────────────────────────────────────────
# BUG 6 — vol_r single-bar, vol_r_5d preserved
# ──────────────────────────────────────────────────────────────────────────
@test
def test_vol_r_is_today_vs_prior_20_and_vol_r_5d_preserved():
    bars = make_bars(300, seed=12, vol_base=1_000_000)
    # Make today's volume a clean 3x the prior 20-day average of 1,000,000.
    bars[-1]["vol"] = 3_000_000
    enriched = md._enrich_with_technicals({"ticker": "T", "price": bars[-1]["close"]}, bars)

    assert approx(enriched["vol_r"], 3.0, tol=0.01), (
        f"vol_r should be today/prior-20-avg = 3.0, got {enriched['vol_r']}"
    )
    # Old smoothed ratio: avg(last 5) / avg(last 20) = 1.4M/1.1M = 1.2727...
    assert approx(enriched["vol_r_5d"], 1.27, tol=0.01), enriched["vol_r_5d"]
    assert enriched["vol_r"] != enriched["vol_r_5d"], "the two ratios must differ here"


@test
def test_vol_r_guards_zero_and_short_volume():
    bars = make_bars(300, seed=13, vol_base=0)
    enriched = md._enrich_with_technicals({"ticker": "T", "price": bars[-1]["close"]}, bars)
    assert enriched["vol_r"] == 1.0, "zero denominator must fall back to 1.0"
    assert enriched["vol_r_5d"] == 1.0

    short = make_bars(3, seed=14)
    e2 = md._enrich_with_technicals({"ticker": "T", "price": short[-1]["close"]}, short)
    assert isinstance(e2["vol_r"], float) and e2["vol_r"] > 0


# ──────────────────────────────────────────────────────────────────────────
# BUG 7 — dividendYield unit auto-detection
# ──────────────────────────────────────────────────────────────────────────
def _scale_div(raw):
    """Mirrors the branch in _fetch_quote_sync — kept in lockstep by the
    source-text assertion below."""
    return raw * 100 if raw <= 1.0 else raw


@test
def test_div_yield_unit_autodetect():
    import inspect
    src = inspect.getsource(md._fetch_quote_sync)
    assert "div_yield * 100 if div_yield <= 1.0 else div_yield" in src, (
        "dividendYield scaling branch changed — update this test"
    )
    # Old-style fraction
    assert approx(_scale_div(0.0132), 1.32, tol=1e-9)
    # New-style already-percent
    assert approx(_scale_div(1.32), 1.32, tol=1e-9)
    assert approx(_scale_div(4.5), 4.5, tol=1e-9)
    # Boundary: exactly 1.0 treated as a fraction (a 1% yield either way is far
    # more likely than a 1.0 that is already a percent at the boundary)
    assert approx(_scale_div(1.0), 100.0, tol=1e-9)


# ──────────────────────────────────────────────────────────────────────────
# BUG 8 — DualMomentum volatility window == the 12-1 span
# ──────────────────────────────────────────────────────────────────────────
@test
def test_dual_momentum_vol_window_excludes_last_month():
    """Bars 0..-22 are calm; the LAST 21 bars are violently volatile. The
    reported ann_vol must be the calm number — the skipped month can't leak in."""
    rng = random.Random(3)
    bars = []
    price = 50.0
    for i in range(300):
        price *= (1 + 0.002 + rng.uniform(-0.0005, 0.0005))   # very calm
        bars.append(_bar(price, price * 1.001, price * 0.999, price, i=i))
    for j in range(21):                                        # violent last month
        price *= (1 + (0.09 if j % 2 == 0 else -0.085))
        bars.append(_bar(price, price * 1.05, price * 0.95, price, i=300 + j))

    res = STRATEGIES["dual_momentum"].candidate(bars, len(bars) - 1)
    assert res is not None, "dual_momentum should still qualify"

    closes = [b["close"] for b in bars[-260:]]
    window = np.array(closes[-253:-21], dtype=float)
    rets = np.diff(window) / window[:-1]
    expected = float(np.std(rets) * math.sqrt(252)) * 100
    assert approx(res["ann_vol"], round(expected, 1), tol=0.11), (
        f"ann_vol {res['ann_vol']} != 12-1 window vol {expected:.1f}"
    )

    # The buggy window (closes[-232:-1]) includes the violent month and would be
    # dramatically larger — prove the two are clearly distinguishable here.
    buggy_window = np.array(closes[-232:-1], dtype=float)
    buggy_rets = np.diff(buggy_window) / buggy_window[:-1]
    buggy = float(np.std(buggy_rets) * math.sqrt(252)) * 100
    assert buggy > expected * 3, "fixture failed to separate the two windows"
    assert abs(res["ann_vol"] - buggy) > 1.0, "ann_vol still reflects the old window"


@test
def test_dual_momentum_details_match_code():
    d = STRATEGIES["dual_momentum"].details
    joined = " ".join(d["rules"]) + " " + d["scoring"]
    assert "231-bar" not in joined, "details still advertise the old 231-bar window"
    assert "252 -> 21" in d["scoring"], "scoring text must name the 12-1 span"
    assert any(p[0] == "Volatility window" for p in d["parameters"])


# ──────────────────────────────────────────────────────────────────────────
# BUG 9 — VolatilityBreakout ATR average excludes today
# ──────────────────────────────────────────────────────────────────────────
@test
def test_volatility_breakout_atr_avg_excludes_today():
    from services.indicators import calc_atr

    rng = random.Random(8)
    bars = []
    price = 100.0
    for i in range(140):
        price *= (1 + 0.002 + rng.uniform(-0.002, 0.002))
        bars.append(_bar(price, price * 1.004, price * 0.996, price, i=i))
    # Breakout bar: new 20-day high on a much wider range (ATR expansion)
    prior_high = max(b["high"] for b in bars[-21:-1])
    close = prior_high * 1.03
    bars.append(_bar(prior_high, close * 1.02, prior_high * 0.97, close, i=140))

    res = STRATEGIES["volatility_breakout"].candidate(bars, len(bars) - 1)
    assert res is not None, "volatility_breakout should trigger on this fixture"

    window = bars[max(0, len(bars) - 120):]
    highs = [b["high"] for b in window]
    lows = [b["low"] for b in window]
    closes = [b["close"] for b in window]
    valid = [a for a in calc_atr(highs, lows, closes, 14) if a is not None]
    expected = valid[-1] / float(np.mean(valid[-21:-1]))     # today excluded
    buggy = valid[-1] / float(np.mean(valid[-20:]))          # today included

    assert approx(res["vol_expansion"], round(expected, 2), tol=0.011), (
        f"vol_expansion {res['vol_expansion']} != prior-20 ratio {expected:.3f}"
    )
    assert expected > buggy, "fixture failed: today's ATR spike must bias the old avg up"
    assert abs(expected - buggy) > 0.01, "the two averages are indistinguishable here"
    assert abs(res["vol_expansion"] - buggy) > 0.005, "still averaging today in"


@test
def test_volatility_breakout_details_match_code():
    d = STRATEGIES["volatility_breakout"].details
    joined = " ".join(d["rules"]) + " " + d["scoring"]
    assert "prior" in joined.lower() or "PRIOR" in joined, (
        "details must state that the ATR average excludes today"
    )
    assert "avg_prior_20_ATR" in d["scoring"]


# ──────────────────────────────────────────────────────────────────────────
# BUG 10 — schema_v stamping + stale-cache detection
# ──────────────────────────────────────────────────────────────────────────
@test
def test_enriched_quote_is_stamped_with_schema_v2():
    bars = make_bars(300, seed=21)
    enriched = md._enrich_with_technicals({"ticker": "T", "price": bars[-1]["close"]}, bars)
    assert enriched["schema_v"] == 2, f"expected schema_v 2, got {enriched.get('schema_v')}"
    assert md.QUOTE_SCHEMA_VERSION == 2


@test
def test_quote_schema_ok_rejects_legacy_cache_entries():
    assert md._quote_schema_ok({"schema_v": 2, "p52w": 50}) is True
    assert md._quote_schema_ok({"p52w": 50}) is False, "pre-v2 row must read as stale"
    assert md._quote_schema_ok({"schema_v": 1}) is False
    assert md._quote_schema_ok({"schema_v": "2"}) is False
    assert md._quote_schema_ok(None) is False
    assert md._quote_schema_ok({}) is False


@test
def test_get_quote_cache_paths_gate_on_schema():
    """Both cache-read paths in get_quote must consult _quote_schema_ok."""
    import inspect
    src = inspect.getsource(md.get_quote)
    # memory path
    assert "_is_fresh(entry[\"cached_at\"]) and _quote_schema_ok(entry.get(\"data\"))" in src, (
        "memory cache path does not gate on schema version"
    )
    # sqlite path
    assert src.count("_quote_schema_ok") >= 2, "SQLite cache path does not gate on schema version"


@test
def test_enriched_quote_field_contract():
    """The exact field contract consumers are being updated against."""
    bars = make_bars(300, seed=31)
    e = md._enrich_with_technicals({"ticker": "T", "price": bars[-1]["close"], "beta": 1.1}, bars)
    for key in ("schema_v", "gc_event", "dc_event", "ma_state", "gc", "dc",
                "p52w", "vol_r", "vol_r_5d", "ann_ret", "ann_ret_1m"):
        assert key in e, f"missing contract field: {key}"
    assert "earn_beat" not in e, "earn_beat must be gone from the enriched quote"
    assert e["ma_state"] in {"bull", "bear", None}
    assert isinstance(e["gc_event"], bool) and isinstance(e["dc_event"], bool)


# ──────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────
@test
def test_bad_refetch_never_replaces_good_history():
    """An empty/truncated Yahoo answer used to overwrite years of cached bars
    (BK/HOLX/MMC/SEE ended with 0 bars and dropped out of every backtest)."""
    def bars(n, last_day=28):
        return [{"date": f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}", "close": 1.0} for i in range(n)]
    cached = bars(1254)
    assert md.history_fetch_rejection(cached, []) is not None
    assert md.history_fetch_rejection(cached, cached[-27:]) is not None, "27 of 1254 bars is truncated"
    assert md.history_fetch_rejection(cached, cached[:-3]) is not None, "must not move backwards in time"
    assert md.history_fetch_rejection(cached, cached[1:] + [{"date": "2099-01-01", "close": 1.0}]) is None, \
        "a normal rolling refetch (drop oldest, add newest) is accepted"
    assert md.history_fetch_rejection([], bars(10)) is None, "nothing cached: accept anything"


def main() -> int:
    passed = failed = 0
    failures = []
    for fn in _TESTS:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            failures.append((fn.__name__, exc, traceback.format_exc()))
            print(f"FAIL  {fn.__name__}: {exc}")
        else:
            passed += 1
            print(f"PASS  {fn.__name__}")

    print("\n" + "=" * 56)
    print(f"  {passed} passed, {failed} failed, {len(_TESTS)} total")
    print("=" * 56)
    if failures:
        print("\n--- failure tracebacks ---")
        for name, _exc, tb in failures:
            print(f"\n[{name}]\n{tb}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
