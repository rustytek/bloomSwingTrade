# ML4T gap tracker

Design gaps found when the app was evaluated against *Machine Learning for Trading*,
3rd edition (Stefan Jansen, 2026) on 2026-09-24. Worked **one at a time**, in order, and
only when the user asks to start the next one. Chapter numbers refer to the book. The book
itself lives in `Coding notes/` and is git-ignored (copyrighted — never commit it).

Status: `done` · `in progress` · `todo` · `won't do` (with the reason)

| # | Gap | Book | Status | Release |
|---|---|---|---|---|
| 1 | Trades filled at the same close that produced the signal | §16.2 | done | v1.22.0 |
| 2 | No sealed holdout; 32-cell search not corrected for; runs not logged; no Deflated Sharpe | Ch 6, §7.4, §16.7 | done | v1.23.0 |
| 3 | Overlapping 5-day periods overstate confidence (Wilson assumes independence) | §7.2 | done | v1.24.0 |
| 4 | Regimes used to hard-switch strategies instead of as a risk lens / continuous input | §1.4, §9.5 | done | v1.24.0 |
| 5 | No same-universe equal-weight baseline — SPY is the only benchmark | §16.6, §17.4 | done | v1.24.0 |
| 6 | One cost setting — no gross-vs-net, turnover, break-even cost or sensitivity sweep | §16.6, §18.8 | done | v1.24.0 |
| 7 | No account-level kill switches (daily loss / drawdown breaker, graduated escalation) | §19.8, §26.5 | done | v1.24.0 |
| 8 | Robinhood: no client order id (retry can duplicate), holdings mismatch doesn't block trading, no quote-freshness check | §25.5, §25.7 | done | v1.24.0 |
| 9 | Data governance: 5-year cache overwritten not versioned; prices as JSON in SQLite, not columnar; no point-in-time membership | Ch 2 | done | v1.24.0 |
| 10 | AI report output is never validated | §22.6 | done | v1.24.0 |

## Detail

### 1 — Next-session fills · done (v1.22.0)
Entries are next-day limit orders at the top of the ATR zone (`min(open, limit)`, no fill
on a gap above the limit or through the stop); time/regime exits sell at the next open;
rotation trades open-to-open. `METHOD_VERSION` 3. Measured impact: momentum trade_plan
CAGR 13.6 → 9.5 %, breakout rotation 2.2 → 0.1 %, pullback slightly up. See CLAUDE.md →
Backtesting.

### 2 — Holdout, multiple testing, trial log, Deflated Sharpe · done (v1.23.0)
- The edge matrix judges regime tags on the same history it reports, and "confirmed"
  needs only a positive mean and compounded return on ≥ 20 periods — with 32
  strategy × regime cells, several will pass by chance.
- Those verdicts already act: the Weekly Plan un-ticks buys from `mis-tagged` strategies,
  and the (off-by-default) evidence override would switch strategies on and off.
- Strategy Lab runs are not recorded, so the number of variants tried is unknown, and no
  run reports a Sharpe confidence interval or a Deflated Sharpe Ratio.

**2a (edge matrix, `METHOD_VERSION` 4).** A verdict now needs a Benjamini-Hochberg-significant
mean (q ≤ 0.10, AR(1) effective n) and agreement between the discovery segment (before
2019-01-01) and the holdout (2019+). Without the 20-year archive every cell is unproven.
Measured on the 20-year DB (2007-07 → 2026-09, 32 tests, ~20 min build on the dev PC):
- **Confirmed (6):** every strategy tagged for trending_bull — momentum, pullback, breakout
  volume, dual momentum, volatility breakout, sector rotation. Each held up on both sides of 2019.
- **Everything else unproven**, including every tagged choppy / bear cell (mean reversion,
  sector rotation and low-vol in choppy_calm; low-vol in choppy_volatile; dual momentum in
  trending_bear — negative before 2019, positive after). **No cell is mis-tagged** and none earns
  an untagged-edge, so the Weekly Plan un-ticks nothing on evidence grounds.
- Closest miss: mean reversion in trending_bull (q 0.065, both segments positive) — untagged and
  below the 0.25 %/period promotion bar.
- The 5-year cache alone gives 23 tests and zero verdicts (no pre-2019 segment), as designed.

**2b (Strategy Lab).** Every run carries `sharpe_inference` (skew/kurtosis-aware 95 % CI,
P(Sharpe > 0)) and is logged to `backtest_runs`; `selection_bias` reports trials, the luck
hurdle and the Deflated Sharpe. Real check, momentum rotation, 5-year window, three `top_n`
variants: Sharpe 0.33 → DSR 0.77 (1 trial); 0.56 → 0.87 (2 trials); 0.03 → 0.36, "likely luck"
(3 trials, hurdle 0.19). `GET /api/backtest/trials` lists them; no delete by design.

**Follow-on:** gap 3 should replace the AR(1) effective n with episode-level clustering — the
lag-1 correction barely moved most cells (n_eff ≈ n), which is exactly gap 3's point.

### 3 — Overlapping periods · done (v1.24.0)
Edge-matrix significance now clusters by regime EPISODE (a stay in a quadrant; stays of the
same quadrant split by ≤ 4 periods of another are merged — ADX flicker is not a new episode)
and uses the more conservative of the cluster-robust and AR(1) standard errors, with df from
the episode count; win-rate intervals use the effective n. `METHOD_VERSION` 5. Real 20-year
data: trending_bull 88 raw → 35 merged episodes, trending_bear 30 → 17; verdicts unchanged
from v4 (six trending_bull confirmed), q-values rose (volatility breakout 0.065 → 0.087).

### 4 — Regimes as a risk lens · done (v1.24.0)
`services/regime_risk.py`: a smooth 0.5–1.0 risk multiplier from SPY vs 200MA, VIX/realized
vol and ADX (no step at any threshold; can only cut risk), soft quadrant memberships and a
per-strategy fit weight (full / fading in / fading out). Applied to live sizing only when
the per-user `regime_sizing` setting is on (ships OFF; off = byte-identical sizing). Also a
trade_plan backtest flag. Real data: max drawdown fell in all six runs (3–11 pts); CAGR
mixed (down in 3 of 6, notably 2008-era breakout/pullback).

### 5 — Same-universe baseline · done (v1.24.0)
Every walk-forward reports an equal-weight portfolio of the same eligible tickers, same
schedule, same next-open fills and cost model (`baseline`, `baseline_metrics`,
`baseline_comparison`). Latest 5 years: every strategy tested trails it (baseline ≈ 11.4 %
CAGR; momentum trade_plan 6.7 %, breakout trade_plan 0.9 %).

### 6 — Cost sensitivity · done (v1.24.0)
`cost_analysis`: gross vs net, cost drag, turnover, break-even cost vs cash and vs the baseline,
and a 0/5/10/25/50 bps sweep from the same run (exact for rotation; first-order for trade_plan,
plus per-trade R before/after costs). Breakout rotation: ~3,600 %/yr turnover, 12.3 % gross →
4.5 % net.

### 7 — Kill switches · done (v1.24.0)
`services/circuit_breaker.py`: daily equity snapshots; warn (daily loss ≥ 2 %), reduce
(drawdown ≥ 8 % or 5 losses in a row → 50 % size), halt (drawdown ≥ 12 % → no new entries in
Weekly Plan, Plan-a-Trade or live BUYs; sells never blocked). Halt latches until recovery or
acknowledgement. Account-size edits are rebased as deposits, never read as drawdowns.

### 8 — Robinhood robustness · done (v1.24.0)
Client order id on every order, an `unknown` state after an ambiguous place result that
blocks re-placement until resolved, duplicate-order refusal; live SELLs of mismatched
holdings blocked and buys need acknowledgement when holdings disagree; stale quotes refused
(20 min in session). Mock-tested only — real Robinhood behaviour unverified.

### 9 — Data governance · done (v1.24.0)
Compact columnar price storage (numpy + zlib BLOBs, exact round trip, ~24 % of the JSON size),
a content version per series, a `data_versions` fingerprint on every walk-forward, revisions
kept when a refetch changes stored bars (bounded retention), and per-ticker source reporting.
Point-in-time membership remains unavailable from free data (disclosed). One-way: after the
conversion an older add-on version sees no history — restore a backup to roll back.

### 10 — AI output validation · done (v1.24.0)
`services/ai_validation.py` checks report/chat/analysis text against the numbers the model was
given (unknown tickers, price/RSI/MA claims within tolerance, above/below and
overbought/oversold contradictions, signal schema) and annotates — never rewrites — with a
visible claim-check panel. It cannot judge opinions, forecasts or unprovided metrics.
