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
| 3 | Overlapping 5-day periods overstate confidence (Wilson assumes independence) | §7.2 | todo | |
| 4 | Regimes used to hard-switch strategies instead of as a risk lens / continuous input | §1.4, §9.5 | todo | |
| 5 | No same-universe equal-weight baseline — SPY is the only benchmark | §16.6, §17.4 | todo | |
| 6 | One cost setting — no gross-vs-net, turnover, break-even cost or sensitivity sweep | §16.6, §18.8 | todo | |
| 7 | No account-level kill switches (daily loss / drawdown breaker, graduated escalation) | §19.8, §26.5 | todo | |
| 8 | Robinhood: no client order id (retry can duplicate), holdings mismatch doesn't block trading, no quote-freshness check | §25.5, §25.7 | todo | |
| 9 | Data governance: 5-year cache overwritten not versioned; prices as JSON in SQLite, not columnar; no point-in-time membership | Ch 2 | todo | |
| 10 | AI report output is never validated | §22.6 | todo | |

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

### 3 — Overlapping periods
Regime cells count 5-day periods, but a regime lasts months — 23 "trending bear" periods
may be one or two bear markets. Needs block/cluster-aware effective sample size. Interacts
with 2 (the significance tests there should use the effective n).

### 4 — Regimes as a risk lens
Hard on/off switching by quadrant is fragile near the boundaries. Book prefers scaling
exposure or sizing continuously with regime inputs.

### 5 — Same-universe baseline
Report an equal-weight portfolio of the same eligible tickers, same costs, same rebalance
schedule, next to SPY.

### 6 — Cost sensitivity
Gross vs net, turnover, break-even cost, and a sweep over `cost_bps`.

### 7 — Kill switches
Daily-loss and drawdown circuit breakers with graduated escalation (warn → size down →
pause new entries).

### 8 — Robinhood robustness
Client order id / idempotency key on every order; block (or require acknowledgement) when
holdings disagree; refuse to size from a stale quote.

### 9 — Data governance
Versioned price snapshots, columnar storage (Parquet), and a note that point-in-time index
membership isn't available from free data.

### 10 — AI output validation
Check the daily report's claims (tickers, numbers, levels) against the computed indicators
before showing it.
