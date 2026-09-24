# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SwingTrader is a self-hosted swing trading screener — a FastAPI backend + vanilla JS frontend deployed via Docker or as a Home Assistant OS (HAOS) native add-on. It screens S&P 500 and ETF tickers with technical indicators (RSI, MACD, Bollinger Bands, MA50/200 trend state, Golden/Death Cross events), per-user watchlists/portfolios, and optional AI analysis via Anthropic Claude, OpenAI, or LiteLLM (the local/self-hosted path — see aiProxy's `CLAUDE.md`). LiteLLM is the only AI backend this app talks to — it never calls a model runtime directly. **`AI_MODEL`/`REPORT_MODEL` must always be a LiteLLM tier alias** (e.g. `tooling_high`), never a raw provider model name, so the physical model behind an alias can change without a config edit here. No model name is hardcoded anywhere in the codebase; `services/ai_service.py::looks_like_raw_model_name()` flags a raw name at startup, and `test_passes.py` fails the build if one is reintroduced.

## Common Commands

```bash
# Run locally (dev)
pip install -r requirements.txt
python main.py

# Docker (recommended)
docker-compose up --build -d
docker-compose logs -f swingtrader

# Run test suites
python test_passes.py            # core logic: trade plans, strategies, backtest math, journal
python test_indicators.py        # calculation regressions: indicators / market_data / strategies
python test_portfolio_risk.py    # correlation, open risk, concentration, edge-weighted sizing
python test_backtest.py          # backtest engine, exit rules, no-look-ahead, Wilson CI
python test_edge.py              # edge matrix verdicts, evidence override safety, scorecard
python test_plan_persistence.py  # plan-intent columns, notes backfill, entry_chasing honesty
python test_jobs.py              # background jobs: worker subprocess, dedupe, stale reaping
python test_weekly_plan.py       # weekly-plan decisions, risk-aware buy pick, strategy rationale coverage
python test_broker.py            # Robinhood OAuth/MCP (mocked), order validation, paper safety, fill sync
```

**235 tests across nine suites** (`python test_broker.py` covers the Robinhood/Trade layer, fully mocked). All are self-contained (no network) except
`test_jobs.py`, which deliberately **launches a real worker subprocess** against a
throwaway SQLite file in a temp dir — mocking the subprocess would let the very
layer it guards break while the test still passed. `test_passes.py`
installs permissive import stubs for `fastapi`, `jose`, `passlib` and `bcrypt` (appended to the
**end** of `sys.meta_path`, so a real install always wins) purely so the pure helper functions
living in `api/*.py` can be imported without the full web stack.

> **Counting routes:** this FastAPI version stores `_IncludedRouter` lazy references, so
> `len(app.routes)` UNDERCOUNTS and filtering on `hasattr(r, "path")` silently omits every
> router-mounted endpoint. Always verify through `app.openapi()["paths"]` (currently 80; the OAuth callback is excluded from the schema).

The app runs on HTTPS at `https://localhost:8443`. Swagger docs at `/api/docs`.

## Architecture

### Entry Points
- `haos_entry.py` — Used by Docker/HAOS; reads `/data/options.json` (HA config) or falls back to env vars
- `main.py` — FastAPI app: registers routers, initializes DB, generates SSL cert, starts uvicorn + APScheduler

### Request Flow
```
React SPA (static/*.html) → FastAPI (main.py)
  → auth/ (JWT login/register)
  → api/ (stocks, screener, watchlist, portfolio, ai, charts, today, weekly_plan, broker,
          settings, journal, backtest, scorecard, jobs)
  → services/ (market_data, indicators, ai_service, chart_service, report_service,
               strategies, strategy_rationale, regime, trade_plan, today, weekly_plan,
               broker_service, brokers/robinhood_mcp, secrets_box)
  → SQLite via SQLAlchemy (database/)
```

### Page Routes
- `/` — the **Playbook** (`static/today.html`). Three-cell regime band (quadrant/ADX/SPY/VIX · the `transition.reason` "time to change strategy" signal · the risk budget from `GET /api/portfolio/risk`, with `unstopped_warning` shown loudly in red). Main column: setups **grouped by strategy**, each group header carrying that strategy's tested edge in the *current* quadrant from `GET /api/edge-matrix`, verdict-coloured (confirmed green / unproven amber / mis-tagged red). A collapsed "standing down" strip explains every off-regime strategy. Right rail: positions sorted by urgency with their `actions[]` verbatim, book exposure, and the morning checklist. The **Plan-a-Trade modal** wraps `POST /api/portfolio/assess` and commits through `POST /api/portfolio`; its commit button is disabled while any warning is `block` level. (Also the 404 catch-all fallback.)
- `/plan` — the **Weekly Plan** (`static/plan.html`), the second tab: a five-step walkthrough (read the market → strategies in play and *why* → what to do with each holding → new trades → review) built on `GET /api/weekly-plan`. Every recommendation carries a plain-English `why[]`. Recommended orders are pre-ticked; the ticks are saved to localStorage `st_trade_selection` (a JSON array of order ids) **plus** `st_trade_selection_week` (= the plan's `week_of`), so a selection never carries into a new week. "Continue to Trade" hands the selection to `/trade`. Nothing is placed from this page.
- `/trade` — the **Trade** page (`static/trade.html`), deliberately the **last** tab: connect Robinhood → paper/live → pick orders from `/api/weekly-plan` (honouring the week-scoped selection keys) → preview → place, plus an order-history panel. See "Trade / Robinhood" below.
- `/screener` — the screener (`static/index.html`, formerly served at `/`).
- `/backtest` — the **Strategy Lab** (`static/backtest.html`). Headlined by the strategy × regime edge matrix, then a single-strategy walk-forward with a rotation/trade_plan mode toggle, sortable trade log with exit-kind distribution, and a severity-sorted caveats panel.
- `/scorecard` — the **Scorecard** (`static/scorecard.html`): realized expectancy vs the backtest's expected R, per-strategy drift with a sample-size guard, and execution-quality leaks ranked by realized R cost. Metrics the app cannot compute get their own "Not Measurable Yet" panel — never `0`, never `--` — each naming the field that must be persisted first.
- `/journal` — trade journal (`static/journal.html`).
- `/charts` — chart dashboard (`static/charts.html`).
- `/admin` — user management (`static/admin.html`), admin-only. Create/edit/delete users, reset passwords, toggle admin, and set/clear each user's per-user LiteLLM key. The header "Users" link (in `common.js`) is shown only to admins.
- Shared page chrome is centralized in **`static/js/common.js`** (auth/fetch helpers, `NAV_LINKS`, `buildTopbar`/`initHeader`, the ticker-detail modal) and **`static/css/common.css`** (`.topbar`, `.brand`, `.btn*`, `.hint` tooltips, `.ticker-link`, `.modal*`, and the `.muted`/`.mono`/`.up`/`.down`/`.error` text primitives), both served via the `/static` StaticFiles mount. **Every page links `common.css` and loads `common.js` before its own inline script**, and builds its bar with `<div id="topbar" class="topbar"></div>` + `initHeader('<path>', leftExtraHtml)`. Never hand-write nav markup or re-inline the palette — both drifted badly in the past (pages stuck on "Today"/"Backtest" with no Scorecard link, and journal/admin quietly using a different green for P&L). Page-specific bar controls go through `leftExtraHtml`; because `initHeader` injects them with `innerHTML`, **their listeners must be attached after the returned promise settles**.
- **Never redeclare a `common.js` top-level name in a page script.** A page `const`/`let` that reuses one is a `SyntaxError` that blanks the whole page — including over one of its `function`s, because a global function declaration creates a non-configurable global property. This bit twice during the refactor (`_tdChart` in charts.html, `fmtPct` in index.html). Note `token` is a **function**, and `authHdr()` returns only `Authorization` — any request with a JSON body must add `Content-Type` itself (see `jsonHdr()` in report.html, `jsonAuthHdr()` in index.html). `charts.html` deliberately overrides `openTickerDetail`/`closeTickerModal` with its richer tabbed modal, naming its chart handle `_tdLwChart` to avoid the clash.
- `static/index.html` is React + in-browser Babel, so it reads `NAV_LINKS` as data and renders it with JSX; it must **not** call `initHeader()`, which would fight React's DOM. Babel standalone appends the transpiled code as a real `<script>`, so its top-level bindings share global scope with `common.js` just like any other page.
- `test_passes.py` guards all of this: shared-chrome usage, NAV_LINKS as the only nav source, every nav href having a route, the palette living only in `common.css`, and — via `node --check` on `common.js` + each page's inline script — that they actually compile together.

### Key Files
| File | Purpose |
|---|---|
| `config.py` | Pydantic Settings — all env vars with defaults |
| `database/models.py` | ORM models: User, WatchlistItem, PortfolioPosition, StockCache, AICache, ReportCache, ClosedTrade, HistoryArchive |
| `database/db.py` | SQLAlchemy engine, `get_db()` FastAPI dependency |
| `auth/deps.py` | `get_current_user` / `get_current_admin` JWT dependencies |
| `services/market_data.py` | yfinance wrapper, dual-layer cache (in-memory dict + SQLite), indicator calculation; emits the `schema_v: 2` quote contract (see "Quote Field Contract" below) |
| `services/universe.py` | ~450 tickers: S&P 500 constituents + ETF lists |
| `services/ai_service.py` | Abstract `AIService` base + Mock/LiteLLM implementations (Anthropic/OpenAI stubbed, not yet implemented) |
| `services/indicators.py` | Technical indicators; includes `calc_atr(highs, lows, closes, period=14)` (Wilder-smoothed) and `calc_adx(highs, lows, closes, period=14)` (Wilder ADX/+DI/-DI — trend-strength gauge behind `services/regime.py`; **the DI series is offset by `period`, not 1** — getting that wrong silently yields look-ahead values). `compute_performance_metrics` returns **both** `ann_ret` (full-history annualized %, the figure Sharpe/Sortino/Calmar/Info Ratio/Treynor are computed from) and `ann_ret_1m` (21-bar annualized %); both annualize over `n-1` return periods |
| `services/regime.py` | ADX + 200MA + VIX/realized-vol market regime classifier — see "Market Regime" below |
| `services/strategies.py` | 9-strategy framework; registry `STRATEGIES` (see below) |
| `services/trade_plan.py` | `build_trade_plan(...)`: ATR entry zone, stop, fixed-fractional sizing, R-multiple target (see below) |
| `services/portfolio_risk.py` | Portfolio-level risk: returns-based correlation matrix (date-aligned, 30-bar minimum overlap), open R / unstopped-position detection, sector + weighted-beta concentration, portfolio heat, and `assess_new_position()` pre-trade block/warn checks |
| `services/exits.py` | Pluggable exit rules — `FixedStopTarget`, `AtrTrailingStop` (chandelier), `TimeStop`, `PartialProfitTaking`, `RegimeExit` — plus the priority-ordered `resolve_exit()` resolver. Stop fills are assumed to precede target fills within a bar. `build_exit_rules(..., strategy_id=…)` resolves the time stop's `max_bars` as: explicit caller value > the strategy's numeric `horizon_days` > `DEFAULT_TIME_STOP_BARS` (20). `strategy_horizon_days()` reads only the numeric attribute — `details["horizon"]` is prose and is never parsed |
| `api/portfolio.py` | Positions CRUD/close/CSV-import, plus `GET /api/portfolio/risk` (heat, open risk, concentration, correlation matrix) and `POST /api/portfolio/assess` (pre-trade check). The `GET /api/portfolio` summary carries `open_r` and `portfolio_heat` |
| `services/today.py` | Builds the `/api/today` payload; exposes `position_flags()` helper (see below) |
| `services/strategy_rationale.py` | Written **why** for every strategy: `why_it_works`, `best_when`, `fails_when`, and a `regime_fit` sentence per quadrant. `rationale_for(id, quadrant)` adds `fit_now`. Surfaced in `/api/today`'s `strategy_regime_status[].rationale`, the Strategy Lab catalog, and the Weekly Plan. It is opinion, not evidence — the UI always labels it as such next to the edge-matrix verdict. `test_weekly_plan.py` fails if a registered strategy or quadrant has no text |
| `services/weekly_plan.py` / `api/weekly_plan.py` | `GET /api/weekly-plan` — the decision layer on top of `build_today` (so the two pages can never disagree). See "Weekly Plan" below |
| `services/edge_matrix.py` | Strategy × regime EVIDENCE matrix — one walk-forward per actionable strategy, bucketed by the `quadrant` already stamped on each rebalance period. Per-cell Wilson CI + `confidence`, and a `verdict` (`confirmed`/`unproven`/`mis-tagged`/`untagged-edge`) saying whether the hand-written `Strategy.regimes` tag is actually supported. 12h per-user cache |
| `services/scorecard.py` | Realized (journal) vs expected (edge matrix) per strategy — `drift` plus a sample-size-guarded recommendation — and `execution_quality()` (entry chasing / stop loosening / overstayed horizon / off-regime entries), which returns any metric it cannot compute as explicitly UNAVAILABLE with the field it needs |
| `api/scorecard.py` | `GET /api/scorecard`, `GET /api/edge-matrix` (cold call returns "not computed yet"; `?refresh=true` builds), `POST /api/edge-matrix/invalidate` |
| `api/today.py` | `GET /api/today` — daily dashboard |
| `api/settings.py` | `GET`/`PUT /api/settings` — account_size, risk_pct, max_positions, atr_stop_mult, r_multiple |
| `api/journal.py` | `GET /api/journal` — `trades` is the LIMIT-ed display page, while `stats`, `by_strategy` and `equity_curve` are all computed from **every** closed trade. `_stats` splits `scratch` (pnl == 0) out of `losses`, so `win_rate = wins/(wins+losses)`, and reports `r_sample` (how many trades actually carry an `r_multiple`) because `avg_r`/`expectancy_r` use that subset while `win_rate` spans all decided trades. `DELETE /api/journal/{id}` |
| `api/backtest.py` | Walk-forward backtest; `GET /api/backtest/strategies` returns each strategy's `details` (horizon/how/rules/scoring/parameters) plus a `backtestable` flag, rendered as a detail panel on the backtest page. Only **actionable** strategies are accepted by `/walk-forward` (`BACKTESTABLE_STRATEGIES`) — the engine is long-only, so watchlist-only signals like `bear_reversal_watch` are rejected there while staying visible in the catalog |

### Caching Strategy
Two-layer cache: in-memory Python dict (`_mem_cache`) → SQLite `StockCache` table. Data is considered "fresh" if cached **after the most recent NYSE market close (4pm ET)** — not a rolling TTL. Quote and history TTLs are configurable via env vars but default to daily refresh.

History cache window is **5 years** in `services/market_data.py` (`get_history`/`_fetch_history_sync` defaults, and the `api/stocks.py` `/history` endpoint default — keep these in sync so a detail-view refetch can't shrink the shared cache). Existing shorter caches upgrade to 5y on the next post-close refresh (or a manual screener refresh). For backtests over **arbitrary historical eras** beyond the rolling window, see `services/history_archive.py` (below).

### Quote Field Contract (`schema_v: 2`)

Enriched quotes carry `"schema_v": QUOTE_SCHEMA_VERSION` (currently `2`, defined in
`services/market_data.py`). **Both** quote cache-read paths in `get_quote` — the in-memory
`_mem_cache` and the SQLite `StockCache.quote_json` row — treat a cached quote whose `schema_v`
is behind the current version as **stale regardless of its timestamp** and re-fetch/re-enrich
it. `invalidate_legacy_schema_cache()` runs at startup so the background universe sweep picks
old rows up too (it selects stale tickers by timestamp alone and would otherwise miss them).

**Bump `QUOTE_SCHEMA_VERSION` whenever the *meaning* of an enriched field changes** — not
merely when a field is added. The front end must also fall back to `--` on any new key rather
than rendering `undefined`/`NaN`, since a stale row can still arrive mid-refresh.

| Field | Meaning |
|---|---|
| `p52w` | Position within the true 252-bar (52-week) range. **Was** computed over the full 5-year cache, which made a stock at its 52-week low report ~60%. |
| `gc` / `dc` | Golden/Death **crossover EVENT** within the last 5 bars — *not* a standing state. Previously one of the two was always true, so every uptrending stock was badged "Golden Cross". |
| `gc_event` / `dc_event` | Explicit aliases of `gc`/`dc`. Prefer these in new code. |
| `ma_state` | `"bull"` \| `"bear"` \| `None` — whether MA50 is currently above/below MA200. This is the state `gc`/`dc` used to carry; anything that wants trend state (scoring, ranking, filters) must read this. |
| `ann_ret` | **Full-history** annualized %. Consistent with `sharpe`/`sortino`/`calmar`, which are computed from the same series. |
| `ann_ret_1m` | 21-bar (1-month) annualized %. This is what the old `ann_ret` key actually held. |
| `vol_r` | Today's volume ÷ the prior 20-day average (conventional single-bar volume confirmation). |
| `vol_r_5d` | The former 5-day ÷ 20-day average ratio, preserved. |
| `earn_beat` | **Removed** — it read a yfinance field (`earningsBeat`) that does not exist, so it was always false and its `compute_score` point could never be earned. |
| `schema_v` | Internal marker, value `2`. Never displayed. |

**Consumers of this contract.** Anything reading a quote must pick the right half of each pair:
- `api/screener.py` — the `gc`/`dc` filters mean a **cross event in the last 5 bars** (the UI labels them "Golden/Death Cross (last 5d)"); the `ma_state` filter (`""`/`"bull"`/`"bear"`) is how to filter on the standing MA50-vs-MA200 trend. `vol_r_min` filters the single-bar ratio, `vol_r_5d_min` the smoothed one.
- `api/watchlist.py::composite_rank` (module-level, so it is unit-testable) — the 0.5 trend bonus keys off `ma_state == "bull"`; a fresh `gc_event` adds a separate 0.25. Keying the state bonus off `gc` made it effectively never fire and silently reordered the watchlist.
- `api/ai.py` — the technicals whitelist passed to the LLM sends `ma_state` **and** `gc`/`dc`, plus `ann_ret_1m` **and** `ann_ret`, so the model cannot conflate a fresh cross with an established trend, or a 1-month pace with a long-run rate. `services/ai_service.py::generate_signals`' docstring is the written-down version of the same contract.
- `services/backtest.py::build_decision_cockpit` uses `gc_event`/`dc_event` and says "crossed"; `services/today.py::position_flags` uses `ann_ret_1m`.

> **Gotcha:** `ScreenerFilters` is a plain `BaseModel`, so pydantic v2's default `extra="ignore"` applies — a filter key the front end sends but the model does not declare is **silently dropped, not rejected**. A new filter must be added to the model *and* to `_passes()`, or the control will appear to work and quietly do nothing.

`div_yield` is unit-autodetected: yfinance changed `dividendYield` from a fraction to a
percent across versions, so a raw value `<= 1.0` is multiplied by 100 and anything larger is
taken as already-percent.

### Strategy Framework (`services/strategies.py`)
Registry `STRATEGIES` maps ids to `Strategy` instances:
- `momentum_rotation` — Clenow-style 90-bar exponential regression slope × R².
- `pullback_50ma` — buy dips to the 50MA in an uptrend.
- `breakout_volume` — 60-day-high breakout on >=1.5x volume.
- `mean_reversion` — Connors-style RSI(2) < 10 washout above the 200MA (lower Bollinger touch confirms); short 3–10 day snap-back holds.
- `dual_momentum` — Antonacci-style: 12-1 month absolute momentum gate (must beat cash) ranked by risk-adjusted score — the return divided by its volatility over that **same** 252→21 bar span (the skipped last month is excluded from the volatility too).
- `volatility_breakout` — Donchian 20-day-high breakout confirmed by ATR expansion: current ATR(14) above the average of the **prior** 20 ATR readings, today excluded (vs. `breakout_volume`'s share-volume confirmation, which excludes today the same way).
- `sector_rotation` — same regression-slope×R² approach as `momentum_rotation`, restricted to the 11 SPDR sector ETFs via `applies_to()`.
- `bear_reversal_watch` — RSI(2) > 90 inside a confirmed downtrend; `actionable = False` (watchlist-only, never gets a trade plan — this is a long-only app).
- `low_vol_trend` — uptrend names ranked by lowest realized volatility; defensive tilt for choppy/volatile regimes.

Each `Strategy` exposes `candidate(bars, idx)` (backtest hook, uses `bars[:idx+1]`) and `scan(ticker, bars, quote)` (live hook returning a `Setup`). Two additional class attributes drive regime awareness and safety: `regimes: list[str]` (which `services/regime.py` quadrants a strategy is built for) and `actionable: bool` (whether it should ever get a live trade plan). `applies_to(ticker)` restricts a strategy to a ticker sub-universe (used by `sector_rotation`); it's checked by both `scan()` and the walk-forward backtest.

### Market Regime (`services/regime.py`)
Two independent axes classify the market into one of four quadrants (`QUADRANTS`): **direction** (SPY above/below its 200-day MA) and **trend strength**, measured by `calc_adx` (ADX(14) on SPY — the "when to change strategies" indicator: ADX >= 25 is trending, < 20 is choppy, 20–25 is a transition band), plus a VIX-based (or, for historical backtests without a per-date VIX, SPY's own realized-volatility proxy) crisis overlay:
- `trending_bull` / `trending_bear` — ADX >= ~20 and above/below the 200MA.
- `choppy_calm` / `choppy_volatile` — ADX < 25 with normal/elevated volatility.

`classify_regime()` also returns a `transition` event (e.g. "ADX fell below 20 — trend exhausted, favor mean-reversion") whenever a threshold was crossed in the last few sessions — this is what `today.html`'s regime banner surfaces as the "time to change strategy" signal. `services/today.py::build_today` routes through `_resolve_active_strategies()`, which uses `strategies_for_regime_with_evidence` + `services/edge_matrix.get_cached_matrix(user_id)` when the per-user `User.use_evidence_regimes` setting (or the module-level `set_evidence_override`) is on. Three invariants: it **never builds** the matrix (a cold build is minutes — a cache miss means "no evidence", not "wait"); a missing, stale or malformed matrix degrades to **exactly** the hand-written-tag behaviour; and it **ships OFF**. Each `strategy_regime_status[]` entry carries an `evidence` sub-object (`effect`: excluded/promoted/unchanged, `verdict`, `n`) and the payload a top-level `evidence_regimes` summary, so the UI can explain why a strategy was stood down or promoted.

`strategies_for_regime(quadrant)` returns the strategy ids tagged for a quadrant; `services/today.py::build_today` uses it to mark each strategy active/inactive for the live setups list, and `services/backtest.py::run_walk_forward_backtest` uses the same `classify_quadrant`/`trend_strength_label` helpers to tag every historical rebalance period, producing a `regime_breakdown` (per-quadrant CAGR/win-rate) in the walk-forward response — the Backtest page's Strategy Comparison and single-strategy results both surface this.

### Long Builds Are Background Jobs — Never HTTP Requests

**This rule exists because breaking it took the whole add-on down (2026-09-23).**
`/api/edge-matrix?refresh=true` ran one full walk-forward per strategy inside the
request. It was already offloaded to a worker thread with `anyio.to_thread`, which
stops the build *blocking* the event loop but does nothing about **GIL contention**:
sustained CPU left uvicorn too few cycles to complete a TLS handshake, so cloudflared
logged `net/http: TLS handshake timeout` against **every** endpoint — `/auth/me` and
`/api/today` included — and Cloudflare returned 502 for the entire app. The Supervisor
watchdog then restarted the container, which killed the in-flight build, wiped the
in-memory chart cache, and left the user retrying into the same wall.

The diagnostic tell: **uvicorn's access log only prints on response completion**, so
the requests that hung are invisible in the add-on log. An app log that looks healthy
while cloudflared logs handshake timeouts means a starved loop, not a slow endpoint.

| File | Purpose |
|---|---|
| `database/models.py::BackgroundJob` | `background_jobs` table: status, `progress` 0–1, `progress_detail`, `heartbeat_at`, `result_json`, `params_key` |
| `services/jobs.py` | API-side registry. `enqueue()` **reuses an in-flight job with the same `(user, kind, params_key)`** — the outage log shows `?refresh=true` arriving twice concurrently, doubling the load. `reap_stale()` fails a job whose heartbeat is older than `STALE_AFTER` (12 min), so a killed worker can't block the feature forever. `to_dict()` deliberately omits the result payload; polls must stay cheap |
| `services/job_worker.py` | `python -m services.job_worker <job_id>` — a **fresh interpreter with its own GIL**. Must have no import side effects (it runs per job) and must never import `main.py`. Every path ends in a terminal status. Carries a `selftest` kind that exercises the whole pipeline in seconds |
| `api/jobs.py` | `GET /api/jobs`, `GET /api/jobs/{id}`, `POST /api/jobs/{id}/cancel`. Cancel marks the row inactive so a rebuild can start; it does **not** kill the detached worker, and says so |
| `main.py::orphan_background_jobs()` | Runs at startup. Workers die with the container, so anything still `queued`/`running` is dead — resetting it immediately avoids a 12-minute window where no rebuild can start after every restart |

`subprocess.Popen` with `-m`, not `fork` (inherits locks from a process already running
an asyncio loop plus threads) and not `spawn` (re-imports `main.py` and would start a
second web app). `build_edge_matrix(..., progress=cb)` drives the heartbeat; a failing
callback is swallowed so progress reporting can never sink a finished build.

`GET /api/edge-matrix?refresh=true` returns `{status:"building", job:{...}}` in
milliseconds. **Three states, not two** — `not_computed` / `building` / `cached`; a UI
that renders `building` as `not_computed` invites a second build onto a loaded box.
Cached reads consult the **job table first**, because `edge_matrix._cache` is in-memory
and empties on every restart.

`services/chart_service.py`'s cache is also memory-only, so `main.py::_warm_chart_cache()`
fills it in the background at startup rather than making the first `/charts` visitor pay
for the cold fetch inside their own request.

### Evidence Layer (`services/edge_matrix.py`, `services/scorecard.py`, `api/scorecard.py`)

`Strategy.regimes` is a hand-written literal — an opinion. The edge matrix turns the backtest into evidence about whether that opinion holds, per quadrant, and the scorecard compares what a strategy was *supposed* to deliver against what the journal says it *did*.

- `GET /api/edge-matrix` — **a cold call deliberately does NOT build**: it returns `{status:"not_computed", matrix:null, message, last_error, strategies[], quadrants[], thresholds{}}` so the page paints instantly. `?refresh=true` **enqueues a background job and returns at once** with `{status:"building", job:{...}}` — it does not build inline (see "Long Builds Are Background Jobs" above; doing so 502'd the whole app). **The payload is NESTED under `matrix`** — `matrix.strategies[id].cells[quadrant]` carries `{n, avg_period_return, win_rate, win_rate_ci_low/high, cum_return, confidence, tagged, verdict, verdict_detail}`, plus `matrix.strategies[id].overall`, `matrix.evidence_regimes` and `matrix.caveats`. Assigning the whole envelope to a variable and reading `.strategies` off it silently returns nothing — that exact bug once made the Playbook's evidence chips invisible even with a fully built matrix.
- Verdicts: `confirmed` / `unproven` / `mis-tagged` / `untagged-edge`. **A thin cell never produces a confident verdict** — a low-`n` losing cell is `unproven`, not `mis-tagged`. `evidence_regimes` carries only `confirmed` and `untagged-edge` cells forward: it answers "what has the data shown", not "what do we still believe".
- `GET /api/scorecard` — `{scorecard, execution_quality}`. Only a **trade_plan-mode** matrix yields a per-trade expectancy; a rotation-mode matrix returns no `expected_r` rather than converting a period return into a pseudo-R.
- `execution_quality` splits into `ranked[]` (computable, sorted by `r_cost` descending) and `unavailable[]` (each with `reason` + `needs`). **UI contract: an unavailable metric must never render as `0` or as a dash.** "Not measurable yet" and "measured, found nothing" have to be visually distinct — see `static/scorecard.html`. A missing field is not a clean bill of health.

Safety invariant for the regime override: a stale, empty or malformed matrix falls back to the hand tags and **can never return an empty strategy set**. Pinned by `test_edge.py`.

### Trade Plans (`services/trade_plan.py`)
`build_trade_plan(...)` produces an ATR-based entry zone, a stop (`entry − atr_mult×ATR`), a fixed-fractional position size from account size + risk %, and an R-multiple target. The returned plan also carries `initial_stop` (same value as `stop`) — the value that must be persisted to `PortfolioPosition.initial_stop` at entry, since it is the R denominator and must never be rewritten when the live stop is trailed up.

Optional `edge_multiplier` (default `1.0`) scales the risk budget (`risk_dollars = account_size × risk_pct/100 × edge_multiplier`) so a setup with a tested edge can be sized up and a marginal one down. It is clamped to **[0.5, 1.5]** (`clamp_edge_multiplier`, `EDGE_MULTIPLIER_MIN/MAX`) so a bad edge estimate cannot blow up sizing; non-numeric/NaN falls back to 1.0. The default reproduces the un-weighted sizing exactly (pinned by a test).

### Portfolio Risk (`services/portfolio_risk.py`)
Portfolio-level risk, all pure functions (no DB, no network — covered by `test_portfolio_risk.py`):
- `correlation_matrix(histories, window=90)` — Pearson on **DAILY RETURNS, never price levels**. This matters: two *independent* rising random walks measure −0.07 on returns but **0.77 on levels**, so a levels-based matrix would call every uptrending pair the same trade. Series are aligned by **DATE, not list index** — tickers have different listing dates, and zipping raw lists silently misaligns them. A pair with fewer than `MIN_OVERLAP` (30) shared observations returns `None`, not a number.
- `open_risk(positions, quotes)` — `shares × (price − stop)` floored at 0 (a stop above price is a free roll). A position with **no stop has unbounded risk**: it is never counted as zero, it is reported via `positions_without_stop` / `unstopped_warning`.
- `concentration(...)` — sector weights of market value plus MV-weighted beta, threshold-flagged, with `beta_coverage_pct` so a book of beta-less quotes is not silently reported as beta 1.
- `portfolio_heat(...)` — open R vs budget, slots vs `max_positions`.
- `assess_new_position(...)` — the pre-trade check behind the Plan-a-Trade screen. Returns `{level: "block"|"warn"|"info", code, message}` warnings plus the projected before → after state. Thresholds: correlation 0.70 warn / 0.85 block, sector weight 30% warn / 40% block, open R ≥80% of budget warn / over budget block, book at `max_positions` blocks.
- **Risk budget:** `max_open_r` is the user's CHOSEN ceiling when `User.max_open_r` is set, and otherwise falls back to the derived `max_positions × risk_pct` (`implied_max_open_r` — every slot at full risk). `api/settings.py::resolve_max_open_r(user) -> (budget, basis)` decides which, and `api/portfolio.py::_with_basis()` stamps the resulting `budget_basis` over the derived note this pure module emits, so **a derived number is never presented as a deliberate decision**. `services/portfolio_risk.py` stays pure and always emits the derived note; the real value is passed in.

### Today Dashboard (`services/today.py`)
Builds the `/api/today` payload (regime light, position health, top setups with trade plans, checklist, capacity). Exposes the shared `position_flags()` helper, also reused by `build_decision_cockpit`. Per-user in-memory cache (15-min TTL), invalidated by the scheduler.

`position_flags()` reads `quote["ann_ret_1m"]` (the 21-bar annualized return) for its "weak 1M annualized return" flag — **not** `ann_ret`, which is now the full-history annualized figure that Sharpe/Sortino/Calmar are built from.

### Weekly Plan (`services/weekly_plan.py`)
`build_weekly_plan(db, user)` turns the Playbook payload into decisions:
- **Position reviews** (`_review_position`) — `stop_hit` → recommended full **sell**; `target_hit` → recommended **trim** (half); `trend_break` → optional sell; held past the user's own `time_stop_days` with < 0.5R → recommended sell (past only the *strategy's* `horizon_days` → optional); otherwise **raise_stop** when the ATR trail is above the current stop, **watch** near the stop, or **hold**. Sell limits are the last price − 1% (`SELL_LIMIT_CUSHION`).
- **Buys** (`_buy_orders`) — setups walked in Playbook order through `portfolio_risk.assess_new_position` against a **hypothetical book** that already contains every buy picked before it (and excludes recommended full sells). So the second of two highly-correlated names, a sector pile-up, or a blown open-R budget is blocked even though each would pass alone. Also not pre-selected: `forming` setups, strategies the cached edge matrix marks `mis-tagged` in this regime, and anything past `MAX_NEW_BUYS_PER_WEEK` (4). Buy limit = top of the ATR entry zone (no chasing).
- **Order contract** (consumed by the Trade page): `{id: "buy:T"|"sell:T"|"trim:T", kind, side, ticker, shares:int, limit_price, est_value, recommended, skip_reason, headline, why[], strategy, strategy_name, plan:{stop, target, strategy, strategy_name, planned_entry, planned_entry_high, thesis, invalidation, time_stop_days}}`; buys also carry `trade_plan`, `risk_warnings[]`, `evidence_verdict`, `state`. The `plan` block is what gets persisted to `PortfolioPosition` on fill.
- The edge matrix is read from **cache only**, never built (same rule as the Playbook).

`/api/today` setups now also carry `rank` / `pool_size` ("ranked #2 of 31 that passed the rules"), the strategy's `metrics`, and `price`; the payload has a top-level `selection` block (scanned count, per-strategy pass counts, the cut-offs, and a prose `method`). `regime.above_200ma` is now included.

**Sector concentration is measured against max(book, account_size)** in `assess_new_position`. It used to be the invested book alone, so the first trade into an empty book was "100% of the book" and **blocked**, and a second name in a new sector (~50%) blocked too — a new or small account could never open a position through Plan-a-Trade. Pinned by `test_first_position_in_empty_book_is_not_a_sector_block`.

### Trade / Robinhood (`services/broker_service.py`, `services/brokers/robinhood_mcp.py`, `api/broker.py`)
Live trading goes through **Robinhood's official Agentic Trading MCP server** (`https://agent.robinhood.com/mcp/trading`), not the private app API and not `robin_stocks` (global session, `input()` prompts, unbounded polling, pickled tokens — unusable in a multi-user server). Verified 2026-09-24 by probing: unauthenticated → 401 with `resource_metadata`; the AS metadata at `/.well-known/oauth-authorization-server/mcp/trading` gives authorize `https://robinhood.com/oauth`, token `https://api.robinhood.com/oauth2/token/`, **RFC 7591 dynamic registration**, public client (`none`), PKCE S256, scope `internal`. The app is the MCP client:
- `POST /api/broker/connect` discovers metadata, registers a client once per redirect URI, stores a PKCE verifier + single-use `state` (10 min, user-bound) and returns `authorize_url`. `GET /api/broker/oauth/callback` takes **no bearer token** — `state` is the auth. Redirect URI = `PUBLIC_URL` (config / add-on option `public_url`) or the request origin via `X-Forwarded-*`. Tokens are Fernet-encrypted (`services/secrets_box.py`; key from `BROKER_ENCRYPTION_KEY` or `broker.key` next to the DB — **losing it means reconnecting**, not data loss).
- Tool names/schemas are **not public**: the client calls `tools/list` and `map_arguments()` maps our order fields onto each tool's `inputSchema` (synonyms, enum aliases, coercion). An unmappable **required** property fails the order with the schema in the error — never a silent guess. `GET /api/broker/tools` exposes what was discovered.
- **Safety invariants** (pinned in `test_broker.py`): paper mode makes no network call and never touches the portfolio; live mode needs a connected account + `confirm` to switch **and** `confirm:true` per submit; `review_equity_order` runs before every `place_equity_order` and a failed review places nothing; limit orders, whole shares, ≤20 per batch, limit within ±15% of the cached quote; a fill is applied to the portfolio exactly once (`applied_to_portfolio`), full sells journal through `api/portfolio.py::journal_close()` (the close endpoint's body, shared). No response ever carries a token. Paper buying power (Settings account size − holdings cost) only **warns**; live buying power blocks.
- **Only mock-tested.** Discovery/registration (incl. whether Robinhood accepts our redirect host), the OAuth round trip, every MCP tool call and the result parsing (order id/state/fill field names are best guesses) have never run against the real service. Robinhood may hold an order for in-app approval (`pending_approval`, text shown verbatim). When there's no order-status tool, `POST /orders/{id}/resolve` lets the user mark fills by hand.
- Models: `BrokerAccount` (one per user; client_id, redirect_uri, encrypted tokens, mode, tools cache) and `BrokerOrder` (the order log, with the plan intent in `plan_json`). New tables → `create_all`, no `_ensure_columns`.

### Backtesting (`api/backtest.py`)
Walk-forward backtest takes a `strategy` param — one of the **actionable** strategy ids (`BACKTESTABLE_STRATEGIES`); non-actionable, watchlist-only strategies such as `bear_reversal_watch` are rejected by the query pattern and again defensively inside `run_walk_forward_backtest`, which returns an explanatory `notes` payload rather than a 500. `source` can be `universe` (the whole cached universe). `GET /api/backtest/strategies` lists all strategies with a `backtestable` flag. The backtest also accepts `period` (1Y/2Y/all), `start_date`, `end_date`, and `archive` params to bound the test window.

**Two simulation modes (`mode`, default `rotation`).**
- `rotation` — the original equal-weight top-N rotation with a turnover cost, marked to market only on rebalance dates. It is a **frozen regression surface**: `test_backtest.py` pins its output to a recorded digest that was diffed against the pre-change engine. Do not change it without re-recording that digest deliberately.
- `trade_plan` — simulates what the user actually does live: each entry is sized by `services/trade_plan.py::build_trade_plan` (fixed-fractional risk + max-position-value cap), the book is capped at `max_positions` slots (rejected signals counted in `signals_skipped_full_book`), cash is tracked explicitly and **earns nothing**, and **exits are evaluated bar-by-bar between rebalances** by `services/exits.py`. Accepts `account_size` / `risk_pct` / `max_positions` / `atr_stop_mult` / `r_multiple` / `exit_rules` overrides, each falling back to the user's saved settings then to the same defaults `services/today.py::build_today` uses (10000, 1%, 8, 2.5, 2.0). Adds a `trade_log` (per-trade entry/exit, R-multiple, exit reason, bars held) and the metrics `avg_r`, `expectancy_r`, `win_rate_trades`, `trades_taken`, `signals_skipped_full_book`, `avg_bars_held`.

This distinction is the point: **`rotation` measures a different system than the one you trade**, so its CAGR never predicted live results. `trade_plan` is the mode to trust for sizing and strategy-selection decisions.

**No look-ahead is the cardinal rule.** Entries evaluate `strategy.candidate(bars, idx)` (which reads only `bars[:idx+1]`) and fill at that bar's close; exit rules are handed one enriched bar at a time and never the series. `test_backtest.py` asserts that truncating the series immediately after a decision bar does not change the decision made at that bar. Never relax this.

**Sample-size honesty.** Every `regime_breakdown` row carries a Wilson 95% interval (`win_rate_ci_low`/`win_rate_ci_high`) and a `confidence` label from the module constants `MIN_PERIODS_LOW_CONFIDENCE` (30) / `MIN_PERIODS_HIGH_CONFIDENCE` (100). Every response carries a structured top-level `caveats` list (`{id, severity, title, detail}`) covering survivorship bias (the universe is *today's* S&P 500 constituents, which inflates every result), which mode produced the result, and — in rotation mode — that stops/targets are not simulated. These are structured for the UI to render, not prose buried in `notes`.

**Cost & risk-free conventions.** Turnover cost is charged via `services/backtest.py::turnover_cost()`: `symmetric_difference` counts both sells and buys, so the round-trip charge is `(len(sym_diff) / top_n) * cost`, **uncapped** — a full rotation costs `2 × cost`, twice a 50% rotation. (It was previously clipped at `1.0 × cost`, which charged a complete rotation the same as a half one.) Sharpe and Sortino in `_metrics` subtract the module constant `RISK_FREE_RATE = 0.05`, de-annualized to the rebalance period as `(1 + rf)**(1/ppy) - 1`. **This must stay in sync with the same 5% rate in `services/indicators.py::compute_performance_metrics`**, or cross-page Sharpe comparisons become meaningless.

**Arbitrary-era backtests (`archive=true`)** route through `services/history_archive.py` / the `history_archive` table instead of the rolling StockCache: it fetches the requested `start_date`→`end_date` span (plus a ~480-day warmup buffer) from yfinance once, stores the widest range per ticker, and reuses it. Best with Watchlist/Portfolio sources (Full Universe = many on-demand fetches). Requires both dates.

### Database / Migrations
- New `ClosedTrade` model (`closed_trades` table) backs the trade journal — records realized `pnl`, `pnl_pct`, and `r_multiple` (when a stop was set).
- `User` gained trading-settings columns: `account_size`, `risk_pct`, `max_positions`, `atr_stop_mult`, `r_multiple`.
- `User.litellm_api_key` (nullable) gives each user their own LiteLLM virtual key so AI token usage is separate. The AI path (`api/ai.py` `llm_headers`/`call_chat_model`, `services/ai_service.py` `LiteLLMAIService`/`ai_service` dependency, `services/report_service.py` `_call_llm`/`generate_daily_report`) takes an optional `api_key` and falls back to the global config key when a user has none. The auth API never returns the raw key — only a `has_litellm_key` boolean. The 05:30 scheduler now generates a report for every user with their own key.
- `PortfolioPosition` gained `stop_loss`, `target`, `entry_date`, `strategy`.
- `PortfolioPosition` and `ClosedTrade` both gained the trade plan's INTENT, previously packed into `notes` as `THESIS:` / `INVALIDATION:` / `TIME STOP:` / `PLAN: entry <x>` prefixed lines: `planned_entry`, `planned_entry_high`, `thesis`, `invalidation`, `time_stop_days` (all nullable). `planned_entry`/`planned_entry_high` are **write-once** like `initial_stop` — they are the reference the fill is judged against by `services/scorecard.py`'s `entry_chasing` metric, so an edit must never rewrite them. `services/trade_plan.py::parse_plan_notes()` is the one decoder of the legacy prefixed-notes shape, and `main.py::backfill_plan_fields_from_notes()` runs it once inside `ensure_schema_migrations()`: it fills only NULL columns, **never touches `notes`** (the only free-form copy of a trade's reasoning), and is therefore idempotent.
- `User.max_open_r` (Float, nullable) — the CHOSEN open-R ceiling. NULL is meaningful and is deliberately **not** repaired to a default: it means "not chosen", and consumers fall back to the derived `max_positions × risk_pct`.
- `User.use_evidence_regimes` (Boolean, default False) — per-user opt-in for the edge-matrix regime override on the Playbook. See "Market Regime".
- `PortfolioPosition` and `ClosedTrade` both gained `initial_stop` (nullable) — the stop the position was **opened** with. It is written once (on creation, or the first time a stop is supplied) and never overwritten, so trailing a stop up cannot inflate the recorded R. `api/portfolio.py::compute_r_multiple(avg_cost, exit_price, initial_stop, stop_loss)` is the single source of that math and falls back to `stop_loss` for legacy rows where `initial_stop` is NULL.
- Closing a position (`POST /api/portfolio/{ticker}/close`) archives it to the journal and deletes it; `DELETE /api/portfolio/{ticker}` remains a non-journaled hard delete for correcting mistaken entries.
- Lightweight SQLite `ALTER` migrations are handled by `ensure_schema_migrations()` in `main.py` (renamed from `ensure_cache_columns`, now with a generic `_ensure_columns` helper).

### Frontend
Vanilla JS + Fetch API in `static/` — no build step. Files are served directly by FastAPI as static assets. Edit `.html` files directly; changes are live when using Docker volume mount (`./static:/app/static`).

### Ticker Symbols — Universal Rule
**Every ticker symbol displayed anywhere in the UI must be clickable and open a full detail view.** This is a hard requirement across all pages and all contexts (tables, charts, legends, correlation matrices, etc.).

- **`index.html` (screener):** Clicking a screener row opens the `StockDetail` React component (panel slide-in). Tickers in Portfolio/Watchlist panels call `handleSelectByTicker(ticker)`. Never render a bare ticker string in a table row without an `onClick` handler.
- **`charts.html`:** All ticker symbols call `openTickerDetail(ticker)` — a vanilla JS modal that fetches `/api/stocks/{ticker}` (quote) + `/api/stocks/{ticker}/history` (OHLCV + indicators) and renders a tabbed detail view (Chart, Metrics, Signals). The modal HTML lives at `#tdModal`, and `closeTickerModal()` tears it down.
- **New pages:** Must implement an equivalent detail trigger — either reuse the `openTickerDetail()` function from `charts.html` or the React `StockDetail` component from `index.html`.
- **Styling:** Use CSS class `.ticker-link` (`color:#f0a500`, `cursor:pointer`, `font-family:JetBrains Mono`) on `<span onclick="openTickerDetail(...)">` elements in vanilla-JS pages. Never use bare `<a href="/?ticker=...">` navigation for tickers — that loses page context.

### Chart / ETF Data Caching (`services/chart_service.py`)
The in-memory cache `_cache` uses `_cache_get(key)` which returns `None` on miss. Always use **truthy** checks (`if cached:`) — not `if cached is not None` — for list-typed caches (VIX, sectors, ETF groups). An empty `[]` from a failed previous fetch will otherwise block re-fetches until the 6-hour TTL expires. Macro data uses a dict so `if cached is not None` is acceptable there.

Sector ETF data (`get_sector_data`) uses individual `yf.Ticker().history()` calls via a `ThreadPoolExecutor`, not `yf.download()`. The bulk `yf.download()` approach was removed due to breaking changes in newer yfinance versions where MultiIndex column access patterns changed.

## Configuration

Copy `.env.example` to `.env`. `Settings` sets `extra = "ignore"`, so unknown keys left in a local `.env` — or in Home Assistant's **stored** add-on options after a setting is retired from `config.json` — are skipped instead of raising a pydantic `ValidationError` at import time, which previously prevented the app from starting at all. (HA keeps stored options that are no longer in the schema; they are inert, but they do linger in the add-on's config and are worth clearing.) Key variables:

| Variable | Default | Notes |
|---|---|---|
| `SECRET_KEY` | weak default | Change this — used for JWT signing |
| `ADMIN_USER` / `ADMIN_PASS` | `admin` / `changeme` | Synced to the configured admin account on startup |
| `AI_PROVIDER` | `litellm` | `none` \| `anthropic` \| `openai` \| `litellm` |
| `AI_API_KEY` | — | Required when provider is `anthropic` or `openai` |
| `AI_MODEL` | `tooling_high` | LiteLLM tier alias — see aiProxy's `CLAUDE.md` Tier Aliases |
| `FRED_API_KEY` | — | Optional; enables macro chart data |
| `LITELLM_URL` | `http://192.168.0.21:4000` | LiteLLM proxy base URL |
| `REPORT_MODEL` | `tooling_high` | LiteLLM tier alias for daily report generation |
| `PUBLIC_URL` | *(empty)* | Public origin for the Robinhood OAuth redirect (`public_url` add-on option). Blank = derived from the request |
| `BROKER_ENCRYPTION_KEY` | *(empty)* | Fernet key for stored broker tokens; blank = `broker.key` next to the DB |

For HAOS, config goes through the add-on UI (mapped to `/data/options.json`).

## Deployment Notes

- **Versioning before push**: Any push to the remote repo must include a Home Assistant-visible version bump so HA detects the update. Keep `config.json` (`version`), `build.json` (`io.hass.version`), and `main.py` (`FastAPI(... version=...)`) in sync. Do not push functional changes without updating these version fields. If HA still does not show the update after a normal patch bump, use a clearer next version bump (for example `1.5.9` -> `1.6.0`), push it, then tell the user to reload/check updates in the HA Add-on Store because HA can cache add-on repository metadata.
- **SSL**: Auto-generated self-signed cert on first run, stored in `./ssl/` (or `/data/ssl/` in HAOS). Persists across restarts.
- **Database**: `./data/swingtrader.db` (SQLite). Survives all restarts; back up by copying this file.
- **Scheduler**: APScheduler runs daily report generation at 05:30 local time using the configured AI provider.

## Adding New AI Providers

Subclass `AIService` in `services/ai_service.py` and implement:
- `analyze_stock(ticker, data) -> dict`
- `generate_signals(ticker, technicals) -> list[dict]`
- `chat(ticker, question, context) -> str`
- `summarize_sector(sector, stocks) -> str`

Then register the new class in the provider factory in `api/ai.py`.
