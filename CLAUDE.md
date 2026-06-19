# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SwingTrader is a self-hosted swing trading screener — a FastAPI backend + vanilla JS frontend deployed via Docker or as a Home Assistant OS (HAOS) native add-on. It screens S&P 500 and ETF tickers with technical indicators (RSI, MACD, Bollinger Bands, MA50/200, Golden/Death Cross), per-user watchlists/portfolios, and optional AI analysis via Anthropic Claude, OpenAI, or local Ollama.

## Common Commands

```bash
# Run locally (dev)
pip install -r requirements.txt
python main.py

# Docker (recommended)
docker-compose up --build -d
docker-compose logs -f swingtrader

# Run test suite
python test_passes.py
```

The app runs on HTTPS at `https://localhost:8443`. Swagger docs at `/api/docs`.

## Architecture

### Entry Points
- `haos_entry.py` — Used by Docker/HAOS; reads `/data/options.json` (HA config) or falls back to env vars
- `main.py` — FastAPI app: registers routers, initializes DB, generates SSL cert, starts uvicorn + APScheduler

### Request Flow
```
React SPA (static/*.html) → FastAPI (main.py)
  → auth/ (JWT login/register)
  → api/ (stocks, screener, watchlist, portfolio, ai, charts, today, settings, journal, backtest)
  → services/ (market_data, indicators, ai_service, chart_service, report_service,
               strategies, trade_plan, today)
  → SQLite via SQLAlchemy (database/)
```

### Page Routes
- `/` — Today dashboard (`static/today.html`): regime light, position health, top setups with trade plans, checklist, capacity. (Also the 404 catch-all fallback.)
- `/screener` — the screener (`static/index.html`, formerly served at `/`).
- `/journal` — trade journal (`static/journal.html`).
- `/charts` — chart dashboard (`static/charts.html`).
- `/admin` — user management (`static/admin.html`), admin-only. Create/edit/delete users, reset passwords, toggle admin, and set/clear each user's per-user LiteLLM key. The header "Users" link (in `common.js`) is shown only to admins.
- Shared front-end helpers live in `static/js/common.js`, served via a `/static` StaticFiles mount in `main.py`.

### Key Files
| File | Purpose |
|---|---|
| `config.py` | Pydantic Settings — all env vars with defaults |
| `database/models.py` | ORM models: User, WatchlistItem, PortfolioPosition, StockCache, AICache, ReportCache, ClosedTrade, HistoryArchive |
| `database/db.py` | SQLAlchemy engine, `get_db()` FastAPI dependency |
| `auth/deps.py` | `get_current_user` / `get_current_admin` JWT dependencies |
| `services/market_data.py` | yfinance wrapper, dual-layer cache (in-memory dict + SQLite), indicator calculation |
| `services/universe.py` | ~450 tickers: S&P 500 constituents + ETF lists |
| `services/ai_service.py` | Abstract `AIService` base + Mock/Anthropic/OpenAI/Ollama implementations |
| `services/indicators.py` | Technical indicators; includes `calc_atr(highs, lows, closes, period=14)` (Wilder-smoothed) |
| `services/strategies.py` | 3-strategy framework; registry `STRATEGIES` (see below) |
| `services/trade_plan.py` | `build_trade_plan(...)`: ATR entry zone, stop, fixed-fractional sizing, R-multiple target (see below) |
| `services/today.py` | Builds the `/api/today` payload; exposes `position_flags()` helper (see below) |
| `api/today.py` | `GET /api/today` — daily dashboard |
| `api/settings.py` | `GET`/`PUT /api/settings` — account_size, risk_pct, max_positions, atr_stop_mult, r_multiple |
| `api/journal.py` | `GET /api/journal` (stats + per-strategy breakdown + `equity_curve`: cumulative realized P&L over time from all closed trades), `DELETE /api/journal/{id}` |
| `api/backtest.py` | Walk-forward backtest; `GET /api/backtest/strategies` returns each strategy's `details` (horizon/how/rules/scoring/parameters), rendered as a detail panel on the backtest page |

### Caching Strategy
Two-layer cache: in-memory Python dict (`_mem_cache`) → SQLite `StockCache` table. Data is considered "fresh" if cached **after the most recent NYSE market close (4pm ET)** — not a rolling TTL. Quote and history TTLs are configurable via env vars but default to daily refresh.

History cache window is **5 years** in `services/market_data.py` (`get_history`/`_fetch_history_sync` defaults, and the `api/stocks.py` `/history` endpoint default — keep these in sync so a detail-view refetch can't shrink the shared cache). Existing shorter caches upgrade to 5y on the next post-close refresh (or a manual screener refresh). For backtests over **arbitrary historical eras** beyond the rolling window, see `services/history_archive.py` (below).

### Strategy Framework (`services/strategies.py`)
Registry `STRATEGIES` maps ids to `Strategy` instances:
- `momentum_rotation` — Clenow-style 90-bar exponential regression slope × R².
- `pullback_50ma` — buy dips to the 50MA in an uptrend.
- `breakout_volume` — 60-day-high breakout on >=1.5x volume.

Each `Strategy` exposes `candidate(bars, idx)` (backtest hook, uses `bars[:idx+1]`) and `scan(ticker, bars, quote)` (live hook returning a `Setup`).

### Trade Plans (`services/trade_plan.py`)
`build_trade_plan(...)` produces an ATR-based entry zone, a stop (`entry − atr_mult×ATR`), a fixed-fractional position size from account size + risk %, and an R-multiple target.

### Today Dashboard (`services/today.py`)
Builds the `/api/today` payload (regime light, position health, top setups with trade plans, checklist, capacity). Exposes the shared `position_flags()` helper, also reused by `build_decision_cockpit`. Per-user in-memory cache (15-min TTL), invalidated by the scheduler.

### Backtesting (`api/backtest.py`)
Walk-forward backtest now takes a `strategy` param (one of the 3 ids); `source` can be `universe` (the whole cached universe). `GET /api/backtest/strategies` lists available strategies. The backtest also accepts `period` (1Y/2Y/all), `start_date`, `end_date`, and `archive` params to bound the test window.

**Arbitrary-era backtests (`archive=true`)** route through `services/history_archive.py` / the `history_archive` table instead of the rolling StockCache: it fetches the requested `start_date`→`end_date` span (plus a ~480-day warmup buffer) from yfinance once, stores the widest range per ticker, and reuses it. Best with Watchlist/Portfolio sources (Full Universe = many on-demand fetches). Requires both dates.

### Database / Migrations
- New `ClosedTrade` model (`closed_trades` table) backs the trade journal — records realized `pnl`, `pnl_pct`, and `r_multiple` (when a stop was set).
- `User` gained trading-settings columns: `account_size`, `risk_pct`, `max_positions`, `atr_stop_mult`, `r_multiple`.
- `User.litellm_api_key` (nullable) gives each user their own LiteLLM virtual key so AI token usage is separate. The AI path (`api/ai.py` `llm_headers`/`call_chat_model`, `services/ai_service.py` `LiteLLMAIService`/`ai_service` dependency, `services/report_service.py` `_call_ollama`/`generate_daily_report`) takes an optional `api_key` and falls back to the global config key when a user has none. The auth API never returns the raw key — only a `has_litellm_key` boolean. The 05:30 scheduler now generates a report for every user with their own key.
- `PortfolioPosition` gained `stop_loss`, `target`, `entry_date`, `strategy`.
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

Copy `.env.example` to `.env`. Key variables:

| Variable | Default | Notes |
|---|---|---|
| `SECRET_KEY` | weak default | Change this — used for JWT signing |
| `ADMIN_USER` / `ADMIN_PASS` | `admin` / `changeme` | Synced to the configured admin account on startup |
| `AI_PROVIDER` | `litellm` | `none` \| `anthropic` \| `openai` \| `ollama` \| `litellm` |
| `AI_API_KEY` | — | Required when provider is `anthropic` or `openai` |
| `FRED_API_KEY` | — | Optional; enables macro chart data |
| `OLLAMA_URL` | `http://192.168.10.21:11434` | Local Ollama server |
| `REPORT_MODEL` | `deepseek-r1:8b` | Model for daily report generation |

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
