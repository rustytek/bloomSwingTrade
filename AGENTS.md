# CLAUDE.md

This file provides guidance to agents when working with code in this repository.

## Project Overview

SwingTrader is a self-hosted swing trading screener â€” a FastAPI backend + vanilla JS frontend deployed via Docker or as a Home Assistant OS (HAOS) native add-on. It screens S&P 500 and ETF tickers with technical indicators (RSI, MACD, Bollinger Bands, MA50/200, Golden/Death Cross), per-user watchlists/portfolios, and optional AI analysis via Anthropic Claude, OpenAI, or LiteLLM (the local/self-hosted path â€” see aiProxy's `CLAUDE.md`). This app never calls Ollama or any other model runtime directly; `AI_MODEL`/`REPORT_MODEL` should always be a LiteLLM tier alias (e.g. `tooling_high`), not a raw provider model name.

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
- `haos_entry.py` â€” Used by Docker/HAOS; reads `/data/options.json` (HA config) or falls back to env vars
- `main.py` â€” FastAPI app: registers routers, initializes DB, generates SSL cert, starts uvicorn + APScheduler

### Request Flow
```
React SPA (static/*.html) â†’ FastAPI (main.py)
  â†’ auth/ (JWT login/register)
  â†’ api/ (stocks, screener, watchlist, portfolio, ai, charts)
  â†’ services/ (market_data, indicators, ai_service, chart_service, report_service)
  â†’ SQLite via SQLAlchemy (database/)
```

### Key Files
| File | Purpose |
|---|---|
| `config.py` | Pydantic Settings â€” all env vars with defaults |
| `database/models.py` | ORM models: User, WatchlistItem, PortfolioPosition, StockCache, AICache, ReportCache |
| `database/db.py` | SQLAlchemy engine, `get_db()` FastAPI dependency |
| `auth/deps.py` | `get_current_user` / `get_current_admin` JWT dependencies |
| `services/market_data.py` | yfinance wrapper, dual-layer cache (in-memory dict + SQLite), indicator calculation |
| `services/universe.py` | ~450 tickers: S&P 500 constituents + ETF lists |
| `services/ai_service.py` | Abstract `AIService` base + Mock/LiteLLM implementations (Anthropic/OpenAI stubbed, not yet implemented) |

### Caching Strategy
Two-layer cache: in-memory Python dict (`_mem_cache`) â†’ SQLite `StockCache` table. Data is considered "fresh" if cached **after the most recent NYSE market close (4pm ET)** â€” not a rolling TTL. Quote and history TTLs are configurable via env vars but default to daily refresh.

### Frontend
Vanilla JS + Fetch API in `static/` â€” no build step. Files are served directly by FastAPI as static assets. Edit `.html` files directly; changes are live when using Docker volume mount (`./static:/app/static`).

### Ticker Symbols â€” Universal Rule
**Every ticker symbol displayed anywhere in the UI must be clickable and open a full detail view.** This is a hard requirement across all pages and all contexts (tables, charts, legends, correlation matrices, etc.).

- **`index.html` (screener):** Clicking a screener row opens the `StockDetail` React component (panel slide-in). Tickers in Portfolio/Watchlist panels call `handleSelectByTicker(ticker)`. Never render a bare ticker string in a table row without an `onClick` handler.
- **`charts.html`:** All ticker symbols call `openTickerDetail(ticker)` â€” a vanilla JS modal that fetches `/api/stocks/{ticker}` (quote) + `/api/stocks/{ticker}/history` (OHLCV + indicators) and renders a tabbed detail view (Chart, Metrics, Signals). The modal HTML lives at `#tdModal`, and `closeTickerModal()` tears it down.
- **New pages:** Must implement an equivalent detail trigger â€” either reuse the `openTickerDetail()` function from `charts.html` or the React `StockDetail` component from `index.html`.
- **Styling:** Use CSS class `.ticker-link` (`color:#f0a500`, `cursor:pointer`, `font-family:JetBrains Mono`) on `<span onclick="openTickerDetail(...)">` elements in vanilla-JS pages. Never use bare `<a href="/?ticker=...">` navigation for tickers â€” that loses page context.

### Chart / ETF Data Caching (`services/chart_service.py`)
The in-memory cache `_cache` uses `_cache_get(key)` which returns `None` on miss. Always use **truthy** checks (`if cached:`) â€” not `if cached is not None` â€” for list-typed caches (VIX, sectors, ETF groups). An empty `[]` from a failed previous fetch will otherwise block re-fetches until the 6-hour TTL expires. Macro data uses a dict so `if cached is not None` is acceptable there.

Sector ETF data (`get_sector_data`) uses individual `yf.Ticker().history()` calls via a `ThreadPoolExecutor`, not `yf.download()`. The bulk `yf.download()` approach was removed due to breaking changes in newer yfinance versions where MultiIndex column access patterns changed.

## Configuration

Copy `.env.example` to `.env`. Key variables:

| Variable | Default | Notes |
|---|---|---|
| `SECRET_KEY` | weak default | Change this â€” used for JWT signing |
| `ADMIN_USER` / `ADMIN_PASS` | `admin` / `changeme` | Synced to the configured admin account on startup |
| `AI_PROVIDER` | `litellm` | `none` \| `anthropic` \| `openai` \| `litellm` |
| `AI_API_KEY` | â€” | Required when provider is `anthropic` or `openai` |
| `AI_MODEL` | `tooling_high` | LiteLLM tier alias â€” see aiProxy's `CLAUDE.md` Tier Aliases |
| `FRED_API_KEY` | â€” | Optional; enables macro chart data |
| `LITELLM_URL` | `http://192.168.0.21:4000` | LiteLLM proxy base URL |
| `REPORT_MODEL` | `tooling_high` | LiteLLM tier alias for daily report generation |

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

