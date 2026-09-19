# SwingTrader

A self-hosted swing trading screener with:
- Live data via **Yahoo Finance** (yfinance)
- **HTTPS** with auto-generated self-signed certificate
- **Multi-user login** (JWT auth, bcrypt passwords)
- Per-user **watchlist** and **portfolio** tracking (SQLite)
- Technical indicators: RSI, MACD, Bollinger Bands, MA50/200, Golden/Death Cross
- F / T / M scoring system (Fundamental / Technical / Momentum)
- Decision cockpit: risk flags, opportunity queue, exit pressure, and signal changes
- Walk-forward backtest lab for a regime-aware relative-strength strategy
- **AI analysis hooks** — ready for Anthropic Claude or OpenAI (mock by default)
- Dark terminal aesthetic React UI

---

## Quick Start (Docker)

```bash
# 1. Clone / copy the project
cd swingtrader

# 2. Create your .env file (optional — defaults work for local use)
cp .env.example .env
# Edit .env to set a strong SECRET_KEY and change ADMIN_PASS

# 3. Build and start
docker-compose up --build -d

# 4. Open in browser (accept the self-signed cert warning)
https://localhost:8443
```

Default login: **admin** / **changeme**
Change the password via the admin panel or API after first login.

Key pages:
- `/` — screener, portfolio, watchlist, and stock detail views
- `/charts` — macro, breadth, correlation, annualized return, and Wyckoff views
- `/backtest` — decision cockpit and walk-forward strategy backtest
- `/report` — daily AI-assisted market report.

---

## Configuration (.env)

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | *(weak default)* | JWT signing key — **change this!** |
| `ADMIN_USER` | `admin` | Initial admin username (first run only) |
| `ADMIN_PASS` | `changeme` | Initial admin password (first run only) |
| `PORT` | `8443` | HTTPS port |
| `AI_PROVIDER` | `none` | `none` \| `anthropic` \| `openai` \| `litellm` |
| `AI_API_KEY` | *(empty)* | API key for direct OpenAI/Anthropic-style providers; not used for LiteLLM |
| `AI_MODEL` | `tooling_high` | LiteLLM tier alias — never a raw provider model name (e.g. not `ollama/qwen3.5:9-mlx`) |
| `LITELLM_URL` | *(empty)* | LiteLLM OpenAI-compatible base URL |
| `LITELLM_API_KEY` | *(empty)* | Required when `AI_PROVIDER=litellm`; use the restricted LiteLLM virtual key for this user/app |
| `FRED_API_KEY` | *(empty)* | Optional FRED key for Macro & Liquidity charts: M2, Fed Funds, 2yr/10yr yields |
| `QUOTE_CACHE_TTL` | `900` | Quote cache lifetime in seconds (15 min) |
| `HISTORY_CACHE_TTL` | `3600` | History cache lifetime in seconds (1 hr) |

---

## Enabling AI Analysis

### LiteLLM

LiteLLM is the recommended path — this app talks to it exclusively for AI analysis; it never calls Ollama or another model runtime directly. Use a stable tier alias for `AI_MODEL`/`REPORT_MODEL`, not a raw model name, so the physical model behind it can change without editing this config.

```env
AI_PROVIDER=litellm
LITELLM_URL=http://192.168.0.21:4000
AI_MODEL=tooling_high
REPORT_MODEL=tooling_high
```

If your LiteLLM proxy requires a key, set `LITELLM_API_KEY` to the user's restricted LiteLLM virtual key.

### Anthropic Claude

```env
AI_PROVIDER=anthropic
AI_API_KEY=sk-ant-...
AI_MODEL=claude-opus-4-6   # optional
```

Then implement `AnthropicAIService` in `services/ai_service.py` (template provided as comments).
Restart the container: `docker-compose restart`

### OpenAI

```env
AI_PROVIDER=openai
AI_API_KEY=sk-...
AI_MODEL=gpt-4o   # optional
```

Implement `OpenAIAIService` in `services/ai_service.py` (template in comments).

---

## Home Assistant OS (HAOS) — Native Add-on

SwingTrader ships as a proper HA add-on — no Portainer, no terminal, no docker-compose needed.

### Install

1. Go to **Settings → Add-ons → Add-on Store**
2. Click the **⋮ menu** (top-right) → **Repositories**
3. Add: `https://github.com/rustytek/bloomSwingTrade`
4. Find **SwingTrader** in the store → **Install**
5. Open the **Configuration** tab — set a strong `secret_key` and change `admin_pass`
6. Click **Start** → open the **Web UI**

Optional: set `fred_api_key` in the add-on **Configuration** tab to enable Macro & Liquidity charts.

On first launch, a self-signed SSL cert is auto-generated. Accept the browser warning once, and you're in.

### FRED Macro Data

The HA add-on reads `fred_api_key` from the add-on **Configuration** tab and maps it to `FRED_API_KEY` inside the container. Set this value if you want `/charts` to populate M2 Money Supply, Fed Funds, 2yr/10yr Treasury yields, and yield spread. After changing it, restart the add-on.

### Use Your HA SSL Certificate (optional)

If you have Let's Encrypt configured in HA:

1. Set `use_ha_ssl: true`
2. `certfile: fullchain.pem` / `keyfile: privkey.pem` (defaults)
3. Restart — no more cert warning

### Updating

**Settings → Add-ons → SwingTrader → Update**. Your database survives all updates.

### Port Forwarding (external access)

Forward port `8443` on your router to your HA machine IP. For external HTTPS with a trusted cert, put it behind NGINX Proxy Manager (HA add-on).

---

## API Reference

Full interactive docs available at: `https://localhost:8443/api/docs`

### Auth
| Method | Endpoint | Description |
|---|---|---|
| POST | `/auth/login` | Get JWT token |
| GET | `/auth/me` | Current user info |
| POST | `/auth/register` | Create user (admin only) |
| GET | `/auth/users` | List users (admin only) |
| PUT | `/auth/users/{id}` | Update user (admin only) |
| DELETE | `/auth/users/{id}` | Delete user (admin only) |

### Market Data
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/stocks/{ticker}` | Enriched quote + indicators |
| GET | `/api/stocks/{ticker}/history` | OHLCV bars + indicator series |
| POST | `/api/stocks/batch` | Multiple tickers in one call |

### Screener
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/screener` | Filter watchlist with parameters |

### Watchlist
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/watchlist` | List watchlist tickers |
| POST | `/api/watchlist` | Add ticker |
| POST | `/api/watchlist/batch` | Add multiple tickers |
| DELETE | `/api/watchlist/{ticker}` | Remove ticker |

### Portfolio
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/portfolio` | Positions with live P&L |
| POST | `/api/portfolio` | Add/update position |
| DELETE | `/api/portfolio/{ticker}` | Remove position |
| POST | `/api/portfolio/import` | Import Fidelity CSV |

### AI (hooks)
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/ai/status` | AI provider status |
| GET | `/api/ai/{ticker}/analysis` | Full AI analysis |
| GET | `/api/ai/{ticker}/signals` | AI trading signals |
| POST | `/api/ai/{ticker}/chat` | Chat about a stock |

---

## User Management

Admins can manage users via the API (see docs) or directly in the SQLite DB.

```bash
# Create a new user via API (requires admin JWT)
curl -k -X POST https://localhost:8443/auth/register \
  -H "Authorization: Bearer <admin-token>" \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "password": "secret", "is_admin": false}'
```

---

## Data Persistence

All data is stored in `./data/swingtrader.db` (SQLite).
The `./data/` directory is mounted as a Docker volume — data survives container restarts and updates.

To back up: `cp data/swingtrader.db data/swingtrader.db.bak`

---

## Accepting the Self-Signed Certificate

Since the cert is self-signed, browsers will warn you on first visit.

| Browser | Steps |
|---|---|
| Chrome | Click "Advanced" → "Proceed to localhost (unsafe)" |
| Firefox | Click "Advanced" → "Accept the Risk and Continue" |
| Safari | Click "Visit Website" in the warning dialog |
| iOS Safari | Settings → General → About → Certificate Trust Settings → enable trust |

The cert is valid for 10 years, so you only need to accept it once per browser/device.

---

## Project Structure

```
swingtrader/
├── main.py               # FastAPI app + HTTPS startup
├── config.py             # Settings from env vars
├── generate_ssl.py       # Self-signed cert generator
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── database/
│   ├── db.py             # SQLAlchemy engine + session
│   └── models.py         # ORM models
├── auth/
│   ├── router.py         # Auth endpoints
│   ├── deps.py           # get_current_user dependency
│   └── utils.py          # JWT + bcrypt
├── api/
│   ├── stocks.py         # Market data endpoints
│   ├── screener.py       # Screening endpoint
│   ├── watchlist.py      # Watchlist CRUD
│   ├── portfolio.py      # Portfolio CRUD + CSV import
│   └── ai.py             # AI hooks endpoints
├── services/
│   ├── market_data.py    # yfinance wrapper + cache
│   ├── indicators.py     # RSI, MACD, MA, BB, scoring
│   └── ai_service.py     # AI abstraction + MockAIService
├── static/
│   ├── index.html        # React SPA (main screener)
│   └── login.html        # Login page
├── data/                 # SQLite DB (Docker volume)
└── ssl/                  # SSL certs (Docker volume)
```
