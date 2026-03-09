# SwingTrader

A self-hosted swing trading screener with:
- Live data via **Yahoo Finance** (yfinance)
- **HTTPS** with auto-generated self-signed certificate
- **Multi-user login** (JWT auth, bcrypt passwords)
- Per-user **watchlist** and **portfolio** tracking (SQLite)
- Technical indicators: RSI, MACD, Bollinger Bands, MA50/200, Golden/Death Cross
- F / T / M scoring system (Fundamental / Technical / Momentum)
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

---

## Configuration (.env)

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | *(weak default)* | JWT signing key — **change this!** |
| `ADMIN_USER` | `admin` | Initial admin username (first run only) |
| `ADMIN_PASS` | `changeme` | Initial admin password (first run only) |
| `PORT` | `8443` | HTTPS port |
| `AI_PROVIDER` | `none` | `none` \| `anthropic` \| `openai` |
| `AI_API_KEY` | *(empty)* | API key for your AI provider |
| `AI_MODEL` | *(default)* | Override model (e.g. `claude-opus-4-6`) |
| `QUOTE_CACHE_TTL` | `900` | Quote cache lifetime in seconds (15 min) |
| `HISTORY_CACHE_TTL` | `3600` | History cache lifetime in seconds (1 hr) |

---

## Enabling AI Analysis

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

## Home Assistant OS (HAOS) Deployment

### Option 1 — Docker via Terminal Add-on

1. Install the **SSH & Web Terminal** add-on in HAOS
2. Copy the `swingtrader/` directory to your HA machine (e.g. via Samba share)
3. In the terminal:
   ```bash
   cd /path/to/swingtrader
   docker-compose up -d
   ```
4. Access via `https://<HA-IP>:8443`

### Option 2 — Portainer Add-on

1. Install the **Portainer** add-on in HAOS
2. In Portainer → Stacks → Add Stack
3. Paste the contents of `docker-compose.yml`
4. Set environment variables in the Portainer UI
5. Deploy

### Port Forwarding (external access)

If you want to access SwingTrader from outside your LAN, forward port 8443 on your router to your HA machine's IP.
For production use, consider putting it behind a reverse proxy (e.g. NGINX Proxy Manager add-on) with a real SSL certificate.

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
