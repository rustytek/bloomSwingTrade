# SwingTrader Add-on Documentation

## Installation

1. In Home Assistant, go to **Settings → Add-ons → Add-on Store**
2. Click the three-dot menu (⋮) in the top-right → **Repositories**
3. Add: `https://github.com/rustytek/bloomSwingTrade`
4. Find **SwingTrader** in the store and click **Install**
5. Go to the **Configuration** tab and set your options (see below)
6. Click **Start**

## First Login

Open the Web UI button (or navigate to `https://<ha-ip>:8443`).

- If you see a certificate warning, click **Advanced → Proceed** — this is expected for a self-signed cert.
- Default credentials: `admin` / `changeme` *(change via the admin panel after first login)*

## Configuration Options

| Option | Description |
|---|---|
| `secret_key` | JWT signing key — **change this to a random 32+ character string** |
| `admin_user` | Username for the initial admin account (first run only) |
| `admin_pass` | Password for the initial admin account (first run only) |
| `use_ha_ssl` | Use HA's own SSL certificate instead of a self-signed cert |
| `certfile` | HA certificate file (only used when `use_ha_ssl` is `true`) |
| `keyfile` | HA key file (only used when `use_ha_ssl` is `true`) |
| `ai_provider` | AI provider: `none`, `anthropic`, `openai`, `ollama`, or `litellm` |
| `ai_api_key` | API key for your AI provider |
| `ai_model` | Model override (e.g. `claude-opus-4-6`) — leave blank for default |

| `litellm_url` | LiteLLM OpenAI-compatible base URL. If blank, the app falls back to `ollama_url` |
| `litellm_api_key` | Optional LiteLLM proxy key. If blank, the app falls back to `ai_api_key` |
| `fred_api_key` | Optional FRED API key for Macro & Liquidity charts: M2, Fed Funds, 2yr/10yr yields, and yield spread |

## Enabling FRED Macro Data

1. Get a free API key from `https://fred.stlouisfed.org`.
2. In Home Assistant, open **Settings -> Add-ons -> SwingTrader -> Configuration**.
3. Paste the key into `fred_api_key`.
4. Restart the add-on.

Without this key, the Macro & Liquidity section still loads but shows a notice and leaves FRED-backed charts blank.

## Using Your Home Assistant SSL Certificate

If you have the **Let's Encrypt** or **Duck DNS** add-on configured:

1. Set `use_ha_ssl: true`
2. Set `certfile` to `fullchain.pem` (or your cert filename)
3. Set `keyfile` to `privkey.pem` (or your key filename)
4. Restart the add-on

This uses HA's trusted certificate, eliminating the browser warning.

## Data Persistence

All data is stored in HA's persistent add-on storage (`/data/`).
- **Database**: `/data/swingtrader.db` (SQLite — survives updates and restarts)
- **SSL certs**: `/data/ssl/` (self-signed only; HA certs are read-only from `/ssl/`)

To back up your data, include the add-on data directory in your HA backup.

## Enabling AI Analysis

### LiteLLM
1. Set `ai_provider: litellm`.
2. Set `litellm_url` to your LiteLLM proxy base URL, for example `http://192.168.0.21:4000`.
3. Set `ai_model` and `report_model` to the LiteLLM model IDs you want to use, for example `ollama/qwen3.5:9-mlx`.
4. If your proxy requires auth, set `litellm_api_key`.
5. Restart the add-on.

### Anthropic Claude
1. Set `ai_provider: anthropic`
2. Set `ai_api_key` to your Anthropic API key
3. Optionally set `ai_model: claude-opus-4-6`
4. Restart the add-on

### OpenAI
1. Set `ai_provider: openai`
2. Set `ai_api_key` to your OpenAI API key
3. Optionally set `ai_model: gpt-4o`
4. Restart the add-on

The AI panel in the stock detail view will populate automatically once configured.

## Updating

In Home Assistant: **Settings → Add-ons → SwingTrader → Update**

Your database and SSL certificates are preserved across updates.

## Support

GitHub: https://github.com/rustytek/bloomSwingTrade
