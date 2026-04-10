"""
Chart Data Service
==================
Provides chart-ready JSON for the dashboard and LLM report context.

Data sources:
  - FRED API  : M2, Fed Funds Rate, 2yr/10yr yields
  - yfinance  : VIX (^VIX), sector ETFs (XLK, XLE, …)
  - Screener  : Market breadth computed from the existing StockCache
                (no external API — uses data we already have)

All FRED/yfinance calls are cached in-memory for 6 hours so the
dashboard loads instantly on repeat visits.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta

import httpx
import yfinance as yf
from sqlalchemy.orm import Session

from config import get_settings
from database.models import StockCache
from services.universe import SP500

logger = logging.getLogger(__name__)
settings = get_settings()

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
CACHE_TTL = timedelta(hours=6)

# ── In-memory cache { key: (data, expires_at) } ──────────────────────────────
_cache: dict[str, tuple] = {}


def _cache_get(key: str):
    entry = _cache.get(key)
    if entry and datetime.now(timezone.utc) < entry[1]:
        return entry[0]
    return None


def _cache_set(key: str, data):
    _cache[key] = (data, datetime.now(timezone.utc) + CACHE_TTL)


# ── FRED helpers ──────────────────────────────────────────────────────────────

async def _fred(series_id: str, limit: int = 60) -> list[dict]:
    """Fetch observations from FRED. Returns [{date, value}] oldest→newest."""
    cached = _cache_get(f"fred:{series_id}")
    if cached is not None:
        return cached

    params = {
        "series_id": series_id,
        "api_key": settings.fred_api_key,
        "file_type": "json",
        "limit": limit,
        "sort_order": "desc",
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(FRED_BASE, params=params)
        resp.raise_for_status()
        raw = resp.json()

    result = [
        {"date": o["date"], "value": float(o["value"])}
        for o in raw.get("observations", [])
        if o.get("value", ".") != "."
    ]
    result.reverse()   # oldest first for charts
    _cache_set(f"fred:{series_id}", result)
    return result


# ── Macro data (FRED) ─────────────────────────────────────────────────────────

async def get_macro_data() -> dict:
    """
    Returns M2, Fed Funds Rate, 2yr yield, 10yr yield, yield spread.
    All FRED series, cached 6 hours.
    """
    cached = _cache_get("macro")
    if cached:
        return cached

    m2, fedfunds, dgs2, dgs10 = await asyncio.gather(
        _fred("M2SL",      limit=48),   # 4 years monthly
        _fred("FEDFUNDS",  limit=48),   # 4 years monthly
        _fred("DGS2",      limit=756),  # ~3 years daily
        _fred("DGS10",     limit=756),
    )

    # Yield spread: 10yr − 2yr (inversion < 0 = recession signal)
    dgs2_map = {o["date"]: o["value"] for o in dgs2}
    spread = [
        {"date": o["date"], "value": round(o["value"] - dgs2_map[o["date"]], 3)}
        for o in dgs10 if o["date"] in dgs2_map
    ]

    # Trend signal from last 3 monthly M2 readings
    m2_trend = "flat"
    if len(m2) >= 4:
        if m2[-1]["value"] > m2[-4]["value"] * 1.002:
            m2_trend = "rising"
        elif m2[-1]["value"] < m2[-4]["value"] * 0.998:
            m2_trend = "falling"

    current_spread = spread[-1]["value"] if spread else None
    result = {
        "m2":              m2,
        "fedfunds":        fedfunds,
        "dgs2":            dgs2,
        "dgs10":           dgs10,
        "yield_spread":    spread,
        "m2_trend":        m2_trend,
        "m2_current":      round(m2[-1]["value"], 1) if m2 else None,
        "fed_rate":        fedfunds[-1]["value"] if fedfunds else None,
        "yield_2yr":       dgs2[-1]["value"] if dgs2 else None,
        "yield_10yr":      dgs10[-1]["value"] if dgs10 else None,
        "yield_spread_now": current_spread,
        "yield_inverted":  current_spread is not None and current_spread < 0,
    }
    _cache_set("macro", result)
    return result


# ── VIX (yfinance) ────────────────────────────────────────────────────────────

async def get_vix_data() -> list[dict]:
    """VIX 90-day daily history."""
    cached = _cache_get("vix")
    if cached:  # truthy check — avoids serving a stale empty list
        return cached

    def _fetch():
        hist = yf.Ticker("^VIX").history(period="3mo", interval="1d", auto_adjust=True)
        if hist.empty:
            return []
        return [
            {"date": dt.strftime("%Y-%m-%d"), "value": round(float(row["Close"]), 2)}
            for dt, row in hist.iterrows()
        ]

    data = await asyncio.to_thread(_fetch)
    _cache_set("vix", data)
    return data


# ── Sector rotation (yfinance) ────────────────────────────────────────────────

SECTOR_ETFS = {
    "XLK":  "Technology",
    "XLF":  "Financials",
    "XLE":  "Energy",
    "XLV":  "Health Care",
    "XLI":  "Industrials",
    "XLY":  "Cons. Discr.",
    "XLP":  "Cons. Staples",
    "XLU":  "Utilities",
    "XLRE": "Real Estate",
    "XLB":  "Materials",
    "XLC":  "Comm. Services",
}


async def get_sector_data() -> list[dict]:
    """5d / 1m / 3m returns for all sector ETFs."""
    cached = _cache_get("sectors")
    if cached:  # truthy check — avoids serving a stale empty list
        return cached

    def _fetch_one(item):
        ticker, name = item
        try:
            hist = yf.Ticker(ticker).history(period="3mo", interval="1d", auto_adjust=True)
            if hist.empty:
                return None
            closes = hist["Close"].dropna()
            if len(closes) < 2:
                return None
            c = closes.values
            return {
                "ticker": ticker,
                "name":   name,
                "price":  round(float(c[-1]), 2),
                "ret_5d": round((c[-1] / c[-6]  - 1) * 100, 2) if len(c) >= 6  else None,
                "ret_1m": round((c[-1] / c[-22] - 1) * 100, 2) if len(c) >= 22 else None,
                "ret_3m": round((c[-1] / c[0]   - 1) * 100, 2),
            }
        except Exception as e:
            logger.warning("Sector fetch error %s: %s", ticker, e)
            return None

    def _fetch():
        from concurrent.futures import ThreadPoolExecutor
        results = []
        with ThreadPoolExecutor(max_workers=6) as pool:
            for result in pool.map(_fetch_one, SECTOR_ETFS.items()):
                if result:
                    results.append(result)
        return sorted(results, key=lambda x: x.get("ret_1m") or -999, reverse=True)

    data = await asyncio.to_thread(_fetch)
    _cache_set("sectors", data)
    return data


# ── ETF Groups (international / commodities / style / sector indices) ─────────

ETF_GROUPS = {
    "international": {
        "SPY":  "S&P 500 (benchmark)",
        "VXUS": "All-World ex-US",
        "VEA":  "Developed Markets",
        "VGK":  "Europe",
        "VPL":  "Pacific",
        "VWO":  "Emerging Mkts",
        "IEMG": "Core Emerging Mkts",
        "FDT":  "Dev Mkts ex-US SC",
        "AVDV": "Intl Small Value",
        "SCHY": "Intl Dividend",
        "EFV":  "Intl Value",
    },
    "commodities": {
        "GLD":  "Gold",
        "SLV":  "Silver",
        "GSG":  "S&P GSCI Cmdty",
        "COMT": "PIMCO Cmdty",
        "DBC":  "DB Commodity",
        "PDBC": "Optimum Yield",
        "GCC":  "Continuous Cmdty",
        "USCI": "US Commodity Idx",
        "USO":  "Crude Oil",
        "UNG":  "Natural Gas",
        "GDX":  "Gold Miners",
        "GDXJ": "Jr Gold Miners",
    },
    "style": {
        "SPY":  "S&P 500",
        "SPYG": "S&P 500 Growth",
        "SPYV": "S&P 500 Value",
        "QQQ":  "Nasdaq 100",
        "IWM":  "Russell 2000",
        "MDYG": "MidCap Growth",
        "MDYV": "MidCap Value",
        "SLYG": "SmallCap Growth",
        "SLYV": "SmallCap Value",
        "IWC":  "Micro Cap",
    },
    "sector_indices": {
        "^SOX":    "Philadelphia Semiconductor",
        "^BKX":    "KBW Banking Index",
        "^XAU":    "Gold/Silver Mining",
        "^OSX":    "PHLX Oil Service",
        "^BTK":    "AMEX Biotech",
        "^XBD":    "AMEX Broker/Dealer",
        "^DRG":    "NYSE Arca Pharma",
        "^DJUSRT": "DJ US Retail",
    },
}


async def get_etf_group_data() -> dict:
    """Returns 5d/1m/3m returns for all ETF groups, cached 6 hours."""
    cached = _cache_get("etf_groups")
    if cached:  # truthy check — avoids serving a stale empty dict
        return cached

    def _fetch_one(ticker: str):
        try:
            hist = yf.Ticker(ticker).history(period="3mo", interval="1d", auto_adjust=True)
            if hist.empty:
                return ticker, None
            closes = hist["Close"].dropna()
            if len(closes) < 2:
                return ticker, None
            c = closes.values
            return ticker, {
                "price":  round(float(c[-1]), 2),
                "ret_5d": round((c[-1] / c[-6]  - 1) * 100, 2) if len(c) >= 6  else None,
                "ret_1m": round((c[-1] / c[-22] - 1) * 100, 2) if len(c) >= 22 else None,
                "ret_3m": round((c[-1] / c[0]   - 1) * 100, 2),
            }
        except Exception as e:
            logger.warning("ETF group fetch error %s: %s", ticker, e)
            return ticker, None

    def _fetch():
        # Collect all unique tickers across all groups
        all_tickers = list({t for g in ETF_GROUPS.values() for t in g})
        prices: dict = {}
        # Use a thread pool for parallel fetches
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as pool:
            for ticker, data in pool.map(_fetch_one, all_tickers):
                if data:
                    prices[ticker] = data

        result = {}
        for group_name, etfs in ETF_GROUPS.items():
            items = []
            for ticker, name in etfs.items():
                if ticker in prices:
                    items.append({"ticker": ticker, "name": name, **prices[ticker]})
            result[group_name] = sorted(items, key=lambda x: x.get("ret_1m") or -999, reverse=True)
        return result

    data = await asyncio.to_thread(_fetch)
    _cache_set("etf_groups", data)
    return data


# ── Market breadth (screener DB — no external API) ────────────────────────────

def get_breadth_data(db: Session) -> dict:
    """
    Compute S&P 500 breadth from the screener's StockCache.
    Uses data already cached by the universe refresh — zero extra API calls.
    """
    sp500_set = set(SP500)
    rows = db.query(StockCache).filter(StockCache.ticker.in_(list(sp500_set))).all()

    total = above_50 = above_200 = adv = dec = 0
    sector_map: dict[str, dict] = {}

    for row in rows:
        if not row.quote_json:
            continue
        try:
            q = json.loads(row.quote_json)
        except Exception:
            continue

        total += 1
        vs50  = q.get("vs_ma50")
        vs200 = q.get("vs_ma200")
        chg   = q.get("chg_pct")
        sec   = q.get("sector") or "Unknown"

        if vs50  is not None and vs50  > 0: above_50  += 1
        if vs200 is not None and vs200 > 0: above_200 += 1
        if chg is not None:
            if chg > 0: adv += 1
            elif chg < 0: dec += 1

        s = sector_map.setdefault(sec, {"total": 0, "above_200": 0, "adv": 0})
        s["total"] += 1
        if vs200 is not None and vs200 > 0: s["above_200"] += 1
        if chg is not None and chg > 0:     s["adv"]       += 1

    pct200 = round(above_200 / total * 100, 1) if total else 0
    regime = (
        "Bull"    if pct200 >= 70 else
        "Neutral" if pct200 >= 50 else
        "Caution" if pct200 >= 30 else
        "Bear"
    )

    sector_breadth = sorted(
        [
            {
                "sector":       sec,
                "total":        d["total"],
                "pct_above_200": round(d["above_200"] / d["total"] * 100, 1) if d["total"] else 0,
                "pct_adv":      round(d["adv"]       / d["total"] * 100, 1) if d["total"] else 0,
            }
            for sec, d in sector_map.items() if d["total"] > 0
        ],
        key=lambda x: x["pct_above_200"],
        reverse=True,
    )

    return {
        "total_stocks":    total,
        "pct_above_50ma":  round(above_50  / total * 100, 1) if total else 0,
        "pct_above_200ma": pct200,
        "advancing":       adv,
        "declining":       dec,
        "ad_ratio":        round(adv / max(dec, 1), 2),
        "regime":          regime,
        "sector_breadth":  sector_breadth,
    }


# ── Convenience: everything in one async call ─────────────────────────────────

async def get_all_chart_data(db: Session) -> dict:
    macro, vix, sectors, etf_groups = await asyncio.gather(
        get_macro_data(),
        get_vix_data(),
        get_sector_data(),
        get_etf_group_data(),
    )
    breadth = get_breadth_data(db)
    return {"macro": macro, "vix": vix, "sectors": sectors, "breadth": breadth, "etf_groups": etf_groups}
