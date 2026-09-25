"""
Daily OHLCV fetchers for the 20-year history archive: yfinance (primary) and
Tiingo (fallback), plus the checks that decide whether a download is usable.

Both return the SAME bar shape the rolling StockCache uses
(`{i, date, open, high, low, close, vol}`), split- and dividend-adjusted, so a
bar from either source means the same thing to every strategy.

Nothing here writes to the database — services/long_history.py owns storage.
"""
from __future__ import annotations

import logging
import math
import time
from datetime import date, timedelta

logger = logging.getLogger(__name__)

TIINGO_URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"

# Tiingo's free tier allows 50 requests/hour and 1,000/day. The backfill stays
# under the hourly cap per run; anything left over is picked up by the next run.
TIINGO_MAX_PER_RUN = 45
TIINGO_SPACING_SECONDS = 1.5


class TiingoRateLimited(Exception):
    """Tiingo answered 429 — stop using it for the rest of this run."""


def _round(v, nd=4):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, nd)


def clean_bars(bars: list[dict]) -> tuple[list[dict], int]:
    """Sort by date, de-duplicate (last wins), drop bars without a positive
    close. Returns (bars, dropped_count). Never invents a value."""
    by_date: dict[str, dict] = {}
    dropped = 0
    for b in bars or []:
        d = str(b.get("date") or "")[:10]
        close = _round(b.get("close"))
        if len(d) != 10 or close is None or close <= 0:
            dropped += 1
            continue
        by_date[d] = {
            "date": d,
            "open": _round(b.get("open")) or close,
            "high": _round(b.get("high")) or close,
            "low": _round(b.get("low")) or close,
            "close": close,
            "vol": int(_round(b.get("vol"), 0) or 0),
        }
    out = [by_date[d] for d in sorted(by_date)]
    for i, b in enumerate(out):
        b["i"] = i
    return out, dropped


def fetch_yfinance(ticker: str, start: str, end: str) -> list[dict]:
    """[start, end] inclusive from yfinance, adjusted. [] on any failure."""
    try:
        import yfinance as yf
        end_excl = (date.fromisoformat(end) + timedelta(days=1)).isoformat()
        hist = yf.Ticker(ticker).history(start=start, end=end_excl, interval="1d", auto_adjust=True)
        if hist is None or hist.empty:
            return []
        bars = [{
            "date": dt.strftime("%Y-%m-%d"),
            "open": row["Open"], "high": row["High"], "low": row["Low"],
            "close": row["Close"], "vol": row["Volume"],
        } for dt, row in hist.iterrows()]
        return clean_bars(bars)[0]
    except Exception as exc:  # noqa: BLE001 — one bad ticker must not stop a backfill
        logger.warning("yfinance fetch failed for %s: %s", ticker, exc)
        return []


def tiingo_symbol(ticker: str) -> str | None:
    """Tiingo writes class shares with a hyphen (brk-b) and has no indices."""
    t = (ticker or "").strip().upper()
    if not t or t.startswith("^"):
        return None
    return t.replace(".", "-").lower()


def fetch_tiingo(ticker: str, start: str, end: str, api_key: str, transport=None) -> list[dict]:
    """[start, end] from Tiingo's EOD endpoint using its ADJUSTED fields.

    Raises TiingoRateLimited on 429 so the caller stops spending the quota.
    Returns [] for an unknown ticker or any other failure."""
    sym = tiingo_symbol(ticker)
    if not sym or not api_key:
        return []
    import httpx
    try:
        with httpx.Client(timeout=30.0, transport=transport) as client:
            r = client.get(
                TIINGO_URL.format(ticker=sym),
                params={"startDate": start, "endDate": end, "resampleFreq": "daily", "format": "json"},
                headers={"Authorization": f"Token {api_key}", "Content-Type": "application/json"},
            )
    except httpx.HTTPError as exc:
        logger.warning("Tiingo request failed for %s: %s", ticker, exc)
        return []
    if r.status_code == 429:
        raise TiingoRateLimited("Tiingo rate limit reached (free tier: 50 requests/hour).")
    if r.status_code != 200:
        logger.warning("Tiingo %s for %s: %s", r.status_code, ticker, r.text[:200])
        return []
    try:
        rows = r.json()
    except ValueError:
        return []
    if not isinstance(rows, list):
        return []
    bars = [{
        "date": str(row.get("date", ""))[:10],
        "open": row.get("adjOpen"), "high": row.get("adjHigh"), "low": row.get("adjLow"),
        "close": row.get("adjClose"), "vol": row.get("adjVolume"),
    } for row in rows if isinstance(row, dict)]
    return clean_bars(bars)[0]


def assess(bars: list[dict], *, target_start: str, today: str,
           known_first: str | None = None, calendar: list[str] | None = None) -> dict:
    """Is this download usable? Returns {ok, first, last, bars, issues[], missing}.

    - `known_first`: the earliest date we KNOW the ticker traded (its 5-year
      cache start). A download starting materially later than that is
      truncated, not a young listing.
    - `calendar`: SPY's trading dates. More than 2% of them missing inside the
      ticker's own span means a holed download.
    """
    issues: list[str] = []
    if not bars:
        return {"ok": False, "first": None, "last": None, "bars": 0, "missing": 0,
                "issues": ["no data returned"]}
    first, last = bars[0]["date"], bars[-1]["date"]
    if known_first and first > (date.fromisoformat(known_first) + timedelta(days=7)).isoformat():
        issues.append(f"starts {first}, but the ticker is known to trade from {known_first} — truncated")
    if last < (date.fromisoformat(today) - timedelta(days=10)).isoformat():
        issues.append(f"ends {last} — more than 10 days old")
    missing = 0
    if calendar:
        have = {b["date"] for b in bars}
        span = [d for d in calendar if first <= d <= last]
        missing = sum(1 for d in span if d not in have)
        if span and missing / len(span) > 0.02:
            issues.append(f"{missing} of {len(span)} trading days missing")
    return {"ok": not issues, "first": first, "last": last, "bars": len(bars),
            "missing": missing, "issues": issues}


def better(a: tuple[list[dict], dict], b: tuple[list[dict], dict]) -> tuple[list[dict], dict]:
    """Pick the more complete of two (bars, assessment) pairs: usable beats not
    usable, then the earlier start, then fewer missing days. `a` wins ties, so
    pass the primary source first."""
    (ab, aa), (bb, ba) = a, b
    if aa["ok"] != ba["ok"]:
        return a if aa["ok"] else b
    if not bb:
        return a
    if not ab:
        return b
    if aa["first"] != ba["first"]:
        return a if aa["first"] < ba["first"] else b
    return a if aa["missing"] <= ba["missing"] else b


def pause(seconds: float) -> None:
    """Indirection so tests can make backfill pacing instant."""
    if seconds > 0:
        time.sleep(seconds)
