"""
Daily Market Analysis Report Service.

Workflow:
  1. Load portfolio positions + watchlist for a user from the DB
  2. Fetch enriched quotes (price, technicals, performance metrics) for all tickers
  3. Compute "Same Shape" correlation groups from 30-day price history
  4. Build a structured context payload and send to Ollama (report_model)
  5. Store the resulting markdown in ReportCache (last 10 per user)
  6. Return the markdown string
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx
import numpy as np
from sqlalchemy.orm import Session

from config import get_settings
from database.db import SessionLocal
from database.models import PortfolioPosition, WatchlistItem, StockCache, ReportCache
from services.chart_service import get_all_chart_data

logger = logging.getLogger(__name__)
settings = get_settings()


def current_settings():
    return get_settings()


# ── Sector keyword sets for correlation group labelling ───────────────────────

_ENERGY = {
    "CVX", "XOM", "COP", "DVN", "OXY", "HAL", "SLB", "BKR", "MPC", "VLO",
    "PSX", "EOG", "FANG", "APA", "WMB", "OKE", "KMI", "LNG", "TRGP", "EQT",
    "RRC", "AR", "AROC", "LBRT", "PARR", "XLE", "VDE", "USO", "UNG",
}
_COMMODITY = {
    "GLD", "SLV", "GDX", "GDXJ", "IAU", "PDBC", "DBA", "FCX", "NEM",
    "ALB", "CF", "MOS", "GOLD",
}
_TECH = {
    "NVDA", "AMD", "INTC", "SMCI", "AVGO", "AAPL", "MSFT", "META", "GOOGL",
    "GOOG", "AMZN", "TSLA", "PLTR", "MU", "SNDK", "QCOM", "TXN", "AMAT",
    "LRCX", "KLAC", "SOXX", "SMH", "QQQ", "TQQQ", "SOXL", "APP", "NOW",
    "CRM", "ADBE", "INTU", "PANW",
}


# ── Prompt & Template ─────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a professional swing-trading analyst generating a structured daily market report.

Rules:
1. Follow the template EXACTLY — replace every [PLACEHOLDER] with real analysis or data.
2. Use actual ticker symbols and numbers from the JSON data provided.
3. All macro data (M2, Fed Funds, yields, VIX, breadth) is provided as LIVE DATA — use the exact values.
4. Wyckoff Signal logic (use vol_r field — volume ratio vs 20-day avg):
   TOP signal  : price rising (chg_pct > 0) AND vol_r < 0.85  →  distribution
               OR price falling (chg_pct < 0) AND vol_r > 1.2  →  panic selling into weakness
   BOTTOM signal: price rising (chg_pct > 0) AND vol_r > 1.2  →  accumulation breakout
               OR price falling (chg_pct < 0) AND vol_r < 0.85 →  low-volume pullback / shakeout
   NEUTRAL: mixed or no clear signal
5. Sell/Exit candidates: ann_ret (1-month annualized) < 10% OR sharpe < 0.5
6. Annualized return formula: cost_multiplier ^ (52 / weeks_held)  then convert to %
7. Keep the markdown table formatting intact.
8. "Same Shape" groups are PRE-COMPUTED — use the provided correlation_groups directly.
"""

_TEMPLATE = """\
# [{date}] Market Analysis & Execution

## 1. Macro Regime & Liquidity

**M2 Money Supply:** ${m2_current}B — Trend: **{m2_trend}**
**Fed Funds Rate:** {fed_rate}%
**2yr Yield:** {yield_2yr}% | **10yr Yield:** {yield_10yr}% | **Spread (10y-2y):** {yield_spread_now:+.3f}% {yield_inverted_flag}
**VIX:** {vix_current:.2f} — Zone: **{vix_zone}**

**Liquidity Status:** [ANALYZE: Based on M2 trend={m2_trend} and Fed Funds={fed_rate}%, describe current liquidity environment and implications for equities]

**Fed Signaling:** [ANALYZE: With 2yr at {yield_2yr}% and the yield curve {yield_curve_state}, describe likely Fed path and market implications]

**Reversal Signs (Wyckoff):**
- Top/Bear: Price rises with falling volume / Price falls with rising volume.
- Bottom/Bull: Price rises with rising volume / Price falls with falling volume.
- **Current Portfolio Signal:** [STATE: TOP / BOTTOM / NEUTRAL — cite 2-3 specific tickers and their vol_r + chg_pct to justify]

---

## 2. Market Breadth & Ranking

**S&P 500 Breadth:** {pct_above_200ma}% above 200MA | {pct_above_50ma}% above 50MA | A/D Ratio: {ad_ratio} | Regime: **{breadth_regime}**

**Sector Breadth (% above 200MA):**
{sector_breadth_table}

**Macro Regime Ranking:**
| Rank | Indicator | Current Status | Regime Signal |
| :--- | :--- | :--- | :--- |
| 1 | M2 / Liquidity | ${m2_current}B ({m2_trend}) | [Risk On / Risk Off based on M2 trend] |
| 2 | Yield Curve | {yield_spread_now:+.3f}% {yield_inverted_flag} | [Risk On / Risk Off based on inversion status] |
| 3 | Market Breadth | {pct_above_200ma}% above 200MA ({breadth_regime}) | [Risk On / Risk Off based on breadth regime] |

---

## 3. Portfolio Performance
Link: https://digital.fidelity.com/ftgw/digital/portfolio/positions

**Holdings:**
| Ticker | Shares | Avg Cost | Price | Mkt Value | P&L% | 1M Ann% | Sharpe | RSI | Signal |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
[INSERT ONE ROW PER PORTFOLIO POSITION using the portfolio data]

**Portfolio Math:** {cost_mult} × (1 − {loss_pct}) = {net_result}; {net_result}^(52/{weeks}) = **[COMPUTE +XX%/yr]**

**Total Cost Basis:** ${total_cost:,.0f} | **Market Value:** ${total_mv:,.0f} | **P&L:** {total_pnl_pct:+.1f}%

---

## 4. Position Logic & "Same Shape" Analysis

**Grouped Correlations** (r > 0.75 over last 30 trading days):

{correlation_section}

**Uncorrelated / Standalone Positions:** [List any portfolio tickers not in a correlation group]

---

## 5. Tactical Actions

**Sell / Exit:** [List tickers where 1M Ann% < 10% or Sharpe < 0.5 — state the metric that triggered it]

**Buy / Add Winners:** [From watchlist, highlight highest-score uncorrelated candidates. Priority tickers if present: SNDK, MU, LBRT, AROC, PARR]

**Trend Scan — New Candidates:** [From watchlist, list tickers with score ≥ 7 and RSI 45–65 as fresh entry candidates]
"""


# ── Helper: extract close prices from cached history ─────────────────────────

def _closes_from_cache(ticker: str, db: Session) -> list[float]:
    row = db.query(StockCache).filter(StockCache.ticker == ticker).first()
    if not row or not row.history_json:
        return []
    try:
        bars = json.loads(row.history_json)
        return [b["close"] for b in bars if b.get("close") is not None]
    except Exception:
        return []


# ── Correlation group computation ─────────────────────────────────────────────

def _compute_correlation_groups(tickers: list[str], db: Session) -> list[dict]:
    """
    Return list of { label, tickers } for groups with pairwise r > 0.75
    computed over the last 30 trading days of price history.
    """
    if len(tickers) < 2:
        return []

    series: dict[str, np.ndarray] = {}
    for t in tickers:
        closes = _closes_from_cache(t, db)
        if len(closes) >= 31:
            arr = np.array(closes[-31:], dtype=float)
            ret = np.diff(arr) / arr[:-1]          # 30 daily returns
            series[t] = ret

    if len(series) < 2:
        return []

    # Align to shortest series
    min_len = min(len(v) for v in series.values())
    aligned = {t: v[-min_len:] for t, v in series.items()}
    tlist = list(aligned.keys())

    used: set[str] = set()
    groups: list[list[str]] = []

    for i, t1 in enumerate(tlist):
        if t1 in used:
            continue
        group = [t1]
        for t2 in tlist[i + 1:]:
            if t2 in used:
                continue
            try:
                r = float(np.corrcoef(aligned[t1], aligned[t2])[0, 1])
            except Exception:
                continue
            if r > 0.75:
                group.append(t2)
        if len(group) > 1:
            for t in group:
                used.add(t)
            groups.append(group)

    # Label each group by dominant sector
    result = []
    for group in groups:
        g = set(group)
        if g & _ENERGY:
            label = "Energies"
        elif g & _COMMODITY:
            label = "Commodities"
        elif g & _TECH:
            label = "Tech/AI"
        else:
            label = "Mixed"
        result.append({"label": label, "tickers": group})

    return result


# ── Context builder ───────────────────────────────────────────────────────────

async def _gather_context(user_id: int, db: Session) -> dict:
    from services import market_data as md

    positions = (
        db.query(PortfolioPosition)
        .filter(PortfolioPosition.user_id == user_id)
        .all()
    )
    watchlist = (
        db.query(WatchlistItem)
        .filter(WatchlistItem.user_id == user_id)
        .all()
    )

    port_tickers = [p.ticker for p in positions]
    watch_tickers = [w.ticker for w in watchlist]
    all_tickers = list(set(port_tickers + watch_tickers))

    quotes = await md.get_batch(all_tickers, db)
    qmap = {q["ticker"]: q for q in quotes}

    # ── Portfolio rows ────────────────────────────────────────────────────
    port_rows = []
    total_cost = total_mv = 0.0
    oldest_added: Optional[datetime] = None

    for pos in positions:
        q = qmap.get(pos.ticker, {})
        price = q.get("price") or pos.avg_cost
        cost_basis = pos.shares * pos.avg_cost
        mv = pos.shares * price
        pnl_pct = ((mv - cost_basis) / cost_basis * 100) if cost_basis > 0 else 0.0
        total_cost += cost_basis
        total_mv += mv
        if oldest_added is None or pos.added_at < oldest_added:
            oldest_added = pos.added_at

        port_rows.append({
            "ticker": pos.ticker,
            "shares": pos.shares,
            "avg_cost": round(pos.avg_cost, 2),
            "price": round(price, 2),
            "market_value": round(mv, 2),
            "cost_basis": round(cost_basis, 2),
            "pnl_pct": round(pnl_pct, 2),
            "sector": q.get("sector", "Unknown"),
            "rsi": q.get("rsi"),
            "vol_r": q.get("vol_r"),
            "chg_pct": q.get("chg_pct"),
            "ann_ret": q.get("ann_ret"),      # 1-month annualized
            "sharpe": q.get("sharpe"),
            "macd_sig": q.get("macd_sig"),
            "vs_ma50": q.get("vs_ma50"),
            "vs_ma200": q.get("vs_ma200"),
            "gc": q.get("gc"),
            "score": q.get("score"),
        })

    # ── Watchlist rows ────────────────────────────────────────────────────
    watch_rows = []
    for w in watchlist:
        q = qmap.get(w.ticker, {})
        watch_rows.append({
            "ticker": w.ticker,
            "price": q.get("price"),
            "chg_pct": q.get("chg_pct"),
            "rsi": q.get("rsi"),
            "score": q.get("score"),
            "ann_ret": q.get("ann_ret"),
            "sharpe": q.get("sharpe"),
            "sector": q.get("sector", "Unknown"),
            "macd_sig": q.get("macd_sig"),
            "vs_ma200": q.get("vs_ma200"),
            "notes": w.notes,
        })

    # ── Portfolio-level math ──────────────────────────────────────────────
    total_pnl_pct = ((total_mv - total_cost) / total_cost * 100) if total_cost > 0 else 0.0
    now_utc = datetime.now(timezone.utc)
    if oldest_added:
        if oldest_added.tzinfo is None:
            oldest_added = oldest_added.replace(tzinfo=timezone.utc)
        weeks_held = max(1, int((now_utc - oldest_added).days / 7))
    else:
        weeks_held = 1
    cost_mult = round(total_mv / total_cost, 4) if total_cost > 0 else 1.0

    # ── Correlation groups (with market values) ───────────────────────────
    raw_groups = _compute_correlation_groups(port_tickers, db)
    corr_groups = []
    for g in raw_groups:
        group_mv = sum(
            r["market_value"] for r in port_rows if r["ticker"] in g["tickers"]
        )
        group_pct = (group_mv / total_mv * 100) if total_mv > 0 else 0.0
        corr_groups.append({
            "label": g["label"],
            "tickers": g["tickers"],
            "total_value": round(group_mv, 2),
            "portfolio_pct": round(group_pct, 1),
        })

    return {
        "portfolio": port_rows,
        "watchlist": watch_rows,
        "correlation_groups": corr_groups,
        "summary": {
            "total_cost": round(total_cost, 2),
            "total_mv": round(total_mv, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "weeks_held": weeks_held,
            "cost_multiplier": cost_mult,
        },
    }


# ── Ollama call ───────────────────────────────────────────────────────────────

async def _call_ollama(system: str, user_msg: str, model: str | None = None,
                       api_key: str | None = None) -> str:
    s = current_settings()
    provider = s.ai_provider.lower()
    timeout_seconds = 900.0
    if provider == "none":
        raise RuntimeError("AI provider is disabled")

    model = model or s.report_model or s.ai_model or s.ollama_model
    logger.info(
        "Report LLM request provider=%s model=%s system_chars=%s user_chars=%s",
        provider,
        model,
        len(system or ""),
        len(user_msg or ""),
    )
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
    }
    if provider in ("litellm", "openai"):
        base_url = "https://api.openai.com" if provider == "openai" else (s.litellm_url or s.ollama_url).rstrip("/")
        url = base_url + "/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if provider == "litellm":
            key = api_key or s.litellm_api_key
            if not key:
                raise RuntimeError("LITELLM_API_KEY is required when AI_PROVIDER=litellm")
            headers["Authorization"] = f"Bearer {key}"
        else:
            if not s.ai_api_key:
                raise RuntimeError("AI_API_KEY is required when AI_PROVIDER=openai")
            headers["Authorization"] = f"Bearer {s.ai_api_key}"
        client_timeout = httpx.Timeout(timeout_seconds, connect=15.0, read=timeout_seconds, write=60.0, pool=15.0)
        logger.info("Report LLM HTTP POST url=%s model=%s timeout=%s", url, model, timeout_seconds)
        try:
            async with httpx.AsyncClient(timeout=client_timeout) as client:
                resp = await client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as e:
            logger.error(
                "Report LLM timeout provider=%s url=%s model=%s timeout=%s error_type=%s error_repr=%r",
                provider,
                url,
                model,
                timeout_seconds,
                type(e).__name__,
                e,
            )
            raise RuntimeError(f"{provider} request timed out after {timeout_seconds:.0f}s ({type(e).__name__})") from e
        except httpx.RequestError as e:
            logger.error(
                "Report LLM request error provider=%s url=%s model=%s error_type=%s error_repr=%r",
                provider,
                url,
                model,
                type(e).__name__,
                e,
            )
            raise RuntimeError(f"{provider} request failed before a response: {type(e).__name__}: {e!r}") from e
        else:
            logger.info(
                "Report LLM response provider=%s model=%s status=%s response_chars=%s",
                provider,
                model,
                resp.status_code,
                len(resp.text or ""),
            )
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                logger.error(
                    "Report LLM HTTP error provider=%s url=%s model=%s status=%s body=%s",
                    provider,
                    url,
                    model,
                    resp.status_code,
                    resp.text[:1000],
                )
                raise RuntimeError(f"{provider} returned HTTP {resp.status_code}: {resp.text[:500]}") from e
            try:
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                logger.error(
                    "Report LLM parse error provider=%s model=%s body=%s",
                    provider,
                    model,
                    resp.text[:1000],
                )
                raise RuntimeError(f"{provider} returned an unexpected response shape: {e}") from e

    if provider != "ollama":
        raise RuntimeError(f"AI provider '{provider}' is not supported for report generation")

    url = s.ollama_url.rstrip("/") + "/api/chat"
    client_timeout = httpx.Timeout(timeout_seconds, connect=15.0, read=timeout_seconds, write=60.0, pool=15.0)
    logger.info("Report Ollama HTTP POST url=%s model=%s timeout=%s", url, model, timeout_seconds)
    try:
        async with httpx.AsyncClient(timeout=client_timeout) as client:
            resp = await client.post(url, json=payload)
    except httpx.TimeoutException as e:
        logger.error(
            "Report Ollama timeout url=%s model=%s timeout=%s error_type=%s error_repr=%r",
            url,
            model,
            timeout_seconds,
            type(e).__name__,
            e,
        )
        raise RuntimeError(f"ollama request timed out after {timeout_seconds:.0f}s ({type(e).__name__})") from e
    except httpx.RequestError as e:
        logger.error(
            "Report Ollama request error url=%s model=%s error_type=%s error_repr=%r",
            url,
            model,
            type(e).__name__,
            e,
        )
        raise RuntimeError(f"ollama request failed before a response: {type(e).__name__}: {e!r}") from e
    else:
        logger.info("Report Ollama response model=%s status=%s response_chars=%s", model, resp.status_code, len(resp.text or ""))
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(
                "Report Ollama HTTP error url=%s model=%s status=%s body=%s",
                url,
                model,
                resp.status_code,
                resp.text[:1000],
            )
            raise RuntimeError(f"ollama returned HTTP {resp.status_code}: {resp.text[:500]}") from e
        try:
            return resp.json()["message"]["content"]
        except Exception as e:
            logger.error("Report Ollama parse error model=%s body=%s", model, resp.text[:1000])
            raise RuntimeError(f"ollama returned an unexpected response shape: {e}") from e


# ── Public entry point ────────────────────────────────────────────────────────

async def generate_daily_report(
    db: Session,
    user_id: int,
    triggered_by: str = "user",
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    """
    Generate the daily market analysis report for *user_id*.
    Stores the result in ReportCache (keeps last 10 per user).
    Returns the raw markdown string.
    """
    logger.info(
        "Generating daily report for user_id=%s triggered_by=%s requested_model=%s",
        user_id,
        triggered_by,
        model or "(default)",
    )

    # Fetch live chart/macro data concurrently with portfolio context
    import asyncio

    async def gather_charts():
        chart_db = SessionLocal()
        try:
            return await get_all_chart_data(chart_db)
        finally:
            chart_db.close()

    logger.info("Daily report gathering context user_id=%s", user_id)
    try:
        ctx, charts = await asyncio.gather(
            _gather_context(user_id, db),
            gather_charts(),
        )
    except Exception:
        logger.exception("Daily report context gathering failed user_id=%s", user_id)
        raise
    logger.info(
        "Daily report context ready user_id=%s portfolio=%s watchlist=%s correlation_groups=%s chart_keys=%s",
        user_id,
        len(ctx.get("portfolio", [])),
        len(ctx.get("watchlist", [])),
        len(ctx.get("correlation_groups", [])),
        sorted(charts.keys()) if isinstance(charts, dict) else type(charts).__name__,
    )
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    s = ctx["summary"]

    # ── Unpack macro scalars ──────────────────────────────────────────────────
    macro = charts.get("macro", {})
    vix_data = charts.get("vix", [])
    breadth = charts.get("breadth", {})

    vix_current = vix_data[-1]["value"] if vix_data else 0.0
    vix_zone_raw = (
        "extreme_fear" if vix_current > 30 else
        "fear"         if vix_current > 20 else
        "neutral"      if vix_current > 15 else
        "complacency"
    )
    vix_zone_label = vix_zone_raw.replace("_", " ").title()

    yield_spread = macro.get("yield_spread_now") or 0.0
    yield_inverted = macro.get("yield_inverted", False)
    yield_inverted_flag = "⚠️ INVERTED" if yield_inverted else ""
    yield_curve_state = "inverted (recession signal)" if yield_inverted else "normal"

    # ── Sector breadth table ──────────────────────────────────────────────────
    sector_rows = breadth.get("sector_breadth", [])
    if sector_rows:
        sb_lines = ["| Sector | % Above 200MA | % Advancing |",
                    "| :--- | ---: | ---: |"]
        for sr in sector_rows:
            sb_lines.append(
                f"| {sr['sector']} | {sr['pct_above_200']}% | {sr['pct_adv']}% |"
            )
        sector_breadth_table = "\n".join(sb_lines)
    else:
        sector_breadth_table = "_Breadth data unavailable (screener cache empty)._"

    # ── Correlation section ───────────────────────────────────────────────────
    if ctx["correlation_groups"]:
        corr_lines = []
        for g in ctx["correlation_groups"]:
            tickers_str = ", ".join(g["tickers"])
            corr_lines.append(
                f"**Same Shape — {g['label']}:** {tickers_str} "
                f"| Total Value: ${g['total_value']:,.0f} ({g['portfolio_pct']:.1f}%)"
            )
        correlation_section = "\n\n".join(corr_lines)
    else:
        correlation_section = "_No correlation groups detected (fewer than 2 positions or insufficient history)._"

    # Fill static template values (numeric math pre-computed for the LLM)
    loss_pct = round(max(0.0, (s["total_cost"] - s["total_mv"]) / s["total_cost"]), 4) if s["total_cost"] > 0 else 0.0
    net_result = round(s["cost_multiplier"] * (1 - loss_pct), 4)

    template_filled = _TEMPLATE.format(
        date=today,
        m2_current=macro.get("m2_current") or "N/A",
        m2_trend=macro.get("m2_trend") or "unknown",
        fed_rate=macro.get("fed_rate") or "N/A",
        yield_2yr=macro.get("yield_2yr") or "N/A",
        yield_10yr=macro.get("yield_10yr") or "N/A",
        yield_spread_now=yield_spread,
        yield_inverted_flag=yield_inverted_flag,
        yield_curve_state=yield_curve_state,
        vix_current=vix_current,
        vix_zone=vix_zone_label,
        pct_above_200ma=breadth.get("pct_above_200ma", 0),
        pct_above_50ma=breadth.get("pct_above_50ma", 0),
        ad_ratio=breadth.get("ad_ratio", 0),
        breadth_regime=breadth.get("regime", "Unknown"),
        sector_breadth_table=sector_breadth_table,
        cost_mult=s["cost_multiplier"],
        loss_pct=loss_pct,
        net_result=net_result,
        weeks=s["weeks_held"],
        total_cost=s["total_cost"],
        total_mv=s["total_mv"],
        total_pnl_pct=s["total_pnl_pct"],
        correlation_section=correlation_section,
    )

    user_msg = f"""\
Today: {datetime.now(timezone.utc).strftime("%Y-%m-%d")}

=== LIVE MACRO DATA (FRED + yfinance) ===
M2 Supply: ${macro.get("m2_current")}B  Trend: {macro.get("m2_trend")}
Fed Funds Rate: {macro.get("fed_rate")}%
2yr Yield: {macro.get("yield_2yr")}%  10yr Yield: {macro.get("yield_10yr")}%  Spread: {yield_spread:+.3f}% {yield_inverted_flag}
VIX: {vix_current:.2f}  Zone: {vix_zone_label}

=== MARKET BREADTH (S&P 500, from screener cache) ===
{json.dumps(breadth, indent=2)}

=== PORTFOLIO DATA (enriched quotes + technicals) ===
{json.dumps(ctx["portfolio"], indent=2)}

=== WATCHLIST DATA ===
{json.dumps(ctx["watchlist"], indent=2)}

=== PORTFOLIO SUMMARY ===
{json.dumps(s, indent=2)}

=== PRE-COMPUTED CORRELATION GROUPS ===
{json.dumps(ctx["correlation_groups"], indent=2)}

=== FILL THIS TEMPLATE ===
{template_filled}

Replace every [ANALYZE] and [STATE] placeholder with real analysis derived from the data above.
Use the live macro numbers provided — do not use placeholder text like [CHECK: url].
Do not add new sections. Return only the completed markdown — no preamble."""

    try:
        markdown = await _call_ollama(_SYSTEM_PROMPT, user_msg, model=model, api_key=api_key)
    except Exception as e:
        logger.error(
            "Report generation LLM call failed provider=%s model=%s error_type=%s error_repr=%r",
            current_settings().ai_provider,
            model or "(default)",
            type(e).__name__,
            e,
        )
        raise RuntimeError(f"Report generation failed: {e}") from e
    logger.info("Daily report LLM completed user_id=%s markdown_chars=%s", user_id, len(markdown or ""))

    # Persist — keep only the 10 most recent per user
    try:
        old = (
            db.query(ReportCache)
            .filter(ReportCache.user_id == user_id)
            .order_by(ReportCache.generated_at.desc())
            .offset(9)
            .all()
        )
        for r in old:
            db.delete(r)
        db.add(ReportCache(
            user_id=user_id,
            report_markdown=markdown,
            triggered_by=triggered_by,
        ))
        db.commit()
        logger.info("Daily report persisted user_id=%s triggered_by=%s", user_id, triggered_by)
    except Exception as e:
        logger.warning("Could not persist report to DB: %s", e)

    return markdown
