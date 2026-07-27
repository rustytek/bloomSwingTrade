from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
import logging
import time
from database.db import get_db
from database.models import User, ReportCache
from auth.deps import get_current_user
from services import market_data
from services.ai_service import AIService, ai_service
from config import get_settings

router = APIRouter(prefix="/api/ai", tags=["ai"])
settings = get_settings()
logger = logging.getLogger(__name__)

# Default system prompt for market chat. Users may override this per-account via
# PUT /api/ai/prompts; when their override is null this default is used.
DEFAULT_CHAT_SYSTEM_PROMPT = (
    "You are a professional swing-trading assistant with access to the user's live portfolio "
    "and watchlist data. Answer questions concisely and specifically using the data provided. "
    "Use markdown formatting. Cite specific tickers and numbers from the data when relevant. "
    "Keep answers focused and under 400 words unless a detailed breakdown is requested."
)


def current_settings():
    return get_settings()


def llm_base_url() -> str:
    s = current_settings()
    provider = s.ai_provider.lower()
    if provider == "litellm":
        return s.litellm_url.rstrip("/")
    return "https://api.openai.com"


def llm_headers(api_key: str | None = None) -> dict:
    s = current_settings()
    provider = s.ai_provider.lower()
    headers = {"Content-Type": "application/json"}
    if provider == "litellm":
        key = api_key or s.litellm_api_key
        if not key:
            raise RuntimeError("LITELLM_API_KEY is required when AI_PROVIDER=litellm")
        headers["Authorization"] = f"Bearer {key}"
    elif provider == "openai":
        if not s.ai_api_key:
            raise RuntimeError("AI_API_KEY is required when AI_PROVIDER=openai")
        headers["Authorization"] = f"Bearer {s.ai_api_key}"
    return headers


async def call_chat_model(system: str, user_msg: str, model: str | None = None, timeout: float = 120.0,
                          api_key: str | None = None) -> str:
    import httpx

    s = current_settings()
    provider = s.ai_provider.lower()
    if provider == "none":
        raise RuntimeError("AI provider is disabled")

    model_name = model or s.ai_model
    if provider in ("litellm", "openai"):
        url = f"{llm_base_url()}/v1/chat/completions"
        payload = {
            "model": model_name,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
        }
        logger.info(
            "Market chat LLM request provider=%s base_url=%s model=%s user_chars=%s timeout=%s",
            provider,
            llm_base_url(),
            model_name,
            len(user_msg),
            timeout,
        )
        client_timeout = httpx.Timeout(timeout, connect=15.0, read=timeout, write=60.0, pool=15.0)
        try:
            async with httpx.AsyncClient(timeout=client_timeout) as client:
                resp = await client.post(url, json=payload, headers=llm_headers(api_key))
        except httpx.TimeoutException as e:
            logger.error(
                "Market chat LLM timeout provider=%s url=%s model=%s timeout=%s error_type=%s error_repr=%r",
                provider,
                url,
                model_name,
                timeout,
                type(e).__name__,
                e,
            )
            raise RuntimeError(f"{provider} request timed out after {timeout:.0f}s ({type(e).__name__})") from e
        except httpx.RequestError as e:
            logger.error(
                "Market chat LLM request error provider=%s url=%s model=%s error_type=%s error_repr=%r",
                provider,
                url,
                model_name,
                type(e).__name__,
                e,
            )
            raise RuntimeError(f"{provider} request failed before a response: {type(e).__name__}: {e!r}") from e
        else:
            logger.info(
                "Market chat LLM response provider=%s model=%s status=%s response_chars=%s",
                provider,
                model_name,
                resp.status_code,
                len(resp.text or ""),
            )
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                logger.error(
                    "Market chat LLM HTTP error provider=%s url=%s model=%s status=%s body=%s",
                    provider,
                    url,
                    model_name,
                    resp.status_code,
                    resp.text[:1000],
                )
                raise RuntimeError(f"{provider} returned HTTP {resp.status_code}: {resp.text[:500]}") from e
            try:
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                logger.error(
                    "Market chat LLM parse error provider=%s model=%s body=%s",
                    provider,
                    model_name,
                    resp.text[:1000],
                )
                raise RuntimeError(f"{provider} returned an unexpected response shape: {e}") from e

    raise RuntimeError(f"AI provider '{provider}' is not supported by this endpoint")


class ChatRequest(BaseModel):
    question: str


class MarketChatRequest(BaseModel):
    question: str
    report_context: str | None = None
    model: str | None = None            # override chat model


class ReportRequest(BaseModel):
    model: str | None = None            # override report model


class DebugLiteLLMRequest(BaseModel):
    model: str | None = None
    mode: str = "short"                 # short | market | report
    timeout: float = 120.0
    target_chars: int | None = None


def _pad_debug_text(text: str, target_chars: int | None) -> str:
    if not target_chars or len(text) >= target_chars:
        return text
    filler = "\n\n=== SIZE PADDING ===\n" + ("Synthetic context row. " * 1000)
    return text + filler[:target_chars - len(text)]


def _debug_llm_user_message(mode: str, target_chars: int | None = None) -> str:
    import json

    mode = mode.lower()
    if mode == "short":
        return "Say OK and name the model you are running as."

    sample_position = {
        "ticker": "SAMPLE",
        "shares": 10,
        "avg_cost": 100.0,
        "price": 104.2,
        "market_value": 1042.0,
        "pnl_pct": 4.2,
        "rsi": 55.1,
        "vol_r": 1.15,
        "chg_pct": 0.8,
        "ann_ret": 24.5,
        "sharpe": 1.1,
        "sector": "Technology",
        "macd_sig": "bullish",
    }
    sample_watch = {
        "ticker": "WATCH",
        "price": 42.1,
        "rsi": 49.8,
        "score": {"o": 7},
        "chg_pct": -0.3,
        "sector": "Energy",
        "notes": "Synthetic smoke-test row.",
    }

    portfolio = [{**sample_position, "ticker": f"P{i:02d}"} for i in range(31)]
    watchlist = [{**sample_watch, "ticker": f"W{i:02d}"} for i in range(16)]

    if mode == "market":
        text = (
            "=== PORTFOLIO SMOKE TEST ===\n"
            + json.dumps(portfolio, indent=2)
            + "\n\n=== WATCHLIST SMOKE TEST ===\n"
            + json.dumps(watchlist, indent=2)
            + "\n\nQuestion: reply in one short paragraph with the strongest and weakest synthetic tickers."
        )
        return _pad_debug_text(text, target_chars or 13655)

    if mode == "report":
        macro = {
            "m2_current": 22000,
            "m2_trend": "rising",
            "fed_rate": 4.5,
            "yield_2yr": 4.1,
            "yield_10yr": 4.3,
            "yield_spread_now": 0.2,
            "vix_current": 16.4,
            "breadth": {"pct_above_200ma": 58, "pct_above_50ma": 51, "ad_ratio": 1.2},
        }
        template = """
# [DATE] Market Analysis & Execution

## 1. Macro Regime & Liquidity
[ANALYZE macro regime using the JSON data]

## 2. Market Breadth & Ranking
[ANALYZE breadth]

## 3. Portfolio Performance
[INSERT portfolio table]

## 4. Position Logic & Same Shape Analysis
[ANALYZE correlation groups]

## 5. Tactical Actions
[LIST sell/add/watch candidates]
"""
        text = (
            "Today: smoke-test\n\n"
            "=== LIVE MACRO DATA ===\n"
            + json.dumps(macro, indent=2)
            + "\n\n=== PORTFOLIO DATA ===\n"
            + json.dumps(portfolio, indent=2)
            + "\n\n=== WATCHLIST DATA ===\n"
            + json.dumps(watchlist, indent=2)
            + "\n\n=== FILL THIS TEMPLATE ===\n"
            + template
            + "\nReturn completed markdown only."
        )
        return _pad_debug_text(text, target_chars or 26639)

    raise ValueError("mode must be one of: short, market, report")


@router.get("/status")
def ai_status(user: User = Depends(get_current_user)):
    s = current_settings()
    provider = s.ai_provider.lower()
    return {
        "configured": provider != "none",
        "provider": provider,
        "model": s.ai_model or "default",
        "base_url": llm_base_url() if provider == "litellm" else None,
    }


class PromptsUpdate(BaseModel):
    report_system_prompt: str | None = None
    chat_system_prompt: str | None = None


def _prompts_payload(user: User) -> dict:
    """Per-user prompt overrides plus the built-in defaults (so the UI can show
    the template and offer a reset)."""
    from services.report_service import _SYSTEM_PROMPT as DEFAULT_REPORT_SYSTEM_PROMPT
    return {
        "report": {
            "custom": user.report_system_prompt,
            "default": DEFAULT_REPORT_SYSTEM_PROMPT,
        },
        "chat": {
            "custom": user.chat_system_prompt,
            "default": DEFAULT_CHAT_SYSTEM_PROMPT,
        },
    }


@router.get("/prompts")
def get_prompts(user: User = Depends(get_current_user)):
    """Return this user's AI system-prompt overrides and the built-in defaults."""
    return _prompts_payload(user)


@router.put("/prompts")
def update_prompts(
    req: PromptsUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Set or clear this user's AI system-prompt overrides. An omitted field is
    left unchanged; an empty/whitespace value clears the override (revert to
    the built-in default)."""
    data = req.model_dump(exclude_unset=True)
    if "report_system_prompt" in data:
        v = (data["report_system_prompt"] or "").strip()
        user.report_system_prompt = v or None
    if "chat_system_prompt" in data:
        v = (data["chat_system_prompt"] or "").strip()
        user.chat_system_prompt = v or None
    db.commit()
    db.refresh(user)
    return _prompts_payload(user)


@router.get("/{ticker}/analysis")
async def analyze_stock(
    ticker: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    svc: AIService = Depends(ai_service),
):
    ticker = ticker.upper()
    data = await market_data.get_quote(ticker, db)
    if data is None:
        raise HTTPException(status_code=404, detail=f"No data for {ticker}")

    analysis = await svc.analyze_stock(ticker, data)
    return {"ticker": ticker, "analysis": analysis}


@router.get("/{ticker}/signals")
async def get_signals(
    ticker: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    svc: AIService = Depends(ai_service),
):
    ticker = ticker.upper()
    data = await market_data.get_quote(ticker, db)
    if data is None:
        raise HTTPException(status_code=404, detail=f"No data for {ticker}")

    technicals = {k: data.get(k) for k in [
        "rsi", "macd_sig", "vs_ma50", "vs_ma200", "gc", "dc",
        "vol_r", "p52w", "chg_pct", "score"
    ]}
    signals = await svc.generate_signals(ticker, technicals)
    return {"ticker": ticker, "signals": signals}


@router.post("/{ticker}/chat")
async def chat(
    ticker: str,
    req: ChatRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    svc: AIService = Depends(ai_service),
):
    ticker = ticker.upper()
    data = await market_data.get_quote(ticker, db) or {}
    answer = await svc.chat(ticker, req.question, data)
    return {"ticker": ticker, "question": req.question, "answer": answer}


@router.get("/sector/{sector}/summary")
async def sector_summary(
    sector: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    svc: AIService = Depends(ai_service),
):
    summary = await svc.summarize_sector(sector, [])
    return {"sector": sector, "summary": summary}


# ── Daily Report Endpoints ────────────────────────────────────────────────────

@router.get("/models")
async def list_models(user: User = Depends(get_current_user)):
    """Return available models from the configured LiteLLM/OpenAI backend."""
    import httpx
    s = current_settings()
    provider = s.ai_provider.lower()
    if provider not in ("litellm", "openai"):
        return {"models": [], "provider": provider, "base_url": None}
    url = f"{llm_base_url()}/v1/models"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=llm_headers(getattr(user, "litellm_api_key", None)))
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Cannot reach LLM server: {e}")

    models = []
    for m in data.get("data", []):
        name = m.get("id") or m.get("name")
        if name:
            models.append({"name": name, "size_gb": None, "modified_at": None})
    # Sort: put embedding models last, everything else alphabetical
    models.sort(key=lambda m: (1 if "embed" in m["name"] else 0, m["name"]))
    return {"models": models, "provider": provider, "base_url": llm_base_url()}


@router.post("/debug/litellm")
async def debug_litellm(
    req: DebugLiteLLMRequest,
    user: User = Depends(get_current_user),
):
    """Run a direct OpenAI-compatible chat-completions smoke test from this app process."""
    s = current_settings()
    provider = s.ai_provider.lower()
    model_name = req.model or s.ai_model or s.report_model
    try:
        user_msg = _debug_llm_user_message(req.mode, req.target_chars)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    system = "You are a concise SwingTrader LiteLLM diagnostic assistant."
    started = time.perf_counter()
    try:
        answer = await call_chat_model(system, user_msg, model=model_name, timeout=req.timeout,
                                       api_key=getattr(user, "litellm_api_key", None))
    except Exception as e:
        elapsed = time.perf_counter() - started
        logger.error(
            "Debug LiteLLM failed user_id=%s provider=%s model=%s mode=%s elapsed=%.2fs error_type=%s error_repr=%r",
            user.id,
            provider,
            model_name,
            req.mode,
            elapsed,
            type(e).__name__,
            e,
        )
        raise HTTPException(
            status_code=503,
            detail={
                "provider": provider,
                "base_url": llm_base_url() if provider == "litellm" else None,
                "model": model_name,
                "mode": req.mode,
                "elapsed_seconds": round(elapsed, 2),
                "user_chars": len(user_msg),
                "error_type": type(e).__name__,
                "error_repr": repr(e),
                "error": str(e),
            },
        )

    elapsed = time.perf_counter() - started
    return {
        "ok": True,
        "provider": provider,
        "base_url": llm_base_url() if provider == "litellm" else None,
        "model": model_name,
        "mode": req.mode,
        "elapsed_seconds": round(elapsed, 2),
        "user_chars": len(user_msg),
        "answer_chars": len(answer or ""),
        "answer_preview": (answer or "")[:1000],
    }


@router.post("/daily-report")
async def generate_report(
    req: ReportRequest = ReportRequest(),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Trigger on-demand report generation for the current user."""
    from services.report_service import generate_daily_report
    try:
        logger.info(
            "Daily report requested user_id=%s username=%s model=%s",
            user.id,
            user.username,
            req.model or "(default)",
        )
        result = await generate_daily_report(db, user.id, triggered_by="user", model=req.model,
                                             api_key=user.litellm_api_key,
                                             system_prompt=user.report_system_prompt)
    except RuntimeError as e:
        logger.error("Daily report request failed user_id=%s model=%s error=%s", user.id, req.model or "(default)", e)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Daily report request crashed user_id=%s model=%s", user.id, req.model or "(default)")
        raise HTTPException(status_code=503, detail=f"Report generation crashed: {e}")
    return {
        "markdown": result["markdown"],
        "model": result["model"],
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }


@router.get("/report/latest")
def get_latest_report(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Return the most recently generated report for the current user."""
    report = (
        db.query(ReportCache)
        .filter(ReportCache.user_id == user.id)
        .order_by(ReportCache.generated_at.desc())
        .first()
    )
    if not report:
        return {"markdown": None, "generated_at": None, "triggered_by": None, "model": None}
    return {
        "markdown": report.report_markdown,
        "generated_at": report.generated_at.isoformat(),
        "triggered_by": report.triggered_by,
        "model": report.model,
    }


@router.get("/reports")
def list_reports(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """List all stored reports for the current user (newest first)."""
    rows = (
        db.query(ReportCache)
        .filter(ReportCache.user_id == user.id)
        .order_by(ReportCache.generated_at.desc())
        .all()
    )
    return [
        {
            "id": r.id,
            "generated_at": r.generated_at.isoformat(),
            "triggered_by": r.triggered_by,
            "model": r.model,
        }
        for r in rows
    ]


@router.post("/market-chat")
async def market_chat(
    req: MarketChatRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    General market/report chat. Loads the user's live portfolio + watchlist
    as context, then calls the configured LLM with the question (and optional report markdown).
    """
    import json
    from database.models import PortfolioPosition, WatchlistItem
    from services import market_data

    # Gather live portfolio + watchlist data
    positions = db.query(PortfolioPosition).filter(PortfolioPosition.user_id == user.id).all()
    watchlist = db.query(WatchlistItem).filter(WatchlistItem.user_id == user.id).all()
    all_tickers = list({p.ticker for p in positions} | {w.ticker for w in watchlist})
    logger.info(
        "Market chat requested user_id=%s username=%s model=%s question_chars=%s report_context_chars=%s tickers=%s",
        user.id,
        user.username,
        req.model or "(default)",
        len(req.question or ""),
        len(req.report_context or ""),
        len(all_tickers),
    )

    quotes = await market_data.get_batch(all_tickers, db) if all_tickers else []
    qmap = {q["ticker"]: q for q in quotes}
    logger.info("Market chat context loaded user_id=%s quotes=%s", user.id, len(qmap))

    portfolio_data = []
    total_cost = total_mv = 0.0
    for pos in positions:
        q = qmap.get(pos.ticker, {})
        price = q.get("price") or pos.avg_cost
        cost = pos.shares * pos.avg_cost
        mv = pos.shares * price
        total_cost += cost
        total_mv += mv
        portfolio_data.append({
            "ticker": pos.ticker, "shares": pos.shares, "avg_cost": pos.avg_cost,
            "price": round(price, 2),
            "market_value": round(mv, 2),
            "pnl_pct": round((mv - cost) / cost * 100, 2) if cost > 0 else 0,
            "rsi": q.get("rsi"), "vol_r": q.get("vol_r"), "chg_pct": q.get("chg_pct"),
            "ann_ret": q.get("ann_ret"), "sharpe": q.get("sharpe"),
            "sector": q.get("sector", "Unknown"), "macd_sig": q.get("macd_sig"),
        })

    watchlist_data = [
        {"ticker": w.ticker, "price": qmap.get(w.ticker, {}).get("price"),
         "rsi": qmap.get(w.ticker, {}).get("rsi"), "score": qmap.get(w.ticker, {}).get("score"),
         "chg_pct": qmap.get(w.ticker, {}).get("chg_pct"),
         "sector": qmap.get(w.ticker, {}).get("sector", "Unknown"), "notes": w.notes}
        for w in watchlist
    ]

    system = user.chat_system_prompt or DEFAULT_CHAT_SYSTEM_PROMPT

    context_parts = [
        f"=== PORTFOLIO ({len(portfolio_data)} positions, Total MV: ${total_mv:,.0f}) ===",
        json.dumps(portfolio_data, indent=2),
        f"\n=== WATCHLIST ({len(watchlist_data)} tickers) ===",
        json.dumps(watchlist_data, indent=2),
    ]
    if req.report_context:
        context_parts += ["\n=== TODAY'S MARKET REPORT ===", req.report_context[:6000]]

    user_msg = "\n".join(context_parts) + f"\n\n=== QUESTION ===\n{req.question}"

    try:
        answer = await call_chat_model(system, user_msg, model=req.model, timeout=300.0,
                                       api_key=user.litellm_api_key)
    except Exception as e:
        logger.error(
            "Market chat failed user_id=%s model=%s error_type=%s error_repr=%r",
            user.id,
            req.model or "(default)",
            type(e).__name__,
            e,
        )
        raise HTTPException(status_code=503, detail=f"LLM error: {e}")

    logger.info("Market chat completed user_id=%s model=%s answer_chars=%s", user.id, req.model or "(default)", len(answer or ""))
    return {"answer": answer}


@router.get("/report/{report_id}")
def get_report_by_id(
    report_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Fetch a specific past report by ID."""
    report = (
        db.query(ReportCache)
        .filter(ReportCache.id == report_id, ReportCache.user_id == user.id)
        .first()
    )
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return {
        "id": report.id,
        "markdown": report.report_markdown,
        "generated_at": report.generated_at.isoformat(),
        "triggered_by": report.triggered_by,
        "model": report.model,
    }
