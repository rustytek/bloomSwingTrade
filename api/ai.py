from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from database.db import get_db
from database.models import User, ReportCache
from auth.deps import get_current_user
from services import market_data
from services.ai_service import AIService, ai_service
from config import get_settings

router = APIRouter(prefix="/api/ai", tags=["ai"])
settings = get_settings()


def llm_base_url() -> str:
    if settings.ai_provider.lower() == "litellm":
        return (settings.litellm_url or settings.ollama_url).rstrip("/")
    return settings.ollama_url.rstrip("/")


def llm_headers() -> dict:
    if settings.ai_provider.lower() != "litellm":
        return {"Content-Type": "application/json"}
    api_key = settings.litellm_api_key or settings.ai_api_key
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


async def call_chat_model(system: str, user_msg: str, model: str | None = None, timeout: float = 120.0) -> str:
    import httpx

    provider = settings.ai_provider.lower()
    model_name = model or settings.ai_model or settings.ollama_model
    if provider == "litellm":
        payload = {
            "model": model_name,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{llm_base_url()}/v1/chat/completions", json=payload, headers=llm_headers())
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    payload = {
        "model": model or settings.ollama_model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{settings.ollama_url.rstrip('/')}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"]


class ChatRequest(BaseModel):
    question: str


class MarketChatRequest(BaseModel):
    question: str
    report_context: str | None = None
    model: str | None = None            # override chat model


class ReportRequest(BaseModel):
    model: str | None = None            # override report model


@router.get("/status")
def ai_status(user: User = Depends(get_current_user)):
    provider = settings.ai_provider.lower()
    return {
        "configured": provider != "none",
        "provider": provider,
        "model": settings.ai_model or settings.ollama_model or "default",
        "base_url": llm_base_url() if provider in ("litellm", "ollama") else None,
    }


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
@router.get("/ollama/models")
async def ollama_models(user: User = Depends(get_current_user)):
    """Return available local models from Ollama or LiteLLM."""
    import httpx
    provider = settings.ai_provider.lower()
    url = f"{llm_base_url()}/v1/models" if provider == "litellm" else settings.ollama_url.rstrip("/") + "/api/tags"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=llm_headers())
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Cannot reach local LLM server: {e}")

    models = []
    if provider == "litellm":
        for m in data.get("data", []):
            name = m.get("id") or m.get("name")
            if name:
                models.append({"name": name, "size_gb": None, "modified_at": None})
    else:
        for m in data.get("models", []):
            size_bytes = m.get("size", 0)
            models.append({
                "name": m["name"],
                "size_gb": round(size_bytes / 1e9, 1) if size_bytes else None,
                "modified_at": m.get("modified_at"),
            })
    # Sort: put embedding models last, everything else alphabetical
    models.sort(key=lambda m: (1 if "embed" in m["name"] else 0, m["name"]))
    return {"models": models, "provider": provider, "base_url": llm_base_url()}


@router.post("/daily-report")
async def generate_report(
    req: ReportRequest = ReportRequest(),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Trigger on-demand report generation for the current user."""
    from services.report_service import generate_daily_report
    try:
        markdown = await generate_daily_report(db, user.id, triggered_by="user", model=req.model)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {
        "markdown": markdown,
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
        return {"markdown": None, "generated_at": None, "triggered_by": None}
    return {
        "markdown": report.report_markdown,
        "generated_at": report.generated_at.isoformat(),
        "triggered_by": report.triggered_by,
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
    as context, then calls Ollama with the question (and optional report markdown).
    """
    import json
    from database.models import PortfolioPosition, WatchlistItem
    from services import market_data

    # Gather live portfolio + watchlist data
    positions = db.query(PortfolioPosition).filter(PortfolioPosition.user_id == user.id).all()
    watchlist = db.query(WatchlistItem).filter(WatchlistItem.user_id == user.id).all()
    all_tickers = list({p.ticker for p in positions} | {w.ticker for w in watchlist})

    quotes = await market_data.get_batch(all_tickers, db) if all_tickers else []
    qmap = {q["ticker"]: q for q in quotes}

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

    system = (
        "You are a professional swing-trading assistant with access to the user's live portfolio "
        "and watchlist data. Answer questions concisely and specifically using the data provided. "
        "Use markdown formatting. Cite specific tickers and numbers from the data when relevant. "
        "Keep answers focused and under 400 words unless a detailed breakdown is requested."
    )

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
        answer = await call_chat_model(system, user_msg, model=req.model, timeout=120.0)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM error: {e}")

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
    }
