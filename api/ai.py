from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from database.db import get_db
from database.models import User
from auth.deps import get_current_user
from services import market_data
from services.ai_service import AIService, ai_service
from config import get_settings

router = APIRouter(prefix="/api/ai", tags=["ai"])
settings = get_settings()


class ChatRequest(BaseModel):
    question: str


@router.get("/status")
def ai_status(user: User = Depends(get_current_user)):
    provider = settings.ai_provider.lower()
    return {
        "configured": provider != "none",
        "provider": provider,
        "model": settings.ai_model or "default",
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
