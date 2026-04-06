from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database.db import get_db
from database.models import User
from auth.deps import get_current_user
from services.chart_service import (
    get_macro_data, get_vix_data, get_sector_data,
    get_breadth_data, get_all_chart_data,
)

router = APIRouter(prefix="/api/charts", tags=["charts"])


@router.get("/macro")
async def macro(user: User = Depends(get_current_user)):
    """M2, Fed Funds Rate, 2yr/10yr yields and spread from FRED."""
    try:
        return await get_macro_data()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"FRED API error: {e}")


@router.get("/vix")
async def vix(user: User = Depends(get_current_user)):
    """VIX 90-day history from yfinance."""
    try:
        data = await get_vix_data()
        current = data[-1]["value"] if data else None
        zone = (
            "extreme_fear" if current and current > 30 else
            "fear"         if current and current > 20 else
            "neutral"      if current and current > 15 else
            "complacency"
        )
        return {"vix": data, "current": current, "zone": zone}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"VIX fetch error: {e}")


@router.get("/sectors")
async def sectors(user: User = Depends(get_current_user)):
    """Sector ETF rotation: 5d / 1m / 3m returns."""
    try:
        return {"sectors": await get_sector_data()}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Sector fetch error: {e}")


@router.get("/breadth")
def breadth(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Market breadth from screener cache (% above 50/200 MA, A/D ratio)."""
    return get_breadth_data(db)


@router.get("/all")
async def all_charts(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """All chart data in a single request — used by the dashboard on load."""
    try:
        return await get_all_chart_data(db)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
