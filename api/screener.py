import json
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from database.db import get_db
from database.models import User, StockCache
from auth.deps import get_current_user
from services.market_data import get_universe_status
from services.universe import UNIVERSE

router = APIRouter(prefix="/api/screener", tags=["screener"])


class ScreenerFilters(BaseModel):
    # Fundamental
    pe_min: Optional[float] = None
    pe_max: Optional[float] = None
    fwd_pe_min: Optional[float] = None
    fwd_pe_max: Optional[float] = None
    eps_grw_min: Optional[float] = None
    eps_grw_max: Optional[float] = None
    rev_grw_min: Optional[float] = None
    rev_grw_max: Optional[float] = None
    pm_min: Optional[float] = None
    pm_max: Optional[float] = None
    debt_eq_min: Optional[float] = None
    debt_eq_max: Optional[float] = None
    fcf_only: bool = False

    # Technical
    rsi_min: Optional[float] = None
    rsi_max: Optional[float] = None
    macd: Optional[str] = None           # "bullish" | "bearish" | "neutral" | None
    vs_ma200: Optional[str] = None       # "above" | "below" | None
    vs_ma50: Optional[str] = None
    gc: bool = False
    dc: bool = False

    # Momentum
    vol_r_min: Optional[float] = None
    p52w_min: Optional[float] = None
    p52w_max: Optional[float] = None
    earn_beat: bool = False
    earn_soon: bool = False

    # Portfolio / Meta
    beta_min: Optional[float] = None
    beta_max: Optional[float] = None
    div_min: Optional[float] = None
    div_max: Optional[float] = None
    sectors: list[str] = []

    # Universe filters
    asset_type: Optional[str] = None     # "stocks" | "etfs" | None (all)
    cap_tier: Optional[str] = None       # "Large" | "Mid" | "Small" | None (all)

    # Score filters
    score_min: Optional[int] = None
    score_f_min: Optional[int] = None
    score_t_min: Optional[int] = None
    score_m_min: Optional[int] = None

    force_refresh: bool = False


def _passes(stock: dict, f: ScreenerFilters) -> bool:
    def between(val, lo, hi):
        if val is None:
            return lo is None and hi is None
        if lo is not None and val < lo:
            return False
        if hi is not None and val > hi:
            return False
        return True

    qt = stock.get("quote_type", "EQUITY")
    if f.asset_type == "stocks" and qt in ("ETF", "MUTUALFUND"):
        return False
    if f.asset_type == "etfs" and qt not in ("ETF", "MUTUALFUND"):
        return False
    mc = stock.get("mkt_cap")
    if f.cap_tier == "Large" and (mc is None or mc < 10):
        return False
    if f.cap_tier == "Mid" and (mc is None or mc < 2 or mc >= 10):
        return False
    if f.cap_tier == "Small" and (mc is None or mc >= 2):
        return False

    if not between(stock.get("pe"), f.pe_min, f.pe_max): return False
    if not between(stock.get("fwd_pe"), f.fwd_pe_min, f.fwd_pe_max): return False
    if not between(stock.get("eps_grw"), f.eps_grw_min, f.eps_grw_max): return False
    if not between(stock.get("rev_grw"), f.rev_grw_min, f.rev_grw_max): return False
    if not between(stock.get("pm"), f.pm_min, f.pm_max): return False
    if not between(stock.get("debt_eq"), f.debt_eq_min, f.debt_eq_max): return False
    if f.fcf_only and not stock.get("fcf_pos"): return False

    if not between(stock.get("rsi"), f.rsi_min, f.rsi_max): return False
    if f.macd and stock.get("macd_sig") != f.macd: return False
    if f.vs_ma200 == "above" and (stock.get("vs_ma200") or 0) <= 0: return False
    if f.vs_ma200 == "below" and (stock.get("vs_ma200") or 0) >= 0: return False
    if f.vs_ma50 == "above" and (stock.get("vs_ma50") or 0) <= 0: return False
    if f.vs_ma50 == "below" and (stock.get("vs_ma50") or 0) >= 0: return False
    if f.gc and not stock.get("gc"): return False
    if f.dc and not stock.get("dc"): return False

    if not between(stock.get("vol_r"), f.vol_r_min, None): return False
    if not between(stock.get("p52w"), f.p52w_min, f.p52w_max): return False
    if f.earn_beat and not stock.get("earn_beat"): return False
    if f.earn_soon and not stock.get("earn_soon"): return False

    if not between(stock.get("beta"), f.beta_min, f.beta_max): return False
    if not between(stock.get("div"), f.div_min, f.div_max): return False
    if f.sectors and stock.get("sector") not in f.sectors: return False

    sc = stock.get("score") or {}
    if f.score_min is not None and (sc.get("o") or 0) < f.score_min: return False
    if f.score_f_min is not None and (sc.get("f") or 0) < f.score_f_min: return False
    if f.score_t_min is not None and (sc.get("t") or 0) < f.score_t_min: return False
    if f.score_m_min is not None and (sc.get("m") or 0) < f.score_m_min: return False

    return True


@router.post("")
async def screen(
    filters: ScreenerFilters,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Screen the full S&P 500 + ETF universe from cache."""
    rows = db.query(StockCache).filter(StockCache.quote_json.isnot(None)).all()

    all_quotes = []
    for row in rows:
        try:
            all_quotes.append(json.loads(row.quote_json))
        except Exception:
            pass

    passed = [q for q in all_quotes if _passes(q, filters)]
    universe_status = get_universe_status(db, UNIVERSE)

    return {
        "results": passed,
        "total": len(all_quotes),
        "filtered": len(passed),
        "universe": universe_status,
    }


@router.get("/universe/status")
def universe_status_endpoint(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return get_universe_status(db, UNIVERSE)
