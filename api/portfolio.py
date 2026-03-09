from fastapi import APIRouter, Depends, HTTPException, Query, status, UploadFile, File
from sqlalchemy.orm import Session
from pydantic import BaseModel
from database.db import get_db
from database.models import User, PortfolioPosition
from auth.deps import get_current_user
from services import market_data
import csv

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])


class PositionRequest(BaseModel):
    ticker: str
    shares: float
    avg_cost: float
    notes: str | None = None


@router.get("")
async def get_portfolio(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    positions = db.query(PortfolioPosition).filter(PortfolioPosition.user_id == user.id).all()
    if not positions:
        return {"positions": [], "summary": {"total_cost": 0, "total_mv": 0, "total_pnl": 0, "total_pnl_pct": 0}}

    tickers = [p.ticker for p in positions]
    quotes = await market_data.get_batch(tickers, db)
    price_map = {q["ticker"]: q.get("price") for q in quotes}

    result = []
    total_cost = 0.0
    total_mv = 0.0

    for pos in positions:
        price = price_map.get(pos.ticker)
        cost_basis = pos.shares * pos.avg_cost
        mv = pos.shares * price if price else cost_basis
        pnl = mv - cost_basis
        pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0

        total_cost += cost_basis
        total_mv += mv

        result.append({
            "ticker": pos.ticker,
            "shares": pos.shares,
            "avg_cost": pos.avg_cost,
            "current_price": price,
            "cost_basis": round(cost_basis, 2),
            "market_value": round(mv, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "added_at": pos.added_at.isoformat(),
            "notes": pos.notes,
        })

    total_pnl = total_mv - total_cost
    return {
        "positions": result,
        "summary": {
            "total_cost": round(total_cost, 2),
            "total_mv": round(total_mv, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl / total_cost * 100 if total_cost > 0 else 0, 2),
        },
    }


@router.post("", status_code=status.HTTP_201_CREATED)
def upsert_position(
    req: PositionRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    ticker = req.ticker.upper().strip()
    pos = (
        db.query(PortfolioPosition)
        .filter(PortfolioPosition.user_id == user.id, PortfolioPosition.ticker == ticker)
        .first()
    )
    if pos:
        pos.shares = req.shares
        pos.avg_cost = req.avg_cost
        if req.notes is not None:
            pos.notes = req.notes
    else:
        pos = PortfolioPosition(
            user_id=user.id, ticker=ticker,
            shares=req.shares, avg_cost=req.avg_cost, notes=req.notes
        )
        db.add(pos)

    db.commit()
    return {"ticker": ticker, "shares": pos.shares, "avg_cost": pos.avg_cost}


@router.delete("/{ticker}", status_code=status.HTTP_204_NO_CONTENT)
def remove_position(
    ticker: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    pos = (
        db.query(PortfolioPosition)
        .filter(PortfolioPosition.user_id == user.id, PortfolioPosition.ticker == ticker.upper())
        .first()
    )
    if not pos:
        raise HTTPException(status_code=404, detail="Position not found")
    db.delete(pos)
    db.commit()


@router.post("/import")
async def import_fidelity_csv(
    file: UploadFile = File(...),
    mode: str = Query("overwrite", description="merge | overwrite | sync"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Import portfolio positions from a Fidelity CSV export.

    Modes:
      merge     — add NEW tickers only; existing positions are untouched
      overwrite — add new AND update existing positions (default)
      sync      — replace entire portfolio with CSV (deletes positions not in CSV)

    Handles:
      - BOM / UTF-8-sig encoding (common from Excel saves)
      - Multiple accounts in one file (same ticker in multiple rows → aggregated)
      - Fidelity footer text rows (copyright, date stamps) → safely skipped
      - Cash/money-market positions (SPAXX**, CORE**, empty quantity) → skipped
      - Dollar signs, commas, plus/minus signs in numeric columns
      - Fidelity slash notation (BRK/B → BRK-B for yfinance)
      - Fidelity mutual fund tickers (FZROX, FSKAX, etc.) → tracked normally
    """
    if mode not in ("merge", "overwrite", "sync"):
        raise HTTPException(status_code=400, detail="mode must be merge, overwrite, or sync")

    content = await file.read()

    # Decode — try UTF-8 with BOM first (Excel/Windows), then plain UTF-8, then latin-1
    text = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = content.decode(enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if text is None:
        text = content.decode("utf-8", errors="replace")

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        raise HTTPException(status_code=400, detail="The uploaded file appears to be empty")

    # Find the header row — must contain both 'symbol' and 'quantity'/'shares'
    header_idx = -1
    for i, line in enumerate(lines):
        low = line.lower()
        if "symbol" in low and ("quantity" in low or "shares" in low):
            header_idx = i
            break
    if header_idx == -1:
        raise HTTPException(
            status_code=400,
            detail="Could not find a header row with 'Symbol' and 'Quantity' columns. "
                   "Export from Fidelity → Positions → Download (CSV)."
        )

    reader = csv.DictReader(lines[header_idx:])
    rows = list(reader)
    if not rows:
        raise HTTPException(status_code=400, detail="No data rows found after the header")

    # Find column names by keyword — operates on dict keys (header names), not values
    def find_col(row: dict, keywords: list[str]) -> str | None:
        for k in row:
            if k is None:
                continue
            kl = k.lower().strip()
            if any(kw in kl for kw in keywords):
                return k
        return None

    sample = rows[0]
    sym_col  = find_col(sample, ["symbol"])
    qty_col  = find_col(sample, ["quantity", "shares"])
    cost_col = find_col(sample, ["average cost basis", "average cost", "avg cost",
                                  "cost basis per share", "cost/share"])

    if not sym_col or not qty_col:
        raise HTTPException(
            status_code=400,
            detail=f"Could not locate required columns. Found: {list(k for k in sample if k)}"
        )

    # Helper: safely read a cell value — DictReader fills short rows with None
    def cell(row: dict, col: str, default: str = "") -> str:
        return (row.get(col) or default).strip()

    # Skip list for known non-position rows
    SKIP_TICKERS = {"", "--", "SPAXX**", "CORE**", "PENDING ACTIVITY", "N/A"}

    # Parse rows — aggregate duplicate tickers (same stock in multiple accounts)
    csv_positions: dict[str, dict] = {}   # ticker → {shares, avg_cost}
    skipped: list[str] = []

    for row in rows:
        raw_ticker = cell(row, sym_col)

        # Normalize: Fidelity uses BRK/B; yfinance uses BRK-B
        raw_ticker = raw_ticker.replace("/", "-").upper()

        if raw_ticker in SKIP_TICKERS:
            continue

        # Skip footer text, totals rows, and obviously non-ticker values.
        # Valid tickers: 1-10 chars, alphanumeric + hyphens (allows fund codes with
        # digits like target-date funds, employer plan codes, QACA holdings, etc.)
        # Pure-number strings ("1234.56") are rejected; spaces indicate footer text.
        clean = raw_ticker.replace("-", "").replace(".", "")
        if not clean or len(raw_ticker) > 10 or not clean.isalnum() or clean.isnumeric():
            continue

        # Parse quantity — skip if blank or zero (e.g. SPAXX with no quantity)
        qty_raw = cell(row, qty_col).replace(",", "").replace("$", "").lstrip("+")
        if not qty_raw:
            continue
        try:
            shares = float(qty_raw)
        except ValueError:
            skipped.append(raw_ticker)
            continue

        if shares <= 0:
            continue

        # Parse average cost basis per share
        avg_cost = 0.0
        if cost_col:
            cost_raw = cell(row, cost_col).replace("$", "").replace(",", "").lstrip("+")
            try:
                avg_cost = float(cost_raw)
            except ValueError:
                pass  # leave 0.0 — user can edit manually

        # Aggregate duplicate tickers (multiple accounts in same file)
        if raw_ticker in csv_positions:
            prev = csv_positions[raw_ticker]
            total_shares = prev["shares"] + shares
            # Weighted average cost
            total_cost = prev["shares"] * prev["avg_cost"] + shares * avg_cost
            csv_positions[raw_ticker] = {
                "shares": total_shares,
                "avg_cost": round(total_cost / total_shares, 6) if total_shares else 0.0,
            }
        else:
            csv_positions[raw_ticker] = {"shares": shares, "avg_cost": avg_cost}

    if not csv_positions:
        raise HTTPException(
            status_code=400,
            detail=f"No valid positions found in the file. "
                   f"Rows examined: {len(rows)}. Skipped: {skipped[:10]}. "
                   f"Make sure you are exporting from Fidelity Positions, not Activity."
        )

    # Apply import mode
    imported: list[str] = []
    deleted:  list[str] = []

    if mode == "sync":
        existing = db.query(PortfolioPosition).filter(PortfolioPosition.user_id == user.id).all()
        for pos in existing:
            if pos.ticker not in csv_positions:
                db.delete(pos)
                deleted.append(pos.ticker)

    for ticker, data in csv_positions.items():
        pos = (
            db.query(PortfolioPosition)
            .filter(PortfolioPosition.user_id == user.id, PortfolioPosition.ticker == ticker)
            .first()
        )
        if pos:
            if mode in ("overwrite", "sync"):
                pos.shares = data["shares"]
                pos.avg_cost = data["avg_cost"]
                imported.append(ticker)
            # merge mode: skip existing positions
        else:
            db.add(PortfolioPosition(
                user_id=user.id, ticker=ticker,
                shares=data["shares"], avg_cost=data["avg_cost"],
            ))
            imported.append(ticker)

    db.commit()
    return {
        "mode": mode,
        "imported": imported,
        "skipped": skipped,
        "deleted": deleted,
        "count": len(imported),
    }
