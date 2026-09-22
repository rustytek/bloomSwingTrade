from datetime import date, datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status, UploadFile, File
from sqlalchemy.orm import Session
from pydantic import BaseModel
from database.db import get_db
from database.models import User, PortfolioPosition, ClosedTrade
from auth.deps import get_current_user
from services import market_data, portfolio_risk
from services.tickers import normalize_ticker
from services.trade_plan import build_trade_plan
from api.settings import resolve_max_open_r
import csv

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])


class PositionRequest(BaseModel):
    ticker: str
    shares: float
    avg_cost: float
    notes: str | None = None
    stop_loss: float | None = None
    target: float | None = None
    entry_date: date | None = None
    strategy: str | None = None
    # The trade plan's intent. `planned_entry` / `planned_entry_high` are
    # WRITE-ONCE (like initial_stop) — they are the denominator of the
    # entry-chasing execution metric, so an edit must not rewrite them.
    planned_entry: float | None = None
    planned_entry_high: float | None = None
    thesis: str | None = None
    invalidation: str | None = None
    time_stop_days: int | None = None


class AssessRequest(BaseModel):
    """Pre-trade assessment request for the 'Plan a Trade' screen.

    Everything but `ticker` is optional: with no overrides the plan is built from
    cached bars and the user's saved risk settings.
    """
    ticker: str
    entry: float | None = None
    shares: float | None = None
    stop: float | None = None
    atr_mult: float | None = None
    r_multiple: float | None = None
    edge_multiplier: float | None = None


class CloseRequest(BaseModel):
    exit_price: float
    exit_date: date | None = None
    notes: str | None = None


def compute_r_multiple(avg_cost: float, exit_price: float,
                       initial_stop: float | None, stop_loss: float | None):
    """Realized R for a closed long, measured against the INITIAL stop.

    R must reflect the risk actually taken when the trade was opened. Using the
    current (possibly trailed-up) stop as the denominator shrinks the
    denominator over the life of the trade and inflates every recorded R.
    `initial_stop` is null on legacy rows written before the column existed —
    those fall back to stop_loss. Returns None when no valid stop below cost
    exists (a stop at or above cost has no meaningful R).
    """
    r_stop = initial_stop if initial_stop is not None else stop_loss
    if r_stop is None or avg_cost - r_stop <= 0:
        return None, r_stop
    return round((exit_price - avg_cost) / (avg_cost - r_stop), 2), r_stop


def _risk_settings(user: User) -> dict:
    """User risk settings for services.portfolio_risk.

    `max_open_r` is the user's CHOSEN ceiling when `User.max_open_r` is set, and
    otherwise falls back to the derived `max_positions x risk_pct` (the exposure
    of a fully loaded book at full per-trade risk). `max_open_r_basis` records
    which of the two is in play — see api/settings.py::resolve_max_open_r.
    """
    budget, basis = resolve_max_open_r(user)
    return {
        "account_size": user.account_size,
        "risk_pct": user.risk_pct,
        "max_positions": user.max_positions,
        "atr_stop_mult": user.atr_stop_mult,
        "r_multiple": user.r_multiple,
        "max_open_r": budget,
        "max_open_r_basis": basis,
        "max_open_r_chosen": getattr(user, "max_open_r", None),
    }


def _with_basis(heat: dict, basis: str) -> dict:
    """Overwrite portfolio_risk's derived-budget note with the real basis.

    services/portfolio_risk.py is a pure module with no access to the User row,
    so it always stamps the DERIVED note. When the user has chosen a ceiling
    that note is wrong — correct it here rather than editing that module.
    """
    if isinstance(heat, dict):
        heat["budget_basis"] = basis
    return heat


def _risk_unit(user: User) -> float | None:
    """Dollar value of 1R for this user, or None if settings are unusable."""
    try:
        unit = float(user.account_size) * float(user.risk_pct) / 100.0
    except (TypeError, ValueError):
        return None
    return unit if unit > 0 else None


@router.get("/risk")
async def get_portfolio_risk(
    window: int = Query(portfolio_risk.DEFAULT_WINDOW, ge=20, le=500,
                        description="Trailing daily returns used for correlation"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Portfolio-level risk: heat, open risk, concentration, correlation matrix.

    Correlations are Pearson on DAILY RETURNS (never price levels), aligned by
    date, and reported as null for pairs with fewer than
    `portfolio_risk.MIN_OVERLAP` shared observations.
    """
    positions = db.query(PortfolioPosition).filter(PortfolioPosition.user_id == user.id).all()
    settings = _risk_settings(user)
    unit = _risk_unit(user)

    if not positions:
        empty_heat = _with_basis(portfolio_risk.portfolio_heat(
            [], {}, settings["max_open_r"], risk_unit=unit,
            max_positions=user.max_positions,
        ), settings["max_open_r_basis"])
        return {
            "settings": settings,
            "open_risk": portfolio_risk.open_risk([], {}, risk_unit=unit),
            "concentration": portfolio_risk.concentration([], {}),
            "portfolio_heat": empty_heat,
            "correlation": portfolio_risk.correlation_matrix({}, window=window),
        }

    tickers = [p.ticker for p in positions]
    quotes = await market_data.get_batch(tickers, db)
    histories = {t: await market_data.get_history(t, db) for t in tickers}

    return {
        "settings": settings,
        "open_risk": portfolio_risk.open_risk(positions, quotes, risk_unit=unit),
        "concentration": portfolio_risk.concentration(positions, quotes),
        "portfolio_heat": _with_basis(portfolio_risk.portfolio_heat(
            positions, quotes, settings["max_open_r"], risk_unit=unit,
            max_positions=user.max_positions,
        ), settings["max_open_r_basis"]),
        "correlation": portfolio_risk.correlation_matrix(histories, window=window),
    }


@router.post("/assess")
async def assess_position(
    req: AssessRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Pre-trade check for a candidate position — the 'Plan a Trade' backend.

    Builds a trade plan from cached bars + the user's risk settings (any field
    can be overridden on the request), then runs it through
    `portfolio_risk.assess_new_position` for correlation / sector / open-R /
    slot checks and a before -> after projection.
    """
    ticker = normalize_ticker(req.ticker)
    bars = await market_data.get_history(ticker, db)
    if not bars:
        raise HTTPException(status_code=404, detail=f"No price history available for {ticker}")

    plan = build_trade_plan(
        bars,
        account_size=user.account_size,
        risk_pct=user.risk_pct,
        entry=req.entry,
        atr_mult=req.atr_mult if req.atr_mult is not None else user.atr_stop_mult,
        r_multiple=req.r_multiple if req.r_multiple is not None else user.r_multiple,
        edge_multiplier=req.edge_multiplier if req.edge_multiplier is not None else 1.0,
    )
    if not plan:
        raise HTTPException(status_code=422,
                            detail=f"Could not build a trade plan for {ticker} from cached data")

    # Manual overrides win over the computed plan.
    if req.stop is not None:
        plan = dict(plan)
        plan["stop"] = req.stop
        plan["initial_stop"] = req.stop
    if req.shares is not None:
        plan = dict(plan)
        plan["shares"] = req.shares

    positions = db.query(PortfolioPosition).filter(PortfolioPosition.user_id == user.id).all()
    held = [p.ticker for p in positions]
    quote_tickers = sorted(set(held) | {ticker})
    quotes = await market_data.get_batch(quote_tickers, db)
    histories = {t: await market_data.get_history(t, db) for t in held}
    histories[ticker] = bars

    settings = _risk_settings(user)
    assessment = portfolio_risk.assess_new_position(
        ticker, plan, positions, quotes, histories, settings
    )
    if isinstance(assessment, dict):
        assessment["budget_basis"] = settings["max_open_r_basis"]
    return assessment


@router.get("")
async def get_portfolio(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    positions = db.query(PortfolioPosition).filter(PortfolioPosition.user_id == user.id).all()
    budget, basis = resolve_max_open_r(user)
    if not positions:
        return {
            "positions": [],
            "summary": {
                "total_cost": 0, "total_mv": 0, "total_pnl": 0, "total_pnl_pct": 0,
                "open_r": 0.0,
                "portfolio_heat": _with_basis(portfolio_risk.portfolio_heat(
                    [], {}, budget,
                    risk_unit=_risk_unit(user), max_positions=user.max_positions,
                ), basis),
            },
        }

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
            "stop_loss": pos.stop_loss,
            "initial_stop": pos.initial_stop,
            "target": pos.target,
            "entry_date": pos.entry_date.isoformat() if pos.entry_date else None,
            "strategy": pos.strategy,
            "planned_entry": pos.planned_entry,
            "planned_entry_high": pos.planned_entry_high,
            "thesis": pos.thesis,
            "invalidation": pos.invalidation,
            "time_stop_days": pos.time_stop_days,
        })

    total_pnl = total_mv - total_cost
    heat = _with_basis(portfolio_risk.portfolio_heat(
        positions, quotes, budget,
        risk_unit=_risk_unit(user), max_positions=user.max_positions,
    ), basis)
    return {
        "positions": result,
        "summary": {
            "total_cost": round(total_cost, 2),
            "total_mv": round(total_mv, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl / total_cost * 100 if total_cost > 0 else 0, 2),
            "open_r": heat["open_r"],
            "portfolio_heat": heat,
        },
    }


@router.post("", status_code=status.HTTP_201_CREATED)
def upsert_position(
    req: PositionRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    ticker = normalize_ticker(req.ticker)
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
        if req.stop_loss is not None:
            pos.stop_loss = req.stop_loss
            # Record the FIRST stop only — never overwrite it. Trailing a stop up
            # must not change the risk the trade was originally taken with.
            if pos.initial_stop is None:
                pos.initial_stop = req.stop_loss
        if req.target is not None:
            pos.target = req.target
        if req.entry_date is not None:
            pos.entry_date = req.entry_date
        if req.strategy is not None:
            pos.strategy = req.strategy
        # Plan intent: the thesis and its invalidation are living text and may
        # be edited. The PLANNED ENTRY is not — it is write-once, for the same
        # reason initial_stop is: it is the reference the fill is judged
        # against, and rewriting it would erase the record of a chased entry.
        if req.thesis is not None:
            pos.thesis = req.thesis
        if req.invalidation is not None:
            pos.invalidation = req.invalidation
        if req.time_stop_days is not None:
            pos.time_stop_days = req.time_stop_days
        if req.planned_entry is not None and pos.planned_entry is None:
            pos.planned_entry = req.planned_entry
        if req.planned_entry_high is not None and pos.planned_entry_high is None:
            pos.planned_entry_high = req.planned_entry_high
    else:
        pos = PortfolioPosition(
            user_id=user.id, ticker=ticker,
            shares=req.shares, avg_cost=req.avg_cost, notes=req.notes,
            stop_loss=req.stop_loss, initial_stop=req.stop_loss, target=req.target,
            entry_date=req.entry_date, strategy=req.strategy,
            planned_entry=req.planned_entry, planned_entry_high=req.planned_entry_high,
            thesis=req.thesis, invalidation=req.invalidation,
            time_stop_days=req.time_stop_days,
        )
        db.add(pos)

    db.commit()
    return {"ticker": ticker, "shares": pos.shares, "avg_cost": pos.avg_cost}


@router.post("/{ticker}/close", status_code=status.HTTP_201_CREATED)
def close_position(
    ticker: str,
    req: CloseRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Close (sell) a position and archive it to the trade journal.

    Use this — not DELETE — when you actually exit a trade, so the realized
    P&L and R-multiple are recorded. DELETE remains for correcting mistaken
    entries that were never really held.
    """
    ticker = normalize_ticker(ticker)
    pos = (
        db.query(PortfolioPosition)
        .filter(PortfolioPosition.user_id == user.id, PortfolioPosition.ticker == ticker)
        .first()
    )
    if not pos:
        raise HTTPException(status_code=404, detail="Position not found")

    cost_basis = pos.shares * pos.avg_cost
    pnl = pos.shares * req.exit_price - cost_basis
    pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0.0

    # R-multiple against the INITIAL stop (legacy rows fall back to stop_loss).
    r_multiple, r_stop = compute_r_multiple(
        pos.avg_cost, req.exit_price, pos.initial_stop, pos.stop_loss
    )

    trade = ClosedTrade(
        user_id=user.id,
        ticker=ticker,
        shares=pos.shares,
        avg_cost=pos.avg_cost,
        exit_price=req.exit_price,
        entry_date=pos.entry_date,
        exit_date=req.exit_date or date.today(),
        stop_loss=pos.stop_loss,
        initial_stop=r_stop,
        target=pos.target,
        strategy=pos.strategy,
        pnl=round(pnl, 2),
        pnl_pct=round(pnl_pct, 2),
        r_multiple=r_multiple,
        # Carry the plan's intent into the journal — without planned_entry the
        # entry-chasing execution metric has nothing to measure the fill against.
        planned_entry=pos.planned_entry,
        planned_entry_high=pos.planned_entry_high,
        thesis=pos.thesis,
        invalidation=pos.invalidation,
        time_stop_days=pos.time_stop_days,
        notes=req.notes or pos.notes,
        opened_at=pos.added_at,
        closed_at=datetime.now(timezone.utc),
    )
    db.add(trade)
    db.delete(pos)
    db.commit()
    db.refresh(trade)
    return {
        "id": trade.id,
        "ticker": trade.ticker,
        "pnl": trade.pnl,
        "pnl_pct": trade.pnl_pct,
        "r_multiple": trade.r_multiple,
    }


@router.delete("/{ticker}", status_code=status.HTTP_204_NO_CONTENT)
def remove_position(
    ticker: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    pos = (
        db.query(PortfolioPosition)
        .filter(PortfolioPosition.user_id == user.id, PortfolioPosition.ticker == normalize_ticker(ticker))
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
        raw_ticker = raw_ticker.replace("/", "-").upper().strip()

        if raw_ticker in SKIP_TICKERS:
            continue

        # Skip footer text, totals rows, and obviously non-ticker values.
        # Valid tickers: 1-10 chars, alphanumeric + hyphens (allows fund codes with
        # digits like target-date funds, employer plan codes, QACA holdings, etc.)
        # Pure-number strings ("1234.56") are rejected; spaces indicate footer text.
        clean = raw_ticker.replace("-", "").replace(".", "")
        if not clean or len(raw_ticker) > 10 or not clean.isalnum() or clean.isnumeric():
            continue
        raw_ticker = normalize_ticker(raw_ticker)

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
