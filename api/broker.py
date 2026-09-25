"""
/api/broker — the Trade page's backend (Robinhood via its official agentic
trading MCP server, plus paper mode). All logic lives in
services/broker_service.py; this module maps it to HTTP.

No response from here ever carries a token or client secret.
"""
from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from auth.deps import get_current_user
from config import get_settings
from database.db import get_db
from database.models import User
from services import broker_service as svc
from services import secrets_box
from services.brokers import robinhood_mcp as mcp

router = APIRouter(prefix="/api/broker", tags=["broker"])

CALLBACK_PATH = "/api/broker/oauth/callback"


class PlanIn(BaseModel):
    stop: float | None = None
    target: float | None = None
    strategy: str | None = None
    strategy_name: str | None = None
    planned_entry: float | None = None
    planned_entry_high: float | None = None
    thesis: str | None = None
    invalidation: str | None = None
    time_stop_days: int | None = None


class OrderIn(BaseModel):
    plan_item_id: str | None = Field(default=None, max_length=64)
    ticker: str = Field(max_length=16)
    side: str
    shares: float
    limit_price: float
    time_in_force: str = "gfd"
    plan: PlanIn | None = None


class OrdersIn(BaseModel):
    orders: list[OrderIn]
    confirm: bool = False


class ModeIn(BaseModel):
    mode: str
    confirm: bool = False


class ResolveIn(BaseModel):
    outcome: str
    fill_price: float | None = None
    filled_quantity: float | None = None


def public_origin(request: Request) -> str:
    """This app's public origin: PUBLIC_URL if set, else the request's own
    origin honouring X-Forwarded-Proto/Host (the add-on sits behind cloudflared)."""
    configured = (get_settings().public_url or "").strip().rstrip("/")
    if configured:
        return configured
    proto = (request.headers.get("x-forwarded-proto") or request.url.scheme or "https").split(",")[0].strip()
    host = (request.headers.get("x-forwarded-host") or request.headers.get("host")
            or request.url.netloc).split(",")[0].strip()
    return f"{proto}://{host}"


def redirect_uri_for(request: Request) -> str:
    return public_origin(request) + CALLBACK_PATH


def _orders_payload(body: OrdersIn) -> list[dict]:
    out = []
    for o in body.orders:
        d = o.model_dump()
        d["plan"] = (o.plan.model_dump() if o.plan else {})
        out.append(d)
    return out


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, mcp.ReauthRequired):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, secrets_box.SecretsUnavailable):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, mcp.BrokerError):
        return HTTPException(status_code=502, detail=str(exc))
    if isinstance(exc, LookupError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    raise exc


@router.get("/status")
def broker_status(request: Request, db: Session = Depends(get_db),
                  user: User = Depends(get_current_user)):
    return svc.status(db, user, redirect_uri_for(request))


@router.post("/connect")
async def broker_connect(request: Request, db: Session = Depends(get_db),
                         user: User = Depends(get_current_user)):
    """Start the Robinhood sign-in. Returns {status:"redirect", authorize_url}
    for the browser to navigate to, or {status:"error", message}."""
    return await svc.begin_connect(db, user, redirect_uri_for(request))


@router.get("/oauth/callback", include_in_schema=False)
async def broker_oauth_callback(
    state: str | None = None, code: str | None = None,
    error: str | None = None, error_description: str | None = None,
    db: Session = Depends(get_db),
):
    """Browser redirect from robinhood.com — no bearer token; the single-use,
    user-bound, 10-minute `state` is the authentication."""
    try:
        ok, message = await svc.complete_oauth(db, state, code, error, error_description)
    except Exception as exc:  # noqa: BLE001 — never show a traceback page mid-OAuth
        ok, message = False, "Connecting failed: " + (str(exc) or type(exc).__name__)
    if ok:
        return RedirectResponse("/trade?connected=1", status_code=302)
    return RedirectResponse("/trade?error=" + quote(message[:400]), status_code=302)


@router.post("/disconnect")
def broker_disconnect(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return svc.disconnect(db, user)


@router.put("/mode")
def broker_mode(body: ModeIn, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    try:
        return svc.set_mode(db, user, body.mode, body.confirm)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/tools")
async def broker_tools(refresh: bool = False, db: Session = Depends(get_db),
                       user: User = Depends(get_current_user)):
    """Discovered MCP tool names + input schemas (debugging aid)."""
    try:
        return await svc.list_tools(db, user, refresh=refresh)
    except Exception as exc:  # noqa: BLE001
        raise _http_error(exc)


@router.get("/account")
async def broker_account(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    try:
        return await svc.account(db, user)
    except Exception as exc:  # noqa: BLE001
        raise _http_error(exc)


class ImportIn(BaseModel):
    tickers: list[str] = Field(max_length=svc.MAX_IMPORT)


@router.get("/holdings")
async def broker_holdings(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Read-only: Robinhood Agentic holdings vs the SwingTrader portfolio."""
    try:
        return await svc.holdings(db, user)
    except Exception as exc:  # noqa: BLE001
        raise _http_error(exc)


@router.post("/holdings/import")
async def broker_import_holdings(body: ImportIn, db: Session = Depends(get_db),
                                 user: User = Depends(get_current_user)):
    """Add the chosen Robinhood-only holdings to the portfolio. Shares/cost are
    re-read from Robinhood server-side; existing positions are never changed."""
    try:
        return await svc.import_holdings(db, user, body.tickers)
    except Exception as exc:  # noqa: BLE001
        raise _http_error(exc)


@router.post("/orders/preview")
async def broker_preview(body: OrdersIn, db: Session = Depends(get_db),
                         user: User = Depends(get_current_user)):
    try:
        return await svc.preview(db, user, _orders_payload(body))
    except Exception as exc:  # noqa: BLE001
        raise _http_error(exc)


@router.post("/orders")
async def broker_place(body: OrdersIn, db: Session = Depends(get_db),
                       user: User = Depends(get_current_user)):
    try:
        return await svc.place(db, user, _orders_payload(body), body.confirm)
    except Exception as exc:  # noqa: BLE001
        raise _http_error(exc)


@router.get("/orders")
def broker_orders(limit: int = Query(50, ge=1, le=500), db: Session = Depends(get_db),
                  user: User = Depends(get_current_user)):
    return {"orders": svc.list_orders(db, user, limit)}


@router.post("/orders/sync")
async def broker_sync(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    try:
        return await svc.sync(db, user)
    except Exception as exc:  # noqa: BLE001
        raise _http_error(exc)


@router.post("/orders/{order_id}/cancel")
async def broker_cancel(order_id: int, db: Session = Depends(get_db),
                        user: User = Depends(get_current_user)):
    try:
        return await svc.cancel(db, user, order_id)
    except Exception as exc:  # noqa: BLE001
        raise _http_error(exc)


@router.post("/orders/{order_id}/resolve")
def broker_resolve(order_id: int, body: ResolveIn, db: Session = Depends(get_db),
                   user: User = Depends(get_current_user)):
    """Record a fill/cancel the user confirmed in the Robinhood app — for when
    the MCP server exposes no order-status tool."""
    try:
        return svc.resolve_manually(db, user, order_id, body.outcome, body.fill_price,
                                    body.filled_quantity)
    except Exception as exc:  # noqa: BLE001
        raise _http_error(exc)
