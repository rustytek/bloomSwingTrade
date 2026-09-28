"""
Broker orchestration for the Trade page — paper mode and Robinhood (official
agentic-trading MCP server, see services/brokers/robinhood_mcp.py).

Safety model, in order of importance:
  1. PAPER BY DEFAULT. A new link starts in paper mode; paper orders are
     recorded as "simulated", nothing is sent anywhere, and the portfolio is
     never touched.
  2. LIVE NEEDS TWO YESES. Switching to live needs a connected account plus an
     explicit confirm, and every live submit needs its own confirm:true.
  3. LIMIT ORDERS ONLY, sanity-checked. Whole shares, max 20 per batch, no
     duplicate ticker+side, limit within ±15% of the last known price, sells no
     larger than the shares held, buys within buying power when it is known.
  4. ROBINHOOD REVIEWS EVERY LIVE ORDER FIRST. review_equity_order runs before
     place_equity_order and its words are shown verbatim.
  5. FILLS REACH THE PORTFOLIO EXACTLY ONCE (BrokerOrder.applied_to_portfolio).
  6. NEVER THE SAME ORDER TWICE (ML4T gap 8). Every live order row is written
     with a client UUID BEFORE the network call (status `sending`); a place
     call that times out or breaks after it may have reached Robinhood leaves
     the row `unknown`. While any `sending`/`unknown`/open order exists for a
     ticker+side, another is refused; an identical order already filled today
     is refused too. The UUID is passed to Robinhood when the place tool's
     schema has an idempotency field (ref_id, client_order_id, …).
  7. THE BOOK MUST MATCH BEFORE LIVE TRADING. Live preview/place compare
     Robinhood's holdings with SwingTrader's portfolio: a sell whose ticker is
     `shares_differ` or `swingtrader_only` is blocked; buys while holdings
     disagree (`shares_differ` / `robinhood_only`, or holdings unreadable)
     need an explicit acknowledgement.
  8. NO SIZING OFF A STALE PRICE. Live orders need a quote younger than
     QUOTE_MAX_AGE_OPEN_S while the market is open, or taken after the last
     close when it is shut. Preview refreshes stale quotes; place refuses them.

Network calls run in a worker thread (anyio.to_thread) — they are short I/O
bounded by per-request timeouts, not CPU work, so they don't need the
background-job subprocess (see CLAUDE.md "Long Builds Are Background Jobs").
Only the event-loop thread touches the DB session.
"""
from __future__ import annotations

import asyncio
import json
import re
import secrets
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

import anyio
from sqlalchemy.orm import Session

from database.models import BrokerAccount, BrokerOrder, PortfolioPosition, User
from services import circuit_breaker, secrets_box
from services.brokers import robinhood_mcp as mcp
from services.journal import journal_close

MAX_ORDERS = 20
MAX_SHARES = 100_000
LIMIT_BAND = 0.15          # limit must be within ±15% of the last price
OAUTH_STATE_TTL = 600      # seconds
TERMINAL = {"filled", "cancelled", "canceled", "rejected", "failed", "simulated", "expired"}
# A live order whose placement outcome we never saw. Non-terminal: blocks
# re-placing the same ticker+side until sync or a manual resolve settles it.
UNRESOLVED = {"sending", "unknown"}
# Live orders are sized/sanity-checked off the cached quote. While the market
# is open it must be at most this old; when shut, taken after the last close.
QUOTE_MAX_AGE_OPEN_S = 20 * 60
# Holdings-comparison statuses (compare_holdings) that gate live orders.
HOLDINGS_SELL_BLOCK = ("shares_differ", "swingtrader_only")
HOLDINGS_BUY_ACK = ("shares_differ", "robinhood_only")
PLAN_KEYS = ("stop", "target", "strategy", "strategy_name", "planned_entry",
             "planned_entry_high", "thesis", "invalidation", "time_stop_days")
_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.-]{0,14}$")

# state -> {user_id, verifier, client_id, redirect_uri, meta, created}
_pending_oauth: dict[str, dict] = {}
_locks: dict[int, asyncio.Lock] = {}


def _lock(user_id: int) -> asyncio.Lock:
    lk = _locks.get(user_id)
    if lk is None:
        lk = _locks[user_id] = asyncio.Lock()
    return lk


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── Quote source (swappable in tests) ──────────────────────────────────────

async def _app_quotes(tickers: list[str], db: Session, refresh: bool = False) -> dict[str, dict]:
    """{TICKER: {"price": float, "as_of": iso-str|None}}. `refresh=True`
    re-fetches (used by a live preview when a cached quote is too old)."""
    from services import market_data
    out: dict[str, dict] = {}
    try:
        if refresh:
            quotes = [await market_data.get_quote(t, db, force_refresh=True) for t in tickers]
        else:
            quotes = await market_data.get_batch(tickers, db)
        for q in quotes:
            if q and q.get("ticker") and q.get("price"):
                out[q["ticker"].upper()] = {"price": float(q["price"]),
                                            "as_of": q.get("last_updated_iso")}
    except Exception:  # noqa: BLE001 — a quote outage becomes "no price", handled per order
        pass
    return out

QUOTE_SOURCE: Callable = _app_quotes


def _parse_ts(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _split_quotes(raw: dict) -> tuple[dict[str, float], dict[str, datetime | None]]:
    """Accepts {T: price} or {T: {"price", "as_of"}} -> (prices, as_of)."""
    prices: dict[str, float] = {}
    as_of: dict[str, datetime | None] = {}
    for t, v in (raw or {}).items():
        if isinstance(v, dict):
            p, ts = v.get("price"), _parse_ts(v.get("as_of"))
        else:
            p, ts = v, None
        try:
            p = float(p)
        except (TypeError, ValueError):
            continue
        if p > 0:
            prices[str(t).upper()] = p
            as_of[str(t).upper()] = ts
    return prices, as_of


def quote_staleness(as_of: datetime | None, now: datetime | None = None,
                    market_open: bool | None = None, last_close: datetime | None = None) -> str | None:
    """None when a quote is fresh enough to size a LIVE order, else why not.

    Market open: at most QUOTE_MAX_AGE_OPEN_S old. Market shut: taken at or
    after the most recent NYSE close (services/market_data's own freshness
    rule — a price from before the close is not the closing price). Unknown
    timestamp = stale: an order is never sized off a price of unknown age."""
    if as_of is None:
        return "has no timestamp"
    now = now or _now()
    as_of = as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)
    if market_open is None or (not market_open and last_close is None):
        from services import market_data
        if market_open is None:
            market_open = market_data.is_market_open()
        if not market_open and last_close is None:
            last_close = market_data._last_market_close()
    if market_open:
        age = (now - as_of).total_seconds()
        if age > QUOTE_MAX_AGE_OPEN_S:
            return (f"is {int(age // 60)} min old (the market is open; live orders need a price "
                    f"under {QUOTE_MAX_AGE_OPEN_S // 60} min old)")
        return None
    if as_of < last_close:
        return "was taken before the last market close"
    return None


# ── Account row helpers ────────────────────────────────────────────────────

def get_account(db: Session, user_id: int, create: bool = False) -> BrokerAccount | None:
    acct = db.query(BrokerAccount).filter(BrokerAccount.user_id == user_id).first()
    if acct is None and create:
        acct = BrokerAccount(user_id=user_id, broker="robinhood", mode="paper")
        db.add(acct)
        db.commit()
        db.refresh(acct)
    return acct


def is_connected(acct: BrokerAccount | None) -> bool:
    return bool(acct and acct.access_token_enc)


def _mask(value: str | None) -> str | None:
    if not value:
        return None
    return "****" + str(value)[-4:]


def _tools(acct: BrokerAccount) -> list[dict]:
    try:
        return json.loads(acct.tools_json) if acct and acct.tools_json else []
    except ValueError:
        return []


def _clear_tokens(acct: BrokerAccount, reason: str | None = None) -> None:
    acct.access_token_enc = None
    acct.refresh_token_enc = None
    acct.token_expires_at = None
    acct.mcp_session_id = None
    acct.connected_at = None
    if reason:
        acct.last_error = reason


def status(db: Session, user: User, redirect_uri: str | None = None) -> dict:
    acct = get_account(db, user.id)
    tools = _tools(acct) if acct else []
    pending = any(p["user_id"] == user.id and time.time() - p["created"] < OAUTH_STATE_TTL
                  for p in _pending_oauth.values())
    return {
        "broker": "robinhood",
        "connected": is_connected(acct),
        "mode": (acct.mode if acct else "paper") or "paper",
        "account_number_masked": _mask(acct.account_number) if acct else None,
        "connected_at": acct.connected_at.isoformat() if acct and acct.connected_at else None,
        "live_enabled_at": acct.live_enabled_at.isoformat() if acct and acct.live_enabled_at else None,
        "last_error": acct.last_error if acct else None,
        "redirect_uri": redirect_uri,
        "registered_redirect_uri": acct.redirect_uri if acct else None,
        "oauth_pending": pending,
        "tools_discovered": len(tools),
        "review_tool": (mcp.find_tool(tools, "review") or {}).get("name"),
        "place_tool": (mcp.find_tool(tools, "place") or {}).get("name"),
        "order_status_tool": (mcp.find_tool(tools, "order") or {}).get("name"),
    }


# ── OAuth connect ──────────────────────────────────────────────────────────

def _prune_pending() -> None:
    cutoff = time.time() - OAUTH_STATE_TTL
    for k in [k for k, v in _pending_oauth.items() if v["created"] < cutoff]:
        _pending_oauth.pop(k, None)


async def begin_connect(db: Session, user: User, redirect_uri: str) -> dict:
    """Discover, register (once per redirect URI), and return the authorize URL."""
    acct = get_account(db, user.id, create=True)
    reuse_client = acct.client_id if acct.redirect_uri == redirect_uri else None

    def work():
        meta = mcp.discover()
        client_id = reuse_client or mcp.register_client(meta, redirect_uri)
        return meta, client_id

    try:
        meta, client_id = await anyio.to_thread.run_sync(work)
    except mcp.BrokerError as exc:
        acct.last_error = str(exc)
        db.commit()
        return {"status": "error", "message": str(exc), "redirect_uri": redirect_uri}

    acct.client_id = client_id
    acct.redirect_uri = redirect_uri
    acct.last_error = None
    db.commit()

    _prune_pending()
    state = secrets.token_urlsafe(32)
    verifier, challenge = mcp.pkce_pair()
    _pending_oauth[state] = {"user_id": user.id, "verifier": verifier, "client_id": client_id,
                             "redirect_uri": redirect_uri, "meta": meta, "created": time.time()}
    return {"status": "redirect", "redirect_uri": redirect_uri,
            "authorize_url": mcp.authorize_url(meta, client_id, redirect_uri, state, challenge)}


def _store_tokens(acct: BrokerAccount, tokens: dict) -> None:
    acct.access_token_enc = secrets_box.encrypt(tokens["access_token"])
    if tokens.get("refresh_token"):
        acct.refresh_token_enc = secrets_box.encrypt(tokens["refresh_token"])
    try:
        secs = int(tokens.get("expires_in") or 0)
    except (TypeError, ValueError):
        secs = 0
    acct.token_expires_at = (_now() + timedelta(seconds=secs)) if secs > 0 else None


async def complete_oauth(db: Session, state: str | None, code: str | None,
                         error: str | None = None, error_description: str | None = None) -> tuple[bool, str]:
    """Handle the browser's return from robinhood.com. The `state` IS the auth:
    it is single-use, expires after 10 minutes, and is bound to one user."""
    _prune_pending()
    pending = _pending_oauth.pop(state, None) if state else None
    if not pending:
        return False, ("This Robinhood sign-in link is invalid or expired (links last 10 minutes "
                       "and work once). Start again from the Trade page.")
    acct = get_account(db, pending["user_id"], create=True)
    if error:
        msg = "Robinhood did not authorize SwingTrader: " + (error_description or error)
        acct.last_error = msg[:500]
        db.commit()
        return False, msg
    if not code:
        return False, "Robinhood returned without an authorization code."

    def work():
        tokens = mcp.exchange_code(pending["meta"], pending["client_id"], pending["redirect_uri"],
                                   code, pending["verifier"])
        # Open the MCP session and learn the tools right away, so problems
        # surface on the Connect step rather than on the first live order.
        session_id = None
        tools: list[dict] = []
        note = None
        try:
            with mcp.McpSession(tokens["access_token"]) as s:
                s.initialize()
                session_id = s.session_id
                tools = s.list_tools()
                account_number = None
                acct_tool = mcp.find_tool(tools, "account")
                if acct_tool:
                    try:
                        r = s.call_tool(acct_tool["name"], mcp.map_arguments(acct_tool, {}))
                        account_number = mcp.pick_agentic_account(r["data"])
                    except mcp.BrokerError:
                        pass
        except mcp.BrokerError as exc:
            note = "Signed in, but the trading service did not answer yet: " + str(exc)
            account_number = None
        return tokens, session_id, tools, account_number, note

    try:
        tokens, session_id, tools, account_number, note = await anyio.to_thread.run_sync(work)
    except mcp.BrokerError as exc:
        acct.last_error = str(exc)[:500]
        db.commit()
        return False, str(exc)

    _store_tokens(acct, tokens)
    acct.mcp_session_id = session_id
    acct.tools_json = json.dumps(tools) if tools else acct.tools_json
    if account_number:
        acct.account_number = account_number
    acct.connected_at = _now()
    acct.last_error = note
    db.commit()
    return True, "Connected to Robinhood."


def disconnect(db: Session, user: User) -> dict:
    acct = get_account(db, user.id)
    if acct:
        _clear_tokens(acct)
        acct.tools_json = None
        acct.account_number = None
        acct.mode = "paper"
        acct.live_enabled_at = None
        acct.last_error = None
        db.commit()
    for k in [k for k, v in _pending_oauth.items() if v["user_id"] == user.id]:
        _pending_oauth.pop(k, None)
    return {"status": "disconnected",
            "message": "Tokens deleted from SwingTrader. You can also revoke access in the "
                       "Robinhood app (Account → Settings → Security → connected apps)."}


def set_mode(db: Session, user: User, mode: str, confirm: bool) -> dict:
    if mode not in ("paper", "live"):
        raise ValueError("mode must be 'paper' or 'live'")
    acct = get_account(db, user.id, create=True)
    if mode == "live":
        if not is_connected(acct):
            raise ValueError("Connect your Robinhood Agentic account before switching to live.")
        if not confirm:
            raise ValueError("Switching to live needs confirm: true — live orders use real money.")
        acct.live_enabled_at = _now()
    acct.mode = mode
    db.commit()
    return {"mode": acct.mode}


# ── Live MCP calls ─────────────────────────────────────────────────────────

async def _live(db: Session, acct: BrokerAccount, fn: Callable[[mcp.McpSession, list[dict]], Any]):
    """Run fn(session, tools) in a thread with a fresh MCP session; persist any
    refreshed token / new session id / tool list; on ReauthRequired, disconnect."""
    if not is_connected(acct):
        raise mcp.ReauthRequired("Robinhood is not connected.")
    access = secrets_box.decrypt(acct.access_token_enc)
    refresh = secrets_box.decrypt_opt(acct.refresh_token_enc)
    client_id = acct.client_id
    cached_tools = _tools(acct)
    holder: dict = {}

    def refresher():
        if not refresh or not client_id:
            return None
        try:
            tokens = mcp.refresh_access_token(mcp.discover(), client_id, refresh)
        except mcp.BrokerError:
            return None
        holder["tokens"] = tokens
        return tokens["access_token"]

    def work():
        with mcp.McpSession(access, session_id=acct.mcp_session_id, refresher=refresher) as s:
            tools = cached_tools
            if not tools:
                tools = s.list_tools()
                holder["tools"] = tools
            result = fn(s, tools)
            holder["session_id"] = s.session_id
            return result

    try:
        return await anyio.to_thread.run_sync(work)
    except mcp.ReauthRequired as exc:
        _clear_tokens(acct, str(exc))
        db.commit()
        raise
    finally:
        if "tokens" in holder:
            _store_tokens(acct, holder["tokens"])
        if holder.get("tools"):
            acct.tools_json = json.dumps(holder["tools"])
        if holder.get("session_id"):
            acct.mcp_session_id = holder["session_id"]
        if holder:
            db.commit()


async def list_tools(db: Session, user: User, refresh: bool = False) -> dict:
    acct = get_account(db, user.id)
    if not is_connected(acct):
        return {"connected": False, "tools": []}
    if refresh:
        acct.tools_json = None
        db.commit()
        await _live(db, acct, lambda s, tools: None)
    return {"connected": True, "tools": _tools(acct)}


def _order_fields(o: dict, account_number: str | None) -> dict:
    return {
        "symbol": o["ticker"].replace("-", "."),
        "side": o["side"],
        "quantity": o["shares"],
        "order_type": "limit",
        "limit_price": round(float(o["limit_price"]), 4 if o["limit_price"] < 1 else 2),
        "time_in_force": o.get("time_in_force") or "gfd",
        "account_number": account_number,
    }


# ── Account snapshot ───────────────────────────────────────────────────────

def _paper_account(db: Session, user: User) -> dict:
    positions = db.query(PortfolioPosition).filter(PortfolioPosition.user_id == user.id).all()
    cost = sum((p.shares or 0) * (p.avg_cost or 0) for p in positions)
    size = float(user.account_size or 10000.0)
    return {
        "source": "paper",
        "buying_power": round(max(0.0, size - cost), 2),
        "cash": round(max(0.0, size - cost), 2),
        "positions": [{"ticker": p.ticker, "shares": p.shares, "avg_cost": p.avg_cost} for p in positions],
        "note": "Paper account = your SwingTrader portfolio; buying power = account size minus cost basis.",
    }


async def _live_account(db: Session, acct: BrokerAccount) -> dict:
    def fn(s: mcp.McpSession, tools: list[dict]):
        out: dict = {"source": "robinhood", "buying_power": None, "cash": None,
                     "positions": None, "note": None, "account_text": None}
        t = mcp.find_tool(tools, "account")
        if t:
            try:
                r = s.call_tool(t["name"], mcp.map_arguments(t, {"account_number": acct.account_number}))
                out["account_text"] = r["text"][:1500]
                number = acct.account_number or mcp.pick_agentic_account(r["data"])
                entry = mcp.account_entry(r["data"], number)
                out["buying_power"] = mcp.to_float(mcp.find_value(entry, ("buying_power",)))
                out["cash"] = mcp.to_float(mcp.find_value(entry, ("cash", "cash_balance")))
                out["account_number"] = number
            except mcp.MappingError as exc:
                out["note"] = str(exc)
        t = mcp.find_tool(tools, "positions")
        if t:
            try:
                # Use the account just resolved above — on the first call
                # acct.account_number is still empty, and an unscoped positions
                # call could return the MAIN account's holdings.
                number = out.get("account_number") or acct.account_number
                r = s.call_tool(t["name"], mcp.map_arguments(t, {"account_number": number}))
                out["positions"] = mcp.parse_positions(r["data"])
            except mcp.MappingError as exc:
                out["note"] = str(exc)
        return out

    out = await _live(db, acct, fn)
    if out.get("account_number") and not acct.account_number:
        acct.account_number = out["account_number"]
        db.commit()
    out.pop("account_number", None)
    return out


async def account(db: Session, user: User) -> dict:
    acct = get_account(db, user.id)
    mode = (acct.mode if acct else "paper") or "paper"
    if mode == "live" and is_connected(acct):
        async with _lock(user.id):
            data = await _live_account(db, acct)
    else:
        data = _paper_account(db, user)
    data["mode"] = mode
    data["connected"] = is_connected(acct)
    return data


# ── Holdings: Robinhood vs SwingTrader ─────────────────────────────────────
#
# READ-ONLY comparison. Works in paper mode too (reading holdings sends no
# order). Nothing here changes the portfolio except import_holdings(), which
# only ever ADDS tickers SwingTrader does not track yet, and only the ones the
# user explicitly asked for.

MAX_IMPORT = 100
_SHARE_EPS = 1e-6


def _key(ticker: str) -> str:
    """Robinhood writes BRK.B; SwingTrader (yfinance) writes BRK-B."""
    return str(ticker or "").upper().strip().replace(".", "-")


def compare_holdings(rh_positions: list[dict], st_positions: list[PortfolioPosition]) -> list[dict]:
    """One row per ticker held on either side. Pure — no DB, no network."""
    rh = {}
    for p in rh_positions:
        k = _key(p.get("ticker"))
        if k:
            rh[k] = p
    st = {_key(p.ticker): p for p in st_positions}
    rows = []
    for k in sorted(set(rh) | set(st)):
        r, s = rh.get(k), st.get(k)
        rh_sh = mcp.to_float(r.get("shares")) if r else None
        rh_cost = mcp.to_float(r.get("avg_cost")) if r else None
        st_sh = float(s.shares) if s else None
        if r and s:
            status_ = "match" if abs((rh_sh or 0) - st_sh) <= _SHARE_EPS else "shares_differ"
        else:
            status_ = "robinhood_only" if r else "swingtrader_only"
        importable = status_ == "robinhood_only" and bool(rh_sh and rh_sh > 0) \
            and bool(rh_cost and rh_cost > 0)
        rows.append({
            "ticker": k, "status": status_, "importable": importable,
            "robinhood_shares": rh_sh, "robinhood_avg_cost": rh_cost,
            "swingtrader_shares": st_sh, "swingtrader_avg_cost": float(s.avg_cost) if s else None,
            "reason": None if importable or status_ != "robinhood_only"
            else "Robinhood did not report an average cost, so it can't be imported with a correct cost basis.",
        })
    return rows


async def _read_rh_positions(db: Session, user: User) -> tuple[BrokerAccount, dict]:
    acct = get_account(db, user.id)
    if not is_connected(acct):
        raise mcp.ReauthRequired("Connect Robinhood first (step 1).")
    async with _lock(user.id):
        snap = await _live_account(db, acct)
    return acct, snap


def _st_positions(db: Session, user: User) -> list[PortfolioPosition]:
    return db.query(PortfolioPosition).filter(PortfolioPosition.user_id == user.id).all()


async def holdings(db: Session, user: User) -> dict:
    acct = get_account(db, user.id)
    base = {"connected": is_connected(acct), "mode": (acct.mode if acct else "paper") or "paper",
            "account_number_masked": _mask(acct.account_number) if acct else None}
    if not base["connected"]:
        return {**base, "readable": False, "rows": [], "summary": {},
                "note": "Connect Robinhood in step 1 to compare holdings."}
    acct, snap = await _read_rh_positions(db, user)
    base["account_number_masked"] = _mask(acct.account_number)
    positions = snap.get("positions")
    if positions is None:
        return {**base, "readable": False, "rows": [], "summary": {},
                "note": snap.get("note") or ("Robinhood's connection did not return holdings in a shape "
                                             "SwingTrader recognizes. See Connection details below."),
                "buying_power": snap.get("buying_power"), "cash": snap.get("cash")}
    rows = compare_holdings(positions, _st_positions(db, user))
    summary = {s: sum(1 for r in rows if r["status"] == s)
               for s in ("match", "shares_differ", "robinhood_only", "swingtrader_only")}
    summary["importable"] = sum(1 for r in rows if r["importable"])
    return {**base, "readable": True, "rows": rows, "summary": summary, "note": snap.get("note"),
            "buying_power": snap.get("buying_power"), "cash": snap.get("cash"),
            "fetched_at": _now().isoformat()}


async def import_holdings(db: Session, user: User, tickers: list[str]) -> dict:
    """Add the requested Robinhood-only holdings to the SwingTrader portfolio.

    Share counts and costs are RE-READ from Robinhood here — never taken from
    the request — and a ticker SwingTrader already tracks is never touched.
    Imported rows carry no stop, so the Playbook flags them as unstopped until
    the user sets one (they are never silently counted as zero risk).
    """
    wanted = []
    for t in tickers or []:
        k = _key(t)
        if k and k not in wanted:
            wanted.append(k)
    if not wanted:
        raise ValueError("Pick at least one holding to import.")
    if len(wanted) > MAX_IMPORT:
        raise ValueError(f"At most {MAX_IMPORT} holdings per import.")
    _acct, snap = await _read_rh_positions(db, user)
    if snap.get("positions") is None:
        raise mcp.BrokerError(snap.get("note") or "Robinhood holdings could not be read; nothing was imported.")
    rows = {r["ticker"]: r for r in compare_holdings(snap["positions"], _st_positions(db, user))}
    today = date.today()
    imported, skipped = [], []
    for k in wanted:
        r = rows.get(k)
        if r is None or r["status"] == "swingtrader_only":
            skipped.append({"ticker": k, "reason": "Not held in your Robinhood Agentic account."})
        elif r["status"] != "robinhood_only":
            skipped.append({"ticker": k, "reason": "Already in your SwingTrader portfolio — left unchanged."})
        elif not r["importable"]:
            skipped.append({"ticker": k, "reason": r["reason"] or "Missing shares or cost from Robinhood."})
        else:
            sh, cost = r["robinhood_shares"], r["robinhood_avg_cost"]
            db.add(PortfolioPosition(
                user_id=user.id, ticker=k, shares=sh, avg_cost=cost,
                notes=(f"Imported from the Robinhood Agentic account on {today.isoformat()} "
                       f"({sh:g} shares @ ${cost:,.2f}). No stop or plan was recorded — set a stop."),
            ))
            imported.append({"ticker": k, "shares": sh, "avg_cost": cost})
    if imported:
        db.commit()
        try:
            from services.today import invalidate_cache
            invalidate_cache(user.id)
        except Exception:  # noqa: BLE001
            pass
    return {"imported": imported, "skipped": skipped}


# ── Validation ─────────────────────────────────────────────────────────────

def _clean_plan(plan: Any) -> dict:
    if not isinstance(plan, dict):
        return {}
    return {k: plan.get(k) for k in PLAN_KEYS if plan.get(k) is not None}


def validate_orders(orders: list[dict], quotes: dict[str, float], held: dict[str, float],
                    buying_power: float | None, strict_quotes: bool,
                    stale: dict[str, str] | None = None,
                    holdings: dict[str, dict] | None = None,
                    holdings_unknown: bool = False) -> tuple[list[dict], dict]:
    """Pure validation. Returns (rows, totals). Never raises on bad input.

    `stale` ({TICKER: reason}) marks quotes too old to size a live order —
    an error when strict (live), a warning otherwise. `holdings` is the
    compare_holdings() result keyed by _key(ticker) (live only): sells of a
    `shares_differ`/`swingtrader_only` ticker are errors, and buys while any
    ticker is `shares_differ`/`robinhood_only` — or when `holdings_unknown` —
    carry `needs_ack` items the user must acknowledge before placing."""
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    batch_errors: list[str] = []
    stale = stale or {}
    if not orders:
        batch_errors.append("No orders selected.")
    if len(orders) > MAX_ORDERS:
        batch_errors.append(f"At most {MAX_ORDERS} orders per batch (got {len(orders)}).")
    mismatched = sorted(k for k, h in (holdings or {}).items() if h.get("status") in HOLDINGS_BUY_ACK)
    buy_ack = None
    if holdings_unknown:
        buy_ack = ("Robinhood holdings could not be read, so SwingTrader can't confirm its portfolio "
                   "matches your account — sizing and risk checks use SwingTrader's numbers.")
    elif mismatched:
        buy_ack = ("SwingTrader's portfolio doesn't match Robinhood for "
                   + ", ".join(mismatched[:8]) + (" …" if len(mismatched) > 8 else "")
                   + " — sizing and risk checks use SwingTrader's numbers. Reconcile in the "
                   "Holdings panel, or acknowledge to trade anyway.")

    for i, o in enumerate(orders or []):
        o = o if isinstance(o, dict) else {}
        errors: list[str] = []
        warnings: list[str] = []
        ticker = str(o.get("ticker") or "").upper().strip().replace("/", "-")
        if not _TICKER_RE.fullmatch(ticker):
            errors.append(f"Invalid ticker {o.get('ticker')!r}.")
        side = str(o.get("side") or "").lower()
        if side not in ("buy", "sell"):
            errors.append("Side must be buy or sell.")
        raw_shares = o.get("shares")
        shares = None
        try:
            f = float(raw_shares)
            if f != int(f):
                errors.append("Whole shares only.")
            shares = int(f)
            if not 1 <= shares <= MAX_SHARES:
                errors.append(f"Shares must be between 1 and {MAX_SHARES:,}.")
        except (TypeError, ValueError):
            errors.append("Shares must be a whole number.")
        limit = None
        try:
            limit = float(o.get("limit_price"))
            if not limit > 0 or limit != limit:
                errors.append("Limit price must be above $0.")
                limit = None
        except (TypeError, ValueError):
            errors.append("Limit price is required.")
        tif = str(o.get("time_in_force") or "gfd").lower()
        if tif not in ("gfd", "gtc"):
            errors.append("Time in force must be gfd (day) or gtc.")

        key = (ticker, side)
        if key in seen:
            errors.append(f"Duplicate {side} order for {ticker} in this batch.")
        seen.add(key)

        last = quotes.get(ticker)
        if limit is not None and ticker:
            if last:
                if abs(limit - last) / last > LIMIT_BAND:
                    errors.append(f"Limit ${limit:,.2f} is more than {int(LIMIT_BAND * 100)}% away from the "
                                  f"last price ${last:,.2f} — probably a typo.")
                elif side == "buy" and limit > last * 1.03:
                    warnings.append(f"Buy limit is {((limit / last) - 1) * 100:.1f}% above the last price — "
                                    "you may pay up to your limit.")
                elif side == "sell" and limit < last * 0.97:
                    warnings.append(f"Sell limit is {(1 - limit / last) * 100:.1f}% below the last price.")
            elif strict_quotes:
                errors.append(f"No current price for {ticker}, so the limit can't be sanity-checked.")
            else:
                warnings.append(f"No current price for {ticker}; limit not sanity-checked.")
        if last and ticker in stale:
            msg = (f"The price for {ticker} {stale[ticker]} — press Preview to refresh prices "
                   "before placing.")
            (errors if strict_quotes else warnings).append(msg)

        if side == "sell" and shares is not None and ticker:
            have = held.get(ticker, 0.0)
            if have <= 0:
                errors.append(f"You don't hold {ticker}.")
            elif shares > have + 1e-9:
                errors.append(f"Selling {shares} but only {have:g} held.")

        h = (holdings or {}).get(_key(ticker)) if ticker else None
        if side == "sell" and h and h.get("status") in HOLDINGS_SELL_BLOCK:
            if h["status"] == "shares_differ":
                errors.append(f"Robinhood holds {h.get('robinhood_shares') or 0:g} {ticker} but SwingTrader "
                              f"records {h.get('swingtrader_shares') or 0:g} — reconcile in the Holdings "
                              "panel before selling.")
            else:
                errors.append(f"Your Robinhood Agentic account holds no {ticker} (SwingTrader records "
                              f"{h.get('swingtrader_shares') or 0:g}) — it may be held at another broker; "
                              "sell it there.")
        needs_ack = []
        if buy_ack and (side == "buy" or holdings_unknown):
            needs_ack.append(buy_ack)

        est = round(shares * limit, 2) if (shares and limit) else None
        rows.append({
            "index": i, "plan_item_id": o.get("plan_item_id"), "ticker": ticker, "side": side,
            "shares": shares, "limit_price": limit, "time_in_force": tif, "last_price": last,
            "est_value": est, "ok": not errors, "errors": errors, "warnings": warnings,
            "needs_ack": needs_ack, "plan": _clean_plan(o.get("plan")),
        })

    buy_total = round(sum(r["est_value"] or 0 for r in rows if r["ok"] and r["side"] == "buy"), 2)
    sell_total = round(sum(r["est_value"] or 0 for r in rows if r["ok"] and r["side"] == "sell"), 2)
    batch_warnings: list[str] = []
    if buying_power is not None and buy_total > buying_power + 0.01:
        msg = (f"Buys total ${buy_total:,.2f} but buying power is ${buying_power:,.2f}. "
               "Deselect some buys or reduce shares.")
        if strict_quotes:
            batch_errors.append(msg)
            for r in rows:
                if r["side"] == "buy" and r["ok"]:
                    r["ok"] = False
                    r["errors"].append("Exceeds buying power as a batch.")
        else:
            # Paper buying power is DERIVED (Settings account size minus the
            # cost of what the app thinks you hold), and a book imported from
            # another broker routinely exceeds a default account size. Blocking
            # on it would make paper mode — the recommended first step —
            # unusable, so it is a warning here; live mode uses Robinhood's real
            # figure and blocks.
            batch_warnings.append(msg + " (Paper mode: shown as a warning — paper buying power "
                                  "is your Settings account size minus the cost of your holdings.)")
    totals = {
        "buy_total": buy_total, "sell_total": sell_total, "buying_power": buying_power,
        "buying_power_known": buying_power is not None,
        "errors": batch_errors, "warnings": batch_warnings,
    }
    _recount(totals, rows)
    return rows, totals


def _recount(totals: dict, rows: list[dict]) -> dict:
    """Refresh the row-derived totals after rows were changed in place."""
    totals["ok_count"] = sum(1 for r in rows if r["ok"])
    totals["error_count"] = sum(1 for r in rows if not r["ok"])
    totals["ok"] = not totals["errors"] and all(r["ok"] for r in rows)
    items: list[str] = []
    for r in rows:
        if r["ok"]:
            for m in r.get("needs_ack") or []:
                if m not in items:
                    items.append(m)
    totals["needs_ack"] = bool(items)
    totals["ack_items"] = items
    return totals


def _dedupe_key(user_id: int, r: dict) -> str:
    return (f"{user_id}|{r['side']}|{r['ticker']}|{r['shares']}|{float(r['limit_price']):.4f}"
            f"|{r.get('plan_item_id') or ''}")[:200]


def duplicate_blocks(db: Session, user: User, rows: list[dict]) -> None:
    """Refuse live orders that could duplicate one already sent (gap 8).

    Per ok row, against this user's LIVE order log for the same ticker+side:
      * an unresolved (`sending`/`unknown`) order  -> blocked until resolved
      * any still-open order                        -> blocked (cancel it first)
      * an identical order (same dedupe key) filled today -> blocked
    A terminal failure (failed/rejected/cancelled/expired) never blocks."""
    tickers = sorted({r["ticker"] for r in rows if r["ok"] and r.get("ticker")})
    if not tickers:
        return
    prior = (db.query(BrokerOrder)
             .filter(BrokerOrder.user_id == user.id, BrokerOrder.mode == "live",
                     BrokerOrder.ticker.in_(tickers))
             .order_by(BrokerOrder.id.desc()).all())
    today = _now().date()
    for r in rows:
        if not r["ok"]:
            continue
        key = _dedupe_key(user.id, r)
        for o in prior:
            if o.ticker != r["ticker"] or o.side != r["side"]:
                continue
            msg = None
            if o.status in UNRESOLVED:
                msg = (f"An earlier {o.side} order for {o.ticker} (#{o.id}) has an unknown outcome — it may "
                       "have reached Robinhood. Check the Robinhood app, then press Check fills or use "
                       "Mark filled / Mark cancelled before placing again.")
            elif o.status not in TERMINAL:
                msg = (f"You already have an open {o.side} order for {o.ticker} (#{o.id}, {o.status}). "
                       "Cancel it before placing another.")
            elif (o.status == "filled" and o.dedupe_key == key and o.created_at is not None
                  and o.created_at.date() == today):
                msg = (f"An identical {o.side} order for {o.ticker} (#{o.id}) already filled today — "
                       "not placing it twice.")
            if msg:
                r["ok"] = False
                r["errors"].append(msg)
                break


async def _context(db: Session, user: User, orders: list[dict], refresh_stale: bool = False) -> dict:
    """Everything validate_orders needs: acct, mode, quotes, stale, held,
    buying power, the holdings comparison (live) and notes. `refresh_stale`
    (preview only) re-fetches quotes too old for a live order."""
    acct = get_account(db, user.id)
    mode = (acct.mode if acct else "paper") or "paper"
    live = mode == "live" and is_connected(acct)
    tickers = sorted({str((o or {}).get("ticker") or "").upper().strip().replace("/", "-")
                      for o in orders if isinstance(o, dict)} - {""})
    quotes, as_of = _split_quotes(await QUOTE_SOURCE(tickers, db) if tickers else {})
    notes: list[str] = []
    stale: dict[str, str] = {}
    if live:
        stale = {t: why for t in quotes if (why := quote_staleness(as_of.get(t)))}
        if stale and refresh_stale:
            fresh, fresh_as_of = _split_quotes(await QUOTE_SOURCE(sorted(stale), db, refresh=True))
            quotes.update(fresh)
            as_of.update(fresh_as_of)
            stale = {t: why for t in quotes if (why := quote_staleness(as_of.get(t)))}
            refreshed = [t for t in fresh if t not in stale]
            if refreshed:
                notes.append("Refreshed prices for " + ", ".join(refreshed) + ".")
    holdings: dict[str, dict] | None = None
    holdings_unknown = False
    if live:
        snap = await _live_account(db, acct)
        bp = snap.get("buying_power")
        if snap.get("positions") is not None:
            held = {p["ticker"].replace(".", "-"): p["shares"] for p in snap["positions"]}
            holdings = {r["ticker"]: r for r in compare_holdings(snap["positions"], _st_positions(db, user))}
        else:
            paper = _paper_account(db, user)
            held = {p["ticker"]: p["shares"] for p in paper["positions"]}
            holdings_unknown = True
            notes.append("Robinhood positions could not be read; sells were checked against your "
                         "SwingTrader portfolio instead, and every order needs your acknowledgement.")
        if bp is None:
            notes.append("Robinhood buying power could not be read; Robinhood will check it.")
    else:
        paper = _paper_account(db, user)
        held = {p["ticker"]: p["shares"] for p in paper["positions"]}
        bp = paper["buying_power"]
        notes.append("Paper mode: holdings are compared with Robinhood and price freshness is enforced "
                     "only for live orders (paper makes no Robinhood calls).")
    return {"acct": acct, "mode": mode, "live": live, "quotes": quotes, "stale": stale, "held": held,
            "bp": bp, "holdings": holdings, "holdings_unknown": holdings_unknown, "notes": notes}


def _validate(orders: list[dict], ctx: dict, db: Session | None = None,
              user: User | None = None) -> tuple[list[dict], dict]:
    rows, totals = validate_orders(orders, ctx["quotes"], ctx["held"], ctx["bp"],
                                   strict_quotes=(ctx["mode"] == "live"), stale=ctx["stale"],
                                   holdings=ctx["holdings"], holdings_unknown=ctx["holdings_unknown"])
    if db is not None and user is not None:
        # Account-level circuit breaker (gap 7): a HALT refuses live BUYs
        # (paper buys are flagged); sells are never touched.
        state = circuit_breaker.current_state(db, user, record=False)
        circuit_breaker.gate_orders(rows, state, live=ctx["live"])
        totals["circuit_breaker"] = {"level": state.get("level"),
                                     "reasons": [r.get("message") for r in state.get("reasons") or []]}
        _recount(totals, rows)
    return rows, totals


async def preview(db: Session, user: User, orders: list[dict]) -> dict:
    async with _lock(user.id):
        ctx = await _context(db, user, orders, refresh_stale=True)
        acct, mode, notes = ctx["acct"], ctx["mode"], ctx["notes"]
        rows, totals = _validate(orders, ctx, db, user)
        if ctx["live"]:
            duplicate_blocks(db, user, rows)
            _recount(totals, rows)
            account_number = acct.account_number

            def fn(s: mcp.McpSession, tools: list[dict]):
                review = mcp.find_tool(tools, "review")
                if not review:
                    for r in rows:
                        if r["ok"]:
                            r["warnings"].append("Robinhood exposes no order-review tool; "
                                                 "the order will be sent without a broker preview.")
                    return
                for r in rows:
                    if not r["ok"]:
                        continue
                    try:
                        res = s.call_tool(review["name"],
                                          mcp.map_arguments(review, _order_fields(r, account_number)))
                    except mcp.MappingError as exc:
                        r["ok"] = False
                        r["errors"].append(str(exc))
                        continue
                    r["broker_review"] = res["text"]
                    if not res["ok"]:
                        r["ok"] = False
                        r["errors"].append("Robinhood review: " + (res["text"] or "rejected"))

            await _live(db, acct, fn)
            _recount(totals, rows)
    return {"mode": mode, "orders": rows, "totals": totals, "notes": notes}


# ── Placement ──────────────────────────────────────────────────────────────

def _new_order(user: User, mode: str, r: dict, status_: str) -> BrokerOrder:
    return BrokerOrder(
        user_id=user.id, broker="robinhood", mode=mode, plan_item_id=r.get("plan_item_id"),
        ticker=r["ticker"], side=r["side"], order_type="limit", quantity=r["shares"],
        limit_price=r["limit_price"], time_in_force=r["time_in_force"], status=status_,
        plan_json=json.dumps(r.get("plan") or {}),
    )


async def place(db: Session, user: User, orders: list[dict], confirm: bool,
                acknowledge: bool = False) -> dict:
    async with _lock(user.id):
        ctx = await _context(db, user, orders)
        acct, mode, notes = ctx["acct"], ctx["mode"], ctx["notes"]
        if mode == "live":
            if not is_connected(acct):
                raise ValueError("Live mode, but Robinhood is not connected. Reconnect or switch to paper.")
            if not confirm:
                raise ValueError("Live orders need confirm: true.")
        rows, totals = _validate(orders, ctx, db, user)
        if mode == "live":
            duplicate_blocks(db, user, rows)
            _recount(totals, rows)
        if totals["errors"] and not any(r["ok"] for r in rows):
            return {"mode": mode, "results": rows, "totals": totals, "notes": notes, "placed": 0}

        if mode != "live":
            for r in rows:
                if r["ok"]:
                    o = _new_order(user, "paper", r, "simulated")
                    o.filled_quantity = None
                    db.add(o)
                    db.flush()
                    r["order_id"] = o.id
                    r["status"] = "simulated"
                    r["message"] = "Simulated — nothing was sent to Robinhood and your portfolio is unchanged."
            db.commit()
            return {"mode": mode, "results": rows, "totals": totals, "notes": notes,
                    "placed": sum(1 for r in rows if r.get("order_id"))}

        if totals["needs_ack"] and not acknowledge:
            totals["ok"] = False
            totals["errors"] = totals["errors"] + [
                "Nothing was placed: acknowledge the holdings warning(s) first — tick "
                "\u201cI understand\u201d and place again."]
            return {"mode": mode, "results": rows, "totals": totals, "notes": notes, "placed": 0,
                    "needs_ack": True}

        # Write every order row BEFORE any network call, with its client id.
        # If the process dies mid-call the row stays `sending`, which blocks a
        # re-placement exactly like `unknown` does.
        by_index: dict[int, BrokerOrder] = {}
        for r in rows:
            if not r["ok"]:
                continue
            o = _new_order(user, "live", r, "sending")
            o.client_order_id = str(uuid.uuid4())
            o.dedupe_key = _dedupe_key(user.id, r)
            db.add(o)
            db.flush()
            r["order_id"] = o.id
            r["client_order_id"] = o.client_order_id
            by_index[r["index"]] = o
        db.commit()

        account_number = acct.account_number
        outcomes: dict[int, dict] = {}
        inflight: dict = {"index": None}

        def fn(s: mcp.McpSession, tools: list[dict]):
            review = mcp.find_tool(tools, "review")
            placer = mcp.find_tool(tools, "place")
            for r in rows:
                if r["index"] not in by_index:
                    continue
                i = r["index"]
                if not placer:
                    outcomes[i] = {"error": "Robinhood exposes no order-placement tool."}
                    continue
                fields = _order_fields(r, account_number)
                rev_text = None
                try:
                    if review:
                        rev = s.call_tool(review["name"], mcp.map_arguments(review, fields))
                        rev_text = rev["text"]
                        if not rev["ok"]:
                            outcomes[i] = {"error": "Robinhood review rejected the order: "
                                           + (rev["text"] or "no reason given"), "review": rev_text}
                            continue
                    place_args = mcp.map_arguments(placer, dict(fields, client_order_id=r["client_order_id"]))
                except mcp.ReauthRequired:
                    raise
                except mcp.BrokerError as exc:   # incl. MappingError; a review never places anything
                    outcomes[i] = {"error": str(exc), "review": rev_text}
                    continue
                inflight["index"] = i
                try:
                    res = s.call_tool(placer["name"], place_args)
                except mcp.OutcomeUnknown as exc:
                    inflight["index"] = None
                    outcomes[i] = {"unknown": str(exc), "review": rev_text}
                    continue
                except mcp.ReauthRequired:
                    # 401: Robinhood rejected the call — the order was not accepted.
                    inflight["index"] = None
                    raise
                except mcp.BrokerError as exc:
                    # Rejected with an HTTP error before being acted on.
                    inflight["index"] = None
                    outcomes[i] = {"error": str(exc), "review": rev_text}
                    continue
                inflight["index"] = None
                outcomes[i] = {"review": rev_text, "text": res["text"], "ok": res["ok"],
                               "parsed": mcp.parse_order(res["data"])}
            return None

        fatal: Exception | None = None
        try:
            await _live(db, acct, fn)
        except Exception as exc:  # noqa: BLE001 — record every row's fate, then re-raise
            fatal = exc
        for r in rows:
            o = by_index.get(r["index"])
            if o is None:
                continue
            out = outcomes.get(r["index"])
            if out is None:
                if fatal is not None and inflight["index"] == r["index"]:
                    out = {"unknown": f"The connection failed while this order was being sent ({fatal})."}
                else:
                    out = {"error": "Not sent: " + (str(fatal) if fatal else "no outcome recorded.")}
            if out.get("unknown"):
                o.status = "unknown"
                o.error = (f"Robinhood did not confirm this order ({out['unknown']}). It MAY have been "
                           "placed. Check the Robinhood app, then press Check fills or use Mark filled / "
                           f"Mark cancelled. SwingTrader won't place another {o.side} for {o.ticker} "
                           "until then.")[:1000]
            elif out.get("error") or not out.get("ok"):
                o.error = (out.get("error") or ("Robinhood: " + (out.get("text") or "order rejected")))[:1000]
                o.status = "failed"
            else:
                parsed = out["parsed"]
                o.broker_order_id = parsed["broker_order_id"]
                text = (out.get("text") or "").lower()
                o.status = parsed["status"] or ("pending_approval" if "approv" in text else "submitted")
                o.filled_quantity = parsed["filled_quantity"]
                o.avg_fill_price = parsed["avg_fill_price"]
            o.response_json = json.dumps({"review": out.get("review"), "result": out.get("text")})[:4000]
            r["status"] = o.status
            r["message"] = o.error or out.get("text") or ""
            r["ok"] = o.status != "failed"
            if o.status in TERMINAL and (o.filled_quantity or 0) > 0:
                apply_fill(db, o, commit=False)
        db.commit()
        if fatal is not None:
            raise fatal
    return {"mode": "live", "results": rows, "totals": totals, "notes": notes,
            "placed": sum(1 for r in rows if r.get("status") not in (None, "failed")),
            "unknown": sum(1 for r in rows if r.get("status") == "unknown")}


# ── Order log, sync, cancel, manual resolve ───────────────────────────────

def order_to_dict(o: BrokerOrder) -> dict:
    try:
        resp = json.loads(o.response_json) if o.response_json else None
    except ValueError:
        resp = None
    return {
        "id": o.id, "mode": o.mode, "plan_item_id": o.plan_item_id, "ticker": o.ticker,
        "side": o.side, "order_type": o.order_type, "quantity": o.quantity,
        "limit_price": o.limit_price, "time_in_force": o.time_in_force, "status": o.status,
        "open": o.status not in TERMINAL, "broker_order_id": o.broker_order_id,
        "outcome_unknown": o.status in UNRESOLVED, "client_order_id": o.client_order_id,
        "filled_quantity": o.filled_quantity, "avg_fill_price": o.avg_fill_price,
        "error": o.error, "broker_messages": resp, "applied_to_portfolio": bool(o.applied_to_portfolio),
        "created_at": o.created_at.isoformat() if o.created_at else None,
        "updated_at": o.updated_at.isoformat() if o.updated_at else None,
    }


def list_orders(db: Session, user: User, limit: int = 50) -> list[dict]:
    rows = (db.query(BrokerOrder).filter(BrokerOrder.user_id == user.id)
            .order_by(BrokerOrder.created_at.desc(), BrokerOrder.id.desc()).limit(limit).all())
    return [order_to_dict(o) for o in rows]


def apply_fill(db: Session, order: BrokerOrder, commit: bool = True) -> str | None:
    """Write a filled live order to the portfolio ONCE. Returns a note."""
    if order.applied_to_portfolio or order.mode != "live":
        return None
    qty = float(order.filled_quantity or 0)
    if qty <= 0:
        return None
    px = float(order.avg_fill_price or order.limit_price)
    try:
        plan = json.loads(order.plan_json) if order.plan_json else {}
    except ValueError:
        plan = {}
    pos = (db.query(PortfolioPosition)
           .filter(PortfolioPosition.user_id == order.user_id, PortfolioPosition.ticker == order.ticker)
           .first())
    note = None
    if order.side == "buy":
        stop = plan.get("stop")
        if pos:
            total = pos.shares + qty
            pos.avg_cost = (pos.shares * pos.avg_cost + qty * px) / total
            pos.shares = total
            if pos.stop_loss is None and stop is not None:
                pos.stop_loss = stop
            if pos.initial_stop is None and stop is not None:
                pos.initial_stop = stop
            if pos.target is None and plan.get("target") is not None:
                pos.target = plan.get("target")
        else:
            thesis = plan.get("thesis")
            db.add(PortfolioPosition(
                user_id=order.user_id, ticker=order.ticker, shares=qty, avg_cost=px,
                stop_loss=stop, initial_stop=stop, target=plan.get("target"),
                entry_date=date.today(), strategy=plan.get("strategy"),
                planned_entry=plan.get("planned_entry"), planned_entry_high=plan.get("planned_entry_high"),
                thesis=thesis, invalidation=plan.get("invalidation"),
                time_stop_days=plan.get("time_stop_days"),
                notes=f"Opened by Robinhood order (SwingTrader #{order.id}) at ${px:,.2f}."
                      + (f"\nTHESIS: {thesis}" if thesis else ""),
            ))
        note = f"Added {qty:g} {order.ticker} to your portfolio at ${px:,.2f}."
    else:
        if not pos:
            note = f"Sold {qty:g} {order.ticker}, but SwingTrader had no position to reduce."
            order.error = note
        elif qty >= pos.shares - 1e-9:
            order.applied_to_portfolio = True   # journal_close commits
            journal_close(db, pos, px, date.today(),
                          f"Closed by Robinhood order (SwingTrader #{order.id}) at ${px:,.2f}.")
            note = f"Closed {order.ticker} and recorded it in your journal."
        else:
            pos.shares = pos.shares - qty
            note = f"Reduced {order.ticker} by {qty:g} shares (partial sells are not journaled)."
    order.applied_to_portfolio = True
    if commit:
        db.commit()
    try:
        from services.today import invalidate_cache
        invalidate_cache(order.user_id)
    except Exception:  # noqa: BLE001
        pass
    return note


async def sync(db: Session, user: User) -> dict:
    acct = get_account(db, user.id)
    open_orders = (db.query(BrokerOrder)
                   .filter(BrokerOrder.user_id == user.id, BrokerOrder.mode == "live",
                           BrokerOrder.status.notin_(list(TERMINAL))).all())
    notes: list[str] = []
    # Terminal-but-unapplied fills (e.g. an earlier sync crashed mid-way).
    for o in (db.query(BrokerOrder)
              .filter(BrokerOrder.user_id == user.id, BrokerOrder.mode == "live",
                      BrokerOrder.applied_to_portfolio.is_(False),
                      BrokerOrder.status.in_(["filled", "cancelled", "canceled", "rejected", "expired"]))
              .all()):
        n = apply_fill(db, o)
        if n:
            notes.append(n)
    if not open_orders:
        return {"checked": 0, "updated": 0, "notes": notes}
    if not is_connected(acct):
        return {"checked": 0, "updated": 0,
                "notes": notes + ["Robinhood is not connected — reconnect to check fills."]}

    snapshot = [(o.id, o.broker_order_id, o.client_order_id) for o in open_orders]

    def fn(s: mcp.McpSession, tools: list[dict]):
        t = mcp.find_tool(tools, "order")
        if not t:
            return None
        out = {}
        for oid, bid, cid in snapshot:
            if not bid:
                if cid:
                    out[oid] = _reconcile_by_client_id(s, t, cid, acct.account_number)
                continue
            try:
                r = s.call_tool(t["name"], mcp.map_arguments(t, {"order_id": bid,
                                                               "account_number": acct.account_number}))
            except mcp.MappingError as exc:
                out[oid] = {"error": str(exc)}
                continue
            data = r["data"]
            # A history tool returns many orders; pick ours.
            if isinstance(data, (dict, list)):
                match = _find_order(data, bid)
                if match is not None:
                    data = match
            out[oid] = mcp.parse_order(data)
        return out

    async with _lock(user.id):
        results = await _live(db, acct, fn)
    if results is None:
        return {"checked": 0, "updated": 0, "notes": notes + [
            "Robinhood exposes no order-status tool. Check the order in the Robinhood app, then use "
            "'Mark filled' / 'Mark cancelled' here so your portfolio stays in step."]}
    updated = 0
    by_id = {o.id: o for o in open_orders}
    for oid, parsed in results.items():
        o = by_id[oid]
        if parsed.get("unresolved"):
            notes.append(f"{o.ticker} #{o.id}: outcome still unknown — {parsed['unresolved']} Check the "
                         "Robinhood app, then use Mark filled / Mark cancelled.")
            continue
        if parsed.get("error"):
            notes.append(f"{o.ticker}: {parsed['error']}")
            continue
        changed = False
        if not o.broker_order_id and parsed.get("broker_order_id"):
            o.broker_order_id = parsed["broker_order_id"]
            changed = True
        if o.status in UNRESOLVED and not parsed.get("status"):
            parsed["status"] = "submitted"
        if parsed.get("status") and parsed["status"] != o.status:
            o.status = parsed["status"]
            changed = True
        if parsed.get("filled_quantity") is not None:
            o.filled_quantity = parsed["filled_quantity"]
        if parsed.get("avg_fill_price") is not None:
            o.avg_fill_price = parsed["avg_fill_price"]
        updated += int(changed)
        if o.status in TERMINAL and (o.filled_quantity or 0) > 0:
            n = apply_fill(db, o, commit=False)
            if n:
                notes.append(n)
    db.commit()
    return {"checked": len(results), "updated": updated, "notes": notes}


_CLIENT_ID_KEYS = ("client_order_id", "ref_id", "idempotency_key", "client_id",
                   "client_ref_id", "external_id", "request_id")


def _find_by_client_id(data: Any, cid: str, depth: int = 0):
    """The order record carrying our client id, anywhere in a tool result."""
    if depth > 6:
        return None
    if isinstance(data, dict):
        if any(str(data.get(k) or "") == str(cid) for k in _CLIENT_ID_KEYS):
            return data
        for v in data.values():
            m = _find_by_client_id(v, cid, depth + 1)
            if m is not None:
                return m
    elif isinstance(data, list):
        for v in data:
            m = _find_by_client_id(v, cid, depth + 1)
            if m is not None:
                return m
    return None


def _reconcile_by_client_id(s: mcp.McpSession, tool: dict, cid: str, account_number: str | None) -> dict:
    """Look an `unknown` order up by our client id. Only possible when the
    order tool can LIST orders (no required order id). Not finding it is NOT
    proof it doesn't exist (the list may be paged), so it stays unresolved."""
    try:
        args = mcp.map_arguments(tool, {"account_number": account_number})
    except mcp.MappingError:
        return {"unresolved": "Robinhood's order tool needs an order id, and none came back for this order."}
    try:
        r = s.call_tool(tool["name"], args)
    except mcp.ReauthRequired:
        raise
    except mcp.BrokerError as exc:
        return {"unresolved": f"the order lookup failed ({exc})."}
    match = _find_by_client_id(r["data"], cid)
    if match is None:
        return {"unresolved": "it isn't in the orders Robinhood returned."}
    return mcp.parse_order(match)


def _find_order(data: Any, bid: str, depth: int = 0):
    if depth > 5:
        return None
    if isinstance(data, dict):
        if str(data.get("id") or data.get("order_id") or "") == str(bid):
            return data
        for v in data.values():
            m = _find_order(v, bid, depth + 1)
            if m is not None:
                return m
    elif isinstance(data, list):
        for v in data:
            m = _find_order(v, bid, depth + 1)
            if m is not None:
                return m
    return None


def _user_order(db: Session, user: User, order_id: int) -> BrokerOrder:
    o = db.query(BrokerOrder).filter(BrokerOrder.id == order_id, BrokerOrder.user_id == user.id).first()
    if not o:
        raise LookupError("Order not found.")
    return o


async def cancel(db: Session, user: User, order_id: int) -> dict:
    o = _user_order(db, user, order_id)
    if o.status in TERMINAL:
        raise ValueError(f"Order is already {o.status}.")
    if o.status in UNRESOLVED and not o.broker_order_id:
        raise ValueError("This order's outcome is unknown, so SwingTrader can't cancel it for you. Check the "
                         "Robinhood app: cancel it there if it exists, then use Mark cancelled (or Mark "
                         "filled) here.")
    acct = get_account(db, user.id)
    if o.mode != "live" or not o.broker_order_id:
        o.status = "cancelled"
        db.commit()
        return order_to_dict(o)

    bid = o.broker_order_id

    def fn(s: mcp.McpSession, tools: list[dict]):
        t = mcp.find_tool(tools, "cancel")
        if not t:
            return None
        return s.call_tool(t["name"], mcp.map_arguments(t, {"order_id": bid,
                                                           "account_number": acct.account_number}))

    async with _lock(user.id):
        res = await _live(db, acct, fn)
    if res is None:
        raise ValueError("Robinhood exposes no cancel tool here. Cancel the order in the Robinhood app, "
                         "then use 'Mark cancelled'.")
    if not res["ok"]:
        raise ValueError("Robinhood did not cancel the order: " + (res["text"] or "no reason given"))
    o.status = "cancelled"
    o.response_json = json.dumps({"cancel": res["text"]})[:4000]
    db.commit()
    return order_to_dict(o)


def resolve_manually(db: Session, user: User, order_id: int, outcome: str,
                     fill_price: float | None, filled_quantity: float | None) -> dict:
    """For when Robinhood offers no status tool: the user reports the outcome."""
    o = _user_order(db, user, order_id)
    if o.mode != "live":
        raise ValueError("Only live orders can be resolved.")
    if o.status in TERMINAL:
        raise ValueError(f"Order is already {o.status}.")
    if outcome == "cancelled":
        o.status = "cancelled"
        if filled_quantity:
            o.filled_quantity = float(filled_quantity)
            o.avg_fill_price = float(fill_price or o.limit_price)
    elif outcome == "filled":
        qty = float(filled_quantity or o.quantity)
        if not 0 < qty <= o.quantity:
            raise ValueError("Filled quantity must be between 0 and the order quantity.")
        px = float(fill_price or o.limit_price)
        if px <= 0:
            raise ValueError("Fill price must be above $0.")
        o.status = "filled"
        o.filled_quantity = qty
        o.avg_fill_price = px
    else:
        raise ValueError("outcome must be 'filled' or 'cancelled'.")
    note = apply_fill(db, o, commit=False) if (o.filled_quantity or 0) > 0 else None
    db.commit()
    d = order_to_dict(o)
    d["note"] = note
    return d
