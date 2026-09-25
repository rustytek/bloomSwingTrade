"""
Trade page / Robinhood broker tests — no network.

Robinhood's agentic-trading MCP server and its OAuth endpoints are simulated
with httpx.MockTransport (services/brokers/robinhood_mcp.TRANSPORT). Nothing
here can reach the internet: an unexpected URL fails the test.

Covers secrets_box, OAuth discovery / dynamic registration / PKCE / callback
state handling, MCP session (initialize, SSE, 401 -> refresh), tool-schema
argument mapping, review -> place sequencing, paper-mode isolation, order
validation, fill sync -> portfolio exactly once, manual resolve, secret-free
API responses, and that static/trade.html compiles next to common.js.

    python test_broker.py
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.parse import parse_qs, urlparse

_TMP = tempfile.mkdtemp(prefix="swingtrader-broker-")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_TMP, "t.db").replace("\\", "/")
os.environ.pop("BROKER_ENCRYPTION_KEY", None)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import httpx  # noqa: E402

from database.db import Base, SessionLocal, engine  # noqa: E402
from database.models import (BrokerAccount, BrokerOrder, ClosedTrade,  # noqa: E402
                             PortfolioPosition, User)
from services import broker_service as svc  # noqa: E402
from services import secrets_box  # noqa: E402
from services.brokers import robinhood_mcp as mcp  # noqa: E402

Base.metadata.create_all(bind=engine)

_TESTS = []


def test(fn):
    _TESTS.append(fn)
    return fn


REDIRECT = "https://invest.example.com/api/broker/oauth/callback"
ORDER_SCHEMA = {
    "type": "object",
    "properties": {
        "symbol": {"type": "string"},
        "side": {"type": "string", "enum": ["BUY", "SELL"]},
        "quantity": {"type": "number"},
        "order_type": {"type": "string", "enum": ["market", "limit"]},
        "limit_price": {"type": "number"},
        "time_in_force": {"type": "string", "enum": ["day", "gtc"]},
        "account_number": {"type": "string"},
    },
    "required": ["symbol", "side", "quantity", "order_type", "limit_price", "account_number"],
}


class FakeRobinhood:
    """In-memory stand-in for Robinhood's OAuth + MCP endpoints."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.tool_calls: list[tuple[str, dict]] = []
        self.registrations: list[dict] = []
        self.token_requests: list[dict] = []
        self.valid_tokens = set()
        self.challenge = None          # captured from the authorize URL
        self.allow_refresh = True
        self.reject_registration = False
        self.review_ok = True
        self.order_state = {"state": "queued", "cumulative_quantity": "0", "average_price": None}
        self.use_sse = False
        self.n = 0
        self.positions = [{"symbol": "MSFT", "quantity": "10", "average_buy_price": "400"}]
        self.tools = [
            {"name": "review_equity_order", "description": "Preview an order", "inputSchema": ORDER_SCHEMA},
            {"name": "place_equity_order", "description": "Place an order", "inputSchema": ORDER_SCHEMA},
            {"name": "get_accounts", "description": "Accounts", "inputSchema": {"type": "object", "properties": {}}},
            {"name": "get_positions", "description": "Positions", "inputSchema": {
                "type": "object", "properties": {"account_number": {"type": "string"}}}},
            {"name": "get_order", "description": "One order", "inputSchema": {
                "type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
        ]

    def _issue(self):
        self.n += 1
        tok = f"AT-secret-{self.n}"
        self.valid_tokens = {tok}
        return {"access_token": tok, "refresh_token": f"RT-secret-{self.n}", "expires_in": 3600,
                "token_type": "Bearer"}

    def tool(self, name, args):
        self.tool_calls.append((name, args))
        if name == "get_accounts":
            return {"structuredContent": {"accounts": [
                {"account_number": "111222", "type": "individual", "buying_power": "0"},
                {"account_number": "999888", "type": "agentic", "buying_power": "5000.00"}]},
                "content": [{"type": "text", "text": "2 accounts"}]}
        if name == "get_positions":
            return {"structuredContent": {"positions": self.positions},
                    "content": [{"type": "text", "text": "positions"}]}
        if name == "review_equity_order":
            if not self.review_ok:
                return {"isError": True, "content": [{"type": "text", "text": "Insufficient buying power."}]}
            return {"content": [{"type": "text", "text": "Estimated cost $1,000. Market is open."}]}
        if name == "place_equity_order":
            return {"structuredContent": {"order": {"id": "rh-ord-1", "state": "queued"}},
                    "content": [{"type": "text", "text": "Order submitted. Approve it in the Robinhood app."}]}
        if name == "get_order":
            return {"structuredContent": dict(self.order_state, id=args.get("order_id"))}
        return {"isError": True, "content": [{"type": "text", "text": "unknown tool"}]}

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.calls.append((request.method, url))
        if url == mcp.RESOURCE_METADATA_URL:
            return httpx.Response(200, json={"authorization_servers": ["https://agent.robinhood.com/mcp/trading"],
                                             "resource": mcp.MCP_URL, "scopes_supported": ["internal"]})
        if url == mcp.AS_METADATA_URL:
            return httpx.Response(200, json={
                "issuer": "https://agent.robinhood.com/mcp/trading",
                "authorization_endpoint": "https://robinhood.com/oauth",
                "token_endpoint": "https://api.robinhood.com/oauth2/token/",
                "registration_endpoint": "https://agent.robinhood.com/oauth/trading/register"})
        if url == "https://agent.robinhood.com/oauth/trading/register":
            body = json.loads(request.content)
            self.registrations.append(body)
            if self.reject_registration:
                return httpx.Response(400, json={"error": "invalid_redirect_uri",
                                                 "error_description": "redirect host not allowed"})
            return httpx.Response(201, json={"client_id": f"cid-{len(self.registrations)}"})
        if url == "https://api.robinhood.com/oauth2/token/":
            form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
            self.token_requests.append(form)
            if form.get("grant_type") == "authorization_code":
                digest = base64.urlsafe_b64encode(
                    hashlib.sha256(form["code_verifier"].encode()).digest()).decode().rstrip("=")
                if form.get("code") != "good-code" or digest != self.challenge:
                    return httpx.Response(400, json={"error": "invalid_grant"})
                return httpx.Response(200, json=self._issue())
            if form.get("grant_type") == "refresh_token" and self.allow_refresh:
                return httpx.Response(200, json=self._issue())
            return httpx.Response(400, json={"error": "invalid_grant"})
        if url == mcp.MCP_URL:
            auth = request.headers.get("authorization", "")
            if auth.replace("Bearer ", "") not in self.valid_tokens:
                return httpx.Response(401)
            msg = json.loads(request.content)
            if "id" not in msg:
                return httpx.Response(202)
            if msg["method"] == "initialize":
                result = {"protocolVersion": mcp.PROTOCOL_VERSION, "capabilities": {"tools": {}},
                          "serverInfo": {"name": "fake-rh"}}
                return self._reply(msg["id"], result, {"mcp-session-id": "sess-1"})
            if msg["method"] == "tools/list":
                return self._reply(msg["id"], {"tools": self.tools})
            if msg["method"] == "tools/call":
                p = msg["params"]
                return self._reply(msg["id"], self.tool(p["name"], p["arguments"]))
        raise AssertionError("unexpected request " + request.method + " " + url)

    def _reply(self, rid, result, headers=None):
        payload = {"jsonrpc": "2.0", "id": rid, "result": result}
        if self.use_sse:
            text = "event: message\ndata: " + json.dumps(payload) + "\n\n"
            return httpx.Response(200, text=text, headers={"content-type": "text/event-stream", **(headers or {})})
        return httpx.Response(200, json=payload, headers=headers or {})


def _install(fake: FakeRobinhood | None):
    mcp._meta_cache.clear()
    if fake is None:
        def boom(request):
            raise AssertionError("network used: " + str(request.url))
        mcp.TRANSPORT = httpx.MockTransport(boom)
    else:
        mcp.TRANSPORT = httpx.MockTransport(fake.handler)


async def _quotes(tickers, db):
    table = {"AAPL": 100.0, "MSFT": 410.0, "NVDA": 120.0}
    return {t: table[t] for t in tickers if t in table}

svc.QUOTE_SOURCE = _quotes

_uid = [100]


def _user(db, **kw):
    _uid[0] += 1
    u = User(id=_uid[0], username=f"u{_uid[0]}", password_hash="x", account_size=10000.0, **kw)
    db.add(u)
    db.commit()
    return u


def _run(coro):
    return asyncio.run(coro)


async def _connect(db, user, fake):
    r = await svc.begin_connect(db, user, REDIRECT)
    assert r["status"] == "redirect", r
    q = {k: v[0] for k, v in parse_qs(urlparse(r["authorize_url"]).query).items()}
    fake.challenge = q["code_challenge"]
    ok, msg = await svc.complete_oauth(db, q["state"], "good-code")
    assert ok, msg
    return q


def _order(ticker="AAPL", side="buy", shares=5, limit=100.5, **kw):
    d = {"plan_item_id": f"{side}:{ticker}", "ticker": ticker, "side": side, "shares": shares,
         "limit_price": limit, "time_in_force": "gfd",
         "plan": {"stop": 95.0, "target": 110.0, "strategy": "pullback_50ma",
                  "planned_entry": 100.0, "planned_entry_high": 101.0, "thesis": "dip in uptrend",
                  "invalidation": "close below 50MA", "time_stop_days": 20}}
    d.update(kw)
    return d


# ── secrets ────────────────────────────────────────────────────────────────

@test
def test_secrets_round_trip_and_key_file_next_to_db():
    secrets_box.reset_cache()
    c = secrets_box.encrypt("hello-token")
    assert c != "hello-token" and "hello" not in c
    assert secrets_box.decrypt(c) == "hello-token"
    assert os.path.isfile(os.path.join(_TMP, "broker.key")), "key file must live next to the DB"


@test
def test_secrets_wrong_key_is_a_clear_error():
    secrets_box.reset_cache()
    c = secrets_box.encrypt("x")
    from cryptography.fernet import Fernet
    os.environ["BROKER_ENCRYPTION_KEY"] = Fernet.generate_key().decode()
    secrets_box.reset_cache()
    try:
        secrets_box.decrypt(c)
        raise AssertionError("expected SecretsUnavailable")
    except secrets_box.SecretsUnavailable as exc:
        assert "reconnect" in str(exc).lower()
    finally:
        os.environ.pop("BROKER_ENCRYPTION_KEY", None)
        secrets_box.reset_cache()


# ── OAuth ──────────────────────────────────────────────────────────────────

@test
def test_pkce_is_s256():
    v, c = mcp.pkce_pair()
    assert 43 <= len(v) <= 128 and re.fullmatch(r"[A-Za-z0-9_\-]+", v)
    assert c == base64.urlsafe_b64encode(hashlib.sha256(v.encode()).digest()).decode().rstrip("=")
    assert "=" not in c


@test
def test_discovery_uses_metadata_then_falls_back():
    fake = FakeRobinhood()
    _install(fake)
    meta = mcp.discover()
    assert meta["token_endpoint"] == "https://api.robinhood.com/oauth2/token/"
    assert meta["registration_endpoint"].endswith("/oauth/trading/register")
    assert meta["resource"] == mcp.MCP_URL

    def down(request):
        raise httpx.ConnectError("down")
    mcp._meta_cache.clear()
    mcp.TRANSPORT = httpx.MockTransport(down)
    meta = mcp.discover()
    assert meta["authorization_endpoint"] == mcp.FALLBACK_METADATA["authorization_endpoint"]
    mcp._meta_cache.clear()


@test
def test_registration_is_public_client_and_reused():
    fake = FakeRobinhood()
    _install(fake)
    db = SessionLocal()
    u = _user(db)
    r1 = _run(svc.begin_connect(db, u, REDIRECT))
    r2 = _run(svc.begin_connect(db, u, REDIRECT))
    assert len(fake.registrations) == 1, "client must be registered once per redirect URI"
    reg = fake.registrations[0]
    assert reg["token_endpoint_auth_method"] == "none"
    assert reg["redirect_uris"] == [REDIRECT]
    assert set(reg["grant_types"]) == {"authorization_code", "refresh_token"}
    q = parse_qs(urlparse(r2["authorize_url"]).query)
    assert q["code_challenge_method"] == ["S256"] and q["client_id"] == ["cid-1"]
    assert q["redirect_uri"] == [REDIRECT] and q["resource"] == [mcp.MCP_URL]
    assert q["response_type"] == ["code"] and q["state"][0]
    assert r1["authorize_url"].startswith("https://robinhood.com/oauth?")
    _run(svc.begin_connect(db, u, "https://other.example.com/api/broker/oauth/callback"))
    assert len(fake.registrations) == 2, "a new redirect URI needs a new registration"
    db.close()


@test
def test_registration_rejection_is_surfaced_not_raised():
    fake = FakeRobinhood()
    fake.reject_registration = True
    _install(fake)
    db = SessionLocal()
    u = _user(db)
    r = _run(svc.begin_connect(db, u, REDIRECT))
    assert r["status"] == "error" and "redirect host not allowed" in r["message"], r
    assert "redirect host not allowed" in svc.status(db, u)["last_error"]
    db.close()


@test
def test_callback_state_is_single_use_user_bound_and_expires():
    fake = FakeRobinhood()
    _install(fake)
    db = SessionLocal()
    u = _user(db)
    ok, msg = _run(svc.complete_oauth(db, "forged-state", "good-code"))
    assert not ok and "invalid or expired" in msg

    r = _run(svc.begin_connect(db, u, REDIRECT))
    q = {k: v[0] for k, v in parse_qs(urlparse(r["authorize_url"]).query).items()}
    svc._pending_oauth[q["state"]]["created"] -= svc.OAUTH_STATE_TTL + 5
    ok, msg = _run(svc.complete_oauth(db, q["state"], "good-code"))
    assert not ok and "expired" in msg

    q = _run(_connect(db, u, fake))
    ok, _ = _run(svc.complete_oauth(db, q["state"], "good-code"))
    assert not ok, "state must be single-use"
    db.close()


@test
def test_callback_exchanges_code_with_verifier_and_stores_encrypted():
    fake = FakeRobinhood()
    _install(fake)
    db = SessionLocal()
    u = _user(db)
    _run(_connect(db, u, fake))
    tr = [t for t in fake.token_requests if t["grant_type"] == "authorization_code"][-1]
    assert tr["client_id"] == "cid-1" and tr["redirect_uri"] == REDIRECT
    assert tr["resource"] == mcp.MCP_URL and tr["code_verifier"]
    acct = svc.get_account(db, u.id)
    db.refresh(acct)
    raw = json.dumps({c.name: str(getattr(acct, c.name)) for c in acct.__table__.columns})
    assert "AT-secret" not in raw and "RT-secret" not in raw, "tokens must be encrypted at rest"
    assert secrets_box.decrypt(acct.access_token_enc).startswith("AT-secret")
    assert acct.mcp_session_id == "sess-1"
    assert len(json.loads(acct.tools_json)) == 5
    assert acct.account_number == "999888", "must pick the agentic account"
    st = svc.status(db, u)
    assert st["connected"] and st["review_tool"] == "review_equity_order" and st["place_tool"] == "place_equity_order"
    assert st["account_number_masked"] == "****9888"
    db.close()


@test
def test_callback_error_from_robinhood():
    fake = FakeRobinhood()
    _install(fake)
    db = SessionLocal()
    u = _user(db)
    r = _run(svc.begin_connect(db, u, REDIRECT))
    state = parse_qs(urlparse(r["authorize_url"]).query)["state"][0]
    ok, msg = _run(svc.complete_oauth(db, state, None, "access_denied", "User declined"))
    assert not ok and "User declined" in msg
    assert not svc.status(db, u)["connected"]
    db.close()


# ── MCP session ───────────────────────────────────────────────────────────

@test
def test_mcp_initialize_sse_and_tools():
    fake = FakeRobinhood()
    fake.use_sse = True
    _install(fake)
    tok = fake._issue()["access_token"]
    with mcp.McpSession(tok) as s:
        tools = s.list_tools()
        assert s.session_id == "sess-1"
    assert [t["name"] for t in tools][:2] == ["review_equity_order", "place_equity_order"]
    posts = [u for m, u in fake.calls if m == "POST" and u == mcp.MCP_URL]
    assert len(posts) == 3, "initialize + notifications/initialized + tools/list"


@test
def test_401_refreshes_once_and_persists_new_token():
    fake = FakeRobinhood()
    _install(fake)
    db = SessionLocal()
    u = _user(db)
    _run(_connect(db, u, fake))
    fake.valid_tokens = set()          # server-side expiry of the access token
    data = _run(svc.account(db, u)) if svc.set_mode(db, u, "live", True) else None
    assert data["source"] == "robinhood" and data["buying_power"] == 5000.0
    acct = svc.get_account(db, u.id)
    db.refresh(acct)
    assert secrets_box.decrypt(acct.access_token_enc) in fake.valid_tokens, "refreshed token persisted"
    assert any(t.get("grant_type") == "refresh_token" for t in fake.token_requests)
    db.close()


@test
def test_failed_refresh_disconnects():
    fake = FakeRobinhood()
    _install(fake)
    db = SessionLocal()
    u = _user(db)
    _run(_connect(db, u, fake))
    svc.set_mode(db, u, "live", True)
    fake.valid_tokens = set()
    fake.allow_refresh = False
    try:
        _run(svc.account(db, u))
        raise AssertionError("expected ReauthRequired")
    except mcp.ReauthRequired:
        pass
    st = svc.status(db, u)
    assert not st["connected"] and "reconnect" in (st["last_error"] or "").lower()
    db.close()


# ── Argument mapping ──────────────────────────────────────────────────────

@test
def test_map_arguments_synonyms_enums_types():
    fields = {"symbol": "BRK.B", "side": "buy", "quantity": 3, "order_type": "limit",
              "limit_price": 412.5, "time_in_force": "gfd", "account_number": "999888"}
    args = mcp.map_arguments({"name": "place_equity_order", "inputSchema": ORDER_SCHEMA}, fields)
    assert args == {"symbol": "BRK.B", "side": "BUY", "quantity": 3.0, "order_type": "limit",
                    "limit_price": 412.5, "time_in_force": "day", "account_number": "999888"}, args
    alt = {"type": "object", "required": ["ticker", "shares", "direction", "price", "type"],
           "properties": {"ticker": {"type": "string"}, "shares": {"type": "integer"},
                          "direction": {"type": "string"}, "price": {"type": "string"},
                          "type": {"type": "string"}, "tif": {"type": "string", "enum": ["GTC", "DAY"]},
                          "note": {"type": "string"}}}
    args = mcp.map_arguments({"name": "x", "inputSchema": alt}, fields)
    assert args == {"ticker": "BRK.B", "shares": 3, "direction": "buy", "price": "412.5",
                    "type": "limit", "tif": "DAY"}, args


@test
def test_unmappable_required_property_fails_with_schema():
    schema = {"type": "object", "required": ["symbol", "confirmation_token"],
              "properties": {"symbol": {"type": "string"}, "confirmation_token": {"type": "string"}}}
    try:
        mcp.map_arguments({"name": "place_equity_order", "inputSchema": schema}, {"symbol": "AAPL"})
        raise AssertionError("expected MappingError")
    except mcp.MappingError as exc:
        assert "confirmation_token" in str(exc) and "Tool schema" in str(exc) and "place_equity_order" in str(exc)
    bad_enum = {"type": "object", "required": ["side"],
                "properties": {"side": {"type": "string", "enum": ["LONG", "SHORT"]}}}
    try:
        mcp.map_arguments({"name": "t", "inputSchema": bad_enum}, {"side": "buy"})
        raise AssertionError("expected MappingError")
    except mcp.MappingError:
        pass


# ── Validation ────────────────────────────────────────────────────────────

@test
def test_validation_rules():
    q = {"AAPL": 100.0, "MSFT": 400.0}
    held = {"MSFT": 10}
    rows, t = svc.validate_orders([
        _order("AAPL", shares=2.5),                      # fractional
        _order("AAPL", "sell", shares=1, limit=100),     # not held
        _order("MSFT", "sell", shares=11, limit=400),    # more than held
        _order("MSFT", "buy", shares=1, limit=500),      # >15% from last
        _order("MSFT", "buy", shares=1, limit=401),      # duplicate ticker+side? no: first buy MSFT errored but seen
        _order("ZZZZ", shares=1, limit=10),              # no quote (strict)
    ], q, held, None, strict_quotes=True)
    assert "Whole shares only." in rows[0]["errors"]
    assert any("don't hold" in e for e in rows[1]["errors"])
    assert any("only 10" in e for e in rows[2]["errors"])
    assert any("15%" in e for e in rows[3]["errors"])
    assert any("Duplicate" in e for e in rows[4]["errors"])
    assert any("No current price" in e for e in rows[5]["errors"])
    rows, t = svc.validate_orders([_order("ZZZZ", shares=1, limit=10)], q, held, None, strict_quotes=False)
    assert rows[0]["ok"] and rows[0]["warnings"]
    # Live (strict): a buying-power shortfall blocks the batch.
    rows, t = svc.validate_orders([_order("AAPL", shares=50, limit=100)], q, held, 1000.0, True)
    assert not rows[0]["ok"] and t["errors"] and "buying power" in t["errors"][0]
    # Paper: derived buying power only warns, or paper mode is unusable for a
    # book imported from another broker that exceeds the Settings account size.
    rows, t = svc.validate_orders([_order("AAPL", shares=50, limit=100)], q, held, 1000.0, False)
    assert rows[0]["ok"] and not t["errors"] and "buying power" in t["warnings"][0]
    rows, t = svc.validate_orders([_order("AAPL", shares=1, limit=100)] * 21, q, held, None, False)
    assert any("At most 20" in e for e in t["errors"])
    rows, t = svc.validate_orders([], q, held, None, False)
    assert t["errors"] == ["No orders selected."]


# ── Paper mode ────────────────────────────────────────────────────────────

@test
def test_paper_orders_never_touch_network_or_portfolio():
    _install(None)   # any network use raises
    db = SessionLocal()
    u = _user(db)
    before = db.query(PortfolioPosition).filter(PortfolioPosition.user_id == u.id).count()
    prev = _run(svc.preview(db, u, [_order("AAPL", shares=5, limit=100.5)]))
    assert prev["mode"] == "paper" and prev["orders"][0]["ok"], prev
    res = _run(svc.place(db, u, [_order("AAPL", shares=5, limit=100.5)], confirm=False))
    assert res["placed"] == 1 and res["results"][0]["status"] == "simulated"
    after = db.query(PortfolioPosition).filter(PortfolioPosition.user_id == u.id).count()
    assert before == after == 0
    o = db.query(BrokerOrder).filter(BrokerOrder.user_id == u.id).one()
    assert o.mode == "paper" and o.status == "simulated" and not o.applied_to_portfolio
    assert json.loads(o.plan_json)["stop"] == 95.0
    s = _run(svc.sync(db, u))
    assert s["checked"] == 0
    db.close()


@test
def test_live_mode_needs_connection_and_confirm():
    _install(None)
    db = SessionLocal()
    u = _user(db)
    for args in (("live", True), ("live", False)):
        try:
            svc.set_mode(db, u, *args)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    fake = FakeRobinhood()
    _install(fake)
    _run(_connect(db, u, fake))
    try:
        svc.set_mode(db, u, "live", False)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "confirm" in str(exc)
    svc.set_mode(db, u, "live", True)
    try:
        _run(svc.place(db, u, [_order()], confirm=False))
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "confirm" in str(exc)
    assert not any(n == "place_equity_order" for n, _ in fake.tool_calls)
    db.close()


# ── Live: review -> place, sync, resolve ─────────────────────────────────

@test
def test_live_review_runs_before_place_and_rejection_blocks_place():
    fake = FakeRobinhood()
    _install(fake)
    db = SessionLocal()
    u = _user(db)
    _run(_connect(db, u, fake))
    svc.set_mode(db, u, "live", True)
    prev = _run(svc.preview(db, u, [_order("AAPL", shares=5, limit=100.5)]))
    assert prev["orders"][0]["broker_review"].startswith("Estimated cost"), prev
    assert prev["totals"]["buying_power"] == 5000.0
    fake.tool_calls.clear()
    res = _run(svc.place(db, u, [_order("AAPL", shares=5, limit=100.5)], confirm=True))
    names = [n for n, _ in fake.tool_calls if n in ("review_equity_order", "place_equity_order")]
    assert names == ["review_equity_order", "place_equity_order"], names
    placed_args = [a for n, a in fake.tool_calls if n == "place_equity_order"][0]
    assert placed_args["account_number"] == "999888" and placed_args["side"] == "BUY"
    assert placed_args["time_in_force"] == "day" and placed_args["order_type"] == "limit"
    r = res["results"][0]
    assert r["status"] == "queued" and "Approve it in the Robinhood app" in r["message"]
    o = db.query(BrokerOrder).filter(BrokerOrder.id == r["order_id"]).one()
    assert o.broker_order_id == "rh-ord-1"

    fake.review_ok = False
    fake.tool_calls.clear()
    res = _run(svc.place(db, u, [_order("NVDA", shares=1, limit=120)], confirm=True))
    assert res["results"][0]["status"] == "failed"
    assert "Insufficient buying power" in res["results"][0]["message"]
    assert not any(n == "place_equity_order" for n, _ in fake.tool_calls), "rejected review must not place"
    db.close()


@test
def test_sync_applies_fills_exactly_once_and_full_sell_journals():
    fake = FakeRobinhood()
    _install(fake)
    db = SessionLocal()
    u = _user(db)
    _run(_connect(db, u, fake))
    svc.set_mode(db, u, "live", True)
    _run(svc.place(db, u, [_order("AAPL", shares=5, limit=100.5)], confirm=True))
    s = _run(svc.sync(db, u))
    assert s["checked"] == 1 and not db.query(PortfolioPosition).filter(
        PortfolioPosition.user_id == u.id).count(), "queued order must not touch the portfolio"
    fake.order_state = {"state": "filled", "cumulative_quantity": "5.00000", "average_price": "100.2500"}
    s = _run(svc.sync(db, u))
    pos = db.query(PortfolioPosition).filter(PortfolioPosition.user_id == u.id,
                                             PortfolioPosition.ticker == "AAPL").one()
    assert pos.shares == 5 and abs(pos.avg_cost - 100.25) < 1e-9
    assert pos.stop_loss == 95.0 and pos.initial_stop == 95.0 and pos.target == 110.0
    assert pos.strategy == "pullback_50ma" and pos.planned_entry == 100.0 and pos.time_stop_days == 20
    s2 = _run(svc.sync(db, u))
    assert s2["checked"] == 0
    assert db.query(PortfolioPosition).filter(PortfolioPosition.user_id == u.id).one().shares == 5

    # Full sell -> journal
    fake.positions = [{"symbol": "AAPL", "quantity": "5"}]
    fake.order_state = {"state": "queued", "cumulative_quantity": "0"}
    _run(svc.place(db, u, [_order("AAPL", "sell", shares=5, limit=104)], confirm=True))
    fake.order_state = {"state": "filled", "cumulative_quantity": "5", "average_price": "104.00"}
    _run(svc.sync(db, u))
    assert not db.query(PortfolioPosition).filter(PortfolioPosition.user_id == u.id).count()
    ct = db.query(ClosedTrade).filter(ClosedTrade.user_id == u.id).one()
    assert ct.exit_price == 104.0 and ct.r_multiple is not None and ct.planned_entry == 100.0
    _run(svc.sync(db, u))
    assert db.query(ClosedTrade).filter(ClosedTrade.user_id == u.id).count() == 1
    db.close()


@test
def test_manual_resolve_when_no_status_tool():
    fake = FakeRobinhood()
    fake.tools = [t for t in fake.tools if t["name"] != "get_order"]
    _install(fake)
    db = SessionLocal()
    u = _user(db)
    _run(_connect(db, u, fake))
    svc.set_mode(db, u, "live", True)
    res = _run(svc.place(db, u, [_order("AAPL", shares=4, limit=100.5)], confirm=True))
    oid = res["results"][0]["order_id"]
    s = _run(svc.sync(db, u))
    assert any("no order-status tool" in n for n in s["notes"]), s
    d = svc.resolve_manually(db, u, oid, "filled", 100.1, 4)
    assert d["status"] == "filled" and d["applied_to_portfolio"]
    try:
        svc.resolve_manually(db, u, oid, "filled", 100.1, 4)
        raise AssertionError("second resolve must fail")
    except ValueError:
        pass
    assert db.query(PortfolioPosition).filter(PortfolioPosition.user_id == u.id).one().shares == 4
    db.close()


# ── Holdings: compare + import ─────────────────────────────────────────────

@test
def test_compare_holdings_statuses_and_ticker_forms():
    st = [PortfolioPosition(ticker="BRK-B", shares=2, avg_cost=400.0),
          PortfolioPosition(ticker="AAPL", shares=5, avg_cost=100.0),
          PortfolioPosition(ticker="VTI", shares=3, avg_cost=250.0)]
    rh = [{"ticker": "BRK.B", "shares": 2.0, "avg_cost": 410.0},
          {"ticker": "AAPL", "shares": 7.0, "avg_cost": 101.0},
          {"ticker": "MSFT", "shares": 10.0, "avg_cost": 400.0},
          {"ticker": "NVDA", "shares": 4.0, "avg_cost": None}]
    rows = {r["ticker"]: r for r in svc.compare_holdings(rh, st)}
    assert rows["BRK-B"]["status"] == "match", "BRK.B and BRK-B are the same holding"
    assert rows["AAPL"]["status"] == "shares_differ" and not rows["AAPL"]["importable"]
    assert rows["MSFT"]["status"] == "robinhood_only" and rows["MSFT"]["importable"]
    assert rows["NVDA"]["status"] == "robinhood_only" and not rows["NVDA"]["importable"] \
        and "average cost" in rows["NVDA"]["reason"]
    assert rows["VTI"]["status"] == "swingtrader_only" and not rows["VTI"]["importable"]


@test
def test_holdings_disconnected_makes_no_network_call():
    _install(None)
    db = SessionLocal()
    u = _user(db)
    h = _run(svc.holdings(db, u))
    assert h["connected"] is False and h["readable"] is False and h["rows"] == []
    try:
        _run(svc.import_holdings(db, u, ["MSFT"]))
        raise AssertionError("import without a connection must fail")
    except mcp.ReauthRequired:
        pass
    db.close()


@test
def test_holdings_read_in_paper_mode_scoped_to_agentic_account_and_read_only():
    fake = FakeRobinhood()
    _install(fake)
    db = SessionLocal()
    u = _user(db)
    _run(_connect(db, u, fake))
    db.add(PortfolioPosition(user_id=u.id, ticker="VTI", shares=3, avg_cost=250.0))
    db.commit()
    h = _run(svc.holdings(db, u))
    assert h["mode"] == "paper" and h["readable"], h
    pos_calls = [a for n, a in fake.tool_calls if n == "get_positions"]
    assert pos_calls and all(a.get("account_number") == "999888" for a in pos_calls), \
        "positions must be read from the AGENTIC account, even on the very first call"
    rows = {r["ticker"]: r["status"] for r in h["rows"]}
    assert rows == {"MSFT": "robinhood_only", "VTI": "swingtrader_only"}, rows
    assert h["summary"]["importable"] == 1
    assert not any(n in ("review_equity_order", "place_equity_order") for n, _ in fake.tool_calls)
    assert db.query(PortfolioPosition).filter(PortfolioPosition.user_id == u.id).count() == 1, \
        "loading holdings must never change the portfolio"
    db.close()


@test
def test_import_rereads_robinhood_adds_only_new_and_never_edits_existing():
    fake = FakeRobinhood()
    fake.positions = [{"symbol": "MSFT", "quantity": "10", "average_buy_price": "400"},
                      {"symbol": "AAPL", "quantity": "9", "average_buy_price": "90"},
                      {"symbol": "NVDA", "quantity": "4"}]
    _install(fake)
    db = SessionLocal()
    u = _user(db)
    _run(_connect(db, u, fake))
    db.add(PortfolioPosition(user_id=u.id, ticker="AAPL", shares=5, avg_cost=100.0, stop_loss=95.0))
    db.commit()
    r = _run(svc.import_holdings(db, u, ["msft", "AAPL", "NVDA", "TSLA"]))
    assert [x["ticker"] for x in r["imported"]] == ["MSFT"], r
    reasons = {x["ticker"]: x["reason"] for x in r["skipped"]}
    assert set(reasons) == {"AAPL", "NVDA", "TSLA"}, reasons
    msft = db.query(PortfolioPosition).filter(PortfolioPosition.user_id == u.id,
                                              PortfolioPosition.ticker == "MSFT").one()
    assert msft.shares == 10 and msft.avg_cost == 400.0, "shares/cost come from Robinhood"
    assert msft.stop_loss is None and msft.initial_stop is None and "Robinhood" in msft.notes
    aapl = db.query(PortfolioPosition).filter(PortfolioPosition.user_id == u.id,
                                              PortfolioPosition.ticker == "AAPL").one()
    assert aapl.shares == 5 and aapl.avg_cost == 100.0 and aapl.stop_loss == 95.0, \
        "an existing position must never be edited by an import"
    r2 = _run(svc.import_holdings(db, u, ["MSFT"]))
    assert not r2["imported"] and db.query(PortfolioPosition).filter(
        PortfolioPosition.user_id == u.id, PortfolioPosition.ticker == "MSFT").count() == 1, \
        "a second import must not duplicate"
    try:
        _run(svc.import_holdings(db, u, []))
        raise AssertionError("empty import must be rejected")
    except ValueError:
        pass
    db.close()


# ── API surface: no secrets, callback redirects ──────────────────────────

@test
def test_api_never_returns_tokens_and_callback_redirects():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.broker import router
    from auth.deps import get_current_user
    from database.db import get_db

    fake = FakeRobinhood()
    _install(fake)
    db = SessionLocal()
    u = _user(db)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: db.query(User).filter(User.id == u.id).one()
    client = TestClient(app, base_url="https://testserver")

    r = client.post("/api/broker/connect", headers={"x-forwarded-host": "invest.example.com",
                                                   "x-forwarded-proto": "https"})
    body = r.json()
    assert body["redirect_uri"] == REDIRECT, body
    q = {k: v[0] for k, v in parse_qs(urlparse(body["authorize_url"]).query).items()}
    fake.challenge = q["code_challenge"]
    bad = client.get("/api/broker/oauth/callback", params={"state": "nope", "code": "good-code"},
                     follow_redirects=False)
    assert bad.status_code == 302 and bad.headers["location"].startswith("/trade?error=")
    ok = client.get("/api/broker/oauth/callback", params={"state": q["state"], "code": "good-code"},
                    follow_redirects=False)
    assert ok.status_code == 302 and ok.headers["location"] == "/trade?connected=1", ok.headers
    client.put("/api/broker/mode", json={"mode": "live", "confirm": True})
    texts = [
        client.get("/api/broker/status").text,
        client.get("/api/broker/tools").text,
        client.get("/api/broker/account").text,
        client.get("/api/broker/holdings").text,
        client.post("/api/broker/holdings/import", json={"tickers": ["MSFT"]}).text,
        client.post("/api/broker/orders/preview", json={"orders": [_order()]}).text,
        client.post("/api/broker/orders", json={"orders": [_order()], "confirm": True}).text,
        client.get("/api/broker/orders").text,
        client.post("/api/broker/orders/sync").text,
    ]
    for t in texts:
        assert "AT-secret" not in t and "RT-secret" not in t and "code_verifier" not in t, t[:300]
    assert client.post("/api/broker/orders", json={"orders": [_order()], "confirm": False}).status_code == 400
    assert client.post("/api/broker/disconnect").json()["status"] == "disconnected"
    assert client.get("/api/broker/status").json()["connected"] is False
    db.close()


# ── Page ──────────────────────────────────────────────────────────────────

@test
def test_trade_page_shares_chrome_and_compiles_with_common_js():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "static", "trade.html"), encoding="utf-8") as fh:
        html = fh.read()
    with open(os.path.join(here, "static", "js", "common.js"), encoding="utf-8") as fh:
        common = fh.read()
    assert 'href="/static/css/common.css"' in html and re.search(r'<script[^>]+src="[^"]*common\.js"', html)
    assert re.search(r"initHeader\(\s*['\"]/trade", html)
    names = set(re.findall(r"^(?:const|let|var|async function|function|class)\s+([A-Za-z_$][\w$]*)",
                           common, re.M))
    blocks = re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, re.S)
    for blk in blocks:
        for kw, name in re.findall(r"^\s{0,6}(const|let|function|async function)\s+([A-Za-z_$][\w$]*)", blk, re.M):
            assert name not in names, "trade.html redeclares common.js name " + name
    with open(os.path.join(here, "main.py"), encoding="utf-8") as fh:
        assert '@app.get("/trade")' in fh.read()
    try:
        subprocess.run(["node", "--version"], capture_output=True, check=True, timeout=20)
    except Exception:
        return
    fd, tmp = tempfile.mkstemp(suffix=".js")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(common + "\n;\n" + "\n".join(blocks))
        proc = subprocess.run(["node", "--check", tmp], capture_output=True, text=True, timeout=60)
    finally:
        os.unlink(tmp)
    assert proc.returncode == 0, proc.stderr[:600]


def main_runner() -> int:
    passed = failed = 0
    for fn in _TESTS:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print("FAIL  " + fn.__name__ + "  --  " + repr(exc))
        else:
            passed += 1
            print("PASS  " + fn.__name__)
    print("\n" + "=" * 56)
    print(f"  {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 56)
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        code = main_runner()
    finally:
        mcp.TRANSPORT = None
        engine.dispose()
        shutil.rmtree(_TMP, ignore_errors=True)
    sys.exit(code)
