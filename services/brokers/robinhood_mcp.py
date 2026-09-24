"""
Client for Robinhood's OFFICIAL agentic-trading MCP server.

    MCP endpoint   https://agent.robinhood.com/mcp/trading   (Streamable HTTP, JSON-RPC 2.0)
    Auth           OAuth 2.1 authorization-code + PKCE (S256), public client,
                   RFC 7591 dynamic client registration, scope "internal".

No Robinhood password ever reaches this app. The user logs in on robinhood.com,
Robinhood redirects back to /api/broker/oauth/callback with a code, and we
exchange it for a revocable token. Trading only happens inside the user's
dedicated Robinhood **Agentic account** (equities only, beta).

Two things are deliberately NOT hard-coded:
  * OAuth endpoints — discovered at runtime from the protected-resource and
    authorization-server metadata; the constants below are documented fallbacks
    (values verified live on 2026-09-24).
  * Tool argument names — Robinhood has not published the tool schemas. We call
    tools/list and map our order fields onto whatever property names each
    tool's inputSchema declares (map_arguments). A REQUIRED property we cannot
    map fails that order with the schema in the message — never a silent guess.

Every function here is synchronous and bounded by one HTTP timeout per request;
services/broker_service.py runs them off the event loop. Nothing logs a token.
The transport is injectable (TRANSPORT) so test_broker.py runs with no network.
"""
from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
from typing import Any, Callable
from urllib.parse import urlencode

import httpx

MCP_URL = "https://agent.robinhood.com/mcp/trading"
RESOURCE_METADATA_URL = "https://agent.robinhood.com/.well-known/oauth-protected-resource/mcp/trading"
AS_METADATA_URL = "https://agent.robinhood.com/.well-known/oauth-authorization-server/mcp/trading"
FALLBACK_METADATA = {
    "issuer": "https://agent.robinhood.com/mcp/trading",
    "authorization_endpoint": "https://robinhood.com/oauth",
    "token_endpoint": "https://api.robinhood.com/oauth2/token/",
    "registration_endpoint": "https://agent.robinhood.com/oauth/trading/register",
    "resource": MCP_URL,
    "scope": "internal",
}
PROTOCOL_VERSION = "2025-06-18"
CLIENT_NAME = "SwingTrader (self-hosted)"
DEFAULT_TIMEOUT = 20.0

REVIEW_TOOL = "review_equity_order"
PLACE_TOOL = "place_equity_order"

# Test hook: an httpx transport (e.g. httpx.MockTransport). None = real network.
TRANSPORT: httpx.BaseTransport | None = None


class BrokerError(Exception):
    """A failure whose message is safe to show the user."""


class ReauthRequired(BrokerError):
    """Token rejected and refresh failed — the user must reconnect."""


class RegistrationError(BrokerError):
    """Robinhood refused dynamic client registration."""


class MappingError(BrokerError):
    """A tool requires an argument we don't know how to fill."""


def http_client(timeout: float = DEFAULT_TIMEOUT) -> httpx.Client:
    return httpx.Client(transport=TRANSPORT, timeout=timeout, follow_redirects=False)


def _json(res: httpx.Response) -> Any:
    try:
        return res.json()
    except Exception:  # noqa: BLE001
        return None


def _err_text(body: Any, fallback: str) -> str:
    if isinstance(body, dict):
        for k in ("error_description", "detail", "message", "error"):
            v = body.get(k)
            if isinstance(v, list) and v:
                v = v[0]
            if isinstance(v, str) and v.strip():
                return v.strip()[:400]
    return fallback


# ── OAuth ──────────────────────────────────────────────────────────────────

_meta_cache: dict[str, tuple[dict, float]] = {}
_META_TTL = 3600.0


def discover(http: httpx.Client | None = None) -> dict:
    """OAuth endpoints for the MCP resource, via RFC 9728 + RFC 8414 discovery.

    Falls back field-by-field to FALLBACK_METADATA, so a discovery outage or a
    renamed well-known path degrades to the verified values instead of failing.
    """
    hit = _meta_cache.get("meta")
    if hit and time.time() - hit[1] < _META_TTL:
        return dict(hit[0])
    own = http is None
    http = http or http_client()
    meta = dict(FALLBACK_METADATA)
    try:
        as_url = AS_METADATA_URL
        try:
            prm = _json(http.get(RESOURCE_METADATA_URL)) or {}
            if isinstance(prm, dict):
                if prm.get("resource"):
                    meta["resource"] = prm["resource"]
                servers = prm.get("authorization_servers") or []
                if servers and isinstance(servers[0], str):
                    issuer = servers[0].rstrip("/")
                    # RFC 8414: insert the well-known segment between host and path.
                    scheme, rest = issuer.split("://", 1)
                    host, _, path = rest.partition("/")
                    as_url = f"{scheme}://{host}/.well-known/oauth-authorization-server" + (
                        "/" + path if path else "")
                scopes = prm.get("scopes_supported") or []
                if scopes:
                    meta["scope"] = " ".join(scopes)
        except httpx.HTTPError:
            pass
        try:
            asm = _json(http.get(as_url)) or {}
            if isinstance(asm, dict):
                for k in ("issuer", "authorization_endpoint", "token_endpoint", "registration_endpoint"):
                    if isinstance(asm.get(k), str) and asm[k]:
                        meta[k] = asm[k]
        except httpx.HTTPError:
            pass
    finally:
        if own:
            http.close()
    _meta_cache["meta"] = (dict(meta), time.time())
    return meta


def register_client(meta: dict, redirect_uri: str, http: httpx.Client | None = None) -> str:
    """RFC 7591 dynamic registration as a public client. Returns client_id."""
    own = http is None
    http = http or http_client()
    try:
        try:
            res = http.post(meta["registration_endpoint"], json={
                "client_name": CLIENT_NAME,
                "redirect_uris": [redirect_uri],
                "token_endpoint_auth_method": "none",
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "scope": meta.get("scope") or "internal",
            })
        except httpx.HTTPError as exc:
            raise RegistrationError("Could not reach Robinhood to register this app ("
                                    + type(exc).__name__ + ").") from exc
        body = _json(res)
        if res.status_code >= 400 or not isinstance(body, dict) or not body.get("client_id"):
            raise RegistrationError(
                "Robinhood refused to register SwingTrader for redirect URI "
                + redirect_uri + ": "
                + _err_text(body, f"HTTP {res.status_code}")
            )
        return str(body["client_id"])
    finally:
        if own:
            http.close()


def pkce_pair() -> tuple[str, str]:
    """(code_verifier, S256 code_challenge) per RFC 7636."""
    verifier = secrets.token_urlsafe(64)[:96]
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("ascii")).digest()).decode("ascii").rstrip("=")
    return verifier, challenge


def authorize_url(meta: dict, client_id: str, redirect_uri: str, state: str,
                  code_challenge: str) -> str:
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": meta.get("scope") or "internal",
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "resource": meta.get("resource") or MCP_URL,
    }
    sep = "&" if "?" in meta["authorization_endpoint"] else "?"
    return meta["authorization_endpoint"] + sep + urlencode(params)


def _token_request(meta: dict, data: dict, http: httpx.Client | None) -> dict:
    own = http is None
    http = http or http_client()
    try:
        try:
            res = http.post(meta["token_endpoint"], data=data,
                            headers={"Accept": "application/json"})
        except httpx.HTTPError as exc:
            raise BrokerError("Could not reach Robinhood's token endpoint ("
                              + type(exc).__name__ + ").") from exc
        body = _json(res)
        if res.status_code >= 400 or not isinstance(body, dict) or not body.get("access_token"):
            raise BrokerError("Robinhood did not issue a token: "
                              + _err_text(body, f"HTTP {res.status_code}"))
        return body
    finally:
        if own:
            http.close()


def exchange_code(meta: dict, client_id: str, redirect_uri: str, code: str,
                  code_verifier: str, http: httpx.Client | None = None) -> dict:
    return _token_request(meta, {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "code_verifier": code_verifier,
        "resource": meta.get("resource") or MCP_URL,
    }, http)


def refresh_access_token(meta: dict, client_id: str, refresh_token: str,
                         http: httpx.Client | None = None) -> dict:
    return _token_request(meta, {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
        "resource": meta.get("resource") or MCP_URL,
    }, http)


# ── MCP session (Streamable HTTP) ──────────────────────────────────────────

def _parse_rpc_body(res: httpx.Response, want_id: int | None) -> dict | None:
    """A JSON-RPC response from a plain-JSON or SSE (text/event-stream) body."""
    ctype = (res.headers.get("content-type") or "").lower()
    if "text/event-stream" in ctype:
        found = None
        for event in res.text.replace("\r\n", "\n").split("\n\n"):
            data = "\n".join(line[5:].lstrip() for line in event.split("\n")
                             if line.startswith("data:"))
            if not data:
                continue
            try:
                msg = json.loads(data)
            except ValueError:
                continue
            msgs = msg if isinstance(msg, list) else [msg]
            for m in msgs:
                if isinstance(m, dict) and ("result" in m or "error" in m) and (
                        want_id is None or m.get("id") == want_id):
                    found = m
        return found
    body = _json(res)
    if isinstance(body, list):
        body = next((m for m in body if isinstance(m, dict) and m.get("id") == want_id), None)
    return body if isinstance(body, dict) else None


class McpSession:
    """One user's MCP connection. Not thread-safe; use one per operation.

    `refresher` is called on a 401 and must return a new access token or None.
    After use, check `tokens_changed` / `session_id` and persist them.
    """

    def __init__(self, access_token: str, session_id: str | None = None,
                 refresher: Callable[[], str | None] | None = None,
                 url: str = MCP_URL, timeout: float = DEFAULT_TIMEOUT):
        self.access_token = access_token
        self.session_id = session_id
        self.refresher = refresher
        self.url = url
        self.tokens_changed = False
        self._next_id = 1
        self._initialized = bool(session_id)
        self._http = http_client(timeout)

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _headers(self) -> dict:
        h = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "Authorization": "Bearer " + (self.access_token or ""),
            "MCP-Protocol-Version": PROTOCOL_VERSION,
        }
        if self.session_id:
            h["Mcp-Session-Id"] = self.session_id
        return h

    def _post(self, payload: dict) -> httpx.Response:
        try:
            return self._http.post(self.url, json=payload, headers=self._headers())
        except httpx.TimeoutException as exc:
            raise BrokerError("Robinhood's trading service did not respond in time.") from exc
        except httpx.HTTPError as exc:
            raise BrokerError("Could not reach Robinhood's trading service ("
                              + type(exc).__name__ + ").") from exc

    def _send(self, payload: dict, _retried: bool = False) -> httpx.Response:
        res = self._post(payload)
        if res.status_code == 401 and not _retried:
            new = self.refresher() if self.refresher else None
            if not new:
                raise ReauthRequired("Your Robinhood connection expired — reconnect on the Trade page.")
            self.access_token = new
            self.tokens_changed = True
            return self._send(payload, _retried=True)
        if res.status_code == 401:
            raise ReauthRequired("Robinhood rejected the connection — reconnect on the Trade page.")
        if res.status_code == 404 and self.session_id and not _retried and payload.get("method") != "initialize":
            # Session expired server-side: start a new one and replay once.
            self.session_id = None
            self._initialized = False
            self.initialize()
            return self._send(payload, _retried=True)
        if res.status_code == 429:
            raise BrokerError("Robinhood is rate-limiting requests. Wait a minute and retry.")
        return res

    def _rpc(self, method: str, params: dict | None = None) -> Any:
        if not self._initialized and method != "initialize":
            self.initialize()
        rid = self._next_id
        self._next_id += 1
        payload = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            payload["params"] = params
        res = self._send(payload)
        if res.status_code >= 400:
            raise BrokerError(_err_text(_json(res), f"Robinhood MCP returned HTTP {res.status_code}."))
        msg = _parse_rpc_body(res, rid)
        if msg is None:
            raise BrokerError("Robinhood MCP returned an unreadable response to " + method + ".")
        if msg.get("error"):
            err = msg["error"]
            raise BrokerError("Robinhood MCP error on " + method + ": "
                              + str((err or {}).get("message") or err)[:400])
        return msg.get("result")

    def initialize(self) -> dict:
        self._initialized = True   # set first: _rpc must not recurse into initialize
        rid = self._next_id
        self._next_id += 1
        res = self._send({"jsonrpc": "2.0", "id": rid, "method": "initialize", "params": {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "SwingTrader", "version": "1"},
        }})
        if res.status_code >= 400:
            self._initialized = False
            raise BrokerError(_err_text(_json(res), f"Robinhood MCP initialize failed (HTTP {res.status_code})."))
        sid = res.headers.get("mcp-session-id")
        if sid:
            self.session_id = sid
        msg = _parse_rpc_body(res, rid) or {}
        # Notifications have no id and expect 202/200 with no body.
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return msg.get("result") or {}

    def list_tools(self) -> list[dict]:
        tools: list[dict] = []
        cursor = None
        for _ in range(10):
            result = self._rpc("tools/list", {"cursor": cursor} if cursor else {}) or {}
            for t in result.get("tools") or []:
                if isinstance(t, dict) and t.get("name"):
                    tools.append({"name": t["name"], "description": (t.get("description") or "")[:600],
                                  "inputSchema": t.get("inputSchema") or {}})
            cursor = result.get("nextCursor")
            if not cursor:
                break
        return tools

    def call_tool(self, name: str, arguments: dict) -> dict:
        """Returns {ok, text, data} — `text` is the tool's own words (shown
        verbatim to the user), `data` its structured content when present."""
        result = self._rpc("tools/call", {"name": name, "arguments": arguments}) or {}
        return tool_result(result)


def tool_result(result: dict) -> dict:
    texts = [c.get("text") for c in (result.get("content") or [])
             if isinstance(c, dict) and c.get("type") == "text" and c.get("text")]
    text = "\n".join(texts).strip()
    data = result.get("structuredContent")
    if data is None and text:
        try:
            data = json.loads(text)
        except ValueError:
            data = None
    return {"ok": not result.get("isError"), "text": text[:4000], "data": data}


# ── Tool discovery & argument mapping ──────────────────────────────────────

def find_tool(tools: list[dict], role: str) -> dict | None:
    """Pick the tool for a role by exact known name, else by name keywords."""
    by_name = {t["name"]: t for t in tools}
    exact = {"review": REVIEW_TOOL, "place": PLACE_TOOL}.get(role)
    if exact and exact in by_name:
        return by_name[exact]

    def has(t, *words):
        n = t["name"].lower()
        return all(w in n for w in words)

    for t in tools:
        n = t["name"].lower()
        if role == "review" and has(t, "review", "order"):
            return t
        if role == "place" and has(t, "place", "order"):
            return t
        if role == "cancel" and has(t, "cancel", "order"):
            return t
        if role == "order" and "order" in n and not any(w in n for w in ("review", "place", "cancel")):
            return t
        if role == "positions" and "position" in n:
            return t
        if role == "account" and any(w in n for w in ("account", "buying_power", "portfolio")) \
                and "order" not in n and "position" not in n:
            return t
    return None


# canonical field -> property names a tool might use for it
SYNONYMS = {
    "symbol": ("symbol", "ticker", "stock_symbol", "instrument_symbol", "equity_symbol"),
    "side": ("side", "direction", "action", "order_side", "buy_or_sell"),
    "quantity": ("quantity", "shares", "qty", "share_quantity", "number_of_shares", "share_count"),
    "order_type": ("type", "order_type", "ordertype"),
    "limit_price": ("limit_price", "price", "limit", "limitprice"),
    "time_in_force": ("time_in_force", "tif", "timeinforce", "duration"),
    "account_number": ("account_number", "account_id", "account", "accountnumber"),
    "order_id": ("order_id", "id", "orderid"),
}
_ENUM_ALIASES = {
    "gfd": ("gfd", "day", "good_for_day", "good_for_the_day"),
    "gtc": ("gtc", "good_till_cancelled", "good_til_canceled", "good_till_canceled", "good_til_cancelled"),
    "limit": ("limit", "limit_order"),
    "buy": ("buy",),
    "sell": ("sell",),
}


def _canonical_for(prop_name: str) -> str | None:
    key = prop_name.lower().replace("-", "_")
    for canon, names in SYNONYMS.items():
        if key in names or key.replace("_", "") in {n.replace("_", "") for n in names}:
            return canon
    return None


def _coerce(value, prop_schema: dict):
    ptype = prop_schema.get("type")
    if isinstance(ptype, list):
        ptype = next((t for t in ptype if t != "null"), None)
    enum = prop_schema.get("enum")
    if enum:
        wanted = [str(value)] + list(_ENUM_ALIASES.get(str(value).lower(), ()))
        for w in wanted:
            for e in enum:
                if isinstance(e, str) and e.lower() == str(w).lower():
                    return e
        raise MappingError(f"value {value!r} is not one of {enum}")
    if ptype == "string":
        return str(value)
    if ptype == "integer":
        return int(value)
    if ptype == "number":
        return float(value)
    return value


def map_arguments(tool: dict, fields: dict) -> dict:
    """Map canonical `fields` onto `tool`'s inputSchema property names.

    Raises MappingError naming the tool and showing its schema when a REQUIRED
    property has no canonical match or no value. Optional properties we can't
    fill are simply omitted.
    """
    schema = tool.get("inputSchema") or {}

    def fill(obj_schema: dict, path: str) -> dict:
        props = obj_schema.get("properties") or {}
        required = set(obj_schema.get("required") or [])
        out: dict = {}
        for pname, pschema in props.items():
            pschema = pschema if isinstance(pschema, dict) else {}
            if pschema.get("type") == "object" and pschema.get("properties"):
                if pname in required:
                    out[pname] = fill(pschema, path + pname + ".")
                continue
            canon = _canonical_for(pname)
            value = fields.get(canon) if canon else None
            if value is None:
                if pname in required:
                    raise MappingError(
                        f"Tool '{tool.get('name')}' requires '{path}{pname}', which SwingTrader "
                        f"cannot fill. Tool schema: {json.dumps(schema)[:1500]}")
                continue
            try:
                out[pname] = _coerce(value, pschema)
            except (MappingError, ValueError, TypeError) as exc:
                raise MappingError(
                    f"Tool '{tool.get('name')}' property '{path}{pname}': {exc}. "
                    f"Tool schema: {json.dumps(schema)[:1500]}") from exc
        return out

    return fill(schema, "")


# ── Result parsing (best effort — schemas are unpublished) ────────────────

def find_value(obj: Any, keys: tuple[str, ...], depth: int = 0):
    """First value under any of `keys` in a nested dict/list structure."""
    if depth > 6:
        return None
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and obj[k] not in (None, ""):
                return obj[k]
        for v in obj.values():
            r = find_value(v, keys, depth + 1)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_value(v, keys, depth + 1)
            if r is not None:
                return r
    return None


def to_float(v) -> float | None:
    try:
        return float(str(v).replace("$", "").replace(",", "")) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def parse_order(data: Any) -> dict:
    """{broker_order_id, status, filled_quantity, avg_fill_price} from a tool result."""
    oid = find_value(data, ("order_id", "id"))
    state = find_value(data, ("state", "status", "order_state", "order_status"))
    return {
        "broker_order_id": str(oid) if oid is not None else None,
        "status": str(state).lower() if state is not None else None,
        "filled_quantity": to_float(find_value(data, ("cumulative_quantity", "filled_quantity",
                                                      "quantity_filled", "executed_quantity"))),
        "avg_fill_price": to_float(find_value(data, ("average_price", "avg_fill_price",
                                                     "average_fill_price", "fill_price"))),
    }


def parse_positions(data: Any) -> list[dict] | None:
    """[{ticker, shares, avg_cost}] or None when the shape is unrecognized."""
    rows = None
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        for k in ("positions", "results", "holdings", "data"):
            if isinstance(data.get(k), list):
                rows = data[k]
                break
    if rows is None:
        return None
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        sym = find_value(r, ("symbol", "ticker"))
        qty = to_float(find_value(r, ("quantity", "shares", "qty")))
        if sym and qty:
            out.append({"ticker": str(sym).upper(), "shares": qty,
                        "avg_cost": to_float(find_value(r, ("average_buy_price", "avg_cost",
                                                            "average_cost", "cost_basis_per_share")))})
    return out


def account_entry(data: Any, account_number: str | None) -> Any:
    """The account record for `account_number` inside a multi-account result
    (so buying power is read from the AGENTIC account, not whichever comes
    first). Returns `data` unchanged when it isn't a list of accounts."""
    rows = None
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        for k in ("accounts", "results", "data"):
            if isinstance(data.get(k), list):
                rows = data[k]
                break
    if not rows or not account_number:
        return data
    for r in rows:
        if isinstance(r, dict) and str(find_value(r, ("account_number", "account_id"))) == str(account_number):
            return r
    return data


def pick_agentic_account(data: Any) -> str | None:
    """The agentic account's number if the account tool lists several."""
    rows = []
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        for k in ("accounts", "results", "data"):
            if isinstance(data.get(k), list):
                rows = data[k]
                break
        if not rows:
            rows = [data]
    first = None
    for r in rows:
        if not isinstance(r, dict):
            continue
        num = find_value(r, ("account_number", "account_id"))
        if num is None:
            continue
        first = first or str(num)
        blob = json.dumps(r).lower()
        if "agentic" in blob or "agent" in blob:
            return str(num)
    return first
