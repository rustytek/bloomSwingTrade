from sqlalchemy import (
    Column, Integer, String, Float, Boolean, Date, DateTime, Text,
    ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from database.db import Base


def utcnow():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    password_hash = Column(String(256), nullable=False)
    email = Column(String(128), nullable=True)
    is_admin = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=utcnow)
    last_login = Column(DateTime, nullable=True)

    # Per-user LiteLLM virtual key — keeps each user's AI token usage separate.
    # When null, the AI path falls back to the global key from config.
    litellm_api_key = Column(String(256), nullable=True)

    # Per-user AI system-prompt overrides. When null, the built-in default
    # (services/report_service._SYSTEM_PROMPT / api.ai.DEFAULT_CHAT_SYSTEM_PROMPT)
    # is used. The fixed report _TEMPLATE is never user-editable.
    report_system_prompt = Column(Text, nullable=True)
    chat_system_prompt = Column(Text, nullable=True)

    # Trading settings (fixed-fractional risk model)
    account_size = Column(Float, default=10000, nullable=False)
    risk_pct = Column(Float, default=1.0, nullable=False)
    max_positions = Column(Integer, default=8, nullable=False)
    atr_stop_mult = Column(Float, default=2.5, nullable=False)
    r_multiple = Column(Float, default=2.0, nullable=False)

    # Chosen open-R ceiling. NULL means "not chosen" — consumers then fall back
    # to the DERIVED budget (max_positions x risk_pct, see
    # services/portfolio_risk.implied_max_open_r). The distinction is reported
    # through `budget_basis` so the UI never presents a derived number as a
    # deliberate one.
    max_open_r = Column(Float, nullable=True)

    # Opt-in: let services/edge_matrix.py evidence override the hand-written
    # Strategy.regimes tags on the Today playbook. Ships OFF.
    use_evidence_regimes = Column(Boolean, default=False, nullable=True)

    watchlist = relationship("WatchlistItem", back_populates="user", cascade="all, delete-orphan")
    portfolio = relationship("PortfolioPosition", back_populates="user", cascade="all, delete-orphan")

    @property
    def has_litellm_key(self) -> bool:
        return bool(self.litellm_api_key)


class WatchlistItem(Base):
    __tablename__ = "watchlist_items"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    ticker = Column(String(16), nullable=False)
    added_at = Column(DateTime, default=utcnow)
    notes = Column(Text, nullable=True)

    user = relationship("User", back_populates="watchlist")

    __table_args__ = (UniqueConstraint("user_id", "ticker", name="uq_watchlist_user_ticker"),)


class WatchlistSnapshot(Base):
    __tablename__ = "watchlist_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    week_start = Column(Date, nullable=False, index=True)
    tickers_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=utcnow)
    notes = Column(Text, nullable=True)

    user = relationship("User")

    __table_args__ = (UniqueConstraint("user_id", "week_start", name="uq_watchlist_snapshot_user_week"),)


class PortfolioPosition(Base):
    __tablename__ = "portfolio_positions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    ticker = Column(String(16), nullable=False)
    shares = Column(Float, nullable=False)
    avg_cost = Column(Float, nullable=False)
    added_at = Column(DateTime, default=utcnow)
    notes = Column(Text, nullable=True)
    stop_loss = Column(Float, nullable=True)
    # The FIRST stop ever set on this position. Written once at creation (or the
    # first time a stop is supplied) and never overwritten — R-multiples must be
    # measured against the initial risk, otherwise trailing a stop up silently
    # inflates the recorded R on every trade.
    initial_stop = Column(Float, nullable=True)
    target = Column(Float, nullable=True)
    entry_date = Column(Date, nullable=True)
    strategy = Column(String(32), nullable=True)

    # ── The trade plan's INTENT, captured at commit time ────────────────────
    # These were previously stuffed into `notes` as "THESIS:"/"INVALIDATION:"/
    # "TIME STOP:"/"PLAN: entry <x>" prefixed lines because no columns existed.
    # `planned_entry` is the price the plan said to pay and is WRITE-ONCE (like
    # initial_stop): it is the denominator for the entry-chasing execution
    # metric, so overwriting it on an edit would erase the evidence of a chase.
    planned_entry = Column(Float, nullable=True)
    planned_entry_high = Column(Float, nullable=True)   # top of the plan's entry zone
    thesis = Column(Text, nullable=True)
    invalidation = Column(Text, nullable=True)
    time_stop_days = Column(Integer, nullable=True)

    user = relationship("User", back_populates="portfolio")

    __table_args__ = (UniqueConstraint("user_id", "ticker", name="uq_portfolio_user_ticker"),)


class ClosedTrade(Base):
    """Journal entry created when a portfolio position is closed (sold)."""
    __tablename__ = "closed_trades"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    ticker = Column(String(16), nullable=False)
    shares = Column(Float, nullable=False)
    avg_cost = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=False)
    entry_date = Column(Date, nullable=True)
    exit_date = Column(Date, nullable=True)
    stop_loss = Column(Float, nullable=True)     # stop in force at the moment of exit
    initial_stop = Column(Float, nullable=True)  # stop at entry — the R denominator
    target = Column(Float, nullable=True)
    strategy = Column(String(32), nullable=True)
    pnl = Column(Float, nullable=False)
    pnl_pct = Column(Float, nullable=False)
    r_multiple = Column(Float, nullable=True)   # null when no valid stop was set
    # Carried over from PortfolioPosition on close — the plan's intent, so the
    # journal can compare what was PLANNED against what was actually done.
    planned_entry = Column(Float, nullable=True)
    planned_entry_high = Column(Float, nullable=True)
    thesis = Column(Text, nullable=True)
    invalidation = Column(Text, nullable=True)
    time_stop_days = Column(Integer, nullable=True)
    notes = Column(Text, nullable=True)
    opened_at = Column(DateTime, nullable=True)
    closed_at = Column(DateTime, default=utcnow)

    user = relationship("User")


class StockCache(Base):
    __tablename__ = "stock_cache"

    ticker = Column(String(16), primary_key=True)
    quote_json = Column(Text, nullable=True)
    history_json = Column(Text, nullable=True)
    cached_at = Column(DateTime, default=utcnow)
    quote_cached_at = Column(DateTime, nullable=True)
    history_cached_at = Column(DateTime, nullable=True)


class HistoryArchive(Base):
    """On-demand long-history store for arbitrary-era backtests.

    Separate from the rolling StockCache (which stays current for the live
    screener). Each row holds the widest [start_date, end_date] daily range
    fetched for a ticker; backtests over old eras fetch the missing span once
    here and reuse it thereafter.
    """
    __tablename__ = "history_archive"

    ticker = Column(String(16), primary_key=True)
    bars_json = Column(Text, nullable=False)
    start_date = Column(String(10), nullable=False)   # earliest bar date covered (YYYY-MM-DD)
    end_date = Column(String(10), nullable=False)     # latest bar date covered
    fetched_at = Column(DateTime, default=utcnow)
    # ── 20-year backfill bookkeeping (services/long_history.py) ───────────
    # `requested_start` is how far back we ASKED for. A row whose first bar is
    # later than that is still complete when the ticker simply listed later —
    # that is how a young listing is told apart from an unfinished row.
    source = Column(String(16), nullable=True)        # "yfinance" | "tiingo"
    requested_start = Column(String(10), nullable=True)
    issues = Column(Text, nullable=True)              # JSON list of problems found, if any
    checked_at = Column(DateTime, nullable=True)


class BacktestRun(Base):
    """One row per distinct Strategy Lab configuration a user has tried
    (services/trial_log.py). The count of rows per (user, strategy) is the
    number of trials behind the Deflated Sharpe Ratio — ML4T 3e §7.4/§16.7:
    the best of many tries looks good by luck, so every try is recorded.
    Re-running an identical configuration updates its row instead of adding
    one; rows are never deleted by the app (deleting would hide the search)."""
    __tablename__ = "backtest_runs"
    __table_args__ = (UniqueConstraint("user_id", "strategy", "params_key", name="uq_backtest_run"),)

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    strategy = Column(String(64), nullable=False)
    mode = Column(String(16), nullable=False)
    params_key = Column(String(32), nullable=False)   # md5 of the sorted parameters
    params_json = Column(Text, nullable=False)
    window_start = Column(String(10), nullable=True)
    window_end = Column(String(10), nullable=True)
    periods = Column(Integer, nullable=True)
    sharpe_annual = Column(Float, nullable=True)
    cagr = Column(Float, nullable=True)
    runs = Column(Integer, default=1)                  # times this exact configuration was run
    first_run_at = Column(DateTime, default=utcnow)
    last_run_at = Column(DateTime, default=utcnow)


class AICache(Base):
    __tablename__ = "ai_cache"

    ticker = Column(String(16), primary_key=True)
    analysis_json = Column(Text, nullable=False)
    analyzed_at = Column(DateTime, default=utcnow)


class ReportCache(Base):
    __tablename__ = "report_cache"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    report_markdown = Column(Text, nullable=False)
    generated_at = Column(DateTime, default=utcnow)
    triggered_by = Column(String(16), default="user")   # "user" | "schedule"
    model = Column(String(128), nullable=True)          # resolved model that generated it


class BackgroundJob(Base):
    """A long-running build that must NOT be an HTTP request.

    WHY THIS EXISTS — a real outage, 2026-09-23. `/api/edge-matrix?refresh=true`
    ran one full walk-forward per strategy inside the request. Offloading it to a
    worker thread (anyio.to_thread) stopped it BLOCKING the event loop but not
    starving it: sustained GIL-bound CPU left uvicorn too few cycles to finish a
    TLS handshake, so cloudflared logged `net/http: TLS handshake timeout` for
    EVERY endpoint — `/auth/me` and `/api/today` included — and Cloudflare
    returned 502 for the whole add-on. The Supervisor watchdog then restarted the
    container, which destroyed the in-flight build and wiped the in-memory chart
    cache, so each retry started from zero and failed the same way.

    The fix is that this work runs in a SEPARATE PROCESS (its own GIL — see
    services/job_worker.py) and reports through this table. Two consequences
    matter as much as the offloading itself:
      * the result is PERSISTED, so a restart no longer throws the work away, and
      * progress is readable, so the page can poll instead of holding a socket
        open for minutes.

    Never reintroduce a synchronous HTTP path for work measured in minutes.
    """

    __tablename__ = "background_jobs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    kind = Column(String(48), nullable=False, index=True)   # "edge_matrix" | "strategy_compare"
    # Canonical, sorted JSON of the build parameters. Two requests with the same
    # key are the SAME job — the outage log shows ?refresh=true arriving twice
    # concurrently, doubling the load, because nothing deduped them.
    params_key = Column(String(512), nullable=False, index=True)
    params_json = Column(Text, nullable=False, default="{}")

    # queued -> running -> done | failed | cancelled
    status = Column(String(16), nullable=False, default="queued", index=True)
    progress = Column(Float, nullable=False, default=0.0)      # 0..1
    progress_detail = Column(String(256), nullable=True)       # "momentum_rotation (3/8)"

    result_json = Column(Text, nullable=True)
    error = Column(Text, nullable=True)

    pid = Column(Integer, nullable=True)        # worker pid, for staleness checks
    created_at = Column(DateTime, default=utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    # Written by the worker on every progress tick. A running job whose heartbeat
    # has gone cold was killed (watchdog restart, OOM) and is reaped as failed —
    # otherwise it would sit at "running" forever and block every future refresh.
    heartbeat_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)


class BrokerAccount(Base):
    """A user's link to Robinhood's official agentic-trading MCP server.

    No Robinhood password ever reaches this app: the user logs in on
    robinhood.com (OAuth 2.1 + PKCE) and Robinhood hands back a revocable
    token. Tokens are Fernet-encrypted by services/secrets_box.py and NEVER
    leave the server — the API reports only booleans and masked identifiers.
    `mode` ships "paper": nothing is sent to the broker until the user
    explicitly switches to live with a confirmation.
    """
    __tablename__ = "broker_accounts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True, index=True)
    broker = Column(String(32), nullable=False, default="robinhood")
    # From RFC 7591 dynamic client registration — reused while redirect_uri is unchanged.
    client_id = Column(String(256), nullable=True)
    redirect_uri = Column(String(512), nullable=True)
    access_token_enc = Column(Text, nullable=True)
    refresh_token_enc = Column(Text, nullable=True)
    token_expires_at = Column(DateTime, nullable=True)
    mcp_session_id = Column(String(256), nullable=True)
    tools_json = Column(Text, nullable=True)          # cached tools/list result
    account_number = Column(String(64), nullable=True)
    mode = Column(String(8), nullable=False, default="paper")   # "paper" | "live"
    connected_at = Column(DateTime, nullable=True)
    live_enabled_at = Column(DateTime, nullable=True)
    last_error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)


class BrokerOrder(Base):
    """Audit log of every order the Trade page submitted (or simulated).

    `plan_json` carries the trade plan's intent (stop/target/strategy/thesis…)
    so a fill can be written to the portfolio exactly as a manual commit would.
    `applied_to_portfolio` makes that write happen once, however often the
    fills are synced.
    """
    __tablename__ = "broker_orders"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    broker = Column(String(32), nullable=False, default="robinhood")
    mode = Column(String(8), nullable=False)                     # "paper" | "live"
    plan_item_id = Column(String(64), nullable=True)
    ticker = Column(String(16), nullable=False)
    side = Column(String(8), nullable=False)                     # "buy" | "sell"
    order_type = Column(String(16), nullable=False, default="limit")
    quantity = Column(Float, nullable=False)
    limit_price = Column(Float, nullable=False)
    time_in_force = Column(String(8), nullable=False, default="gfd")
    # simulated | queued | unconfirmed | confirmed | partially_filled | filled
    # | cancelled | rejected | failed
    status = Column(String(24), nullable=False, index=True)
    broker_order_id = Column(String(64), nullable=True, index=True)
    filled_quantity = Column(Float, nullable=True)
    avg_fill_price = Column(Float, nullable=True)
    plan_json = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    response_json = Column(Text, nullable=True)                  # sanitized, trimmed
    applied_to_portfolio = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=utcnow, index=True)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)
