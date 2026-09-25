"""
Portfolio-level risk: correlation, open risk, concentration, heat, pre-trade checks.

Why this module exists
----------------------
Position sizing and correlation move P&L more than the choice of entry strategy.
Four "independent" trades that are all 0.8-correlated Technology names with a
weighted beta of 1.4 are one leveraged bet, not four trades. Everything here is
pure-function and DB-free so it can be unit-tested without network or SQLite;
`api/portfolio.py` does the loading and calls in.

Key modelling decisions
-----------------------
* **Correlation is computed on DAILY RETURNS, never on price levels.** Any two
  rising series correlate near 1.0 on levels regardless of whether they actually
  move together — that number is meaningless for diversification.
* **Series are aligned by DATE, not by list index.** Tickers have different
  history lengths and listing dates; zipping raw lists silently compares
  Monday's AAPL against Thursday's NVDA.
* **A pair with fewer than `MIN_OVERLAP` shared return observations returns
  `None`**, not a number. A 4-bar correlation is noise, and noise presented as
  0.93 is worse than an honest "unknown".
* **A position with no stop has UNBOUNDED risk.** It is never counted as zero
  risk; it is reported separately in `positions_without_stop` so the UI can
  shout about it.
* **Risk budget (`max_open_r`) is implicit: `max_positions x risk_pct`.** There
  is no dedicated user setting for it yet; see `DEFAULT_MAX_OPEN_R_NOTE`.

Every numeric path is NaN-guarded: cached quotes can carry NaN floats and
Starlette serializes with `allow_nan=False`, so a single NaN 500s the response.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

# ── Documented thresholds ────────────────────────────────────────────────────
MIN_OVERLAP = 30           # minimum shared daily-return observations for a pair
DEFAULT_WINDOW = 90        # trailing daily returns used for correlation

CORR_WARN = 0.70           # "these two are the same trade-ish"
CORR_BLOCK = 0.85          # "these two ARE the same trade"

SECTOR_WARN_PCT = 30.0     # sector weight of portfolio market value
SECTOR_BLOCK_PCT = 40.0

HEAT_WARN_PCT = 80.0       # % of the open-R budget consumed

DEFAULT_MAX_OPEN_R_NOTE = (
    "max_open_r is derived as max_positions x risk_pct (every slot filled at full "
    "per-trade risk). A dedicated user setting would be better long-term."
)


# ── numeric guards ───────────────────────────────────────────────────────────
def _num(v, default=None):
    """Coerce to a finite float, else `default`. Rejects None, NaN, inf, junk."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def _round(v, ndigits=2):
    f = _num(v)
    return None if f is None else round(f, ndigits)


def _as_quote_map(quotes) -> dict[str, dict]:
    """Accept either {ticker: quote} or a list of quote dicts."""
    if isinstance(quotes, dict):
        return {str(k).upper(): (v or {}) for k, v in quotes.items()}
    out: dict[str, dict] = {}
    for q in quotes or []:
        if isinstance(q, dict) and q.get("ticker"):
            out[str(q["ticker"]).upper()] = q
    return out


def _pos_get(pos, key, default=None):
    """Read a field from either a dict or an ORM PortfolioPosition."""
    if isinstance(pos, dict):
        return pos.get(key, default)
    return getattr(pos, key, default)


def _position_view(pos, quote_map: dict[str, dict]) -> dict | None:
    """Normalize one position + its quote into a plain dict of finite numbers."""
    ticker = _pos_get(pos, "ticker")
    if not ticker:
        return None
    ticker = str(ticker).upper()
    shares = _num(_pos_get(pos, "shares"), 0.0) or 0.0
    avg_cost = _num(_pos_get(pos, "avg_cost"), 0.0) or 0.0
    q = quote_map.get(ticker, {}) or {}
    price = _num(q.get("price"))
    if price is None or price <= 0:
        price = avg_cost if avg_cost > 0 else None
    stop = _num(_pos_get(pos, "stop_loss"))
    sector = q.get("sector") or "Unknown"
    beta = _num(q.get("beta"))
    return {
        "ticker": ticker,
        "shares": shares,
        "avg_cost": avg_cost,
        "price": price,
        "stop": stop,
        # Alias so a view can be fed back through _views() unchanged — the
        # before/after projection in assess_new_position does exactly that.
        "stop_loss": stop,
        "sector": str(sector),
        "beta": beta,
        "market_value": (shares * price) if (price is not None) else (shares * avg_cost),
    }


def _views(positions, quotes) -> list[dict]:
    qm = _as_quote_map(quotes)
    out = []
    for p in positions or []:
        v = _position_view(p, qm)
        if v is not None:
            out.append(v)
    return out


# ── correlation ──────────────────────────────────────────────────────────────
def daily_returns(bars: Sequence[dict]) -> dict[str, float]:
    """Map `date -> simple daily return` from OHLCV bars.

    Bars are `{date, open, high, low, close, vol}` as stored in
    `StockCache.history_json`. Closes are yfinance `auto_adjust=True` closes, so
    these are total returns and are comparable across tickers. Bars are sorted by
    date defensively; non-finite or non-positive closes are dropped, and the
    return is keyed by the LATER of the two dates it spans.
    """
    clean: list[tuple[str, float]] = []
    for b in bars or []:
        if not isinstance(b, dict):
            continue
        d = b.get("date")
        c = _num(b.get("close"))
        if not d or c is None or c <= 0:
            continue
        clean.append((str(d), c))
    clean.sort(key=lambda x: x[0])

    rets: dict[str, float] = {}
    for i in range(1, len(clean)):
        prev_c = clean[i - 1][1]
        cur_d, cur_c = clean[i]
        if prev_c <= 0:
            continue
        r = cur_c / prev_c - 1.0
        if math.isfinite(r):
            rets[cur_d] = r
    return rets


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2:
        return None
    a = np.asarray(xs, dtype=float)
    b = np.asarray(ys, dtype=float)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return None
    if a.std() == 0 or b.std() == 0:
        return None          # a flat series has no correlation, not 0 and not 1
    with np.errstate(invalid="ignore", divide="ignore"):
        c = float(np.corrcoef(a, b)[0, 1])
    if not math.isfinite(c):
        return None
    return max(-1.0, min(1.0, c))


def pair_correlation(
    returns_a: dict[str, float],
    returns_b: dict[str, float],
    window: int = DEFAULT_WINDOW,
    min_overlap: int = MIN_OVERLAP,
) -> float | None:
    """Pearson correlation of two date-keyed daily-return series.

    Aligned on shared DATES, restricted to the most recent `window` shared
    dates. Returns None when fewer than `min_overlap` dates overlap.
    """
    shared = sorted(set(returns_a) & set(returns_b))
    if len(shared) < max(2, min_overlap):
        return None
    shared = shared[-window:] if window and window > 0 else shared
    if len(shared) < max(2, min_overlap):
        return None
    return _pearson([returns_a[d] for d in shared], [returns_b[d] for d in shared])


def correlation_matrix(
    histories: dict[str, list[dict]],
    window: int = DEFAULT_WINDOW,
    min_overlap: int = MIN_OVERLAP,
) -> dict:
    """Full pairwise correlation matrix of DAILY RETURNS for the given tickers.

    `histories` maps ticker -> OHLCV bar list. Returns:
        {
          "tickers": [...],
          "window": int, "min_overlap": int,
          "matrix": {a: {b: float | None}},     # self-correlation is 1.0
          "pairs":  [{"a","b","corr"}, ...],    # sorted by |corr| desc
          "high_pairs": [...],                  # corr >= CORR_WARN
          "insufficient": [[a, b], ...],        # pairs below min_overlap
        }
    """
    tickers = sorted({str(t).upper() for t in (histories or {})})
    rets = {t: daily_returns((histories or {}).get(t) or (histories or {}).get(t.lower()) or [])
            for t in tickers}
    # histories keys may not be uppercase — rebuild defensively
    for k, v in (histories or {}).items():
        ku = str(k).upper()
        if ku in rets and not rets[ku]:
            rets[ku] = daily_returns(v or [])

    matrix: dict[str, dict[str, float | None]] = {a: {} for a in tickers}
    pairs: list[dict] = []
    insufficient: list[list[str]] = []

    for i, a in enumerate(tickers):
        matrix[a][a] = 1.0
        for b in tickers[i + 1:]:
            c = pair_correlation(rets[a], rets[b], window=window, min_overlap=min_overlap)
            matrix[a][b] = None if c is None else round(c, 4)
            matrix[b][a] = matrix[a][b]
            if c is None:
                insufficient.append([a, b])
            else:
                pairs.append({"a": a, "b": b, "corr": round(c, 4)})

    pairs.sort(key=lambda p: abs(p["corr"]), reverse=True)
    return {
        "tickers": tickers,
        "window": window,
        "min_overlap": min_overlap,
        "matrix": matrix,
        "pairs": pairs,
        "high_pairs": [p for p in pairs if p["corr"] >= CORR_WARN],
        "insufficient": insufficient,
    }


# ── open risk ────────────────────────────────────────────────────────────────
def open_risk(positions, quotes, risk_unit: float | None = None) -> dict:
    """Total dollars currently at risk across open positions.

    Per position: `shares x (current_price - stop)`, floored at 0 — once the stop
    sits at or above the current price the trade is a free roll and carries no
    downside. A position with NO stop has unbounded risk: it contributes nothing
    to `open_risk_dollars` (we cannot know the number) but is listed in
    `positions_without_stop`, which callers must surface loudly.

    `risk_unit` is the dollar value of 1R (account_size x risk_pct/100). When
    supplied, `open_r` is the risk expressed in R; otherwise `open_r` is None.
    """
    views = _views(positions, quotes)
    unit = _num(risk_unit)
    if unit is not None and unit <= 0:
        unit = None

    details = []
    total = 0.0
    at_risk = 0
    no_stop: list[str] = []

    for v in views:
        if v["stop"] is None or v["price"] is None:
            if v["stop"] is None:
                no_stop.append(v["ticker"])
            details.append({
                "ticker": v["ticker"], "shares": v["shares"], "price": v["price"],
                "stop": v["stop"], "risk_dollars": None, "risk_r": None,
                "unbounded": v["stop"] is None,
            })
            continue
        risk = max(0.0, v["shares"] * (v["price"] - v["stop"]))
        total += risk
        if risk > 0:
            at_risk += 1
        details.append({
            "ticker": v["ticker"], "shares": v["shares"], "price": _round(v["price"]),
            "stop": _round(v["stop"]), "risk_dollars": round(risk, 2),
            "risk_r": round(risk / unit, 2) if unit else None,
            "unbounded": False,
            "free_roll": risk == 0.0,
        })

    return {
        "open_risk_dollars": round(total, 2),
        "open_r": round(total / unit, 2) if unit else None,
        "risk_unit": round(unit, 2) if unit else None,
        "positions_at_risk": at_risk,
        "positions_without_stop": no_stop,
        "unstopped_count": len(no_stop),
        "unstopped_warning": (
            f"{len(no_stop)} position(s) have NO stop — their downside is unbounded and is "
            f"NOT included in open risk: {', '.join(no_stop)}"
        ) if no_stop else None,
        "positions": details,
    }


# ── concentration ────────────────────────────────────────────────────────────
def concentration(positions, quotes, sector_threshold_pct: float = SECTOR_WARN_PCT) -> dict:
    """Sector weights and market-value-weighted portfolio beta.

    Weights are of total portfolio MARKET VALUE (not account equity — cash is not
    modelled here). Beta is weighted only over positions that actually have a
    finite beta; `beta_coverage_pct` says how much of the book that covered.
    """
    views = _views(positions, quotes)
    total_mv = sum(v["market_value"] or 0.0 for v in views)
    sectors: dict[str, float] = {}
    beta_num = 0.0
    beta_mv = 0.0
    largest = None

    for v in views:
        mv = v["market_value"] or 0.0
        sectors[v["sector"]] = sectors.get(v["sector"], 0.0) + mv
        if v["beta"] is not None:
            beta_num += v["beta"] * mv
            beta_mv += mv
        if largest is None or mv > largest[1]:
            largest = (v["ticker"], mv)

    weights = {
        s: round(mv / total_mv * 100.0, 2) if total_mv > 0 else 0.0
        for s, mv in sorted(sectors.items(), key=lambda kv: kv[1], reverse=True)
    }
    flagged = [{"sector": s, "weight_pct": w} for s, w in weights.items()
               if w >= sector_threshold_pct]

    return {
        "total_market_value": round(total_mv, 2),
        "position_count": len(views),
        "sector_weights": weights,
        "sector_threshold_pct": sector_threshold_pct,
        "flagged_sectors": flagged,
        "weighted_beta": round(beta_num / beta_mv, 2) if beta_mv > 0 else None,
        "beta_coverage_pct": round(beta_mv / total_mv * 100.0, 1) if total_mv > 0 else 0.0,
        "largest_position": (
            {"ticker": largest[0],
             "weight_pct": round(largest[1] / total_mv * 100.0, 2) if total_mv > 0 else 0.0}
            if largest else None
        ),
    }


# ── heat ─────────────────────────────────────────────────────────────────────
def implied_max_open_r(max_positions: int | float | None, risk_pct: float | None) -> float | None:
    """The implicit open-R budget: every slot filled at full per-trade risk.

    There is no `max_open_r` user setting (no DB column), so the budget is
    derived: `max_positions x risk_pct` R-units of exposure. See
    DEFAULT_MAX_OPEN_R_NOTE.
    """
    mp = _num(max_positions)
    rp = _num(risk_pct)
    if mp is None or rp is None or mp <= 0 or rp <= 0:
        return None
    return round(mp * rp, 4)


def resolve_max_open_r(user) -> tuple[float | None, str]:
    """The open-R budget actually in force, plus an honest basis string.

    A CHOSEN ceiling (`user.max_open_r`) wins. When it is NULL the budget falls
    back to the derived `max_positions x risk_pct` — the exposure of a fully
    loaded book at full per-trade risk, which is a ceiling nobody picked. The
    returned string says which of the two it is, so the UI can stop labelling a
    derived number as a decision.

    Lives here (not in api/settings.py) so every caller — api/portfolio,
    services/weekly_plan — shares one implementation without a services->api
    import. `user` is duck-typed (max_open_r/max_positions/risk_pct) to keep
    this module DB-free.
    """
    chosen = getattr(user, "max_open_r", None)
    try:
        chosen = float(chosen) if chosen is not None else None
    except (TypeError, ValueError):
        chosen = None
    if chosen is not None and chosen > 0:
        return round(chosen, 4), (
            f"max_open_r is a chosen ceiling of {round(chosen, 4):g}R (Settings), "
            "not derived from max_positions x risk_pct."
        )
    derived = implied_max_open_r(
        getattr(user, "max_positions", None), getattr(user, "risk_pct", None)
    )
    # The note still says a dedicated setting "would be better" — it now
    # exists, so point at it rather than editing anything else.
    return derived, (
        DEFAULT_MAX_OPEN_R_NOTE
        + " Set `max_open_r` in Settings to choose a ceiling instead."
    )


def portfolio_heat(positions, quotes, max_open_r: float | None,
                   risk_unit: float | None = None,
                   max_positions: int | None = None) -> dict:
    """Open R against the budget, and slots used against max_positions."""
    risk = open_risk(positions, quotes, risk_unit=risk_unit)
    budget = _num(max_open_r)
    open_r = risk["open_r"]
    heat_pct = (
        round(open_r / budget * 100.0, 1)
        if (budget and budget > 0 and open_r is not None) else None
    )
    slots = len(_views(positions, quotes))
    mp = _num(max_positions)
    return {
        "open_r": open_r,
        "open_risk_dollars": risk["open_risk_dollars"],
        "max_open_r": budget,
        "heat_pct": heat_pct,
        "over_budget": bool(budget and open_r is not None and open_r > budget),
        "slots_used": slots,
        "max_positions": int(mp) if mp else None,
        "slots_available": max(0, int(mp) - slots) if mp else None,
        "at_max_positions": bool(mp and slots >= int(mp)),
        "positions_without_stop": risk["positions_without_stop"],
        "budget_basis": DEFAULT_MAX_OPEN_R_NOTE,
    }


# ── pre-trade assessment ─────────────────────────────────────────────────────
def _warn(level: str, code: str, message: str, **extra) -> dict:
    w = {"level": level, "code": code, "message": message}
    w.update(extra)
    return w


def assess_new_position(ticker: str, plan: dict, positions, quotes,
                        histories: dict[str, list[dict]] | None = None,
                        settings: dict | None = None) -> dict:
    """Pre-trade check for adding `ticker` sized per `plan`.

    `plan` is a `services.trade_plan.build_trade_plan` result (needs at minimum
    `entry`, `stop`, `shares`). `settings` carries `account_size`, `risk_pct`,
    `max_positions`, and optionally `max_open_r` (defaults to the implied
    `max_positions x risk_pct` budget).

    Returns `{ok, warnings, before, after, ...}` where each warning is
    `{level: "block"|"warn"|"info", code, message}`. Thresholds:
      correlation  >= 0.85 block, >= 0.70 warn
      sector wt    >= 40%  block, >= 30%  warn   (AFTER the add)
      open R       >  budget block, >= 80% of budget warn
      book already at max_positions -> block
    Nothing here refuses the trade on its own — it hands the UI a structured
    verdict to render before -> after.
    """
    ticker = str(ticker or "").upper()
    settings = settings or {}
    plan = plan or {}
    histories = histories or {}

    account_size = _num(settings.get("account_size"), 0.0) or 0.0
    risk_pct = _num(settings.get("risk_pct"), 0.0) or 0.0
    max_positions = _num(settings.get("max_positions"))
    risk_unit = account_size * risk_pct / 100.0 if account_size > 0 and risk_pct > 0 else None
    budget = _num(settings.get("max_open_r")) or implied_max_open_r(max_positions, risk_pct)

    qm = _as_quote_map(quotes)
    held = [v for v in _views(positions, qm) if v["ticker"] != ticker]
    already_held = any(v["ticker"] == ticker for v in _views(positions, qm))

    entry = _num(plan.get("entry"))
    stop = _num(plan.get("stop"))
    shares = _num(plan.get("shares"), 0.0) or 0.0
    new_mv = (entry * shares) if (entry is not None) else 0.0
    new_risk = max(0.0, shares * (entry - stop)) if (entry is not None and stop is not None) else None

    # ── before state ───────────────────────────────────────────────────────
    before_conc = concentration(held, qm)
    before_heat = portfolio_heat(held, qm, budget, risk_unit=risk_unit,
                                 max_positions=int(max_positions) if max_positions else None)

    # ── after state: synthesize the candidate as a pseudo-position ─────────
    new_q = qm.get(ticker, {}) or {}
    candidate = {
        "ticker": ticker, "shares": shares,
        "avg_cost": entry if entry is not None else 0.0,
        "stop_loss": stop,
    }
    after_quotes = dict(qm)
    after_quotes[ticker] = {
        "ticker": ticker,
        "price": entry if entry is not None else new_q.get("price"),
        "sector": new_q.get("sector") or "Unknown",
        "beta": new_q.get("beta"),
    }
    after_positions = list(held) + [candidate]
    after_conc = concentration(after_positions, after_quotes)
    after_heat = portfolio_heat(after_positions, after_quotes, budget, risk_unit=risk_unit,
                                max_positions=int(max_positions) if max_positions else None)

    warnings: list[dict] = []

    # 1. slots
    if max_positions and len(held) >= int(max_positions):
        warnings.append(_warn(
            "block", "max_positions",
            f"Book is already at max_positions ({len(held)}/{int(max_positions)}). "
            f"Close something before adding {ticker}.",
            slots_used=len(held), max_positions=int(max_positions),
        ))

    if already_held:
        warnings.append(_warn(
            "info", "already_held",
            f"{ticker} is already held — this assessment treats the plan as a replacement, "
            f"not an addition.",
        ))

    # 2. correlation with held names
    cand_rets = daily_returns(histories.get(ticker) or histories.get(ticker.lower()) or [])
    corr_rows = []
    if cand_rets:
        for v in held:
            other = histories.get(v["ticker"]) or histories.get(v["ticker"].lower()) or []
            c = pair_correlation(cand_rets, daily_returns(other))
            corr_rows.append({"ticker": v["ticker"],
                              "corr": None if c is None else round(c, 4)})
            if c is None:
                continue
            if c >= CORR_BLOCK:
                warnings.append(_warn(
                    "block", "correlation",
                    f"{ticker} is {c:.2f}-correlated with {v['ticker']} (>= {CORR_BLOCK:.2f}) — "
                    f"that is the same trade twice, not diversification.",
                    ticker=v["ticker"], corr=round(c, 4), threshold=CORR_BLOCK,
                ))
            elif c >= CORR_WARN:
                warnings.append(_warn(
                    "warn", "correlation",
                    f"{ticker} is {c:.2f}-correlated with {v['ticker']} (>= {CORR_WARN:.2f}) — "
                    f"sizing both at full risk doubles one bet.",
                    ticker=v["ticker"], corr=round(c, 4), threshold=CORR_WARN,
                ))
    else:
        warnings.append(_warn(
            "info", "no_history",
            f"No usable price history for {ticker} — correlation with held names was not checked.",
        ))
    corr_rows.sort(key=lambda r: (r["corr"] is None, -(r["corr"] or 0)))

    # 3. sector weight after the add
    new_sector = after_quotes[ticker].get("sector") or "Unknown"
    after_w = after_conc["sector_weights"].get(new_sector, 0.0)
    before_w = before_conc["sector_weights"].get(new_sector, 0.0)
    # Measure against the ACCOUNT, not just the invested book, while the book is
    # smaller than the account. Otherwise the first position in an empty book is
    # always "100% of the book" and blocks — as does a second name in a new
    # sector — so a small or new account could never open a trade at all.
    after_total = _num(after_conc.get("total_market_value"), 0.0) or 0.0
    before_total = _num(before_conc.get("total_market_value"), 0.0) or 0.0
    if account_size > after_total > 0:
        after_w = round(after_w * after_total / account_size, 2)
    if account_size > before_total > 0:
        before_w = round(before_w * before_total / account_size, 2)
    if after_w >= SECTOR_BLOCK_PCT:
        warnings.append(_warn(
            "block", "sector_concentration",
            f"{new_sector} would be {after_w:.1f}% of the book after this add "
            f"(was {before_w:.1f}%, limit {SECTOR_BLOCK_PCT:.0f}%).",
            sector=new_sector, before_pct=before_w, after_pct=after_w,
            threshold=SECTOR_BLOCK_PCT,
        ))
    elif after_w >= SECTOR_WARN_PCT:
        warnings.append(_warn(
            "warn", "sector_concentration",
            f"{new_sector} would be {after_w:.1f}% of the book after this add "
            f"(was {before_w:.1f}%, soft limit {SECTOR_WARN_PCT:.0f}%).",
            sector=new_sector, before_pct=before_w, after_pct=after_w,
            threshold=SECTOR_WARN_PCT,
        ))

    # 4. open R after the add
    after_r = after_heat["open_r"]
    if budget and after_r is not None:
        if after_r > budget:
            warnings.append(_warn(
                "block", "open_risk_budget",
                f"Open risk would be {after_r:.2f}R against a {budget:.2f}R budget "
                f"(max_positions x risk_pct).",
                after_open_r=after_r, max_open_r=budget,
            ))
        elif after_r >= budget * HEAT_WARN_PCT / 100.0:
            warnings.append(_warn(
                "warn", "open_risk_budget",
                f"Open risk would be {after_r:.2f}R — {after_r / budget * 100:.0f}% of the "
                f"{budget:.2f}R budget.",
                after_open_r=after_r, max_open_r=budget,
            ))

    # 5. no stop on the plan itself
    if stop is None:
        warnings.append(_warn(
            "block", "no_stop",
            f"The plan for {ticker} has no stop — risk is unbounded and cannot be sized.",
        ))

    # 6. unstopped positions already in the book
    if before_heat["positions_without_stop"]:
        names = ", ".join(before_heat["positions_without_stop"])
        warnings.append(_warn(
            "warn", "unstopped_positions",
            f"Existing positions with NO stop carry unbounded risk that is not in these "
            f"numbers: {names}.",
            tickers=before_heat["positions_without_stop"],
        ))

    blocks = [w for w in warnings if w["level"] == "block"]
    warns = [w for w in warnings if w["level"] == "warn"]
    return {
        "ticker": ticker,
        "ok": not blocks,
        "verdict": "block" if blocks else ("warn" if warns else "ok"),
        "warnings": warnings,
        "block_count": len(blocks),
        "warn_count": len(warns),
        "plan": plan,
        "new_position_value": round(new_mv, 2),
        "new_position_risk_dollars": round(new_risk, 2) if new_risk is not None else None,
        "correlations": corr_rows,
        "thresholds": {
            "corr_warn": CORR_WARN, "corr_block": CORR_BLOCK,
            "sector_warn_pct": SECTOR_WARN_PCT, "sector_block_pct": SECTOR_BLOCK_PCT,
            "heat_warn_pct": HEAT_WARN_PCT, "min_overlap": MIN_OVERLAP,
        },
        "before": {
            "sector_weights": before_conc["sector_weights"],
            "weighted_beta": before_conc["weighted_beta"],
            "open_r": before_heat["open_r"],
            "open_risk_dollars": before_heat["open_risk_dollars"],
            "heat_pct": before_heat["heat_pct"],
            "slots_used": before_heat["slots_used"],
        },
        "after": {
            "sector_weights": after_conc["sector_weights"],
            "weighted_beta": after_conc["weighted_beta"],
            "open_r": after_heat["open_r"],
            "open_risk_dollars": after_heat["open_risk_dollars"],
            "heat_pct": after_heat["heat_pct"],
            "slots_used": after_heat["slots_used"],
        },
        "max_open_r": budget,
        "budget_basis": DEFAULT_MAX_OPEN_R_NOTE,
    }
