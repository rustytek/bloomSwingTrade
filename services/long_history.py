"""
20-year price history for backtests and the edge matrix.

WHY
---
The rolling StockCache keeps 5 years. After the ~1-year warm-up the strategies
need, that leaves tests starting mid/late 2022 — after most of that bear market —
so the bear-regime cells of the edge matrix were thin by construction. This
module keeps a 20-year archive (the `history_archive` table) and hands the
backtest one continuous series per ticker:

    archive bars BEFORE the cache's first date  +  the live 5-year cache

The cache stays authoritative for the recent part (it is refreshed daily);
the archive only ever supplies the older years, so it does not need daily
maintenance.

REBASING
--------
Prices are split- and dividend-ADJUSTED, and adjusted as of the day they were
downloaded. A split or dividend after the backfill shifts the whole cache
relative to the archive, and a naive join would show a fake jump (a 10:1 split
would look like a 90% crash). splice() measures the ratio between the two on
dates they share and rescales the archive onto the cache's basis. If the ratio
is not consistent across the overlap the sources disagree, and the archive is
NOT used for that ticker (reported, never guessed around).

MEMORY
------
Nothing here loads the universe at once. The backtest asks for one ticker at a
time (services/backtest.py scans per ticker), because 20 years x ~540 tickers
as Python dicts is ~1.25 GB — too much for the Home Assistant host.

The backfill itself runs only in the job worker (services/job_worker.py), never
in the web process.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone

from sqlalchemy.orm import Session

from database.models import HistoryArchive, StockCache
from services import history_sources as src

logger = logging.getLogger(__name__)

LONG_HISTORY_YEARS = 20
VIX_TICKER = "^VIX"
REFERENCE_TICKER = "SPY"

# Pacing between yfinance downloads — polite enough not to get throttled on a
# ~540-ticker run (~5 minutes of sleeping in total).
YF_SPACING_SECONDS = 0.5
# A ticker whose last attempt had problems is retried, but not more than once a day.
RETRY_AFTER = timedelta(hours=20)
# Overlap ratios must agree within this, or the two sources disagree.
MAX_REBASE_MISMATCH = 0.01


def target_start(today: date | None = None) -> str:
    today = today or date.today()
    try:
        return today.replace(year=today.year - LONG_HISTORY_YEARS).isoformat()
    except ValueError:                      # Feb 29
        return today.replace(year=today.year - LONG_HISTORY_YEARS, day=28).isoformat()


# ── splice ──────────────────────────────────────────────────────────────────
def splice(archive: list[dict], cache: list[dict]) -> tuple[list[dict], dict]:
    """archive (older) + cache (recent) on the cache's price basis. Pure.

    Returns (bars, meta). meta: {used_archive_bars, ratio, mismatch, reason}.
    """
    meta = {"used_archive_bars": 0, "ratio": None, "mismatch": None, "reason": None}
    if not archive:
        meta["reason"] = "no archive"
        return cache, meta
    if not cache:
        meta["reason"] = "no cache — archive only"
        meta["used_archive_bars"] = len(archive)
        return archive, meta
    pivot = cache[0]["date"]
    if archive[0]["date"] >= pivot:
        meta["reason"] = "archive adds nothing before the cache"
        return cache, meta
    arch = {b["date"]: b for b in archive if b["date"] >= pivot}
    pairs = [(c, arch[c["date"]]) for c in cache[:60]
             if c["date"] in arch and c.get("close") and arch[c["date"]].get("close")]
    if not pairs:
        meta["reason"] = "archive does not overlap the cache — cannot rebase"
        return cache, meta
    ratios = [float(c["close"]) / float(a["close"]) for c, a in pairs[:10]]
    ratio = ratios[0]
    mismatch = max(abs(r / ratio - 1) for r in ratios)
    meta["ratio"], meta["mismatch"] = round(ratio, 6), round(mismatch, 6)
    if mismatch > MAX_REBASE_MISMATCH:
        meta["reason"] = (f"archive and cache disagree on shared dates (ratio varies "
                          f"{mismatch * 100:.1f}%) — archive not used")
        return cache, meta
    split_like = ratio < 0.67 or ratio > 1.5
    older = []
    for b in archive:
        if b["date"] >= pivot:
            break
        nb = dict(b)
        for k in ("open", "high", "low", "close"):
            if nb.get(k) is not None:
                nb[k] = round(float(nb[k]) * ratio, 4)
        if split_like and nb.get("vol"):
            nb["vol"] = int(round(float(nb["vol"]) / ratio))
        older.append(nb)
    meta["used_archive_bars"] = len(older)
    return older + cache, meta


# ── loading ─────────────────────────────────────────────────────────────────
def _json_bars(text: str | None) -> list[dict]:
    if not text:
        return []
    try:
        bars = json.loads(text)
    except (TypeError, ValueError):
        return []
    return bars if isinstance(bars, list) else []


def load_cache_bars(db: Session, ticker: str) -> list[dict]:
    row = db.query(StockCache.history_json).filter(StockCache.ticker == ticker).first()
    return _json_bars(row[0] if row else None)


def load_archive_bars(db: Session, ticker: str) -> list[dict]:
    row = db.query(HistoryArchive.bars_json).filter(HistoryArchive.ticker == ticker).first()
    return _json_bars(row[0] if row else None)


def load_long_bars(db: Session, ticker: str,
                   trim_before: tuple[str, int] | None = None) -> tuple[list[dict], dict]:
    """One ticker's archive+cache series (raw bar dicts) and the splice meta.

    `trim_before=(date, keep)` drops archive bars more than `keep` bars before
    `date` BEFORE splicing. Archive bars after that point (including the
    overlap with the cache used for rebasing) are kept, so the result from
    that point on is identical to splicing everything and trimming after."""
    from bisect import bisect_left
    cache = load_cache_bars(db, ticker)
    if trim_before and cache:
        start, keep = trim_before
        if bisect_left([b["date"] for b in cache], start) - keep >= 0:
            # The cache alone reaches back far enough (warm-up included): the
            # archive would be trimmed away entirely, so don't even parse it.
            return cache, {"used_archive_bars": 0, "ratio": None, "mismatch": None,
                           "reason": "cache covers the window"}
    archive = load_archive_bars(db, ticker)
    if trim_before and archive:
        start, keep = trim_before
        cut = bisect_left([b["date"] for b in archive], start) - keep
        if cut > 0:
            archive = archive[cut:]
    return splice(archive, cache)


def load_vix_by_date(db: Session) -> dict[str, float]:
    """date -> VIX close, from the archive (+ cache if it has ^VIX). Used by the
    backtest's regime tagging in place of the realized-volatility proxy."""
    bars, _ = load_long_bars(db, VIX_TICKER)
    out = {}
    for b in bars:
        try:
            out[b["date"]] = float(b["close"])
        except (KeyError, TypeError, ValueError):
            continue
    return out


# ── status ──────────────────────────────────────────────────────────────────
def is_complete(row: HistoryArchive | None, want_start: str) -> bool:
    """Asked from far enough back, and nothing wrong with the result."""
    if row is None or not row.requested_start:
        return False
    slack = (date.fromisoformat(want_start) + timedelta(days=5)).isoformat()
    return row.requested_start <= slack and not _issues(row)


def _issues(row: HistoryArchive) -> list[str]:
    try:
        v = json.loads(row.issues) if row.issues else []
    except (TypeError, ValueError):
        return ["unreadable issues field"]
    return v if isinstance(v, list) else []


def backfill_tickers(db: Session) -> list[str]:
    """SPY and VIX first (SPY is the trading calendar every other download is
    checked against), then the universe plus every watchlist ticker."""
    from database.models import WatchlistItem
    from services.universe import UNIVERSE
    rest = set(UNIVERSE)
    rest.update(t for (t,) in db.query(WatchlistItem.ticker).distinct().all() if t)
    rest.discard(REFERENCE_TICKER)
    rest.discard(VIX_TICKER)
    return [REFERENCE_TICKER, VIX_TICKER] + sorted(rest)


def status(db: Session) -> dict:
    """Coverage summary for the Strategy Lab. Reads only small columns."""
    want = target_start()
    tickers = backfill_tickers(db)
    rows = {r.ticker: r for r in db.query(
        HistoryArchive.ticker, HistoryArchive.start_date, HistoryArchive.end_date,
        HistoryArchive.source, HistoryArchive.requested_start, HistoryArchive.issues,
        HistoryArchive.checked_at).all()}
    complete = partial = missing = 0
    sources: dict[str, int] = {}
    problems = []
    for t in tickers:
        r = rows.get(t)
        if r is None:
            missing += 1
            continue
        sources[r.source or "unknown"] = sources.get(r.source or "unknown", 0) + 1
        if is_complete(r, want):
            complete += 1
        else:
            partial += 1
            iss = _issues(r)
            if iss and len(problems) < 30:
                problems.append({"ticker": t, "issues": iss, "start": r.start_date, "source": r.source})
    spy, vix = rows.get(REFERENCE_TICKER), rows.get(VIX_TICKER)
    return {
        "target_years": LONG_HISTORY_YEARS,
        "target_start": want,
        "tickers_total": len(tickers),
        "complete": complete,
        "partial": partial,
        "missing": missing,
        "sources": sources,
        "spy_start": spy.start_date if spy else None,
        "vix_start": vix.start_date if vix else None,
        "problems": problems,
        "ready": bool(spy is not None and is_complete(spy, want)),
    }


# ── backfill (job worker only) ──────────────────────────────────────────────
def _store(db: Session, ticker: str, bars: list[dict], source: str | None,
           requested_start: str, issues: list[str]) -> None:
    row = db.query(HistoryArchive).filter(HistoryArchive.ticker == ticker).first()
    now = datetime.now(timezone.utc)
    if bars:
        old = _json_bars(row.bars_json) if row else []
        # Never replace a longer stored series with a shorter download.
        if old and (bars[0]["date"] > old[0]["date"] and bars[-1]["date"] <= old[-1]["date"]):
            bars = old
            issues = issues + ["kept the previously stored, longer series"]
    if row is None:
        row = HistoryArchive(ticker=ticker, bars_json="[]", start_date="", end_date="")
        db.add(row)
    if bars:
        row.bars_json = json.dumps(bars)
        row.start_date, row.end_date = bars[0]["date"], bars[-1]["date"]
        row.fetched_at = now
        row.source = source
    row.requested_start = requested_start
    row.issues = json.dumps(issues) if issues else None
    row.checked_at = now
    db.commit()


def run_backfill(db: Session, *, tiingo_key: str | None = None, force: bool = False,
                 progress=None, today: date | None = None, tickers: list[str] | None = None) -> dict:
    """Download 20 years for every ticker that does not have it yet.

    Resumable: complete tickers are skipped, so a run that was interrupted (or
    hit Tiingo's hourly cap) simply continues next time. yfinance is tried
    first; Tiingo is used when the yfinance download is empty, truncated,
    stale or holed and a TIINGO_API_KEY is configured.
    """
    today = today or date.today()
    want = target_start(today)
    today_s = today.isoformat()
    todo = tickers or backfill_tickers(db)

    def tick(frac, detail):
        if progress:
            try:
                progress(frac, detail)
            except Exception:  # noqa: BLE001
                pass

    calendar: list[str] | None = None
    counts = {"done": 0, "skipped": 0, "yfinance": 0, "tiingo": 0, "failed": 0, "with_issues": 0}
    tiingo_calls = 0
    tiingo_blocked = None
    failures: list[dict] = []
    for n, t in enumerate(todo):
        tick(0.02 + 0.96 * n / max(1, len(todo)), f"{t} ({n + 1}/{len(todo)})")
        row = db.query(HistoryArchive).filter(HistoryArchive.ticker == t).first()
        if t == REFERENCE_TICKER and row is not None and row.bars_json:
            calendar = [b["date"] for b in _json_bars(row.bars_json)]
        # VIX is not in the daily cache, so its archive IS its only source —
        # refresh it whenever it has fallen behind instead of skipping it.
        vix_stale = (t == VIX_TICKER and row is not None and row.end_date
                     and row.end_date < (today - timedelta(days=3)).isoformat())
        if not force and not vix_stale and is_complete(row, want):
            counts["skipped"] += 1
            continue
        if not force and row is not None and row.checked_at is not None:
            checked = row.checked_at if row.checked_at.tzinfo else row.checked_at.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) - checked < RETRY_AFTER and _issues(row):
                counts["skipped"] += 1
                continue

        cache = load_cache_bars(db, t)
        known_first = cache[0]["date"] if cache else None
        cal = None if t in (REFERENCE_TICKER, VIX_TICKER) else calendar

        yf_bars = src.fetch_yfinance(t, want, today_s)
        src.pause(YF_SPACING_SECONDS)
        best = (yf_bars, src.assess(yf_bars, target_start=want, today=today_s,
                                    known_first=known_first, calendar=cal))
        chosen_source = "yfinance" if yf_bars else None

        if not best[1]["ok"] and tiingo_key and src.tiingo_symbol(t) and tiingo_blocked is None:
            if tiingo_calls >= src.TIINGO_MAX_PER_RUN:
                tiingo_blocked = (f"Tiingo fallback paused after {tiingo_calls} calls this run "
                                  "(free tier: 50/hour). Run the backfill again later to continue.")
            else:
                tiingo_calls += 1
                try:
                    tg = src.fetch_tiingo(t, want, today_s, tiingo_key)
                except src.TiingoRateLimited as exc:
                    tg, tiingo_blocked = [], str(exc)
                src.pause(src.TIINGO_SPACING_SECONDS)
                if tg:
                    cand = (tg, src.assess(tg, target_start=want, today=today_s,
                                           known_first=known_first, calendar=cal))
                    picked = src.better(best, cand)
                    if picked is cand:
                        best, chosen_source = cand, "tiingo"

        bars, verdict = best
        issues = list(verdict["issues"])
        _store(db, t, bars, chosen_source if bars else None, want, issues)
        if t == REFERENCE_TICKER and bars:
            calendar = [b["date"] for b in bars]
        if bars:
            counts["done"] += 1
            counts[chosen_source] += 1
            if issues:
                counts["with_issues"] += 1
        else:
            counts["failed"] += 1
        if issues and len(failures) < 40:
            failures.append({"ticker": t, "source": chosen_source, "issues": issues})

    tick(0.99, "Finishing…")
    return {
        "target_start": want,
        "tickers": len(todo),
        "counts": counts,
        "tiingo_configured": bool(tiingo_key),
        "tiingo_calls": tiingo_calls,
        "tiingo_note": tiingo_blocked,
        "problems": failures,
    }
