"""
Price-bar storage: compact columnar encoding, data versions and revisions.
ML4T 3e, Ch 2 (data governance).

WHY
---
Every stored series used to be a JSON list of dicts in a TEXT column — ~370 MB
for the 20-year archive plus the 5-year cache, parsed in full on every load,
and OVERWRITTEN on every refetch, so a backtest could not be reproduced once
Yahoo re-adjusted a series (every dividend does) and nobody could tell that
the history under a result had changed.

THREE THINGS LIVE HERE
----------------------
1. The codec. A series is stored as one BLOB:

       b"SBR1" | uint32 header length | header JSON (uncompressed) | zlib(body)

   The header lists each field (name, kind, width, has-null-mask) in the
   series' own key order and the bar count, so a caller can count bars or read
   the field list without decompressing. The body holds one column per field:
   float64 ("f8"), int64 ("i8") or fixed-width ASCII ("s"), byte-shuffled so
   zlib sees similar bytes together, each preceded by a packed null mask when
   the column contains None. Decoding yields EXACTLY the dicts that were
   encoded — same keys in the same order, same types, NaN/inf/-0.0 intact —
   which is what lets the engine stay unchanged. Anything the codec cannot
   hold exactly (bools, mixed int/float columns, non-ASCII strings, rows with
   different keys) is simply left as JSON: `encode_bars` returns None and the
   writer falls back. numpy and zlib only — pyarrow/Parquet would add well over
   100 MB to the add-on image for one table's worth of columns.

2. Data versions. `bars_version(bars)` is the SHA-1 (16 hex chars) of the
   canonical `json.dumps(bars)`. A JSON row written by this app IS that text,
   so its version is a hash of the stored string (no parsing), and the same
   bars have the same version before and after migration to the BLOB format.
   Every write stores the version; every walk-forward reports the versions it
   read (services/backtest.py -> `data_versions`), so a result can be tied to
   the exact data under it.

3. Revisions. When a refetch materially changes bars that were already stored
   (a new split/dividend adjustment factor, or corrected values), the PREVIOUS
   series is kept in `history_revisions` (encoded, with the reason and a diff
   summary) before it is overwritten. Volume is not compared (Yahoo revises it
   routinely), nor is the last stored bar (it may have been a session still in
   progress). Retention is bounded: REVISIONS_PER_SERIES per ticker and store,
   and REVISIONS_MAX_BYTES in total, oldest pruned first.

MIGRATION
---------
`migrate()` (job worker only, kind `bar_storage_migration`) converts existing
JSON rows: encode, decode again, require the result to serialize identically to
the stored JSON, and only then store the BLOB and clear the JSON. Batches are
committed separately, so it is resumable and never holds the write lock for
long. A row that fails verification keeps its JSON and is reported. SQLite does
not shrink the file on its own: freed pages are reused by later writes, and
`vacuum=True` compacts the file at the end (it locks the database while it
runs, so it is opt-in).

After migration, versions of SwingTrader older than the one that introduced
this module cannot read the cached history (they see an empty JSON column).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import struct
import zlib
from bisect import bisect_left
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)

MAGIC = b"SBR1"
FORMAT = "sbr1"
_HDR = struct.Struct("<I")
_MAX_STR_WIDTH = 64
_I8_MIN, _I8_MAX = -(2 ** 63), 2 ** 63 - 1

# Revision detection and retention.
REVISION_REL_TOL = 5e-4          # 5 bps on any of open/high/low/close
REVISIONS_PER_SERIES = 3         # newest kept per (ticker, store)
REVISIONS_MAX_BYTES = 64 * 1024 * 1024
# A uniform ratio across the changed bars (spread below this) that covers a
# PREFIX of the overlap is an adjustment-factor change, not a correction.
_ADJUSTMENT_RATIO_SPREAD = 0.002


# ── codec ───────────────────────────────────────────────────────────────────
def _kind_of(values: list) -> tuple[str, int] | None:
    """(kind, string width) for one column, or None when it can't round-trip."""
    types = set()
    for v in values:
        if v is None:
            continue
        t = type(v)
        if t is bool:
            return None
        types.add(t)
        if len(types) > 1:
            return None
    if not types or types == {float}:
        return "f8", 0
    if types == {int}:
        if any(v is not None and not (_I8_MIN <= v <= _I8_MAX) for v in values):
            return None
        return "i8", 0
    if types == {str}:
        width = 1
        for v in values:
            if v is None:
                continue
            if "\x00" in v or not v.isascii() or len(v) > _MAX_STR_WIDTH:
                return None
            width = max(width, len(v))
        return "s", width
    return None


def _shuffle(arr: np.ndarray) -> bytes:
    size = arr.dtype.itemsize
    return np.ascontiguousarray(arr).view(np.uint8).reshape(len(arr), size).T.tobytes()


def _unshuffle(buf: bytes, n: int, dtype: str) -> np.ndarray:
    size = np.dtype(dtype).itemsize
    raw = np.frombuffer(buf, dtype=np.uint8).reshape(size, n).T
    return np.ascontiguousarray(raw).view(dtype).reshape(n)


def encode_bars(bars) -> bytes | None:
    """The BLOB for `bars`, or None when the series can't be stored exactly
    (the caller then keeps JSON). Never raises for odd input."""
    try:
        return _encode(bars)
    except Exception:  # noqa: BLE001 — an unencodable series falls back to JSON
        logger.debug("bar encode fell back to JSON", exc_info=True)
        return None


def _encode(bars) -> bytes | None:
    if not isinstance(bars, list):
        return None
    n = len(bars)
    keys: list[str] = []
    if n:
        first = bars[0]
        if type(first) is not dict:
            return None
        keys = list(first)
        if not all(type(k) is str for k in keys):
            return None
        kt = tuple(keys)
        for b in bars:
            if type(b) is not dict or tuple(b) != kt:
                return None
    fields, chunks = [], []
    for k in keys:
        vals = [b[k] for b in bars]
        kind = _kind_of(vals)
        if kind is None:
            return None
        kind, width = kind
        mask = [v is None for v in vals]
        has_mask = any(mask)
        if kind == "f8":
            arr = np.array([0.0 if v is None else v for v in vals], dtype="<f8")
        elif kind == "i8":
            arr = np.array([0 if v is None else v for v in vals], dtype="<i8")
        else:
            arr = np.array(["" if v is None else v for v in vals], dtype=f"S{width}")
        if has_mask:
            chunks.append(np.packbits(np.array(mask, dtype=bool)).tobytes())
        chunks.append(_shuffle(arr))
        fields.append([k, kind, width, has_mask])
    header = json.dumps({"v": 1, "n": n, "f": fields}, separators=(",", ":")).encode()
    return MAGIC + _HDR.pack(len(header)) + header + zlib.compress(b"".join(chunks), 6)


def is_blob(value) -> bool:
    return isinstance(value, (bytes, bytearray, memoryview)) and bytes(value[:4]) == MAGIC


def _header(blob: bytes) -> tuple[dict, int]:
    (hlen,) = _HDR.unpack_from(blob, 4)
    start = 4 + _HDR.size
    return json.loads(blob[start:start + hlen]), start + hlen


def blob_count(blob: bytes) -> int:
    """Number of bars in a BLOB, read from the header without decompressing."""
    return int(_header(bytes(blob))[0]["n"])


def decode_bars(blob, since: str | None = None, keep_before: int = 0) -> list[dict]:
    """The bars stored in `blob` (list of dicts, exactly as encoded).

    With `since`, only bars from `keep_before` bars before the first bar dated
    >= `since` are built — the same cut `services/long_history.py` applies to
    a parsed series, without building dicts for the rest."""
    blob = bytes(blob)
    if not is_blob(blob):
        raise ValueError("not a bar blob")
    header, off = _header(blob)
    n = int(header["n"])
    body = zlib.decompress(blob[off:])
    pos = 0
    keys, cols = [], []
    for name, kind, width, has_mask in header["f"]:
        mask = None
        if has_mask:
            mlen = (n + 7) // 8
            mask = np.unpackbits(np.frombuffer(body[pos:pos + mlen], dtype=np.uint8), count=n).astype(bool)
            pos += mlen
        dtype = "<f8" if kind == "f8" else "<i8" if kind == "i8" else f"S{width}"
        size = np.dtype(dtype).itemsize * n
        arr = _unshuffle(body[pos:pos + size], n, dtype)
        pos += size
        if kind == "s":
            vals = arr.astype(f"U{width}").tolist()
        else:
            vals = arr.tolist()
        if mask is not None:
            for i in np.flatnonzero(mask).tolist():
                vals[i] = None
        keys.append(name)
        cols.append(vals)
    cut = 0
    if since is not None and "date" in keys and n:
        dates = cols[keys.index("date")]
        cut = max(0, bisect_left(dates, since) - int(keep_before or 0))
    if cut:
        cols = [c[cut:] for c in cols]
    if not keys:
        return [{} for _ in range(n - cut)] if n else []
    return _row_builder(tuple(keys))(*cols)


_BUILDERS: dict[tuple, object] = {}


def _row_builder(keys: tuple):
    """A function building the list of dicts from the columns with a dict
    DISPLAY (`{"date": v1, ...}`), which is markedly faster than
    dict(zip(keys, row)) — dict construction is most of a decode. Keys are
    embedded with repr(), so any string key is safe; one builder per key
    layout (in practice two: the cache's and the archive's)."""
    fn = _BUILDERS.get(keys)
    if fn is None:
        args = ", ".join(f"c{i}" for i in range(len(keys)))
        names = ", ".join(f"v{i}" for i in range(len(keys)))
        body = ", ".join(f"{k!r}: v{i}" for i, k in enumerate(keys))
        src = (f"def _build({args}):\n"
               f"    return [{{{body}}} for {names}{',' if len(keys) == 1 else ''} in zip({args})]\n")
        ns: dict = {}
        exec(compile(src, "<bar_store row builder>", "exec"), ns)  # noqa: S102 — keys are repr()-quoted
        fn = _BUILDERS[keys] = ns["_build"]
    return fn


# ── reading / writing rows ──────────────────────────────────────────────────
def _json_list(text) -> list[dict]:
    if not text:
        return []
    try:
        bars = json.loads(text)
    except (TypeError, ValueError):
        return []
    return bars if isinstance(bars, list) else []


def read_bars(blob, text, since: str | None = None, keep_before: int = 0) -> list[dict]:
    """Bars from a (BLOB, JSON) column pair: the BLOB when there is one, else
    the JSON (rows not migrated yet, or series the codec can't hold). A BLOB
    that fails to decode falls back to the JSON rather than returning nothing."""
    if blob:
        try:
            return decode_bars(blob, since, keep_before)
        except Exception:  # noqa: BLE001
            logger.warning("Unreadable bar BLOB; falling back to JSON", exc_info=True)
    bars = _json_list(text)
    if since is not None and bars:
        cut = bisect_left([b.get("date") or "" for b in bars], since) - int(keep_before or 0)
        if cut > 0:
            bars = bars[cut:]
    return bars


def bar_count(blob, text) -> int:
    if blob:
        try:
            return blob_count(blob)
        except Exception:  # noqa: BLE001
            pass
    return len(_json_list(text))


def bars_version(bars: list[dict]) -> str:
    """Content version of a series: SHA-1 of its canonical JSON (16 hex)."""
    return hashlib.sha1(json.dumps(bars).encode()).hexdigest()[:16]


def text_version(text: str | None) -> str | None:
    """Version of a stored JSON row without parsing it. Equal to
    bars_version(json.loads(text)) for every row this app wrote."""
    if not text:
        return None
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def row_version(blob, text, stored: str | None) -> str | None:
    """The version of whatever a (BLOB, JSON, version) column triple holds."""
    if stored:
        return stored
    if blob:
        try:
            return bars_version(decode_bars(blob))
        except Exception:  # noqa: BLE001
            return None
    return text_version(text)


def payload_for(bars: list[dict]) -> tuple[bytes | None, str | None, str]:
    """(blob, json_text, version) to store for `bars`: the BLOB when the codec
    can hold the series exactly, otherwise JSON. Exactly one of the first two
    is set."""
    version = bars_version(bars)
    blob = encode_bars(bars)
    if blob is not None:
        return blob, None, version
    return None, json.dumps(bars), version


# ── revisions ───────────────────────────────────────────────────────────────
def _pos_finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0


def compare_series(old: list[dict], new: list[dict]) -> dict | None:
    """Did `new` materially change bars `old` already had? None when not.

    Only dates both series cover are compared (a longer or rolled-forward
    series is not a revision); `old`'s last bar is skipped because it may
    have been a session still in progress when it was stored. Pure."""
    if not old or not new:
        return None
    new_by = {b.get("date"): b for b in new if isinstance(b, dict)}
    first, last = new[0].get("date"), new[-1].get("date")
    if first is None or last is None:
        return None
    overlap = changed = removed = 0
    ratios: list[float] = []
    changed_dates: list[str] = []
    max_rel = 0.0
    unchanged_seen = False
    prefix = True
    for b in old[:-1]:
        d = b.get("date") if isinstance(b, dict) else None
        if d is None or d < first or d > last:
            continue
        nb = new_by.get(d)
        if nb is None:
            removed += 1
            continue
        overlap += 1
        worst = 0.0
        for k in ("open", "high", "low", "close"):
            o, v = b.get(k), nb.get(k)
            if _pos_finite(o) and _pos_finite(v):
                worst = max(worst, abs(v / o - 1))
            elif _pos_finite(o) != _pos_finite(v):
                worst = max(worst, 1.0)
        if worst > REVISION_REL_TOL:
            changed += 1
            changed_dates.append(d)
            max_rel = max(max_rel, worst)
            if _pos_finite(b.get("close")) and _pos_finite(nb.get("close")):
                ratios.append(nb["close"] / b["close"])
            if unchanged_seen:
                prefix = False
        else:
            unchanged_seen = True
    if not changed and not removed:
        return None
    kind, factor = "correction", None
    if changed and ratios and prefix and len(ratios) == changed:
        lo, hi = min(ratios), max(ratios)
        if lo > 0 and hi / lo - 1 < _ADJUSTMENT_RATIO_SPREAD:
            factor = sorted(ratios)[len(ratios) // 2]
            kind = "split" if (factor < 0.67 or factor > 1.5) else "adjustment"
    if kind == "correction":
        reason = f"values changed on {changed} of {overlap} shared dates"
        if removed:
            reason += f"; {removed} previously stored date(s) no longer present"
    else:
        reason = (f"{'split' if kind == 'split' else 'dividend/adjustment'} factor {factor:.6f} "
                  f"applied to bars through {changed_dates[-1]}")
    return {
        "kind": kind,
        "reason": reason,
        "overlap": overlap,
        "changed": changed,
        "removed": removed,
        "max_rel_change": round(max_rel, 6),
        "factor": round(factor, 6) if factor is not None else None,
        "first_changed": changed_dates[0] if changed_dates else None,
        "last_changed": changed_dates[-1] if changed_dates else None,
    }


def record_revision(db, ticker: str, store: str, old_bars: list[dict], change: dict,
                    prev_version: str | None, new_version: str | None) -> None:
    """Keep `old_bars` before they are overwritten, then enforce retention.
    Adds to the caller's session; the caller's commit makes the data write and
    its revision atomic."""
    from database.models import HistoryRevision
    blob = encode_bars(old_bars)
    if blob is not None:
        payload, fmt = blob, FORMAT
    else:
        payload, fmt = zlib.compress(json.dumps(old_bars).encode(), 6), "json.zlib"
    db.add(HistoryRevision(
        ticker=ticker, store=store, captured_at=datetime.now(timezone.utc),
        kind=change.get("kind"), reason=str(change.get("reason") or "")[:500],
        summary_json=json.dumps(change), prev_version=prev_version, new_version=new_version,
        prev_bars=len(old_bars),
        prev_start=(old_bars[0].get("date") if old_bars else None),
        prev_end=(old_bars[-1].get("date") if old_bars else None),
        prev_format=fmt, prev_payload=payload, payload_bytes=len(payload),
    ))
    db.flush()
    prune_revisions(db, ticker, store)


def prune_revisions(db, ticker: str | None = None, store: str | None = None) -> int:
    """Enforce REVISIONS_PER_SERIES (for one series, or all when ticker is
    None) and REVISIONS_MAX_BYTES overall, oldest first. Returns rows removed."""
    from sqlalchemy import func
    from database.models import HistoryRevision
    removed = 0
    q = db.query(HistoryRevision.ticker, HistoryRevision.store).distinct()
    if ticker is not None:
        q = q.filter(HistoryRevision.ticker == ticker)
        if store is not None:
            q = q.filter(HistoryRevision.store == store)
    for t, s in q.all():
        ids = [i for (i,) in (db.query(HistoryRevision.id)
                              .filter(HistoryRevision.ticker == t, HistoryRevision.store == s)
                              .order_by(HistoryRevision.captured_at.desc(), HistoryRevision.id.desc())
                              .all())]
        for rid in ids[REVISIONS_PER_SERIES:]:
            db.query(HistoryRevision).filter(HistoryRevision.id == rid).delete()
            removed += 1
    total = db.query(func.coalesce(func.sum(HistoryRevision.payload_bytes), 0)).scalar() or 0
    if total > REVISIONS_MAX_BYTES:
        for rid, size in (db.query(HistoryRevision.id, HistoryRevision.payload_bytes)
                          .order_by(HistoryRevision.captured_at.asc(), HistoryRevision.id.asc())
                          .all()):
            if total <= REVISIONS_MAX_BYTES:
                break
            db.query(HistoryRevision).filter(HistoryRevision.id == rid).delete()
            total -= int(size or 0)
            removed += 1
    if removed:
        db.flush()
    return removed


def revision_bars(rev) -> list[dict]:
    """The series a revision preserved (to reproduce an old result)."""
    if rev.prev_format == FORMAT:
        return decode_bars(rev.prev_payload)
    return json.loads(zlib.decompress(rev.prev_payload).decode())


def note_rewrite(db, ticker: str, store: str, old_bars: list[dict], new_bars: list[dict],
                 prev_version: str | None, new_version: str | None) -> dict | None:
    """compare_series + record_revision, never raising: bookkeeping must not
    stop fresh data from being stored. Returns the change (or None)."""
    try:
        change = compare_series(old_bars, new_bars)
        if change:
            record_revision(db, ticker, store, old_bars, change, prev_version, new_version)
            logger.info("History revision kept for %s (%s): %s", ticker, store, change["reason"])
        return change
    except Exception:  # noqa: BLE001
        logger.warning("Could not record a history revision for %s", ticker, exc_info=True)
        return None


# ── migration (job worker only) ─────────────────────────────────────────────
def _verified_blob(text: str) -> tuple[bytes, str] | None:
    """(blob, version) for a JSON row when the BLOB decodes to exactly the
    same series, else None (the row stays JSON)."""
    bars = _json_list(text)
    if not bars and text.strip() != "[]":
        return None
    blob = encode_bars(bars)
    if blob is None:
        return None
    canonical = json.dumps(bars)
    if json.dumps(decode_bars(blob)) != canonical:
        return None
    return blob, hashlib.sha1(canonical.encode()).hexdigest()[:16]


def _pending(db):
    from database.models import HistoryArchive, StockCache
    cache = [t for (t,) in db.query(StockCache.ticker)
             .filter(StockCache.history_json.isnot(None), StockCache.history_json != "",
                     StockCache.history_blob.is_(None)).all()]
    archive = [t for (t,) in db.query(HistoryArchive.ticker)
               .filter(HistoryArchive.bars_json.isnot(None), HistoryArchive.bars_json != "",
                       HistoryArchive.bars_blob.is_(None)).all()]
    return cache, archive


def migrate(db, progress=None, batch: int = 10, max_rows: int | None = None,
            vacuum: bool = False) -> dict:
    """Convert JSON rows to verified BLOBs. Resumable and idempotent: only rows
    with JSON and no BLOB are touched, each batch is committed on its own, and
    a second run finds nothing to do. `max_rows` stops early (tests, or a
    deliberately short run). Rows whose JSON the codec can't reproduce exactly
    stay JSON and are listed in `kept_json`.

    Each conversion is a CONDITIONAL update — it applies only if the row still
    holds the exact JSON that was read — because the web process keeps
    refreshing cache rows while this runs; a row rewritten in between is left
    for the next run (`raced`), never overwritten with stale bars."""
    from database.models import HistoryArchive, StockCache

    def tick(frac, detail):
        if progress:
            try:
                progress(frac, detail)
            except Exception:  # noqa: BLE001
                pass

    cache_todo, archive_todo = _pending(db)
    todo = [("cache", t) for t in cache_todo] + [("archive", t) for t in archive_todo]
    counts = {"cache": 0, "archive": 0, "raced": 0, "json_bytes_freed": 0, "blob_bytes_written": 0}
    kept_json: list[dict] = []
    done = 0
    for i, (store, t) in enumerate(todo):
        if max_rows is not None and done >= max_rows:
            break
        if i % 10 == 0:
            tick(0.02 + 0.9 * i / max(1, len(todo)), f"{store} {t} ({i + 1}/{len(todo)})")
        if store == "cache":
            model, col = StockCache, StockCache.history_json
            got_row = db.query(col).filter(StockCache.ticker == t, StockCache.history_blob.is_(None)).first()
        else:
            model, col = HistoryArchive, HistoryArchive.bars_json
            got_row = db.query(col).filter(HistoryArchive.ticker == t, HistoryArchive.bars_blob.is_(None)).first()
        text = got_row[0] if got_row else None
        if not text:
            continue
        got = _verified_blob(text)
        if got is None:
            if len(kept_json) < 50:
                kept_json.append({"store": store, "ticker": t})
            continue
        blob, version = got
        if store == "cache":
            values = {StockCache.history_blob: blob, StockCache.history_version: version,
                      StockCache.history_json: None}
        else:
            values = {HistoryArchive.bars_blob: blob, HistoryArchive.data_version: version,
                      HistoryArchive.bars_json: ""}
        changed = (db.query(model)
                   .filter(model.ticker == t, col == text)
                   .update(values, synchronize_session=False))
        if not changed:
            counts["raced"] += 1
            continue
        counts[store] += 1
        counts["json_bytes_freed"] += len(text)
        counts["blob_bytes_written"] += len(blob)
        done += 1
        if done % max(1, batch) == 0:
            db.commit()
    db.commit()
    db.expire_all()
    remaining_cache, remaining_archive = _pending(db)
    result = {
        **counts,
        "kept_json": kept_json,
        "remaining": len(remaining_cache) + len(remaining_archive),
        "vacuumed": False,
    }
    if vacuum and not result["remaining"]:
        tick(0.95, "Compacting the database file (VACUUM)…")
        try:
            db.close()
            from database.db import engine
            with engine.connect() as conn:
                conn.exec_driver_sql("VACUUM")
            result["vacuumed"] = True
        except Exception as exc:  # noqa: BLE001
            result["vacuum_error"] = f"{type(exc).__name__}: {exc}"
    tick(0.99, "Finishing…")
    return result


def storage_status(db) -> dict:
    """How much of the stored history is columnar, from small columns only."""
    from sqlalchemy import func
    from database.models import HistoryArchive, HistoryRevision, StockCache
    cache_blob = db.query(func.count()).select_from(StockCache).filter(StockCache.history_blob.isnot(None)).scalar()
    arch_blob = db.query(func.count()).select_from(HistoryArchive).filter(HistoryArchive.bars_blob.isnot(None)).scalar()
    pending_cache, pending_archive = _pending(db)
    revs = db.query(func.count(HistoryRevision.id), func.coalesce(func.sum(HistoryRevision.payload_bytes), 0)).one()
    return {
        "format": FORMAT,
        "cache_columnar": int(cache_blob or 0),
        "archive_columnar": int(arch_blob or 0),
        "pending_cache": len(pending_cache),
        "pending_archive": len(pending_archive),
        "migrated": not pending_cache and not pending_archive,
        "revisions": int(revs[0] or 0),
        "revision_bytes": int(revs[1] or 0),
        "revision_limits": {"per_series": REVISIONS_PER_SERIES, "max_bytes": REVISIONS_MAX_BYTES},
    }
