"""
AI output validation (ML4T 3e §22.6 — LLM output is a claim, not a fact).

The app computes every indicator itself and hands the numbers to the model;
the model only narrates. Nothing checked the narration. A model that writes
"NVDA, RSI 28 and oversold" when the data says RSI 74, or confidently analyses
a ticker it was never given, used to be shown to the user exactly like a
correct report.

This module checks LLM output against the numbers that were in the prompt and
ANNOTATES it. It never rewrites the model's text. Pure functions, no DB, no
network. Every public entry point has a `safe_*` wrapper that turns an
internal error into a `validator_error` issue, so validation can never make
report generation (or an AI endpoint) fail.

What is checked
---------------
* Tickers: a symbol the text mentions that was NOT in the data given to the
  model -> `unknown_ticker` (anything said about it is unverified).
* Numbers attributed to a ticker, in prose (nearest preceding ticker in the
  same sentence) and in markdown tables (the row's ticker, the column header):
  price (±2 %), RSI (±3 pts), day change %, P&L %, distance from the 50/200-day
  MA — each compared with the value that was supplied. A metric the data did
  not include is reported as `unverifiable`, never as wrong.
* Directional contradictions: "above/below the 50/200-day" vs the supplied
  vs_ma50 / vs_ma200 sign; "oversold" with RSI >= 50 or "overbought" with
  RSI <= 50. Negated or conditional phrasing ("not oversold", "needs to
  reclaim the 200-day", "if it falls below the 50-day") is skipped.
* Structured outputs: signals (list of {type, signal, message, strength}) and
  per-ticker analyses are schema-checked; invalid signal items are DROPPED from
  what is shown, and every drop is reported.

What it cannot catch: opinions, forecasts and reasoning quality; numbers not
attributable to one ticker; claims about metrics the app never computed
(support/resistance, fundamentals it wasn't given); paraphrased tickers
("Nvidia" rather than NVDA); common-word tickers written without a `$`.
"""
from __future__ import annotations

import json
import math
import re

VALIDATION_VERSION = 1

# ── tolerances ───────────────────────────────────────────────────────────────
PRICE_TOL_PCT = 2.0      # relative, percent of the true price
RSI_TOL = 3.0            # absolute RSI points
PCT_TOL_ABS = 0.5        # absolute percentage points for % metrics …
PCT_TOL_REL = 0.10       # … or this share of the true value, whichever is larger
MA_DIST_TOL_ABS = 1.5    # distance-from-MA claims are usually rounded harder

SIGNAL_TYPES = ("entry", "exit", "warning", "info")
SIGNAL_STRENGTHS = ("strong", "moderate", "weak")
SENTIMENTS = ("bullish", "bearish", "neutral")

# Upper-case words that are also real tickers, or common acronyms in trading
# prose. Recognised as tickers only when written as $TICKER or when they are in
# the data given to the model.
_AMBIGUOUS = {
    "A", "ALL", "AM", "AN", "ARE", "AT", "BE", "BIG", "BY", "C", "CAN", "CAT",
    "D", "DD", "DO", "ED", "F", "FAST", "GO", "HAS", "HD", "IT", "J", "K",
    "KEY", "L", "LOW", "MA", "MO", "NOW", "O", "ON", "ONE", "OR", "PH", "R",
    "SO", "T", "TV", "UP", "V", "WELL", "X",
}
_ACRONYMS = {
    "AI", "API", "ATH", "ATR", "BB", "CEO", "CFO", "CPI", "DMA", "EMA", "EOD",
    "EPS", "ETF", "ETFS", "EU", "FED", "FOMC", "GDP", "IPO", "LLM", "MACD",
    "MV", "NA", "NYSE", "OK", "PCE", "PE", "PM", "PNL", "Q1", "Q2", "Q3", "Q4",
    "RSI", "SMA", "SP", "TA", "TBD", "UK", "US", "USA", "USD", "VWAP", "YOY",
    "YTD", "QOQ", "ADX", "DI", "ROI", "EV", "FCF", "IV", "OTM", "ITM", "ATM",
    "NASDAQ", "SEC", "FX", "HY", "IG", "ZERO", "HOLD", "BUY", "SELL", "TRIM",
    "WATCH", "NOTE", "RISK", "EXIT", "ENTRY", "STOP", "TARGET", "BULL", "BEAR",
    "NEW", "TOP", "MAX", "MIN", "AVG", "N", "M", "B", "MM", "BN", "TL", "DR",
    "INVERTED", "M2", "VIX",
}

_TICKER_TOKEN = re.compile(r"(?<![A-Za-z0-9^$])(\$?\^?[A-Z]{1,5}(?:[.\-][A-Z])?)(?![A-Za-z0-9])")

_NEGATION = re.compile(
    r"\b(not|n't|no longer|never|if|would|could|should|might|may|needs? to|"
    r"has to|have to|reclaim\w*|break\w*|fall\w*|drop\w*|dip\w*|mov\w+|cross\w*|"
    r"slip\w*|push\w*|close\w*|get back|return\w*|recover\w*|approach\w*|"
    r"toward\w*|risk\w*|unless|until|once|watch\w*)\b[^.;:|]{0,24}$",
    re.IGNORECASE,
)


def _universe() -> set[str]:
    try:
        from services.universe import UNIVERSE
        return set(UNIVERSE)
    except Exception:  # noqa: BLE001
        return set()


def _norm_ticker(tok: str) -> str:
    return tok.lstrip("$^").upper().replace(".", "-")


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _parse_number(text: str):
    """First number in a table cell / capture, tolerant of $, %, +, commas, **."""
    m = re.search(r"[-+−]?\$?\s*\d[\d,]*(?:\.\d+)?", text or "")
    if not m:
        return None
    raw = m.group(0).replace("$", "").replace(",", "").replace(" ", "").replace("−", "-")
    return _num(raw)


# ── facts ────────────────────────────────────────────────────────────────────
_FACT_KEYS = ("price", "rsi", "chg_pct", "pnl_pct", "vs_ma50", "vs_ma200", "ma_state")


def build_facts(*row_lists) -> dict[str, dict]:
    """{TICKER: {price, rsi, chg_pct, pnl_pct, vs_ma50, vs_ma200, ma_state}}
    from any rows/quotes that carry a `ticker`. Later rows fill gaps only."""
    facts: dict[str, dict] = {}
    for rows in row_lists:
        if isinstance(rows, dict):
            rows = [rows]
        for row in rows or []:
            if not isinstance(row, dict) or not row.get("ticker"):
                continue
            t = _norm_ticker(str(row["ticker"]))
            entry = facts.setdefault(t, {})
            for k in _FACT_KEYS:
                if entry.get(k) is None and row.get(k) is not None:
                    entry[k] = row.get(k)
    return facts


def tickers_in_text(text: str, universe: set[str] | None = None) -> set[str]:
    """Every ticker-looking token in `text` that is a real symbol (universe) or
    written with a `$`. Used to learn which tickers the PROMPT contained."""
    universe = _universe() if universe is None else universe
    found = set()
    for m in _TICKER_TOKEN.finditer(text or ""):
        raw = m.group(1)
        t = _norm_ticker(raw)
        if raw.startswith("$") or t in universe:
            found.add(t)
    return found


# ── result block ─────────────────────────────────────────────────────────────
class _Result:
    def __init__(self):
        self.checked = 0
        self.verified = 0
        self.issues: list[dict] = []

    def issue(self, severity, kind, ticker, claim, expected):
        self.issues.append({
            "severity": severity, "kind": kind, "ticker": ticker,
            "claim": (claim or "")[:240], "expected": expected,
        })

    def block(self, extra: dict | None = None) -> dict:
        contradicted = sum(1 for i in self.issues if i["severity"] == "high")
        unverifiable = sum(1 for i in self.issues if i["severity"] in ("warn", "info"))
        out = {
            "version": VALIDATION_VERSION,
            "checked": self.checked,
            "verified": self.verified,
            "contradicted": contradicted,
            "unverifiable": unverifiable,
            "score": round(self.verified / self.checked, 3) if self.checked else None,
            "pass": contradicted == 0,
            "issues": self.issues,
        }
        if extra:
            out.update(extra)
        return out


def _within(claimed: float, actual: float, abs_tol: float, rel_tol: float) -> bool:
    return abs(claimed - actual) <= max(abs_tol, abs(actual) * rel_tol)


def _check_metric(res: _Result, ticker: str, metric: str, claimed, facts: dict,
                  claim_text: str, sign_hint: int | None = None) -> None:
    """Compare one numeric claim with the supplied value."""
    claimed = _num(claimed)
    if claimed is None:
        return
    res.checked += 1
    actual = _num((facts.get(ticker) or {}).get(metric))
    if actual is None:
        res.issue("info", "unverifiable", ticker, claim_text,
                  f"no {metric} was supplied for {ticker}")
        return
    if sign_hint is not None:
        claimed = abs(claimed) * sign_hint
    if metric == "price":
        ok = abs(claimed - actual) <= abs(actual) * PRICE_TOL_PCT / 100.0
        kind = "price_mismatch"
    elif metric == "rsi":
        ok = abs(claimed - actual) <= RSI_TOL
        kind = "rsi_mismatch"
    elif metric in ("vs_ma50", "vs_ma200"):
        ok = _within(claimed, actual, MA_DIST_TOL_ABS, PCT_TOL_REL)
        kind = "pct_mismatch"
    else:
        ok = _within(claimed, actual, PCT_TOL_ABS, PCT_TOL_REL)
        kind = "pct_mismatch"
    if ok:
        res.verified += 1
    else:
        res.issue("high", kind, ticker, claim_text, f"{metric} = {actual:g}")


# ── prose claims ─────────────────────────────────────────────────────────────
_RSI_RE = re.compile(
    r"\bRSI(?:\s*\(\s*\d+\s*\))?\s*(?:of|at|is|was|=|:|near|around|~|≈|reading(?: of)?|sits at|now)?\s*"
    r"(?:\*\*)?(\d{1,3}(?:\.\d+)?)(?!\s*%|\d)", re.IGNORECASE)
_PRICE_RE = re.compile(
    r"\b(?:price(?:d)?|trading at|trades at|trading near|closed at|closes at|close of|last(?: price)?|"
    r"currently(?: at)?|sits at|now at|at a price of|quoted at)\s*(?:of|is|was|:|=|~|≈|around|near)?\s*"
    r"(?:\*\*)?\$\s?(\d[\d,]*(?:\.\d+)?)", re.IGNORECASE)
_CHG_RE = re.compile(
    r"\b(up|down|gained|gains|lost|loses|rose|rises|fell|falls|climbed|dropped|drops|jumped|slid|slipped|added|shed)\s+"
    r"(\d+(?:\.\d+)?)\s?%\s*(?:today|on the day|in (?:today's|the) session|intraday|on the session)",
    re.IGNORECASE)
_PNL_RE = re.compile(
    r"\b(?:P&L|PnL|P/L|unrealized(?: gain| loss| P&L)?|position (?:is )?(?:up|down))\s*"
    r"(?:of|is|at|:|=)?\s*([+\-−]?\d+(?:\.\d+)?)\s?%", re.IGNORECASE)
_MA_DIST_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s?%\s*(above|below)\s+(?:its\s+|the\s+)?(50|200)[\s\-]?"
    r"(?:day|DMA|dma|MA|d\b|-day)", re.IGNORECASE)
_MA_SIDE_RE = re.compile(
    r"\b(above|below|over|under)\s+(?:its\s+|the\s+)?(50|200)[\s\-]?"
    r"(?:day|DMA|dma|MA\b|d\b|-day)", re.IGNORECASE)
_OVERSOLD_RE = re.compile(r"\b(oversold|overbought)\b", re.IGNORECASE)

_DOWN_WORDS = {"down", "lost", "loses", "fell", "falls", "dropped", "drops", "slid", "slipped", "shed"}


def _segments(text: str) -> list[str]:
    """Lines, then sentences. Decimal points and $1.50 stay intact."""
    out = []
    for line in (text or "").splitlines():
        if line.strip().startswith("|"):
            continue   # tables are checked cell-by-cell in _check_tables
        for seg in re.split(r"(?<=[.!?;])\s+(?=[A-Z*$#(\[])", line):
            if seg.strip():
                out.append(seg)
    return out


def _mentions(seg: str, known: set[str], universe: set[str]) -> list[tuple[int, str, bool]]:
    """(position, ticker, is_known) for every ticker token in a segment."""
    shouting = _is_shouting(seg)
    out = []
    for m in _TICKER_TOKEN.finditer(seg):
        raw = m.group(1)
        t = _norm_ticker(raw)
        dollar = raw.startswith("$")
        if t in known:
            out.append((m.start(), t, True))
            continue
        if not dollar:
            if shouting or t in _AMBIGUOUS or t in _ACRONYMS or t not in universe:
                continue
        elif t in _ACRONYMS and not t.startswith("^"):
            continue
        out.append((m.start(), t, False))
    return out


def _is_shouting(seg: str) -> bool:
    words = re.findall(r"[A-Za-z]{2,}", seg)
    if len(words) < 3:
        return False
    return sum(1 for w in words if w.isupper()) / len(words) > 0.6


def _owner(mentions, pos: int, default: str | None = None):
    """Nearest KNOWN ticker mentioned before `pos` in the same segment, else
    `default` (single-ticker outputs: analysis, signals, ticker chat)."""
    owner = default
    for p, t, is_known in mentions:
        if p > pos:
            break
        if is_known:
            owner = t
    return owner


def _negated(seg: str, pos: int) -> bool:
    return bool(_NEGATION.search(seg[max(0, pos - 40):pos]))


def _check_prose(res: _Result, text: str, facts: dict, known: set[str], universe: set[str],
                 default: str | None = None) -> None:
    flagged_unknown: set[str] = set()
    for seg in _segments(text):
        mentions = _mentions(seg, known, universe)
        for _p, t, is_known in mentions:
            if not is_known and t not in flagged_unknown:
                flagged_unknown.add(t)
                res.checked += 1
                res.issue("warn", "unknown_ticker", t, seg.strip(),
                          f"{t} was not in the data given to the model — anything said about it is unverified")
        if default is None and not any(k for _p, _t, k in mentions):
            continue
        claim = seg.strip()

        for m in _RSI_RE.finditer(seg):
            t = _owner(mentions, m.start(), default)
            if t:
                _check_metric(res, t, "rsi", m.group(1), facts, claim)
        for m in _PRICE_RE.finditer(seg):
            t = _owner(mentions, m.start(), default)
            if t:
                _check_metric(res, t, "price", m.group(1).replace(",", ""), facts, claim)
        for m in _CHG_RE.finditer(seg):
            t = _owner(mentions, m.start(), default)
            if t:
                sign = -1 if m.group(1).lower() in _DOWN_WORDS else 1
                _check_metric(res, t, "chg_pct", m.group(2), facts, claim, sign_hint=sign)
        for m in _PNL_RE.finditer(seg):
            t = _owner(mentions, m.start(), default)
            if t:
                val = m.group(1).replace("−", "-")
                hint = -1 if "down" in m.group(0).lower() else None
                _check_metric(res, t, "pnl_pct", val, facts, claim, sign_hint=hint)
        dist_spans = []
        for m in _MA_DIST_RE.finditer(seg):
            t = _owner(mentions, m.start(), default)
            if not t:
                continue
            dist_spans.append((m.start(), m.end()))
            sign = 1 if m.group(2).lower() == "above" else -1
            _check_metric(res, t, f"vs_ma{m.group(3)}", m.group(1), facts, claim, sign_hint=sign)
        for m in _MA_SIDE_RE.finditer(seg):
            if any(a <= m.start() < b for a, b in dist_spans):
                continue   # already judged numerically above
            t = _owner(mentions, m.start(), default)
            if not t or _negated(seg, m.start()):
                continue
            actual = _num((facts.get(t) or {}).get(f"vs_ma{m.group(2)}"))
            res.checked += 1
            if actual is None:
                res.issue("info", "unverifiable", t, claim, f"no vs_ma{m.group(2)} was supplied for {t}")
                continue
            says_above = m.group(1).lower() in ("above", "over")
            if actual == 0 or (actual > 0) == says_above:
                res.verified += 1
            else:
                res.issue("high", "contradiction", t, claim,
                          f"price is {abs(actual):g}% {'above' if actual > 0 else 'below'} the {m.group(2)}-day MA")
        for m in _OVERSOLD_RE.finditer(seg):
            t = _owner(mentions, m.start(), default)
            if not t or _negated(seg, m.start()):
                continue
            rsi = _num((facts.get(t) or {}).get("rsi"))
            res.checked += 1
            if rsi is None:
                res.issue("info", "unverifiable", t, claim, f"no RSI was supplied for {t}")
                continue
            word = m.group(1).lower()
            bad = (word == "oversold" and rsi >= 50) or (word == "overbought" and rsi <= 50)
            if bad:
                strong = (word == "oversold" and rsi > 70) or (word == "overbought" and rsi < 30)
                res.issue("high" if strong else "warn", "contradiction", t, claim,
                          f"RSI = {rsi:g} ({'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral'})")
            else:
                res.verified += 1


# ── tables ───────────────────────────────────────────────────────────────────
def _column_metric(header: str) -> str | None:
    h = header.lower().replace("*", "").strip()
    if not h:
        return None
    if "rsi" in h:
        return "rsi"
    if any(w in h for w in ("target", "stop", "entry", "avg", "cost", "support", "resist", "value", "basis")):
        return None
    if "200" in h and ("ma" in h or "day" in h):
        return "vs_ma200"
    if "50" in h and ("ma" in h or "day" in h):
        return "vs_ma50"
    if "p&l" in h or "pnl" in h or "p/l" in h or "gain" in h:
        return "pnl_pct"
    if "chg" in h or "change" in h or "day %" in h or "today" in h:
        return "chg_pct"
    if h in ("price", "last", "close", "last price", "current price", "px"):
        return "price"
    return None


def _cells(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _check_tables(res: _Result, text: str, facts: dict, known: set[str], universe: set[str]) -> None:
    lines = (text or "").splitlines()
    i = 0
    while i < len(lines) - 1:
        head, sep = lines[i], lines[i + 1]
        if head.strip().startswith("|") and re.match(r"^\s*\|?\s*:?-{2,}", sep.strip() or "x"):
            metrics = [_column_metric(h) for h in _cells(head)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = _cells(lines[i])
                ticker = None
                for c in cells[:2]:
                    for tok in _TICKER_TOKEN.findall(c.replace("*", "").replace("`", "")):
                        t = _norm_ticker(tok)
                        if t in known:
                            ticker = t
                            break
                    if ticker:
                        break
                if ticker:
                    for metric, cell in zip(metrics, cells):
                        if metric and cell and cell not in ("—", "-", "--", "N/A", "n/a"):
                            _check_metric(res, ticker, metric, _parse_number(cell), facts,
                                          lines[i].strip())
                i += 1
            continue
        i += 1


# ── public: free text ────────────────────────────────────────────────────────
def validate_text(text: str, facts: dict, known_tickers=None, default_ticker: str | None = None) -> dict:
    """Validate free text (report markdown, chat answer, analysis prose).

    `facts`: build_facts(...) output. `known_tickers`: every ticker that was in
    the prompt (tickers_in_text(prompt)); defaults to the facts' tickers.
    `default_ticker`: for single-ticker outputs, the ticker a claim belongs to
    when no ticker is named before it in the sentence."""
    universe = _universe()
    known = {_norm_ticker(t) for t in (known_tickers or [])} | set(facts or {})
    res = _Result()
    _check_tables(res, text, facts or {}, known, universe)
    default = _norm_ticker(default_ticker) if default_ticker else None
    _check_prose(res, text, facts or {}, known, universe, default)
    return res.block()


# ── public: structured outputs ───────────────────────────────────────────────
def validate_signals(signals, ticker: str, facts: dict | None = None) -> tuple[list[dict], dict]:
    """(signals safe to show, validation). Items that are not a dict with a
    valid type/strength and non-empty signal+message are dropped and reported —
    never shown half-formed. Kept items' messages are then text-checked."""
    res = _Result()
    ticker = _norm_ticker(ticker)
    if isinstance(signals, dict):
        signals = signals.get("signals", [signals])
    if not isinstance(signals, list):
        res.checked += 1
        res.issue("high", "invalid_output", ticker, str(signals)[:200],
                  "a JSON array of {type, signal, message, strength}")
        return [], res.block()
    kept = []
    for idx, item in enumerate(signals):
        res.checked += 1
        problems = []
        if not isinstance(item, dict):
            problems.append("not an object")
        else:
            if str(item.get("type", "")).lower() not in SIGNAL_TYPES:
                problems.append(f"type {item.get('type')!r} not in {SIGNAL_TYPES}")
            if str(item.get("strength", "")).lower() not in SIGNAL_STRENGTHS:
                problems.append(f"strength {item.get('strength')!r} not in {SIGNAL_STRENGTHS}")
            for key in ("signal", "message"):
                if not isinstance(item.get(key), str) or not item.get(key).strip():
                    problems.append(f"missing {key}")
        if problems:
            res.issue("warn", "invalid_item", ticker, json.dumps(item, default=str)[:200],
                      "dropped: " + "; ".join(problems))
            continue
        res.verified += 1
        clean = dict(item)
        clean["type"] = item["type"].lower()
        clean["strength"] = item["strength"].lower()
        kept.append(clean)
    if signals and not kept:
        # Nothing usable survived: the output as a whole failed, not just an item.
        res.issue("high", "invalid_output", ticker, json.dumps(signals, default=str)[:200],
                  "at least one valid {type, signal, message, strength} item")
    if facts:
        text = "\n".join(f"{s['signal']}. {s['message']}" for s in kept)
        _merge(res, validate_text(text, facts, [ticker], default_ticker=ticker))
    return kept, res.block({"dropped": len(signals) - len(kept)})


def validate_analysis(analysis, ticker: str, facts: dict | None = None) -> dict:
    """Schema/range checks on a per-ticker analysis dict, then the text checks
    over its prose fields (each prefixed with the ticker so claims attribute)."""
    res = _Result()
    ticker = _norm_ticker(ticker)
    if not isinstance(analysis, dict):
        res.checked += 1
        res.issue("high", "invalid_output", ticker, str(analysis)[:200], "a JSON object")
        return res.block()

    def field(ok, key, expected):
        res.checked += 1
        if ok:
            res.verified += 1
        else:
            res.issue("warn", "invalid_item", ticker, f"{key} = {analysis.get(key)!r}", expected)

    if "sentiment" in analysis:
        field(str(analysis.get("sentiment", "")).lower() in SENTIMENTS, "sentiment", f"one of {SENTIMENTS}")
    if analysis.get("confidence") is not None:
        c = _num(analysis.get("confidence"))
        field(c is not None and 0.0 <= c <= 1.0, "confidence", "a number from 0 to 1")
    if analysis.get("ai_score") is not None:
        s = _num(analysis.get("ai_score"))
        field(s is not None and 1 <= s <= 10, "ai_score", "a number from 1 to 10")
    if facts:
        parts = []
        for key in ("summary", "entry_strategy", "exit_strategy"):
            if isinstance(analysis.get(key), str):
                parts.append(analysis[key])
        for key in ("key_factors", "risks", "opportunities"):
            for item in analysis.get(key) or []:
                if isinstance(item, str):
                    parts.append(item)
        _merge(res, validate_text("\n".join(parts), facts, [ticker], default_ticker=ticker))
    return res.block()


def _merge(res: _Result, block: dict) -> None:
    res.checked += block.get("checked") or 0
    res.verified += block.get("verified") or 0
    res.issues.extend(block.get("issues") or [])


# ── never-raise wrappers ─────────────────────────────────────────────────────
def error_block(exc: Exception) -> dict:
    return {
        "version": VALIDATION_VERSION, "checked": 0, "verified": 0, "contradicted": 0,
        "unverifiable": 0, "score": None, "pass": None,
        "issues": [{"severity": "warn", "kind": "validator_error", "ticker": None,
                    "claim": "", "expected": f"validation did not run: {type(exc).__name__}: {exc}"[:240]}],
    }


def safe_validate_text(text, facts, known_tickers=None, default_ticker=None) -> dict:
    try:
        return validate_text(text, facts, known_tickers, default_ticker)
    except Exception as exc:  # noqa: BLE001 — validation must never sink the caller
        return error_block(exc)


def safe_validate_signals(signals, ticker, facts=None):
    try:
        return validate_signals(signals, ticker, facts)
    except Exception as exc:  # noqa: BLE001
        return (signals if isinstance(signals, list) else []), error_block(exc)


def safe_validate_analysis(analysis, ticker, facts=None) -> dict:
    try:
        return validate_analysis(analysis, ticker, facts)
    except Exception as exc:  # noqa: BLE001
        return error_block(exc)
