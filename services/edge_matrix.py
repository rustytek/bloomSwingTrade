"""
Strategy x regime EVIDENCE matrix.

WHY THIS EXISTS
---------------
`Strategy.regimes` in services/strategies.py is a hand-written list literal. It
is an *opinion*: "momentum_rotation belongs in trending_bull". The app then uses
that opinion, every single day, to decide what to recommend — even though it
already owns the machinery (services/backtest.py) to find out whether the
opinion is true. A strategy tagged for a quadrant it demonstrably loses money in
will keep being recommended forever, and nothing in the app would ever say so.

This module closes that loop. For every ACTIONABLE strategy it runs the
walk-forward backtest ONCE, then buckets the per-period results by the
`quadrant` that services/backtest.py already stamps on every rebalance period.
One run per strategy, then group — never one run per (strategy, quadrant).

The output is deliberately conservative. Every cell carries its sample size, a
Wilson 95% interval on the win rate, and a `confidence` label; a thin cell can
only ever produce the verdict "unproven". Under-claiming is the correct failure
mode here: this drives real money decisions, and "we don't know yet" is a far
cheaper mistake than "the evidence says stop using this" off eleven periods.

THRESHOLDS (all module-level, all documented, all deliberately conservative)
---------------------------------------------------------------------------
See MIN_PERIODS_* / UNTAGGED_* below. Rationale for each is inline.

CAVEAT: every number in here inherits services/universe.py's survivorship bias
(see UNIVERSE_CAVEAT) — the cells are relative evidence between strategies, not
an absolute forecast.
"""
from __future__ import annotations

import inspect
import math
import time
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from services.backtest import run_walk_forward_backtest, wilson_interval
from services.regime import QUADRANTS, QUADRANT_INFO
from services.stats import bh_adjust, mean_test
from services.strategies import STRATEGIES
from services.universe import UNIVERSE_AS_OF, UNIVERSE_CAVEAT

# ── Sample-size thresholds ──────────────────────────────────────────────────
# A quadrant CELL is a subset of one backtest run, so it is always thinner than
# the run as a whole. These are cell-level thresholds and are intentionally
# lower than services/backtest.py's run-level 30/100, but never low enough to
# let a handful of periods masquerade as evidence.

# Below this a cell is "insufficient" and can only ever be "unproven". 20
# rebalance periods ~= 100 trading days at the default 5-day cadence, i.e. a
# minimum of roughly five months spent in that quadrant.
MIN_PERIODS_JUDGE = 20

# At/above this a cell is "high" confidence; between the two it is "low".
MIN_PERIODS_HIGH = 50

# To DEMOTE a tagged strategy ("mis-tagged") we require an adequate sample AND
# losses on both measures (mean period return and compounded return). A cell
# that is negative on one and positive on the other is a coin flip, not a case.
MIN_PERIODS_DEMOTE = MIN_PERIODS_JUDGE

# To PROMOTE an untagged cell ("untagged-edge") the bar is higher than for
# confirming a tag: a bigger sample, a materially positive mean, a positive
# compounded return, and a Wilson LOWER bound above a coin flip. Promotion adds
# a strategy to the live playbook in a regime nobody designed it for, so it must
# clear more than "not negative".
MIN_PERIODS_PROMOTE = 30
UNTAGGED_MIN_AVG_RETURN = 0.25   # percent, mean per-period return
UNTAGGED_MIN_CI_LOW = 50.0       # percent, Wilson 95% lower bound on win rate

# ── Out-of-time confirmation and multiple testing (ML4T gap 2) ─────────────
# The matrix tests every strategy in every quadrant — 32+ cells — so with a
# "positive mean on 20 periods" bar several cells pass or fail by luck alone,
# and the history that produced the verdict is the only history it was judged
# on. Since METHOD_VERSION 4 a confident verdict needs all three of:
#   1. the full-sample mean distinguishable from zero after a Benjamini-
#      Hochberg correction across every judged cell (FDR_Q), using an AR(1)
#      effective sample size (services/stats.py::mean_test);
#   2. the DISCOVERY segment (before HOLDOUT_START) pointing the same way;
#   3. the HOLDOUT segment (HOLDOUT_START onward) pointing the same way.
# The split puts a bear market on each side: discovery holds 2008, 2011,
# 2015-16 and late 2018; the holdout holds the 2020 crash and 2022.
# The holdout stays meaningful only while the hand-written Strategy.regimes
# tags are not edited in response to it — change a tag because of a holdout
# number and the holdout is spent (it becomes in-sample).
HOLDOUT_START = "2019-01-01"
MIN_SEGMENT_PERIODS = 8          # per segment; below it the segment can't agree or disagree
FDR_Q = 0.10                     # Benjamini-Hochberg false-discovery rate

# Cache: mirrors the per-user TTL pattern in services/today.py. The matrix is
# expensive (8 strategies x a full walk-forward over ~450 tickers), so the TTL
# is long and the API exposes an explicit refresh instead of rebuilding on read.
_TTL_SECONDS = 12 * 60 * 60
_cache: dict[tuple, tuple[dict, float]] = {}

# Whether services/backtest.py already grew the trade_plan simulator. When it
# has, we prefer it: it produces a real per-trade expectancy_r (stops, targets
# and sizing simulated) instead of an un-risk-managed rotation return.
_BACKTEST_PARAMS = set(inspect.signature(run_walk_forward_backtest).parameters)
_SUPPORTS_TRADE_PLAN = "mode" in _BACKTEST_PARAMS
DEFAULT_MODE = "trade_plan" if _SUPPORTS_TRADE_PLAN else "rotation"

# Bump whenever the way cells are MEASURED changes, so a matrix built by the
# old method is never served as if it were current (see is_current()).
#   2 — builds run with spy_regime=False, and periods that held nothing are
#       excluded from a cell (reported as `idle`). v1 ran every strategy with
#       the SPY>200MA entry filter ON, which blocks every entry in exactly the
#       regimes defined by SPY<200MA — so trending_bear / choppy_volatile cells
#       were almost entirely cash periods scored as 0% "results".
#   3 — fills moved to the session AFTER each decision (next-day limit entries,
#       next-open time/regime exits, open-to-open rotation). v2 cells were
#       measured with same-close fills and overstate breakout/momentum edges.
#   4 — verdicts need a Benjamini-Hochberg-significant mean (AR(1) effective
#       n) AND agreement in both the discovery and the holdout segment (see
#       HOLDOUT_START). v3 confirmed any cell with a positive mean on 20
#       periods, uncorrected for testing 32 cells on one history.
METHOD_VERSION = 4

VERDICTS = ("confirmed", "unproven", "mis-tagged", "untagged-edge")

VERDICT_INFO = {
    "confirmed": "Tagged for this regime and the backtest agrees — positive edge on an adequate sample.",
    "unproven": "Not enough evidence either way. Too few periods in this regime to judge.",
    "mis-tagged": "Tagged for this regime but LOSES money here on an adequate sample. The tag is wrong.",
    "untagged-edge": "NOT tagged for this regime, yet shows a solid positive edge here. Candidate to add.",
}


# ── numeric guards ──────────────────────────────────────────────────────────
def _num(value, default=None):
    """Any non-finite value becomes `default`. Starlette serializes with
    allow_nan=False — one NaN anywhere in this payload 500s the endpoint."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _round(value, digits=2, default=None):
    f = _num(value)
    return round(f, digits) if f is not None else default


def cell_confidence(n: int) -> str:
    """Cell-level sample-size label. See MIN_PERIODS_* above."""
    if n < MIN_PERIODS_JUDGE:
        return "insufficient"
    if n < MIN_PERIODS_HIGH:
        return "low"
    return "high"


# ── verdicts ────────────────────────────────────────────────────────────────
def judge_cell(cell: dict, is_tagged: bool) -> tuple[str, str]:
    """(verdict, reason) for one (strategy, quadrant) cell.

    Hard rule: a thin cell NEVER produces a confident verdict. A low-n losing
    cell is "unproven", not "mis-tagged" — we do not strip a strategy out of the
    user's playbook on eleven periods of noise.

    Since METHOD_VERSION 4 the sign of the cell is only the CANDIDATE verdict;
    it stands only if the mean survives the multiple-testing correction
    (`q_value` <= FDR_Q, set by build_edge_matrix across all judged cells) and
    both the discovery and the holdout segment point the same way. `reason`
    says which gate stopped it, for the UI.
    """
    n = int(cell.get("n") or 0)
    avg = _num(cell.get("avg_period_return"))
    cum = _num(cell.get("cum_return"))
    ci_low = _num(cell.get("win_rate_ci_low"))

    if n < MIN_PERIODS_JUDGE or avg is None or cum is None:
        return "unproven", f"only {n} invested periods — {MIN_PERIODS_JUDGE} needed before judging"

    if is_tagged:
        if avg > 0 and cum > 0:
            candidate, sign = "confirmed", 1
        elif n >= MIN_PERIODS_DEMOTE and avg < 0 and cum < 0:
            candidate, sign = "mis-tagged", -1
        else:
            # Mixed signs / dead flat: honest answer is "we can't tell".
            return "unproven", "average and compounded return disagree in sign"
    else:
        if (
            n >= MIN_PERIODS_PROMOTE
            and avg >= UNTAGGED_MIN_AVG_RETURN
            and cum > 0
            and ci_low is not None
            and ci_low >= UNTAGGED_MIN_CI_LOW
        ):
            candidate, sign = "untagged-edge", 1
        else:
            return "unproven", "not strong enough to add an untagged regime"

    q = _num(cell.get("q_value"))
    if q is None or q > FDR_Q:
        shown = "n/a" if q is None else f"{q:.2f}"
        return "unproven", (f"not distinguishable from noise after correcting for every cell "
                            f"tested (q={shown}, need <= {FDR_Q:.2f})")

    for key, label in (("discovery", f"before {HOLDOUT_START[:4]}"),
                       ("holdout", f"{HOLDOUT_START[:4]} onward")):
        seg = cell.get(key) or {}
        seg_n = int(seg.get("n") or 0)
        seg_avg = _num(seg.get("avg_period_return"))
        if seg_n < MIN_SEGMENT_PERIODS or seg_avg is None:
            return "unproven", (f"{key} segment ({label}) has {seg_n} periods — "
                                f"{MIN_SEGMENT_PERIODS} needed to confirm out of time")
        if seg_avg * sign <= 0:
            return "unproven", f"{key} segment ({label}) points the other way ({seg_avg:+.2f}%/period)"

    return candidate, "significant after multiple-testing correction; discovery and holdout agree"


def classify_verdict(cell: dict, is_tagged: bool) -> str:
    """One of VERDICTS for a single cell — see judge_cell for the rules."""
    return judge_cell(cell, is_tagged)[0]


# ── cell construction ───────────────────────────────────────────────────────
def _segment(rets: list[float]) -> dict:
    n = len(rets)
    cum = 1.0
    for r in rets:
        cum *= (1 + r / 100.0)
    return {
        "n": n,
        "avg_period_return": _round(sum(rets) / n, 2) if n else None,
        "win_rate": _round(sum(1 for r in rets if r > 0) / n * 100, 1) if n else None,
        "cum_return": _round((cum - 1) * 100, 2) if n else None,
    }


def _cell_from_returns(quadrant: str, rets: list[float], idle: int = 0,
                       dates: list[str] | None = None) -> dict:
    """`rets` are periods the strategy was actually INVESTED in, in date order.
    `idle` counts periods in this regime where it held nothing — reported,
    never scored: a cash period says nothing about how the strategy trades in
    that regime. `dates` (ISO, parallel to `rets`) enables the discovery /
    holdout split; without it the cell has no segments and cannot be judged
    out of time."""
    n = len(rets)
    wins = [r for r in rets if r > 0]
    cum = 1.0
    for r in rets:
        cum *= (1 + r / 100.0)
    ci_low, ci_high = wilson_interval(len(wins), n) if n else (0.0, 100.0)
    test = mean_test(rets)
    cell = {
        "quadrant": quadrant,
        "label": QUADRANT_INFO[quadrant]["label"],
        "n": n,
        "idle_periods": idle,
        "avg_period_return": _round(sum(rets) / n, 2) if n else None,
        "win_rate": _round(len(wins) / n * 100, 1) if n else None,
        "win_rate_ci_low": _round(ci_low, 1, 0.0),
        "win_rate_ci_high": _round(ci_high, 1, 100.0),
        "cum_return": _round((cum - 1) * 100, 2) if n else None,
        "confidence": cell_confidence(n),
        "t_stat": _round(test["t_stat"], 2),
        "p_value": _round(test["p_value"], 4, 1.0),
        "n_effective": _round(test["n_eff"], 1, 0.0),
        "q_value": None,   # set across the whole matrix by build_edge_matrix
    }
    if dates is not None and len(dates) == n:
        cell["discovery"] = _segment([r for r, d in zip(rets, dates) if str(d) < HOLDOUT_START])
        cell["holdout"] = _segment([r for r, d in zip(rets, dates) if str(d) >= HOLDOUT_START])
    return cell


def apply_verdicts(strategies: dict) -> int:
    """Benjamini-Hochberg across every judged cell in the matrix, then the
    verdict of every cell. Mutates `strategies` in place; returns how many
    cells the correction covered (reported, so the UI can say "of N tests")."""
    judged = [
        c for entry in strategies.values() for c in (entry.get("cells") or {}).values()
        if int(c.get("n") or 0) >= MIN_PERIODS_JUDGE and _num(c.get("p_value")) is not None
    ]
    for c, q in zip(judged, bh_adjust([_num(c.get("p_value")) for c in judged])):
        c["q_value"] = _round(q, 4, 1.0)
    for entry in strategies.values():
        for c in (entry.get("cells") or {}).values():
            c["verdict"], c["verdict_reason"] = judge_cell(c, bool(c.get("tagged")))
            c["verdict_detail"] = VERDICT_INFO[c["verdict"]]
    return len(judged)


def actionable_strategy_ids() -> list[str]:
    return [sid for sid, s in STRATEGIES.items() if getattr(s, "actionable", True)]


def _run_one(db: Session, user_id: int, strategy_id: str, source: str, kwargs: dict,
             progress=None) -> dict:
    """One walk-forward run, bucketed into quadrant cells.

    run_walk_forward_backtest is called with KEYWORD arguments only and only
    with parameters that exist in its current signature — it is being extended
    concurrently and we must not depend on anything that isn't there today.
    """
    call = {k: v for k, v in kwargs.items() if k in _BACKTEST_PARAMS}
    if progress is not None and "progress" in _BACKTEST_PARAMS:
        call["progress"] = progress
    try:
        result = run_walk_forward_backtest(
            db=db, user_id=user_id, strategy_id=strategy_id, source=source, **call
        )
    except Exception as exc:  # noqa: BLE001 — one bad strategy must not sink the matrix
        return {"error": f"{type(exc).__name__}: {exc}", "cells": {}, "overall": {}}

    trades = result.get("trades") or []
    by_q: dict[str, list[float]] = {q: [] for q in QUADRANTS}
    dates_q: dict[str, list[str]] = {q: [] for q in QUADRANTS}
    idle: dict[str, int] = {q: 0 for q in QUADRANTS}
    for t in trades:
        q = t.get("quadrant")
        if q not in by_q:
            continue
        # Held nothing all period (and exited nothing during it): cash, not a
        # measurement of the strategy. Counted separately, never scored.
        if not t.get("holdings") and not t.get("exits"):
            idle[q] += 1
            continue
        r = _num(t.get("period_return"))
        if r is not None:
            by_q[q].append(r)
            dates_q[q].append(str(t.get("date") or ""))

    metrics = result.get("metrics") or {}
    overall = {
        "periods": len(trades),
        "cagr": _round(metrics.get("cagr"), 2),
        "sharpe": _round(metrics.get("sharpe"), 2),
        "max_drawdown": _round(metrics.get("max_drawdown"), 2),
        # Per-TRADE stats only exist in trade_plan mode; absent -> None, never 0.
        "expectancy_r": _round(metrics.get("expectancy_r"), 3),
        "avg_r": _round(metrics.get("avg_r"), 3),
        "trades_taken": metrics.get("trades_taken"),
        "win_rate_trades": _round(metrics.get("win_rate_trades"), 1),
    }
    return {
        "error": None,
        "cells": {q: _cell_from_returns(q, rets, idle[q], dates_q[q]) for q, rets in by_q.items()},
        "overall": overall,
        "data_quality": result.get("data_quality"),
    }


def build_edge_matrix(db: Session, user_id: int, source: str = "universe",
                      progress=None, **backtest_kwargs) -> dict:
    """Run every actionable strategy once and bucket its periods by regime.

    Extra keyword arguments are forwarded to run_walk_forward_backtest (filtered
    to parameters that actually exist in its signature today). The result is
    cached per (user, source, kwargs) — see get_cached_matrix/invalidate_cache.

    `progress` is an optional `(fraction: float, detail: str) -> None` callback.
    This build takes MINUTES, so it must never be run inside an HTTP request —
    it is driven by services/job_worker.py, and the callback is how that worker
    keeps its heartbeat alive and tells the polling page where it has got to. A
    failing callback is swallowed: progress reporting must not sink a build.
    """
    kwargs = _normalize_kwargs(backtest_kwargs)

    def _tick(fraction: float, detail: str) -> None:
        if progress is None:
            return
        try:
            progress(fraction, detail)
        except Exception:  # noqa: BLE001
            pass

    started = time.time()
    strategies: dict[str, dict] = {}
    errors: list[dict] = []
    data_quality = None

    _all = actionable_strategy_ids()
    _total = max(1, len(_all))
    for _i, sid in enumerate(_all):
        strat = STRATEGIES[sid]
        # Reported BEFORE the run, so the detail names the strategy currently
        # being worked on rather than the one just finished.
        _tick(_i / _total, f"{getattr(strat, 'name', sid)} ({_i + 1}/{_total})")
        tagged = list(getattr(strat, "regimes", []) or [])
        label = getattr(strat, "name", sid)

        def _sub(fraction, detail, _i=_i, _label=label):
            # A 20-year run is minutes per strategy; ticking from INSIDE it
            # keeps the job's heartbeat well within jobs.STALE_AFTER.
            _tick((_i + max(0.0, min(1.0, float(fraction)))) / _total,
                  f"{_label} ({_i + 1}/{_total}): {detail}")

        run = _run_one(db, user_id, sid, source, kwargs, progress=_sub)
        if run["error"]:
            errors.append({"strategy": sid, "error": run["error"]})
        if data_quality is None and run.get("data_quality"):
            data_quality = run["data_quality"]

        cells = {}
        for q in QUADRANTS:
            cell = run["cells"].get(q) or _cell_from_returns(q, [])
            cell = dict(cell)
            cell["tagged"] = q in tagged
            cells[q] = cell   # verdicts come after every strategy has run (apply_verdicts)

        strategies[sid] = {
            "id": sid,
            "name": getattr(strat, "name", sid),
            "tagged_regimes": tagged,
            "cells": cells,
            "overall": run["overall"],
            "error": run["error"],
        }

    tests_corrected = apply_verdicts(strategies)
    discovery_periods = sum(
        int(((c.get("discovery") or {}).get("n")) or 0)
        for entry in strategies.values() for c in entry["cells"].values()
    )

    matrix = {
        "method_version": METHOD_VERSION,
        "holdout_start": HOLDOUT_START,
        "fdr_q": FDR_Q,
        "tests_corrected_for": tests_corrected,
        "spy_regime_filter": kwargs.get("spy_regime"),
        "data_quality": data_quality,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "build_seconds": round(time.time() - started, 1),
        "source": source,
        "mode": kwargs.get("mode", "rotation"),
        "quadrants": [{"id": q, **QUADRANT_INFO[q]} for q in QUADRANTS],
        "strategies": strategies,
        "verdict_info": VERDICT_INFO,
        "thresholds": {
            "min_periods_judge": MIN_PERIODS_JUDGE,
            "min_periods_high_confidence": MIN_PERIODS_HIGH,
            "min_periods_demote": MIN_PERIODS_DEMOTE,
            "min_periods_promote": MIN_PERIODS_PROMOTE,
            "untagged_min_avg_return": UNTAGGED_MIN_AVG_RETURN,
            "untagged_min_ci_low": UNTAGGED_MIN_CI_LOW,
            "min_segment_periods": MIN_SEGMENT_PERIODS,
            "fdr_q": FDR_Q,
            "holdout_start": HOLDOUT_START,
        },
        "errors": errors,
        "caveats": [
            {
                "id": "survivorship_bias",
                "severity": "high",
                "title": "Survivorship bias inflates every cell",
                "detail": UNIVERSE_CAVEAT,
                "universe_as_of": UNIVERSE_AS_OF,
            },
            {
                "id": "relative_not_absolute",
                "severity": "info",
                "title": "Read this matrix as relative, not absolute",
                "detail": (
                    "Every strategy here is measured on the same biased universe over the "
                    "same periods, so comparing cells against each other is meaningful. "
                    "Reading any single cell as a forecast of future return is not."
                ),
            },
            {
                "id": "no_regime_filter",
                "severity": "info",
                "title": "Measured without the SPY 200-day entry filter",
                "detail": (
                    "The filter blocks every entry whenever SPY is below its 200-day MA — which "
                    "is how trending_bear and most of choppy_volatile are defined — so with it on, "
                    "those cells would only measure cash. Periods a strategy spent holding "
                    "nothing are excluded from each cell and shown as idle."
                ),
            },
            {
                "id": "multiple_testing",
                "severity": "info",
                "title": f"Verdicts are corrected for {tests_corrected} simultaneous tests",
                "detail": (
                    f"Testing every strategy in every regime means some cells look good or bad by "
                    f"luck alone. A cell is only confirmed, mis-tagged or promoted when its average "
                    f"return is significant at a {FDR_Q:.0%} false-discovery rate (Benjamini-Hochberg) "
                    f"using a sample size shrunk for correlation between consecutive periods, AND the "
                    f"periods before {HOLDOUT_START[:4]} and from {HOLDOUT_START[:4]} on both point the "
                    f"same way. Expect many cells to stay unproven — that is the honest answer."
                ),
            },
            {
                "id": "holdout_integrity",
                "severity": "info",
                "title": f"The {HOLDOUT_START[:4]}+ holdout only counts while the regime tags are left alone",
                "detail": (
                    "The regime tags on each strategy are hand-written, not fitted to this data, which is "
                    "what lets the later segment act as an out-of-time check. Editing a tag because of a "
                    "number in this matrix spends the holdout: after that, the check is in-sample."
                ),
            },
        ],
    }
    if discovery_periods == 0:
        matrix["caveats"].append({
            "id": "no_discovery_segment",
            "severity": "high",
            "title": f"No history before {HOLDOUT_START[:4]} — nothing can be confirmed",
            "detail": (
                f"Every tested period falls on or after {HOLDOUT_START}, so there is no earlier segment "
                "to check the holdout against and every cell stays unproven. Download the 20-year "
                "history (panel above, admin) and rebuild the matrix."
            ),
        })
    if data_quality:
        from services.backtest import data_quality_caveats
        matrix["caveats"].extend(data_quality_caveats(data_quality))
    matrix["evidence_regimes"] = evidence_backed_regimes(matrix)
    _cache[_cache_key(user_id, source, kwargs)] = (matrix, time.time())
    _cache[(user_id, "__latest__")] = (matrix, time.time())
    return matrix


def evidence_backed_regimes(matrix: dict | None) -> dict[str, list[str]]:
    """The quadrants each strategy has EARNED from the data — the evidence-side
    counterpart to the hand-written `Strategy.regimes` literal.

    Only "confirmed" and "untagged-edge" cells count. "unproven" deliberately
    does NOT carry a tag forward: this dict answers "what has the data shown",
    not "what do we still believe". Callers that need the belief (and the safe
    default) use services/regime.strategies_for_regime_with_evidence, which
    falls back to the hand tags rather than emptying the playbook.
    """
    out: dict[str, list[str]] = {}
    if not matrix:
        return out
    for sid, entry in (matrix.get("strategies") or {}).items():
        earned = [
            q for q, cell in (entry.get("cells") or {}).items()
            if cell.get("verdict") in ("confirmed", "untagged-edge")
        ]
        out[sid] = [q for q in QUADRANTS if q in earned]
    return out


def is_current(matrix: dict | None) -> bool:
    """True when `matrix` was built by the current measurement method."""
    return isinstance(matrix, dict) and matrix.get("method_version") == METHOD_VERSION


def _normalize_kwargs(backtest_kwargs: dict) -> dict:
    kwargs = dict(backtest_kwargs)
    kwargs.setdefault("mode", DEFAULT_MODE)
    if "mode" not in _BACKTEST_PARAMS:
        kwargs.pop("mode", None)
    # The matrix measures strategies PER REGIME; the SPY>200MA entry filter
    # would pre-empt exactly the regimes being measured. See METHOD_VERSION.
    if "spy_regime" in _BACKTEST_PARAMS:
        kwargs.setdefault("spy_regime", False)
    # Measure on the 20-year archive (+ cache). Per ticker it falls back to
    # the 5-year cache until the backfill has run, and data_quality says which.
    if "history" in _BACKTEST_PARAMS:
        kwargs.setdefault("history", "20y")
    # Strategy Lab runs are capped at a 5-year window; the matrix is not — it
    # exists to see every regime (2008, 2018, 2020, 2022) and runs in the worker.
    if "max_window_years" in _BACKTEST_PARAMS:
        kwargs.setdefault("max_window_years", None)
    return kwargs


# ── cache plumbing (mirrors services/today.py) ──────────────────────────────
def _cache_key(user_id: int, source: str, kwargs: dict) -> tuple:
    return (user_id, source, tuple(sorted((k, repr(v)) for k, v in kwargs.items())))


def _from_job_table(user_id: int) -> dict | None:
    """The newest successful matrix build from the background_jobs table.

    Builds run in a separate worker process (services/job_worker.py), so they
    never land in THIS process's `_cache`. Without this, the Playbook's
    evidence override, the Weekly Plan's verdicts and the Scorecard's expected
    R never saw a built matrix at all. Read-only; never builds."""
    try:
        from database.db import SessionLocal
        from services import jobs as jobsvc
    except Exception:  # noqa: BLE001
        return None
    db = SessionLocal()
    try:
        payload = jobsvc.result_of(jobsvc.latest_done(db, user_id, "edge_matrix"))
    except Exception:  # noqa: BLE001 — a cache read must never raise into a page
        payload = None
    finally:
        db.close()
    matrix = payload.get("matrix") if isinstance(payload, dict) else None
    return matrix if is_current(matrix) else None


def get_cached_matrix(user_id: int, source: str | None = None, **backtest_kwargs) -> dict | None:
    """The cached matrix if one is fresh AND built by the current method, else
    None. Falls back to the job table (worker-built results). Never builds —
    the build is slow enough that an HTTP handler must be able to say "not
    computed yet"."""
    if source is None:
        key = (user_id, "__latest__")
    else:
        key = _cache_key(user_id, source, _normalize_kwargs(backtest_kwargs))
    entry = _cache.get(key)
    if entry:
        matrix, ts = entry
        if time.time() - ts <= _TTL_SECONDS and is_current(matrix):
            return matrix
        _cache.pop(key, None)
    if source is not None:
        return None
    matrix = _from_job_table(user_id)
    if matrix is not None:
        _cache[key] = (matrix, time.time())
    return matrix


def invalidate_cache(user_id: int | None = None) -> None:
    if user_id is None:
        _cache.clear()
    else:
        for key in [k for k in _cache if k and k[0] == user_id]:
            _cache.pop(key, None)
