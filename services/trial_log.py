"""
Strategy Lab trial log and the Deflated Sharpe Ratio (ML4T 3e §7.4, §16.7).

Every walk-forward a user runs is a TRIAL. Run fifteen variants and keep the
best, and the kept Sharpe is inflated by the search itself — the book's
"backtest overfitting". The fix has two halves, both here:

  1. record every distinct configuration tried (`backtest_runs` table), per
     user and strategy — the trial count is otherwise unknowable;
  2. report the Deflated Sharpe Ratio: the probability the run's Sharpe beats
     the best Sharpe that many skill-less variants would show by luck
     (Bailey & Lopez de Prado 2014), using the spread of Sharpes actually
     observed across the user's trials.

Re-running an identical configuration is not a new trial (it updates the row).
Different windows ARE different trials: trying eras until one looks good is a
search too. Nothing here ever deletes a row — hiding tries defeats the point.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from statistics import pvariance

from sqlalchemy.orm import Session

from database.models import BacktestRun
from services.stats import deflated_sharpe, expected_max_sharpe

# DSR at/above this is "likely real"; below DSR_WEAK "likely luck".
DSR_STRONG = 0.95
DSR_WEAK = 0.50


def params_key(strategy: str, mode: str, params: dict) -> str:
    blob = json.dumps({"strategy": strategy, "mode": mode, "params": params},
                      sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.md5(blob.encode()).hexdigest()


def _window(result: dict) -> tuple[str | None, str | None]:
    eq = result.get("equity") or []
    if not eq:
        return None, None
    return str(eq[0].get("date"))[:10], str(eq[-1].get("date"))[:10]


def record_run(db: Session, user_id: int, result: dict) -> BacktestRun | None:
    """Upsert this run's configuration. Returns the row, or None when the run
    produced no Sharpe to record (e.g. not enough data)."""
    inf = result.get("sharpe_inference") or {}
    params = result.get("parameters") or {}
    strategy = str(params.get("strategy") or (result.get("strategy") or {}).get("id") or "")
    if not strategy or inf.get("sharpe_annual") is None:
        return None
    mode = str(result.get("mode") or "rotation")
    key = params_key(strategy, mode, params)
    start, end = _window(result)
    now = datetime.now(timezone.utc)
    row = (db.query(BacktestRun)
           .filter(BacktestRun.user_id == user_id, BacktestRun.strategy == strategy,
                   BacktestRun.params_key == key)
           .first())
    if row is None:
        row = BacktestRun(user_id=user_id, strategy=strategy, mode=mode, params_key=key,
                          params_json=json.dumps(params, sort_keys=True, default=str),
                          runs=0, first_run_at=now)
        db.add(row)
    row.window_start, row.window_end = start, end
    row.periods = inf.get("periods")
    row.sharpe_annual = inf.get("sharpe_annual")
    row.cagr = (result.get("metrics") or {}).get("cagr")
    row.runs = int(row.runs or 0) + 1
    row.last_run_at = now
    db.commit()
    return row


def selection_bias(inference: dict | None, trial_sharpes_annual: list[float]) -> dict | None:
    """The Deflated Sharpe block for one run, given the annualized Sharpes of
    every distinct trial of that strategy (this run included). Pure."""
    if not inference or inference.get("sharpe_period") is None:
        return None
    ppy = float(inference.get("periods_per_year") or 252 / 5)
    trials = max(1, len(trial_sharpes_annual))
    # Trials may use different rebalance cadences; compare them annualized and
    # convert the spread back to THIS run's per-period units.
    var_annual = pvariance(trial_sharpes_annual) if trials >= 2 else 0.0
    var_period = var_annual / ppy
    sr = float(inference["sharpe_period"])
    n = int(inference.get("periods") or 0)
    dsr = deflated_sharpe(sr, n, float(inference.get("skew") or 0.0),
                          float(inference.get("kurtosis") or 3.0), trials, var_period)
    hurdle = expected_max_sharpe(trials, var_period) * math.sqrt(ppy)
    if dsr >= DSR_STRONG:
        label = "likely_real"
    elif dsr >= DSR_WEAK:
        label = "inconclusive"
    else:
        label = "likely_luck"
    return {
        "trials": trials,
        "trial_sharpe_std_annual": round(math.sqrt(var_annual), 3),
        "luck_hurdle_sharpe_annual": round(hurdle, 3),
        "deflated_sharpe": round(dsr, 4),
        "label": label,
        "detail": (
            f"You have tried {trials} distinct configuration{'s' if trials != 1 else ''} of this "
            f"strategy. The best of that many skill-less tries would show an annualized Sharpe of "
            f"about {hurdle:.2f} by luck alone; the probability this run's Sharpe beats that is "
            f"{dsr:.0%}." if trials > 1 else
            f"First configuration tried for this strategy, so there is no search to deflate for: "
            f"the probability its true Sharpe is above zero is {dsr:.0%}. Every further variant "
            f"you try raises the bar for all of them."
        ),
    }


def trial_sharpes(db: Session, user_id: int, strategy: str) -> list[float]:
    rows = (db.query(BacktestRun.sharpe_annual)
            .filter(BacktestRun.user_id == user_id, BacktestRun.strategy == strategy,
                    BacktestRun.sharpe_annual.isnot(None))
            .all())
    return [float(r[0]) for r in rows]


def annotate(db: Session, user_id: int, result: dict) -> dict:
    """Record the run, then attach `selection_bias` and, when warranted, a
    caveat. Never raises into the request: a logging failure must not lose
    the user a finished backtest."""
    try:
        row = record_run(db, user_id, result)
        if row is None:
            return result
        block = selection_bias(result.get("sharpe_inference"),
                               trial_sharpes(db, user_id, row.strategy))
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        result["selection_bias"] = {"error": f"{type(exc).__name__}: {exc}"}
        return result
    result["selection_bias"] = block
    if block and block["label"] != "likely_real":
        result.setdefault("caveats", []).append({
            "id": "selection_bias",
            "severity": "high" if block["label"] == "likely_luck" else "medium",
            "title": (f"Deflated Sharpe {block['deflated_sharpe']:.0%} — "
                      + ("this result is likely luck" if block["label"] == "likely_luck"
                         else "not distinguishable from luck yet")),
            "detail": block["detail"],
        })
    return result


def list_trials(db: Session, user_id: int, strategy: str | None = None) -> list[dict]:
    q = db.query(BacktestRun).filter(BacktestRun.user_id == user_id)
    if strategy:
        q = q.filter(BacktestRun.strategy == strategy)
    out = []
    for r in q.order_by(BacktestRun.last_run_at.desc()).all():
        out.append({
            "strategy": r.strategy, "mode": r.mode, "params": json.loads(r.params_json or "{}"),
            "window_start": r.window_start, "window_end": r.window_end, "periods": r.periods,
            "sharpe_annual": r.sharpe_annual, "cagr": r.cagr, "runs": r.runs,
            "first_run_at": r.first_run_at.isoformat() if r.first_run_at else None,
            "last_run_at": r.last_run_at.isoformat() if r.last_run_at else None,
        })
    return out
