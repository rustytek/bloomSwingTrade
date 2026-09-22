"""
Pluggable exit-rule framework.

The app has nine ENTRY strategies (services/strategies.py) and, historically,
exactly one exit: a fixed ATR stop with an R-multiple target. Most of the
edge in swing trading lives in the exit, so exits are modelled here as small,
composable, independently testable rules that the walk-forward backtest
(`services/backtest.py`, mode="trade_plan") and, later, the live trade plan
can share.

CONTRACT
--------
    rule.evaluate(bar, state) -> ExitSignal | None

`bar` is a normal OHLCV dict (`date/open/high/low/close/vol`) that the engine
enriches with two derived fields before handing it over:
    bar["atr"]       ATR(14) at THIS bar, precomputed once per ticker.
    bar["quadrant"]  market regime quadrant at this bar (services/regime.py).
A rule may look at that one bar and at `state` only — never at future bars.
That is the whole no-look-ahead guarantee: the rules are structurally
incapable of seeing tomorrow.

Rules may also implement `update(bar, state)`, which the engine calls AFTER a
bar has been evaluated and survived. That is where path-dependent levels (a
trailing stop) advance, so the level checked on bar *t* was always derived
from bars up to *t-1*'s close.

PRIORITY / TIE-BREAKING
-----------------------
A position can carry several rules. When more than one fires on the same bar,
`resolve_exit()` picks the winner by ExitSignal.kind, in this order:

    stop  ->  partial  ->  target  ->  time  ->  regime

i.e. STOP BEFORE TARGET BEFORE TIME, exactly as documented. Ties inside one
kind are broken by `rule.priority` (lower first) and then registration order.

INTRABAR CONVENTION (deliberately pessimistic)
----------------------------------------------
Daily bars do not tell us whether the low or the high came first. When a bar's
low breaches the stop AND its high reaches the target, we ASSUME THE STOP
FILLED FIRST. Backtests that assume the opposite quietly manufacture winners.
Gaps are handled the same way: if a bar OPENS beyond the level, the fill is
the open, not the level (you cannot get filled at a price that never traded).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# Lower number = resolved first when several signals fire on the same bar.
KIND_PRIORITY: dict[str, int] = {
    "stop": 0,
    "partial": 1,
    "target": 2,
    "time": 3,
    "regime": 4,
}


def _finite(v) -> bool:
    """True only for real, finite numbers (rejects None, NaN, +/-inf)."""
    return isinstance(v, (int, float)) and math.isfinite(v)


def _safe_float(value):
    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    except (TypeError, ValueError):
        return None


@dataclass
class ExitSignal:
    """One rule's verdict for one bar."""
    reason: str
    price: float
    kind: str = "stop"              # see KIND_PRIORITY
    rule: str = ""                  # rule id, for attribution in the trade log
    fraction: float = 1.0           # <1.0 = scale out, position stays open


@dataclass
class PositionState:
    """Everything an exit rule is allowed to know about an open position."""
    ticker: str
    entry_date: str
    entry_price: float
    entry_idx: int
    shares: int
    initial_shares: int
    stop: float
    initial_stop: float
    risk_per_share: float
    target: float | None = None
    strategy: str = ""
    highest_close: float = 0.0
    bars_held: int = 0
    scaled_out: bool = False
    trail_stop: float | None = None
    realized_cash: float = 0.0      # proceeds already booked from partials
    events: list[str] = field(default_factory=list)

    def r_at(self, price: float) -> float:
        """R-multiple of `price` measured against the INITIAL risk per share.

        Initial risk is the denominator on purpose: moving a stop up must not
        retroactively inflate the R of a trade.
        """
        if not self.risk_per_share:
            return 0.0
        return (price - self.entry_price) / self.risk_per_share


def _stop_fill(bar: dict, level: float) -> float:
    """Fill price for a downside breach — a gap-through fills at the open."""
    open_ = _safe_float(bar.get("open"))
    if open_ is not None and open_ <= level:
        return open_
    return level


def _target_fill(bar: dict, level: float) -> float:
    """Fill price for an upside touch — a gap-through fills at the open."""
    open_ = _safe_float(bar.get("open"))
    if open_ is not None and open_ >= level:
        return open_
    return level


class ExitRule:
    """Base class. Subclasses override `evaluate` and optionally `update`."""

    id: str = "exit_rule"
    name: str = "Exit Rule"
    priority: int = 50              # tie-break inside one ExitSignal.kind

    def evaluate(self, bar: dict, position_state: PositionState) -> ExitSignal | None:
        raise NotImplementedError

    def update(self, bar: dict, position_state: PositionState) -> None:
        """Advance path-dependent state AFTER a surviving bar. No-op default."""
        return None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.id}>"


class FixedStopTarget(ExitRule):
    """The baseline — today's live behaviour.

    A hard ATR-derived stop plus a fixed R-multiple target, both set at entry
    and never moved (unless another rule, e.g. PartialProfitTaking, moves the
    stop). Stop is checked before target on the same bar.
    """

    id = "fixed"
    name = "Fixed stop + R target"
    priority = 10

    def evaluate(self, bar: dict, position_state: PositionState) -> ExitSignal | None:
        low = _safe_float(bar.get("low"))
        high = _safe_float(bar.get("high"))
        st = position_state
        if low is not None and _finite(st.stop) and low <= st.stop:
            price = _stop_fill(bar, st.stop)
            return ExitSignal(
                reason=f"stop hit at {price:.2f}",
                price=price, kind="stop", rule=self.id,
            )
        if high is not None and st.target and high >= st.target:
            price = _target_fill(bar, st.target)
            return ExitSignal(
                reason=f"target hit at {price:.2f}",
                price=price, kind="target", rule=self.id,
            )
        return None


class AtrTrailingStop(ExitRule):
    """Chandelier-style trailing stop.

    Level = highest CLOSE since entry - atr_mult * ATR, and it never moves
    down. The level checked on bar *t* is the one computed at the close of
    bar *t-1* (see `update`), so a bar can never stop itself out using its own
    close.
    """

    id = "trail"
    name = "ATR trailing stop"
    priority = 5                    # tighter than the fixed stop -> checked first

    def __init__(self, atr_mult: float = 3.0):
        self.atr_mult = float(atr_mult)

    def evaluate(self, bar: dict, position_state: PositionState) -> ExitSignal | None:
        level = position_state.trail_stop
        low = _safe_float(bar.get("low"))
        if level is None or low is None or not _finite(level):
            return None
        if low <= level:
            price = _stop_fill(bar, level)
            return ExitSignal(
                reason=f"ATR trailing stop hit at {price:.2f}",
                price=price, kind="stop", rule=self.id,
            )
        return None

    def update(self, bar: dict, position_state: PositionState) -> None:
        atr = _safe_float(bar.get("atr"))
        if atr is None or atr <= 0:
            return
        candidate = position_state.highest_close - self.atr_mult * atr
        if not _finite(candidate):
            return
        current = position_state.trail_stop
        position_state.trail_stop = candidate if current is None else max(current, candidate)


class TimeStop(ExitRule):
    """Cut a trade that has gone nowhere.

    After `max_bars` bars held, exit at the close unless the trade has reached
    at least `min_r`. Dead capital is a real cost: a position sitting at +0.1R
    for six weeks is occupying a slot another signal could use.
    """

    id = "time"
    name = "Time stop"
    priority = 30

    def __init__(self, max_bars: int = 20, min_r: float = 0.5):
        self.max_bars = int(max_bars)
        self.min_r = float(min_r)

    def evaluate(self, bar: dict, position_state: PositionState) -> ExitSignal | None:
        close = _safe_float(bar.get("close"))
        if close is None or position_state.bars_held < self.max_bars:
            return None
        r_now = position_state.r_at(close)
        if r_now >= self.min_r:
            return None
        return ExitSignal(
            reason=f"time stop: {position_state.bars_held} bars held at {r_now:+.2f}R "
                   f"(< {self.min_r:g}R)",
            price=close, kind="time", rule=self.id,
        )


class PartialProfitTaking(ExitRule):
    """Scale out at a first R target, then ride the rest risk-free.

    At `first_r`, sell `fraction` of the position and move the stop to
    breakeven. Emits an ExitSignal with fraction < 1.0 — the engine treats
    that as a scale-out, not a close.
    """

    id = "partial"
    name = "Partial profit taking"
    priority = 20

    def __init__(self, first_r: float = 1.0, fraction: float = 0.5,
                 move_stop_to_breakeven: bool = True):
        self.first_r = float(first_r)
        self.fraction = min(max(float(fraction), 0.01), 0.99)
        self.move_stop_to_breakeven = bool(move_stop_to_breakeven)

    def evaluate(self, bar: dict, position_state: PositionState) -> ExitSignal | None:
        st = position_state
        if st.scaled_out or not st.risk_per_share:
            return None
        high = _safe_float(bar.get("high"))
        if high is None:
            return None
        level = st.entry_price + self.first_r * st.risk_per_share
        if high < level:
            return None
        price = _target_fill(bar, level)
        return ExitSignal(
            reason=f"scaled out {self.fraction:.0%} at {self.first_r:g}R ({price:.2f})"
                   + (", stop to breakeven" if self.move_stop_to_breakeven else ""),
            price=price, kind="partial", rule=self.id, fraction=self.fraction,
        )


class RegimeExit(ExitRule):
    """Leave when the market stops being the market this strategy trades.

    Exits at the close of the first bar whose regime quadrant is outside the
    strategy's tagged `regimes` (services/strategies.py Strategy.regimes).
    """

    id = "regime"
    name = "Regime exit"
    priority = 40

    def __init__(self, regimes: list[str] | None = None):
        self.regimes = list(regimes or [])

    def evaluate(self, bar: dict, position_state: PositionState) -> ExitSignal | None:
        if not self.regimes:
            return None
        quadrant = bar.get("quadrant")
        if not quadrant or quadrant in self.regimes:
            return None
        close = _safe_float(bar.get("close"))
        if close is None:
            return None
        return ExitSignal(
            reason=f"regime left {'/'.join(self.regimes)} (now {quadrant})",
            price=close, kind="regime", rule=self.id,
        )


def resolve_exit(rules: list[ExitRule], bar: dict,
                 position_state: PositionState) -> ExitSignal | None:
    """First triggering rule wins, in documented priority order.

    Collect every signal fired on this bar, then pick by
    (KIND_PRIORITY[kind], rule.priority, registration order) — which enforces
    stop -> partial -> target -> time -> regime globally, across rules, not
    just inside one rule.
    """
    fired: list[tuple[int, int, int, ExitSignal]] = []
    for order, rule in enumerate(rules):
        signal = rule.evaluate(bar, position_state)
        if signal is None:
            continue
        fired.append((KIND_PRIORITY.get(signal.kind, 99), rule.priority, order, signal))
    if not fired:
        return None
    fired.sort(key=lambda row: (row[0], row[1], row[2]))
    return fired[0][3]


def advance(rules: list[ExitRule], bar: dict, position_state: PositionState) -> None:
    """Called once per surviving bar: age the position, then let rules update."""
    close = _safe_float(bar.get("close"))
    position_state.bars_held += 1
    if close is not None:
        position_state.highest_close = max(position_state.highest_close, close)
    for rule in rules:
        rule.update(bar, position_state)


def apply_partial(signal: ExitSignal, position_state: PositionState,
                  rule: PartialProfitTaking | None = None) -> int:
    """Book a scale-out. Returns the number of shares sold (0 if none)."""
    st = position_state
    sold = int(st.shares * signal.fraction)
    if sold < 1:
        # Too small to split (e.g. a 1-share position) — mark it done so the
        # rule stops re-firing every bar, but don't sell a fractional share.
        st.scaled_out = True
        return 0
    st.shares -= sold
    st.scaled_out = True
    if rule is None or rule.move_stop_to_breakeven:
        st.stop = max(st.stop, st.entry_price)
        if st.trail_stop is not None:
            st.trail_stop = max(st.trail_stop, st.entry_price)
    st.events.append(signal.reason)
    return sold


# ──────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────
RULE_IDS = ("fixed", "trail", "time", "partial", "regime")

#: Used when neither the caller nor the strategy supplies a hold horizon.
DEFAULT_TIME_STOP_BARS = 20


def strategy_horizon_days(strategy_id: str | None) -> int | None:
    """A strategy's MACHINE-READABLE max hold, or None.

    Reads the numeric `horizon_days` / `max_hold_days` class attribute only.
    `Strategy.details["horizon"]` is prose ("3-10 day snap-back") and parsing a
    number out of English would be a guess, so it is never consulted.
    Mirrors services/scorecard.py::_horizon_days deliberately — the number that
    judges a trade as "overstayed" must be the same one that exits it.
    """
    if not strategy_id:
        return None
    try:
        from services.strategies import STRATEGIES
        strat = STRATEGIES.get(strategy_id)
    except Exception:  # noqa: BLE001 — exits must never hard-fail on an import
        return None
    if strat is None:
        return None
    for attr in ("horizon_days", "max_hold_days"):
        val = getattr(strat, attr, None)
        if isinstance(val, (int, float)) and math.isfinite(val) and val > 0:
            return int(val)
    return None


def build_exit_rules(
    names: str | list[str] | None,
    atr_mult: float = 2.5,
    trail_mult: float = 3.0,
    time_stop_bars: int | None = None,
    time_stop_min_r: float = 0.5,
    partial_r: float = 1.0,
    partial_fraction: float = 0.5,
    regimes: list[str] | None = None,
    strategy_id: str | None = None,
) -> list[ExitRule]:
    """Build a rule set from ids ("fixed", "fixed,trail", ["time"], ...).

    "fixed" is always included: every position needs a hard stop, and every
    other rule here only ever tightens or shortens the trade.

    `time_stop_bars` resolution order — an EXPLICIT value from the caller always
    wins; otherwise the strategy's own `horizon_days` is used (a mean-reversion
    snap-back has no business being held for a momentum strategy's 21 bars);
    otherwise DEFAULT_TIME_STOP_BARS (20), which is exactly what every existing
    call site already got.
    """
    if isinstance(names, str):
        wanted = [n.strip() for n in names.split(",") if n.strip()]
    else:
        wanted = [str(n).strip() for n in (names or []) if str(n).strip()]
    wanted = [n for n in wanted if n in RULE_IDS]
    if "fixed" not in wanted:
        wanted.insert(0, "fixed")

    rules: list[ExitRule] = []
    for name in wanted:
        if name == "fixed":
            rules.append(FixedStopTarget())
        elif name == "trail":
            rules.append(AtrTrailingStop(atr_mult=trail_mult or atr_mult))
        elif name == "time":
            bars = time_stop_bars
            if bars is None:
                bars = strategy_horizon_days(strategy_id) or DEFAULT_TIME_STOP_BARS
            rules.append(TimeStop(max_bars=bars, min_r=time_stop_min_r))
        elif name == "partial":
            rules.append(PartialProfitTaking(first_r=partial_r, fraction=partial_fraction))
        elif name == "regime":
            rules.append(RegimeExit(regimes=regimes))
    return rules
