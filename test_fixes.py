"""
Regression tests for the review-fix pass (no network, no DB).

Covers:
  1. Settings uses pydantic-v2 model_config (no deprecated `class Config`),
     tolerates unknown .env keys, and declares broker_encryption_key.
  2. journal_close / compute_r_multiple live in services/journal.py and
     api/portfolio.py re-exports the very same objects (no services->api
     layering inversion for the broker fill sync).
  3. services/broker_service.py no longer imports from the api package.
  4. api/ai.py validates tickers with normalize_ticker (consistent 400s).
  5. main.py CORS never combines a wildcard origin with credentials.
  6. get_ai_service() fails loudly for unimplemented/unknown providers and
     returns the mock only for ai_provider="none".
  7. The web-dependency stub below is removed after imports (it must not
     persist process-wide and mask real packages for other test modules).
  8. resolve_max_open_r is defined in services/portfolio_risk.py (shared by
     the API and services layers); api/settings.py re-exports it.

Run:
    python test_fixes.py
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys
import traceback
import types

# ──────────────────────────────────────────────────────────────────────────
# Web-dependency stubs (same mechanism as test_backtest.py — appended to the
# END of sys.meta_path so a real install always wins; inert when present).
# ──────────────────────────────────────────────────────────────────────────
_STUBBED_PACKAGES = ("fastapi", "jose", "passlib", "bcrypt")


class _Stub:
    def __init__(self, name: str):
        self._name = name

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and not kwargs and callable(args[0]) and not isinstance(args[0], _Stub):
            return args[0]
        return _Stub(f"{self._name}()")

    def __getattr__(self, item):
        return _Stub(f"{self._name}.{item}")

    def __repr__(self):
        return f"<stub {self._name}>"


class _StubModule(types.ModuleType):
    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        return _Stub(f"{self.__name__}.{item}")


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in _STUBBED_PACKAGES:
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        module = _StubModule(spec.name)
        module.__path__ = []
        return module

    def exec_module(self, module):
        pass


_STUB_FINDER = _StubFinder()
sys.meta_path.append(_STUB_FINDER)

import config  # noqa: E402
from config import Settings, get_settings  # noqa: E402
from services import journal as journal_svc  # noqa: E402
from services import broker_service  # noqa: E402
from services.ai_service import MockAIService, get_ai_service  # noqa: E402
from services.portfolio_risk import resolve_max_open_r  # noqa: E402
import api.portfolio as portfolio_api  # noqa: E402
import api.settings as settings_api  # noqa: E402

# The stub above is import-time scaffolding only — drop it (and anything it
# provided) so later imports resolve real packages or raise real ImportErrors.
try:
    sys.meta_path.remove(_STUB_FINDER)
except ValueError:
    pass
for _stubbed in [m for m in list(sys.modules) if m.split(".")[0] in _STUBBED_PACKAGES]:
    del sys.modules[_stubbed]

ROOT = os.path.dirname(os.path.abspath(__file__))


def _read(relpath: str) -> str:
    with open(os.path.join(ROOT, relpath), encoding="utf-8") as fh:
        return fh.read()


# ──────────────────────────────────────────────────────────────────────────
# Tiny test runner (same pattern as test_backtest.py)
# ──────────────────────────────────────────────────────────────────────────
_TESTS = []


def test(fn):
    _TESTS.append(fn)
    return fn


@test
def test_settings_model_config():
    """No deprecated `class Config`; unknown keys ignored; broker key declared."""
    assert not hasattr(Settings, "Config"), "class Config must be replaced by model_config"
    assert Settings.model_config.get("extra") == "ignore"
    assert "broker_encryption_key" in Settings.model_fields


@test
def test_journal_lives_in_services():
    """Single journal implementation, re-exported by the API layer."""
    assert portfolio_api.journal_close is journal_svc.journal_close
    assert portfolio_api.compute_r_multiple is journal_svc.compute_r_multiple
    assert broker_service.journal_close is journal_svc.journal_close
    r, stop = journal_svc.compute_r_multiple(100, 120, initial_stop=90, stop_loss=98)
    assert (r, stop) == (2.0, 90)


@test
def test_broker_service_has_no_api_import():
    src = _read(os.path.join("services", "broker_service.py"))
    assert "from api." not in src and "import api." not in src, \
        "services/ must not import from the api/ layer"
    assert "from services.journal import" in src


@test
def test_ai_endpoints_normalize_tickers():
    src = _read(os.path.join("api", "ai.py"))
    assert "ticker.upper()" not in src, "use normalize_ticker for consistent 400s"
    assert src.count("normalize_ticker(ticker)") >= 3


@test
def test_cors_wildcard_without_credentials():
    src = _read("main.py")
    assert "allow_credentials=True" not in src, \
        '"*" origins + credentials are rejected by browsers'
    assert "allow_credentials=bool(_cors_origin)" in src


@test
def test_ai_service_fails_loudly():
    """Unimplemented/unknown providers raise; mock is opt-in via 'none'."""
    from config import get_settings as gs

    sentinel = object()
    saved = os.environ.get("AI_PROVIDER", sentinel)
    try:
        for provider in ("anthropic", "openai", "definitely-not-a-provider"):
            os.environ["AI_PROVIDER"] = provider
            gs.cache_clear()
            try:
                get_ai_service()
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"expected RuntimeError for AI_PROVIDER={provider!r}")
        os.environ["AI_PROVIDER"] = "none"
        gs.cache_clear()
        svc = get_ai_service()
        assert isinstance(svc, MockAIService), type(svc)
    finally:
        if saved is sentinel:
            os.environ.pop("AI_PROVIDER", None)
        else:
            os.environ["AI_PROVIDER"] = saved
        gs.cache_clear()


@test
def test_stub_does_not_persist():
    assert all(f is not _STUB_FINDER for f in sys.meta_path)
    leftovers = [m for m in sys.modules if m.split(".")[0] in _STUBBED_PACKAGES]
    assert not leftovers, f"stub modules leaked into sys.modules: {leftovers}"


@test
def test_budget_rule_lives_in_services():
    """resolve_max_open_r is defined in services; api re-exports it."""
    assert settings_api.resolve_max_open_r is resolve_max_open_r

    class _Chosen:
        max_open_r = 4.0
        max_positions = 8
        risk_pct = 1.0

    class _Derived:
        max_open_r = None
        max_positions = 8
        risk_pct = 1.0

    budget, basis = resolve_max_open_r(_Chosen())
    assert budget == 4.0 and "chosen" in basis
    budget2, basis2 = resolve_max_open_r(_Derived())
    assert budget2 == 8.0 and isinstance(basis2, str)

    src = _read(os.path.join("services", "weekly_plan.py"))
    assert "from api." not in src and "import api." not in src, \
        "services/ must not import from the api/ layer"
    assert "portfolio_risk.resolve_max_open_r" in src


# ──────────────────────────────────────────────────────────────────────────
# ML4T gap 10 — AI output is checked against the numbers it was given
# ──────────────────────────────────────────────────────────────────────────
from services import ai_validation as _aiv  # noqa: E402  (pure module, no web deps)

_FACTS = _aiv.build_facts([
    {"ticker": "NVDA", "price": 120.50, "rsi": 74.2, "chg_pct": 1.8, "pnl_pct": 12.4,
     "vs_ma50": 6.1, "vs_ma200": 22.0},
    {"ticker": "AAPL", "price": 190.00, "rsi": 45.0, "chg_pct": -0.6, "vs_ma200": -3.2},
])


def _kinds(v):
    return [(i["kind"], i["ticker"]) for i in v["issues"]]


@test
def test_ai_validation_correct_text_passes_clean():
    text = (
        "NVDA is trading at $120.40 with an RSI of 74 — overbought, and it is 22% above its 200-day. "
        "AAPL sits below the 200-day MA, RSI 45.\n"
        "NVDA rose 1.8% today. AAPL fell 0.6% today. Your NVDA P&L is +12.4%.\n"
        "It needs to reclaim the 50-day before AAPL looks healthy. AAPL is not oversold.\n\n"
        "| Ticker | Price | RSI | Chg % |\n| :--- | ---: | ---: | ---: |\n"
        "| **NVDA** | $120.50 | 74.2 | +1.8% |\n| AAPL | $190.00 | 45 | -0.6% |\n\n"
        "## MACRO OVERVIEW AND RISK ALL CLEAR\nThe VIX is 18. The Fed held rates. IT spending is up.\n"
    )
    v = _aiv.validate_text(text, _FACTS, ["NVDA", "AAPL", "SPY"])
    assert v["issues"] == [], v["issues"]
    assert v["pass"] is True and v["checked"] >= 10 and v["verified"] == v["checked"], v


@test
def test_ai_validation_flags_fabricated_ticker():
    v = _aiv.validate_text("Consider $ZZZQ and TSLA as new ideas; NVDA looks extended.", _FACTS, ["NVDA"])
    kinds = _kinds(v)
    assert ("unknown_ticker", "ZZZQ") in kinds and ("unknown_ticker", "TSLA") in kinds, kinds
    assert all(i["severity"] == "warn" for i in v["issues"]), v["issues"]
    # Unknown is "unverified", not "wrong": it must not fail the report.
    assert v["pass"] is True and v["unverifiable"] == 2


@test
def test_ai_validation_flags_wrong_price_and_rsi():
    v = _aiv.validate_text("NVDA is trading at $98.00 with an RSI of 28.", _FACTS, ["NVDA"])
    kinds = _kinds(v)
    assert ("price_mismatch", "NVDA") in kinds and ("rsi_mismatch", "NVDA") in kinds, kinds
    assert v["pass"] is False and v["contradicted"] == 2
    by = {i["kind"]: i for i in v["issues"]}
    assert "120.5" in by["price_mismatch"]["expected"] and "74.2" in by["rsi_mismatch"]["expected"]
    # Within tolerance is NOT a mismatch (price ±2 %, RSI ±3).
    ok = _aiv.validate_text("NVDA trades at $118.60, RSI of 71.5.", _FACTS, ["NVDA"])
    assert ok["issues"] == [], ok["issues"]
    # Tables are checked cell by cell against the row's ticker.
    t = _aiv.validate_text("| Ticker | Price | RSI |\n| --- | --- | --- |\n| AAPL | $150.00 | 62 |", _FACTS, ["AAPL"])
    assert ("price_mismatch", "AAPL") in _kinds(t) and ("rsi_mismatch", "AAPL") in _kinds(t), t


@test
def test_ai_validation_flags_contradictions_but_not_conditionals():
    v = _aiv.validate_text("AAPL is above the 200-day. NVDA is deeply oversold here.", _FACTS, ["AAPL", "NVDA"])
    kinds = _kinds(v)
    assert ("contradiction", "AAPL") in kinds and ("contradiction", "NVDA") in kinds, kinds
    nv = next(i for i in v["issues"] if i["ticker"] == "NVDA")
    assert nv["severity"] == "high" and "74.2" in nv["expected"]
    # Negated / conditional phrasing is not a claim about the present.
    c = _aiv.validate_text(
        "AAPL needs to reclaim the 200-day. If NVDA falls below the 50-day, trim. NVDA is not oversold.",
        _FACTS, ["AAPL", "NVDA"])
    assert c["issues"] == [], c["issues"]


@test
def test_ai_validation_missing_metric_is_unverifiable_not_wrong():
    v = _aiv.validate_text("AAPL P&L is +5%.", _FACTS, ["AAPL"])   # no pnl_pct supplied for AAPL
    assert _kinds(v) == [("unverifiable", "AAPL")] and v["pass"] is True, v


@test
def test_ai_validation_drops_malformed_signals():
    raw = [
        {"type": "entry", "signal": "Buy", "message": "RSI of 28 oversold bounce", "strength": "strong"},
        {"type": "moon", "signal": "x", "message": "y", "strength": "huge"},
        "garbage",
        {"type": "INFO", "signal": "Trend", "message": "Holding above the 200-day", "strength": "Weak"},
    ]
    kept, v = _aiv.validate_signals(raw, "NVDA", _FACTS)
    assert [s["signal"] for s in kept] == ["Buy", "Trend"], kept
    assert kept[1]["type"] == "info" and kept[1]["strength"] == "weak"
    assert v["dropped"] == 2 and sum(i["kind"] == "invalid_item" for i in v["issues"]) == 2
    # The kept message's numbers are still checked (RSI 28 vs 74.2).
    assert ("rsi_mismatch", "NVDA") in _kinds(v), v["issues"]
    for bad in ("not json", None, 42, {"weird": True}):
        kept, v = _aiv.validate_signals(bad, "NVDA", _FACTS)
        assert kept == [] and v["pass"] is False, (bad, kept, v)
    # Analysis schema/range checks.
    a = _aiv.validate_analysis({"summary": "Price trades at $120.", "sentiment": "ecstatic",
                                "confidence": 1.7, "ai_score": 8}, "NVDA", _FACTS)
    assert sum(i["kind"] == "invalid_item" for i in a["issues"]) == 2, a["issues"]


@test
def test_ai_validation_failure_never_breaks_the_report():
    """A validator crash becomes a `validator_error` issue; the report is still
    generated, stamped, persisted and returned unchanged."""
    import asyncio
    import json as _json
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from database.db import Base
    from database.models import ReportCache, User
    from services import report_service as rs

    engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()
    db.add(User(id=1, username="u", password_hash="x"))
    db.commit()

    ctx = {"portfolio": [{"ticker": "NVDA", "price": 120.5, "rsi": 74.2}], "watchlist": [],
           "correlation_groups": [],
           "summary": {"total_cost": 100.0, "total_mv": 110.0, "total_pnl_pct": 10.0,
                       "weeks_held": 3, "cost_multiplier": 1.1}}
    model_text = "NVDA is trading at $98.00 with an RSI of 28."

    async def fake_ctx(user_id, _db):
        return ctx

    async def fake_charts(_db):
        return {}

    async def fake_llm(system, user_msg, model=None, api_key=None, **kw):
        return model_text

    saved = (rs._gather_context, rs.get_all_chart_data, rs._call_llm, _aiv.validate_text)
    rs._gather_context, rs.get_all_chart_data, rs._call_llm = fake_ctx, fake_charts, fake_llm
    try:
        out = asyncio.run(rs.generate_daily_report(db, 1, triggered_by="test", model="tooling_high"))
        assert out["markdown"].startswith(model_text), out["markdown"][:80]   # never rewritten
        assert out["validation"]["contradicted"] == 2, out["validation"]
        row = db.query(ReportCache).one()
        assert _json.loads(row.validation_json)["contradicted"] == 2

        def boom(*a, **k):
            raise RuntimeError("validator bug")
        _aiv.validate_text = boom
        out = asyncio.run(rs.generate_daily_report(db, 1, triggered_by="test", model="tooling_high"))
        assert out["markdown"].startswith(model_text)
        kinds = [i["kind"] for i in out["validation"]["issues"]]
        assert kinds == ["validator_error"] and out["validation"]["pass"] is None, out["validation"]
        assert db.query(ReportCache).count() == 2
    finally:
        rs._gather_context, rs.get_all_chart_data, rs._call_llm, _aiv.validate_text = saved
        db.close()


@test
def test_ai_endpoints_return_validation():
    """Every AI endpoint that shows model output annotates it (static check —
    the endpoints need the web stack)."""
    src = _read(os.path.join("api", "ai.py"))
    for needle in ("safe_validate_analysis(", "safe_validate_signals(", "safe_validate_text(",
                   '"validation": stored_validation(report)', '"validation": result.get("validation")'):
        assert needle in src, needle
    assert src.count("safe_validate_text(") >= 2   # ticker chat + market chat



def main() -> int:
    passed = failed = 0
    failures = []
    for fn in _TESTS:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            failures.append((fn.__name__, exc, traceback.format_exc()))
            print(f"FAIL  {fn.__name__}: {exc}")
        else:
            passed += 1
            print(f"PASS  {fn.__name__}")

    print("\n" + "=" * 56)
    print(f"  {passed} passed, {failed} failed, {len(_TESTS)} total")
    print("=" * 56)
    if failures:
        print("\n--- failure tracebacks ---")
        for name, _exc, tb in failures:
            print(f"\n[{name}]\n{tb}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
