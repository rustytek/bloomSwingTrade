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
