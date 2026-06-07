"""
Smoke-test SwingTrader's LiteLLM/OpenAI-compatible chat path.

Examples:
  python scripts/litellm_smoke.py --list-models
  python scripts/litellm_smoke.py --model qwen3.5-mlx --mode short
  python scripts/litellm_smoke.py --model qwen3.5-mlx --mode market
  python scripts/litellm_smoke.py --model nvda/deepseek-ai/deepseek-v4-pro --mode report --timeout 900
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from config import get_settings  # type: ignore  # noqa: E402
except ModuleNotFoundError:
    get_settings = None


class Defaults:
    litellm_url = "http://192.168.0.21:4000"
    ollama_url = "http://192.168.0.21:11434"
    litellm_api_key = ""
    ai_api_key = ""
    ai_model = "ollama/qwen3.5:9-mlx"
    report_model = "ollama/qwen3.5:9-mlx"
    ollama_model = "qwen3.5:9-mlx"


def load_defaults() -> Defaults:
    if get_settings is not None:
        return get_settings()
    import os

    defaults = Defaults()
    defaults.litellm_url = os.getenv("LITELLM_URL", defaults.litellm_url)
    defaults.ollama_url = os.getenv("OLLAMA_URL", defaults.ollama_url)
    defaults.litellm_api_key = os.getenv("LITELLM_API_KEY", defaults.litellm_api_key)
    defaults.ai_api_key = os.getenv("AI_API_KEY", defaults.ai_api_key)
    defaults.ai_model = os.getenv("AI_MODEL", defaults.ai_model)
    defaults.report_model = os.getenv("REPORT_MODEL", defaults.report_model)
    defaults.ollama_model = os.getenv("OLLAMA_MODEL", defaults.ollama_model)
    return defaults


def _pad_to_target(text: str, target_chars: Optional[int]) -> str:
    if not target_chars or len(text) >= target_chars:
        return text
    filler = "\n\n=== SIZE PADDING ===\n" + ("Synthetic context row. " * 1000)
    needed = target_chars - len(text)
    return text + filler[:needed]


def _make_context(mode: str, target_chars: Optional[int] = None) -> str:
    if mode == "short":
        return "Say OK and name the model you are running as."

    sample_position = {
        "ticker": "SAMPLE",
        "shares": 10,
        "avg_cost": 100.0,
        "price": 104.2,
        "market_value": 1042.0,
        "pnl_pct": 4.2,
        "rsi": 55.1,
        "vol_r": 1.15,
        "chg_pct": 0.8,
        "ann_ret": 24.5,
        "sharpe": 1.1,
        "sector": "Technology",
        "macd_sig": "bullish",
    }
    sample_watch = {
        "ticker": "WATCH",
        "price": 42.1,
        "rsi": 49.8,
        "score": {"o": 7},
        "chg_pct": -0.3,
        "sector": "Energy",
        "notes": "Synthetic smoke-test row.",
    }

    if mode == "market":
        portfolio = [{**sample_position, "ticker": f"P{i:02d}"} for i in range(31)]
        watchlist = [{**sample_watch, "ticker": f"W{i:02d}"} for i in range(16)]
        text = (
            "=== PORTFOLIO SMOKE TEST ===\n"
            + json.dumps(portfolio, indent=2)
            + "\n\n=== WATCHLIST SMOKE TEST ===\n"
            + json.dumps(watchlist, indent=2)
            + "\n\nQuestion: reply in one short paragraph with the strongest and weakest synthetic tickers."
        )
        return _pad_to_target(text, target_chars)

    if mode == "report":
        portfolio = [{**sample_position, "ticker": f"P{i:02d}"} for i in range(31)]
        watchlist = [{**sample_watch, "ticker": f"W{i:02d}"} for i in range(16)]
        macro = {
            "m2_current": 22000,
            "m2_trend": "rising",
            "fed_rate": 4.5,
            "yield_2yr": 4.1,
            "yield_10yr": 4.3,
            "yield_spread_now": 0.2,
            "vix_current": 16.4,
            "breadth": {"pct_above_200ma": 58, "pct_above_50ma": 51, "ad_ratio": 1.2},
        }
        template = """
# [DATE] Market Analysis & Execution

## 1. Macro Regime & Liquidity
[ANALYZE macro regime using the JSON data]

## 2. Market Breadth & Ranking
[ANALYZE breadth]

## 3. Portfolio Performance
[INSERT portfolio table]

## 4. Position Logic & Same Shape Analysis
[ANALYZE correlation groups]

## 5. Tactical Actions
[LIST sell/add/watch candidates]
"""
        text = (
            "Today: smoke-test\n\n"
            "=== LIVE MACRO DATA ===\n"
            + json.dumps(macro, indent=2)
            + "\n\n=== PORTFOLIO DATA ===\n"
            + json.dumps(portfolio, indent=2)
            + "\n\n=== WATCHLIST DATA ===\n"
            + json.dumps(watchlist, indent=2)
            + "\n\n=== FILL THIS TEMPLATE ===\n"
            + template
            + "\nReturn completed markdown only."
        )
        return _pad_to_target(text, target_chars)

    raise ValueError(f"Unknown mode: {mode}")


def _headers(api_key: str) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


async def list_models(base_url: str, api_key: str, timeout: float) -> None:
    url = f"{base_url}/v1/models"
    started = time.perf_counter()
    req = Request(url, headers=_headers(api_key), method="GET")
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            status = resp.status
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        elapsed = time.perf_counter() - started
        print(f"GET {url}")
        print(f"status={e.code} elapsed={elapsed:.2f}s bytes={len(body)}")
        print(body[:2000])
        raise SystemExit(1)
    elapsed = time.perf_counter() - started
    print(f"GET {url}")
    print(f"status={status} elapsed={elapsed:.2f}s bytes={len(body)}")
    data = json.loads(body)
    for item in data.get("data", []):
        name = item.get("id") or item.get("name")
        if name:
            print(f"- {name}")


async def chat(base_url: str, api_key: str, model: str, mode: str, timeout: float, target_chars: Optional[int]) -> None:
    url = f"{base_url}/v1/chat/completions"
    system = "You are a concise SwingTrader smoke-test assistant."
    user = _make_context(mode, target_chars)
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    print(f"POST {url}")
    print(f"model={model} mode={mode} timeout={timeout:.0f}s user_chars={len(user)}")
    body = json.dumps(payload).encode("utf-8")
    headers = _headers(api_key)
    headers["Content-Length"] = str(len(body))
    req = Request(url, data=body, headers=headers, method="POST")
    started = time.perf_counter()
    try:
        with urlopen(req, timeout=timeout) as resp:
            resp_body = resp.read().decode("utf-8", errors="replace")
            status = resp.status
    except TimeoutError as e:
        elapsed = time.perf_counter() - started
        print(f"TIMEOUT elapsed={elapsed:.2f}s type={type(e).__name__} repr={e!r}")
        raise
    except HTTPError as e:
        resp_body = e.read().decode("utf-8", errors="replace")
        elapsed = time.perf_counter() - started
        print(f"status={e.code} elapsed={elapsed:.2f}s bytes={len(resp_body)}")
        print(resp_body[:2000])
        raise SystemExit(1)
    except URLError as e:
        elapsed = time.perf_counter() - started
        print(f"REQUEST_ERROR elapsed={elapsed:.2f}s type={type(e).__name__} repr={e!r}")
        raise

    elapsed = time.perf_counter() - started
    print(f"status={status} elapsed={elapsed:.2f}s bytes={len(resp_body)}")
    data = json.loads(resp_body)
    content = data["choices"][0]["message"]["content"]
    print("\n--- response preview ---")
    print(content[:2000])


def parse_args() -> argparse.Namespace:
    settings = load_defaults()
    parser = argparse.ArgumentParser(description="Smoke-test LiteLLM chat completions from SwingTrader config.")
    parser.add_argument("--base-url", default=(settings.litellm_url or settings.ollama_url).rstrip("/"))
    parser.add_argument("--api-key", default=settings.litellm_api_key or settings.ai_api_key)
    parser.add_argument("--model", default=settings.ai_model or settings.report_model or settings.ollama_model)
    parser.add_argument("--mode", choices=["short", "market", "report"], default="short")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--target-chars",
        type=int,
        default=None,
        help="Pad the synthetic user message to this many characters. Defaults: market=13655, report=26639.",
    )
    parser.add_argument("--list-models", action="store_true")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if args.list_models:
        await list_models(args.base_url, args.api_key, args.timeout)
        return
    target_chars = args.target_chars
    if target_chars is None and args.mode == "market":
        target_chars = 13655
    if target_chars is None and args.mode == "report":
        target_chars = 26639
    await chat(args.base_url, args.api_key, args.model, args.mode, args.timeout, target_chars)


if __name__ == "__main__":
    asyncio.run(main())
