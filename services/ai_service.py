"""
AI Service abstraction layer.

This module provides an abstract base class and a mock implementation.
To add a real AI provider:
  1. Create a new class inheriting from AIService (e.g. LiteLLMAIService)
  2. Implement all abstract methods
  3. Set AI_PROVIDER=litellm in your .env file

The rest of the application always calls get_ai_service() to get the current
implementation, so switching providers requires zero changes elsewhere.
"""

import json
import re
import logging
from abc import ABC, abstractmethod
from config import get_settings

import httpx
from fastapi import Depends

from auth.deps import get_current_user
from database.models import User

logger = logging.getLogger(__name__)


def _strip_json_fences(raw: str) -> str:
    return raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


# ── Abstract Interface ─────────────────────────────────────────────────────

class AIService(ABC):
    """
    Contract that all AI provider implementations must fulfil.
    All methods are async and return JSON-serialisable dicts / lists / strings.
    """

    @abstractmethod
    async def analyze_stock(self, ticker: str, data: dict) -> dict:
        """
        Full stock analysis.

        Args:
            ticker: Stock symbol
            data: Enriched quote dict (price, fundamentals, technicals, score)

        Returns:
            {
                "summary": str,           # 2-4 sentence narrative
                "sentiment": str,         # "bullish" | "bearish" | "neutral"
                "confidence": float,      # 0.0 – 1.0
                "key_factors": [str],     # bullet-point observations
                "ai_score": int | None,   # 1-10 overall AI rating
                "risks": [str],           # key risk factors
                "opportunities": [str],   # key opportunity factors
            }
        """

    @abstractmethod
    async def generate_signals(self, ticker: str, technicals: dict) -> list[dict]:
        """
        AI-generated trading signals based on technical data.

        Args:
            ticker: Stock symbol
            technicals: {rsi, macd_sig, vs_ma50, vs_ma200, ma_state, gc, dc,
                         vol_r, p52w, ann_ret_1m, chg_pct, score}
                Quote schema v2 semantics: `gc`/`dc` are crossover EVENTS (MA50
                crossed MA200 within the last 5 bars), while `ma_state`
                ("bull"/"bear"/None) is the standing MA50-vs-MA200 condition.
                `vol_r` is today's volume / prior 20-day average (single bar);
                `ann_ret_1m` is the 21-bar annualized return, NOT the
                full-history `ann_ret`. `p52w` is a true 252-bar range position.

        Returns:
            [
                {
                    "type": str,       # "entry" | "exit" | "warning" | "info"
                    "signal": str,     # short label
                    "message": str,    # explanation
                    "strength": str,   # "strong" | "moderate" | "weak"
                }
            ]
        """

    @abstractmethod
    async def chat(self, ticker: str, question: str, context: dict) -> str:
        """
        Answer a natural-language question about a stock.

        Args:
            ticker: Stock symbol
            question: User's question
            context: Full enriched quote dict for grounding

        Returns:
            Plain text answer (markdown OK)
        """

    @abstractmethod
    async def summarize_sector(self, sector: str, stocks: list[dict]) -> str:
        """
        Generate a sector-level market commentary.

        Args:
            sector: Sector name
            stocks: List of enriched quote dicts in that sector

        Returns:
            Markdown narrative (2-4 sentences)
        """


# ── Mock Implementation (default — no API key required) ───────────────────

class MockAIService(AIService):
    """
    Returns placeholder responses so the UI can be built and tested
    without a real AI provider configured.
    Replace with AnthropicAIService or OpenAIAIService when ready.
    """

    async def analyze_stock(self, ticker: str, data: dict) -> dict:
        rsi = data.get("rsi")
        vs_ma200 = data.get("vs_ma200")
        vs_ma50 = data.get("vs_ma50")
        score = (data.get("score") or {}).get("o", 0)
        return {
            "summary": (
                f"Swing trading analysis for {ticker} requires a configured AI provider. "
                "Set AI_PROVIDER=litellm, anthropic, or openai and the corresponding "
                "API key in your .env to enable real AI-powered swing trading strategy analysis."
            ),
            "sentiment": "neutral",
            "confidence": 0.0,
            "key_factors": [
                f"RSI: {round(rsi, 1) if rsi else '—'} — {'Neutral zone (40-65), good for entry' if rsi and 40 <= rsi <= 65 else 'Outside ideal swing entry zone' if rsi else 'N/A'}",
                f"vs MA200: {'+' if vs_ma200 and vs_ma200 > 0 else ''}{round(vs_ma200, 1) if vs_ma200 else '—'}% — {'Above 200MA (uptrend)' if vs_ma200 and vs_ma200 > 0 else 'Below 200MA (downtrend)' if vs_ma200 else 'N/A'}",
                f"Overall score: {score}/5 — configure an AI provider for a full swing trade plan",
            ],
            "entry_strategy": "Configure an AI provider (LiteLLM/Anthropic/OpenAI) to see specific entry signals, price levels, and setup conditions.",
            "exit_strategy": "Configure an AI provider to see profit targets, stop-loss levels, and trailing stop recommendations.",
            "ai_score": None,
            "risks": ["AI provider not configured — set AI_PROVIDER in .env"],
            "opportunities": ["Enable LiteLLM for local/free analysis, or add Anthropic/OpenAI key"],
        }

    async def generate_signals(self, ticker: str, technicals: dict) -> list[dict]:
        return [
            {
                "type": "info",
                "signal": "AI Not Configured",
                "message": "Connect an AI provider in .env to see AI-generated trading signals.",
                "strength": "weak",
            }
        ]

    async def chat(self, ticker: str, question: str, context: dict) -> str:
        return (
            f"AI chat is not configured. To enable it, set `AI_PROVIDER=litellm`, "
            f"`LITELLM_URL`, and the model names in your `.env` file, then restart the container."
        )

    async def summarize_sector(self, sector: str, stocks: list[dict]) -> str:
        return f"AI sector analysis for {sector} is not configured."


# ── Model-alias hygiene ────────────────────────────────────────────────────
# `AI_MODEL`/`REPORT_MODEL` are LiteLLM TIER ALIASES, never raw provider model
# names. The difference is not cosmetic: an alias is remapped on the LiteLLM
# side, so the physical model can be swapped without touching this app, while a
# raw name pins this app to a model that may be retired underneath it — and the
# failure shows up as an opaque 400 from the proxy at request time, long after
# the config was edited.
#
# This was a real drift: a deployment was found running `ai_model: qwen3.5-mlx`  model-name-ok
# (a raw name) instead of `tooling_high`. Nothing detected it, because nothing
# looked. `check_model_aliases()` runs at startup and says so loudly.

# Markers that a string names a specific model rather than a tier.
_RAW_MODEL_MARKERS = (
    "/",        # provider-prefixed: "<provider>/<model>"
    ":",        # runtime tag: "<model>:<size>"
)
_RAW_MODEL_PREFIXES = (   # this IS the detector's own marker list
    "gpt-", "claude-", "qwen", "llama", "mistral", "mixtral", "gemma",      # model-name-ok
    "gemini", "deepseek", "phi-", "o1-", "o3-", "o4-", "grok-", "command-", # model-name-ok
)


def looks_like_raw_model_name(value: str | None) -> bool:
    """True when `value` looks like a specific model rather than a tier alias.

    Deliberately conservative — a false positive only produces a startup
    warning, never a refusal, because guessing wrong must not stop the app from
    running with whatever the user actually configured.
    """
    if not value:
        return False
    v = str(value).strip().lower()
    if not v:
        return False
    if any(m in v for m in _RAW_MODEL_MARKERS):
        return True
    return any(v.startswith(p) for p in _RAW_MODEL_PREFIXES)


# Tier-alias families from aiProxy's CLAUDE.md "Tier Aliases" that can write a
# text report or answer a chat. vision_* and embedding are aliases too, but a
# report asked of a camera or embedding model is useless, so they are never
# offered. Order here is the dropdown order.
_TEXT_ALIAS_FAMILIES = ("tooling", "tooling_local", "coding", "coding_local",
                        "knowledge", "knowledge_local")
_TIER_ORDER = ("high", "med", "low")
_TIER_ALIAS_RE = re.compile(r"^(?P<family>[a-z]+(?:_local)?)_(?P<tier>high|med|low)$")


def select_tier_aliases(names) -> tuple[list[str], int]:
    """Filter a LiteLLM `/v1/models` id list down to text-capable tier aliases.

    Returns (aliases_in_display_order, hidden_count). LiteLLM already limits
    `/v1/models` to what the calling virtual key may use, so a key restricted
    to specific models yields exactly that user's assignment; an UNRESTRICTED
    key (or the global fallback key) returns the whole catalog, legacy direct
    names included — which is what put legacy OpenAI-compat and `aLocalModel_*`
    names back in the dropdown. The app only ever calls tier aliases (see AI_MODEL rule), so
    everything else is hidden rather than offered.
    """
    seen, picked = set(), []
    total = 0
    for n in names or []:
        if not isinstance(n, str) or n in seen:
            continue
        seen.add(n)
        total += 1
        m = _TIER_ALIAS_RE.match(n.strip().lower())
        if m and m.group("family") in _TEXT_ALIAS_FAMILIES:
            picked.append((_TEXT_ALIAS_FAMILIES.index(m.group("family")),
                           _TIER_ORDER.index(m.group("tier")), n))
    picked.sort()
    aliases = [n for _f, _t, n in picked]
    return aliases, total - len(aliases)


def check_model_aliases(settings=None) -> list[str]:
    """Return a human-readable warning per misconfigured model setting.

    Returns a list rather than logging directly so it is testable and so the
    caller decides where the message goes.
    """
    s = settings or get_settings()
    problems = []
    for field, value in (("AI_MODEL", getattr(s, "ai_model", None)),
                         ("REPORT_MODEL", getattr(s, "report_model", None))):
        if looks_like_raw_model_name(value):
            problems.append(
                f"{field} is set to {value!r}, which looks like a raw provider model "
                f"name rather than a LiteLLM tier alias (e.g. 'tooling_high'). The app "
                f"will still try it, but it pins you to one model and will break when "
                f"that model is retired on the LiteLLM side."
            )
    return problems


# ── LiteLLM Implementation ─────────────────────────────────────────────────

class LiteLLMAIService(AIService):
    """
    LiteLLM OpenAI-compatible proxy service. LiteLLM is the sole AI backend —
    this app never calls a model runtime directly — and `model` must always be
    a stable tier alias (e.g. `tooling_high`), never a raw provider model name,
    so the physical model behind it can change without a config edit here.
    See services/ai_service.py::looks_like_raw_model_name, which flags the
    mistake at startup instead of letting it fail at request time.
    Set AI_PROVIDER=litellm, LITELLM_URL, and model names in config.
    """

    def __init__(self, api_key: str | None = None):
        settings = get_settings()
        self.base_url = settings.litellm_url.rstrip("/")
        self.model = settings.ai_model
        # Per-user virtual key takes precedence; fall back to the global key.
        self.api_key = api_key or settings.litellm_api_key
        self._timeout = 120.0

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if not self.api_key:
            raise RuntimeError("LITELLM_API_KEY is required when AI_PROVIDER=litellm")
        headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _chat(self, system: str, user: str, model: str | None = None, timeout: float | None = None) -> str:
        model_name = model or self.model
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        url = f"{self.base_url}/v1/chat/completions"
        logger.info(
            "LiteLLM service request base_url=%s model=%s system_chars=%s user_chars=%s timeout=%s",
            self.base_url,
            model_name,
            len(system or ""),
            len(user or ""),
            timeout or self._timeout,
        )
        timeout_seconds = timeout or self._timeout
        client_timeout = httpx.Timeout(timeout_seconds, connect=15.0, read=timeout_seconds, write=60.0, pool=15.0)
        try:
            async with httpx.AsyncClient(timeout=client_timeout) as client:
                resp = await client.post(url, json=payload, headers=self._headers())
        except httpx.TimeoutException as e:
            logger.error(
                "LiteLLM service timeout url=%s model=%s timeout=%s error_type=%s error_repr=%r",
                url,
                model_name,
                timeout_seconds,
                type(e).__name__,
                e,
            )
            raise RuntimeError(f"litellm request timed out after {timeout_seconds:.0f}s ({type(e).__name__})") from e
        except httpx.RequestError as e:
            logger.error(
                "LiteLLM service request error url=%s model=%s error_type=%s error_repr=%r",
                url,
                model_name,
                type(e).__name__,
                e,
            )
            raise RuntimeError(f"litellm request failed before a response: {type(e).__name__}: {e!r}") from e
        else:
            logger.info(
                "LiteLLM service response model=%s status=%s response_chars=%s",
                model_name,
                resp.status_code,
                len(resp.text or ""),
            )
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                logger.error(
                    "LiteLLM service HTTP error url=%s model=%s status=%s body=%s",
                    url,
                    model_name,
                    resp.status_code,
                    resp.text[:1000],
                )
                raise RuntimeError(f"litellm returned HTTP {resp.status_code}: {resp.text[:500]}") from e
            try:
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                logger.error("LiteLLM service parse error model=%s body=%s", model_name, resp.text[:1000])
                raise RuntimeError(f"litellm returned an unexpected response shape: {e}") from e

    async def analyze_stock(self, ticker: str, data: dict) -> dict:
        system = (
            "You are an expert swing trader. Analyze the given stock/ETF technical and fundamental data "
            "and respond with a JSON object ONLY - no markdown, no extra text. "
            "Required keys: summary, sentiment, confidence, key_factors, entry_strategy, "
            "exit_strategy, ai_score, risks, opportunities."
        )
        user = f"Analyze {ticker} for swing trading:\n{json.dumps(data, default=str)}"
        try:
            raw = await self._chat(system, user)
            return json.loads(_strip_json_fences(raw))
        except Exception as e:
            logger.error("LiteLLM analyze_stock failed for %s: %s", ticker, e)
            return {
                "summary": f"LiteLLM analysis failed: {e}",
                "sentiment": "neutral",
                "confidence": 0.0,
                "key_factors": [],
                "ai_score": None,
                "risks": ["LLM unavailable"],
                "opportunities": [],
            }

    async def generate_signals(self, ticker: str, technicals: dict) -> list[dict]:
        system = (
            "You are a swing-trading signal generator. Based on the technical data provided, "
            "return a JSON array of signal objects only - no markdown, no extra text. "
            "Each object: type (entry|exit|warning|info), signal, message, strength."
        )
        user = f"Generate signals for {ticker}:\n{json.dumps(technicals, default=str)}"
        try:
            raw = await self._chat(system, user)
            return json.loads(_strip_json_fences(raw))
        except Exception as e:
            logger.error("LiteLLM generate_signals failed for %s: %s", ticker, e)
            return [{"type": "info", "signal": "LLM Error", "message": str(e), "strength": "weak"}]

    async def chat(self, ticker: str, question: str, context: dict) -> str:
        system = (
            "You are a swing-trading assistant. Answer concisely using the stock data provided. "
            "Use markdown formatting. Keep answers under 300 words."
        )
        user = f"Stock: {ticker}\nData: {json.dumps(context, default=str)}\n\nQuestion: {question}"
        try:
            return await self._chat(system, user)
        except Exception as e:
            logger.error("LiteLLM chat failed for %s: %s", ticker, e)
            return f"LiteLLM error: {e}"

    async def summarize_sector(self, sector: str, stocks: list[dict]) -> str:
        system = "You are a market analyst. Write a 2-4 sentence markdown sector summary based on the stock data provided."
        user = f"Sector: {sector}\nStocks: {json.dumps(stocks[:10], default=str)}"
        try:
            return await self._chat(system, user)
        except Exception as e:
            logger.error("LiteLLM summarize_sector failed for %s: %s", sector, e)
            return f"LiteLLM sector summary unavailable: {e}"


# ── Future: Anthropic Implementation ──────────────────────────────────────
#
# class AnthropicAIService(AIService):
#     def __init__(self):
#         import anthropic
#         self.client = anthropic.AsyncAnthropic(api_key=settings.ai_api_key)
#         self.model = settings.ai_model   # configured alias; never hardcode a model
#
#     async def analyze_stock(self, ticker, data):
#         prompt = f"Analyze this stock data and return JSON: {json.dumps(data)}"
#         response = await self.client.messages.create(
#             model=self.model, max_tokens=1024,
#             messages=[{"role": "user", "content": prompt}]
#         )
#         return json.loads(response.content[0].text)
#
#     # ... implement generate_signals, chat, summarize_sector similarly


# ── Future: OpenAI Implementation ─────────────────────────────────────────
#
# class OpenAIAIService(AIService):
#     def __init__(self):
#         from openai import AsyncOpenAI
#         self.client = AsyncOpenAI(api_key=settings.ai_api_key)
#         self.model = settings.ai_model   # configured alias; never hardcode a model
#
#     async def analyze_stock(self, ticker, data):
#         ...


# ── Factory ───────────────────────────────────────────────────────────────

def get_ai_service(api_key: str | None = None) -> AIService:
    """Build the configured AI service. `api_key` overrides the global LiteLLM key
    (used to give each user their own virtual key)."""
    settings = get_settings()
    provider = settings.ai_provider.lower()

    if provider == "litellm":
        return LiteLLMAIService(api_key=api_key)

    if provider == "none":
        return MockAIService()

    if provider == "anthropic":
        # AnthropicAIService is not implemented — fail loudly, never mock.
        # (No mock fallback: a misconfigured provider must error.)
        raise RuntimeError("AI_PROVIDER=anthropic is not implemented — use 'litellm' (or 'none' for placeholder responses).")

    if provider == "openai":
        # OpenAIAIService is not implemented — fail loudly, never mock.
        # (No mock fallback: a misconfigured provider must error.)
        raise RuntimeError("AI_PROVIDER=openai is not implemented — use 'litellm' (or 'none' for placeholder responses).")

    raise RuntimeError(f"Unknown AI_PROVIDER={provider!r} — expected 'litellm' or 'none'.")


def ai_service(user: "User" = Depends(get_current_user)) -> AIService:
    """FastAPI dependency: build an AI service bound to the current user's
    LiteLLM key (falls back to the global key when the user has none)."""
    return get_ai_service(api_key=getattr(user, "litellm_api_key", None))
