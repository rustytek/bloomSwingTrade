"""
AI Service abstraction layer.

This module provides an abstract base class and a mock implementation.
To add a real AI provider:
  1. Create a new class inheriting from AIService (e.g. AnthropicAIService)
  2. Implement all abstract methods
  3. Set AI_PROVIDER=anthropic (or openai) in your .env file

The rest of the application always calls get_ai_service() to get the current
implementation, so switching providers requires zero changes elsewhere.
"""

import json
import logging
from abc import ABC, abstractmethod
from config import get_settings

import httpx

logger = logging.getLogger(__name__)
settings = get_settings()


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
            technicals: {rsi, macd_sig, vs_ma200, gc, dc, vol_r, p52w, ...}

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
        return {
            "summary": (
                f"AI analysis for {ticker} is not yet configured. "
                "Set AI_PROVIDER=anthropic (or openai) and AI_API_KEY in your .env file "
                "to enable real AI-powered analysis."
            ),
            "sentiment": "neutral",
            "confidence": 0.0,
            "key_factors": [
                "Configure an AI provider to see key factors",
                "Supports Anthropic Claude and OpenAI GPT models",
            ],
            "ai_score": None,
            "risks": ["AI provider not configured"],
            "opportunities": ["Add AI_PROVIDER to .env to unlock this feature"],
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
            f"AI chat is not configured. To enable it, set `AI_PROVIDER=anthropic` "
            f"(or `openai`) and `AI_API_KEY` in your `.env` file, then restart the container."
        )

    async def summarize_sector(self, sector: str, stocks: list[dict]) -> str:
        return f"AI sector analysis for {sector} is not configured."


# ── Ollama Implementation ─────────────────────────────────────────────────

class OllamaAIService(AIService):
    """
    Ollama local LLM service.
    Calls the Ollama /api/chat endpoint on your local network.
    Set AI_PROVIDER=ollama in .env to activate.
    """

    def __init__(self):
        self.base_url = settings.ollama_url.rstrip("/")
        self.model = settings.ollama_model
        self._timeout = 120.0

    async def _chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json()["message"]["content"]

    async def analyze_stock(self, ticker: str, data: dict) -> dict:
        system = (
            "You are a professional stock analyst. Analyze the given stock data and respond "
            "with a JSON object only — no markdown, no extra text. Keys: summary (str), "
            "sentiment (bullish|bearish|neutral), confidence (0.0-1.0), key_factors ([str]), "
            "ai_score (1-10 int), risks ([str]), opportunities ([str])."
        )
        user = f"Analyze {ticker}:\n{json.dumps(data, default=str)}"
        try:
            raw = await self._chat(system, user)
            # Strip any markdown code fences the model may add
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(raw)
        except Exception as e:
            logger.error(f"Ollama analyze_stock failed for {ticker}: {e}")
            return {
                "summary": f"Ollama analysis failed: {e}",
                "sentiment": "neutral", "confidence": 0.0,
                "key_factors": [], "ai_score": None,
                "risks": ["LLM unavailable"], "opportunities": [],
            }

    async def generate_signals(self, ticker: str, technicals: dict) -> list[dict]:
        system = (
            "You are a swing-trading signal generator. Based on the technical data provided, "
            "return a JSON array of signal objects only — no markdown, no extra text. "
            "Each object: type (entry|exit|warning|info), signal (short label), "
            "message (explanation), strength (strong|moderate|weak)."
        )
        user = f"Generate signals for {ticker}:\n{json.dumps(technicals, default=str)}"
        try:
            raw = await self._chat(system, user)
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(raw)
        except Exception as e:
            logger.error(f"Ollama generate_signals failed for {ticker}: {e}")
            return [{"type": "info", "signal": "LLM Error", "message": str(e), "strength": "weak"}]

    async def chat(self, ticker: str, question: str, context: dict) -> str:
        system = (
            "You are a swing-trading assistant. Answer concisely using the stock data provided. "
            "Use markdown formatting. Keep answers under 300 words."
        )
        user = (
            f"Stock: {ticker}\nData: {json.dumps(context, default=str)}\n\nQuestion: {question}"
        )
        try:
            return await self._chat(system, user)
        except Exception as e:
            logger.error(f"Ollama chat failed for {ticker}: {e}")
            return f"Ollama error: {e}"

    async def summarize_sector(self, sector: str, stocks: list[dict]) -> str:
        system = (
            "You are a market analyst. Write a 2-4 sentence markdown sector summary "
            "based on the stock data provided."
        )
        user = f"Sector: {sector}\nStocks: {json.dumps(stocks[:10], default=str)}"
        try:
            return await self._chat(system, user)
        except Exception as e:
            logger.error(f"Ollama summarize_sector failed for {sector}: {e}")
            return f"Ollama sector summary unavailable: {e}"


# ── Future: Anthropic Implementation ──────────────────────────────────────
#
# class AnthropicAIService(AIService):
#     def __init__(self):
#         import anthropic
#         self.client = anthropic.AsyncAnthropic(api_key=settings.ai_api_key)
#         self.model = settings.ai_model or "claude-opus-4-6"
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
#         self.model = settings.ai_model or "gpt-4o"
#
#     async def analyze_stock(self, ticker, data):
#         ...


# ── Factory ───────────────────────────────────────────────────────────────

def get_ai_service() -> AIService:
    """Return the configured AI service singleton."""
    provider = settings.ai_provider.lower()

    if provider == "ollama":
        return OllamaAIService()

    if provider == "anthropic":
        # Uncomment when AnthropicAIService is implemented above
        # return AnthropicAIService()
        logger.warning("Anthropic provider selected but not yet implemented. Using mock.")
        return MockAIService()

    if provider == "openai":
        # Uncomment when OpenAIAIService is implemented above
        # return OpenAIAIService()
        logger.warning("OpenAI provider selected but not yet implemented. Using mock.")
        return MockAIService()

    return MockAIService()


# Singleton — created once at import time
_ai_service: AIService = get_ai_service()


def ai_service() -> AIService:
    """FastAPI dependency: returns the singleton AI service."""
    return _ai_service
