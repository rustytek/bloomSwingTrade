import re

from fastapi import HTTPException


_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.-]{0,14}$")


def normalize_ticker(value: str) -> str:
    ticker = (value or "").upper().strip().replace("/", "-")
    if not _TICKER_RE.fullmatch(ticker):
        raise HTTPException(status_code=400, detail=f"Invalid ticker: {value}")
    return ticker


def normalize_tickers(values: list[str], limit: int | None = None) -> list[str]:
    seen = set()
    normalized = []
    for raw in values[:limit] if limit else values:
        ticker = normalize_ticker(raw)
        if ticker not in seen:
            seen.add(ticker)
            normalized.append(ticker)
    return normalized
