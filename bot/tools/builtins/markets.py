"""Tools: currency conversion and live prices (fiat, crypto, stocks).
[CA][IV][REH][PA].

Rates and prices change daily; the model's remembered numbers are stale and it
gets them wrong ("10 cad is about 5.8 gbp"). These look the numbers up:

- Fiat rates: Frankfurter (European Central Bank reference rates, no key).
- Crypto: CoinGecko public API (no key).
- Stocks: Yahoo Finance chart endpoint (no key).

Read-only HTTP GETs with fixed hosts; the model only supplies codes/symbols,
which are validated before they reach a URL. Never raises.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote

from bot.utils.logging import get_logger

from ..types import ToolContext, ToolResult, ToolSpec

logger = get_logger(__name__)

FRANKFURTER_URL = "https://api.frankfurter.dev/v1/latest"
COINGECKO_PRICE_URL = "https://api.coingecko.com/api/v3/simple/price"
COINGECKO_SEARCH_URL = "https://api.coingecko.com/api/v3/search"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

_FIAT_RE = re.compile(r"^[A-Z]{3}$")
_SYMBOL_RE = re.compile(r"^[A-Z0-9.\-^=]{1,15}$")

# Common coins resolved without a search round trip. [CMV][PA]
KNOWN_COINS: dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "USDT": "tether",
    "USDC": "usd-coin",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "DOGE": "dogecoin",
    "ADA": "cardano",
    "LTC": "litecoin",
    "TRX": "tron",
    "AVAX": "avalanche-2",
    "DOT": "polkadot",
    "LINK": "chainlink",
    "XMR": "monero",
    "SHIB": "shiba-inu",
    "PEPE": "pepe",
    "TON": "the-open-network",
}


def _request_config() -> Any:
    from bot.http_client import RequestConfig

    return RequestConfig(connect_timeout=2.0, read_timeout=6.0, total_timeout=8.0, max_retries=1)


async def _get_json(url: str, params: dict[str, Any] | None = None) -> Any | None:
    """GET JSON from a fixed API host. None on any failure."""
    from bot.http_client import get_http_client

    try:
        client = await get_http_client()
        response = await client.get(url, config=_request_config(), params=params or {}, headers={"User-Agent": "Mozilla/5.0"})
    except Exception as exc:  # noqa: BLE001 — network failure is a normal "couldn't check" answer
        logger.info("tool.markets.http_failed url=%s error=%s", url[:80], exc)
        return None
    if response.status_code != 200:
        logger.info("tool.markets.http_status url=%s status=%s", url[:80], response.status_code)
        return None
    try:
        return response.json()
    except ValueError:
        return None


def _code(value: Any) -> str:
    return str(value or "").strip().upper().lstrip("$")


async def resolve_coin(symbol: str) -> tuple[str, str] | None:
    """(coingecko_id, display name) for a ticker like SOL, or None."""
    if symbol in KNOWN_COINS:
        return KNOWN_COINS[symbol], symbol
    data = await _get_json(COINGECKO_SEARCH_URL, {"query": symbol})
    coins = (data or {}).get("coins") or []
    exact = [c for c in coins if str(c.get("symbol", "")).upper() == symbol]
    pick = (exact or coins or [None])[0]
    if not pick or not pick.get("id"):
        return None
    return str(pick["id"]), str(pick.get("name") or symbol)


async def _coin_price(coin_id: str, fiat: str) -> tuple[float, float | None] | None:
    vs = fiat.lower()
    data = await _get_json(COINGECKO_PRICE_URL, {"ids": coin_id, "vs_currencies": vs, "include_24hr_change": "true"})
    row = (data or {}).get(coin_id) or {}
    if vs not in row:
        return None
    return float(row[vs]), row.get(f"{vs}_24h_change")


def _fmt(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1:
        return f"{value:,.2f}"
    return f"{value:.6g}"


# ── convert_currency ─────────────────────────────────────────────────────────

CONVERT_DESCRIPTION = "Convert an amount between currencies at today's rate: regular currencies (USD, GBP, CAD, EUR, JPY...) and crypto (BTC, ETH, SOL...). Always use this instead of guessing an exchange rate."

CONVERT_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "amount": {"type": "number", "description": "Amount to convert.", "default": 1},
        "from_currency": {"type": "string", "description": "Code to convert from, e.g. CAD or BTC."},
        "to_currency": {"type": "string", "description": "Code to convert to, e.g. GBP or USD."},
    },
    "required": ["from_currency", "to_currency"],
}


async def _convert_fiat(amount: float, src: str, dst: str) -> ToolResult:
    data = await _get_json(FRANKFURTER_URL, {"base": src, "symbols": dst})
    rate = ((data or {}).get("rates") or {}).get(dst)
    if rate is None:
        return ToolResult.failure(f"no rate for {src} to {dst} (unsupported currency or service down)")
    day = (data or {}).get("date", "today")
    return ToolResult.success(f"{_fmt(amount)} {src} = {_fmt(amount * float(rate))} {dst} (1 {src} = {float(rate):.5g} {dst}; European Central Bank reference rate, {day})")


async def _crypto_in(code: str) -> tuple[float, str] | None:
    """USD price of one unit of ``code`` (crypto or fiat) and its display name."""
    if code == "USD":
        return 1.0, "USD"
    if code in KNOWN_COINS or not _FIAT_RE.match(code):
        coin = await resolve_coin(code)
        if coin is None:
            return None
        price = await _coin_price(coin[0], "USD")
        return (price[0], coin[1]) if price else None
    data = await _get_json(FRANKFURTER_URL, {"base": code, "symbols": "USD"})
    rate = ((data or {}).get("rates") or {}).get("USD")
    return (float(rate), code) if rate is not None else None


async def convert_currency(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    src, dst = _code(arguments.get("from_currency")), _code(arguments.get("to_currency"))
    if not (_SYMBOL_RE.match(src) and _SYMBOL_RE.match(dst)):
        return ToolResult.failure("from_currency and to_currency must be currency or coin codes like GBP or BTC")
    raw_amount = arguments.get("amount")
    try:
        amount = 1.0 if raw_amount is None else float(raw_amount)
    except (TypeError, ValueError):
        return ToolResult.failure("amount must be a number")
    logger.info("tool.convert_currency %s %s->%s", amount, src, dst)
    both_fiat = _FIAT_RE.match(src) and _FIAT_RE.match(dst) and src not in KNOWN_COINS and dst not in KNOWN_COINS
    if both_fiat:
        return await _convert_fiat(amount, src, dst)
    a, b = await _crypto_in(src), await _crypto_in(dst)
    if a is None or b is None:
        return ToolResult.failure(f"couldn't price {src if a is None else dst}")
    result = amount * a[0] / b[0]
    return ToolResult.success(f"{_fmt(amount)} {a[1]} = {_fmt(result)} {b[1]} (live price via CoinGecko)")


# ── get_price ────────────────────────────────────────────────────────────────

PRICE_DESCRIPTION = "Current price of a stock/ETF/index (ticker like AAPL, TSLA, SPY, ^GSPC) or a crypto coin (BTC, ETH...), with today's change. Use instead of remembering prices."

PRICE_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "symbol": {"type": "string", "description": "Ticker or coin symbol."},
        "asset_type": {"type": "string", "enum": ["stock", "crypto"], "description": "stock (default) or crypto."},
    },
    "required": ["symbol"],
}


async def _stock_price(symbol: str) -> ToolResult:
    data = await _get_json(YAHOO_CHART_URL.format(symbol=quote(symbol, safe="")), {"range": "1d", "interval": "1d"})
    results = ((data or {}).get("chart") or {}).get("result") or []
    meta = (results[0] or {}).get("meta") if results else None
    if not meta or meta.get("regularMarketPrice") is None:
        return ToolResult.failure(f"no stock quote for {symbol}")
    change = meta.get("regularMarketChangePercent")
    change_txt = f", {change:+.2f}% today" if isinstance(change, int | float) else ""
    name = meta.get("shortName") or meta.get("longName") or symbol
    return ToolResult.success(
        f"{name} ({symbol}): {_fmt(float(meta['regularMarketPrice']))} {meta.get('currency', '')}{change_txt} on {meta.get('fullExchangeName') or meta.get('exchangeName') or 'market'} (Yahoo Finance; may be delayed)"
    )


async def _crypto_price(symbol: str) -> ToolResult:
    coin = await resolve_coin(symbol)
    price = await _coin_price(coin[0], "USD") if coin else None
    if not coin or not price:
        return ToolResult.failure(f"no crypto price for {symbol}")
    change_txt = f", {price[1]:+.2f}% in 24h" if isinstance(price[1], int | float) else ""
    label = symbol if coin[1] == symbol else f"{coin[1]} ({symbol})"
    return ToolResult.success(f"{label}: ${_fmt(price[0])}{change_txt} (CoinGecko)")


async def get_price(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    symbol = _code(arguments.get("symbol"))
    if not _SYMBOL_RE.match(symbol):
        return ToolResult.failure("symbol must be a ticker like AAPL or BTC")
    asset = str(arguments.get("asset_type") or "").lower()
    is_crypto = asset == "crypto" or (asset != "stock" and symbol in KNOWN_COINS)
    logger.info("tool.get_price symbol=%s crypto=%s", symbol, is_crypto)
    return await (_crypto_price(symbol) if is_crypto else _stock_price(symbol))


CONVERT_SPEC = ToolSpec(
    name="convert_currency",
    description=CONVERT_DESCRIPTION,
    parameters=CONVERT_PARAMETERS,
    handler=convert_currency,
    timeout_s=20.0,
)
PRICE_SPEC = ToolSpec(
    name="get_price",
    description=PRICE_DESCRIPTION,
    parameters=PRICE_PARAMETERS,
    handler=get_price,
    timeout_s=20.0,
)
