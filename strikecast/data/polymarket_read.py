"""Read-only Polymarket client for BTC 5-minute Up/Down markets.

SAFETY: This module uses httpx for HTTP calls. It does NOT import
py-clob-client, and it MUST NEVER import any order, signing, or
wallet module. NFR-001 is enforced by test_no_order_path.py.

Discovery: recurring BTC 5m markets use slug ``btc-updown-5m-<window_open_ts>``
(title e.g. "Bitcoin Up or Down - … 9:50PM-9:55PM ET"). Gamma's ``tag=``
filter does not return these; use ``/public-search`` and filter by slug prefix.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

import httpx
import pandas as pd

from strikecast.constants import LABEL_COLUMNS, MARKET_COLUMNS, WINDOW_SECONDS

logger = logging.getLogger(__name__)

GAMMA_API_BASE = "https://gamma-api.polymarket.com"
BTC_UPDOWN_5M_SLUG_PREFIX = "btc-updown-5m-"
BTC_UPDOWN_SEARCH_QUERY = "bitcoin up or down"


def fetch_market_metadata(
    start_ts: int,
    end_ts: int,
    timeout: float = 30.0,
    max_pages: int = 500,
) -> pd.DataFrame:
    """Fetch closed BTC 5-minute Up/Down markets in ``[start_ts, end_ts)``."""
    rows: list[dict] = []
    seen_windows: set[int] = set()

    for page in range(1, max_pages + 1):
        resp = httpx.get(
            f"{GAMMA_API_BASE}/public-search",
            params={
                "q": BTC_UPDOWN_SEARCH_QUERY,
                "limit_per_type": 100,
                "page": page,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        events = payload.get("events", [])
        if not events:
            break

        for event in events:
            slug = event.get("slug") or ""
            if not slug.startswith(BTC_UPDOWN_5M_SLUG_PREFIX):
                continue

            window_open_ts = _window_open_ts_from_slug(slug)
            if window_open_ts is None:
                continue
            if window_open_ts in seen_windows:
                continue
            if not (start_ts <= window_open_ts < end_ts):
                continue

            for market in event.get("markets", []):
                parsed = _parse_market(event, market, window_open_ts=window_open_ts)
                if parsed is None:
                    continue
                seen_windows.add(window_open_ts)
                rows.append(parsed)
                break

        pagination = payload.get("pagination") or {}
        if not pagination.get("hasMore"):
            break

    logger.info(
        "Polymarket: found %d BTC 5m Up/Down markets in [%d, %d)",
        len(rows),
        start_ts,
        end_ts,
    )
    if not rows:
        return pd.DataFrame(columns=MARKET_COLUMNS)
    return pd.DataFrame(rows)[MARKET_COLUMNS]


def _window_open_ts_from_slug(slug: str) -> int | None:
    suffix = slug.removeprefix(BTC_UPDOWN_5M_SLUG_PREFIX)
    try:
        ts = int(suffix)
    except ValueError:
        return None
    if ts % WINDOW_SECONDS != 0:
        return None
    return ts


def _parse_market(
    event: dict,
    market: dict,
    *,
    window_open_ts: int | None = None,
) -> dict | None:
    try:
        if window_open_ts is None:
            slug = event.get("slug") or ""
            window_open_ts = _window_open_ts_from_slug(slug)
            if window_open_ts is None:
                event_start = market.get("eventStartTime") or market.get("startDate", "")
                if event_start:
                    dt = datetime.fromisoformat(event_start.replace("Z", "+00:00"))
                    window_open_ts = int(dt.timestamp())
                else:
                    end_date = market.get("endDate", "")
                    if not end_date:
                        return None
                    dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                    window_open_ts = int(dt.timestamp()) - WINDOW_SECONDS

        if window_open_ts % WINDOW_SECONDS != 0:
            return None

        price_to_beat = _extract_price_to_beat(
            event.get("description", "") + " " + market.get("description", "")
        )

        outcome_prices = _parse_json_list(market.get("outcomePrices", "[]"))
        clob_token_ids = _parse_json_list(market.get("clobTokenIds", "[]"))

        if len(outcome_prices) < 2 or len(clob_token_ids) < 2:
            return None

        return {
            "window_open_ts": window_open_ts,
            "condition_id": market.get("id", ""),
            "token_id_up": clob_token_ids[0],
            "token_id_down": clob_token_ids[1],
            "price_to_beat": float("nan") if price_to_beat is None else price_to_beat,
            "price_up": float(outcome_prices[0]),
            "price_down": float(outcome_prices[1]),
            "captured_ts": int(datetime.now(timezone.utc).timestamp()),
        }
    except (ValueError, IndexError, KeyError) as exc:
        logger.debug("Skipping unparseable market: %s", exc)
        return None


def _extract_price_to_beat(description: str) -> float | None:
    match = re.search(r"price to beat[:\s]*\$?([\d,]+(?:\.\d+)?)", description, re.I)
    if match:
        return float(match.group(1).replace(",", ""))
    match = re.search(r"\$([\d,]+(?:\.\d+)?)", description)
    if match:
        return float(match.group(1).replace(",", ""))
    return None


def _parse_json_list(raw: str | list) -> list[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw]
    try:
        parsed = json.loads(raw)
        return [str(x) for x in parsed]
    except (json.JSONDecodeError, TypeError):
        return []


def fetch_resolution_labels(
    candles_df: pd.DataFrame,
    markets_df: pd.DataFrame,
) -> pd.DataFrame:
    if candles_df.empty or markets_df.empty:
        return pd.DataFrame(columns=LABEL_COLUMNS)

    merged = pd.merge(
        markets_df[["window_open_ts", "price_to_beat"]],
        candles_df[["window_open_ts", "open", "close"]],
        on="window_open_ts",
        how="inner",
    )

    # Strike = Chainlink price at window open; proxy with Coinbase open until RTDS wired.
    strike = merged["price_to_beat"].fillna(merged["open"])

    return pd.DataFrame(
        {
            "window_open_ts": merged["window_open_ts"],
            "oracle_close": merged["close"],
            "coinbase_close": merged["close"],
            # Polymarket resolves Up when end price >= open (Chainlink); >= with proxy.
            "outcome_up": merged["close"] >= strike,
        }
    )
