"""Read-only Polymarket client for BTC 5-minute Up/Down markets.

SAFETY: This module uses httpx for HTTP calls. It does NOT import
py-clob-client, and it MUST NEVER import any order, signing, or
wallet module. NFR-001 is enforced by test_no_order_path.py.
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


def fetch_market_metadata(
    start_ts: int,
    end_ts: int,
    timeout: float = 30.0,
) -> pd.DataFrame:
    resp = httpx.get(
        f"{GAMMA_API_BASE}/events",
        params={
            "tag": "btc-5-minute",
            "closed": True,
            "limit": 100,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    events = resp.json()

    rows: list[dict] = []
    for event in events:
        for market in event.get("markets", []):
            parsed = _parse_market(event, market)
            if parsed is None:
                continue
            if start_ts <= parsed["window_open_ts"] < end_ts:
                rows.append(parsed)

    if not rows:
        return pd.DataFrame(columns=MARKET_COLUMNS)
    return pd.DataFrame(rows)[MARKET_COLUMNS]


def _parse_market(event: dict, market: dict) -> dict | None:
    try:
        end_date = market.get("endDate", "")
        if not end_date:
            return None

        dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        window_close_ts = int(dt.timestamp())
        window_open_ts = window_close_ts - WINDOW_SECONDS

        if window_open_ts % WINDOW_SECONDS != 0:
            return None

        price_to_beat = _extract_price_to_beat(event.get("description", ""))
        if price_to_beat is None:
            return None

        outcome_prices = _parse_json_list(market.get("outcomePrices", "[]"))
        clob_token_ids = _parse_json_list(market.get("clobTokenIds", "[]"))

        if len(outcome_prices) < 2 or len(clob_token_ids) < 2:
            return None

        return {
            "window_open_ts": window_open_ts,
            "condition_id": market.get("id", ""),
            "token_id_up": clob_token_ids[0],
            "token_id_down": clob_token_ids[1],
            "price_to_beat": price_to_beat,
            "price_up": float(outcome_prices[0]),
            "price_down": float(outcome_prices[1]),
            "captured_ts": int(datetime.now(timezone.utc).timestamp()),
        }
    except (ValueError, IndexError, KeyError) as exc:
        logger.debug("Skipping unparseable market: %s", exc)
        return None


def _extract_price_to_beat(description: str) -> float | None:
    match = re.search(r"\$?([\d,]+(?:\.\d+)?)", description)
    if match:
        return float(match.group(1).replace(",", ""))
    return None


def _parse_json_list(raw: str) -> list[str]:
    try:
        return json.loads(raw)
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
        candles_df[["window_open_ts", "close"]],
        on="window_open_ts",
        how="inner",
    )

    return pd.DataFrame(
        {
            "window_open_ts": merged["window_open_ts"],
            "oracle_close": merged["close"],
            "coinbase_close": merged["close"],
            "outcome_up": merged["close"] > merged["price_to_beat"],
        }
    )
