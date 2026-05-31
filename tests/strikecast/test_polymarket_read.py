from unittest.mock import MagicMock, patch

import httpx
import pytest

from strikecast.constants import WINDOW_SECONDS

WINDOW_OPEN_TS = 1768435200  # 2026-01-15 00:00:00 UTC, 300s grid

SAMPLE_SEARCH_RESPONSE = {
    "events": [
        {
            "id": "event_1",
            "slug": f"btc-updown-5m-{WINDOW_OPEN_TS}",
            "title": "Bitcoin Up or Down - January 14, 7:00PM-7:05PM ET",
            "markets": [
                {
                    "id": "cond_abc",
                    "question": "Bitcoin Up or Down - January 14, 7:00PM-7:05PM ET",
                    "outcomes": ["Up", "Down"],
                    "outcomePrices": "[0.52, 0.48]",
                    "clobTokenIds": '["tok_up_1", "tok_down_1"]',
                    "closed": True,
                    "eventStartTime": "2026-01-15T00:00:00Z",
                    "endDate": "2026-01-15T00:05:00Z",
                    "description": (
                        "Resolves Up if Chainlink BTC/USD at end >= price at start."
                    ),
                }
            ],
            "description": "Resolves on Chainlink BTC/USD stream.",
        }
    ],
    "pagination": {"hasMore": False, "totalResults": 1},
}


class TestFetchMarketMetadata:
    def test_returns_market_dict(self) -> None:
        from strikecast.data.polymarket_read import fetch_market_metadata

        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_SEARCH_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch("strikecast.data.polymarket_read.httpx.get", return_value=mock_response):
            result = fetch_market_metadata(
                start_ts=WINDOW_OPEN_TS,
                end_ts=WINDOW_OPEN_TS + WINDOW_SECONDS,
            )

        assert len(result) == 1
        row = result.iloc[0]
        assert row["window_open_ts"] == WINDOW_OPEN_TS
        assert row["condition_id"] == "cond_abc"
        assert row["token_id_up"] == "tok_up_1"
        assert row["price_up"] == pytest.approx(0.52)

    def test_returns_empty_when_no_markets(self) -> None:
        from strikecast.data.polymarket_read import fetch_market_metadata

        mock_response = MagicMock()
        mock_response.json.return_value = {"events": [], "pagination": {"hasMore": False}}
        mock_response.raise_for_status = MagicMock()

        with patch("strikecast.data.polymarket_read.httpx.get", return_value=mock_response):
            result = fetch_market_metadata(
                start_ts=WINDOW_OPEN_TS,
                end_ts=WINDOW_OPEN_TS + WINDOW_SECONDS,
            )

        assert len(result) == 0

    def test_skips_non_5m_slug(self) -> None:
        from strikecast.data.polymarket_read import fetch_market_metadata

        payload = {
            "events": [
                {
                    "slug": "btc-updown-15m-1771868700",
                    "markets": [{"id": "x", "outcomePrices": "[0.5,0.5]", "clobTokenIds": '["a","b"]', "endDate": "2026-02-23T18:00:00Z"}],
                }
            ],
            "pagination": {"hasMore": False},
        }
        mock_response = MagicMock()
        mock_response.json.return_value = payload
        mock_response.raise_for_status = MagicMock()

        with patch("strikecast.data.polymarket_read.httpx.get", return_value=mock_response):
            result = fetch_market_metadata(
                start_ts=WINDOW_OPEN_TS,
                end_ts=WINDOW_OPEN_TS + WINDOW_SECONDS,
            )

        assert len(result) == 0


class TestModuleSafety:
    def test_does_not_import_py_clob_client(self) -> None:
        """polymarket_read.py must never import py-clob-client."""
        import ast
        from pathlib import Path

        source = Path("strikecast/data/polymarket_read.py").read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "clob" not in alias.name.lower(), (
                        f"polymarket_read.py imports '{alias.name}' — "
                        "py-clob-client must not be imported"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module and "clob" in node.module.lower():
                    raise AssertionError(
                        f"polymarket_read.py imports from '{node.module}' — "
                        "py-clob-client must not be imported"
                    )
